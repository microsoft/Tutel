# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import torch
import tutel_custom_kernel

from torch.utils.cpp_extension import IS_HIP_EXTENSION

if 'OP_LOADER' not in os.environ:
    if not IS_HIP_EXTENSION:
        suffix = 'cuda'
    else:
        suffix = 'rocm'
    os.environ['OP_LOADER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), suffix)

def pad_at_dim(x, dim, new_size):
  padded_shape = list(x.shape)
  if padded_shape[dim] == new_size:
    return x
  padded_shape[dim] = new_size
  y = torch.empty(padded_shape, dtype=x.dtype, device=x.device)
  y.narrow(dim, 0, x.size(dim)).copy_(x)
  y.narrow(dim, x.size(dim), new_size - x.size(dim)).zero_()
  return y

def to_float8_rowwise(
    x: torch.Tensor, dim=-1, dtype: torch.dtype = torch.float8_e4m3fn, max_scale=None
):
    # sum_val = x.float().sum(dim=dim, keepdim=True) / x.size(dim)
    # x = x - sum_val
    min_val, max_val = x.aminmax(dim=dim, keepdim=True)
    amax = torch.maximum(min_val.abs(), max_val.abs()).clamp(min=1e-12)
    fp8_max = max_scale if max_scale is not None else 224.0
    scale = fp8_max / amax.float()
    x_scl_sat = (x * scale).clamp(min=-fp8_max, max=fp8_max)
    x_scl_sat = x_scl_sat.to(dtype).contiguous()
    assert x_scl_sat.dtype == torch.float8_e4m3fn
    x_scl_sat.view(torch.int8)[x_scl_sat.view(torch.int8) == 128] = 0
    x_scl_sat.scale_inv = scale.float().reciprocal().squeeze(dim)
    # x_scl_sat.scale_mean = sum_val.squeeze(dim)
    return x_scl_sat

def to_float8_blockwise(w, block_size=128):
  shape = w.shape
  assert w.dim() in (2, 3)
  if w.dim() == 2:
    w = w.unsqueeze(0)
  ws_shape = [w.size(0), (w.size(1) + block_size - 1) // block_size, (w.size(2) + block_size - 1) // block_size]
  w = torch.nn.functional.pad(w, (0, ws_shape[-1] * block_size - w.size(-1), 0, ws_shape[-2] * block_size - w.size(-2))).view(w.size(0), ws_shape[-2], block_size, ws_shape[-1], block_size)
  ws = w.abs().amax(dim=-3, keepdim=True).amax(dim=-1, keepdim=True).float() / 224.0
  w = (w / ws.clip(min=1e-5)).clip(max=224.0)
  w, ws = w.view(w.size(0), ws_shape[1] * block_size, ws_shape[2] * block_size)[:, :shape[-2], :shape[-1]].to(torch.float8_e4m3fn), ws.view(ws_shape)
  if len(shape) == 2:
    w, ws = w.squeeze(0), ws.squeeze(0)
  w.view(torch.uint8)[w.view(torch.uint8) == 128] = 0
  return w, ws

def from_float8_blockwise(w, ws, block_size=128, dtype=torch.bfloat16):
  shape = w.shape
  assert w.dtype == torch.float8_e4m3fn
  assert w.dim() == ws.dim() and w.dim() in (2, 3)
  if w.dim() == 2:
    w, ws = w.unsqueeze(0), ws.unsqueeze(0)
  else:
    assert w.size(0) == ws.size(0)
  ph = torch.empty([ws.size(0), ws.size(1) * block_size, ws.size(2) * block_size], dtype=w.dtype, device=w.device)
  ph[:, :w.size(1), :w.size(2)] = w
  ph = (ph.view(w.size(0), ws.size(1), block_size, ws.size(2), block_size).to(ws.dtype) * ws.view(w.size(0), ws.size(1), 1, ws.size(2), 1)).view(ph.shape)
  return ph[:, :w.size(1), :w.size(2)].to(dtype).view(shape)

def to_float4_groupwise(w):
  assert w.size(-1) % 16 == 0
  x = w.view(-1, 16)
  scale_b = x.abs().amax(-1, keepdim=True).float() / 6
  scale_b = torch.where(scale_b > 1e-4, scale_b, 1e-4).to(scale_b.dtype)
  boundaries = torch.tensor([-10, -5, -3.5, -2.5, -1.75, -1.25, -0.75, -0.25, 0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5], dtype=torch.float32, device=w.device)
  map_ids = torch.tensor([0, 15, 14, 13, 12, 11, 10,  9,  0,  8,  1,  2,  3,  4,  5,  6,  7], dtype=torch.uint8, device=w.device)
  scale_o = scale_b.amax() / 448
  fp4_vals = map_ids.index_select(0, torch.bucketize(x / scale_b, boundaries, right=True, out_int32=True).clamp(1, 16).flatten()).view(*w.shape[:-1], w.shape[-1] // 2, 2).chunk(2, dim=-1)
  scale_b = (scale_b / scale_o).to(torch.float8_e4m3fn).view(*w.shape[:-1], w.shape[-1] // 16)
  return (fp4_vals[0] + (fp4_vals[1] << 4)).squeeze(-1), scale_b, scale_o

def from_float4_groupwise(w, scale, scale_2, input_scale=None):
  assert w.dtype == torch.uint8
  w = w.to(torch.int16)
  fp4_e2m1_table = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.bfloat16, device=w.device)
  w = ((w & 15) + ((w >> 4) << 8)).view(torch.int8).to(torch.int32)
  w = fp4_e2m1_table.index_select(0, w.flatten()).view(*scale.shape, -1) * scale.bfloat16().unsqueeze(scale.dim())
  return w.flatten(-2) * scale_2

def from_mxfp4(p):
  w = p.to(torch.int16)
  fp4_e2m1_table = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.float32, device=w.device)
  w = ((w & 15) + ((w >> 4) << 8)).view(torch.int8).to(torch.int32)
  w = fp4_e2m1_table.index_select(0, w.flatten()).view(*p.scales.shape, -1)
  s = torch.pow(2.0, p.scales.view(torch.int8) - 127).view(*w.shape[:-1], -1)
  w = (w * s).bfloat16().flatten(-2)
  w.bias = p.bias
  return w


def __getattr__(name):
    fn = getattr(torch.ops.tutel_ops, name)
    return torch.compiler.disable(fn, recursive=True)

