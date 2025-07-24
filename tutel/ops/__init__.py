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

def to_float4_groupwise(w):
  assert w.size(-1) % 16 == 0
  x = w.view(-1, 16)
  scale_b = x.abs().amax(-1, keepdim=True).float() / 6
  scale_b = torch.where(scale_b > 1e-4, scale_b, 1e-4).to(scale_b.dtype)
  boundaries = torch.tensor([-10, -5, -3.5, -2.5, -1.75, -1.25, -0.75, -0.25, 0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5], dtype=torch.float32, device=w.device)
  map_ids = torch.tensor([0, 15, 14, 13, 12, 11, 10,  9,  0,  8,  1,  2,  3,  4,  5,  6,  7], dtype=torch.uint8, device=w.device)
  scale_o = scale_b.amax() / 448
  fp4_vals = map_ids.index_select(0, torch.bucketize(x / scale_b, boundaries, right=True, out_int32=True).clamp(1, 16).flatten()).view(*w.shape[:-1],   w.shape[-1] // 2, 2).chunk(2, dim=-1)
  scale_b = (scale_b / scale_o).to(torch.float8_e4m3fn).view(*w.shape[:-1], w.shape[-1] // 16)
  return (fp4_vals[0] + (fp4_vals[1] << 4)).squeeze(-1), scale_b, scale_o

def from_float4_groupwise(w, scale, scale_2, input_scale=None):
  assert w.dtype == torch.uint8
  w = w.to(torch.int16)
  fp4_e2m1_table = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.bfloat16, device=w.device)
  w = ((w & 15) + ((w >> 4) << 8)).view(torch.int8).to(torch.int32)
  w = fp4_e2m1_table.index_select(0, w.flatten()).view(*scale.shape, -1) * scale.bfloat16().unsqueeze(scale.dim())
  return w.flatten(-2) * scale_2


def __getattr__(name):
    fn = getattr(torch.ops.tutel_ops, name)
    return torch.compiler.disable(fn, recursive=True)

