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

def from_float8_blockwise(w, w_scale, block_size=128, dtype=torch.bfloat16):
  assert (w.size(-1) + block_size - 1) // block_size == w_scale.size(-1)
  assert (w.size(-2) + block_size - 1) // block_size == w_scale.size(-2)
  w_out = torch.empty(list(w.shape[:-2]) + [w_scale.size(-2) * block_size, w_scale.size(-1) * block_size], dtype=dtype, device=w.device)
  w_out.narrow(-2, 0, w.size(-2)).narrow(-1, 0, w.size(-1)).copy_(w.to(w_out.dtype))
  w_view = w_out.view(list(w.shape[:-2]) + [w_scale.size(-2), block_size, w_scale.size(-1), block_size])
  w_view *= w_scale.unsqueeze(-2).unsqueeze(-1).to(w_out.dtype)
  return w_out.narrow(-2, 0, w.size(-2)).narrow(-1, 0, w.size(-1)).contiguous()

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

def from_float4_groupwise(w, scale, scale_o=None, input_scale=None, dtype=torch.bfloat16):
  assert w.dtype == torch.uint8
  fp4_e2m1_table = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.bfloat16, device=w.device)
  w = w.to(torch.int16)
  w = ((w & 15) + ((w >> 4) << 8)).view(torch.int8).to(torch.int32)
  w = (fp4_e2m1_table.index_select(0, w.flatten()).view(*scale.shape, -1) * scale.bfloat16().unsqueeze(scale.dim())).flatten(-2)
  if scale_o is None:
    return w
  prefix = w.shape[:-2]
  scale_o = scale_o.view(*prefix, -1)
  w = w.view(*prefix, scale_o.size(-1), w.size(-2) // scale_o.size(-1), w.size(-1)).float() * scale_o.unsqueeze(-1).unsqueeze(-1)
  return w.view(*prefix, -1, w.size(-1)).to(dtype)

def from_int4_groupwise(x, s, dtype=torch.bfloat16):
  assert x.dim() == 2
  assert s.size(-1) * 4 == x.size(-1)
  x = x.view(torch.uint8).view(x.size(0), -1)
  N, half_M = x.shape

  x16 = x.to(torch.int16)
  low = x16 & 0x0F
  high = (x16 >> 4) & 0x0F
  low_signed = torch.where(low >= 8, low - 16, low).to(torch.int8)
  high_signed = torch.where(high >= 8, high - 16, high).to(torch.int8)
  out = torch.empty((N, half_M * 2), dtype=torch.int8, device=x.device)
  out[:, 0::2] = low_signed
  out[:, 1::2] = high_signed
  out = out.to(dtype)
  return (out.view(*s.shape, -1).to(dtype) * s.unsqueeze(-1).to(dtype)).flatten(-2)


def from_mxfp4(p):
  w = p.to(torch.int16)
  fp4_e2m1_table = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.float32, device=w.device)
  w = ((w & 15) + ((w >> 4) << 8)).view(torch.int8).to(torch.int32)
  w = fp4_e2m1_table.index_select(0, w.flatten()).view(*p.scales.shape, -1)
  s = torch.pow(2.0, p.scales.view(torch.int8) - 127).view(*w.shape[:-1], -1)
  w = (w * s).bfloat16().flatten(-2)
  w.bias = p.bias
  return w


import numpy as np

def marlin_pack(w_unpacked):

    def _get_marlin_perms(device):
        perm = []
        for i in range(32):
            perm1 = []
            col = i // 4
            for block in [0, 1]:
                for row in [
                    2 * (i % 4),
                    2 * (i % 4) + 1,
                    2 * (i % 4 + 4),
                    2 * (i % 4 + 4) + 1
                ]:
                    perm1.append(16 * row + col + 8 * block)
            for j in range(4):
                perm.extend([p + 256 * j for p in perm1])
        perm = np.array(perm)
        interleave = np.array([0, 2, 4, 6, 1, 3, 5, 7])
        perm = perm.reshape((-1, 8))[:, interleave].ravel()
        return torch.from_numpy(perm).long().to(device)

    def unpack_weight_4d(packed_weight):
        # only for 4-bit weight packed in 8-bit (such as 2 fp4 in one uint8)
        device = packed_weight.device
        # shape: [total_num_experts, out_feature, in_feature // group_size, group_size // 2]
        E, N, G, half_group_size = packed_weight.shape
        w_bytes = packed_weight.view(E, N, G * half_group_size)
        low = w_bytes & 0x0F
        high = (w_bytes >> 4) & 0x0F
        # stack(..., dim=-1) -> [E, N, G*group_size//2, 2] -> view -> [E, N, K]
        w_unpacked = torch.stack([low, high], dim=-1).view(E, N, -1)
        # Marlin [In, Out]，GPTOSS [Out, In]
        return w_unpacked.transpose(1, 2).contiguous()

    w_unpacked = unpack_weight_4d(w_unpacked)
    device = w_unpacked.device
    tile = 16
    E, K, N = w_unpacked.shape
    # Shape: [E, K, N] -> [E, K//16, 16, N//16, 16]
    w = w_unpacked.reshape(E, K // tile, tile, N // tile, tile)
    # Permute -> [E, K//16, N//16, 16, 16]
    w = w.permute(0, 1, 3, 2, 4)
    # Reshape -> [E, K//16, N * 16]
    w = w.reshape(E, K // tile, N * tile)
    perm = _get_marlin_perms(device)
    num_cols = w.shape[-1]
    assert num_cols % perm.numel() == 0
    w_flat = w.reshape(-1, perm.numel())
    w_permuted = w_flat[:, perm]
    w = w_permuted.reshape(E, K // tile, N * tile)
    # pack into Marlin int32
    out_cols = w.shape[-1] // 8
    marlin_weight = torch.zeros((E, K // tile, out_cols), dtype=torch.int32, device=device)
    w = w.to(torch.int32)
    for i in range(8):
        marlin_weight |= (w[:, :, i::8] << (4 * i))
    return marlin_weight


def marlin_permute_scales(s, group_size):

    def get_scale_perms():
        scale_perm: list[int] = []
        for i in range(8):
            scale_perm.extend([i + 8 * j for j in range(8)])
        scale_perm_single: list[int] = []
        for i in range(4):
            scale_perm_single.extend([2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
        return scale_perm, scale_perm_single

    def nvfp4_marlin_process_scales(marlin_scales):
        # convert to half first, we would convert to fp8 later
        marlin_scales = marlin_scales.to(torch.half)

        # 8 is the number of scale number using by one thread
        marlin_scales = marlin_scales.view(marlin_scales.size(0), marlin_scales.size(1) // 2, 2, -1, 8)
        marlin_scales = marlin_scales.permute(0, 1, 3, 2, 4).reshape(
            marlin_scales.size(0), marlin_scales.size(1) * 2, -1
        )

        # fit the layout of fp8 dequantization
        marlin_scales = marlin_scales.view(marlin_scales.size(0), -1, 4)[:, :, [0, 2, 1, 3]].view(
            marlin_scales.size(0), marlin_scales.size(1), -1
        )

        # We assume that weight_scale (FP8-S1E4M3) is always greater
        # than or equal to 0. So we can convert
        # (weight_scale * (2 ** 7) to a special FP8-S0E5M3 format.
        # After multiplying by 2 ** 7, the top bit of FP8-S0E5M3 would always be 1
        # when weight_scale > 0. This allows us to have an exponent bias
        # closer to zero after dequantization.

        marlin_scales = (marlin_scales * (2**7)).view(torch.int16) << 1
        marlin_scales = marlin_scales.view(torch.float8_e4m3fn)
        marlin_scales = marlin_scales[:, :, 1::2].contiguous()
        return marlin_scales

    def mxfp4_marlin_process_scales(marlin_scales):
        marlin_scales = marlin_scales.view(marlin_scales.size(0), marlin_scales.size(1) // 2, 2, -1, 8)
        marlin_scales = marlin_scales.permute(0, 1, 3, 2, 4).reshape(
            marlin_scales.size(0), marlin_scales.size(1) * 2, -1
        )
        marlin_scales = marlin_scales.view(marlin_scales.size(0), -1, 4)[:, :, [0, 2, 1, 3]].view(
            marlin_scales.size(0), marlin_scales.size(1), -1
        )
        marlin_scales = marlin_scales.to(torch.float8_e8m0fnu)
        return marlin_scales

    if group_size != 16:
      s = s.view(torch.float8_e8m0fnu)
    s = s.transpose(1, 2).to(torch.bfloat16)
    size_k, size_n = s.size(-2) * group_size, s.size(-1)
    scale_perm, scale_perm_single = get_scale_perms()
    if group_size < size_k and group_size != -1:
        s = s.reshape((s.size(0), -1, len(scale_perm)))[:, :, scale_perm]
    else:
        s = s.reshape((s.size(0), -1, len(scale_perm_single)))[:, :, scale_perm_single]
    s = s.reshape((s.size(0), -1, size_n)).contiguous()
    if group_size != 16:
        s = mxfp4_marlin_process_scales(s)
    else:
        s = nvfp4_marlin_process_scales(s)
    s.group_size = group_size
    return s

def marlin_nvfp4_process_global_scale(global_scale):
    global_scale = global_scale.to(torch.bfloat16)
    assert global_scale.dtype in [torch.half, torch.bfloat16]
    fp4_exponent = 2
    target_exponent = 8
    # exponent_bias_fp16 = 2 ** 4 - 2 ** 1 = 14
    # exponent_bias_bf16 = 2 ** 7 - 2 ** 1 = 126
    exponent_bias = 2 ** (target_exponent - 1) - 2 ** (fp4_exponent - 1)
    return global_scale * (2.0 ** (exponent_bias - 7))

def marlin_unpack(marlin_weight, group_size):
    # reverse process of `marlin_pack()`
    # group_size 32/16 for mxfp4/nvfp4

    def _get_marlin_perms(device):
        perm = []
        for i in range(32):
            perm1 = []
            col = i // 4
            for block in [0, 1]:
                for row in [
                    2 * (i % 4),
                    2 * (i % 4) + 1,
                    2 * (i % 4 + 4),
                    2 * (i % 4 + 4) + 1
                ]:
                    perm1.append(16 * row + col + 8 * block)
            for j in range(4):
                perm.extend([p + 256 * j for p in perm1])
        perm = np.array(perm)
        interleave = np.array([0, 2, 4, 6, 1, 3, 5, 7])
        perm = perm.reshape((-1, 8))[:, interleave].ravel()
        return torch.from_numpy(perm).long().to(device)
    
    def pack_weight_4d(w_unpacked):
        # reverse process of `unpack_weight_4d()``
        w = w_unpacked.transpose(1, 2).contiguous()
        E, N, K = w.shape
        if K % group_size != 0:
            raise ValueError(f"In_features ({K}) should be able to be divided by group_size ({group_size})")
        
        # 2. Pack 4-bit into 8-bit
        # [E, N, K] -> [E, N, K//2, 2]
        w_pairs = w.view(E, N, K // 2, 2)
        low = w_pairs[..., 0].to(torch.uint8)
        high = w_pairs[..., 1].to(torch.uint8)
        # uint8: low | (high << 4)
        packed_val = low | (high << 4)
        # shape: [E, N, K//2] -> [E, N, G, half_group_size]
        # where G = K // group_size, half_group_size = group_size // 2
        packed_weight = packed_val.view(E, N, K // group_size, group_size // 2)
        return packed_weight

    device = marlin_weight.device
    E, K_div_16, out_cols = marlin_weight.shape
    
    # pack_marlin : out_cols = N * 16 // 8 = N * 2
    N = out_cols // 2
    K = K_div_16 * 16
    tile = 16
    
    # shape: [E, K//16, N*2] -> [E, K//16, N*2, 8]
    unpacked_bits = torch.zeros((E, K_div_16, out_cols, 8), dtype=torch.int32, device=device)
    
    for i in range(8):
        # marlin_weight: [E, K//16, out_cols]
        # unpacked_bits[..., i]: [E, K//16, out_cols]
        unpacked_bits[..., i] = (marlin_weight >> (4 * i)) & 0xF
        
    w = unpacked_bits.view(E, K_div_16, -1)
    
    perm = _get_marlin_perms(device)
    inv_perm = torch.argsort(perm)
    w_flat = w.reshape(-1, perm.numel())
    w_unpermuted = w_flat[:, inv_perm]
    w = w_unpermuted.reshape(E, K_div_16, N * tile)
    w = w.reshape(E, K // tile, N // tile, tile, tile)
    w = w.permute(0, 1, 3, 2, 4)
    
    w_unpacked = pack_weight_4d(w.reshape(E, K, N))
    
    return w_unpacked

def marlin_nvfp4_pack(x: torch.Tensor) -> torch.Tensor:
    """
    Input:  x.shape = [E, N, K // 8, 8], dtype=torch.uint8
    Output: y.shape = [E, K // 16, N << 1], dtype=torch.int32
    """
    E, N, G, half_group_size = x.shape
    K = G * half_group_size * 2

    x_view = x.view(E, N, K // 2)
    w_unpacked = torch.empty((E, N, K), dtype=torch.int8, device=x.device)
    w_unpacked[..., 0::2] = x_view & 0x0F
    w_unpacked[..., 1::2] = (x_view >> 4) & 0x0F
    w = w_unpacked.transpose(1, 2).contiguous()
    w = w.reshape(E, K // 16, 2, 4, 2, N // 64, 4, 2, 8)
    w = w.permute(0, 1, 5, 8, 3, 6, 4, 7, 2)
    w_pack = w.reshape(E, K // 16, N * 2, 8).to(torch.int32)
    marlin_weight = w_pack[..., 0].clone()
    for i in range(1, 8):
        marlin_weight |= (w_pack[..., i] << (4 * i))
    return marlin_weight


def marlin_nvfp4_revert(y: torch.Tensor) -> torch.Tensor:
    """
    Input: y.shape = [E, K // 16, N << 1], dtype=torch.int32
    Output:  x.shape = [E, N, K // 8, 8], dtype=torch.uint8
    """
    E = y.shape[0]
    K = y.shape[1] * 16
    N = y.shape[2] // 2

    w_pack = torch.empty((E, K // 16, N * 2, 8), dtype=torch.int8, device=y.device)
    for i in range(8):
        w_pack[..., i] = (y >> (4 * i)) & 0x0F
    w_permuted = w_pack.reshape(E, K // 16, N // 64, 8, 4, 4, 2, 2, 2)
    w_reshaped = w_permuted.permute(0, 1, 8, 4, 6, 2, 5, 7, 3).contiguous()
    w = w_reshaped.reshape(E, K, N)
    w_unpacked = w.transpose(1, 2).contiguous()
    low = w_unpacked[..., 0::2]
    high = w_unpacked[..., 1::2]
    packed_bytes = low.to(torch.uint8) | (high.to(torch.uint8) << 4)
    x = packed_bytes.view(E, N, K // 16, 8).to(torch.uint8)
    return x

def marlin_nvfp4_revert_transposed(y: torch.Tensor, nvfp4_groupscale: torch.Tensor, nvfp4_oscale: torch.Tensor) -> torch.Tensor:
    """
    Input: y.shape = [E, K // 16, N * 2], dtype=torch.int32
           nvfp4_groupscale.shape = [E, N, K // group_size], dtype=torch.float8_e4m3fn
           nvfp4_oscale.shape = [E, L], dtype=torch.float32
    Output:  z.shape = [E, N, K], dtype=torch.bfloat16
    """
    x = marlin_nvfp4_revert(y)
    E, N, _, _ = x.shape
    K = y.shape[1] * 16
    x_flat = x.view(E, N, K // 2)
    low = x_flat & 0x0F
    high = (x_flat >> 4) & 0x0F
    unpacked = torch.stack((low, high), dim=-1).view(E, N, K)
    lut = torch.tensor([
         0.0,  0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0
    ], dtype=torch.bfloat16, device=y.device)
    w_bf16 = lut[unpacked.to(torch.long)]
    group_size = K // nvfp4_groupscale.shape[-1]
    scales_bf16 = nvfp4_groupscale.to(torch.bfloat16)
    scales_expanded = scales_bf16.repeat_interleave(group_size, dim=-1)
    z = w_bf16 * scales_expanded
    z = z.view(z.size(0), nvfp4_oscale.size(1), -1, z.size(-1)).float() * nvfp4_oscale.unsqueeze(-1).unsqueeze(-1)
    return z.flatten(1, 2).to(scales_bf16.dtype)


def __getattr__(name):
  fn = getattr(torch.ops.tutel_ops, name)
  return torch.compiler.disable(fn, recursive=True)

