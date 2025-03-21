# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import torch
import tutel_custom_kernel

if 'OP_LOADER' not in os.environ:
    os.environ['OP_LOADER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.')

def input_to_float8(
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

