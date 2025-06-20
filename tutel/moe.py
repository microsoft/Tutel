# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


# Low-level Ops
from .jit_kernels.gating import fast_cumsum_sub_one
from .impls.fast_dispatch import fast_dispatcher, extract_critical, fast_encode, fast_decode, get_reversed_sample_ids, get_topk_selection

top_k_routing = extract_critical

# High-level Ops
from .impls.moe_layer import moe_layer
