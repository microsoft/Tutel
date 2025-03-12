#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import time
import argparse

from tutel import system, net

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--count', type=int, default=229376)
parser.add_argument('--loop', type=int, default=50)
parser.add_argument('--warmup', type=int, default=5, help='Number of warmup iterations')
args = parser.parse_args()

parallel_env = system.init_data_model_parallel(backend='nccl' if args.device == 'cuda' else 'gloo')
local_device = parallel_env.local_device

x = torch.randn([args.count], device=local_device, dtype=torch.float32)

if args.device == 'cuda':
  wait = lambda: torch.cuda.synchronize() or time.perf_counter()
else:
  wait = lambda: time.perf_counter()

# Warmup phase (excluded from any measurement)
with torch.no_grad():
  for _ in range(args.warmup + args.loop):
    torch.ops.tutel_ops.test_allreduce_bf16(args.count)

