#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import time
import argparse

from tutel import system, net

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--size_mb', type=int, default=256)
parser.add_argument('--loop', type=int, default=50)
parser.add_argument('--warmup', type=int, default=5, help='Number of warmup iterations')
args = parser.parse_args()

parallel_env = system.init_data_model_parallel(backend='nccl' if args.device == 'cuda' else 'gloo')
local_device = parallel_env.local_device

x = torch.randn([((args.size_mb + 3) // 4 * 1024 * 1024 + parallel_env.global_size - 1) // parallel_env.global_size * parallel_env.global_size], device=local_device, dtype=torch.float32)

if args.device == 'cuda':
  wait = lambda: torch.cuda.synchronize() or time.perf_counter()
else:
  wait = lambda: time.perf_counter()

# Warmup phase (excluded from any measurement)
with torch.no_grad():
  for _ in range(args.warmup):
    net.simple_all_to_all(x.view(parallel_env.global_size, -1))
    net.simple_all_reduce(x.view(-1), inplace=True)
    net.simple_all_gather(x.view(parallel_env.global_size, -1)[parallel_env.global_rank])
    net.simple_reduce_scatter(x.view(parallel_env.global_size, -1))

# Measurement phase - accumulate times and compute average bandwidth
with torch.no_grad():
  total_time_a2a = 0.0
  total_time_ar = 0.0
  total_time_ag = 0.0
  total_time_rs = 0.0

  for _ in range(args.loop):
    # AllToAll
    t0 = wait()
    net.simple_all_to_all(x.view(parallel_env.global_size, -1))
    t1 = wait()
    total_time_a2a += (t1 - t0)
    
    # AllReduce
    t0 = wait()
    net.simple_all_reduce(x.view(-1), inplace=True)
    t1 = wait()
    total_time_ar += (t1 - t0)
    
    # AllGather
    t0 = wait()
    net.simple_all_gather(x.view(parallel_env.global_size, -1)[parallel_env.global_rank])
    t1 = wait()
    total_time_ag += (t1 - t0)

    # ReduceScatter
    t0 = wait()
    net.simple_reduce_scatter(x.view(parallel_env.global_size, -1))
    t1 = wait()
    total_time_rs += (t1 - t0)

  # Calculate and print the average bandwidth for each collective
  parallel_env.dist_print(
    f'AllToAll average bandwidth across {parallel_env.global_size} node(s) = '
    f'{((x.numel() * 4) * 1e-9 * args.loop) / total_time_a2a:.4f} GB/s'
  )
  parallel_env.dist_print(
    f'AllReduce average bandwidth across {parallel_env.global_size} node(s) = '
    f'{((x.numel() * 4) * 1e-9 * args.loop) / total_time_ar:.4f} GB/s'
  )
  parallel_env.dist_print(
    f'AllGather average bandwidth across {parallel_env.global_size} node(s) = '
    f'{((x.numel() * 4) * 1e-9 * args.loop) / total_time_ag:.4f} GB/s'
  )
  parallel_env.dist_print(
    f'ReduceScatter average bandwidth across {parallel_env.global_size} node(s) = '
    f'{((x.numel() * 4) * 1e-9 * args.loop) / total_time_rs:.4f} GB/s'
  )