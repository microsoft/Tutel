#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import time
import torch
import torch.nn.functional as F

from tutel import system
from tutel import moe
from tutel import net

if torch.cuda.is_available():
  dist = system.init_data_model_parallel(backend='nccl')
else:
  dist = system.init_data_model_parallel(backend='gloo')

outer_batch, sequence_length = 16, 1024
num_samples = outer_batch * sequence_length
model_dim, hidden_size = 2048, 2048
num_local_experts = 4
num_global_experts = num_local_experts * dist.global_size


class CustomGate(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1)
        self.register_parameter(name='wg', param=torch.nn.Parameter(torch.randn([model_dim, num_global_experts]) * 1e-3))

    def forward(self, x):
        return torch.matmul(x, self.wg)

class CustomExpert(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(dist.global_rank + 1)
        self.register_parameter(name='batched_fc1_w', param=torch.nn.Parameter(torch.randn([num_local_experts, model_dim, hidden_size]) * 1e-3))
        self.register_parameter(name='batched_fc2_w', param=torch.nn.Parameter(torch.randn([num_local_experts, hidden_size, model_dim]) * 1e-3))
        self.register_parameter(name='batched_fc1_bias', param=torch.nn.Parameter(torch.zeros([num_local_experts, 1, hidden_size])))
        self.register_parameter(name='batched_fc2_bias', param=torch.nn.Parameter(torch.zeros([num_local_experts, 1, model_dim])))
        for x in self.parameters(): setattr(x, 'skip_allreduce', True)

    def forward(self, x):
        y = torch.add(torch.matmul(x, self.batched_fc1_w), self.batched_fc1_bias)
        y = F.relu(y)
        y = torch.add(torch.matmul(y, self.batched_fc2_w), self.batched_fc2_bias)
        return y

class CustomMoE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = CustomGate()
        self.expert = CustomExpert()

    def forward(self, x, k=2):
        logits = self.gate(x)
        scores = F.softmax(logits, dim=-1)

        crit, l_aux = moe.top_k_routing(scores, top_k=k, capacity_factor=0)

        topk_ids, total_tokens = moe.get_topk_selection(crit)
        print(f'\n>> Get Topk Selection Result of Shape[B * S, K] = {topk_ids.shape}:\n{topk_ids}')
        print(f'\n>> Get Total Tokens per Expert of Shape[E] = {total_tokens.shape}: {total_tokens}')

        sample_index = moe.get_reversed_sample_ids(crit, return_id_type='sample_id')
        top_index = moe.get_reversed_sample_ids(crit, return_id_type='top_id')
        print(f'\n>> Get Reversed Sample Index Map of Shape[E, CAPACITY] = {sample_index.shape} (-1 for padded tokens):\n{sample_index}')
        print(f'\n>> Get Reversed TopK Index Map of Shape[E, CAPACITY] = {top_index.shape} (-1 for padded tokens):\n{top_index}')

        sample_outer_batch_index = sample_index // sequence_length
        sample_inner_sequence_index = torch.where(sample_index >= 0, sample_index % sequence_length, sample_index)
        print(f'\n>> Get Outer-batch Index Map of Shape[E, CAPACITY] = {sample_outer_batch_index.shape} (-1 for padded tokens):\n{sample_outer_batch_index}')
        print(f'\n>> Get Inner-sequence Index Map of Shape[E, CAPACITY] = {sample_inner_sequence_index.shape} (-1 for padded tokens):\n{sample_inner_sequence_index}')

        y = moe.fast_encode(x, crit)
        y = net.all_to_all(y, 1, 0)

        print(f'\n>> Forwarding Shuffled Expert Input of Shape[E, CAPACITY, M] = {y.shape} ..\n')
        y = self.expert(y)

        y = net.all_to_all(y, 0, 1)
        output = moe.fast_decode(y, crit)
        return output, l_aux

model = CustomMoE().to(dist.local_device)

torch.manual_seed(dist.global_rank + 1)
data = torch.randn([num_samples, model_dim], device=dist.local_device)
label = torch.LongTensor(num_samples).random_(1).to(dist.local_device)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

for i in range(10):
    t_start = system.record_time()

    optimizer.zero_grad()
    result, l_aux = model(data)
    result = F.log_softmax(result, dim=1)
    loss = F.nll_loss(result, label) + 0.0001 * l_aux
    loss.backward()

    for p in model.parameters():
        if not hasattr(p, 'skip_allreduce'):
            p.grad = net.simple_all_reduce(p.grad)
    optimizer.step()

    t_stop = system.record_time()

    dist.dist_print('STEP-%d: loss = %.5f, step_time = %.3f s' % (i, loss, t_stop - t_start))

