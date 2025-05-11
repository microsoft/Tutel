# Tutel

Tutel MoE: An Optimized Mixture-of-Experts Implementation, also the first parallel solution proposing ["No-penalty Parallism/Sparsity/Capacity/.. Switching"](https://mlsys.org/media/mlsys-2023/Slides/2477.pdf) for modern training and inference that have dynamic behaviors.

- Supported Framework: Pytorch (recommend: >= 1.10)
- Supported GPUs: CUDA(fp64/fp32/fp16/bf16), ROCm(fp64/fp32/fp16/bf16)
- Supported CPU: fp64/fp32

#### - ***The First to Support DeepSeek 671B NVFP4 Inference using A100/A800/H100/MI300 resources.***
#### - ***Fastest "Thinking" on MI300X: Complete a 4K reasoning answer in 39 sec, compared with SGLANG in 1 min 35 sec.***

#### FP4 Model: [nvidia/DeepSeek-R1-FP4](https://huggingface.co/nvidia/DeepSeek-R1-FP4)
> |  ***Machine Type*** | ***TRT-LLM***  | ***SGLANG***  |  ***Tutel***  |
> |  ----  | ----  | ----  | ----  |
> | $"A100 \times 8" or "A800 \times 8"$ | N/A | N/A | 63 Generation TPS (bsz = 1) |
> | $"H100 \times 8" or "H800 \times 8"$ | N/A | N/A | 81 Generation TPS (bsz = 1) |
> | $MI300 \times 8$  | N/A | N/A | 112 Generation TPS (bsz = 1) |

#### FP8 Model: [DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1) or [DeepSeek-V3-0324](https://huggingface.co/deepseek-ai/DeepSeek-V3-0324) or [DeepSeek-Prover-V2-671B](https://huggingface.co/deepseek-ai/DeepSeek-Prover-V2-671B)
> |  ***Machine Type*** | ***TRT-LLM***  | ***SGLANG***  |  ***Tutel***  |
> |  ----  | ----  | ----  | ----  |
> | $"A100 \times 8" or "H100 \times 8"$ | OoM | OoM | OoM |
> | $MI300 \times 8$  | N/A | 42 Generation TPS (bsz = 1) | 110 Generation TPS (bsz = 1) |

#### Run DeepSeek 671B in Docker:
```sh
>> Example:
  huggingface-cli download nvidia/DeepSeek-R1-FP4 --local-dir ./nvidia/DeepSeek-R1-FP4

  # For A100/A800/H100/H800/H20/H200 (80G x 8):
  docker run -it --rm --ipc=host --net=host --shm-size=8g --ulimit memlock=-1 --ulimit stack=67108864 --gpus=all \
      -v /:/host -w /host$(pwd) tutelgroup/deepseek-671b:a100x8-chat-20250508 \
      --try_path ./nvidia/DeepSeek-R1-FP4 \
      --serve --listen_port 8000 \
      --prompt "Calculate the indefinite integral of 1/sin(x) + x"

  # For AMD MI300x NVFP4 (192G x 8):
  docker run -it --rm --ipc=host --net=host --shm-size=8g --ulimit memlock=-1 --ulimit stack=67108864 \
      --device=/dev/kfd --device=/dev/dri --group-add=video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
      -v /:/host -w /host$(pwd) tutelgroup/deepseek-671b:mi300x8-chat-20250508 \
      --try_path ./nvidia/DeepSeek-R1-FP4 \
      --try_path ./deepseek-ai/DeepSeek-R1 \
      --try_path ./deepseek-ai/DeepSeek-V3-0324 \
      --try_path ./deepseek-ai/DeepSeek-Prover-V2-671B \
      --try_path ./microsoft/MAI-DS-R1 \
      --serve --listen_port 8000 \
      --prompt "Calculate the indefinite integral of 1/sin(x) + x"
```

## What's New:

> Tutel v0.4.2: Add R1-FP4 for "NVIDIA A100/A800/H100/H800 (80G) x 8" and "AMD MI300 x 8":

> Tutel v0.3.3: Add all-to-all benchmark:
```sh
  >> Example:

    python3 -m torch.distributed.run --nproc_per_node=8 -m tutel.examples.bandwidth_test --size_mb=256
```

> Tutel v0.3.2: Add tensorcore option for extra benchmarks / Extend the example for custom experts / Allow NCCL timeout settings:
```sh
  >> Example of using tensorcore:

    python3 -m tutel.examples.helloworld --dtype=float32
    python3 -m tutel.examples.helloworld --dtype=float32 --use_tensorcore

    python3 -m tutel.examples.helloworld --dtype=float16
    python3 -m tutel.examples.helloworld --dtype=float16 --use_tensorcore

  >> Example of custom gates/experts:
    python3 -m tutel.examples.helloworld_custom_gate_expert --batch_size=16

  >> Example of NCCL timeout settings:
    TUTEL_GLOBAL_TIMEOUT_SEC=60 python3 -m torch.distributed.run --nproc_per_node=8 -m tutel.examples.helloworld --use_tensorcore

```

> Tutel v0.3.1: Add NCCL all_to_all_v and all_gather_v for arbitrary-length message transfers:
```sh
  >> Example:
    # All_to_All_v:
    python3 -m torch.distributed.run --nproc_per_node=2 --master_port=7340 -m tutel.examples.nccl_all_to_all_v
    # All_Gather_v:
    python3 -m torch.distributed.run --nproc_per_node=2 --master_port=7340 -m tutel.examples.nccl_all_gather_v

  >> How to:
    net.batch_all_to_all_v([t_x_cuda, t_y_cuda, ..], common_send_counts)
    net.batch_all_gather_v([t_x_cuda, t_y_cuda, ..])
```

> Tutel v0.3: Add Megablocks solution to improve decoder inference on single-GPU with num_local_expert >= 2:
```sh
  >> Example (capacity_factor=0 required by dropless-MoE):
    # Using BatchMatmul:
    python3 -m tutel.examples.helloworld --megablocks_size=0 --batch_size=1 --num_tokens=32 --top=1 --eval --num_local_experts=128 --capacity_factor=0
    # Using Megablocks with block_size = 1:
    python3 -m tutel.examples.helloworld --megablocks_size=1 --batch_size=1 --num_tokens=32 --top=1 --eval --num_local_experts=128 --capacity_factor=0
    # Using Megablocks with block_size = 2:
    python3 -m tutel.examples.helloworld --megablocks_size=2 --batch_size=1 --num_tokens=32 --top=1 --eval --num_local_experts=128 --capacity_factor=0

  >> How to:
    self._moe_layer.forward(x, .., megablocks_size=1)         # Control the switch of megablocks_size (0 for disabled)
```

> Tutel v0.2: Allow most configurations to be dynamic switchable with free cost:
```sh
  >> Example:
    python3 -m torch.distributed.run --nproc_per_node=8 -m tutel.examples.helloworld_switch --batch_size=16

  >> How to:
    self._moe_layer.forward(x, .., a2a_ffn_overlap_degree=2)  # Control the switch of overlap granularity (1 for no overlapping)
    self._moe_layer.forward(x, .., adaptive_r=1)              # Control the switch of parallelism (0 for DP, 1 for DP + EP, W / E for MP + EP, else for DP + MP + EP)
    self._moe_layer.forward(x, .., capacity_factor=1)         # Control the switch of capacity_volume (positive for padding, negative for no-padding, 0 for dropless)
    self._moe_layer.forward(x, .., top_k=1)                   # Control the switch of top_k sparsity
```

> Tutel v0.1: Optimize the Einsum Complexity of Data Dispatch Encoding and Decoding, add 2DH option to deal with All-to-All at scale:
```sh
  >> Example (suggest enabling 2DH only at scale, note that the value of --nproc_per_node MUST equal to total physical GPU counts per node, e.g. 8 for A100x8):
    python3 -m torch.distributed.run --nproc_per_node=8 -m tutel.examples.helloworld --batch_size=16 --use_2dh
```

-----------
## Getting Started

### 1. Prepare Pytorch (if applicable):
```
* Prepare Recommended Pytorch >= 2.0.0:
        #  Windows/Linux Pytorch for NVIDIA CUDA >= 11.7:
        python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        #  Linux Pytorch for AMD ROCm >= 6.2.2:
        python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2.2
        #  Windows/Linux Pytorch for CPU:
        python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 2. Tutel Installation:
```
* Option-1: Install Tutel Online:

        $ python3 -m pip uninstall tutel -y
        $ python3 -m pip install -v -U --no-build-isolation git+https://github.com/microsoft/tutel@main

* Option-2: Build Tutel from Source:

        $ git clone https://github.com/microsoft/tutel --branch main
        $ python3 -m pip uninstall tutel -y
        $ python3 ./tutel/setup.py install --user
```

### 3. Quick Test for Single Device / CPU:
```
* Quick Test on Single-GPU:

        $ python3 -m tutel.examples.helloworld --batch_size=16               # Test Tutel-optimized MoE + manual distribution
        $ python3 -m tutel.examples.helloworld_ddp --batch_size=16           # Test Tutel-optimized MoE + Pytorch DDP distribution (requires: Pytorch >= 1.8.0)
        $ python3 -m tutel.examples.helloworld_ddp_tutel --batch_size=16     # Test Tutel-optimized MoE + Tutel DDP distribution (ZeRO on optimizors)
        $ python3 -m tutel.examples.helloworld_amp --batch_size=16           # Test Tutel-optimized MoE with AMP data type + manual distribution
        $ python3 -m tutel.examples.helloworld_custom_gate_expert --batch_size=16 # Test Tutel-optimized MoE + custom defined gate/expert layer
        $ python3 -m tutel.examples.helloworld_from_scratch                  # Test Custom MoE implementation from scratch
        $ python3 -m tutel.examples.moe_mnist                                # Test MoE layer in end-to-end MNIST dataset
        $ python3 -m tutel.examples.moe_cifar10                              # Test MoE layer in end-to-end CIFAR10 dataset

        (If building from source, the following method also works:)
        $ python3 ./tutel/examples/helloworld.py --batch_size=16
        ..
```

### 4. Quick Test for 8 GPUs within 1 Machine:
```
        $ python3 -m torch.distributed.run --nproc_per_node=8 -m tutel.examples.helloworld --batch_size=16
```

### 5. Quick Test for Multiple GPUs across Machines:
```
* Run Tutel MoE in Distributed Mode:

        (Option A - Torch launcher for `Multi-Node x Multi-GPU`:)
        $ ssh <node-ip-0> python3 -m torch.distributed.run --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=<node-ip-0> -m tutel.examples.helloworld --batch_size=16
        $ ssh <node-ip-1> python3 -m torch.distributed.run --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=<node-ip-0> -m tutel.examples.helloworld --batch_size=16

        (Option B - Tutel launcher for `Multi-Node x Multi-GPU`, requiring package `openmpi-bin`:)
        # << Single Node >>
        $ mpiexec -bind-to none -host localhost -x LOCAL_SIZE=8 python3 -m tutel.launcher.run -m tutel.examples.helloworld_ddp_tutel --batch_size=16
        $ mpiexec -bind-to none -host localhost -x LOCAL_SIZE=8 python3 -m tutel.launcher.run -m tutel.examples.moe_mnist
        $ mpiexec -bind-to none -host localhost -x LOCAL_SIZE=8 python3 -m tutel.launcher.run -m tutel.examples.moe_cifar10
        ...

        # << MPI-based launch for GPU backend>>
        $ mpiexec -bind-to none -host <node-ip-0>,<node-ip-1>,.. -x MASTER_ADDR=<node-ip-0> -x LOCAL_SIZE=8 python3 -m tutel.launcher.run -m tutel.examples.helloworld --batch_size=16

        # << MPI-based Launch for CPU backend>>
        $ mpiexec -bind-to none -host localhost -x LOCAL_SIZE=1 -x OMP_NUM_THREADS=1024 python3 -m tutel.launcher.run -m tutel.examples.helloworld --batch_size=16 --device cpu
```

-----------

### Advance: Convert Checkpoint Files for Different World Sizes:
Documentation for checkpoint conversion has been moved [here](doc/CHECKPOINT.md).

### Examples: How to import Tutel-optimized MoE in Pytorch:
```
# Input Example:
import torch
x = torch.ones([6, 1024], device='cuda:0')

# Create MoE:
from tutel import moe as tutel_moe
moe_layer = tutel_moe.moe_layer(
    gate_type={'type': 'top', 'k': 2},
    model_dim=x.shape[-1],
    experts={
        'num_experts_per_device': 2,
        'type': 'ffn', 'hidden_size_per_expert': 2048, 'activation_fn': lambda x: torch.nn.functional.relu(x)
    },
    scan_expert_func = lambda name, param: setattr(param, 'skip_allreduce', True),
)

# Cast to GPU
moe_layer = moe_layer.to('cuda:0')

# In distributed model, you need further skip doing allreduce on global parameters that have `skip_allreduce` mask, 
# e.g.
#    for p in moe_layer.parameters():
#        if hasattr(p, 'skip_allreduce'):
#            continue
#        dist.all_reduce(p.grad)


# Forward MoE:
y = moe_layer(x)

print(y)
```

### Reference
You can consult this [paper](https://arxiv.org/pdf/2206.03382.pdf) below to get to know more technical details about Tutel:
```
@article {tutel,
author = {Changho Hwang and Wei Cui and Yifan Xiong and Ziyue Yang and Ze Liu and Han Hu and Zilong Wang and Rafael Salas and Jithin Jose and Prabhat Ram and Joe Chau and Peng Cheng and Fan Yang and Mao Yang and Yongqiang Xiong},
title = {Tutel: Adaptive Mixture-of-Experts at Scale},
year = {2022},
month = jun,
journal = {CoRR},
volume= {abs/2206.03382},
url = {https://arxiv.org/pdf/2206.03382.pdf},
}
```

### Usage of MOELayer:
```
* Usage of MOELayer Args:

        gate_type        : dict-type gate description, e.g. {'type': 'top', 'k': 2, 'capacity_factor': -1.5, ..},
                              or a list of dict-type gate descriptions, e.g. [{'type': 'top', 'k', 2}, {'type': 'top', 'k', 2}],
                              the value of k in top-gating can be also negative, like -2, which indicates one GPU will hold 1/(-k) parameters of an expert
                              capacity_factor X can be positive (factor = X), zero (factor = max(needed_volumes)) or negative (factor = min(-X, max(needed_volumes))).
        model_dim        : the number of channels for MOE's input tensor
        experts          : a dict-type config for builtin expert network
        scan_expert_func : allow users to specify a lambda function to iterate each experts param, e.g. `scan_expert_func = lambda name, param: setattr(param, 'expert', True)`
        result_func      : allow users to specify a lambda function to format the MoE output and aux_loss, e.g. `result_func = lambda output: (output, output.l_aux)`
        group            : specify the explicit communication group of all_to_all
        seeds            : a tuple containing a tripple of int to specify manual seed of (shared params, local params, others params after MoE's)
        a2a_ffn_overlap_degree : the value to control a2a overlap depth, 1 by default for no overlap, 2 for overlap a2a with half gemm, ..
        parallel_type    : the parallel method to compute MoE, valid types: 'auto', 'data', 'model'
        pad_samples      : whether do auto padding on newly-coming input data to maximum data size in history

* Usage of dict-type Experts Config:

        num_experts_per_device : the number of local experts per device (by default, the value is 1 if not specified)
        hidden_size_per_expert : the hidden size between two linear layers for each expert (used for type == 'ffn' only)
        type             : available built-in experts implementation, e.g: ffn
        activation_fn    : the custom-defined activation function between two linear layers (used for type == 'ffn' only)
        has_fc1_bias     : If set to False, the expert bias parameters `batched_fc1_bias` is disabled. Default: True
        has_fc2_bias     : If set to False, the expert bias parameters `batched_fc2_bias` is disabled. Default: True
```

### Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

### Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
