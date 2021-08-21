import torch
import torch.distributed as dist

import os, tempfile

assert torch.cuda.is_available() == True, "This version of Tutel MoE only supports CUDA. More backends will be supported soon."

try:
    import tutel_custom_kernel
except:
    raise Exception("Cannot import JIT optimized kernels. Did you forget to install Custom Kernel Extension?")

try:
    from torch.utils.cpp_extension import IS_HIP_EXTENSION
except:
    IS_HIP_EXTENSION = False

try:
	dist_rank = dist.get_rank()
except:
	dist_rank = 0

class JitKernel:
    @staticmethod
    def create(source):
        if not hasattr(JitKernel, '__CTX__'):
            torch.cuda.init()
            JitKernel.__CTX__ = 0
        __ctx__ = JitKernel.__CTX__
        JitKernel.__CTX__ += 1

        key = dist_rank
        temp_loc = f'{tempfile.mktemp()}-{__ctx__}.MoE'
        with open(temp_loc, 'w') as fp:
            if IS_HIP_EXTENSION:
              fp.write('#include <hip/hip_runtime.h>\n#include <hip/hip_fp16.h>\n')
            else:
              fp.write('#include <cuda_runtime.h>\n#include <cuda_fp16.h>\n')
            fp.write(source)
        os.rename(temp_loc, f'/tmp/{__ctx__}-{key}.cu')

        def func(*inputs):
            tutel_custom_kernel.invoke(inputs, __ctx__, key)
        return func

