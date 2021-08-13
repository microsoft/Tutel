import torch
import custom_kernel
import os, tempfile

try:
    from torch.utils.cpp_extension import IS_HIP_EXTENSION
except:
    IS_HIP_EXTENSION = False

class JitKernel:
    @staticmethod
    def create(source):
        if not hasattr(JitKernel, '__CTX__'):
            torch.cuda.init()
            JitKernel.__CTX__ = 0
        __ctx__ = JitKernel.__CTX__
        JitKernel.__CTX__ += 1

        temp_loc = f'{tempfile.mktemp()}-{__ctx__}.MoE'
        with open(temp_loc, 'w') as fp:
            if IS_HIP_EXTENSION:
              fp.write('#include <hip/hip_runtime.h>\n#include <hip/hip_fp16.h>\n')
            else:
              fp.write('#include <cuda_runtime.h>\n#include <cuda_fp16.h>\n')
            fp.write(source)
        os.rename(temp_loc, f'/tmp/{__ctx__}.cu')

        def func(*inputs):
            custom_kernel.invoke(inputs, __ctx__)
        return func

