// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// HSACO reference from: https://github.com/ROCm/aiter
//   SPDX-License-Identifier: MIT
//   Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include <hip/hip_runtime.h>
#define CUSTOM_MLA_DECODE

#define HIP_CALL(call)                                                                                                           \
    do                                                                                                                           \
    {                                                                                                                            \
        hipError_t err = call;                                                                                                   \
        if (err != hipSuccess)                                                                                                   \
        {                                                                                                                        \
            printf("\n[AITER] %s:%d fail to call %s ---> [HIP error](%s)\n", __FILE__, __LINE__, #call, hipGetErrorString(err)); \
            exit(0);                                                                                                             \
        }                                                                                                                        \
    } while (0)

struct p3
{
    unsigned int _p0;
    unsigned int _p1;
    unsigned int _p2;
};
struct p2
{
    unsigned int _p0;
    unsigned int _p1;
};
struct p1
{
    unsigned int _p0;
};
struct AiterAsmKernelArgs
{
    void *args_ptr;
    void *arg_size_ptr;
    int gdx;
    int gdy;
    int gdz;
    int bdx;
    int bdy;
    int bdz;
    const hipStream_t stream;
};

class AiterAsmKernel
{
private:
    hipModule_t module;
    hipFunction_t kernel_func;

public:
    AiterAsmKernel(const char *name, const char *hsaco)
    {
        HIP_CALL(hipModuleLoadData(&module, hsaco));
        HIP_CALL(hipModuleGetFunction(&kernel_func, module, name));
    };

    ~AiterAsmKernel()
    {
        HIP_CALL(hipModuleUnload(module));
    }

    void launch_kernel(const AiterAsmKernelArgs &kargs)
    {
        void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, kargs.args_ptr,
                          HIP_LAUNCH_PARAM_BUFFER_SIZE, kargs.arg_size_ptr,
                          HIP_LAUNCH_PARAM_END};

        HIP_CALL(hipModuleLaunchKernel(kernel_func,
                                       kargs.gdx, kargs.gdy, kargs.gdz,
                                       kargs.bdx, kargs.bdy, kargs.bdz,
                                       0, kargs.stream, nullptr, (void **)&config));
    };

    void launch_kernel2(const AiterAsmKernelArgs &kargs, void **args)
    {
        HIP_CALL(cuLaunchKernel(kernel_func,
                                       kargs.gdx, kargs.gdy, kargs.gdz,
                                       kargs.bdx, kargs.bdy, kargs.bdz,
                                       0, kargs.stream, args, nullptr));
    };
};

struct __attribute__((packed)) KernelArgs
{
    void *ptr_R;
    p2 _p0;
    void *ptr_LSE;
    p2 _p1;
    void *ptr_Q;
    p2 _p2;
    void *ptr_KV;
    p2 _p3;
    void *ptr_LTP;
    p2 _p4;
    void *ptr_LTD;
    p2 _p5;
    void *ptr_LTL;
    p2 _p6;
    float scalar;
    p3 _p12;
    unsigned int s_MQA;
    p3 _p13;
    unsigned int s_kv_split;
    p3 _p14;
    unsigned int s_Q_Bs;
    p3 _p15;
    unsigned int s_Bs;
    p3 _p16;
    unsigned int s_log2_plen;
    p3 _p17;
};

struct __attribute__((packed)) Kernel2Args
{
    void *p0, *p1, *p2, *p3;
    unsigned int s0, s1, s2, s3, s4;
};

static torch::Tensor mla_decode_fwd(torch::Tensor Q, torch::Tensor KV, torch::Tensor kv_indptr, double softmax_scale) {
    Q = Q.squeeze(1);
    KV = KV.view({-1, 1, 1, KV.size(-1)});
    static const int splits = 32;

    AiterAsmKernel *impl_comb = nullptr;
    if (splits == 32) {
#include "mla_stage2_a16w16_bf16_kvsplit32.h"
      static AiterAsmKernel impl_a16w16_bf16("mla_stage2_a16w16_bf16", mla_stage2_a16w16_bf16);
      impl_comb = &impl_a16w16_bf16;
    } else {
      CHECK_EQ(splits, 16);
#include "mla_stage2_a16w16_bf16_kvsplit16.h"
      static AiterAsmKernel impl_a16w16_bf16("mla_stage2_a16w16_bf16", mla_stage2_a16w16_bf16);
      impl_comb = &impl_a16w16_bf16;
    }

    static torch::Tensor kv_page_indices, kv_last_page_lens, splitData, splitLse, output;
    if (splitData.numel() == 0) {
      int stride = KV.size(0);
      kv_page_indices = torch::arange(0, stride + 1, torch::TensorOptions().dtype(torch::kInt32).device(Q.device()));
      kv_last_page_lens = torch::ones({1}, torch::TensorOptions().dtype(torch::kInt32).device(Q.device()));
      splitData = torch::empty({1, splits, Q.size(1), 512}, torch::TensorOptions().dtype(torch::kFloat32).device(Q.device()));
      splitLse = torch::empty({1, splits, Q.size(1), 1}, torch::TensorOptions().dtype(torch::kFloat32).device(Q.device()));
      output = torch::empty({1, Q.size(1), 512}, torch::TensorOptions().dtype(Q.dtype()).device(Q.device()));
    }

    int num_seqs = Q.size(0);
    int num_heads = Q.size(1);
    int head_size = Q.size(2);
    int page_size = KV.size(1);
    int num_kv_heads = KV.size(2);
    int kv_split = splitData.size(1);
    const int gqa_ratio = num_heads / num_kv_heads;

    int stride_Q = Q.stride(0) * Q.itemsize();
    int stride_Page = KV.stride(0) * KV.itemsize();
    uint32_t log2_page = (uint32_t)log2f(page_size);

    KernelArgs args;
    size_t arg_size = sizeof(args);
    args.ptr_R = splitData.data_ptr();
    args.ptr_LSE = splitLse.data_ptr();
    args.ptr_Q = Q.data_ptr();
    args.ptr_KV = KV.data_ptr();
    args.ptr_LTP = kv_indptr.data_ptr();
    args.ptr_LTD = kv_page_indices.data_ptr();
    args.ptr_LTL = kv_last_page_lens.data_ptr();
    args.scalar = (float)softmax_scale;
    args.s_MQA = gqa_ratio;
    args.s_kv_split = kv_split;
    args.s_Q_Bs = stride_Q;
    args.s_Bs = stride_Page;
    args.s_log2_plen = log2_page;

    const at::cuda::OptionalCUDAGuard device_guard(device_of(Q));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AiterAsmKernel *impl_ptr = nullptr;
    TORCH_CHECK(Q.is_contiguous(),
                __func__, ":only support Q.is_contiguous() for now");
    TORCH_CHECK(num_kv_heads == 1,
                __func__, ":only support num_kv_heads==1 for now");
    TORCH_CHECK(head_size == KV.size(3),
                __func__, ":only support head_size == KV.size(3) for now");
    CHECK_EQ(Q.dtype(), at::ScalarType::BFloat16);

    {
#include "mla_stage1_a16w16_bf16.h"
        static AiterAsmKernel impl_a16w16_bf16("mla_stage1_a16w16_bf16", mla_stage1_a16w16_bf16);
        impl_ptr = &impl_a16w16_bf16;
    }

    TORCH_CHECK(impl_ptr != nullptr,
                __func__, ": unsupport current input type");
    impl_ptr->launch_kernel({&args,
                             &arg_size,
                             (gqa_ratio + 15) / 16, // gdx
                             num_seqs,              // gdy
                             kv_split,              // gdz
                             256,                   // bdx: 4 wv64
                             1,                     // bdy
                             1,                     // bdz
                             stream});

    Kernel2Args args2;
    size_t arg2_size = sizeof(args2);
    args2.p0 = splitData.data_ptr();
    args2.p1 = splitLse.data_ptr();
    args2.p2 = output.data_ptr();
    args2.p3 = kv_indptr.data_ptr();
    args2.s0 = splitLse.stride(0);
    args2.s1 = splitLse.stride(2);
    args2.s2 = splitLse.stride(1);
    args2.s3 = output.stride(0);
    args2.s4 = output.stride(1);

    impl_comb->launch_kernel({&args2,
                             &arg2_size,
                             Q.size(0),             // gdx
                             Q.size(1),             // gdy
                             1,                     // gdz
                             512,                   // bdx: 4 wv64
                             1,                     // bdy
                             1,                     // bdz
                             stream});
    return output;
}

