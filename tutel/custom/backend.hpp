// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

// Plugin reference: https://github.com/microsoft/antares

#if defined(_WIN64)
#include <filesystem>
#include <algorithm>
#endif

#if !defined(CHECK_OK)
#define CHECK_OK(x)  ((x) ? 1 : (fprintf(stderr, "[CheckFail] %s:%d\n", __FILE__, __LINE__), exit(1), 0))
#endif

#if !defined(__RUNTIME_MODE__)
#define GET_STREAM() ((CUstream)stream)
#else
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include <ATen/cuda/CUDAContext.h>
#define GET_STREAM() at::cuda::getCurrentCUDAStream().stream()
#endif

#if !defined(__HIP_PLATFORM_HCC__) && !defined(__HIP_PLATFORM_AMD__)
#include <cuda.h>
#include <nvrtc.h>
#else
#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>
#define cuInit hipInit
#define cuMemAlloc hipMalloc
#define cuMemFree hipFree
#define cuModuleLoad hipModuleLoad
#define cuModuleLoadData hipModuleLoadData
#define cuModuleUnload hipModuleUnload
#define cuModuleGetFunction hipModuleGetFunction
#define cuLaunchKernel hipModuleLaunchKernel
#define cuMemAllocHost hipHostMalloc
#define cuMemFreeHost hipHostFree
#define cuStreamSynchronize hipStreamSynchronize
#define cuCtxSynchronize hipDeviceSynchronize
#define cuMemcpyHtoDAsync hipMemcpyHtoDAsync
#define cuMemcpyDtoDAsync hipMemcpyDtoDAsync
#define cuMemcpyDtoHAsync hipMemcpyDtoHAsync
#define CUdeviceptr hipDeviceptr_t
#define CUmodule hipModule_t
#define CUfunction hipFunction_t
#define CUevent hipEvent_t
#define cuEventElapsedTime hipEventElapsedTime
#define cuEventCreate hipEventCreateWithFlags
#define cuEventDestroy hipEventDestroy
#define cuEventRecord hipEventRecord
#define CUcontext long
#define cuDevicePrimaryCtxRetain(x, y) (*(x) = (CUcontext)((long)(y)), 0)
#define cuCtxSetCurrent(x) hipSetDevice((int)(x))
#define cuCtxGetCurrent(x) hipGetDevice((int*)(x))
#define cuCtxGetDevice(x) hipGetDevice((int*)(x))
#define CUstream hipStream_t
#define nvrtcGetCUBIN hiprtcGetCode
#define nvrtcGetCUBINSize hiprtcGetCodeSize
#endif


namespace ab {

  static int _current_device;
  static std::unordered_map<size_t, std::vector<void*>> _cached_memory;

  int init(int dev) {
    static bool _retained = false;
    if (_retained)
      return _current_device;
    _retained = true;

    CUcontext ctx;
    if (dev < 0) {
      if (0 == cuCtxGetDevice(&_current_device)) {
        cuDevicePrimaryCtxRetain(&ctx, _current_device);
        cuCtxSetCurrent(ctx);
        return _current_device;
      }
      dev = getenv("LOCAL_RANK") ? std::atoi(getenv("LOCAL_RANK")) : 0;
    }
#if !defined(__RUNTIME_MODE__)
    setenv("CUDA_VISIBLE_DEVICES", std::to_string(dev).c_str(), 1);
#else
    _current_device = dev;
#endif
    if (0 != cuInit(0) || 0 != cuDevicePrimaryCtxRetain(&ctx, _current_device) || 0 != cuCtxSetCurrent(ctx))
        throw std::runtime_error("GPU device is not found.\n");
    return _current_device;
  }

  void finalize() {
  }

  inline size_t compute_slotsize(size_t value) {
      if (value >= (1LL << 30))
          return value;
      value -= 1;
      value |= value >> 1;
      value |= value >> 2;
      value |= value >> 4;
      value |= value >> 8;
      value |= value >> 16;
      value |= value >> 32;
      value += 1;
      return value;
  }

  void* alloc(size_t byteSize, const std::vector<size_t> &shape, const std::string &dtype, const std::string &name) {
    init(-1);

    byteSize = compute_slotsize(byteSize);
    auto &it = _cached_memory[byteSize];
    if (it.size()) {
      auto dptr = it.back();
      it.pop_back();
      return dptr;
    }
    void *dptr = nullptr;
    if (byteSize)
      CHECK_OK(0 == cuMemAlloc((CUdeviceptr*)&dptr, byteSize));
    else
      dptr = (void*)1LU;
    return dptr;
  }

  void release(void *dptr, size_t byteSize) {
    byteSize = compute_slotsize(byteSize);
    auto &it = _cached_memory[byteSize];
    it.push_back(dptr);
  }

  void* moduleLoad(const std::string &binary) {
    init(-1);
    const char* data = binary.data();
    CUmodule hmod = nullptr;
    CHECK_OK(0 == cuModuleLoadData(&hmod, data));
    return hmod;
  }

  std::vector<void*> moduleGetFunction(const void *hModule, const std::string &fname, const std::unordered_map<std::string, int> &threads) {
    auto query = [&](const std::string &axis, long defval = 1) -> void* {
      auto it = threads.find(axis);
      if (it == threads.end())
        return (void*)defval;
      return (void*)(long)it->second;
    };

    CUfunction hfunc = nullptr;
    CHECK_OK(0 == cuModuleGetFunction(&hfunc, (CUmodule)hModule, fname.c_str()));
    std::vector<void*> fdata = { hfunc, query("blockIdx.x"), query("blockIdx.y"), query("blockIdx.z"), query("threadIdx.x"), query("threadIdx.y"), query("threadIdx.z") };

    void *item = query("$", 0);
    if (item) {
      fdata.push_back(item);
      fdata.push_back(query("$$", 1));

      for (int i = 0; ; ++i) {
        void *item = query("$" + std::to_string(i), 0);
        if (!item)
          break;
        fdata.push_back(item);
      }
    }
    return fdata;
  }

  void launchKernel(std::vector<void*> &hFunc, const std::vector<void*> &krnl_args, void *stream) {
    std::vector<void*> pargs(krnl_args.size());
    for (int i = 0; i < krnl_args.size(); ++i)
      pargs[i] = (void*)&krnl_args[i];

    if (hFunc.size() > 7) {
      long attrs = (long)hFunc[8];
      for (int i = 9; i < hFunc.size(); ++i) {
        long val = (long)hFunc[i];
        if (val == -1) continue;

        auto ptr = (int*)pargs[i - 9 + (long)hFunc[7]];
        attrs *= (val > 0) ? ((*ptr + val - 1) / val) : (*ptr * (-val));
      }
      hFunc[1] = (void*)attrs;
      if (!hFunc[1]) return;
    }

    CHECK_OK(0 == cuLaunchKernel((CUfunction)hFunc[0], (long)hFunc[1], (long)hFunc[2], (long)hFunc[3], (long)hFunc[4], (long)hFunc[5], (long)hFunc[6],
      0, GET_STREAM(), (void**)pargs.data(), nullptr));
  }

  void memcpyHtoD(void *dptr, void *hptr, size_t byteSize, void *stream) {
    CHECK_OK(0 == cuMemcpyHtoDAsync((CUdeviceptr)dptr, hptr, byteSize, (CUstream)stream));
  }

  void memcpyDtoD(void *dptr, void *dptr0, size_t byteSize, void *stream) {
    CHECK_OK(0 == cuMemcpyDtoDAsync((CUdeviceptr)dptr, (CUdeviceptr)dptr0, byteSize, (CUstream)stream));
  }

  void memcpyDtoH(void *hptr, void *dptr, size_t byteSize, void *stream) {
    CHECK_OK(0 == cuMemcpyDtoHAsync(hptr, (CUdeviceptr)dptr, byteSize, (CUstream)stream));
  }

  void synchronize(void *stream) {
    CHECK_OK(0 == cuStreamSynchronize((CUstream)stream));
  }

  void* recordTime(void *stream) {
    CUevent hEvent;
    CHECK_OK(0 == cuEventCreate(&hEvent, 0));
    CHECK_OK(0 == cuEventRecord(hEvent, (CUstream)stream));
    return hEvent;
  }

  double convertToElapsedTime(void *hStart, void *hStop) {
    CHECK_OK(0 == cuCtxSynchronize());

    float ms;
    CHECK_OK(0 == cuEventElapsedTime(&ms, (CUevent)hStart, (CUevent)hStop));
    CHECK_OK(0 == cuEventDestroy((CUevent)hStart));
    CHECK_OK(0 == cuEventDestroy((CUevent)hStop));
    return ms * 1e-3;
  }
}
