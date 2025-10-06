// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <torch/extension.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupGloo.hpp>
#include <torch/csrc/distributed/c10d/TCPStore.hpp>

#if defined(USE_GPU)
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda.h>
#include <nvrtc.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGraph.h>
#else
#undef USE_NCCL
#endif

#if defined(USE_NCCL)
#include <nccl.h>
#endif

#include <regex>
#include <vector>

#if defined(__linux__)
#include <sys/wait.h>
#endif

#undef CHECK_EQ
#undef CHECK_NE
#undef CHECK_LE
#undef CHECK_CPU
#undef CHECK_CUDA
#undef CHECK_CONTIGUOUS

#define CHECK_EQ(x, y) AT_ASSERTM((x) == (y), "CHECK_EQ fails.")
#define CHECK_NE(x, y) AT_ASSERTM((x) != (y), "CHECK_NE fails.")
#define CHECK_LE(x, y) AT_ASSERTM((x) <= (y), "CHECK_LE fails.")
#define CHECK_CPU(x) AT_ASSERTM(!x.is_cuda(), #x " must be a CPU tensor")
#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")

#if defined(USE_GPU)
#include "antares_ops.h"

#if !defined(__HIP_PLATFORM_HCC__) && !defined(__HIP_PLATFORM_AMD__)
#define IS_NVIDIA_GPU 1
#else
#define IS_NVIDIA_GPU 0
#endif

namespace jit {

inline static std::string file_read(const char *path) {
  FILE *fp = fopen(path, "rb");
  CHECK_EQ(true, fp != nullptr);
  fseek(fp, 0, SEEK_END);
  size_t code_size = ftell(fp);
  fseek(fp, 0, SEEK_SET);
  std::string code;
  code.resize(code_size);
  CHECK_EQ(code_size, fread((void*)code.data(), 1, code_size, fp));
  fclose(fp);
  return code;
}

inline static void file_write(const char *path, const std::string &code) {
  FILE *fp = fopen(path, "wb");
  CHECK_EQ(true, fp != nullptr);
  CHECK_EQ(code.size(), fwrite((void*)code.data(), 1, code.size(), fp));
  fclose(fp);
}

static std::string __sdk_home__;

static void update_sdk_home(const torch::Tensor &sdk_path) {
  CHECK_CPU(sdk_path);
  __sdk_home__ = static_cast<char*>(sdk_path.data_ptr());
}

inline std::string sdk_path(const std::string &rel = "") {
  static std::string cuda_home, cc;
  if (cuda_home.size() == 0) {
#if !defined(__HIP_PLATFORM_HCC__) && !defined(__HIP_PLATFORM_AMD__)
    cc = "bin/nvcc";
#else
    cc = "bin/hipcc";
#endif

#if defined(__linux__)
    cuda_home = __sdk_home__ + std::string("/");
#else
    cuda_home = __sdk_home__ + std::string("\\");
#endif
  }
  if (rel.size() > 0)
    return cuda_home + rel;
  return cuda_home + cc;
}

static std::string nvcc_compile(const char* code, const std::string &arch) {
#if defined(__linux__)
  char code_path[] = "/tmp/torch-tutel-XXXXXX.cu";
  CHECK_NE(-1, mkstemps(code_path, 3));

  file_write(code_path, code);
  std::string fatbin_path = code_path + std::string(".fatbin");

  std::string entry = sdk_path();
  if (access(entry.c_str(), F_OK) != 0) {
    LOG(FATAL) << "Failed to detect CUDA compiler file: " << entry << ", please set CUDA_HOME environment to configure CUDA SDK location correctly.";
    exit(1);
  }
  pid_t  pid = fork();
  if (pid == 0) {
#if !defined(__HIP_PLATFORM_HCC__) && !defined(__HIP_PLATFORM_AMD__)
    CHECK_EQ(-1, execl(entry.c_str(), entry.c_str(), code_path, "-o", fatbin_path.c_str(), "--fatbin", "-O4", "-gencode", ("arch=compute_" + arch + ",code=sm_" + arch).c_str(), (char *)NULL));
#else
    CHECK_EQ(-1, execl(entry.c_str(), entry.c_str(), code_path, "-o", fatbin_path.c_str(), "--genco", "-O4", "-w" , ("--offload-arch=" + arch).c_str(), (char *)NULL));
#endif
    exit(1);
  } else {
    wait(NULL);
  }
  auto image = file_read(fatbin_path.data());
  unlink(fatbin_path.data());
  unlink(code_path);
  return image;
#else
  return "";
#endif
}

static std::string nvrtc_compile(const char* code, const std::string &arch) {
#if !defined(__HIP_PLATFORM_HCC__) && !defined(__HIP_PLATFORM_AMD__)
  std::string arch_option = "--gpu-architecture=compute_" + arch, include_path = "--include-path=" + sdk_path("include");
  std::vector<const char*> param_cstrings = {"--restrict", include_path.c_str(), arch_option.c_str(), "--use_fast_math", "--extra-device-vectorization"};
#else
  std::string arch_option = "--gpu-architecture=" + arch;
  std::vector<const char*> param_cstrings = {arch_option.c_str(), "-O4"};
#endif
  nvrtcProgram prog;

  CHECK_EQ(0, nvrtcCreateProgram(&prog, code, nullptr, 0, nullptr, nullptr));
  nvrtcResult res = nvrtcCompileProgram(prog, param_cstrings.size(), param_cstrings.data());

  size_t log_size;
  CHECK_EQ(0, nvrtcGetProgramLogSize(prog, &log_size));
  std::string log;
  log.resize(log_size);
  CHECK_EQ(0, nvrtcGetProgramLog(prog, &log[0]));
  if (0 != res) {
    static bool once_flag = false;
    if (!once_flag) {
      once_flag = true;
      LOG(WARNING) << log << " Failed to use NVRTC for JIT compilation in this Pytorch version, try another approach using CUDA compiler.. (To always disable NVRTC, please: export USE_NVRTC=0)";
    }
    CHECK_EQ(0, nvrtcDestroyProgram(&prog));
    return "";
  }

  size_t ptx_size;
  CHECK_EQ(0, nvrtcGetPTXSize(prog, &ptx_size));

  std::string ptx;
  ptx.resize(ptx_size);
  CHECK_EQ(0, nvrtcGetPTX(prog, &ptx[0]));
  CHECK_EQ(0, nvrtcDestroyProgram(&prog));
  return ptx;
}

struct ModuleConfig {
  // Handling JIT compilation in Multi-gpu cases
  std::vector<CUfunction> hFunc;
  std::string code, fname;
  dim3 blocks, threads;
};

static std::vector<ModuleConfig> _gms;

inline static CUfunction jit_activate(int fd, int dev) {
  auto &gm = _gms[fd];
  if (gm.hFunc.size() <= dev)
    gm.hFunc.resize(dev + 1);

  if (gm.hFunc[dev] == nullptr) {
#if !defined(__HIP_PLATFORM_HCC__) && !defined(__HIP_PLATFORM_AMD__)
    int major, minor;
    CHECK_EQ(0, cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev));
    CHECK_EQ(0, cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev));
    std::string arch = std::to_string(major) + std::to_string(minor);
#else
    hipDeviceProp_t prop;
    CHECK_EQ(0, hipGetDeviceProperties(&prop, dev));
    std::string arch = prop.gcnArchName;
#endif
    const char *source = gm.code.data(), *pos, *tail;

    int use_nvrtc = getenv("USE_NVRTC") ? std::atoi(getenv("USE_NVRTC")) : 0;
    std::string image;
    if (use_nvrtc || (image = nvcc_compile(source, arch)) == "") {
        image = nvrtc_compile(source, arch);
    }

    long launch_bound;
    { char tag[] = " __launch_bounds__(";  const char *pos = strstr(source, tag); launch_bound = pos ? std::atol(pos + sizeof(tag) - 1) : 1024L; }

    static CUjit_option options[] = {CU_JIT_OPTIMIZATION_LEVEL, CU_JIT_THREADS_PER_BLOCK};
    static void* values[] = {(void*)4L, (void*)launch_bound};

    CUmodule hMod = nullptr;
    CHECK_EQ(0, cuModuleLoadDataEx(&hMod, image.c_str(), sizeof(options) / sizeof(*options), options, values));
    CHECK_NE(nullptr, hMod);

    CHECK_NE(nullptr, (pos = strstr(source, " void ")));
    pos += 6; CHECK_NE(nullptr, (tail = strchr(pos, '(')));

    std::string fname = std::string(pos, tail - pos);
    gm.fname = fname;
    CHECK_EQ(0, cuModuleGetFunction(&gm.hFunc[dev], hMod, fname.c_str()));
    CHECK_NE(nullptr, gm.hFunc[dev]);
  }

  return gm.hFunc[dev];
}

static void jit_execute(const std::vector<const void*> &ppargs, int fd, int dev, const dim3 &blocks, const dim3 &threads, cudaStream_t stream = 0) {
  CHECK_EQ(0, cudaSetDevice(dev));
  CUfunction hfunc = jit_activate(fd, dev);
  CHECK_EQ(0, cuLaunchKernel(hfunc, blocks.x, blocks.y, blocks.z, threads.x, threads.y, threads.z, 0, stream, (void**)ppargs.data(), nullptr));
}

static void jit_execute_with_values(const std::vector<const void*> &pargs, int fd, int dev, const dim3 &blocks, const dim3 &threads, cudaStream_t stream = 0) {
  std::vector<const void*> ppargs(pargs.size());
  for (int i = 0; i < ppargs.size(); ++i)
    ppargs[i] = &pargs[i];
  jit_execute(ppargs, fd, dev, blocks, threads, stream);
}

static int inject_source(const std::string &headless_code) {
  int fd = _gms.size();
  _gms.resize(fd + 1);

  auto &gm = _gms[fd];
#if !defined(__HIP_PLATFORM_HCC__) && !defined(__HIP_PLATFORM_AMD__)
  gm.code = "#include <cuda_runtime.h>\n#include <cuda_fp16.h>\n" + headless_code;
#else
  gm.code = "#include <hip/hip_runtime.h>\n" + headless_code;
#endif

  const char *source = headless_code.c_str();
  { char tag[] = "// [thread_extent] blockIdx.x = ";  const char *pos = strstr(source, tag); gm.blocks.x = pos ? std::atoi(pos + sizeof(tag) - 1) : 1; }
  { char tag[] = "// [thread_extent] blockIdx.y = ";  const char *pos = strstr(source, tag); gm.blocks.y = pos ? std::atoi(pos + sizeof(tag) - 1) : 1; }
  { char tag[] = "// [thread_extent] blockIdx.z = ";  const char *pos = strstr(source, tag); gm.blocks.z = pos ? std::atoi(pos + sizeof(tag) - 1) : 1; }
  { char tag[] = "// [thread_extent] threadIdx.x = "; const char *pos = strstr(source, tag); gm.threads.x = pos ? std::atoi(pos + sizeof(tag) - 1) : 1; }
  { char tag[] = "// [thread_extent] threadIdx.y = "; const char *pos = strstr(source, tag); gm.threads.y = pos ? std::atoi(pos + sizeof(tag) - 1) : 1; }
  { char tag[] = "// [thread_extent] threadIdx.z = "; const char *pos = strstr(source, tag); gm.threads.z = pos ? std::atoi(pos + sizeof(tag) - 1) : 1; }

  return fd;
}

static void invoke(const std::vector<torch::Tensor> &ts, const std::vector<long> &args, const std::vector<int> &blocks, int fd) {
  std::vector<const void*> pargs(ts.size() + args.size()), ppargs(ts.size() + args.size());
  for (int i = 0; i < (int)ts.size(); ++i) {
    CHECK_CUDA(ts[i]);
    pargs[i] = ts[i].data_ptr(), ppargs[i] = &pargs[i];
  }
  for (int i = (int)ts.size(); i < (int)pargs.size(); ++i) {
    pargs[i] = (void*)args[i - ts.size()], ppargs[i] = &pargs[i];
  }

  int dev = ts[0].device().index();
  CHECK_EQ(0, cudaSetDevice(dev));
  if (blocks.size() == 0)
    jit_execute(ppargs, fd, dev, _gms[fd].blocks, _gms[fd].threads, at::cuda::getDefaultCUDAStream().stream());
  else if (blocks.size() == 1)
    jit_execute(ppargs, fd, dev, dim3(blocks[0]), _gms[fd].threads, at::cuda::getDefaultCUDAStream().stream());
  else if (blocks.size() == 2)
    jit_execute(ppargs, fd, dev, dim3(blocks[0], blocks[1]), _gms[fd].threads, at::cuda::getDefaultCUDAStream().stream());
  else
    jit_execute(ppargs, fd, dev, dim3(blocks[0], blocks[1], blocks[2]), _gms[fd].threads, at::cuda::getDefaultCUDAStream().stream());
}

} // namespace jit
#endif

static std::unordered_map<int64_t, c10::intrusive_ptr<c10d::ProcessGroup>> _pg_storage;

static void put_pg_storage(int64_t key, pybind11::object pg_obj) {
    auto pg = pybind11::cast<c10::intrusive_ptr<c10d::ProcessGroup>>(pg_obj);
    _pg_storage[key] = pg;
}

template<typename dtype> static void invoke_cpu(const std::vector<torch::Tensor> &ts, const std::vector<int> &extra, int kernel_type) {
  int samples = extra[0];
  int hidden = extra[1];
  int capacity = extra[2];
  dtype *gates1_s = static_cast<dtype*>(ts[0].data_ptr());
  int *indices1_s = static_cast<int*>(ts[1].data_ptr());
  int *locations1_s = static_cast<int*>(ts[2].data_ptr());
  dtype *reshaped_input = static_cast<dtype*>(ts[3].data_ptr());
  dtype *dispatched_input = static_cast<dtype*>(ts[4].data_ptr());

  for (int i = 0; i < (int)ts.size(); ++i)
    CHECK_CONTIGUOUS(ts[i]);

  if (kernel_type == 0) { //forward
    for (int i = 0; i < samples; ++i) {
      if (locations1_s[i] < capacity && indices1_s[i] >= 0) {
        for (int j = 0; j < hidden; ++j) {
          dispatched_input[(indices1_s[i] * capacity + locations1_s[i]) * (hidden) + j] += gates1_s[i] * reshaped_input[i * (hidden) + j];
        }
      }
    }
  } else if (kernel_type == 1) { //backward_data
    for (int i = 0; i < samples; ++i) {
      if (locations1_s[i] < capacity && indices1_s[i] >= 0) {
        for (int j = 0; j < hidden; ++j) {
          reshaped_input[i * hidden + j] = gates1_s[i] * dispatched_input[(indices1_s[i] * capacity + locations1_s[i]) * (hidden) + j];
        }
      } else {
        for (int j = 0; j < hidden; ++j) {
          reshaped_input[i * hidden + j] = 0;
        }
      }
    }
  } else { //backward_gate
    for (int i = 0; i < samples; ++i) {
      gates1_s[i] = 0;
      if (locations1_s[i] >= capacity || indices1_s[i] < 0)
        continue;
      for (int j = 0; j < hidden; ++j) {
        gates1_s[i] += dispatched_input[(indices1_s[i] * capacity + locations1_s[i]) * (hidden) + j] * reshaped_input[i * hidden + j];
      }
    }
  }
}

#if defined(USE_NCCL)

static ncclComm_t g_nccl_comm = nullptr, shared_nccl_comm = nullptr;
static std::vector<at::cuda::CUDAEvent> g_cuda_events;
static int g_world_size = 0, shared_world_size = 0;
static int g_world_rank = 0, shared_world_rank = 0;
static int g_local_size = 0;
static int g_local_rank = 0;

// jit
static int mem_stride_copy_char_fd = -1;
static int mem_stride_copy_uint4_fd = -1;
static int mem_stride_copy_gridsize = 1;
static int mem_stride_copy_blocksize = 1;

static size_t get_nccl_unique_id_size() {
  return sizeof(ncclUniqueId);
}

static void get_nccl_unique_id(torch::Tensor &nccl_unique_id_tensor) {
  ncclUniqueId nccl_unique_id;

  CHECK_EQ(0, ncclGetUniqueId(&nccl_unique_id));
  CHECK_CPU(nccl_unique_id_tensor);
  CHECK_EQ(nccl_unique_id_tensor.nbytes(), sizeof(ncclUniqueId));
  memcpy((void *)nccl_unique_id_tensor.data_ptr(), &nccl_unique_id, sizeof(ncclUniqueId));
}

static void init_shared_nccl(
    const torch::Tensor &nccl_unique_id_tensor,
    int world_size,
    int world_rank) {
  ncclUniqueId nccl_unique_id;

  CHECK_CPU(nccl_unique_id_tensor);
  CHECK_EQ(nccl_unique_id_tensor.nbytes(), sizeof(ncclUniqueId));
  memcpy(&nccl_unique_id, (void *)nccl_unique_id_tensor.data_ptr(), sizeof(ncclUniqueId));
  CHECK_EQ(0, ncclGroupStart());
  CHECK_EQ(0, ncclCommInitRank(&shared_nccl_comm, world_size, nccl_unique_id, world_rank));
  CHECK_EQ(0, ncclGroupEnd());

  shared_world_size = world_size;
  shared_world_rank = world_rank;
}

static void init_nccl(
    const torch::Tensor &nccl_unique_id_tensor,
    int world_size,
    int world_rank,
    int max_num_split) {
  ncclUniqueId nccl_unique_id;

  CHECK_CPU(nccl_unique_id_tensor);
  CHECK_EQ(nccl_unique_id_tensor.nbytes(), sizeof(ncclUniqueId));
  memcpy(&nccl_unique_id, (void *)nccl_unique_id_tensor.data_ptr(), sizeof(ncclUniqueId));
  CHECK_EQ(0, ncclGroupStart());
  CHECK_EQ(0, ncclCommInitRank(&g_nccl_comm, world_size, nccl_unique_id, world_rank));
  CHECK_EQ(0, ncclGroupEnd());

  g_cuda_events.resize(max_num_split);
  g_world_size = world_size;
  g_world_rank = world_rank;

  if (const char* local_size = std::getenv("LOCAL_SIZE")) {
    g_local_size = std::atoi(local_size);
  } else {
    CHECK_EQ(0, cudaGetDeviceCount(&g_local_size));
  }
  CHECK_EQ(0, ncclCommCuDevice(g_nccl_comm, &g_local_rank));

  // jit for nccl
  if (mem_stride_copy_uint4_fd == -1) {
    std::string mem_stride_copy_cu = R"(
extern "C" __global__ void memStrideCopyKernel(
    $T *__restrict__ out, const $T *__restrict__ in,
    const size_t size, const int height, const int width) {
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (size_t i = tid; i < size * height * width; i += gridDim.x * blockDim.x) {
        const size_t index = i / size, offset = i % size;
        const size_t j = (width * (index % height) + (index / height)) * size + offset;
        out[j] = in[i];
    }
}
    )";
    mem_stride_copy_char_fd = jit::inject_source(std::regex_replace(mem_stride_copy_cu, std::regex("\\$T"), "char"));
    mem_stride_copy_uint4_fd = jit::inject_source(std::regex_replace(mem_stride_copy_cu, std::regex("\\$T"), "uint4"));
    CHECK_NE(-1, mem_stride_copy_char_fd);
    CHECK_NE(-1, mem_stride_copy_uint4_fd);
    CUfunction hfunc = jit::jit_activate(mem_stride_copy_uint4_fd, g_local_rank);
#if !defined(__HIP_PLATFORM_HCC__) && !defined(__HIP_PLATFORM_AMD__)
    CHECK_EQ(0, cuOccupancyMaxPotentialBlockSize(&mem_stride_copy_gridsize, &mem_stride_copy_blocksize, hfunc, 0, 0, 0));
#else
    CHECK_EQ(0, hipModuleOccupancyMaxPotentialBlockSize(&mem_stride_copy_gridsize, &mem_stride_copy_blocksize, hfunc, 0, 0));
#endif
  }
}

inline at::cuda::CUDAStream& get_default_stream() {
  static at::cuda::CUDAStream default_stream = at::cuda::getDefaultCUDAStream();
  return default_stream;
}

inline at::cuda::CUDAStream& get_nccl_stream() {
  static at::cuda::CUDAStream nccl_stream = at::cuda::getStreamFromPool();
  return nccl_stream;
}

static torch::Tensor& current_stream_release(torch::Tensor &tensor, int idx) {
  g_cuda_events[idx].record(at::cuda::getCurrentCUDAStream());
  return tensor;
}

static torch::Tensor& current_stream_acquire(torch::Tensor &tensor, int idx) {
  g_cuda_events[idx].block(at::cuda::getCurrentCUDAStream());
  return tensor;
}

static torch::Tensor& nccl_stream_release(torch::Tensor &tensor, int idx) {
  g_cuda_events[idx].record(get_nccl_stream());
  return tensor;
}

static torch::Tensor& nccl_stream_acquire(torch::Tensor &tensor, int idx) {
  g_cuda_events[idx].block(get_nccl_stream());
  return tensor;
}

void warp_nccl_bcast(const torch::Tensor &t, int64_t root) {
  CHECK_CUDA(t);
  AT_ASSERTM(shared_world_size > 0, "Failed to initialize Shared NCCL");
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  auto dtype = t.dtype();
  int dtype_size = torch::elementSize(torch::typeMetaToScalarType(dtype));
  ncclBcast(t.data_ptr(), t.numel() * dtype_size, ncclInt8, root, (ncclComm_t)shared_nccl_comm, stream);
}

void warp_nccl_all_reduce(const torch::Tensor &t, const torch::Tensor &out) {
  CHECK_CUDA(t);
  AT_ASSERTM(shared_world_size > 0, "Failed to initialize Shared NCCL");
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  ncclDataType_t ncclType;
  if (t.dtype() == torch::kBFloat16)
    ncclType = ncclBfloat16;
  else if (t.dtype() == torch::kFloat16)
    ncclType = ncclFloat16;
  else
      AT_ASSERTM(0, "Unrecognized data type for Nccl AllReduce.");
  ncclAllReduce(t.data_ptr(), out.data_ptr(), t.numel(), ncclType, ncclSum, (ncclComm_t)shared_nccl_comm, stream);
}

static void batch_all_to_all_v(const std::vector<torch::Tensor> &ins, const std::vector<torch::Tensor> &outs, const torch::Tensor &in_sizes_, const torch::Tensor &out_sizes_) {
  AT_ASSERTM(shared_world_size > 0, "Failed to initialize Shared NCCL");

  auto in_sizes_cpu = in_sizes_.to(torch::kCPU).to(torch::kInt64);
  auto out_sizes_cpu = out_sizes_.to(torch::kCPU).to(torch::kInt64);
  auto* in_sizes = (unsigned long long*)in_sizes_cpu.data_ptr();
  auto* out_sizes = (unsigned long long*)out_sizes_cpu.data_ptr();
  auto stream = at::cuda::getCurrentCUDAStream().stream();

  for (int k = 0; k < ins.size(); ++k) {
    ncclGroupStart();
    auto* in_buff = ins[k].data_ptr();
    auto* out_buff = outs[k].data_ptr();
    auto dtype = ins[k].dtype();
    int size = torch::elementSize(torch::typeMetaToScalarType(dtype));
    AT_ASSERTM(size > 0, "Data type of input tensors for batch_all_to_all_v are not recognized.");
    AT_ASSERTM(k == 0 || ins[0].numel() == ins[k].numel(), "Tensors within batch_all_to_all_v are supposed to share same length.");

    unsigned long long in_offset = 0, out_offset = 0;
    for (int i = 0; i < shared_world_size; ++i) {
      if(in_sizes[i])  // only send if partition is non-empty
          ncclSend((char*)in_buff + in_offset, in_sizes[i] * size, ncclInt8, i, (ncclComm_t)shared_nccl_comm, stream);
      if(out_sizes[i]) // only receive if partition is non-empty
          ncclRecv((char*)out_buff + out_offset, out_sizes[i] * size, ncclInt8, i, (ncclComm_t)shared_nccl_comm, stream);
      in_offset += in_sizes[i] * size;
      out_offset += out_sizes[i] * size;
    }
    ncclGroupEnd();
  }
}

static void batch_all_gather_v(const std::vector<torch::Tensor> &ins, const std::vector<torch::Tensor> &outs, const torch::Tensor &out_sizes_) {
  AT_ASSERTM(shared_world_size > 0, "Failed to initialize Shared NCCL");

  auto out_sizes_cpu = out_sizes_.to(torch::kCPU).to(torch::kInt64);
  auto* out_sizes = (unsigned long long*)out_sizes_cpu.data_ptr();
  auto stream = at::cuda::getCurrentCUDAStream().stream();

  for (int k = 0; k < ins.size(); ++k) {
    ncclGroupStart();
    auto* in_buff = ins[k].data_ptr();
    auto* out_buff = outs[k].data_ptr();
    auto dtype = ins[k].dtype();
    int size = torch::elementSize(torch::typeMetaToScalarType(dtype));
    AT_ASSERTM(size > 0, "Data type of input tensors for batch_all_gather_v are not recognized.");
    AT_ASSERTM(k == 0 || ins[0].numel() == ins[k].numel(), "Tensors within batch_all_gather_v are supposed to share same length.");

    unsigned long long out_offset = 0;
    for (int i = 0; i < shared_world_size; ++i) {
      if (out_sizes[shared_world_rank])
        ncclSend((char*)in_buff, out_sizes[shared_world_rank] * size, ncclInt8, i, (ncclComm_t)shared_nccl_comm, stream);
      if (out_sizes[i])
        ncclRecv((char*)out_buff + out_offset, out_sizes[i] * size, ncclInt8, i, (ncclComm_t)shared_nccl_comm, stream);
      out_offset += out_sizes[i] * size;
    }
    ncclGroupEnd();
  }
}

static std::vector<torch::Tensor> nccl_all_to_all_scatter_async(
    const torch::Tensor &input,
    torch::IntArrayRef output_size,
    int num_split,
    int num_slices_per_split,
    bool is_backward) {
  CHECK_CUDA(input);
  CHECK_LE(num_split, g_cuda_events.size());

  CHECK_EQ(0, num_slices_per_split % g_world_size);
  size_t length = input.nbytes();
  size_t num_slices = num_slices_per_split * num_split;
  CHECK_EQ(0, length % num_slices);
  size_t slice_size = length / num_slices;

  // Save original stream and switch to NCCL stream
  // Output tensors must be allocated in NCCL stream context to prevent PyTorch Caching Allocator from recycling it
  const at::cuda::CUDAStream& original_stream = at::cuda::getCurrentCUDAStream();
  at::cuda::setCurrentCUDAStream(get_nccl_stream());

  // Computation stream allocator will add blocking event to nccl stream after nccl kernels
  c10::cuda::CUDACachingAllocator::recordStream(input.storage().data_ptr(), get_nccl_stream());

  std::vector<torch::Tensor> output_list(num_split);
  for (int i = 0; i < num_split; i++) {
    output_list[i] = torch::empty(output_size, torch::TensorOptions().dtype(input.dtype()).device(input.device()));
  }
  // NCCL stream allocator will add blocking event to computation stream after computation kernels
  for (auto& output : output_list) {
    c10::cuda::CUDACachingAllocator::recordStream(output.storage().data_ptr(), original_stream);
  }

  // Acquire 0-th event for single input
  g_cuda_events[0].block(get_nccl_stream());

  for (int i = 0; i < num_split; i++) {
    // Reverse calculation order in backward for pipelining
    int calc_idx = is_backward ? num_split - 1 - i : i;

    CHECK_EQ(0, ncclGroupStart());
    for (int j = 0; j < num_slices_per_split; j++) {
      CHECK_EQ(0, ncclSend(
          ((char*)input.data_ptr()) + (j * num_split + calc_idx) * slice_size,
          slice_size,
          ncclInt8,
          g_world_size * j / num_slices_per_split,
          g_nccl_comm,
          get_nccl_stream().stream()));
      CHECK_EQ(0, ncclRecv(
          ((char*)output_list[calc_idx].data_ptr()) + j * slice_size,
          slice_size,
          ncclInt8,
          g_world_size * j / num_slices_per_split,
          g_nccl_comm,
          get_nccl_stream().stream()));
    }
    CHECK_EQ(0, ncclGroupEnd());

    // Release calc_idx-th event
    g_cuda_events[calc_idx].record(get_nccl_stream());
  }

  // Switch to original stream
  at::cuda::setCurrentCUDAStream(original_stream);

  return output_list;
}

static torch::Tensor nccl_all_to_all_gather_async(
    const std::vector<torch::Tensor> &input_list,
    torch::IntArrayRef output_size,
    int num_split,
    int num_slices_per_split,
    bool is_backward) {
  CHECK_LE(num_split, g_cuda_events.size());
  CHECK_EQ(num_split, input_list.size());
  for (auto& input : input_list) {
    CHECK_CUDA(input);
  }

  CHECK_EQ(0, num_slices_per_split % g_world_size);

  // Save original stream and switch to NCCL stream
  // Output tensor must be allocated in NCCL stream context to prevent PyTorch Caching Allocator from recycling it
  const at::cuda::CUDAStream& original_stream = at::cuda::getCurrentCUDAStream();
  at::cuda::setCurrentCUDAStream(get_nccl_stream());

  // Computation stream allocator will add blocking event to nccl stream after nccl kernels
  for (auto& input : input_list) {
    c10::cuda::CUDACachingAllocator::recordStream(input.storage().data_ptr(), get_nccl_stream());
  }

  torch::Tensor output = torch::empty(output_size, torch::TensorOptions().dtype(input_list[0].dtype()).device(input_list[0].device()));
  size_t length = output.nbytes();
  size_t num_slices = num_slices_per_split * num_split;
  CHECK_EQ(0, length % num_slices);
  size_t slice_size = length / num_slices;
  // NCCL stream allocator will add blocking event to computation stream after computation kernels
  c10::cuda::CUDACachingAllocator::recordStream(output.storage().data_ptr(), original_stream);

  for (int i = 0; i < num_split; i++) {
    // Reverse calculation order in backward for pipelining
    int calc_idx = is_backward ? num_split - 1 - i : i;

    // Acquire calc_idx-th event
    g_cuda_events[calc_idx].block(get_nccl_stream());

    CHECK_EQ(0, ncclGroupStart());
    for (int j = 0; j < num_slices_per_split; j++) {
      CHECK_EQ(0, ncclSend(
          ((char*)input_list[calc_idx].data_ptr()) + j * slice_size,
          slice_size,
          ncclInt8,
          g_world_size * j / num_slices_per_split,
          g_nccl_comm,
          get_nccl_stream().stream()));
      CHECK_EQ(0, ncclRecv(
          ((char*)output.data_ptr()) + (j * num_split + calc_idx) * slice_size,
          slice_size,
          ncclInt8,
          g_world_size * j / num_slices_per_split,
          g_nccl_comm,
          get_nccl_stream().stream()));
    }
    CHECK_EQ(0, ncclGroupEnd());
  }

  // Release 0-th event for single output
  g_cuda_events[0].record(get_nccl_stream());

  // Switch to original stream
  at::cuda::setCurrentCUDAStream(original_stream);

  return output;
}

static torch::Tensor nccl_all_to_all_2d_async(torch::Tensor &input) {
  CHECK_CUDA(input);
  CHECK_CONTIGUOUS(input);

  size_t length = input.nbytes();
  CHECK_EQ(0, length % g_world_size);
  size_t slice_size = length / g_world_size;
  size_t slice_size_uint4 = slice_size / sizeof(uint4);

  // Save original stream and switch to NCCL stream
  // Output tensors must be allocated in NCCL stream context to prevent PyTorch Caching Allocator from recycling it
  const at::cuda::CUDAStream& original_stream = at::cuda::getCurrentCUDAStream();
  at::cuda::setCurrentCUDAStream(get_nccl_stream());

  // Computation stream allocator will add blocking event to nccl stream after nccl kernels
  c10::cuda::CUDACachingAllocator::recordStream(input.storage().data_ptr(), get_nccl_stream());

  int nranks = g_world_size, ngpus = g_local_size;
  CHECK_EQ(0, nranks % ngpus);
  int nnodes = nranks / ngpus;

  torch::Tensor tmp_output = torch::empty_like(input, torch::MemoryFormat::Contiguous);
  void* input_buff = (void*)input.data_ptr();
  void* tmp_output_buff = (void*)tmp_output.data_ptr();

  if (!(ngpus == 1 || nnodes == 1)) {
    int node_rank = g_world_rank / ngpus, local_rank = g_local_rank;

    // phase 0. per-gpu (ngpus) stride copy
    slice_size < sizeof(uint4)
      ? jit::jit_execute(
        {&tmp_output_buff, &input_buff, &slice_size, &ngpus, &nnodes}, mem_stride_copy_char_fd,
        input.device().index(), mem_stride_copy_gridsize, mem_stride_copy_blocksize, get_nccl_stream().stream())
      : jit::jit_execute(
        {&tmp_output_buff, &input_buff, &slice_size_uint4, &ngpus, &nnodes}, mem_stride_copy_uint4_fd,
        input.device().index(), mem_stride_copy_gridsize, mem_stride_copy_blocksize, get_nccl_stream().stream());

    // phase 1. intra-node alltoall
    CHECK_EQ(0, ncclGroupStart());
    for (int g = 0; g < ngpus; g++) {
      CHECK_EQ(0, ncclSend(((char*)tmp_output_buff) + g * nnodes * slice_size, nnodes * slice_size, ncclInt8, g + node_rank * ngpus, g_nccl_comm, get_nccl_stream().stream()));
      CHECK_EQ(0, ncclRecv(((char*)input_buff) + g * nnodes * slice_size, nnodes * slice_size, ncclInt8, g + node_rank * ngpus, g_nccl_comm, get_nccl_stream().stream()));
    }
    CHECK_EQ(0, ncclGroupEnd());

    // phase 2. per-gpu (nnodes) stride copy
    slice_size < sizeof(uint4)
      ? jit::jit_execute(
        {&tmp_output_buff, &input_buff, &slice_size, &nnodes, &ngpus}, mem_stride_copy_char_fd,
        input.device().index(), mem_stride_copy_gridsize, mem_stride_copy_blocksize, get_nccl_stream().stream())
      : jit::jit_execute(
        {&tmp_output_buff, &input_buff, &slice_size_uint4, &nnodes, &ngpus}, mem_stride_copy_uint4_fd,
        input.device().index(), mem_stride_copy_gridsize, mem_stride_copy_blocksize, get_nccl_stream().stream());

    // phase 3. inter-node alltoall
    CHECK_EQ(0, ncclGroupStart());
    for (int n = 0; n < nnodes; n++) {
      CHECK_EQ(0, ncclSend(((char*)tmp_output_buff) + n * ngpus * slice_size, ngpus * slice_size, ncclInt8, n * ngpus + local_rank, g_nccl_comm, get_nccl_stream().stream()));
      CHECK_EQ(0, ncclRecv(((char*)input_buff) + n * ngpus * slice_size, ngpus * slice_size, ncclInt8, n * ngpus + local_rank, g_nccl_comm, get_nccl_stream().stream()));
    }
    CHECK_EQ(0, ncclGroupEnd());

    // Switch to original stream
    at::cuda::setCurrentCUDAStream(original_stream);

    return input;
  } else {
    CHECK_EQ(0, ncclGroupStart());
    for (int r = 0; r < nranks; r++) {
      CHECK_EQ(0, ncclSend(((char*)input_buff) + r * slice_size, slice_size, ncclInt8, r, g_nccl_comm, get_nccl_stream().stream()));
      CHECK_EQ(0, ncclRecv(((char*)tmp_output_buff) + r * slice_size, slice_size, ncclInt8, r, g_nccl_comm, get_nccl_stream().stream()));
    }
    CHECK_EQ(0, ncclGroupEnd());

    // NCCL stream allocator will add blocking event to computation stream after computation kernels
    c10::cuda::CUDACachingAllocator::recordStream(tmp_output.storage().data_ptr(), original_stream);

    // Switch to original stream
    at::cuda::setCurrentCUDAStream(original_stream);

    return tmp_output;
  }
}

#endif


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
#if defined(USE_GPU)
    m.def("update_sdk_home",
        &jit::update_sdk_home,
        "Configure SDK HOME Path for GPU (CUDA)"
    );
    m.def("invoke",
        &jit::invoke,
        "Generic Invoke for GPU (CUDA)"
    );
    m.def("inject_source",
        &jit::inject_source,
        "Inject Source for GPU (CUDA)"
    );
#endif
    m.def("put_pg_storage", &put_pg_storage);

    m.def("invoke_cpu_fp32",
        &invoke_cpu<float>,
        "Invoke for Sparse Ops (CPU)"
    );
    m.def("invoke_cpu_fp64",
        &invoke_cpu<double>,
        "Invoke for Sparse Ops (CPU)"
    );
#if defined(USE_NCCL)
    m.def("get_nccl_unique_id_size",
        &get_nccl_unique_id_size,
        "Get size of ncclUniqueId in bytes"
    );
    m.def("get_nccl_unique_id",
        &get_nccl_unique_id,
        "Get ncclUniqueId for NCCL initialization"
    );
    m.def("init_shared_nccl",
        &init_shared_nccl,
        "NCCL initialization used for global world"
    );
    m.def("init_nccl",
        &init_nccl,
        "NCCL initialization"
    );
    m.def("current_stream_release",
        &current_stream_release,
        "Record CUDA event on current stream to i-th event slot"
    );
    m.def("current_stream_acquire",
        &current_stream_acquire,
        "Let current stream wait CUDA event in i-th event slot"
    );
    m.def("nccl_stream_release",
        &nccl_stream_release,
        "Record CUDA event on NCCL stream to i-th event slot"
    );
    m.def("nccl_stream_acquire",
        &nccl_stream_acquire,
        "Let NCCL stream wait CUDA event in i-th event slot"
    );
    m.def("nccl_all_to_all_scatter_async",
        &nccl_all_to_all_scatter_async,
        "NCCL AllToAll (Scatter Async)"
    );
    m.def("nccl_all_to_all_gather_async",
        &nccl_all_to_all_gather_async,
        "NCCL AllToAll (Gather Async)"
    );
    m.def("nccl_all_to_all_2d_async",
        &nccl_all_to_all_2d_async,
        "NCCL AllToAll (2D Async, In-place if 2DH A2A is enabled)"
    );

    m.def("batch_all_to_all_v", &batch_all_to_all_v, "NCCL AllToAllV Batched.");
    m.def("batch_all_gather_v", &batch_all_gather_v, "NCCL AllGatherV Batched.");
#endif
}


#if defined(USE_GPU)
#include <torch/script.h>
#define DEFINE_KERNEL(x, y)  static int x = -1; if (x == -1) { x = y; }

torch::Tensor warp_cumsum(torch::Tensor x) {
  CHECK_CUDA(x);
  CHECK_EQ(x.dim(), 2);
  x = x.to(torch::kInt32).contiguous();

  auto y = torch::empty_like(x);

  DEFINE_KERNEL(cumsum_fn, jit::inject_source(R"(
extern "C" __global__ void cumsum_fn(int* input0 /* (num_samples, batch_num) */, int* output0 /* (num_samples, batch_num) */, int num_samples) {
    #define thread_num  1024
    #define batch_num ((int)gridDim.x)

    __shared__ int temp[thread_num + 1];
    int thid = threadIdx.x, bid = blockIdx.x;
    int last_sum = -1;

    for (int S = 0; S < num_samples; S += thread_num, output0 += thread_num * batch_num, input0 += thread_num * batch_num) {
        int offset = 1;
        if (S + thid < num_samples)
                temp[thid] = input0[thid * batch_num + bid];
        for (int d = thread_num >> 1; d > 0; d >>= 1) {
                __syncthreads();
                if (thid < d)
                        temp[offset * (2 * thid + 2) - 1] += temp[offset * (2 * thid + 1) - 1];
                offset *= 2;
        }
        if (thid == 0)
                temp[thread_num] = temp[thread_num - 1], temp[thread_num - 1] = 0;
        for (int d = 1; d < thread_num; d *= 2) {
                offset >>= 1;
                __syncthreads();
                if (thid < d) {
                        int ai = offset * (2 * thid + 1) - 1;
                        int bi = offset * (2 * thid + 2) - 1;
                        int t = temp[ai];
                        temp[ai] = temp[bi];
                        temp[bi] += t;
                }
        }
        __syncthreads();
        if (S + thid < num_samples)
                output0[thid * batch_num + bid] = temp[thid + 1] + last_sum;
        __syncthreads();
        last_sum += temp[thread_num];
    }
}
)"));

  jit::jit_execute_with_values({x.data_ptr(), y.data_ptr(), (void*)x.size(0)}, cumsum_fn, x.device().index(), x.size(1), 1024, nullptr);
  return y;
}

torch::Tensor warp_sparse_bmm_infer(const torch::Tensor &x, const torch::Tensor &w, const torch::Tensor &sparse_groups_device, bool w_transpose, int64_t sparse_size) {
  auto sparse_groups = sparse_groups_device.cpu().to(torch::kInt32);
  auto group_ptr = ((int*)sparse_groups.data_ptr());

  auto y = torch::empty({x.size(0), x.size(1), w_transpose ? w.size(1) : w.size(2)}, torch::TensorOptions().dtype(x.dtype()).device(x.device()));

  // auto hCublas = at::cuda::getCurrentCUDABlasHandle();  -- Wait Pytorch to add builtin support for cublasSgemmBatched()
  for (int i = 0; i < sparse_groups.size(0); ++i) {
    int group_size = group_ptr[i];
    if (group_size > 0) {
      auto y_sub = y.select(0, i).narrow(0, 0, int(group_size * sparse_size));
      torch::matmul_out(y_sub, x.select(0, i).narrow(0, 0, int(group_size * sparse_size)), w_transpose ? w.select(0, i).t() : w.select(0, i));
    }
  }
  return y;
}

#if defined(USE_NCCL)

static int get_world_size() {
  static int world_size = getenv("WORLD_SIZE") ? std::atoi(getenv("WORLD_SIZE")) : 1;
  return (world_size);
}

static int get_world_rank() {
  static int world_rank = getenv("RANK") ? std::atoi(getenv("RANK")) : 0;
  return (world_rank);
}

static at::Tensor all_gather_native(const at::Tensor &input) {
  auto it = _pg_storage.find(0);
  CHECK_NE(it, _pg_storage.end());
  auto pg = it->second;
  if (pg.get() == nullptr)
    return input;

  auto shape = input.sizes().vec();
  size_t size_per_group = shape[0];
  shape[0] *= pg->getSize();
  auto output = torch::empty(shape, torch::TensorOptions().dtype(input.dtype()).device(input.device()));

  std::vector<at::Tensor> inputs = {input};
  std::vector<std::vector<at::Tensor>> outputs(inputs.size());
  for (int i = 0; i < output.size(0); ++i)
    outputs[0].push_back(output.narrow(0, i * size_per_group, size_per_group));
  c10::intrusive_ptr<c10d::Work> work = pg->allgather(outputs, inputs);
  work->wait();

  return output;
}

static std::tuple<torch::Tensor, torch::Tensor> uncached_empty_ex(torch::IntArrayRef shape, at::ScalarType dtype) {
  int64_t size = std::accumulate(shape.begin(), shape.end(), 1LL, std::multiplies<int64_t>()) * torch::elementSize(dtype);

  auto device_index = c10::cuda::current_device();
  at::DeviceGuard device_guard(at::Device(at::DeviceType::CUDA, device_index));
  void* buffer = nullptr;
  cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
  auto stream = c10::cuda::getCurrentCUDAStream().stream();
  AT_CUDA_CHECK(cudaThreadExchangeStreamCaptureMode(&mode));

#if defined(USE_ROCM)
  AT_CUDA_CHECK(hipExtMallocWithFlags((void**)&buffer, size, hipDeviceMallocUncached));
#else
  AT_CUDA_CHECK(cudaMalloc((void**)&buffer, size));
#endif
  AT_CUDA_CHECK(cudaMemsetAsync(buffer, 0, size, stream));
  AT_CUDA_CHECK(cudaStreamSynchronize(stream));
  AT_CUDA_CHECK(cudaThreadExchangeStreamCaptureMode(&mode));
  auto t = torch::from_blob(buffer, shape, torch::TensorOptions().dtype(dtype).device(torch::kCUDA));
  CHECK_EQ(t.data_ptr(), buffer);

  auto options = torch::TensorOptions().dtype(torch::kUInt8);
  auto handle = torch::empty({1, static_cast<int64_t>(sizeof(cudaIpcMemHandle_t))}, options.device(torch::kCPU));
  AT_CUDA_CHECK(cudaIpcGetMemHandle((cudaIpcMemHandle_t*)handle.data_ptr(), buffer));

  auto handles = all_gather_native(handle);
  int rank = get_world_rank();

  CHECK_EQ(handles.dim(), 2);
  int scope_size = handles.size(0);
  auto pointers = torch::empty({scope_size}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));

  for (int i = 0; i < scope_size; ++i) {
    if (i == rank)
      pointers[i] = reinterpret_cast<int64_t>(t.data_ptr());
    else {
      void* ipc_ptr = nullptr;
      AT_CUDA_CHECK(cudaIpcOpenMemHandle(
        (void**)&ipc_ptr, *((const cudaIpcMemHandle_t*)handles[i].data_ptr()),
        cudaIpcMemLazyEnablePeerAccess));
      pointers[i] = reinterpret_cast<int64_t>(ipc_ptr);
    }
  }
  return {t, pointers.to(torch::kCUDA)};
}


template<class FN>
static void perform(FN func, int num_runs = 1000) {
  for (int _ = 0; _ < 5; ++_)
    func();
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  puts("=======================");
  void *h1 = ab::recordTime(stream);
  for (int _ = 0; _ < num_runs; ++_)
    func();
  void *h2 = ab::recordTime(stream);
  printf("Perform: %g\n", ab::convertToElapsedTime(h1, h2) / num_runs);
}

static void master_print(const std::vector<torch::Tensor> &xs, int64_t rank = -1) {
  if (get_world_rank() != 0 && rank != 0)
    return;
  puts("=======================");
  for (auto &x: xs) {
    printf("[");
    for (int i = 0; i < x.dim(); ++i) printf("%d, ", x.size(i));
    printf("] (dtype_size = %d) data = ", int(torch::elementSize(torch::typeMetaToScalarType(x.dtype()))));
    auto x_ = x.to(torch::kFloat32).to(torch::kCPU);
    if (x.numel() > 10) {
      for (int i = 0; i < 5; ++i) printf("%g, ", x_.data_ptr<float>()[i]);
      printf("..");
      for (int i = 0; i < 5; ++i) printf("%g, ", x_.data_ptr<float>()[x.numel() - 5 + i]);
    } else {
      for (int i = 0; i < x.numel(); ++i) printf("%g, ", x_.data_ptr<float>()[i]);
    }
    puts("");
  }
}


std::tuple<torch::Tensor, torch::Tensor> warp_to_float8_block(torch::Tensor w) {
  CHECK_CUDA(w);
  CHECK_EQ(w.dtype(), torch::kBFloat16);

  bool has_batch = w.dim() > 2;
  if (!has_batch)
    w = w.unsqueeze(0);
  CHECK_EQ(w.dim(), 3);
  CHECK_EQ(w.size(1) % 128, 0);
  CHECK_EQ(w.size(2) % 128, 0);

  w = w.view({w.size(0), w.size(1) / 128, 128, w.size(2) / 128, 128});
  auto scal = torch::empty({w.size(0), w.size(1), w.size(3)}, torch::TensorOptions().dtype(torch::kFloat32).device(w.device()));
  auto fp8_w = antares::ops::call("to_float8_block", {w.view(torch::kInt32), scal}, {}).view(at::kFloat8_e4m3fn).flatten(1, 2).flatten(2, 3);
  if (!has_batch)
    fp8_w = fp8_w.squeeze(0), scal = scal.squeeze(0);
  return {fp8_w, scal};
}

std::tuple<torch::Tensor, torch::Tensor> warp_to_float8_per_token(const torch::Tensor &w) {
  CHECK_CUDA(w);
  CHECK_EQ(w.dtype(), torch::kBFloat16);

  auto local_w = w.view({-1, 128}).view(torch::kInt32);
  auto scal = torch::empty({local_w.size(0)}, torch::TensorOptions().dtype(torch::kFloat32).device(w.device()));
  auto fp8_w = antares::ops::call("to_float8_per_token", {local_w, scal}, {}).view(at::kFloat8_e4m3fn).view(w.sizes());
  return {fp8_w, scal.view(fp8_w.narrow(-1, 0, fp8_w.size(-1) / 128).sizes())};
}

torch::Tensor warp_scaled_mask_inv(const torch::Tensor &x, const torch::Tensor &range, double scale_inv) {
  CHECK_CUDA(x);
  CHECK_EQ(x.dtype(), torch::kBFloat16);
  CHECK_EQ(range.dtype(), torch::kInt32);
  auto out = antares::ops::call("scaled_mask_inv_bf16", {x.view({-1, x.size(-1)}), range}, {(float)scale_inv}).view(x.sizes());
  return out;
}

torch::Tensor warp_topk_token_sort(
  const torch::Tensor &topk_ids,
  const torch::Tensor &num_tokens_post_padded,
  int64_t num_pages
) {
  const int E = num_tokens_post_padded.numel();
  auto sorted_token_ids = torch::empty({E, num_pages}, torch::TensorOptions().dtype(torch::kInt32).device(topk_ids.device()));
  return antares::ops::call("token_sort_i32", {topk_ids.flatten(), num_tokens_post_padded, sorted_token_ids}, {}).flatten();
}

torch::Tensor warp_scatter_sample_ids(const torch::Tensor &expert_ids, const torch::Tensor &location_ids, const torch::Tensor &out, int64_t capacity, int64_t num_samples, bool return_top_id) {
  CHECK_CUDA(out);
  CHECK_EQ(expert_ids.dtype(), torch::kInt32);
  CHECK_EQ(location_ids.dtype(), torch::kInt32);
  CHECK_EQ(out.dtype(), torch::kInt32);
  CHECK_EQ(capacity > 0, true);

  if (return_top_id)
    antares::ops::call("scatter_top_ids_i32", {expert_ids.flatten(), location_ids.flatten(), out.flatten()}, {capacity, num_samples});
  else
    antares::ops::call("scatter_sample_ids_i32", {expert_ids.flatten(), location_ids.flatten(), out.flatten()}, {capacity, num_samples});
  return out;
}


torch::Tensor warp_to_bfloat16(const torch::Tensor &w, const torch::Tensor &scal) {
  CHECK_CUDA(w);
  if (w.dtype() == torch::kBFloat16)
    return w;

  CHECK_CUDA(scal);
  auto w_ = w, scal_ = scal;
  if (w_.dim() < 3)
    w_ = w_.unsqueeze(0), scal_ = scal_.unsqueeze(0);
  CHECK_EQ(w_.dim(), 3);
  if (scal_.dim() + 1 == w_.dim()) {
    if (w_.size(-1) != scal_.size(-1))
      w_ = (w_.to(torch::kFloat32) * scal_.unsqueeze(-1)).to(torch::kBFloat16);
    else
      w_ = (w_.to(torch::kFloat32) * scal_.unsqueeze(-2)).to(torch::kBFloat16);
  } else {
    CHECK_EQ(scal_.dim(), 3);
    auto padded_w = torch::empty({w_.size(0), (w_.size(1) + 127) / 128, 128, (w_.size(2) + 127) / 128, 128}, torch::TensorOptions().dtype(w_.dtype()).device(w_.device()));
    CHECK_EQ(padded_w.size(0), scal_.size(0));
    CHECK_EQ(padded_w.size(1), scal_.size(1));
    CHECK_EQ(padded_w.size(3), scal_.size(2));
    padded_w.flatten(1, 2).flatten(2, 3).narrow(1, 0, w_.size(1)).narrow(2, 0, w_.size(2)).copy_(w_);
    padded_w = padded_w.view(at::kFloat8_e4m3fn).to(torch::kBFloat16) * scal_.to(torch::kBFloat16).view({scal_.size(0), scal_.size(1), 1, scal_.size(2), 1});
    w_ = padded_w.flatten(1, 2).flatten(2, 3).narrow(1, 0, w_.size(1)).narrow(2, 0, w_.size(2)).contiguous();
  }
  if (w.dim() < 3)
    w_ = w_.squeeze(0);
  return w_;
}

torch::Tensor warp_gemm_nt_bf16xfp8_block_scal_out(const torch::Tensor &x, torch::Tensor w, const torch::Tensor &scal, const ::std::optional<torch::Tensor> &w_alt, const ::std::optional<torch::Tensor> &p_out) {
  CHECK_CUDA(x);
  CHECK_EQ(x.dim(), 3);
  CHECK_EQ(x.dtype(), torch::kBFloat16);
  CHECK_EQ(w.dim(), 2);

  int samples = x.size(0) * x.size(1);
  if (samples > 1 and w_alt.has_value())
    w = w_alt.value();

  auto out = p_out.has_value() ? p_out.value().view({samples, w.size(0)}) : torch::empty({samples, w.size(0)}, torch::TensorOptions().dtype(x.dtype()).device(x.device()));

  CHECK_EQ(out.dtype(), torch::kBFloat16);

  if (w.dtype() == torch::kBFloat16) {
    torch::matmul_out(out, x.view({samples, -1}), w.t());
  } else {
    CHECK_EQ(scal.dim(), 2);
#if IS_NVIDIA_GPU
    antares::ops::call("gemv_nt_bf16xfp8_block_v2", {x.view({samples, x.size(-1)}).view(torch::kInt64), w.view(torch::kInt32), scal, out}, {}, false, 0, 3);
#else
    antares::ops::call("gemv_nt_bf16xfp8_block_v2", {x.view({samples, x.size(-1)}).view(torch::kInt32), w.view(at::kComplexDouble), scal, out}, {}, false, 0, 3);
#endif
  }
  return out.view({x.size(0), x.size(1), w.size(0)});
}

torch::Tensor warp_rmsnorm_bf16(const torch::Tensor &x, const torch::Tensor &rms_w, double eps, int64_t id = 0) {
  CHECK_CUDA(x);
  CHECK_EQ(x.dim(), 3);
  CHECK_EQ(x.dtype(), torch::kBFloat16);
  auto out = torch::empty({x.size(0), x.size(1), rms_w.size(0)}, torch::TensorOptions().dtype(x.dtype()).device(x.device()));
  CHECK_EQ(id % 4, 0);
  antares::ops::call("rmsnorm2_bf16", {x.view({-1, x.size(-1)}).view(torch::kInt64), rms_w.view(torch::kInt64), out}, {eps, id / 4}, false, 0, 2);
  return out;
}

std::tuple<torch::Tensor, torch::Tensor> warp_deepseek_sigmoid_top_8_static_v2(
     const torch::Tensor &logits_bf16,
     const torch::Tensor &moe_gate_b_bf16,
     const ::std::optional<torch::Tensor> &top_v_out_,
     const ::std::optional<torch::Tensor> &top_k_out_) {
  CHECK_CUDA(logits_bf16);
  CHECK_EQ(logits_bf16.dtype(), torch::kBFloat16);
  CHECK_EQ(moe_gate_b_bf16.dtype(), torch::kBFloat16);

  int n_experts = logits_bf16.size(-1);
  int samples = logits_bf16.numel() / n_experts;

  auto device = logits_bf16.device();
  auto top_v_out = top_v_out_.has_value() ? top_v_out_.value().view({samples, -1}) : torch::empty({samples, 8}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
  auto top_k_out = top_k_out_.has_value() ? top_k_out_.value().view({samples, -1}) : torch::empty({samples, 8}, torch::TensorOptions().dtype(torch::kInt32).device(device));
  AT_ASSERTM(top_v_out.dtype() == torch::kFloat32 && top_k_out.dtype() == torch::kInt32, "Output tensor space should be float32 for top_scores and int32 for top_ids.");

  antares::ops::call("deepseek_r1_sigmoid_top_k_routed_scaled_f32", {logits_bf16.view({samples, n_experts}), moe_gate_b_bf16, top_v_out, top_k_out}, {}, false, 0, 3);
  return {top_v_out, top_k_out};
}

std::tuple<torch::Tensor, torch::Tensor> warp_qwen3_moe_top_8_static(
     const torch::Tensor &logits_fp32) {
  CHECK_CUDA(logits_fp32);
  CHECK_EQ(logits_fp32.dtype(), torch::kFloat32);

  int n_experts = logits_fp32.size(-1);
  int samples = logits_fp32.numel() / n_experts;

  auto device = logits_fp32.device();
  auto top_v_out = torch::empty({samples, 8}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
  auto top_k_out = torch::empty({samples, 8}, torch::TensorOptions().dtype(torch::kInt32).device(device));

  antares::ops::call("qwen3_moe_top_k_routed_scaled_f32", {logits_fp32.view({samples, n_experts}), top_v_out, top_k_out}, {}, false, 0, 2);
  return {top_v_out, top_k_out};
}

std::tuple<torch::Tensor, torch::Tensor> warp_kimi_sigmoid_top_8_static_v2(
     const torch::Tensor &logits_bf16,
     const torch::Tensor &moe_gate_b_bf16,
     const ::std::optional<torch::Tensor> &top_v_out_,
     const ::std::optional<torch::Tensor> &top_k_out_) {
  CHECK_CUDA(logits_bf16);
  CHECK_EQ(logits_bf16.dtype(), torch::kBFloat16);
  CHECK_EQ(moe_gate_b_bf16.dtype(), torch::kBFloat16);

  int n_experts = logits_bf16.size(-1);
  int samples = logits_bf16.numel() / n_experts;

  auto device = logits_bf16.device();
  auto top_v_out = top_v_out_.has_value() ? top_v_out_.value().view({samples, -1}) : torch::empty({samples, 8}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
  auto top_k_out = top_k_out_.has_value() ? top_k_out_.value().view({samples, -1}) : torch::empty({samples, 8}, torch::TensorOptions().dtype(torch::kInt32).device(device));
  AT_ASSERTM(top_v_out.dtype() == torch::kFloat32 && top_k_out.dtype() == torch::kInt32, "Output tensor space should be float32 for top_scores and int32 for top_ids.");

  antares::ops::call("kimi_k2_sigmoid_top_k_routed_scaled_f32", {logits_bf16.view({samples, n_experts}), moe_gate_b_bf16, top_v_out, top_k_out}, {}, false, 0, 3);
  return {top_v_out, top_k_out};
}

torch::Tensor warp_qwen3_norm_rotary_kvcache2_bf16(
     const torch::Tensor &cos_cache,
     const torch::Tensor &sin_cache,
     const torch::Tensor &positions,
     const torch::Tensor &qkv_out,
     const torch::Tensor &key_cache,
     const torch::Tensor &val_cache,
     const torch::Tensor &qk_norm,
     int64_t n_heads
) {
  int64_t local_kv_heads = key_cache.size(-2);
  auto q_out = antares::ops::call("qwen3_norm_rotary_kvcache2_bf16", {cos_cache, sin_cache, positions.flatten(),
    qkv_out.view(torch::kInt32), key_cache.view(torch::kInt32), val_cache.view(torch::kInt32), qk_norm.view(torch::kInt32)}, {n_heads, 1e-6, n_heads + local_kv_heads}).view(torch::kBFloat16);
  return q_out.narrow(-2, 0, n_heads);
}

torch::Tensor warp_multi_head_latent_rope_bf16_v3(
  const torch::Tensor &qkv_act,
  const torch::Tensor &cos_sin,
  const torch::Tensor &q_a_norm,
  const torch::Tensor &kv_a_norm,
  const torch::Tensor &q_b_proj,
  const torch::Tensor &k_b_proj,
  const torch::Tensor &kv_ranges,
  const torch::Tensor &kv_indices,
  const torch::Tensor &kv_cache,
  int64_t n_local_heads
) {
  auto x = qkv_act;
  CHECK_CUDA(x);
  CHECK_EQ(x.dtype(), torch::kBFloat16);
  CHECK_EQ(x.dim(), 3);
  CHECK_EQ(x.size(-1), 2112);
  CHECK_EQ(cos_sin.dtype(), torch::kInt64);
  CHECK_EQ(kv_ranges.dtype(), torch::kInt32);
  CHECK_EQ(q_b_proj.dtype(), torch::kBFloat16);

  int batch = qkv_act.size(0), seqlen = qkv_act.size(1);
  int samples = batch * seqlen;

  auto q = antares::ops::call("rope_norms_to_kvcache_bf16", {cos_sin, kv_a_norm, q_a_norm, x.flatten(0, 1), kv_indices, kv_cache.view({-1, 576}), kv_ranges}, {}).view({x.size(0), x.size(1), -1});

  CHECK_EQ(q_b_proj.dtype(), torch::kBFloat16);
  CHECK_EQ(q_b_proj.dim(), 2);
  CHECK_EQ(k_b_proj.dtype(), torch::kBFloat16);
  CHECK_EQ(k_b_proj.dim(), 3);
  CHECK_CONTIGUOUS(k_b_proj.transpose(1, 2));

  auto q_output = torch::empty({batch, seqlen, n_local_heads, 512 + 64}, torch::TensorOptions().dtype(q.dtype()).device(q.device()));
  torch::Tensor qh = (IS_NVIDIA_GPU || samples >= 4) ? torch::matmul(q, q_b_proj.t()).view({samples, n_local_heads, -1}) : \
    antares::ops::call("rope_gmv_bf16", {q.view({samples, -1}).view(torch::kInt32), q_b_proj.view(torch::kInt32)}, {}).view({samples, n_local_heads, -1}); // (BS, 1536) @ (192 x H, 1536)
  auto buffer = q_output.flatten(0, 1).transpose(0, 1).narrow(-1, 0, 512);
  torch::matmul_out(buffer, qh.transpose(0, 1).narrow(-1, 0, 128), k_b_proj); // (H, BS, 128) @ (H, 512, 128)
  antares::ops::call("rope_q_out_bf16", {cos_sin, qh.view({qh.size(0), -1, 3, 64}).view(torch::kInt32), kv_ranges, q_output.view({qh.size(0), -1, 9, 64}).view(torch::kInt32)}, {});
  return q_output;
}

#if IS_NVIDIA_GPU == 0
#if defined(__has_include)
#if __has_include("extensions/mla_decode.h")
#include "extensions/mla_decode.h"
#endif
#endif
#endif

torch::Tensor warp_deepseek_custom_mla_bf16(
  const torch::Tensor &x,
  const torch::Tensor &key_cache,
  const torch::Tensor &kv_range,
  const torch::Tensor &kv_indices,
  const torch::Tensor &wvc,
  double softmax_scale,
  bool update_indices,
  int64_t opt
) {
  CHECK_CUDA(x);
  CHECK_EQ(x.dim(), 4);
  int batch = x.size(0), seqlen = x.size(1);
  CHECK_EQ(seqlen, 1);

  {
    auto Q = x;
#if defined(CUSTOM_MLA_DECODE)
    {
      if (update_indices)
        antares::ops::call("kv_range_to_indice", {kv_range.narrow(0, 0, kv_range.size(0) - 1), kv_indices}, {});

      Q = mla_decode_fwd(Q, key_cache, kv_range, kv_indices, softmax_scale).view({batch * seqlen, Q.size(-2), -1});
      if (opt == 0)
        Q = antares::ops::call("wvc_logits_bf16", {Q.view(torch::kInt32), wvc.view(torch::kInt32)}, {});
      else
        Q = at::einsum("bhc,hdc->bhd", {Q, wvc}).contiguous();
    }
#else
    {
      AT_ASSERTM(false, "Custom MLA not implemented.");
    }
#endif
    return Q.view({batch, seqlen, -1});
  }
}

static torch::Tensor warp_intra_add_allreduce_bf16(const torch::Tensor &x, const torch::Tensor &t,
    const std::tuple<torch::Tensor, torch::Tensor> &sigp, const std::tuple<torch::Tensor, torch::Tensor> &buffer, bool copy = true) {
  CHECK_EQ(t.dtype(), torch::kBFloat16);
  CHECK_CUDA(t);

  if (get_world_size() == 1)
    return x + t;

  auto buf = std::get<0>(buffer).flatten();
  if (copy)
    buf.narrow(0, 0, t.numel()).copy_(t.flatten());
  static torch::Tensor v_count = torch::zeros({1024 * 1024}, torch::TensorOptions().dtype(torch::kInt64).device(x.device()));
  int scope_size = std::get<1>(sigp).numel();
  std::vector<torch::Tensor> args = {x.flatten().view(torch::kInt32), std::get<1>(buffer), std::get<0>(sigp), std::get<1>(sigp), v_count};
  if (scope_size == 8)
    return antares::ops::call("sig_allreduce_bf16_u8", args, {get_world_rank()}).view(torch::typeMetaToScalarType(x.dtype())).view(x.sizes());
  if (scope_size == 4)
    return antares::ops::call("sig_allreduce_bf16_u4", args, {get_world_rank()}).view(torch::typeMetaToScalarType(x.dtype())).view(x.sizes());
  if (scope_size == 2)
    return antares::ops::call("sig_allreduce_bf16_u2", args, {get_world_rank()}).view(torch::typeMetaToScalarType(x.dtype())).view(x.sizes());
  CHECK_EQ(scope_size, 1);
  return x + t;
}

static torch::Tensor warp_x_add_allreduce_y_bf16(const torch::Tensor &x, const torch::Tensor &t) {
  CHECK_EQ(t.dtype(), torch::kBFloat16);
  CHECK_CUDA(t);

  if (get_world_size() == 1)
    return x + t;

  CHECK_NE(shared_nccl_comm, nullptr);
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  static torch::Tensor t_out = torch::empty_like(x);
  ncclAllReduce(t.data_ptr(), t_out.data_ptr(), t.numel(), ncclBfloat16, ncclSum, (ncclComm_t)shared_nccl_comm, stream);
  return x + t_out;
}

torch::Tensor warp_shared_expert_bf16xf8(
  const torch::Tensor &x,
  const torch::Tensor &moe_gate_up_w,
  const torch::Tensor &moe_gate_up_s,
  const torch::Tensor &moe_down_w,
  const torch::Tensor &moe_down_s
) {
    int model_dim = x.size(-1);
    int samples = x.numel() / model_dim;

    CHECK_EQ(moe_gate_up_s.size(0), 1);

    static std::unordered_map<void*, torch::Tensor> shared_gate_up, shared_down;
    auto it = shared_gate_up.find(moe_gate_up_s.data_ptr());
    if (it == shared_gate_up.end()) {
      shared_gate_up[moe_gate_up_s.data_ptr()] = warp_to_bfloat16(moe_gate_up_w, moe_gate_up_s).squeeze(0);
      it = shared_gate_up.find(moe_gate_up_s.data_ptr());
    }
    auto xb = torch::matmul(x, it->second.t());
    it = shared_down.find(moe_down_s.data_ptr());
    if (it == shared_down.end()) {
      shared_down[moe_down_s.data_ptr()] = warp_to_bfloat16(moe_down_w, moe_down_s).squeeze(0);
      it = shared_down.find(moe_down_s.data_ptr());
    }
    xb = antares::ops::call("fused_silu_mul_bf16", {xb.view({samples, 2, xb.size(-1) / 2})}, {}).view({xb.size(0), xb.size(1), -1});
    xb = torch::matmul(xb, it->second.t());
    return xb.view({x.size(0), x.size(1), moe_down_w.size(1)});
}

torch::Tensor warp_glu_expert_bf16xf4_group_scal(
  const torch::Tensor &x,
  const torch::Tensor &expert_ids,
  const torch::Tensor &expert_weight,
  const torch::Tensor &gateup_w,
  const torch::Tensor &gateup_s,
  const torch::Tensor &gateup_i,
  const torch::Tensor &gateup_m,
  const torch::Tensor &down_w,
  const torch::Tensor &down_s,
  const torch::Tensor &down_i,
  const torch::Tensor &down_m,
  const torch::Tensor &out
) {
  CHECK_EQ(x.dim(), 3);
  int samples = x.size(0) * x.size(1), model_dim = x.size(2);
  int select_size = expert_ids.numel();
  int num_top_k = expert_ids.size(-1);
#if IS_NVIDIA_GPU
  auto y = antares::ops::call("fmoe_f16xf4_phase_1_top_k", {x.view({samples, -1, 8}).view(torch::kInt32), gateup_s, gateup_m, expert_ids.view({select_size}), gateup_w.view(torch::kFloat32)}, {num_top_k}).view({select_size, -1});
  antares::ops::call("fmoe_f16xf4_phase_2", {y.view({samples, expert_ids.size(1), 2, -1, 8}).view(torch::kInt32), down_s, down_m, expert_ids, expert_weight, down_w.view(torch::kFloat32), out}, {}, false, 0, 6);
#else
  auto y = antares::ops::call("fmoe_f16xf4_phase_1_top_k", {x.view({samples, -1, 16}).view(torch::kInt32), gateup_s, gateup_m, expert_ids.view({select_size}), gateup_w.view(torch::kFloat64)}, {num_top_k}).view({select_size, -1});
  antares::ops::call("fmoe_f16xf4_phase_2", {y.view({samples, expert_ids.size(1), 2, -1, 16}).view(torch::kInt32), down_s, down_m, expert_ids, expert_weight, down_w.view(torch::kFloat64), out}, {}, false, 0, 6);
#endif
  return out.view(x.sizes());
}


torch::Tensor warp_glu_expert_bf16xf8_block_scal(
  const torch::Tensor &x,
  const torch::Tensor &expert_ids,
  const torch::Tensor &expert_weight,
  const torch::Tensor &moe_gate_up_w,
  const torch::Tensor &moe_gate_up_s,
  const torch::Tensor &moe_down_w,
  const torch::Tensor &moe_down_s,
  const torch::Tensor &out) {

  int model_dim = x.size(-1);
  int samples = x.numel() / model_dim;

  CHECK_CUDA(x);
  CHECK_EQ(x.dtype(), torch::kBFloat16);
  CHECK_EQ(x.dim(), 3);
  CHECK_EQ(expert_ids.dim(), 2);
  CHECK_EQ(expert_weight.dim(), 2);
  CHECK_EQ(expert_ids.size(1), expert_weight.size(1));

  if (samples == 1) {
    auto xb = antares::ops::call("fmoe_f16xf8_blk128_phase_1", {x.view({samples, -1, 16}).view(torch::kInt32), expert_ids, moe_gate_up_w.view(at::kComplexDouble), moe_gate_up_s}, {});
    antares::ops::call("fmoe_f16xf8_blk128_phase_2", {xb.view({samples, expert_ids.size(1), 2, -1, 16}).view(torch::kInt32), expert_weight, expert_ids, moe_down_w.view(at::kComplexDouble), moe_down_s, out}, {}, false, 0, 5);
    return out.view({x.size(0), x.size(1), moe_down_w.size(1)});
  }

  if (moe_down_s.dim() == 2) {
    if (samples >= 4 && moe_gate_up_w.size(0) == 1)
      return warp_shared_expert_bf16xf8(x, moe_gate_up_w, moe_gate_up_s, moe_down_w, moe_down_s);

    if (samples >= 4) {
      AT_ASSERTM(moe_gate_up_w.size(1) == 512, "Branch designed for 8 GPUs.");

      auto out = warp_shared_expert_bf16xf8(x, moe_gate_up_w.narrow(0, -1, 1), moe_gate_up_s.narrow(0, -1, 1), moe_down_w.narrow(0, -1, 1), moe_down_s.narrow(0, -1, 1));
      auto xb = antares::ops::call("fmoe_w8a16_stage_1", {x.view({samples, model_dim}).view(torch::kInt32), expert_ids, moe_gate_up_w.view(torch::kInt16), moe_gate_up_s}, {}).view({expert_ids.size(0), 8, 2, moe_gate_up_w.size(-2) / 2});
      xb = antares::ops::call("fmoe_w8a16_stage_2", {xb.view(torch::kInt32), expert_weight, expert_ids, moe_down_s.view(torch::kInt64)}, {}).view({expert_ids.size(0), 8, xb.size(-1) / 2});
      xb = antares::ops::call("fmoe_w8a16_stage_3", {xb.view(torch::kInt32), out.view({samples, out.size(-1)}), expert_ids, moe_down_w.view(torch::kInt16)}, {});
      return xb.view({x.size(0), x.size(1), moe_down_w.size(1)});
    } else {
      auto xb = antares::ops::call("fmoe_w8a16_vector_1", {x.view({samples, model_dim}).view(torch::kInt32), expert_ids.view({-1}), moe_gate_up_w.view(torch::kInt16), moe_gate_up_s}, {}).view({expert_ids.size(0), expert_ids.size(1), moe_gate_up_w.size(1)});
      xb = antares::ops::call("fmoe_w8a16_vector_2", {xb.view(xb.dtype() == torch::kFloat32 ? torch::kInt64 : torch::kInt32), expert_weight, expert_ids, moe_down_w.view(torch::kInt16), moe_down_s.view(torch::kInt64)}, {});
      return xb.view({x.size(0), x.size(1), moe_down_w.size(1)});
    }
  }

  if (samples <= 4) {
    if (expert_ids.size(-1) == 9) {
      auto xb = antares::ops::call("fmoe_blkvect_phase_1", {x.view({samples, -1, 16}).view(torch::kInt32), expert_ids.flatten(), moe_gate_up_w.view(at::kComplexDouble), moe_gate_up_s}, {});
      return antares::ops::call("fmoe_blkvect_phase_2", {xb.view({samples, 9, -1, 16}).view(torch::kInt32), expert_weight, expert_ids, moe_down_w.view(at::kComplexDouble), moe_down_s}, {}).view({x.size(0), x.size(1), moe_down_w.size(1)});
    }
    auto xb = antares::ops::call("fmoe_blockscal_vector_1", {x.view({samples, model_dim}).view(torch::kInt32), expert_ids, moe_gate_up_w.view(torch::kInt16), moe_gate_up_s}, {});
    return antares::ops::call("fmoe_blockscal_vector_2", {xb.view({expert_ids.size(0), expert_ids.size(1), -1}).view(xb.dtype() == torch::kFloat32 ? torch::kInt64 : torch::kInt32), expert_weight, expert_ids, moe_down_w.view(torch::kInt16), moe_down_s}, {}).view({x.size(0), x.size(1), moe_down_w.size(1)});
  }

  auto partial = warp_shared_expert_bf16xf8(x, moe_gate_up_w.narrow(0, -1, 1), moe_gate_up_s.narrow(0, -1, 1), moe_down_w.narrow(0, -1, 1), moe_down_s.narrow(0, -1, 1));
  if (moe_gate_up_w.size(0) == 1)
    return partial;

  CHECK_EQ(moe_gate_up_w.size(1), 512);
  CHECK_EQ(moe_gate_up_w.size(1), 512);
  CHECK_EQ(expert_ids.size(1), 9);
  auto xb = torch::empty({2, samples, 8, 4, model_dim / 2}, torch::TensorOptions().dtype(torch::kFloat32).device(x.device()));
  antares::ops::call("fmoe_blockscal_stage_1", {x.view({samples, model_dim}).view(torch::kInt32), expert_ids, moe_gate_up_s, xb.select(0, 0), xb.select(0, 1)}, {}, false, 0, 4);
  xb = antares::ops::call("fmoe_blockscal_stage_2", {xb, expert_ids, moe_gate_up_w.view(torch::kInt16)}, {});
  xb = antares::ops::call("fmoe_blockscal_stage_3", {xb.view({xb.size(0), xb.size(1), 2, moe_down_w.size(2)}).view(torch::kInt32), expert_weight}, {});
  xb = antares::ops::call("fmoe_blockscal_stage_4", {xb.view({xb.size(0), xb.size(1), 2, xb.size(2) / 2}), partial.view({samples, model_dim}), expert_ids, moe_down_w.view({moe_down_w.size(0), moe_down_w.size(1), 2, moe_down_w.size(2) / 2}).view(torch::kInt16), moe_down_s}, {}).view({x.size(0), x.size(1), moe_down_w.size(1)});
  return xb;
}

torch::Tensor warp_gate_gemm_out_bf16(const torch::Tensor &xb, const torch::Tensor &gate_w) {
  int samples = xb.numel() / xb.size(-1);
  return samples < 4 ? antares::ops::call("gate_gemm_out_bf16", {xb.view(torch::kInt32).view({samples, -1}), gate_w.view(torch::kInt32)}, {}) : torch::matmul(xb, gate_w.t());
}

torch::Tensor warp_copy_to_device(const std::vector<torch::Tensor> &data) {
  CHECK_NE(data.size(), 0);

  auto shape = data[0].sizes().vec();
  for (int i = 1; i < data.size(); ++i)
    shape[0] += data[i].size(0);

  auto out = torch::empty(shape, torch::TensorOptions().dtype(data[0].dtype()).device(torch::kCUDA));
  char *dptr = (char*)out.data_ptr();
  auto stream = at::cuda::getDefaultCUDAStream().stream();

  for (const auto &t: data) {
    size_t partial_size = t.numel() * torch::elementSize(torch::typeMetaToScalarType(t.dtype()));
    cudaMemcpyAsync(dptr, t.data_ptr(), partial_size, cudaMemcpyHostToDevice, stream);
    dptr += partial_size;
  }
  cudaStreamSynchronize(stream);
  return out;
}

namespace specialized {

torch::Tensor warp_glu_expert_bf16xf8_block_scal_16x16_fnuz(
  const torch::Tensor &x,
  const torch::Tensor &expert_ids,
  const torch::Tensor &expert_weight,
  const torch::Tensor &moe_gate_up_w,
  const torch::Tensor &moe_gate_up_s,
  const torch::Tensor &moe_down_w,
  const torch::Tensor &moe_down_s) {

  int model_dim = x.size(-1);
  int samples = x.numel() / model_dim;

  CHECK_CUDA(x);
  CHECK_EQ(x.dtype(), torch::kBFloat16);
  CHECK_EQ(x.dim(), 3);
  CHECK_EQ(expert_ids.dim(), 2);
  CHECK_EQ(expert_ids.size(-1), 9);
  CHECK_EQ(expert_weight.dim(), 2);

  CHECK_EQ(moe_gate_up_w.dim(), 5); // shape = [256, 32, 448, 16, 16]
  CHECK_EQ(moe_gate_up_s.dim(), 3); // shape = [256, 4, 56]
  CHECK_EQ(moe_gate_up_w.size(2), 448);
  CHECK_EQ(moe_gate_up_w.size(-2), 16);
  CHECK_EQ(moe_gate_up_w.size(-1), 16);

  CHECK_EQ(moe_down_w.dim(), 5); // shape = [256, 448, 16, 16, 16]
  CHECK_EQ(moe_down_s.dim(), 3); // shape = [256, 56, 2]
  CHECK_EQ(moe_down_w.size(1), 448);
  CHECK_EQ(moe_down_w.size(-2), 16);
  CHECK_EQ(moe_down_w.size(-1), 16);

  auto _0 = moe_gate_up_w.view({moe_gate_up_w.size(0), moe_gate_up_w.size(1) * moe_gate_up_w.size(3), moe_gate_up_w.size(2) * moe_gate_up_w.size(4)});
  auto _1 = moe_down_w.view({moe_down_w.size(0), moe_down_w.size(1) * moe_down_w.size(3), moe_down_w.size(2) * moe_down_w.size(4)});
  const char *fn1 = samples < 4 ? "gemm_gate_up_silu_bf16xf8_s_16x16_fnuz_v2" : "gemm_gate_up_silu_bf16xf8_s_16x16_fnuz_bs4_v2";
  const char *fn2 = samples < 4 ? "gemm_down_weight_sum_bf16xf8_s_16x16_fnuz_v2": "gemm_down_weight_sum_bf16xf8_s_16x16_fnuz_bs4_v2";
  auto xb = antares::ops::call(fn1, {x.view({samples, -1, 16}).view(torch::kInt32), expert_ids.flatten(), moe_gate_up_w.view(at::kComplexDouble), moe_gate_up_w.view(_0.sizes()).view(at::kComplexDouble), moe_gate_up_s}, {}).view({samples, 9, 2, _0.size(1) / 2});
  auto yb = antares::ops::call(fn2, {xb.view({samples, 9, -1, 16}).view(torch::kInt32), expert_weight, expert_ids, moe_down_w.view(at::kComplexDouble), moe_down_w.view(_1.sizes()).view(at::kComplexDouble), moe_down_s}, {}).view({x.size(0), x.size(1), _1.size(1)});
  return yb;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> warp_multi_head_latent_rope_bf16_v2(
  const torch::Tensor &qkv_act,
  const torch::Tensor &cos_sin,
  const torch::Tensor &positions,
  const torch::Tensor &q_a_norm,
  const torch::Tensor &kv_a_norm,
  const torch::Tensor &q_b_proj,
  const torch::Tensor &k_b_proj,
  int64_t n_local_heads
) {
  auto x = qkv_act;
  CHECK_CUDA(x);
  CHECK_EQ(x.dtype(), torch::kBFloat16);
  CHECK_EQ(x.dim(), 3);
  CHECK_EQ(x.size(-1), 2112);
  CHECK_EQ(cos_sin.dtype(), torch::kInt64);
  CHECK_EQ(positions.dtype(), torch::kInt64);

  int batch = qkv_act.size(0), seqlen = qkv_act.size(1);
  int samples = batch * seqlen;

  auto q = warp_rmsnorm_bf16(x, q_a_norm, 1e-6f);
  auto v_output = warp_rmsnorm_bf16(x, kv_a_norm, 1e-6f, 1536); // [B, S, 512]
  auto k_output = antares::ops::call("rope_kt_bf16", {v_output.view({-1, 8, 64}).view(torch::kInt32), cos_sin, x.view({-1, 33, 64}).view(torch::kInt32), positions}, {}).view(torch::kBFloat16).view({batch, seqlen, 576});

  auto &w_q_b_proj = q_b_proj;
  CHECK_EQ(w_q_b_proj.dtype(), torch::kBFloat16);
  CHECK_EQ(w_q_b_proj.dim(), 2);
  CHECK_EQ(k_b_proj.dtype(), torch::kBFloat16);
  CHECK_EQ(k_b_proj.dim(), 3);
  CHECK_CONTIGUOUS(k_b_proj.transpose(1, 2));

  auto q_output = torch::empty({batch, seqlen, n_local_heads, 512 + 64}, torch::TensorOptions().dtype(q.dtype()).device(q.device()));
  torch::Tensor qh = (IS_NVIDIA_GPU || samples >= 4) ? torch::matmul(q, w_q_b_proj.t()).view({samples, n_local_heads, -1}) : \
    antares::ops::call("rope_gmv_bf16", {q.view({samples, -1}).view(torch::kInt32), w_q_b_proj.view(torch::kInt32)}, {}).view({samples, n_local_heads, -1}); // (BS, 1536) @ (3072, 1536)
  auto buffer = q_output.flatten(0, 1).transpose(0, 1).narrow(-1, 0, 512);
  torch::matmul_out(buffer, qh.transpose(0, 1).narrow(-1, 0, 128), k_b_proj);

  antares::ops::call("rope_qt_bf16_put", {cos_sin, qh.view({qh.size(0), -1, 3, 64}).view(torch::kInt32), positions, q_output.view({qh.size(0), -1, 9, 64}).view(torch::kInt32)}, {});
  return {q_output, k_output, v_output};
}

torch::Tensor warp_gemm_nt_bf16xfp8_block_scal(const torch::Tensor &x, const torch::Tensor &w, const torch::Tensor &scal, int64_t policy = 0) {
  CHECK_CUDA(x);
  CHECK_EQ(x.dim(), 3);
  CHECK_EQ(x.dtype(), torch::kBFloat16);
  CHECK_EQ(w.dim(), 2);

  int samples = x.size(0) * x.size(1);
  if (w.dtype() == torch::kBFloat16)
    return torch::matmul(x.view({samples, x.size(2)}), w.t()).view({x.size(0), x.size(1), w.size(0)});

  if (scal.dim() == 1) {
    CHECK_EQ(w.size(0), scal.size(0));
    return antares::ops::call("gemv_nt_bf16xfp8_row", {x.view({samples, x.size(2)}).view(torch::kInt32), w.view(torch::kInt16), scal}, {}).view({x.size(0), x.size(1), w.size(0)});
  }

  CHECK_EQ(scal.dim(), 2);
  if (samples < 4)
    return antares::ops::call("gemv_nt_bf16xfp8_block", {x.view({samples, x.size(2)}).view(torch::kInt32), w.view(torch::kInt16), scal}, {}).view({x.size(0), x.size(1), w.size(0)});

  torch::Tensor w_ = w;

  if (policy == 0) {
    static std::unordered_map<void*, torch::Tensor> cached;

    auto dptr = scal.data_ptr();
    auto it = cached.find(dptr);
    if (it == cached.end()) {
      cached[dptr] = antares::ops::call("to_bfloat16_3d", {w.unsqueeze(0), scal.unsqueeze(0)}, {}).squeeze(0);
      it = cached.find(dptr);
    }
    w_ = it->second;
  } else {
    w_ = antares::ops::call("to_bfloat16_3d", {w.unsqueeze(0), scal.unsqueeze(0)}, {}).squeeze(0);
  }
  return torch::matmul(x.view({samples, x.size(2)}), w_.t()).view({x.size(0), x.size(1), w.size(0)});
}

}

#endif


TORCH_LIBRARY(tutel_ops, m) {
  m.def("cumsum", warp_cumsum);
  m.def("sparse_bmm_infer", warp_sparse_bmm_infer);

#if defined(USE_NCCL)
  m.def("uncached_empty_ex", uncached_empty_ex);
  m.def("all_gather_native", all_gather_native);

  m.def("nccl_bcast", warp_nccl_bcast);
  m.def("nccl_all_reduce", &warp_nccl_all_reduce);
  m.def("x_add_allreduce_y_bf16", warp_x_add_allreduce_y_bf16);

  m.def("gate_gemm_out_bf16", warp_gate_gemm_out_bf16);
  m.def("intra_add_allreduce_bf16", warp_intra_add_allreduce_bf16);
  m.def("gemm_nt_bf16xfp8_block_scal_out", warp_gemm_nt_bf16xfp8_block_scal_out);
  m.def("deepseek_custom_mla_bf16", warp_deepseek_custom_mla_bf16);
  m.def("multi_head_latent_rope_bf16_v3", warp_multi_head_latent_rope_bf16_v3);
  m.def("glu_expert_bf16xf8_block_scal", warp_glu_expert_bf16xf8_block_scal);
  m.def("glu_expert_bf16xf4_group_scal", warp_glu_expert_bf16xf4_group_scal);

  m.def("qwen3_moe_scaled_topk", warp_qwen3_moe_top_8_static);
  m.def("qwen3_norm_rotary_kvcache2_bf16", warp_qwen3_norm_rotary_kvcache2_bf16);
  m.def("kimi_moe_sigmoid_scaled_topk", warp_kimi_sigmoid_top_8_static_v2);
  m.def("deepseek_moe_sigmoid_scaled_topk", warp_deepseek_sigmoid_top_8_static_v2);
  m.def("rmsnorm_bf16", warp_rmsnorm_bf16);
  m.def("to_bfloat16", warp_to_bfloat16);
  m.def("to_float8_block", warp_to_float8_block);
  m.def("to_float8_per_token", warp_to_float8_per_token);
  m.def("scaled_mask_inv", warp_scaled_mask_inv);
  m.def("topk_token_sort", warp_topk_token_sort);
  m.def("scatter_sample_ids", warp_scatter_sample_ids);
  m.def("copy_to_device", warp_copy_to_device);

  m.def("multi_head_latent_rope_bf16_v2", specialized::warp_multi_head_latent_rope_bf16_v2);
  m.def("glu_expert_bf16xf8_block_scal_16x16_fnuz", specialized::warp_glu_expert_bf16xf8_block_scal_16x16_fnuz);
  m.def("gemm_nt_bf16xfp8_block_scal", specialized::warp_gemm_nt_bf16xfp8_block_scal);
#endif
}
#endif
