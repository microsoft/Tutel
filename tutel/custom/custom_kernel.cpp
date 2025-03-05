// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <torch/extension.h>

#if defined(USE_GPU)
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda.h>
#include <nvrtc.h>
#include <ATen/cuda/CUDAContext.h>
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
    CHECK_EQ(-1, execl(entry.c_str(), entry.c_str(), code_path, "-o", fatbin_path.c_str(), "--genco", "-O4", "-w" , ("--amdgpu-target=" + arch).c_str(), (char *)NULL));
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

static ncclComm_t g_nccl_comm, shared_nccl_comm;
static std::vector<at::cuda::CUDAEvent> g_cuda_events;
static int g_world_size = 0, shared_world_size = 0;
static int g_world_rank = 0, shared_world_rank = 0;
static int g_local_size = 0;
static int g_local_rank = 0;
static int __dtype_size[256];

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

  __dtype_size[(int)torch::kFloat64] = 8;
  __dtype_size[(int)torch::kInt64] = 8;
  __dtype_size[(int)torch::kFloat32] = 4;
  __dtype_size[(int)torch::kInt32] = 4;
  __dtype_size[(int)torch::kFloat16] = 2;
  __dtype_size[(int)torch::kInt16] = 2;
  __dtype_size[(int)torch::kInt8] = 1;
  __dtype_size[(int)torch::kUInt8] = 1;
  __dtype_size[(int)torch::kBool] = 1;
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

void warp_bcast_index(const torch::Tensor &t, int64_t root) {
  CHECK_CUDA(t);
  AT_ASSERTM(shared_world_size > 0, "Failed to initialize Shared NCCL");
  auto stream = at::cuda::getCurrentCUDAStream();
  ncclBcast(t.data_ptr(), t.numel(), t.dtype() == torch::kInt64 ? ncclInt64 : ncclBfloat16, root, (ncclComm_t)shared_nccl_comm, stream);
}

static torch::Tensor warp_x_add_allreduce_y_f16(const torch::Tensor &x, const torch::Tensor &t) {
  AT_ASSERTM(shared_world_size > 0, "Failed to initialize Shared NCCL");
  auto stream = at::cuda::getCurrentCUDAStream();
  ncclAllReduce(t.data_ptr(), t.data_ptr(), t.numel(), ncclBfloat16, ncclSum, (ncclComm_t)shared_nccl_comm, stream);
  return x + t;
}

static void batch_all_to_all_v(const std::vector<torch::Tensor> &ins, const std::vector<torch::Tensor> &outs, const torch::Tensor &in_sizes_, const torch::Tensor &out_sizes_) {
  AT_ASSERTM(shared_world_size > 0, "Failed to initialize Shared NCCL");

  auto in_sizes_cpu = in_sizes_.to(torch::kCPU).to(torch::kInt64);
  auto out_sizes_cpu = out_sizes_.to(torch::kCPU).to(torch::kInt64);
  auto* in_sizes = (unsigned long long*)in_sizes_cpu.data_ptr();
  auto* out_sizes = (unsigned long long*)out_sizes_cpu.data_ptr();
  auto stream = at::cuda::getCurrentCUDAStream();

  for (int k = 0; k < ins.size(); ++k) {
    ncclGroupStart();
    auto* in_buff = ins[k].data_ptr();
    auto* out_buff = outs[k].data_ptr();
    auto dtype = ins[k].dtype();
    int size = __dtype_size[*(unsigned short*)&dtype];
    AT_ASSERTM(size > 0, "Data type of input tensors for batch_all_to_all_v are not recognized.");
    AT_ASSERTM(k == 0 || ins[0].numel() == ins[k].numel(), "Tensor instances within batch_all_to_all_v are supposed to share same length.");

    unsigned long long in_offset = 0, out_offset = 0;
    for (int i = 0; i < shared_world_size; ++i) {
      ncclSend((char*)in_buff + in_offset, in_sizes[i] * size, ncclInt8, i, (ncclComm_t)shared_nccl_comm, stream);
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
  auto stream = at::cuda::getCurrentCUDAStream();

  for (int k = 0; k < ins.size(); ++k) {
    ncclGroupStart();
    auto* in_buff = ins[k].data_ptr();
    auto* out_buff = outs[k].data_ptr();
    auto dtype = ins[k].dtype();
    int size = __dtype_size[*(unsigned short*)&dtype];
    AT_ASSERTM(size > 0, "Data type of input tensors for batch_all_gather_v are not recognized.");
    AT_ASSERTM(k == 0 || ins[0].numel() == ins[k].numel(), "Tensor instances within batch_all_gather_v are supposed to share same length.");

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

void show(const std::vector<torch::Tensor> &xs) {
  if (shared_world_rank != 0)
    return;
  puts("=======================");
  for (auto &x: xs) {
    printf("[");
    for (int i = 0; i < x.dim(); ++i) printf("%d, ", x.size(i));
    printf("] data = ");
    auto x_ = x.to(torch::kFloat32).to(torch::kCPU);
    for (int i = 0; i < 5; ++i) printf("%g, ", x_.data_ptr<float>()[i]);
    printf("..");
    for (int i = 0; i < 5; ++i) printf("%g, ", x_.data_ptr<float>()[x.numel() - 5 + i]);
    puts("");
  }
}

torch::Tensor warp_gemm_nt_bf16xfp8_block_scal(const torch::Tensor &x, const torch::Tensor &w, const torch::Tensor &scal, int64_t policy = 0) {
  CHECK_CUDA(x);
  CHECK_EQ(x.dim(), 3);
  CHECK_EQ(w.dim(), 2);
  CHECK_EQ(x.dtype(), torch::kBFloat16);

  int samples = x.size(0) * x.size(1);

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

torch::Tensor warp_rmsnorm_bf16(const torch::Tensor &x, const torch::Tensor &rms_w, double eps, const ::std::optional<torch::Tensor> &out = ::std::nullopt) {
  CHECK_CUDA(x);
  CHECK_EQ(x.dtype(), torch::kBFloat16);
  if (out.has_value()) {
    antares::ops::call("rmsnorm_bf16", {x.view({-1, x.size(-1)}), rms_w, out.value().view({-1, x.size(-1)})}, {eps}, false, 0, 2);
    return out.value();
  }
  return antares::ops::call("rmsnorm_bf16", {x.view({-1, x.size(-1)}), rms_w}, {eps}).view(x.sizes());
}

std::tuple<torch::Tensor, torch::Tensor> warp_deepseek_sigmoid_top_8_static(
     const torch::Tensor &logits_bf16,
     const torch::Tensor &moe_gate_b_bf16,
     const ::std::optional<torch::Tensor> &top_v_out_,
     const ::std::optional<torch::Tensor> &top_k_out_) {
  CHECK_CUDA(logits_bf16);
  CHECK_EQ(logits_bf16.dtype(), torch::kBFloat16);
  CHECK_EQ(moe_gate_b_bf16.dtype(), torch::kBFloat16);

  int n_experts = logits_bf16.size(-1);
  AT_ASSERTM(n_experts == 256, "Deepseek R1 requires 256 experts for gating.");
  int samples = logits_bf16.numel() / n_experts;

  auto device = logits_bf16.device();
  auto top_v_out = top_v_out_.has_value() ? top_v_out_.value().view({samples, -1}) : torch::empty({samples, 8}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
  auto top_k_out = top_k_out_.has_value() ? top_k_out_.value().view({samples, -1}) : torch::empty({samples, 8}, torch::TensorOptions().dtype(torch::kInt32).device(device));
  AT_ASSERTM(top_v_out.dtype() == torch::kFloat32 && top_k_out.dtype() == torch::kInt32, "Output tensor space should be float32 for top_scores and int32 for top_ids.");

  antares::ops::call("deepseek_r1_sigmoid_top_k_f32", {logits_bf16.view({samples, n_experts}), moe_gate_b_bf16, top_v_out, top_k_out}, {}, false, 0, 3);
  return {top_v_out, top_k_out};
}

void warp_deepseek_r1_static_gating_f16(
     const torch::Tensor &x,
     const torch::Tensor &gate_moe,
     const torch::Tensor &gate_bias,
     const ::std::optional<torch::Tensor> &top_v_out_,
     const ::std::optional<torch::Tensor> &top_k_out_) {
  CHECK_CUDA(x);
  CHECK_EQ(x.dtype(), torch::kBFloat16);
  CHECK_EQ(gate_moe.dtype(), torch::kBFloat16);
  CHECK_EQ(gate_bias.dtype(), torch::kBFloat16);

  auto logits_bf16 = torch::matmul(x, gate_moe.t());
  warp_deepseek_sigmoid_top_8_static(logits_bf16.view({-1, logits_bf16.size(-1)}), gate_bias, top_v_out_, top_k_out_);
}

namespace {
  int64_t n_local_heads;
  torch::Tensor token_emb, weight_classify;
  torch::Tensor cos_sin;
  torch::Tensor key_cache;
  torch::Tensor val_cache;
  std::vector<torch::Tensor> weight_gate_ups;
  std::vector<torch::Tensor> weight_gate_up_scals;
  std::vector<torch::Tensor> weight_downs;
  std::vector<torch::Tensor> weight_down_scals;
  std::vector<torch::Tensor> moe_gate_up_ws;
  std::vector<torch::Tensor> moe_gate_up_ss;
  std::vector<torch::Tensor> moe_down_ws;
  std::vector<torch::Tensor> moe_down_ss;
  std::vector<torch::Tensor> gate_moes;
  std::vector<torch::Tensor> gate_biases;
  std::vector<torch::Tensor> rms_att_ws;
  std::vector<torch::Tensor> rms_ffn_ws;
  std::vector<torch::Tensor> qkv_a_projs;
  std::vector<torch::Tensor> qkv_a_proj_scals;
  std::vector<torch::Tensor> q_a_norms;
  std::vector<torch::Tensor> kv_a_norms;
  std::vector<torch::Tensor> q_b_projs;
  std::vector<torch::Tensor> q_b_proj_scals;
  std::vector<torch::Tensor> kv_b_projs;
  std::vector<torch::Tensor> kv_b_proj_scals;
  std::vector<torch::Tensor> o_projs;
  std::vector<torch::Tensor> o_proj_scals;
  torch::Tensor shared_exp_id, shared_weights, topk_exp_id, score_weight, rms_end_w;
}


torch::Tensor warp_deepseek_r1_latent_attn_f16(
  const torch::Tensor &data,
  const torch::Tensor &key_cache,
  const torch::Tensor &val_cache,
  const torch::Tensor &qkv_a_proj,
  const torch::Tensor &qkv_a_proj_scal,
  const torch::Tensor &q_a_norm,
  const torch::Tensor &kv_a_norm,
  const torch::Tensor &q_b_proj,
  const torch::Tensor &q_b_proj_scal,
  const torch::Tensor &kv_b_proj,
  const torch::Tensor &kv_b_proj_scal,
  const torch::Tensor &o_proj,
  const torch::Tensor &o_proj_scal,
  int64_t pos
) {
  CHECK_CUDA(data);
  CHECK_EQ(data.dim(), 3);
  auto xb = data;
  int batch = xb.size(0), seqlen = xb.size(1), n_heads = n_local_heads;
  auto qkv = warp_gemm_nt_bf16xfp8_block_scal(xb, qkv_a_proj, qkv_a_proj_scal); // [B, S, *]
  auto q = qkv.narrow(-1, 0, 1536).contiguous(), kv = qkv.narrow(-1, 1536, 512).contiguous(), k_pe = qkv.narrow(-1, 2048, 64).contiguous();
  auto k_pe_out = torch::empty_like(k_pe);
  antares::ops::call("rotary_lookup_bf16", {cos_sin.select(0, 0).select(0, pos), cos_sin.select(0, 1).select(0, pos), k_pe.view({-1, 32, 2}), k_pe_out.view({-1, 2, 32})}, {}, false, 0, 3);

  q = warp_gemm_nt_bf16xfp8_block_scal(warp_rmsnorm_bf16(q, q_a_norm, 1e-6f), q_b_proj, q_b_proj_scal);
  auto query_states = q.view({batch, seqlen, -1, 192});
  auto q_pe = query_states.narrow(-1, 128, 64).contiguous();
  auto q_pe_out = torch::empty_like(q_pe);
  antares::ops::call("rotary_lookup_bf16", {cos_sin.select(0, 0).select(0, pos), cos_sin.select(0, 1).select(0, pos), q_pe.view({-1, 32, 2}), q_pe_out.view({-1, 2, 32})}, {}, false, 0, 3);

  if (val_cache.numel() > 1) {
    kv = warp_gemm_nt_bf16xfp8_block_scal(warp_rmsnorm_bf16(kv, kv_a_norm, 1e-6f), kv_b_proj, kv_b_proj_scal);

    antares::ops::call("cache_fill_bf16", {q_pe_out, k_pe_out, query_states, key_cache.select(0, pos)}, {128}, false, 0, 3);
                                      // [B,S,H,64]  [B,S,64]  [B,S,H,128:]    [B,H,128:]

    antares::ops::call("cache_move_bf16", {kv.view({batch, seqlen, n_heads, 2, 128}), key_cache.narrow(0, pos, seqlen), val_cache.narrow(0, pos, seqlen)}, {}, false, 0, 2);
                                                 // [B,S,H,2,M]                                [S,B,H,:128]                 [S,B,H,:128]

    auto key_states = key_cache.narrow(0, 0, pos + seqlen).view({1, pos + seqlen, batch * n_heads, 192});
    auto value_states = val_cache.narrow(0, 0, pos + seqlen).view({1, pos + seqlen, batch * n_heads, 128});
    query_states = query_states.permute({1, 0, 2, 3}).view({1, seqlen, -1, 192});
    CHECK_EQ(query_states.size(1), 1);

    auto lm = torch::empty({2, batch * n_heads, 64}, torch::TensorOptions().dtype(torch::kBFloat16).device(query_states.device()));
    auto attn_output = antares::ops::call("self_attn_infer_bf16", {query_states.squeeze(0).squeeze(0), key_states.squeeze(0), value_states.squeeze(0), lm}, {0.1352337788608801f});
    xb = torch::matmul(antares::ops::call("self_attn_reduce_bf16", {lm}, {}).unsqueeze(1), attn_output).to(query_states.dtype());

    // xb = std::get<0>(at::native::_scaled_dot_product_attention_math(query_states.permute({0, 2, 1, 3}).to(torch::kBFloat16), key_states.permute({0, 2, 1, 3}).to(torch::kBFloat16), value_states.permute({0, 2, 1, 3}).to(torch::kBFloat16), {}, 0, false, {}, 0.1352337788608801)).permute({0, 2, 1, 3}).to(query_states.dtype());
  } else {
    kv = warp_rmsnorm_bf16(kv, kv_a_norm, 1e-6f); // [B, S, 512]
    key_cache.narrow(0, pos, seqlen).narrow(1, 0, batch) = torch::cat({kv, k_pe_out}, -1).permute({1, 0, 2}); // [S, B, 512 + 64]

    static std::unordered_map<void*, torch::Tensor> wkv_b_;
    auto it = wkv_b_.find(kv_b_proj_scal.data_ptr());
    if (it == wkv_b_.end()) {
      wkv_b_[kv_b_proj_scal.data_ptr()] = antares::ops::call("to_bfloat16_3d", {kv_b_proj.unsqueeze(0), kv_b_proj_scal.unsqueeze(0)}, {}).
        view({n_heads, 2, -1, kv_b_proj.size(-1)}).permute({1, 0, 2, 3}).contiguous(); // 2, H, 128, 512
      it = wkv_b_.find(kv_b_proj_scal.data_ptr());
    }
    auto _0 = it->second.select(0, 0), _1 = it->second.select(0, 1); // H, D(128), C(512)
    // k_pe_out, q_pe_out -- 1, 1, 64 | 1, 1, 16, 64
    auto q_nope = query_states.narrow(-1, 0, 128).contiguous(); // B, S, H, D
    q_nope = at::einsum("bshd,hdc->bshc", {q_nope, _0}).contiguous(); // B, S, H, C(512)

    auto R = key_cache.narrow(0, 0, pos + seqlen); // S2, B, (512 + 64)
    auto scores_ = at::einsum("bshc,tbc->bsht", {torch::cat({q_nope, q_pe_out}, -1), R}) * 0.1352337788608801f;
    _0 = at::einsum("bsht,tbc->bshc", {at::softmax(scores_, -1), R});
    _0 = _0.narrow(-1, 0, 512);

    xb = at::einsum("bshc,hdc->bshd", {_0, _1}).contiguous();
  }
  return warp_gemm_nt_bf16xfp8_block_scal(xb.view({batch, seqlen, -1}), o_proj, o_proj_scal);
}

torch::Tensor warp_glu_expert_f16xf8_block_scal(
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
  CHECK_EQ(expert_weight.dim(), 2);

  if (samples < 4) {
    auto xb = antares::ops::call("gemm_gate_up_silu_bf16xf8_s", {x.view({samples, model_dim}).view(torch::kInt32), expert_ids, moe_gate_up_w.view(torch::kInt16), moe_gate_up_s}, {});
    return antares::ops::call("gemm_down_weight_sum_bf16xf8_s", {xb.view(xb.dtype() == torch::kFloat32 ? torch::kInt64 : torch::kInt32), expert_weight, expert_ids, moe_down_w.view(torch::kInt16), moe_down_s}, {}).view({x.size(0), x.size(1), moe_down_w.size(1)});
  } else if (samples < 32) {
    auto xb = antares::ops::call("gemm_gate_up_silu_bf16xf8_m", {x.view({samples, model_dim}).view(torch::kInt32), expert_ids, moe_gate_up_w.view(torch::kInt16), moe_gate_up_s}, {});
    return antares::ops::call("gemm_down_weight_sum_bf16xf8_m", {xb.view(xb.dtype() == torch::kFloat32 ? torch::kInt64 : torch::kInt32), expert_weight, expert_ids, moe_down_w.view(torch::kInt16), moe_down_s}, {}).view({x.size(0), x.size(1), moe_down_w.size(1)});
  } else {
    auto xb = antares::ops::call("gemm_gate_up_silu_bf16xf8_l", {x.view({samples, model_dim}).view(torch::kInt32), expert_ids, moe_gate_up_w.view(torch::kInt16), moe_gate_up_s}, {});
    return antares::ops::call("gemm_down_weight_sum_bf16xf8_l", {xb.view(xb.dtype() == torch::kFloat32 ? torch::kInt64 : torch::kInt32), expert_weight, expert_ids, moe_down_w.view(torch::kInt16), moe_down_s}, {}).view({x.size(0), x.size(1), moe_down_w.size(1)});
  }
}

void warp_deepseek_r1_prepare_weights(
  int64_t n_local_heads,
  int64_t max_seq_len,
  int64_t batch,
  const torch::Tensor &token_emb,
  const torch::Tensor &weight_classify,
  const torch::Tensor &cos_sin,
  const torch::Tensor &shared_exp_id,
  const torch::Tensor &shared_weights,
  const torch::Tensor &topk_exp_id,
  const torch::Tensor &score_weight,
  const torch::Tensor &rms_end_w,

  const std::vector<torch::Tensor> &rms_att_ws,
  const std::vector<torch::Tensor> &rms_ffn_ws,
  const std::vector<torch::Tensor> &qkv_a_projs,
  const std::vector<torch::Tensor> &qkv_a_proj_scals,
  const std::vector<torch::Tensor> &q_a_norms,
  const std::vector<torch::Tensor> &kv_a_norms,
  const std::vector<torch::Tensor> &q_b_projs,
  const std::vector<torch::Tensor> &q_b_proj_scals,
  const std::vector<torch::Tensor> &kv_b_projs,
  const std::vector<torch::Tensor> &kv_b_proj_scals,
  const std::vector<torch::Tensor> &o_projs,
  const std::vector<torch::Tensor> &o_proj_scals,

  const std::vector<torch::Tensor> &weight_gate_ups,
  const std::vector<torch::Tensor> &weight_gate_up_scals,
  const std::vector<torch::Tensor> &weight_downs,
  const std::vector<torch::Tensor> &weight_down_scals,
  const std::vector<torch::Tensor> &moe_gate_up_ws,
  const std::vector<torch::Tensor> &moe_gate_up_ss,
  const std::vector<torch::Tensor> &moe_down_ws,
  const std::vector<torch::Tensor> &moe_down_ss,
  const std::vector<torch::Tensor> &gate_moes,
  const std::vector<torch::Tensor> &gate_biases
) {
  ::n_local_heads = n_local_heads,
  ::token_emb = token_emb,
  ::weight_classify = weight_classify,
  ::cos_sin = cos_sin;

  int n_layers = o_projs.size();
  if (!getenv("LORA") || std::atoi(getenv("LORA")) == 1) {
    // kv_lora_rank + qk_rope_head_dim
    ::key_cache = torch::zeros({n_layers, max_seq_len, batch, 512 + 64}, torch::TensorOptions().dtype(token_emb.dtype()).device(token_emb.device()));
    ::val_cache = torch::empty({n_layers}, torch::TensorOptions().dtype(token_emb.dtype()).device(token_emb.device()));
  } else {
    // qk_nope_head_dim + qk_rope_head_dim
    ::key_cache = torch::zeros({n_layers, max_seq_len, batch, n_local_heads, 128 + 64}, torch::TensorOptions().dtype(token_emb.dtype()).device(token_emb.device()));
    // v_head_dim
    ::val_cache = torch::zeros({n_layers, max_seq_len, batch, n_local_heads, 128}, torch::TensorOptions().dtype(token_emb.dtype()).device(token_emb.device()));
  }

  ::weight_gate_ups = weight_gate_ups;
  ::weight_gate_ups = weight_gate_ups;
  ::weight_gate_up_scals = weight_gate_up_scals;
  ::weight_downs = weight_downs;
  ::weight_down_scals = weight_down_scals;
  ::moe_gate_up_ws = moe_gate_up_ws;
  ::moe_gate_up_ss = moe_gate_up_ss;
  ::moe_down_ws = moe_down_ws;
  ::moe_down_ss = moe_down_ss;
  ::gate_moes = gate_moes;
  ::gate_biases = gate_biases;
  ::rms_end_w = rms_end_w;
  ::rms_att_ws = rms_att_ws;
  ::rms_ffn_ws = rms_ffn_ws;
  ::qkv_a_projs = qkv_a_projs;
  ::qkv_a_proj_scals = qkv_a_proj_scals;
  ::q_a_norms = q_a_norms;
  ::kv_a_norms = kv_a_norms;
  ::q_b_projs = q_b_projs;
  ::q_b_proj_scals = q_b_proj_scals;
  ::kv_b_projs = kv_b_projs;
  ::kv_b_proj_scals = kv_b_proj_scals;
  ::o_projs = o_projs;
  ::o_proj_scals = o_proj_scals;

  ::shared_exp_id = shared_exp_id;
  ::shared_weights = shared_weights;
  ::topk_exp_id = topk_exp_id;
  ::score_weight = score_weight;
}

torch::Tensor warp_deepseek_r1_forward(
  const torch::Tensor &data,
  int64_t pos
) {
    auto x = data;
    CHECK_CUDA(x);
    CHECK_EQ(x.dim(), 2);

    x = token_emb.index_select(0, x.view({-1})).view({x.size(0), x.size(1), token_emb.size(1)});
    #pragma unroll
    for (int l = 0; l < rms_att_ws.size(); ++l) {
      auto xb = warp_rmsnorm_bf16(x, rms_att_ws[l], 1e-6f);
      xb = warp_deepseek_r1_latent_attn_f16(xb, key_cache[l], val_cache[l], qkv_a_projs[l], qkv_a_proj_scals[l], q_a_norms[l], kv_a_norms[l], q_b_projs[l], q_b_proj_scals[l], kv_b_projs[l], kv_b_proj_scals[l], o_projs[l], o_proj_scals[l], pos);

      x = warp_x_add_allreduce_y_f16(x, xb);
      xb = warp_rmsnorm_bf16(x, rms_ffn_ws[l], 1e-6f);
      if (l < weight_gate_ups.size()) {
        xb = warp_glu_expert_f16xf8_block_scal(xb, shared_exp_id, shared_weights, weight_gate_ups[l], weight_gate_up_scals[l], weight_downs[l], weight_down_scals[l]);
      } else {
        CHECK_EQ(topk_exp_id.dim(), 2);
        auto logits_bf16 = torch::matmul(xb, gate_moes[l - 3].t());
        antares::ops::call("deepseek_r1_sigmoid_top_k_routed_scaled_f32", {logits_bf16.view({-1, logits_bf16.size(-1)}), gate_biases[l - 3], score_weight, topk_exp_id}, {}, false, 0, 3);

        xb = warp_glu_expert_f16xf8_block_scal(xb, topk_exp_id, score_weight, moe_gate_up_ws[l - 3], moe_gate_up_ss[l - 3], moe_down_ws[l - 3], moe_down_ss[l - 3]);
      }
      x = warp_x_add_allreduce_y_f16(x, xb);
    }
    x = warp_rmsnorm_bf16(x, rms_end_w, 1e-6);
    return torch::matmul(x, weight_classify.t());
}


TORCH_LIBRARY(tutel_ops, m) {
  m.def("cumsum", warp_cumsum);
  m.def("sparse_bmm_infer", warp_sparse_bmm_infer);

  m.def("gemm_nt_bf16xfp8_block_scal", warp_gemm_nt_bf16xfp8_block_scal);

  m.def("bcast_index", &warp_bcast_index);
  m.def("x_add_allreduce_y_f16", &warp_x_add_allreduce_y_f16);
  m.def("deepseek_r1_static_gating_f16", warp_deepseek_r1_static_gating_f16);
  m.def("deepseek_r1_attn_f16xf8_block_scal", warp_deepseek_r1_latent_attn_f16);
  m.def("deepseek_r1_prepare_weights", warp_deepseek_r1_prepare_weights);
  m.def("deepseek_r1_forward", warp_deepseek_r1_forward);

  m.def("deepseek_sigmoid_top_8_static", warp_deepseek_sigmoid_top_8_static);
  m.def("rmsnorm_bf16", warp_rmsnorm_bf16);
  m.def("glu_expert_bf16xf8_block_scal", warp_glu_expert_f16xf8_block_scal);
}
#endif
