// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

// Plugin reference: https://github.com/microsoft/antares

#include <c10/core/Allocator.h>
#include <c10/core/DeviceGuard.h>

#include <torch/csrc/Device.h>
#include <torch/extension.h>
#include <torch/library.h>

#undef AT_ASSERTM
#define AT_ASSERTM(cond, ...)  \
  if (C10_UNLIKELY_OR_CONST(!(cond))) { \
    ::c10::detail::torchInternalAssertFail(__func__, "torch_ext.hpp", \
        static_cast<uint32_t>(__LINE__), #cond " INTERNAL ASSERT FAILED at " C10_STRINGIZE("torch_ext.hpp") ":" C10_STRINGIZE( \
            __LINE__) ", please report a bug to PyTorch. ", c10::str(__VA_ARGS__)); \
  }

#undef TORCH_WARN
#define TORCH_WARN(...) \
  ::c10::warn(::c10::Warning(                                \
      ::c10::UserWarning(),                                  \
      {__func__, "torch_ext.hpp", static_cast<uint32_t>(__LINE__)}, \
      WARNING_MESSAGE_STRING(__VA_ARGS__),                   \
      false));


#if defined(__linux__)
typedef ssize_t llong;
#else
typedef long long llong;
#define ssize_t llong
#endif

#define __RUNTIME_MODE__
#include "backend.hpp"

#include <string>
#include <fstream>

#if !defined(Antares)
#define Antares CUDA
#endif

#define ANTARES_DEV c10::DeviceType::Antares

static c10::Device get_device() { static c10::Device dev = c10::Device(ANTARES_DEV, getenv("LOCAL_RANK") ? std::atoi(getenv("LOCAL_RANK")) : 0); return dev; }
static bool is_verbose = false;

#define DEBUG_FUNC(x)  // printf("[DEBUG] ::%s\n", x)

std::string read_file(const std::string &path) {
  std::ifstream t(path, std::ios::binary);
  if (t.fail())
    return "";
  std::string _((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());
  return _;
}

#define OP_LOADER "OP_LOADER"

std::string get_ops_root() {
  static std::string ops_root;
  if (ops_root.size() == 0) {
    auto root_path = getenv(OP_LOADER);
    AT_ASSERTM(root_path != nullptr && *root_path != 0, OP_LOADER " is not set, please configure this environment variable correctly.");
    ops_root = root_path;
  }
  return ops_root;
}

const char* get_backend_type() {
#if !defined(__HIP_PLATFORM_HCC__) && !defined(__HIP_PLATFORM_AMD__)
  return "c-cuda";
#else
  return "c-rocm";
#endif
}


namespace antares {
namespace ops {

at::Tensor call(const void *key, const std::vector<at::Tensor> &ts, const std::vector<at::Scalar> &ps, bool allow_non_contiguous = false, size_t key_length = 0, int output_index = -1) {
  DEBUG_FUNC((const char*)key);

  struct kernel_object {
    std::vector<void*> symbol;
    std::vector<llong> args;

    int output_exist;
    torch::Dtype output_dtype;
    std::vector<ssize_t> output_shape;

    std::string entry_name, name;
  };

  auto key_id = key;

  static std::unordered_map<decltype(key_id), kernel_object> kernel_dict[16];
  const auto &curr_dev = ts.size() > 0 ? ts[0].device() : get_device();
  auto &kernels = kernel_dict[curr_dev.index()];

  if (is_verbose) {
    std::string fname = key_length ? std::string((const char*)key, (const char*)key + key_length) : std::string((const char*)key);
    TORCH_WARN("AutoRT on executing new function: `", fname, "`");
  }

  auto it = kernels.find(key_id);
  if (it == kernels.end()) {
    // load function module

    auto ssplit = [](const std::string &str, const std::string &sub, bool allow_empty = false) -> std::vector<std::string> {
      std::vector<std::string> ret;
      int it = 0, next;
      while (next = str.find(sub, it), next >= 0) {
        if (next > it || allow_empty)
          ret.push_back(str.substr(it, next - it));
        it = next + sub.size();
      }
      if (it < str.size() || allow_empty)
        ret.push_back(str.substr(it));
      return ret;
    };
    std::string fname = key_length ? std::string((const char*)key, (const char*)key + key_length) : std::string((const char*)key); // key.toStringRef();
    // TORCH_WARN("AutoRT on registering new function: `", fname, "`");

    auto image = read_file(get_ops_root() + "/" + fname + ".mod");
    AT_ASSERTM(!image.empty(), "Failed to load operator module: ", fname.c_str());

    int pos = image.find("||");
    AT_ASSERTM(pos >= 0, " ");
    auto meta = image.substr(0, pos);
    auto data = image.substr(pos + 2);

    // load symbol
    std::unordered_map<std::string, int> th;
    std::unordered_map<std::string, std::string> fn;
    auto metas = ssplit(meta, "|");

    if (metas.size() >= 5) {
      AT_ASSERTM(!strcmp(metas[4].data(), get_backend_type()), "External operator module `", fname.c_str(), "` is not designed for current backend.");
    }

    auto fentry = metas[0];
    for (auto sect: ssplit(metas[1], ";")) {
      auto kvs = ssplit(sect, "=");
      th[kvs[0]] = std::atoll(kvs[1].c_str());
    }
    for (auto sect: ssplit(metas[2], ";")) {
      auto kvs = ssplit(sect, "=");
      fn[kvs[0]] = kvs[1];
    }

    kernels[key_id] = {}, it = kernels.find(key_id);
    it->second.symbol = ab::moduleGetFunction(ab::moduleLoad(data), fentry, th);
    it->second.entry_name = fentry;
    it->second.name = fname;

    // load argument config
    for (int i = 0; ; ++i) {
      auto jt = fn.find("arg_" + std::to_string(i));
      if (jt == fn.end())
        break;
      auto options = ssplit(jt->second, ":", true);
      llong input_id = (options[0] == "") ? ~0 : std::atoll(options[0].c_str());
      llong second_ref = std::atoll(options[1].c_str());
      llong use_fp32 = (options[2] == "float32");

      llong comb = (use_fp32 << 63) | (second_ref << 32) | ((unsigned int)input_id);
      it->second.args.push_back(comb);
    }

    // load output config
    auto o_type = ssplit(fn["o_type"], ":", true);
    if (o_type[0] == "infer") {
      it->second.output_exist = -1;

      static std::unordered_map<std::string, decltype(torch::kInt8)> key_to_dtype = {
        {"int8", torch::kInt8}, {"int16", torch::kInt16}, {"int32", torch::kInt32}, {"int64", torch::kInt64},
        {"bfloat8", at::kFloat8_e5m2}, {"float8", at::kFloat8_e4m3fn}, {"bfloat16", torch::kBFloat16}, {"float16", torch::kFloat16}, {"float32", torch::kFloat32}, {"float64", torch::kFloat64},
        {"float2x8", torch::kInt16}, {"bfloat2x16", torch::kInt32}, {"float2x16", torch::kInt32}, {"float2x32", torch::kInt64},
      };

      auto dtype_it = key_to_dtype.find(o_type[1]);
      if (dtype_it != key_to_dtype.end())
        it->second.output_dtype = dtype_it->second;
      else
        it->second.output_dtype = at::kComplexDouble;

      for (auto dim: ssplit(o_type[2], ",")) {
        if (dim[0] == '#')
          it->second.output_shape.push_back(~std::atoll(dim.c_str() + 1));
        else
          it->second.output_shape.push_back(std::atoll(dim.c_str()));
      }
    } else {
      AT_ASSERTM(o_type[0] == "exist", "`o_type` is not recognized: ", o_type);
      it->second.output_exist = std::atoll(o_type[1].c_str());
    }
  }

  auto &prop = it->second;

  std::vector<void*> krnl_args;
  for (int i = 0; i < ts.size(); ++i) {
#if !defined(__aarch64__)
    if (ts[i].device().type() != ANTARES_DEV) {
      std::string error_msg = "\nThe " + std::to_string(i + 1) + "-th argument of `antares.ops." + prop.name + "(...)` is not a CUDA tensor.";
      AT_ASSERTM(0, error_msg);
    }
#endif
    if (!allow_non_contiguous)
      AT_ASSERTM(ts[i].is_contiguous(), "Not contiguous tensor for custom kernel");
    krnl_args.push_back((void*)ts[i].data_ptr());
  }

  int output_exist = prop.output_exist;
  if (output_index >= 0)
    output_exist = output_index;
  if (output_exist == -1)
    krnl_args.push_back(nullptr); // placeholder for output

  size_t param_offset = krnl_args.size();

  // construct argument values
  for (auto it: prop.args) {
    llong use_fp32 = (it & (1LL << 63));
    it = (it & ~(1LL << 63));
    auto *ids = (unsigned int*)&it;
    if (ids[0] != ~0)
      krnl_args.push_back((void*)ts[ids[0]].size(ids[1]));
    else if (!use_fp32)
      krnl_args.push_back(*(void**)ps[ids[1]].data_ptr());
    else {
      float fp32val[2];
      fp32val[0] = (float)*(double*)ps[ids[1]].data_ptr();
      krnl_args.push_back(*(void**)fp32val);
    }
  }
  krnl_args.push_back(nullptr);

  std::vector<ssize_t> shape = prop.output_shape;
  for (int i = 0; i < shape.size(); ++i)
    if (shape[i] < 0)
      shape[i] = (ssize_t)krnl_args[param_offset + (~shape[i])];

  torch::Tensor output;
  if (output_exist == -1) {
    output = torch::empty(shape, torch::TensorOptions().dtype(prop.output_dtype).device(curr_dev));
    krnl_args[param_offset - 1] = (void*)output.data_ptr();
  } else
    output = ts[output_exist];

  ab::launchKernel(prop.symbol, krnl_args, nullptr);
  return output;
}

} // namespace ops
} // namespace antares
