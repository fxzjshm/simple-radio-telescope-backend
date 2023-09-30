/******************************************************************************* 
 * Copyright (c) 2023 fxzjshm
 * This software is licensed under Mulan PubL v2.
 * You can use this software according to the terms and conditions of the Mulan PubL v2.
 * You may obtain a copy of Mulan PubL v2 at:
 *          http://license.coscl.org.cn/MulanPubL-2.0
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PubL v2 for more details.
 ******************************************************************************/

#pragma once
#ifndef __SRTB_VKFFT_WRAPPER__
#define __SRTB_VKFFT_WRAPPER__

#include <concepts>
#include <mutex>

#include "srtb/fft/fft_wrapper.hpp"
#include "srtb/global_variables.hpp"

#define SRTB_VKFFT_BACKEND_VULKAN 0
#define SRTB_VKFFT_BACKEND_CUDA 1
#define SRTB_VKFFT_BACKEND_ROCM 2
#define SRTB_VKFFT_BACKEND_OPENCL 3
#define SRTB_VKFFT_BACKEND_LEVEL_ZERO 4
#define SRTB_VKFFT_BACKEND_METAL 5

#define SRTB_VKFFT_FORWARD -1
#define SRTB_VKFFT_BACKWARD 1

#ifndef VKFFT_BACKEND
#if SRTB_ENABLE_OPENCL_INTEROP
#define VKFFT_BACKEND SRTB_VKFFT_BACKEND_OPENCL
#elif SRTB_ENABLE_ROCM_INTEROP
#define VKFFT_BACKEND SRTB_VKFFT_BACKEND_ROCM
#elif SRTB_ENABLE_CUDA_INTEROP
#define VKFFT_BACKEND SRTB_VKFFT_BACKEND_CUDA
#endif
#endif
#include "vkFFT.h"

#define SRTB_CHECK_VKFFT_WRAPPER(expr, expected)                      \
  SRTB_CHECK(expr, expected,                                          \
             throw std::runtime_error(                                \
                 std::string("[vkfft_wrapper] " #expr " returned ") + \
                 std::to_string(ret)););

#define SRTB_CHECK_VKFFT(expr) SRTB_CHECK_VKFFT_WRAPPER(expr, VKFFT_SUCCESS)

namespace srtb {
namespace fft {

inline std::mutex vkfft_mutex;

inline constexpr auto vkfft_get_backend(int vkfft_backend) -> sycl::backend {
  switch (vkfft_backend) {
    case SRTB_VKFFT_BACKEND_VULKAN:
      throw std::runtime_error("[vkfft_wrapper] no vulkan support now");
    case SRTB_VKFFT_BACKEND_CUDA:
      return srtb::backend::cuda;
    case SRTB_VKFFT_BACKEND_ROCM:
      return srtb::backend::rocm;
    case SRTB_VKFFT_BACKEND_OPENCL:
      return srtb::backend::opencl;
    case SRTB_VKFFT_BACKEND_LEVEL_ZERO:
      throw std::runtime_error("[vkfft_wrapper] no level zero support now");
    case SRTB_VKFFT_BACKEND_METAL:
      throw std::runtime_error("[vkfft_wrapper] no metal support now");
    default:
      throw std::runtime_error("[vkfft_wrapper] no support now");
  }
}

inline namespace detail {

template <typename T, sycl::backend backend>
struct vkfft_trait;

#if (VKFFT_BACKEND == SRTB_VKFFT_BACKEND_OPENCL)
template <typename T>
struct vkfft_trait<T, srtb::backend::opencl> {
  using device_t = cl_device_id;
  using context_t = cl_context;
  using stream_t = cl_command_queue;
  using buffer_t = cl_mem;

  template <typename... Args>
  static inline decltype(auto) StreamSynchronize(Args&&... args) {
    return clFinish(std::forward<Args>(args)...);
  }

  template <typename U>
  static inline auto get_buffer(context_t context, U* ptr, size_t count)
      -> buffer_t {
    cl_int err;
    cl_mem buffer = clCreateBuffer(context, CL_MEM_USE_HOST_PTR,
                                   sizeof(T) * count, ptr, &err);
    if (err != CL_SUCCESS) {
      throw std::runtime_error(
          "[vkfft_wrapper] SVM -> cl_buffer failed with error " +
          std::to_string(err));
    }
    return buffer;
  }
};
#endif  // (VKFFT_BACKEND == SRTB_VKFFT_BACKEND_OPENCL)

#if (VKFFT_BACKEND == SRTB_VKFFT_BACKEND_ROCM)
template <typename T>
struct vkfft_trait<T, srtb::backend::rocm> {
  using device_t = hipDevice_t;
  using context_t = hipCtx_t;
  using stream_t = hipStream_t;
  using buffer_t = void*;

  template <typename... Args>
  static inline decltype(auto) StreamSynchronize(Args&&... args) {
    return hipStreamSynchronize(std::forward<Args>(args)...);
  }

  template <typename U>
  static inline auto get_buffer([[maybe_unused]] context_t context, U* ptr,
                                [[maybe_unused]] size_t count) -> buffer_t {
    return ptr;
  }
};
#endif  // (VKFFT_BACKEND == SRTB_VKFFT_BACKEND_ROCM)

#if (VKFFT_BACKEND == SRTB_VKFFT_BACKEND_CUDA)
template <typename T>
struct vkfft_trait<T, srtb::backend::cuda> {
  using device_t = CUdevice;
  using context_t = CUcontext;
  using stream_t = cudaStream_t;
  using buffer_t = void*;

  template <typename... Args>
  static inline decltype(auto) StreamSynchronize(Args&&... args) {
    return cudaStreamSynchronize(std::forward<Args>(args)...);
  }

  template <typename U>
  static inline auto get_buffer([[maybe_unused]] context_t context, U* ptr,
                                [[maybe_unused]] size_t count) -> buffer_t {
    return ptr;
  }
};
#endif  // (VKFFT_BACKEND == SRTB_VKFFT_BACKEND_CUDA)

}  // namespace detail

/** @brief common wrapper for vkFFT. */
template <srtb::fft::type fft_type, std::floating_point T,
          typename C = srtb::complex<T>,
          sycl::backend backend_ = vkfft_get_backend(VKFFT_BACKEND)>
class vkfft_1d_wrapper
    : public fft_wrapper<vkfft_1d_wrapper<fft_type, T, C, backend_>, fft_type,
                         T, C> {
 public:
  static constexpr sycl::backend backend = backend_;
  using trait = vkfft_trait<T, backend>;
  using super_class =
      fft_wrapper<vkfft_1d_wrapper<fft_type, T, C, backend>, fft_type, T, C>;
  using device_t = typename trait::device_t;
  using context_t = typename trait::context_t;
  using stream_t = typename trait::stream_t;
  using buffer_t = typename trait::buffer_t;
  friend super_class;

 protected:
  device_t device;
  context_t context;
  stream_t stream;
  uint64_t idist, odist;
  VkFFTConfiguration configuration;
  VkFFTApplication app;

 public:
  vkfft_1d_wrapper(size_t n, size_t batch_size, sycl::queue& queue)
      : super_class{n, batch_size, queue} {}

 protected:
  void create_impl(size_t n, size_t batch_size, sycl::queue& q) {
    std::lock_guard lock{srtb::fft::vkfft_mutex};
    // should be equivalent to this
    /*
    plan = fftw_plan_dft_r2c_1d(static_cast<int>(n), tmp_in.get(),
                               reinterpret_cast<fftw_complex*>(tmp_out.get()),
                               FFTW_PATIENT |  FFTW_DESTROY_INPUT);
    */

    device = sycl::get_native<backend>(q.get_device());
#if (VKFFT_BACKEND == SRTB_VKFFT_BACKEND_OPENCL)
    context = sycl::get_native<backend>(q.get_context());
#endif
    stream = srtb::backend::get_native_queue<backend, stream_t>(q);
    const uint64_t n_ = n;

    configuration.FFTdim = 1;
    configuration.size[0] = n_;
    configuration.numberBatches = batch_size;
    configuration.device = &device;
#if (VKFFT_BACKEND == SRTB_VKFFT_BACKEND_OPENCL)
    configuration.context = &context;
#endif
    if constexpr (fft_type == srtb::fft::type::R2C_1D ||
                  fft_type == srtb::fft::type::C2R_1D) {
      configuration.performR2C = 1;
    }
    if constexpr (std::is_same<T, double>::value) {
      configuration.doublePrecision = 1;
    }
    if constexpr (fft_type == srtb::fft::type::R2C_1D ||
                  fft_type == srtb::fft::type::C2C_1D_FORWARD) {
      configuration.makeForwardPlanOnly = 1;
    }
    if constexpr (fft_type == srtb::fft::type::C2R_1D ||
                  fft_type == srtb::fft::type::C2C_1D_BACKWARD) {
      configuration.makeInversePlanOnly = 1;
    }
    if constexpr (fft_type == srtb::fft::type::C2C_1D_FORWARD ||
                  fft_type == srtb::fft::type::C2C_1D_BACKWARD) {
      const long long int n_complex = n_;
      idist = odist = n_complex;
    } else if constexpr (fft_type == srtb::fft::type::R2C_1D) {
      const long long int n_real = n_;
      const long long int n_complex = n_real / 2 + 1;
      idist = n_real;
      odist = n_complex;
    } else if constexpr (fft_type == srtb::fft::type::C2R_1D) {
      const long long int n_real = n_;
      const long long int n_complex = n_real / 2 + 1;
      idist = n_complex;
      odist = n_real;
    } else {
      throw std::runtime_error("[vkfft_like_wrapper] create_impl: TODO");
    }

    SRTB_CHECK_VKFFT(initializeVkFFT(&app, configuration));

    SRTB_LOGI << " [vkfft_wrapper] "
              << "plan finished." << srtb::endl;
  }

  void destroy_impl() {
    std::lock_guard lock{srtb::fft::vkfft_mutex};
    deleteVkFFT(&app);
  }

  // enable only if fft_type_ == srtb::fft::type::R2C_1D
  template <typename..., srtb::fft::type fft_type_ = fft_type,
            typename std::enable_if<(fft_type_ == srtb::fft::type::R2C_1D),
                                    int>::type = 0>
  void process_impl(T* in, C* out) {
    VkFFTLaunchParams launch_params = get_launch_params(in, out);
    constexpr auto direction = SRTB_VKFFT_FORWARD;
    SRTB_CHECK_VKFFT(VkFFTAppend(&app, direction, &launch_params));
    flush();
  }

  template <
      typename..., srtb::fft::type fft_type_ = fft_type,
      typename std::enable_if<(fft_type_ == srtb::fft::type::C2C_1D_FORWARD ||
                               fft_type_ == srtb::fft::type::C2C_1D_BACKWARD),
                              int>::type = 0>
  void process_impl(C* in, C* out) {
    VkFFTLaunchParams launch_params = get_launch_params(in, out);
    constexpr auto direction = (fft_type == srtb::fft::type::C2C_1D_BACKWARD)
                                   ? SRTB_VKFFT_BACKWARD
                                   : SRTB_VKFFT_FORWARD;
    SRTB_CHECK_VKFFT(VkFFTAppend(&app, direction, &launch_params));
    flush();
  }

  template <typename..., srtb::fft::type fft_type_ = fft_type,
            typename std::enable_if<(fft_type_ == srtb::fft::type::C2R_1D),
                                    int>::type = 0>
  void process_impl(C* in, T* out) {
    VkFFTLaunchParams launch_params = get_launch_params(in, out);
    constexpr auto direction = SRTB_VKFFT_BACKWARD;
    SRTB_CHECK_VKFFT(VkFFTAppend(&app, direction, &launch_params));
    flush();
  }

  bool has_inited_impl() {
    // invalid plan causes segmentation fault, so not using plan to check here.
    return true;
  }

  template <typename U, typename V>
  auto get_launch_params(U* in, V* out) -> VkFFTLaunchParams {
    VkFFTLaunchParams launch_params;
    buffer_t out_buffer = trait::get_buffer(context, out, odist);
    launch_params.buffer = &out_buffer;
    if (in != out) {
      buffer_t in_buffer = trait::get_buffer(context, in, idist);
      launch_params.inputBuffer = &in_buffer;
    }
    return launch_params;
  }

  void flush() {
    SRTB_CHECK_VKFFT_API(trait::StreamSynchronize((*this).stream));
  }
};

}  // namespace fft
}  // namespace srtb

#endif  //  __SRTB_VKFFT_WRAPPER__
