/******************************************************************************* 
 * Copyright (c) 2022 fxzjshm
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
#ifndef __SRTB_CUFFT_WRAPPER__
#define __SRTB_CUFFT_WRAPPER__

#include <cuda_runtime.h>
#include <cufft.h>

#include <concepts>

#include "srtb/fft/fft_wrapper.hpp"
#include "srtb/global_variables.hpp"

#define SRTB_CHECK_CUFFT_WRAPPER(expr, expected)                      \
  SRTB_CHECK(expr, expected,                                          \
             throw std::runtime_error(                                \
                 std::string("[cufft_wrapper] " #expr " returned ") + \
                 std::to_string(ret)););

#define SRTB_CHECK_CUFFT(expr) SRTB_CHECK_CUFFT_WRAPPER(expr, CUFFT_SUCCESS)

#define SRTB_CHECK_CUDA(expr) SRTB_CHECK_CUFFT_WRAPPER(expr, cudaSuccess)

namespace srtb {
namespace fft {

template <std::floating_point T>
constexpr inline cufftType get_cufft_type(srtb::fft::type fft_type) {
  if constexpr (std::is_same_v<T, cufftReal>) {
    switch (fft_type) {
      case srtb::fft::type::C2C_1D_FORWARD:
      case srtb::fft::type::C2C_1D_BACKWARD:
        return CUFFT_C2C;
      case srtb::fft::type::R2C_1D:
        return CUFFT_R2C;
      case srtb::fft::type::C2R_1D:
        return CUFFT_C2R;
    }
  } else if constexpr (std::is_same_v<T, cufftDoubleReal>) {
    switch (fft_type) {
      case srtb::fft::type::C2C_1D_FORWARD:
      case srtb::fft::type::C2C_1D_BACKWARD:
        return CUFFT_Z2Z;
      case srtb::fft::type::R2C_1D:
        return CUFFT_D2Z;
      case srtb::fft::type::C2R_1D:
        return CUFFT_Z2D;
    }
  }
}

// ref: https://docs.nvidia.com/cuda/cufft/index.html
template <srtb::fft::type fft_type, std::floating_point T,
          typename Complex = srtb::complex<T> >
class cufft_1d_wrapper
    : public fft_wrapper<cufft_1d_wrapper, fft_type, T, Complex> {
 public:
  using super_class = fft_wrapper<cufft_1d_wrapper, fft_type, T, Complex>;
  friend super_class;
  static_assert((std::is_same_v<T, cufftReal> &&
                 sizeof(Complex) == sizeof(cufftComplex)) ||
                (std::is_same_v<T, cufftDoubleReal> &&
                 sizeof(Complex) == sizeof(cufftDoubleComplex)));

 protected:
  cufftHandle plan;
  size_t workSize;
  cudaStream_t stream;

 public:
  cufft_1d_wrapper(size_t n, size_t batch_size, sycl::queue& queue)
      : super_class{n, batch_size, queue} {}

 protected:
  void create_impl(size_t n, size_t batch_size, sycl::queue& q) {
    // should be equivalent to this
    /*
    plan = fftw_plan_dft_r2c_1d(static_cast<int>(n), tmp_in.get(),
                               reinterpret_cast<fftw_complex*>(tmp_out.get()),
                               FFTW_PATIENT |  FFTW_DESTROY_INPUT);
    */

    // pending: https://github.com/intel/llvm/pull/6649
    // therefore, using non-default device for FFT is not supported on intel/llvm
#ifndef SYCL_IMPLEMENTATION_ONEAPI
    auto device = q.get_device();
    auto native_device = sycl::get_native<srtb::backend::cuda>(device);
    SRTB_CHECK_CUDA(cudaSetDevice(native_device));
#endif  // SYCL_IMPLEMENTATION_ONEAPI

    SRTB_CHECK_CUFFT(cufftCreate(&plan));
    constexpr cufftType cufft_type = get_cufft_type<T>(fft_type);

    long long int n_ = static_cast<long long int>(n);
    long long int idist, odist;
    long long int inembed[1] = {1}, onembed[1] = {1};  // should have no effect
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
      throw std::runtime_error("[cufft_wrapper] create_impl: TODO");
    }

    SRTB_CHECK_CUFFT(cufftMakePlanMany64(/* plan = */ plan,
                                         /* rank = */ 1,
                                         /* n = */ &n_,
                                         /* inembed = */ inembed,
                                         /* istride = */ 1,
                                         /* idist = */ idist,
                                         /* onembed = */ onembed,
                                         /* ostride = */ 1,
                                         /* odist = */ odist,
                                         /* type = */ cufft_type,
                                         /* batch = */ batch_size,
                                         /* worksize = */ &workSize));
    set_queue_impl(q);
  }

  void destroy_impl() { SRTB_CHECK_CUFFT(cufftDestroy(plan)); }

  template <typename..., srtb::fft::type fft_type_ = fft_type,
            typename std::enable_if<(fft_type_ == srtb::fft::type::R2C_1D),
                                    int>::type = 0>
  void process_impl(T* in, Complex* out) {
    if constexpr (std::is_same_v<T, cufftReal>) {
      SRTB_CHECK_CUFFT(cufftExecR2C((*this).plan, static_cast<cufftReal*>(in),
                                    reinterpret_cast<cufftComplex*>(out)));
    } else if constexpr (std::is_same_v<T, cufftDoubleReal>) {
      SRTB_CHECK_CUFFT(
          cufftExecD2Z((*this).plan, static_cast<cufftDoubleReal*>(in),
                       reinterpret_cast<cufftDoubleComplex*>(out)));
    } else {
      throw std::runtime_error("[cufft_wrapper] process_impl: TODO");
    }
    (*this).flush();
  }

  template <
      typename..., srtb::fft::type fft_type_ = fft_type,
      typename std::enable_if<(fft_type_ == srtb::fft::type::C2C_1D_FORWARD ||
                               fft_type_ == srtb::fft::type::C2C_1D_BACKWARD),
                              int>::type = 0>
  void process_impl(Complex* in, Complex* out) {
    constexpr auto direction = (fft_type == srtb::fft::type::C2C_1D_BACKWARD)
                                   ? CUFFT_INVERSE
                                   : CUFFT_FORWARD;
    if constexpr (std::is_same_v<T, cufftReal>) {
      SRTB_CHECK_CUFFT(
          cufftExecC2C((*this).plan, reinterpret_cast<cufftComplex*>(in),
                       reinterpret_cast<cufftComplex*>(out), direction));
    } else if constexpr (std::is_same_v<T, cufftDoubleReal>) {
      SRTB_CHECK_CUFFT(
          cufftExecZ2Z((*this).plan, reinterpret_cast<cufftDoubleComplex*>(in),
                       reinterpret_cast<cufftDoubleComplex*>(out), direction));
    } else {
      throw std::runtime_error("[cufft_wrapper] process_impl<C2C_1D>: ?");
    }
    flush();
  }

  bool has_inited_impl() {
    // ref: https://forums.developer.nvidia.com/t/check-for-a-valid-cufft-plan/34297/4
    size_t work_size;
    auto ret_val = cufftGetSize(plan, &work_size);
    switch (ret_val) {
      [[likely]] case CUFFT_SUCCESS : return true;
      case CUFFT_INVALID_PLAN:
        return false;
      default:
        SRTB_CHECK_CUFFT(ret_val);
        return false;
    }
  }

  void set_queue_impl(sycl::queue& queue) {
#if defined(SYCL_EXT_ONEAPI_BACKEND_CUDA)
    stream = sycl::get_native<sycl::backend::ext_oneapi_cuda>(queue);
    SRTB_CHECK_CUFFT(cufftSetStream(plan, stream));
#elif defined(__HIPSYCL__)
    // ref: https://github.com/illuhad/hipSYCL/issues/722
    cufftResult ret = CUFFT_SUCCESS;
    queue
        .submit([&](sycl::handler& cgh) {
          cgh.hipSYCL_enqueue_custom_operation([&](sycl::interop_handle& h) {
            stream = h.get_native_queue<sycl::backend::cuda>();
          });
        })
        .wait();
    // stream seems to be thread-local, so set it in this thread instead of the lambda above
    ret = cufftSetStream(plan, stream);
    if (ret != CUFFT_SUCCESS) [[unlikely]] {
      throw std::runtime_error("[cufft_wrapper] cufftSetStream returned " +
                               std::to_string(ret));
    }
#elif
#warning cufft_wrapper::set_queue_impl uses default stream
    stream = nullptr;
#endif  // SYCL_EXT_ONEAPI_BACKEND_CUDA or __HIPSYCL__
  }

  void flush() { SRTB_CHECK_CUDA(cudaStreamSynchronize((*this).stream)); }
};

}  // namespace fft
}  // namespace srtb

#endif  //  __SRTB_CUFFT_WRAPPER__
