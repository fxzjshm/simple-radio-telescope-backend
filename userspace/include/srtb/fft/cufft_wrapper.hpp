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

  cufft_1d_wrapper(size_t n, size_t batch_size) : super_class{n, batch_size} {}

 protected:
  void create_impl(size_t n, size_t batch_size) {
    // should be equivalent to this
    /*
    plan = fftw_plan_dft_r2c_1d(static_cast<int>(n), tmp_in.get(),
                               reinterpret_cast<fftw_complex*>(tmp_out.get()),
                               FFTW_PATIENT |  FFTW_DESTROY_INPUT);
    */

    SRTB_CHECK_CUFFT(cufftCreate(&plan));
    long long int n_ = static_cast<long long int>(n);
    long long int inembed[1] = {n_}, onembed[1] = {n_ / 2 + 1};
    cufftType cufft_type = get_cufft_type<T>(fft_type);
    if constexpr (fft_type == srtb::fft::type::C2C_1D_FORWARD ||
                  fft_type == srtb::fft::type::C2C_1D_BACKWARD ||
                  fft_type == srtb::fft::type::R2C_1D) {
      SRTB_CHECK_CUFFT(cufftMakePlanMany64(/* plan = */ plan,
                                           /* rank = */ 1,
                                           /* n = */ &n_,
                                           /* inembed = */ inembed,
                                           /* istride = */ 1,
                                           /* idist = */ n_,
                                           /* onembed = */ onembed,
                                           /* ostride = */ 1,
                                           /* odist = */ n_,
                                           /* type = */ cufft_type,
                                           /* batch = */ batch_size,
                                           /* worksize = */ &workSize));
    } else {
      throw std::runtime_error("[cufft_wrapper] create_impl: TODO");
    }
  }

  void destroy_impl() { SRTB_CHECK_CUFFT(cufftDestroy(plan)); }

  typename std::enable_if<(fft_type == srtb::fft::type::R2C_1D), void>::type
  process_impl(T* in, Complex* out) {
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
            ret = cufftSetStream(plan, stream);
          });
        })
        .wait();
    if (ret != CUFFT_SUCCESS) [[unlikely]] {
      throw std::runtime_error("[cufft_wrapper] cufftSetStream returned " +
                               std::to_string(ret));
    }
#elif
#warning cufft_wrapper::set_queue_impl does nothing
#endif  // SYCL_EXT_ONEAPI_BACKEND_CUDA or __HIPSYCL__
  }

  void flush() { SRTB_CHECK_CUDA(cudaStreamSynchronize((*this).stream)); }

 protected:
  cufftHandle plan;
  size_t workSize;
  cudaStream_t stream = nullptr;
};

}  // namespace fft
}  // namespace srtb

#endif  //  __SRTB_CUFFT_WRAPPER__
