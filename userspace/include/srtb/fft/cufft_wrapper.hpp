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

template <std::floating_point T, typename Complex = srtb::complex<T> >
class cufft_1d_r2c_wrapper;

template <std::floating_point T, typename Complex = srtb::complex<T> >
class cufft_1d_r2c_wrapper_abstract
    : public fft_wrapper<cufft_1d_r2c_wrapper, T, Complex> {
  friend fft_wrapper<cufft_1d_r2c_wrapper, T, Complex>;

 protected:
  void create_abstract(size_t n, cufftType cufft_type) {
    // should be equivalent to this
    /*
    plan = fftw_plan_dft_r2c_1d(static_cast<int>(n), tmp_in.get(),
                               reinterpret_cast<fftw_complex*>(tmp_out.get()),
                               FFTW_PATIENT |  FFTW_DESTROY_INPUT);
    */

    SRTB_CHECK_CUFFT(cufftCreate(&plan));
    long long int n_ = static_cast<long long int>(n);
    long long int inembed[1] = {n_}, onembed[1] = {n_ / 2 + 1};
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
                                         /* batch = */ 1,
                                         /* worksize = */ &workSize));
  }

  void destroy_impl() { SRTB_CHECK_CUFFT(cufftDestroy(plan)); }

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

template <typename Complex>
class cufft_1d_r2c_wrapper<float, Complex>
    : public cufft_1d_r2c_wrapper_abstract<float, Complex> {
  friend fft_wrapper<cufft_1d_r2c_wrapper, float, Complex>;
  static_assert(std::is_same_v<float, cufftReal>);
  static_assert(sizeof(Complex) == sizeof(cufftComplex));

 protected:
  void create_impl(size_t n) { (*this).create_abstract(n, CUFFT_R2C); }

  void process_impl(float* in, Complex* out) {
    SRTB_CHECK_CUFFT(cufftExecR2C((*this).plan, static_cast<cufftReal*>(in),
                                  reinterpret_cast<cufftComplex*>(out)));
    (*this).flush();
  }
};

template <typename Complex>
class cufft_1d_r2c_wrapper<double, Complex>
    : public cufft_1d_r2c_wrapper_abstract<double, Complex> {
  friend fft_wrapper<cufft_1d_r2c_wrapper, double, Complex>;
  static_assert(std::is_same_v<double, cufftDoubleReal>);
  static_assert(sizeof(Complex) == sizeof(cufftDoubleComplex));

 protected:
  void create_impl(size_t n) { (*this).create_abstract(n, CUFFT_D2Z); }

  void process_impl(double* in, Complex* out) {
    SRTB_CHECK_CUFFT(cufftExecD2Z((*this).plan,
                                  static_cast<cufftDoubleReal*>(in),
                                  reinterpret_cast<cufftDoubleComplex*>(out)));
    (*this).flush();
  }
};

}  // namespace fft
}  // namespace srtb

#endif  //  __SRTB_CUFFT_WRAPPER__
