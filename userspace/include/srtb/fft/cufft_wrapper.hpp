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

    cufftResult ret = cufftCreate(&plan);
    if (ret != CUFFT_SUCCESS) [[unlikely]] {
      throw std::runtime_error("[cufft_wrapper] cufftCreate returned " +
                               std::to_string(ret));
    }
    long long int n_ = static_cast<long long int>(n);
    //ret = cufftMakePlanMany64(/* plan = */ plan,
    //                          /* rank = */ 1,
    //                          /* n = */ &n_,
    //                          /* inembed = */ NULL,
    //                          /* istride = */ 1,
    //                          /* idist = */ 0,
    //                          /* onenbed = */ NULL,
    //                          /* ostride = */ 1,
    //                          /* odist = */ 0,
    //                          /* type = */ cufft_type,
    //                          /* batch = */ 1,
    //                          /* worksize = */ &workSize);
    long long int inembed[1] = {n_}, onembed[1] = {n_ / 2 + 1};
    ret = cufftMakePlanMany64(/* plan = */ plan,
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
                              /* worksize = */ &workSize);
    if (ret != CUFFT_SUCCESS) [[unlikely]] {
      throw std::runtime_error("[cufft_wrapper] cufftMakePlanMany64 returned " +
                               std::to_string(ret));
    }
  }

  void destroy_impl() {
    cufftResult ret = cufftDestroy(plan);
    if (ret != CUFFT_SUCCESS) [[unlikely]] {
      throw std::runtime_error("[cufft_wrapper] cufftDestroy returned " +
                               std::to_string(ret));
    }
  }

  bool has_inited_impl() {
    // TODO: check plan?
    return true;
  }

  void set_queue_impl(sycl::queue& queue) {
    cufftResult ret = CUFFT_SUCCESS;
#if defined(SYCL_EXT_ONEAPI_BACKEND_CUDA)
    stream = sycl::get_native<sycl::backend::ext_oneapi_cuda>(queue);
    ret = cufftSetStream(plan, stream);
#elif defined(__HIPSYCL__)
    // ref: https://github.com/illuhad/hipSYCL/issues/722
    q.submit([&](sycl::handler& cgh) {
       cgh.hipSYCL_enqueue_custom_operation([&](sycl::interop_handle& h) {
         stream = h.get_native_queue<sycl::backend::cuda>();
         ret = cufftSetStream(plan, stream);
       });
     }).wait();
#elif
#warning cufft_wrapper::set_queue_impl does nothing
#endif  // SYCL_EXT_ONEAPI_BACKEND_CUDA or __HIPSYCL__
    if (ret != CUFFT_SUCCESS) [[unlikely]] {
      throw std::runtime_error("[cufft_wrapper] cufftSetStream returned " +
                               std::to_string(ret));
    }
  }

 protected:
  cufftHandle plan;
  size_t workSize;
  cudaStream_t stream = nullptr;
};

template <typename Complex>
class cufft_1d_r2c_wrapper<float, Complex>
    : public cufft_1d_r2c_wrapper_abstract<float, Complex> {
  friend fft_wrapper<cufft_1d_r2c_wrapper, float, Complex>;

 protected:
  void create_impl(size_t n) { (*this).create_abstract(n, CUFFT_R2C); }

  void process_impl(float* in, Complex* out) {
    cufftResult fft_ret = CUFFT_SUCCESS;
    fft_ret = cufftExecR2C((*this).plan, static_cast<cufftReal*>(in),
                           reinterpret_cast<cufftComplex*>(out));
    if (fft_ret != CUFFT_SUCCESS) [[unlikely]] {
      throw std::runtime_error("[cufft_wrapper] cufftExecR2C returned " +
                               std::to_string(fft_ret));
    }
    cudaError_t runtime_ret = cudaSuccess;
    runtime_ret = cudaStreamSynchronize((*this).stream);
    if (runtime_ret != cudaSuccess) [[unlikely]] {
      throw std::runtime_error(
          "[cufft_wrapper] cudaStreamSynchronize returned " +
          std::to_string(runtime_ret));
    }
  }
};

template <typename Complex>
class cufft_1d_r2c_wrapper<double, Complex>
    : public cufft_1d_r2c_wrapper_abstract<double, Complex> {
  friend fft_wrapper<cufft_1d_r2c_wrapper, double, Complex>;

 protected:
  void create_impl(size_t n) { (*this).create_abstract(n, CUFFT_D2Z); }

  void process_impl(double* in, Complex* out) {
    cufftResult fft_ret = CUFFT_SUCCESS;
    fft_ret = cufftExecD2Z((*this).plan, static_cast<cufftDoubleReal*>(in),
                           reinterpret_cast<cufftDoubleComplex*>(out));
    if (fft_ret != CUFFT_SUCCESS) [[unlikely]] {
      throw std::runtime_error("[cufft_wrapper] cufftExecR2C returned " +
                               std::to_string(fft_ret));
    }
    cudaError_t runtime_ret = cudaSuccess;
    runtime_ret = cudaStreamSynchronize((*this).stream);
    if (runtime_ret != cudaSuccess) [[unlikely]] {
      throw std::runtime_error(
          "[cufft_wrapper] cudaStreamSynchronize returned " +
          std::to_string(runtime_ret));
    }
  }
};

}  // namespace fft
}  // namespace srtb

#endif  //  __SRTB_CUFFT_WRAPPER__
