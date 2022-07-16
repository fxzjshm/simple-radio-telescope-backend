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
#ifndef __SRTB_HIPFFT_WRAPPER__
#define __SRTB_HIPFFT_WRAPPER__

#define __HIP_PLATFORM_AMD__
#include <hipfft/hipfft.h>

#include <concepts>

#include "srtb/fft/fft_wrapper.hpp"
#include "srtb/global_variables.hpp"

#define SRTB_CHECK_HIPFFT_WRAPPER(expr, expected)                      \
  SRTB_CHECK(expr, expected,                                           \
             throw std::runtime_error(                                 \
                 std::string("[hipfft_wrapper] " #expr " returned ") + \
                 std::to_string(ret)););

#define SRTB_CHECK_HIPFFT(expr) SRTB_CHECK_HIPFFT_WRAPPER(expr, HIPFFT_SUCCESS)

#define SRTB_CHECK_HIP(expr) SRTB_CHECK_HIPFFT_WRAPPER(expr, hipSuccess)

namespace srtb {
namespace fft {

template <std::floating_point T, typename Complex = srtb::complex<T> >
class hipfft_1d_r2c_wrapper;

template <std::floating_point T, typename Complex = srtb::complex<T> >
class hipfft_1d_r2c_wrapper_abstract
    : public fft_wrapper<hipfft_1d_r2c_wrapper, T, Complex> {
  friend fft_wrapper<hipfft_1d_r2c_wrapper, T, Complex>;

 protected:
  void create_abstract(size_t n, hipfftType hipfft_type) {
    // should be equivalent to this
    /*
    plan = fftw_plan_dft_r2c_1d(static_cast<int>(n), tmp_in.get(),
                               reinterpret_cast<fftw_complex*>(tmp_out.get()),
                               FFTW_PATIENT |  FFTW_DESTROY_INPUT);
    */

    SRTB_CHECK_HIPFFT(hipfftCreate(&plan));
    long long int n_ = static_cast<long long int>(n);
    long long int inembed[1] = {n_}, onembed[1] = {n_ / 2 + 1};
    SRTB_CHECK_HIPFFT(
        hipfftMakePlan1d(plan, static_cast<int>(n), hipfft_type, 1, &workSize));

    //SRTB_CHECK_HIPFFT(hipfftMakePlanMany64(/* plan = */ plan,
    //                                       /* rank = */ 1,
    //                                       /* n = */ &n_,
    //                                       /* inembed = */ inembed,
    //                                       /* istride = */ 1,
    //                                       /* idist = */ n_,
    //                                       /* onembed = */ onembed,
    //                                       /* ostride = */ 1,
    //                                       /* odist = */ n_,
    //                                       /* type = */ hipfft_type,
    //                                       /* batch = */ 1,
    //                                       /* worksize = */ &workSize));
  }

  void destroy_impl() { SRTB_CHECK_HIPFFT(hipfftDestroy(plan)); }

  bool has_inited_impl() {
    // TODO: check plan?
    return true;
  }

  void set_queue_impl(sycl::queue& queue) {
#if defined(SYCL_EXT_ONEAPI_BACKEND_HIP)
    stream = sycl::get_native<sycl::backend::ext_oneapi_hip>(queue);
    SRTB_CHECK_HIPFFT(hipfftSetStream(plan, stream));
#elif defined(__HIPSYCL__)
    // ref: https://github.com/illuhad/hipSYCL/issues/722
    hipfftResult ret = HIPFFT_SUCCESS;
    q.submit([&](sycl::handler& cgh) {
       cgh.hipSYCL_enqueue_custom_operation([&](sycl::interop_handle& h) {
         stream = h.get_native_queue<sycl::backend::hip>();
         ret = hipfftSetStream(plan, stream);
       });
     }).wait();
    if (ret != HIPFFT_SUCCESS) [[unlikely]] {
      throw std::runtime_error("[hipfft_wrapper] hipfftSetStream returned " +
                               std::to_string(ret));
    }
#elif
#warning hipfft_wrapper::set_queue_impl does nothing
#endif  // SYCL_EXT_ONEAPI_BACKEND_HIP or __HIPSYCL__
  }

  void flush() { SRTB_CHECK_HIP(hipStreamSynchronize((*this).stream)); }

 protected:
  hipfftHandle plan;
  size_t workSize;
  hipStream_t stream = nullptr;
};

template <typename Complex>
class hipfft_1d_r2c_wrapper<float, Complex>
    : public hipfft_1d_r2c_wrapper_abstract<float, Complex> {
  friend fft_wrapper<hipfft_1d_r2c_wrapper, float, Complex>;

 protected:
  void create_impl(size_t n) { (*this).create_abstract(n, HIPFFT_R2C); }

  void process_impl(float* in, Complex* out) {
    SRTB_CHECK_HIPFFT(hipfftExecR2C((*this).plan, static_cast<hipfftReal*>(in),
                                    reinterpret_cast<hipfftComplex*>(out)));
    (*this).flush();
  }
};

template <typename Complex>
class hipfft_1d_r2c_wrapper<double, Complex>
    : public hipfft_1d_r2c_wrapper_abstract<double, Complex> {
  friend fft_wrapper<hipfft_1d_r2c_wrapper, double, Complex>;

 protected:
  void create_impl(size_t n) { (*this).create_abstract(n, HIPFFT_D2Z); }

  void process_impl(double* in, Complex* out) {
    SRTB_CHECK_HIPFFT(
        hipfftExecD2Z((*this).plan, static_cast<hipfftDoubleReal*>(in),
                      reinterpret_cast<hipfftDoubleComplex*>(out)));
    (*this).flush();
  }
};

}  // namespace fft
}  // namespace srtb

#undef __HIP_PLATFORM_AMD__

#endif  //  __SRTB_HIPFFT_WRAPPER__
