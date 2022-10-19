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

#ifndef __HIP_PLATFORM_AMD__
#define __HIP_PLATFORM_AMD__
#endif

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

template <std::floating_point T>
constexpr inline hipfftType get_hipfft_type(srtb::fft::type fft_type) {
  if constexpr (std::is_same_v<T, hipfftReal>) {
    switch (fft_type) {
      case srtb::fft::type::C2C_1D_FORWARD:
      case srtb::fft::type::C2C_1D_BACKWARD:
        return HIPFFT_C2C;
      case srtb::fft::type::R2C_1D:
        return HIPFFT_R2C;
      case srtb::fft::type::C2R_1D:
        return HIPFFT_C2R;
    }
  } else if constexpr (std::is_same_v<T, hipfftDoubleReal>) {
    switch (fft_type) {
      case srtb::fft::type::C2C_1D_FORWARD:
      case srtb::fft::type::C2C_1D_BACKWARD:
        return HIPFFT_Z2Z;
      case srtb::fft::type::R2C_1D:
        return HIPFFT_D2Z;
      case srtb::fft::type::C2R_1D:
        return HIPFFT_Z2D;
    }
  }
}

template <srtb::fft::type fft_type, std::floating_point T,
          typename Complex = srtb::complex<T> >
class hipfft_1d_wrapper
    : public fft_wrapper<hipfft_1d_wrapper, fft_type, T, Complex> {
 public:
  using super_class = fft_wrapper<hipfft_1d_wrapper, fft_type, T, Complex>;
  friend super_class;
  static_assert((std::is_same_v<T, hipfftReal> &&
                 sizeof(Complex) == sizeof(hipfftComplex)) ||
                (std::is_same_v<T, hipfftDoubleReal> &&
                 sizeof(Complex) == sizeof(hipfftDoubleComplex)));

 protected:
  hipfftHandle plan;
  size_t workSize;
  hipStream_t stream;

 public:
  hipfft_1d_wrapper(size_t n, size_t batch_size, sycl::queue& queue)
      : super_class{n, batch_size, queue} {}

 protected:
  void create_impl(size_t n, size_t batch_size, sycl::queue& q) {
    // should be equivalent to this
    /*
    plan = fftw_plan_dft_r2c_1d(static_cast<int>(n), tmp_in.get(),
                               reinterpret_cast<fftw_complex*>(tmp_out.get()),
                               FFTW_PATIENT |  FFTW_DESTROY_INPUT);
    */

    auto device = q.get_device();
    auto native_device = sycl::get_native<srtb::backend::rocm>(device);
    SRTB_CHECK_HIP(hipSetDevice(native_device));

    SRTB_CHECK_HIPFFT(hipfftCreate(&plan));
    constexpr hipfftType hipfft_type = get_hipfft_type<T>(fft_type);

    // 32-bit version
    //SRTB_CHECK_HIPFFT(hipfftMakePlan1d(plan, static_cast<int>(n), hipfft_type,
    //                                   batch_size, &workSize));

    // 64-bit version
    // TODO: API call returns HIPFFT_NOT_IMPLEMENTED
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
      throw std::runtime_error("[hipfft_wrapper] create_impl: TODO");
    }

    // This returns HIPFFT_NOT_IMPLEMENTED.
    SRTB_CHECK_HIPFFT(hipfftMakePlanMany64(/* plan = */ plan,
                                           /* rank = */ 1,
                                           /* n = */ &n_,
                                           /* inembed = */ inembed,
                                           /* istride = */ 1,
                                           /* idist = */ idist,
                                           /* onembed = */ onembed,
                                           /* ostride = */ 1,
                                           /* odist = */ odist,
                                           /* type = */ hipfft_type,
                                           /* batch = */ batch_size,
                                           /* worksize = */ &workSize));
    SRTB_LOGI << " [hipfft_wrapper] "
              << "plan finished. workSize = " << workSize << srtb::endl;
    set_queue_impl(q);
  }

  void destroy_impl() { SRTB_CHECK_HIPFFT(hipfftDestroy(plan)); }

  template <typename..., srtb::fft::type fft_type_ = fft_type,
            typename std::enable_if<(fft_type_ == srtb::fft::type::R2C_1D),
                                    int>::type = 0>
  void process_impl(T* in, Complex* out) {
    if constexpr (std::is_same_v<T, hipfftReal>) {
      SRTB_CHECK_HIPFFT(hipfftExecR2C((*this).plan,
                                      static_cast<hipfftReal*>(in),
                                      reinterpret_cast<hipfftComplex*>(out)));
    } else if constexpr (std::is_same_v<T, hipfftDoubleReal>) {
      SRTB_CHECK_HIPFFT(
          hipfftExecD2Z((*this).plan, static_cast<hipfftDoubleReal*>(in),
                        reinterpret_cast<hipfftDoubleComplex*>(out)));
    } else {
      throw std::runtime_error("[hipfft_wrapper] process_impl<R2C_1D>: ?");
    }
    flush();
  }

  template <
      typename..., srtb::fft::type fft_type_ = fft_type,
      typename std::enable_if<(fft_type_ == srtb::fft::type::C2C_1D_FORWARD ||
                               fft_type_ == srtb::fft::type::C2C_1D_BACKWARD),
                              int>::type = 0>
  void process_impl(Complex* in, Complex* out) {
    constexpr auto direction = (fft_type == srtb::fft::type::C2C_1D_BACKWARD)
                                   ? HIPFFT_BACKWARD
                                   : HIPFFT_FORWARD;
    if constexpr (std::is_same_v<T, hipfftReal>) {
      SRTB_CHECK_HIPFFT(
          hipfftExecC2C((*this).plan, reinterpret_cast<hipfftComplex*>(in),
                        reinterpret_cast<hipfftComplex*>(out), direction));
    } else if constexpr (std::is_same_v<T, hipfftDoubleReal>) {
      SRTB_CHECK_HIPFFT(hipfftExecZ2Z(
          (*this).plan, reinterpret_cast<hipfftDoubleComplex*>(in),
          reinterpret_cast<hipfftDoubleComplex*>(out), direction));
    } else {
      throw std::runtime_error("[hipfft_wrapper] process_impl<C2C_1D>: ?");
    }
    flush();
  }

  template <typename..., srtb::fft::type fft_type_ = fft_type,
            typename std::enable_if<(fft_type_ == srtb::fft::type::C2R_1D),
                                    int>::type = 0>
  void process_impl(Complex* in, T* out) {
    if constexpr (std::is_same_v<T, hipfftReal>) {
      SRTB_CHECK_HIPFFT(hipfftExecC2R((*this).plan,
                                      reinterpret_cast<hipfftComplex*>(in),
                                      static_cast<hipfftReal*>(out)));
    } else if constexpr (std::is_same_v<T, hipfftDoubleReal>) {
      SRTB_CHECK_HIPFFT(hipfftExecZ2D(
          (*this).plan, reinterpret_cast<hipfftDoubleComplex*>(in),
          static_cast<hipfftDoubleReal*>(out)));
    } else {
      throw std::runtime_error("[hipfft_wrapper] process_impl<R2C_1D>: ?");
    }
    flush();
  }

  bool has_inited_impl() {
    // invalid plan causes segmentation fault, so not using plan to check here.
    return true;
  }

  void set_queue_impl(sycl::queue& queue) {
#if defined(SYCL_EXT_ONEAPI_BACKEND_HIP)
    stream = sycl::get_native<sycl::backend::ext_oneapi_hip>(queue);
    SRTB_CHECK_HIPFFT(hipfftSetStream(plan, stream));
#elif defined(__HIPSYCL__)
    // ref: https://github.com/illuhad/hipSYCL/issues/722
    hipfftResult ret = HIPFFT_SUCCESS;
    queue
        .submit([&](sycl::handler& cgh) {
          cgh.hipSYCL_enqueue_custom_operation([&](sycl::interop_handle& h) {
            stream = h.get_native_queue<sycl::backend::hip>();
          });
        })
        .wait();
    // stream seems to be thread-local, so set it in this thread instead of the lambda above
    ret = hipfftSetStream(plan, stream);
    if (ret != HIPFFT_SUCCESS) [[unlikely]] {
      throw std::runtime_error("[hipfft_wrapper] hipfftSetStream returned " +
                               std::to_string(ret));
    }
#else
#warning hipfft_wrapper::set_queue_impl uses default stream
    stream = nullptr;
#endif  // SYCL_EXT_ONEAPI_BACKEND_HIP or __HIPSYCL__
  }

  void flush() { SRTB_CHECK_HIP(hipStreamSynchronize((*this).stream)); }
};

}  // namespace fft
}  // namespace srtb

#undef __HIP_PLATFORM_AMD__

#endif  //  __SRTB_HIPFFT_WRAPPER__
