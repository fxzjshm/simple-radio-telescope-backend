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
#ifndef __SRTB_FFT__
#define __SRTB_FFT__

#include <complex>
#include <exception>

#include "srtb/config.hpp"
#include "srtb/fft/fft_wrapper.hpp"
#include "srtb/fft/fftw_wrapper.hpp"
//#include "srtb/fft/cufft_wrapper.hpp"
//#include "srtb/fft/rocfft_wrapper.hpp"

namespace srtb {
namespace fft {

/**
 * @brief Get the fftw 1d r2c wrapper object. 
 *        Static local variable is used so that it won't be initialized 
 *        if not used, thus incorrect device_allocator won't cause 
 *        segmentation fault, maybe...
 */
template <typename T = srtb::real, typename C = std::complex<T> >
inline fftw_1d_r2c_wrapper<T, C>& get_fftw_1d_r2c_wrapper() {
  static fftw_1d_r2c_wrapper<T, C> fftw_1d_r2c_wrapper_instance;
  return fftw_1d_r2c_wrapper_instance;
}

#define SRTB_FFT_DISPATCH(queue, type, func, ...)                  \
  {                                                                \
    auto device = queue.get_device();                              \
    if (device.is_cpu() || device.is_host()) {                     \
      get_fftw_##type##_wrapper().func(__VA_ARGS__);               \
    } else {                                                       \
      throw std::runtime_error{" [fft] dispatch_" #type ": TODO"}; \
    }                                                              \
  }

inline void init_1d_r2c(sycl::queue& queue = srtb::queue) {
  SRTB_FFT_DISPATCH(queue, 1d_r2c, create);
}

template <typename T = srtb::real, typename C = std::complex<T> >
inline void dispatch_1d_r2c(T* in, C* out, sycl::queue& queue = srtb::queue) {
  SRTB_FFT_DISPATCH(queue, 1d_r2c, process, in, out);
}

#undef SRTB_FFT_DISPATCH

}  // namespace fft
}  // namespace srtb

#endif  //  __SRTB_FFT__
