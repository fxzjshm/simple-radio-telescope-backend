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

#include "srtb/config.hpp"
#include "srtb/fft/fft_wrapper.hpp"
#include "srtb/fft/fftw_wrapper.hpp"
//#include "srtb/fft/cufft_wrapper.hpp"
//#include "srtb/fft/rocfft_wrapper.hpp"

namespace srtb {
namespace fft {

template <typename T = srtb::real, typename C = std::complex<T> >
inline void dispatch_1d_r2c(T* in, C* out, sycl::queue = srtb::queue) {
  static fftw_1d_r2c_wrapper<T, C> fftw_1d_r2c_wrapper_instance;

  auto device = queue.get_device();
  if (device.is_cpu() || device.is_host()) {
    fftw_1d_r2c_wrapper_instance.process(in, out);
  } else {
    throw std::runtime_error{" [fft] dispatch_1d_r2c: TODO"};
  }
}

}  // namespace fft
}  // namespace srtb

#endif  //  __SRTB_FFT__
