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

#include <exception>

#include "srtb/config.hpp"
#include "srtb/fft/fft_wrapper.hpp"
#include "srtb/fft/fftw_wrapper.hpp"
#ifdef SRTB_ENABLE_CUDA_INTEROP
#include "srtb/fft/cufft_wrapper.hpp"
#endif  // SRTB_ENABLE_CUDA_INTEROP
#ifdef SRTB_ENABLE_ROCM_INTEROP
#include "srtb/fft/hipfft_wrapper.hpp"
#endif  // SRTB_ENABLE_ROCM_INTEROP

namespace srtb {
namespace fft {

/**
 * @brief Get the fftw 1d r2c wrapper object. 
 *        Static local variable is used so that it won't be initialized 
 *        if not used, thus incorrect device_allocator won't cause 
 *        segmentation fault, maybe...
 */
template <typename T = srtb::real, typename C = srtb::complex<T> >
inline fftw_1d_r2c_wrapper<T, C>& get_fftw_1d_r2c_wrapper() {
  static fftw_1d_r2c_wrapper<T, C> fftw_1d_r2c_wrapper_instance;
  return fftw_1d_r2c_wrapper_instance;
}

#ifdef SRTB_ENABLE_CUDA_INTEROP
/**
 * @brief Get the cufft 1d r2c wrapper object. 
 */
template <typename T = srtb::real, typename C = srtb::complex<T> >
inline cufft_1d_r2c_wrapper<T, C>& get_cufft_1d_r2c_wrapper() {
  static cufft_1d_r2c_wrapper<T, C> cufft_1d_r2c_wrapper_instance;
  return cufft_1d_r2c_wrapper_instance;
}
#endif  // SRTB_ENABLE_CUDA_INTEROP

#ifdef SRTB_ENABLE_ROCM_INTEROP
/**
 * @brief Get the hipfft 1d r2c wrapper object. 
 */
template <typename T = srtb::real, typename C = srtb::complex<T> >
inline hipfft_1d_r2c_wrapper<T, C>& get_hipfft_1d_r2c_wrapper() {
  static hipfft_1d_r2c_wrapper<T, C> hipfft_1d_r2c_wrapper_instance;
  return hipfft_1d_r2c_wrapper_instance;
}
#endif  // SRTB_ENABLE_ROCM_INTEROP

#define SRTB_FFT_DISPATCH(queue, type, func, ...)               \
  {                                                             \
    auto device = queue.get_device();                           \
    SRTB_IF_ENABLED_CUDA_INTEROP({                              \
      if (device.get_backend() == srtb::backend::cuda) {        \
        return get_cufft_##type##_wrapper().func(__VA_ARGS__);  \
      }                                                         \
    });                                                         \
                                                                \
    SRTB_IF_ENABLED_ROCM_INTEROP({                              \
      if (device.get_backend() == srtb::backend::rocm) {        \
        return get_hipfft_##type##_wrapper().func(__VA_ARGS__); \
      }                                                         \
    });                                                         \
                                                                \
    if (device.is_cpu() || device.is_host()) {                  \
      return get_fftw_##type##_wrapper().func(__VA_ARGS__);     \
    }                                                           \
                                                                \
    throw std::runtime_error{"[fft] dispatch_" #type ": TODO"}; \
  }

template <typename T = srtb::real, typename C = srtb::complex<T> >
inline void dispatch_1d_r2c(T* in, C* out, sycl::queue& queue = srtb::queue) {
  SRTB_FFT_DISPATCH(queue, 1d_r2c, process, in, out);
}

inline size_t get_size_1d_r2c(sycl::queue& queue = srtb::queue) {
  SRTB_FFT_DISPATCH(queue, 1d_r2c, get_size);
}

inline void set_size_1d_r2c(size_t n) {
  SRTB_FFT_DISPATCH(queue, 1d_r2c, set_size, n);
}

inline void set_queue_1d_r2c(sycl::queue& queue) {
  SRTB_FFT_DISPATCH(queue, 1d_r2c, set_queue, queue);
}

inline void init_1d_r2c(size_t n = default_fft_1d_r2c_input_size(),
                        sycl::queue& queue = srtb::queue) {
  set_size_1d_r2c(n);
  // not only construct the object by accessing it, but also set queue to be used.
  set_queue_1d_r2c(queue);
}

#undef SRTB_FFT_DISPATCH

}  // namespace fft
}  // namespace srtb

#endif  //  __SRTB_FFT__
