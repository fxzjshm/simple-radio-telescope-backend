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

#include "srtb/fft/cufft_like_wrapper.hpp"

namespace srtb {
namespace fft {

// common types & functions that not related to data type
struct cufft_common_trait {
  static constexpr auto backend = srtb::backend::cuda;
  using fft_handle = cufftHandle;
  using stream_t = cudaStream_t;

  static constexpr auto C2C = CUFFT_C2C;
  static constexpr auto R2C = CUFFT_R2C;
  static constexpr auto C2R = CUFFT_C2R;

  static constexpr auto FORWARD = CUFFT_FORWARD;
  static constexpr auto BACKWARD = CUFFT_INVERSE;

  static constexpr auto FFT_SUCCESS = CUFFT_SUCCESS;
  static constexpr auto API_SUCCESS = cudaSuccess;

  template <typename... Args>
  static inline decltype(auto) SetDevice(Args&&... args) {
    return cudaSetDevice(std::forward<Args>(args)...);
  }

  template <typename... Args>
  static inline decltype(auto) StreamSynchronize(Args&&... args) {
    return cudaStreamSynchronize(std::forward<Args>(args)...);
  }

  template <typename... Args>
  static inline decltype(auto) fftCreate(Args&&... args) {
    return cufftCreate(std::forward<Args>(args)...);
  }

  template <typename... Args>
  static inline decltype(auto) fftDestroy(Args&&... args) {
    return cufftDestroy(std::forward<Args>(args)...);
  }

  template <typename... Args>
  static inline decltype(auto) fftMakePlan1d(Args&&... args) {
    return cufftMakePlan1d(std::forward<Args>(args)...);
  }

  template <typename... Args>
  static inline decltype(auto) fftMakePlanMany64(Args&&... args) {
    return cufftMakePlanMany64(std::forward<Args>(args)...);
  }

  template <typename... Args>
  static inline decltype(auto) fftSetStream(Args&&... args) {
    return cufftSetStream(std::forward<Args>(args)...);
  }
};

template <>
struct cufft_like_trait<float, srtb::backend::cuda> : cufft_common_trait {
  using real = cufftReal;
  using complex = cufftComplex;

  static inline constexpr auto get_native_fft_type(srtb::fft::type fft_type) {
    switch (fft_type) {
      case srtb::fft::type::C2C_1D_FORWARD:
      case srtb::fft::type::C2C_1D_BACKWARD:
        return CUFFT_C2C;
      case srtb::fft::type::R2C_1D:
        return CUFFT_R2C;
      case srtb::fft::type::C2R_1D:
        return CUFFT_C2R;
    }
  }

  template <typename... Args>
  static inline decltype(auto) fftExecR2C(Args&&... args) {
    return cufftExecR2C(std::forward<Args>(args)...);
  }

  template <typename... Args>
  static inline decltype(auto) fftExecC2C(Args&&... args) {
    return cufftExecC2C(std::forward<Args>(args)...);
  }

  template <typename... Args>
  static inline decltype(auto) fftExecC2R(Args&&... args) {
    return cufftExecC2R(std::forward<Args>(args)...);
  }
};

template <>
struct cufft_like_trait<double, srtb::backend::cuda> : cufft_common_trait {
  using real = cufftDoubleReal;
  using complex = cufftDoubleComplex;

  static inline constexpr auto get_native_fft_type(srtb::fft::type fft_type) {
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

  template <typename... Args>
  static inline decltype(auto) fftExecR2C(Args&&... args) {
    return cufftExecD2Z(std::forward<Args>(args)...);
  }

  template <typename... Args>
  static inline decltype(auto) fftExecC2C(Args&&... args) {
    return cufftExecZ2Z(std::forward<Args>(args)...);
  }

  template <typename... Args>
  static inline decltype(auto) fftExecC2R(Args&&... args) {
    return cufftExecZ2D(std::forward<Args>(args)...);
  }
};

template <std::floating_point T>
using cufft_trait = cufft_like_trait<T, srtb::backend::cuda>;

template <srtb::fft::type fft_type, std::floating_point T,
          typename C = srtb::complex<T> >
using cufft_1d_wrapper =
    cufft_like_1d_wrapper<srtb::backend::cuda, fft_type, T, C>;

}  // namespace fft
}  // namespace srtb

#endif  //  __SRTB_CUFFT_WRAPPER__
