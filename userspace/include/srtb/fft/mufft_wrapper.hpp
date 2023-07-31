/******************************************************************************* 
 * Copyright (c) 2022-2023 fxzjshm
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
#ifndef __SRTB_MUFFT_WRAPPER__
#define __SRTB_MUFFT_WRAPPER__

#include <musa_runtime.h>
#include <mufft.h>

#include <concepts>

#include "srtb/fft/cufft_like_wrapper.hpp"

namespace srtb {
namespace fft {

// common types & functions that not related to data type
struct mufft_common_trait {
  static constexpr auto backend = srtb::backend::musa;
  using fft_handle = mufftHandle;
  using stream_t = musaStream_t;

  static constexpr auto C2C = MUFFT_C2C;
  static constexpr auto R2C = MUFFT_R2C;
  static constexpr auto C2R = MUFFT_C2R;

  static constexpr auto FORWARD = MUFFT_FORWARD;
  static constexpr auto BACKWARD = MUFFT_INVERSE;

  static constexpr auto FFT_SUCCESS = MUFFT_SUCCESS;
  static constexpr auto API_SUCCESS = musaSuccess;

  template <typename... Args>
  static inline decltype(auto) SetDevice(Args&&... args) {
    return musaSetDevice(std::forward<Args>(args)...);
  }

  template <typename... Args>
  static inline decltype(auto) StreamSynchronize(Args&&... args) {
    return musaStreamSynchronize(std::forward<Args>(args)...);
  }

  template <typename... Args>
  static inline decltype(auto) fftCreate(Args&&... args) {
    return mufftCreate(std::forward<Args>(args)...);
  }

  template <typename... Args>
  static inline decltype(auto) fftDestroy(Args&&... args) {
    return mufftDestroy(std::forward<Args>(args)...);
  }

  template <typename... Args>
  static inline decltype(auto) fftMakePlan1d(Args&&... args) {
    return mufftMakePlan1d(std::forward<Args>(args)...);
  }

  template <typename... Args>
  static inline decltype(auto) fftMakePlanMany64(Args&&... args) {
    // not even defined (was commented out)
    //return mufftMakePlanMany64(std::forward<Args>(args)...);
    return MUFFT_NOT_IMPLEMENTED;
  }

  template <typename... Args>
  static inline decltype(auto) fftSetStream(Args&&... args) {
    return mufftSetStream(std::forward<Args>(args)...);
  }

  template <typename... Args>
  static inline decltype(auto) fftSetWorkArea(Args&&... args) {
    return mufftSetWorkArea(std::forward<Args>(args)...);
  }

  template <typename... Args>
  static inline decltype(auto) fftSetAutoAllocation(Args&&... args) {
    return mufftSetAutoAllocation(std::forward<Args>(args)...);
  }
};

template <>
struct cufft_like_trait<float, srtb::backend::musa> : mufft_common_trait {
  using real = mufftReal;
  using complex = mufftComplex;

  static inline constexpr auto get_native_fft_type(srtb::fft::type fft_type) {
    switch (fft_type) {
      case srtb::fft::type::C2C_1D_FORWARD:
      case srtb::fft::type::C2C_1D_BACKWARD:
        return MUFFT_C2C;
      case srtb::fft::type::R2C_1D:
        return MUFFT_R2C;
      case srtb::fft::type::C2R_1D:
        return MUFFT_C2R;
    }
  }

  template <typename... Args>
  static inline decltype(auto) fftExecR2C(Args&&... args) {
    return mufftExecR2C(std::forward<Args>(args)...);
  }

  template <typename... Args>
  static inline decltype(auto) fftExecC2C(Args&&... args) {
    return mufftExecC2C(std::forward<Args>(args)...);
  }

  template <typename... Args>
  static inline decltype(auto) fftExecC2R(Args&&... args) {
    return mufftExecC2R(std::forward<Args>(args)...);
  }
};

template <>
struct cufft_like_trait<double, srtb::backend::musa> : mufft_common_trait {
  using real = mufftDoubleReal;
  using complex = mufftDoubleComplex;

  static inline constexpr auto get_native_fft_type(srtb::fft::type fft_type) {
    switch (fft_type) {
      case srtb::fft::type::C2C_1D_FORWARD:
      case srtb::fft::type::C2C_1D_BACKWARD:
        return MUFFT_Z2Z;
      case srtb::fft::type::R2C_1D:
        return MUFFT_D2Z;
      case srtb::fft::type::C2R_1D:
        return MUFFT_Z2D;
    }
  }

  template <typename... Args>
  static inline decltype(auto) fftExecR2C(Args&&... args) {
    return mufftExecD2Z(std::forward<Args>(args)...);
  }

  template <typename... Args>
  static inline decltype(auto) fftExecC2C(Args&&... args) {
    return mufftExecZ2Z(std::forward<Args>(args)...);
  }

  template <typename... Args>
  static inline decltype(auto) fftExecC2R(Args&&... args) {
    return mufftExecZ2D(std::forward<Args>(args)...);
  }
};

template <std::floating_point T>
using mufft_trait = cufft_like_trait<T, srtb::backend::musa>;

template <srtb::fft::type fft_type, std::floating_point T,
          typename C = srtb::complex<T> >
using mufft_1d_wrapper =
    cufft_like_1d_wrapper<srtb::backend::musa, fft_type, T, C>;

}  // namespace fft
}  // namespace srtb

#endif  //  __SRTB_MUFFT_WRAPPER__
