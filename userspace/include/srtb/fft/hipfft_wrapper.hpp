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

#include "srtb/fft/cufft_like_wrapper.hpp"

namespace srtb {
namespace fft {

// common types & functions that not related to data type
struct hipfft_common_trait {
  static constexpr auto backend = srtb::backend::rocm;
  using fft_handle = hipfftHandle;
  using stream_t = hipStream_t;

  static constexpr auto C2C = HIPFFT_C2C;
  static constexpr auto R2C = HIPFFT_R2C;
  static constexpr auto C2R = HIPFFT_C2R;

  static constexpr auto FORWARD = HIPFFT_FORWARD;
  static constexpr auto BACKWARD = HIPFFT_BACKWARD;

  static constexpr auto FFT_SUCCESS = HIPFFT_SUCCESS;
  static constexpr auto API_SUCCESS = hipSuccess;

  template <typename... Args>
  static inline decltype(auto) SetDevice(Args&&... args) {
    return hipSetDevice(std::forward<Args>(args)...);
  }

  template <typename... Args>
  static inline decltype(auto) StreamSynchronize(Args&&... args) {
    return hipStreamSynchronize(std::forward<Args>(args)...);
  }

  template <typename... Args>
  static inline decltype(auto) fftCreate(Args&&... args) {
    return hipfftCreate(std::forward<Args>(args)...);
  }

  template <typename... Args>
  static inline decltype(auto) fftDestroy(Args&&... args) {
    return hipfftDestroy(std::forward<Args>(args)...);
  }

  template <typename... Args>
  static inline decltype(auto) fftMakePlan1d(Args&&... args) {
    return hipfftMakePlan1d(std::forward<Args>(args)...);
  }

  template <typename... Args>
  static inline decltype(auto) fftMakePlanMany64(Args&&... args) {
    return hipfftMakePlanMany64(std::forward<Args>(args)...);
  }

  template <typename... Args>
  static inline decltype(auto) fftSetStream(Args&&... args) {
    return hipfftSetStream(std::forward<Args>(args)...);
  }

  template <typename... Args>
  static inline decltype(auto) fftSetWorkArea(Args&&... args) {
    return hipfftSetWorkArea(std::forward<Args>(args)...);
  }

  template <typename... Args>
  static inline decltype(auto) fftSetAutoAllocation(Args&&... args) {
    return hipfftSetAutoAllocation(std::forward<Args>(args)...);
  }
};

template <>
struct cufft_like_trait<float, srtb::backend::rocm> : hipfft_common_trait {
  using real = hipfftReal;
  using complex = hipfftComplex;

  static inline constexpr auto get_native_fft_type(srtb::fft::type fft_type) {
    switch (fft_type) {
      case srtb::fft::type::C2C_1D_FORWARD:
      case srtb::fft::type::C2C_1D_BACKWARD:
        return HIPFFT_C2C;
      case srtb::fft::type::R2C_1D:
        return HIPFFT_R2C;
      case srtb::fft::type::C2R_1D:
        return HIPFFT_C2R;
    }
  }

  template <typename... Args>
  static inline decltype(auto) fftExecR2C(Args&&... args) {
    return hipfftExecR2C(std::forward<Args>(args)...);
  }

  template <typename... Args>
  static inline decltype(auto) fftExecC2C(Args&&... args) {
    return hipfftExecC2C(std::forward<Args>(args)...);
  }

  template <typename... Args>
  static inline decltype(auto) fftExecC2R(Args&&... args) {
    return hipfftExecC2R(std::forward<Args>(args)...);
  }
};

template <>
struct cufft_like_trait<double, srtb::backend::rocm> : hipfft_common_trait {
  using real = hipfftDoubleReal;
  using complex = hipfftDoubleComplex;

  static inline constexpr auto get_native_fft_type(srtb::fft::type fft_type) {
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

  template <typename... Args>
  static inline decltype(auto) fftExecR2C(Args&&... args) {
    return hipfftExecD2Z(std::forward<Args>(args)...);
  }

  template <typename... Args>
  static inline decltype(auto) fftExecC2C(Args&&... args) {
    return hipfftExecZ2Z(std::forward<Args>(args)...);
  }

  template <typename... Args>
  static inline decltype(auto) fftExecC2R(Args&&... args) {
    return hipfftExecZ2D(std::forward<Args>(args)...);
  }
};

template <std::floating_point T>
using hipfft_trait = cufft_like_trait<T, srtb::backend::rocm>;

template <srtb::fft::type fft_type, std::floating_point T,
          typename C = srtb::complex<T> >
using hipfft_1d_wrapper =
    cufft_like_1d_wrapper<srtb::backend::rocm, fft_type, T, C>;

}  // namespace fft
}  // namespace srtb

#endif  //  __SRTB_HIPFFT_WRAPPER__
