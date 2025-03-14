/******************************************************************************* 
 * Copyright (c) 2023 fxzjshm
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
#ifndef __SRTB_MATH__
#define __SRTB_MATH__

#include <type_traits>

#include "srtb/sycl.hpp"
// ---
#include "dsmath_sycl.h"

// defined _SYCL_CPLX_NAMESPACE to avoid namespace conflict
#define _SYCL_CPLX_NAMESPACE argonne_lcf::sycl_cplx
#include "sycl_ext_complex.hpp"

/**
 * @brief Shortcut for namespace of complex type.
 * @see srtb::math::complex
 */
namespace akira {

template <typename T>
using complex = _SYCL_CPLX_NAMESPACE::complex<T>;

}  // namespace akira

namespace srtb {

inline namespace math {

// maybe double on CPU,
//       float on most GPUs, even most professional cards of SOME VENDOR.
using real = float;

/**
 * @brief In memory of Akira Complex, "My Guiding Star".
 */
template <typename T>
using complex = akira::complex<T>;

template <typename C>
concept complex_like = requires(C c) {
  c.real();
  c.imag();
};

template <complex_like C>
inline constexpr auto norm(const C c) noexcept {
  using T = decltype(c.real());
  if constexpr (std::is_same<C, _SYCL_CPLX_NAMESPACE::complex<T> >::value) {
    return _SYCL_CPLX_NAMESPACE::norm(c);
  } else {
    return c.real() * c.real() + c.imag() * c.imag();
  }
}

template <complex_like C>
inline constexpr auto conj(const C c) noexcept -> C {
  using T = decltype(c.real());
  if constexpr (std::is_same<C, _SYCL_CPLX_NAMESPACE::complex<T> >::value) {
    return _SYCL_CPLX_NAMESPACE::conj(c);
  } else {
    return C{c.real(), -c.imag()};
  }
}

template <complex_like C>
inline constexpr auto abs(const C c) noexcept {
  using T = decltype(c.real());
  if constexpr (std::is_same<C, _SYCL_CPLX_NAMESPACE::complex<T> >::value) {
    return _SYCL_CPLX_NAMESPACE::abs(c);
  } else {
    return sycl::hypot(c.real(), c.imag());
  }
}

template <typename T>
inline constexpr T abs(const T x) noexcept {
  if constexpr (std::is_integral_v<T>) {
    return sycl::abs(x);
  } else if constexpr (std::is_floating_point_v<T>) {
    return sycl::fabs(x);
  }
}

template <typename T>
inline constexpr T modf(const T a, T* iptr) noexcept {
  if constexpr (std::is_floating_point_v<T>) {
    return sycl::modf(a, sycl::decorated_private_ptr<T>{iptr});
  } else if constexpr (std::is_same_v<T, dsmath::df64>) {
    float xi, xf, yi, yf;
    xf = sycl::modf(a.x, sycl::decorated_private_ptr<float>{&xi});
    yf = sycl::modf(a.y, sycl::decorated_private_ptr<float>{&yi});
    dsmath::df64 i, f;
    i = dsmath::df64{xi} + dsmath::df64{yi};
    f = dsmath::df64{xf} + dsmath::df64{yf};
    // assume f in [-1.0, 1.0] ?
    constexpr bool assume_range = true;
    constexpr dsmath::df64 one = 1.0f;
    if constexpr (assume_range) {
      if (static_cast<float>(a) > 0.0f) {
        // f should in [0.0, 1.0]
        if (f.x + f.y < 0.0f) {
          f = f + one;
          i = i - one;
        }
      }
      if (static_cast<float>(a) < 0.0f) {
        // f should in [-1.0, 0.0]
        if (f.x + f.y > 0.0f) {
          f = f - one;
          i = i + one;
        }
      }
    } else {
      // not assuming f in [-1.0, 1.0], should in [-2.0, 2.0]
      if (static_cast<float>(a) > 0.0f) {
        // f should in [0.0, 1.0]
        while (f.x + f.y < 0.0f) {
          f = f + one;
          i = i - one;
        }
        while (f.x + f.y > 1.0f) {
          f = f - one;
          i = i + one;
        }
      }
      if (static_cast<float>(a) < 0.0f) {
        // f should in [-1.0, 0.0]
        while (f.x + f.y < -1.0f) {
          f = f + one;
          i = i - one;
        }
        while (f.x + f.y > 0.0f) {
          f = f - one;
          i = i + one;
        }
      }
    }
    *iptr = i;
    return f;
  }
}

}  // namespace math

}  // namespace srtb

#endif  // __SRTB_MATH__
