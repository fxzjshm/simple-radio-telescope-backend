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
#ifndef __SRTB_COMMONS__
#define __SRTB_COMMONS__

/**
 * This file should contain commonly included headers, and forward
 * declaration if needed.
 */

#if defined(__clang__) || defined(__GNUC__) || defined(_MSC_VER)
#define SRTB_HAS_CONSTEXPR_UNMANGLED_TYPE_NAME
#include "unmangled_type_name.hpp"
#else
#include <boost/core/demangle.hpp>
#endif

#include "dsmath_sycl.h"
#include "srtb/sycl.hpp"

// ------ dividing line for clang-format ------

#include "srtb/config.hpp"

#define SRTB_CHECK(expr, expected, handle) \
  {                                        \
    auto ret = expr;                       \
    if (ret != expected) [[unlikely]] {    \
      handle;                              \
    }                                      \
  }

// TODO: ROCm: ref: https://github.com/intel/llvm/pull/6424
//       CUDA: an ptxas "syntax error" encountered if enabled
#if !defined(SRTB_ENABLE_ROCM_INTEROP) && !defined(SRTB_ENABLE_CUDA_INTEROP)
#define SRTB_ASSERT_IN_KERNEL(expr) assert(expr)
#else
#define SRTB_ASSERT_IN_KERNEL(expr)
#endif

namespace srtb {

template <typename T>
inline constexpr auto norm(const srtb::complex<T> c) noexcept -> T {
  return c.real() * c.real() + c.imag() * c.imag();
}

template <typename T>
inline constexpr auto conj(const srtb::complex<T> c) noexcept
    -> srtb::complex<T> {
  return srtb::complex<T>{c.real(), -c.imag()};
}

template <typename T>
inline constexpr auto abs(const srtb::complex<T> c) noexcept -> T {
  return sycl::sqrt(srtb::norm(c));
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
inline constexpr T modf(const T x, T* iptr) noexcept {
  if constexpr (std::is_floating_point_v<T>) {
    return sycl::modf(x, sycl::private_ptr<T>{iptr});
  } else if constexpr (std::is_same_v<T, dsmath::df64>) {
    float xi, xf, yi, yf;
    xf = sycl::modf(x.x, sycl::private_ptr<float>{&xi});
    yf = sycl::modf(x.y, sycl::private_ptr<float>{&yi});
    dsmath::df64 i, f;
    i = dsmath::df64{xi} + dsmath::df64{yi};
    f = dsmath::df64{xf} + dsmath::df64{yf};
    constexpr dsmath::df64 one = 1.0f;
    if (f.x > 1.0f) {
      f = f - one;
      i = i + one;
    }
    *iptr = i;
    return f;
  }
}

#ifdef SRTB_HAS_CONSTEXPR_UNMANGLED_TYPE_NAME
template <typename T>
static inline constexpr auto type_name() {
  return unmangled_type_name::type_name<T>();
}
#else
template <typename T>
static inline auto type_name() {
  return boost::core::demangle(typeid(T).name());
}
#endif  // SRTB_HAS_CONSTEXPR_UNMANGLED_TYPE_NAME

}  // namespace srtb

// ------ dividing line for clang-format ------

#include "srtb/global_variables.hpp"
#include "srtb/log/log.hpp"
#include "srtb/termination_handler.hpp"

#endif  // __SRTB_COMMONS__
