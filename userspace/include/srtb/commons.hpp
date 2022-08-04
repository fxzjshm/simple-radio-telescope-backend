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

// TODO: ref: https://github.com/intel/llvm/pull/6424
#if !defined(SRTB_ENABLE_ROCM_INTEROP) && !defined(SRTB_ENABLE_CUDA_INTEROP)
#define SRTB_ASSERT_IN_KERNEL(expr) assert(expr)
#else
#define SRTB_ASSERT_IN_KERNEL(expr)
#endif

namespace srtb {

template <typename T = srtb::real, typename C = srtb::complex<T> >
inline constexpr auto norm(const C& c) noexcept -> T {
  return c.real() * c.real() + c.imag() * c.imag();
}

template <typename T = srtb::real, typename C = srtb::complex<T> >
inline constexpr auto abs(const C& c) noexcept -> T {
  return sycl::sqrt(srtb::norm(c));
}

}  // namespace srtb

// ------ dividing line for clang-format ------

#include "srtb/global_variables.hpp"
#include "srtb/log/log.hpp"

#endif  // __SRTB_COMMONS__
