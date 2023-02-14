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
#ifndef __SRTB_MEMORY_MEMCPY__
#define __SRTB_MEMORY_MEMCPY__

#include <cstddef>
#include <type_traits>

#ifndef SRTB_USE_CUSTOM_MEMCPY
// it seems that glibc introduces first AVX-based memcpy/memmove for x86 based machine on glibc 2.20
//     https://sourceware.org/git/gitweb.cgi?p=glibc.git;a=commitdiff;h=05f3633da4f9df870d04dd77336e793746e57ed4
//         (from https://www.zhihu.com/question/35172305/answer/73698602)
// but some machines use server processors launched in 2021 w/ CentOS 7 (+ glibc 2.17)
// so even auto-vectorized naive memcpy seems faster...
#ifdef __GLIBC__
#if ((__GLIBC__ < 2) || (__GLIBC__ == 2 && __GLIBC_MINOR__ < 20)) && \
    defined(__AVX__)
#define SRTB_USE_CUSTOM_MEMCPY
#endif
#endif
#endif

namespace srtb {
namespace memory {

/** @deprecated need further profiling */
template <typename T = std::byte,
          typename = typename std::enable_if_t<(sizeof(T) == 1), void> >
inline auto memcpy(T* dest, const T* src, size_t size) -> T* {
#ifdef SRTB_USE_CUSTOM_MEMCPY
#pragma omp simd
  for (size_t i = 0; i < size; i++) {
    dest[i] = src[i];
  }
  return dest;
#else
  return reinterpret_cast<T*>(std::memcpy(dest, src, size));
#endif
}

}  // namespace memory
}  // namespace srtb

#endif  // __SRTB_MEMORY_MEMCPY__
