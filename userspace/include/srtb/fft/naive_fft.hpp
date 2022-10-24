/******************************************************************************* 
 * Copyright (c) 2022 fxzjshm
 * This file is licensed under Mulan PubL v2.
 * You can use this software according to the terms and conditions of the Mulan PubL v2.
 * You may obtain a copy of Mulan PubL v2 at:
 *          http://license.coscl.org.cn/MulanPubL-2.0
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PubL v2 for more details.
 ******************************************************************************/

#pragma once
#ifndef __SRTB_NAIVE_FFT__
#define __SRTB_NAIVE_FFT__

#include <cmath>
#include <concepts>

#include "srtb/config.hpp"
#include "srtb/sycl.hpp"

/** 
 * very naive SYCL FFT, currently Gauss-Cooley-Tukey algorithm
 * often slower than the serial one on CPU ... just used as a fallback though.
 */
namespace naive_fft {

template <std::integral T>
inline T bit_reverse(T x, size_t k) {
  T y = 0;
  while (k--) {
    y <<= 1;
    y |= x & 1;
    x >>= 1;
  }
  return y;
}

template <typename Accessor>
inline void bit_reverse_swap(const size_t k, const size_t i, Accessor input,
                             Accessor output) noexcept {
  const size_t j = bit_reverse(i, k);

  if (i <= j) {
    const auto x = input[i], y = input[j];
    output[i] = y;
    output[j] = x;
  }

  //output[j] = input[i];
}

/**
 * @param n size of the FFT
 * @param k s.t. n == 2**k
 * @param m 2**m is the current half-size of the butterfly
 * @param i thread index
 * @param input Accessor or pointer or something like that of input buffer
 * @param output Accessor or pointer or something like that of output buffer
 * @param invert 1 -> forward, -1 -> backward
 */
template <typename T, typename C = srtb::complex<T>, typename Accessor>
inline void fft_1d_c2c_butterfly(const size_t m, const size_t i,
                                 Accessor output,
                                 const int direction) noexcept {
  /*
  using namespace std::complex_literals;
  const size_t butterfly_size = 1 << (m+1);
  const size_t butterfly_group_id = i / (butterfly_size / 2);
  const size_t butterfly_local_id = i % (butterfly_size / 2);
  const size_t x = butterfly_group_id * butterfly_size + butterfly_local_id;
  const size_t y = x + (butterfly_size / 2);
  const C w = std::exp(2.0 * M_PI * butterfly_local_id / butterfly_size *
                       direction * 1.0i);
  */
  const size_t butterfly_size = 1 << (m + 1);
  const size_t butterfly_group_id = i >> m;
  const size_t butterfly_local_id = i - (butterfly_group_id << m);
  const size_t x = butterfly_group_id * butterfly_size + butterfly_local_id;
  const size_t y = x + (butterfly_size / 2);
  const T theta =
      -T(2.0) * M_PI * butterfly_local_id / butterfly_size * direction;
  const T w_re = sycl::cos(theta), w_im = sycl::sin(theta);
  const C w = C(w_re, w_im);
  //assert(x < n);
  //assert(y < n);
  const C c_x = output[x], c_y = output[y];
  output[x] = c_x + w * c_y;
  output[y] = c_x - w * c_y;
}

/**
 * @param n size of the FFT
 * @param k s.t. n == 2**k
 * @param i thread index, 0 <= i < n / 2
 * @param input Accessor or pointer or something like that of input buffer
 * @param output Accessor or pointer or something like that of output buffer
 * @param invert 1 -> forward, -1 -> backward
 */
template <typename T, typename C = srtb::complex<T>, typename InputAccessor,
          typename OutputAccessor>
inline void fft_1d_c2c(const size_t k, sycl::queue& q, InputAccessor input,
                       OutputAccessor output, const int direction) {
  const size_t n = 1 << k;
  q.parallel_for(sycl::range{n}, [=](sycl::item<1> id) {
     const size_t i = id.get_id(0);
     bit_reverse_swap(k, i, input, output);
   }).wait();

  for (uint m = 0; m < k; ++m) {
    // 2**m is the half-size of the butterfly
    // NOTE: the size of range is n/2 as every thread do 2 points (output[x] and output[y])
    q.parallel_for(sycl::range{n / 2}, [=](sycl::item<1> id) {
       const size_t i = id.get_id(0);
       fft_1d_c2c_butterfly<T, C, OutputAccessor>(m, i, output, direction);
     }).wait();
  }

  if (direction < 0) {
    q.parallel_for(sycl::range{n}, [=](sycl::item<1> id) {
       output[id.get_id(0)] /= T(n);
     }).wait();
  }
}

// TODO: 1D R2C FFT

}  // namespace naive_fft

#endif  // __SRTB_NAIVE_FFT__
