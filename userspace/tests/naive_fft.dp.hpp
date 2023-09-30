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

#include <cstddef>

#include <sycl/sycl.hpp>

#include "sycl_ext_complex.dp.hpp"

/** 
 * very naive SYCL FFT, currently Gauss-Cooley-Tukey algorithm
 * often slower than the serial one on CPU ... just used as a fallback though.
 */
namespace naive_fft {

template <typename T>
inline T bit_reverse(T x, size_t k) {
  T y = 0;
  while (k--) {
    y <<= 1;
    y |= x & 1;
    x >>= 1;
  }
  return y;
}

template <typename InputAccessor, typename OutputAccessor>
inline void bit_reverse_swap(const size_t k, const size_t i,
                             InputAccessor input,
                             OutputAccessor output) noexcept {
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
template <typename T, typename C, typename Accessor>
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
      -T{2.0 * M_PI} * butterfly_local_id / butterfly_size * direction;
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
 * @param direction 1 -> forward, -1 -> backward
 */
template <typename T, typename C, typename InputAccessor,
          typename OutputAccessor>
inline void fft_1d_c2c(const size_t k, InputAccessor input,
                       OutputAccessor output, const int direction,
                       sycl::queue& q) {
  const size_t n = 1 << k;
  q.parallel_for(sycl::range{n}, [=](sycl::item<1> id) {
     const size_t i = id.get_id(0);
     bit_reverse_swap(k, i, input, output);
   }).wait();

  for (unsigned int m = 0; m < k; ++m) {
    // 2**m is the half-size of the butterfly
    // NOTE: the size of range is n/2 as every thread do 2 points (output[x] and output[y])
    q.parallel_for(sycl::range{n / 2}, [=](sycl::item<1> id) {
       const size_t i = id.get_id(0);
       fft_1d_c2c_butterfly<T, C, OutputAccessor>(m, i, output, direction);
     }).wait();
  }

  // normalization is removed to stay in sync with FFTW, cuFFT & hipFFT
}

template <typename T, typename C>
inline constexpr auto reinterpret_as_complex(T* x) -> C* {
  return reinterpret_cast<C*>(x);
}

/**
 * @param n size of input real numbers
 * @param k s.t. n == 2**k
 * @param i thread index, 0 <= i < n / 2
 * @param input Accessor or pointer or something like that of input buffer
 * @param output Accessor or pointer or something like that of output buffer
 * 
 * ref: https://www.cnblogs.com/liam-ji/p/11742941.html
 *      http://www.dspguide.com/ch12/5.htm
 */
template <typename T, typename C, typename InputAccessor,
          typename OutputAccessor>
inline void fft_1d_r2c(const size_t k, InputAccessor input,
                       OutputAccessor output, sycl::queue& q) {
  const size_t n_real = 1 << k;
  //const size_t n_complex = n_real / 2 + 1;
  const size_t N = n_real / 2;
  const auto input_as_complex = reinterpret_as_complex<T, C>(input);
  fft_1d_c2c<T, C>(k - 1, input_as_complex, output, +1, q);
  const auto H = output;
  q.parallel_for(sycl::range{N / 2 + 1}, [=](sycl::item<1> id) {
     const size_t k = id.get_id(0);
     const C H_k = H[k];
     const C H_N_k = ((k == 0) ? (H[0]) : (H[N - k]));

     //const C H_k_conj = _SYCL_CPLX_NAMESPACE::conj(H_k);
     const C H_N_k_conj = _SYCL_CPLX_NAMESPACE::conj(H_N_k);
     const C F_k = (H_k + H_N_k_conj) / T{2};
     const C G_k = (H_k - H_N_k_conj) * (-C{0, 1} / T{2});
     //const C F_N_k = (H_N_k + H_k_conj) / T{2};
     //const C G_N_k = (H_N_k - H_k_conj) * (-C{0, 1} / T{2});
     const C F_N_k = _SYCL_CPLX_NAMESPACE::conj(F_k);
     const C G_N_k = _SYCL_CPLX_NAMESPACE::conj(G_k);

     //const T theta = -T{2.0 * M_PI} * k / n_real;
     const T theta_k = -T{M_PI} * k / N;
     const T w_k_re = sycl::cos(theta_k), w_k_im = sycl::sin(theta_k);
     const C w_k = C{w_k_re, w_k_im};
     //const T theta_N_k = -T{M_PI} * (N - k) / N;
     //const T w_N_k_re = sycl::cos(theta_N_k), w_N_k_im = sycl::sin(theta_N_k);
     //const C w_N_k = C{w_N_k_re, w_N_k_im};
     const C w_N_k = C{-w_k_re, w_k_im};

     const C X_k = F_k + G_k * w_k;
     const C X_N_k = F_N_k + G_N_k * w_N_k;
     output[k] = X_k;
     output[N - k] = X_N_k;
     // can prove X_N also satisfies this formula
     // as F_0 and G_0 is real
   }).wait();
}

}  // namespace naive_fft

#endif  // __SRTB_NAIVE_FFT__
