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

#include <climits>
#include <cmath>
#include <type_traits>

#include "srtb/config.hpp"
#include "srtb/sycl.hpp"

/** 
 * very naive SYCL FFT, currently Gauss-Cooley-Tukey algorithm
 * often slower than the serial one on CPU ... just used as a fallback though.
 */
namespace naive_fft {

namespace detail {

template <size_t bit_width>
struct builtin_bit_reverse {
  static constexpr bool available = false;
};


/**
 * @note not supported on: 
 *         * MUSA (1.4)
 *         * Rusticl (OpBitReverse, only 32bit working)
 */
#define SRTB_NAIVE_FFT_BUILTIN_BIT_REVERSE(bit)                 \
  template <>                                                   \
  struct builtin_bit_reverse<bit> {                             \
    static constexpr bool available = true;                     \
                                                                \
    template <typename T, typename = typename std::enable_if_t< \
                              sizeof(T) * CHAR_BIT == bit> >    \
    constexpr inline auto operator()(T x) const {               \
      return __builtin_bitreverse##bit(x);                      \
    }                                                           \
  };

#if __has_builtin(__builtin_bitreverse8)
SRTB_NAIVE_FFT_BUILTIN_BIT_REVERSE(8)
#endif

#if __has_builtin(__builtin_bitreverse16)
SRTB_NAIVE_FFT_BUILTIN_BIT_REVERSE(16)
#endif

#if __has_builtin(__builtin_bitreverse32)
SRTB_NAIVE_FFT_BUILTIN_BIT_REVERSE(32)
#endif

#if __has_builtin(__builtin_bitreverse64)
SRTB_NAIVE_FFT_BUILTIN_BIT_REVERSE(64)
#endif

#undef SRTB_NAIVE_FFT_BUILTIN_BIT_REVERSE

}  // namespace detail

template <typename T>
inline T bit_reverse(T x, size_t k) {
  constexpr auto bit_width = sizeof(T) * CHAR_BIT;
  using builtin_bit_reverse = detail::builtin_bit_reverse<bit_width>;
  if constexpr (builtin_bit_reverse::available) {
    constexpr builtin_bit_reverse functor;
    return functor(x) >> (bit_width - k);
  } else {
    T y = 0;
    while (k--) {
      y <<= 1;
      y |= x & 1;
      x >>= 1;
    }
    return y;
  }
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
 * @param k s.t. n == 2**k, n is size of input real numbers
 * @param input Accessor or pointer or something like that of input buffer
 * @param output Accessor or pointer or something like that of output buffer
 *
 * ref: https://www.cnblogs.com/liam-ji/p/11742941.html
 *      http://www.dspguide.com/ch12/5.htm
 * 
 * (Translated from 1st reference)
 * 
 * Suppose $ x(n) $ is a real sequence of length $ 2N $ whose discrete Fourier transform is
 * 
 * $$ X(k)=\sum_{n=0}^{2N-1}x(n)W_{2N}^{nk} \ , \ k=0,1,...,2N-1  $$
 * 
 * To efficiently compute the Fourier transform $ X(k) $ , we divide $ x(n) $ into even and odd groups, forming two new sequences $ x(n) $ and $ g(n) $ , i.e
 * 
 * $$ \left\{\begin{matrix}\begin{align*}f(n)&=x(2n)\\ g(n)&=x(2n+1)\end{align*}\end{matrix}\right. , n=0,1,...,N-1  $$
 * 
 * Then $ f(n) $ and $ g(n) $ form a complex sequence $ h(n) $ 
 * 
 * $$ h(n)=f(n)+jg(n), \ n = 0,1,...,N-1  $$
 * 
 * Using FFT to calculate $ h (n) $$ n $ point Fourier transform $ h (k) $ , and $ h (k) $ can be expressed as
 * 
 * $$ H(k)=F(k)+jG(k), \ n = 0,1,...,N-1  $$
 * 
 * Easy to derive from above
 * 
 * $$ \left\{\begin{matrix}\begin{align*}F(k)&=\frac{1}{2}[H(k)+H^{*}(N-k)]\\ G(k)&=-\frac{j}{2}[H(k)-H^{*}(N-k)]\end{align*}\end{matrix}\right. , n=0,1,...,N-1  $$
 * 
 * After obtaining $ F(k) $ and $ G(k) $ , the discrete Fourier transform $ x(k) $ of $ x(n) $ is computed using the following butterfly operation
 * 
 * $$ \left\{\begin{matrix}\begin{align*}X(k)&=F(k)+G(k)W_{2N}^{k}\\ X(k+n)&=F(k)-G(k)W_{2N}^{k}\end{align*}\end{matrix}\right. , n=0,1,...,N-1  $$
 * 
 * The real sequence FFT algorithm can reduce about half of the computation compared with the complex sequence FFT algorithm of the same length.
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
  // operate in-place
  const auto H = output;
  q.parallel_for(sycl::range{N / 2 + 1}, [=](sycl::item<1> id) {
     const size_t k = id.get_id(0);
     const C H_k = H[k];
     const C H_N_k = ((k == 0) ? (H[0]) : (H[N - k]));

     //const C H_k_conj = srtb::conj(H_k);
     const C H_N_k_conj = srtb::conj(H_N_k);
     const C F_k = (H_k + H_N_k_conj) / T{2};
     const C G_k = (H_k - H_N_k_conj) * (-C{0, 1} / T{2});
     //const C F_N_k = (H_N_k + H_k_conj) / T{2};
     //const C G_N_k = (H_N_k - H_k_conj) * (-C{0, 1} / T{2});
     const C F_N_k = srtb::conj(F_k);
     const C G_N_k = srtb::conj(G_k);

     //const T theta = -T{2.0 * M_PI} * k / n_real;
     const T theta_k = -T{M_PI} * k / N;
     T w_k_re, w_k_im;
     w_k_im = sycl::sincos(theta_k, sycl::decorated_private_ptr<T>{&w_k_re});
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
