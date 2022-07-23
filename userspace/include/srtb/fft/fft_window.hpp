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
#ifndef __SRTB_FFT_WINDOW__
#define __SRTB_FFT_WINDOW__

#include <cmath>
#include <memory>

#include "srtb/commons.hpp"

namespace srtb {
namespace fft {

namespace window {

template <size_t K, typename T = srtb::real>
class cosine_sum_window {
 protected:
  std::array<T, K> a;

 public:
  template <typename... Args>
  cosine_sum_window(Args... args) : a{args...} {}

  /**
 * @brief calculate window funtion value at x = n / N for 0 <= n <= N
 * 
 * @param x x = n / N, 0 <= n <= N
 * @return T window funtion value at x
 */
  T operator()(const T& x) const noexcept {
    T ret = 0;
    for (size_t k = 0; k < K; k++) {
      T sign = static_cast<T>(((k & 1) == 0) ? (1) : (-1));
      ret += sign * a[k] * sycl::cos(2 * M_PI * k * x);
    }
    return ret;
  }
};

template <typename T = srtb::real>
struct hann : cosine_sum_window<2, T> {
  hann() : cosine_sum_window<2, T>{T{0.5}, T{0.5}} {}
  using cosine_sum_window<2, T>::operator();
};

template <typename T = srtb::real>
struct hamming : cosine_sum_window<2, T> {
  hamming() : cosine_sum_window<2, T>{T{25.0 / 46.0}, T{21.0 / 46.0}} {}
  using cosine_sum_window<2, T>::operator();
};

}  // namespace window

/**
 * @brief Provide FFT window coefficients for size n, that is, [0, n-1]
 * 
 * @tparam T real type, default to @c srtb::real
 * @tparam Window type of a window funtor, default to @c srtb::fft::window::hann
 */
template <typename T = srtb::real>
struct fft_window_functor {
  size_t n;
  T* coefficients;

  fft_window_functor(size_t n_, T* coefficients_)
      : n{n_}, coefficients{coefficients_} {
    assert(coefficients != nullptr);
  }

  T operator()(size_t pos, T val) const noexcept {
// TODO: ref: https://github.com/intel/llvm/pull/6424
#if !defined(SRTB_ENABLE_ROCM_INTEROP) && !defined(SRTB_ENABLE_CUDA_INTEROP)
    assert(pos < n);
#endif
    return val * coefficients[pos];
  }
};

/**
 * @brief This class manages memory used by @c fft_window_functor at host,
 *        avoiding `is_device_copyable` problems.
 * @see srtb::fft::fft_window_functor
 */
template <typename T = srtb::real>
class fft_window_functor_manager {
 protected:
  size_t n;
  std::shared_ptr<T> shared_coefficients;

 public:
  fft_window_functor<T> functor;

 public:
  template <typename Window>
  fft_window_functor_manager(const Window& window, size_t n_,
                             sycl::queue& q = srtb::queue)
      : n{n_},
        shared_coefficients{srtb::device_allocator.allocate_shared<T>(n_)},
        functor{n_, shared_coefficients.get()} {
    static_assert(std::is_convertible_v<decltype(window(T{0})), T>);
    set_value(window, q);
  }

  template <typename Window>
  void set_value(const Window& window, sycl::queue& q) {
    static_assert(std::is_convertible_v<decltype(window(T{0})), T>);
    T* _coefficients = shared_coefficients.get();
    const size_t _n = n;
    q.parallel_for(sycl::range<1>(n), [=](sycl::item<1> id) {
       const auto i = id.get_id(0);
       _coefficients[i] = window(static_cast<T>(i) / (_n - 1));
     }).wait();
  }
};

}  // namespace fft
}  // namespace srtb

#endif  // __SRTB_FFT_WINDOW__
