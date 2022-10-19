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

template <typename T = srtb::real>
struct rectangle {
  /**
   * @brief coefficients of rectangle window is just 1
   */
  T operator()(const T& x) const noexcept {
    (void)x;
    return T(1);
  }
};

}  // namespace window

using default_window = srtb::fft::window::hamming<>;

/**
 * @brief Provide FFT window coefficients for size n, that is, [0, n-1]
 * 
 * @tparam T real type
 * @tparam Iterator == T* if pre-computed window, or TransformIterator if real-time computed window.
 */
template <typename T, typename Iterator>
struct fft_window_functor {
  size_t n;
  Iterator coefficients;

  fft_window_functor(size_t n_, Iterator coefficients_)
      : n{n_}, coefficients{coefficients_} {
    assert(coefficients != nullptr);
  }

  T operator()(size_t pos, T val) const noexcept {
    SRTB_ASSERT_IN_KERNEL(pos < n);
    return val * coefficients[pos];
  }
};

/**
 * @brief compute window value at i-th point of total n points
 */
template <typename T, typename Window>
struct fft_window_iterator {
  size_t n;
  Window window;

  fft_window_iterator(const size_t n_, Window window_)
      : n{n_}, window{window_} {}

  T operator[](const size_t i) const noexcept {
    return window(static_cast<T>(i) / (n - 1));
  }
};

/**
 * @brief This class manages memory used by @c fft_window_functor at host,
 *        avoiding `sycl::is_device_copyable` problems.
 * @see srtb::fft::fft_window_functor
 */
template <typename T, typename Window,
          bool precomputed = srtb::fft_window_precompute>
class fft_window_functor_manager;

/**
 * @brief implementation using pre-computed coefficients.
 *        More VRAM usage, (maybe) less computation.
 * @note template parameter "Window" doesn't really matters
 *       as in this implementation it can be dynamically set.
 */
template <typename T, typename Window>
class fft_window_functor_manager<T, Window, true> {
 public:
  using Iterator = T*;

 protected:
  size_t n;
  std::shared_ptr<T> shared_coefficients;

 public:
  fft_window_functor<T, Iterator> functor;

 public:
  template <typename TrueWindow>
  fft_window_functor_manager(const TrueWindow& window, size_t n_,
                             sycl::queue& q = srtb::queue)
      : n{n_},
        shared_coefficients{srtb::device_allocator.allocate_shared<T>(n_)},
        functor{n_, shared_coefficients.get()} {
    static_assert(std::is_convertible_v<decltype(window(T{0})), T>);
    set_value(window, q);
  }

 protected:
  template <typename TrueWindow>
  void set_value(const TrueWindow& window, sycl::queue& q) {
    static_assert(std::is_convertible_v<decltype(window(T{0})), T>);
    T* _coefficients = shared_coefficients.get();
    const size_t _n = n;
    fft_window_iterator<T, TrueWindow> iterator{_n, window};
    q.parallel_for(sycl::range<1>(n), [=](sycl::item<1> id) {
       const auto i = id.get_id(0);
       _coefficients[i] = iterator[i];
     }).wait();
  }

 public:
  Iterator get_coefficients() { return shared_coefficients.get(); }
};

// ------------

template <typename T, typename Window>
class fft_window_functor_manager<T, Window, false> {
 public:
  using Iterator = fft_window_iterator<T, Window>;

 protected:
  size_t n;
  Iterator coefficients;

 public:
  fft_window_functor<T, Iterator> functor;

 public:
  fft_window_functor_manager(Window window, size_t n_,
                             sycl::queue& q = srtb::queue)
      : n{n_}, coefficients{n_, window}, functor{n_, coefficients} {
    (void)q;
    static_assert(std::is_convertible_v<decltype(window(T{0})), T>);
  }

  Iterator get_coefficients() { return coefficients; }
};

}  // namespace fft
}  // namespace srtb

#endif  // __SRTB_FFT_WINDOW__
