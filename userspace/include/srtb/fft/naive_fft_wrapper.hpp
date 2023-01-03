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
#ifndef __SRTB_NAIVE_FFT_WRAPPER__
#define __SRTB_NAIVE_FFT_WRAPPER__

#include <bit>

#include "srtb/fft/naive_fft.hpp"

namespace srtb {
namespace fft {

template <srtb::fft::type fft_type, std::floating_point T, typename C>
class naive_fft_1d_wrapper
    : public fft_wrapper<naive_fft_1d_wrapper<fft_type, T, C>, fft_type, T, C> {
 public:
  using super_class = fft_wrapper<naive_fft_1d_wrapper, fft_type, T, C>;
  friend super_class;

 protected:
  size_t k;  // should be 2**k = n

 public:
  naive_fft_1d_wrapper(size_t n, size_t batch_size, sycl::queue& queue)
      : super_class{n, batch_size, queue} {}

 protected:
  void create_impl(size_t n, size_t batch_size, sycl::queue& queue) {
    (void)batch_size;
    (void)queue;
    k = std::bit_width(n);
    // delay error on n != 2**k because this wrapper may not be called
    // as it is naive and slow
  }

  void destroy_impl() {}

  bool has_inited_impl() { return true; }

  // SFINAE ref: https://stackoverflow.com/a/50714150
  template <typename..., srtb::fft::type fft_type_ = fft_type,
            typename std::enable_if<(fft_type_ == srtb::fft::type::R2C_1D),
                                    int>::type = 0>
  void process_impl(T* in, C* out) {
    (void)in;
    (void)out;
    throw std::runtime_error("[naive_fft_wrapper]: R2C TODO");
  }

  template <
      typename..., srtb::fft::type fft_type_ = fft_type,
      typename std::enable_if<(fft_type_ == srtb::fft::type::C2C_1D_FORWARD ||
                               fft_type_ == srtb::fft::type::C2C_1D_BACKWARD),
                              int>::type = 0>
  void process_impl(C* in, C* out) {
    const size_t n = (*this).n;
    if (std::has_single_bit(n)) {
      throw std::runtime_error("[naive_fft_wrapper]: n must be a power of 2");
    }
    constexpr int direction =
        (fft_type == srtb::fft::type::C2C_1D_BACKWARD) ? -1 : +1;
    for (size_t i = 0; i < (*this).batch_size; i++) {
      naive_fft::fft_1d_c2c<T>(k, (*this).q, in + i * n, out + i * n,
                               direction);
    }
  }

  template <typename..., srtb::fft::type fft_type_ = fft_type,
            typename std::enable_if<(fft_type_ == srtb::fft::type::C2R_1D),
                                    int>::type = 0>
  void process_impl(C* in, T* out) {
    (void)in;
    (void)out;
    throw std::runtime_error("[naive_fft_wrapper]: C2R TODO");
  }
};

}  // namespace fft
}  // namespace srtb

#endif  //  __SRTB_NAIVE_FFT_WRAPPER__
