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
#ifndef __SRTB_FFT_WRAPPER__
#define __SRTB_FFT_WRAPPER__

#include <complex>
#include <concepts>
#include <mutex>

#include "srtb/commons.hpp"

namespace srtb {
namespace fft {

/**
 * @brief As functions beside `fftw_execute` are not thread-safe, lock this mutex
 *        when calling other functions
 * @see https://github.com/wenyan4work/SafeFFT
 */
inline std::mutex fft_mutex;

/**
 * @brief Abstract interface of backend-specific FFTs.
 * 
 * @tparam T Data type
 * @tparam Derived CRTP requirement.
 * @tparam C Complex type of T, default to std::complex<T>
 */
template <template <typename, typename> class Derived, std::floating_point T,
          typename C = std::complex<T> >
class fft_wrapper {
 public:
  using sub_class = Derived<T, C>;

  void create(size_t n = srtb::config.unpacked_input_count()) {
    std::lock_guard lock{fft_mutex};
    static_cast<sub_class&>(*this).create_impl(n);
  }

  void destroy() {
    std::lock_guard lock{fft_mutex};
    static_cast<sub_class&>(*this).destroy_impl();
  }

  void process(T* in, C* out) {
    //std::lock_guard lock{fft_mutex};
    static_cast<sub_class&>(*this).process_impl(in, out);
  }

  void update_config() {
    // a lock may be required so that a call to process() won't be inserted between
    std::lock_guard lock{fft_mutex};
    static_cast<sub_class&>(*this).destroy_impl();
    static_cast<sub_class&>(*this).create_impl();
  }

  friend sub_class;

 private:
  fft_wrapper() { create(); }

  ~fft_wrapper() { destroy(); }
};

}  // namespace fft
}  // namespace srtb

#endif  //  __SRTB_FFT_WRAPPER__
