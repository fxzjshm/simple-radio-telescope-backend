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

#include <concepts>
#include <mutex>

#include "srtb/commons.hpp"

namespace srtb {
namespace fft {

/**
 * @brief Abstract interface of backend-specific FFTs. Should be called within one thread.
 * 
 * @tparam T Data type
 * @tparam Derived CRTP requirement.
 * @tparam C Complex type of T, default to srtb::complex<T>
 */
template <template <typename, typename> class Derived, std::floating_point T,
          typename C = srtb::complex<T> >
class fft_wrapper {
 public:
  using sub_class = Derived<T, C>;
  static_assert(sizeof(T) * 2 == sizeof(C));

  fft_wrapper() { create(); }

  ~fft_wrapper() { destroy(); }

  sub_class& sub() { return static_cast<sub_class&>(*this); }

  friend sub_class;

  bool has_inited() { return sub().has_inited_impl(); }

  void process(T* in, C* out) { sub().process_impl(in, out); }

  void update_config() {
    sub().destroy_impl();
    sub().create_impl();
  }

  void set_queue(sycl::queue& queue) { sub().set_queue_impl(queue); }

 private:
  void create(size_t n = srtb::config.unpacked_input_count()) {
    sub().create_impl(n);
  }

  void destroy() {
    if (has_inited()) [[likely]] {
      sub().destroy_impl();
    }
  }
};

}  // namespace fft
}  // namespace srtb

#endif  //  __SRTB_FFT_WRAPPER__
