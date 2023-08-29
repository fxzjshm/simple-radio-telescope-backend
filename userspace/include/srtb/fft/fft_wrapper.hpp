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

#include "srtb/commons.hpp"

namespace srtb {
namespace fft {

enum class type {
  C2C_1D_FORWARD,
  C2C_1D_BACKWARD,
  R2C_1D,
  R2C_1D_FORWARD = R2C_1D,
  C2R_1D,
  C2R_1D_BACKWARD = C2R_1D
};

/**
 * @brief Abstract interface of backend-specific FFTs. Should be called within one thread.
 * 
 * @tparam Derived CRTP requirement.
 * @tparam fft_type type of FFT, see @c srtb::fft::type
 * @tparam T Data type
 * @tparam C Complex type of T, default to srtb::complex<T>
 */
template <typename Derived, srtb::fft::type fft_type, std::floating_point T,
          typename C = srtb::complex<T> >
class fft_wrapper {
 public:
  static_assert(sizeof(T) * 2 == sizeof(C));
  using sub_class = Derived;
  friend sub_class;

 protected:
  /**
   * @brief corresponded length of one FFT operation
   */
  size_t n;
  /**
   * @brief total data count = n * batch_size
   */
  size_t batch_size;
  sycl::queue q;

 protected:
  sub_class& sub() { return static_cast<sub_class&>(*this); }

  void create(size_t n_, size_t batch_size_, sycl::queue& queue_) {
    n = n_;
    batch_size = batch_size_;
    q = queue_;
    sub().create_impl(n_, batch_size_, queue_);
  }

  void destroy() {
    if (has_inited()) [[likely]] {
      sub().destroy_impl();
    }
  }

 public:
  fft_wrapper(size_t n_, size_t batch_size_, sycl::queue& queue_) {
    create(n_, batch_size_, queue_);
  }

  fft_wrapper(fft_wrapper& other) = delete;

  ~fft_wrapper() { destroy(); }

  bool has_inited() { return sub().has_inited_impl(); }

  template <typename DeviceInputAccessor, typename DeviceOutputAccessor>
  void process(DeviceInputAccessor in, DeviceOutputAccessor out) {
    sub().process_impl(in, out);
  }

  void reset() {
    sub().destroy_impl();
    sub().create_impl();
  }

  size_t get_n() const { return n; }

  size_t get_batch_size() const { return batch_size; }

  size_t get_total_size() const { return n * batch_size; }

  /**
   * @brief Re-construct the plan for FFT-operation of another size and batch_size
   */
  void set_size(size_t n_, size_t batch_size_ = 1) {
    if (n != n_ || batch_size != batch_size_) [[likely]] {
      if (has_inited()) [[likely]] {
        destroy();
      }
      create(n_, batch_size_, q);
    }
  }
};

}  // namespace fft
}  // namespace srtb

#endif  //  __SRTB_FFT_WRAPPER__
