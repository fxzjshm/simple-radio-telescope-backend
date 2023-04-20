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
#ifndef __SRTB_SIMPLIFY_SPECTRUM__
#define __SRTB_SIMPLIFY_SPECTRUM__

#include "srtb/commons.hpp"
// --- divide line for clang-format ---
#include "srtb/algorithm/map_reduce.hpp"
#include "srtb/algorithm/map_identity.hpp"

namespace srtb {
namespace spectrum {

/** @brief a very small float number to test equivalence of two float numbers */
inline constexpr srtb::real eps = 1e-5;

/**
 * @brief average the norm of input complex numbers
 *        xxxxxxxxx xxxxxxxxx......xxxxxxxxx
 *        |-------| |-------|      |-------|
 *          \   /     \   /          \   /
 *         \bar{x}   \bar{x}        \bar{x}
 */
template <typename T = srtb::real, typename C = srtb::complex<srtb::real>,
          typename DeviceInputAccessor = C*, typename DeviceOutputAccessor = T*>
void simplify_spectrum_calculate_norm(DeviceInputAccessor d_in, size_t in_count,
                                      DeviceOutputAccessor d_out,
                                      size_t out_count, size_t batch_size = 1,
                                      sycl::queue& q = srtb::queue) {
  static_assert(sizeof(T) * 2 == sizeof(C));
  constexpr auto norm = [=](srtb::complex<srtb::real> c) {
    return srtb::norm(c);
  };

  const srtb::real in_count_real = static_cast<srtb::real>(in_count);
  const srtb::real out_count_real = static_cast<srtb::real>(out_count);
  // count of in data point that average into one out data point
  const srtb::real coverage = in_count_real / out_count_real;

  q.parallel_for(sycl::range<1>{out_count * batch_size}, [=](sycl::item<1> id) {
     const size_t idx = id.get_id(0);
     const size_t j = idx / out_count;
     const size_t in_offset = j * in_count;
     const size_t out_offset = j * out_count;
     const size_t i = idx - out_offset;
     SRTB_ASSERT_IN_KERNEL(i < out_count && j < batch_size);
     SRTB_ASSERT_IN_KERNEL(j * out_count + i == idx);

     srtb::real sum = 0;
     const srtb::real left_accurate = coverage * i,
                      right_accurate = coverage * (i + 1),
                      left_real = sycl::ceil(left_accurate),
                      right_real = sycl::floor(right_accurate);
     const size_t left = static_cast<size_t>(left_real),
                  right = static_cast<size_t>(right_real);
     for (size_t k = left; k < right; k++) {
       SRTB_ASSERT_IN_KERNEL(k < in_count);
       sum += norm(d_in[k + in_offset]);
     }
     SRTB_ASSERT_IN_KERNEL(left_real >= left_accurate);
     if (left_real - left_accurate > eps) [[likely]] {
       const size_t left_left = left - 1;
       // left_left == static_cast<size_t>(sycl::floor(left_real)),
       const size_t left_right = left;
       SRTB_ASSERT_IN_KERNEL(left_left < in_count);
       sum += (left_right - left_accurate) * norm(d_in[left_left + in_offset]);
     }
     SRTB_ASSERT_IN_KERNEL(right_accurate >= right_real);
     if (right_accurate - right_real > eps) [[likely]] {
       const size_t right_left = right;
       // const size_t right_right = right + 1;
       // right_right == static_cast<size_t>(sycl::ceil(right_real));
       SRTB_ASSERT_IN_KERNEL(right_left < in_count);
       sum +=
           (right_accurate - right_left) * norm(d_in[right_left + in_offset]);
     }
     d_out[i + out_offset] = sum;
   }).wait();
}

/**
 * @brief normalize ( intended to scale values to [0, 1] )
 *        using max value in data.
 * @note input is not stable, so this is not a proper way.
 */
template <typename T = srtb::real, typename DeviceInputAccessor = T*>
void simplify_spectrum_normalize_with_max_value(DeviceInputAccessor d_in,
                                                size_t in_count,
                                                size_t batch_size,
                                                sycl::queue& q = srtb::queue) {
  const size_t total_in_count = in_count * batch_size;
  auto d_max_val_shared =
      srtb::device_allocator.allocate_shared<srtb::real>(batch_size);
  auto d_max_val = d_max_val_shared.get();
  q.fill<srtb::real>(d_max_val, srtb::real{0}, batch_size).wait();

  // maybe log() the power?
  //q.parallel_for(sycl::range<1>{total_in_count}, [=](sycl::item<1> id) {
  //   const size_t i = id.get_id(0);
  //   d_in[i] = sycl::log(d_in[i] + 1);
  // }).wait();
  std::vector<sycl::event> events(batch_size);
  for (size_t k = 0; k < batch_size; k++) {
    // k: segment id
    events.at(k) = q.submit([&](sycl::handler& cgh) {
      const size_t k_ = k;  // avoid data race, ... or whatever.
      auto max_reduction_k =
          sycl::reduction(d_max_val + k_, sycl::maximum<srtb::real>{});
      cgh.parallel_for(sycl::range<1>{in_count}, max_reduction_k,
                       [=](sycl::id<1> id, auto& max) {
                         max.combine(d_in[k_ * in_count + id]);
                       });
    });
  }
  for (auto it = events.rbegin(); it != events.rend(); it++) {
    (*it).wait();
  }
  q.parallel_for(sycl::range<1>{total_in_count}, [=](sycl::item<1> id) {
     const size_t i = id.get_id(0), k = i / in_count;
     d_in[i] /= d_max_val[k];
   }).wait();
}

/**
 * @brief normalize ( intended to scale values to [0, 1] )
 *        using average value in data.
 * @return average value
 */
template <typename T = srtb::real, typename DeviceInputAccessor = T*>
auto simplify_spectrum_normalize_with_average_value(
    DeviceInputAccessor d_in, size_t in_count, sycl::queue& q = srtb::queue)
    -> T {
  auto d_avg_val_shared = srtb::algorithm::map_average(
      d_in, in_count, srtb::algorithm::map_identity(), q);
  auto d_avg_val = d_avg_val_shared.get();
  T h_avg_val;
  q.copy(d_avg_val, /* -> */ &h_avg_val, 1).wait();
  const T coeff = T{1.0} / (h_avg_val * 2);
  if (h_avg_val > std::numeric_limits<T>::epsilon()) [[likely]] {
    q.parallel_for(sycl::range<1>{in_count}, [=](sycl::item<1> id) {
       const size_t i = id.get_id(0);
       d_in[i] *= coeff;
     }).wait();
  }
  return h_avg_val;
}

}  // namespace spectrum
}  // namespace srtb

#endif  // __SRTB_SIMPLIFY_SPECTRUM__
