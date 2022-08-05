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
#ifndef __SRTB_SPECTRUM__
#define __SRTB_SPECTRUM__

#include "srtb/commons.hpp"

namespace srtb {
namespace spectrum {

inline constexpr srtb::real eps = 1e-5;

/**
 * @brief average the norm of input complex numbers
 */
template <typename T = srtb::real, typename C = srtb::complex<srtb::real>,
          typename DeviceInputAccessor = C*, typename DeviceOutputAccessor = T*>
void simplify_spectrum_norm_and_sum(DeviceInputAccessor d_in, size_t in_count,
                                    DeviceOutputAccessor d_sum,
                                    size_t out_count,
                                    sycl::queue& q = srtb::queue) {
  const srtb::real in_count_real = static_cast<srtb::real>(in_count);
  const srtb::real out_count_real = static_cast<srtb::real>(out_count);
  // count of in data point that average into one out data point
  const srtb::real coverage = in_count_real / out_count_real;

  q.parallel_for(sycl::range<1>{out_count}, [=](sycl::item<1> id) {
     const size_t i = id.get_id(0);
     srtb::real sum = 0;
     const srtb::real left_accurate = coverage * i,
                      right_accurate = coverage * (i + 1),
                      left_real = sycl::ceil(left_accurate),
                      right_real = sycl::floor(right_accurate);
     const size_t left = static_cast<size_t>(left_real),
                  right = static_cast<size_t>(right_real);
     for (size_t k = left; k < right; k++) {
       SRTB_ASSERT_IN_KERNEL(k < in_count);
       sum += srtb::norm(d_in[k]);
     }
     SRTB_ASSERT_IN_KERNEL(left_real >= left_accurate);
     if (left_real - left_accurate > eps) [[likely]] {
       const size_t left_left = left - 1;
       // left_left == static_cast<size_t>(sycl::floor(left_real)),
       const size_t left_right = left;
       SRTB_ASSERT_IN_KERNEL(left_left < in_count);
       sum += (left_right - left_accurate) * srtb::norm(d_in[left_left]);
     }
     SRTB_ASSERT_IN_KERNEL(right_accurate >= right_real);
     if (right_accurate - right_real > eps) [[likely]] {
       const size_t right_left = right;
       // const size_t right_right = right + 1;
       // right_right == static_cast<size_t>(sycl::ceil(right_real));
       SRTB_ASSERT_IN_KERNEL(right_left < in_count);
       sum += (right_accurate - right_left) * srtb::norm(d_in[right_left]);
     }
     d_sum[i] += sum;
   }).wait();
}

template <typename T = srtb::real, typename C = srtb::complex<srtb::real>,
          typename DeviceInputAccessor = T*, typename HostOutputAccessor = T*>
void simplify_spectrum_normalize(DeviceInputAccessor d_in, size_t in_count,
                                 sycl::queue& q = srtb::queue) {
  auto d_max_val_shared = srtb::device_allocator.allocate_shared<srtb::real>(1);
  auto d_max_val = d_max_val_shared.get();
  q.submit([&](sycl::handler& cgh) {
     auto max_reduction =
         sycl::reduction(d_max_val, sycl::maximum<srtb::real>{});
     cgh.parallel_for(sycl::range<1>{in_count}, max_reduction,
                      [=](sycl::id<1> id, auto& max) {
                        // maybe log() the power?
                        //auto val = sycl::log(d_in[id]);
                        //d_in[id] = val;
                        //max.combine(val);
                        max.combine(d_in[id]);
                      });
   }).wait();
  q.parallel_for(sycl::range<1>{in_count}, [=](sycl::item<1> id) {
     const size_t i = id.get_id(0);
     d_in[i] /= (*d_max_val);
   }).wait();
}
}  // namespace spectrum
}  // namespace srtb

#endif  // __SRTB_SPECTRUM__
