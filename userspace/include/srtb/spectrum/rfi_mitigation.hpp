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
#ifndef __SRTB_SPECTRIM_RFI_MITIGAGION__
#define __SRTB_SPECTRIM_RFI_MITIGAGION__

#include "srtb/commons.hpp"

namespace srtb {
namespace spectrum {

template <typename DeviceInputAccessor, typename TransformFunctor>
auto transform_and_average(DeviceInputAccessor d_in, size_t in_count,
                           TransformFunctor func, sycl::queue& q) {
  using transformed_type = std::remove_reference_t<decltype(func(d_in[0]))>;
  auto d_sum_shared =
      srtb::device_allocator.allocate_shared<transformed_type>(1);
  auto d_sum = d_sum_shared.get();
  q.fill(d_sum, 0, 1).wait();
  q.submit([&](sycl::handler& cgh) {
     auto sum_reduction =
         sycl::reduction(d_sum, sycl::plus<transformed_type>{});
     cgh.parallel_for(
         sycl::range<1>{in_count}, sum_reduction,
         [=](sycl::id<1> id, auto& sum) { sum.combine(func(d_in[id])); });
   }).wait();
  auto d_avg = d_sum;  // same pointer, renaming for different meaning
  q.single_task([=] { (*d_avg) = (*d_sum) / in_count; }).wait();
  return d_sum_shared;  // now d_avg_shared
}

/**
 * @brief Mitigate RFI by setting narrow band interference to 0 directly
 * TODO: use "spectural kurtosis"
 * TODO: compute norm twice or once with temporary buffer for it ?
 */
template <typename T = srtb::real, typename C = srtb::complex<srtb::real>,
          typename DeviceComplexInputAccessor = C*>
void mitigate_rfi(DeviceComplexInputAccessor d_in, size_t in_count,
                  sycl::queue& q = srtb::queue) {
  const srtb::real thereshold = srtb::config.mitigate_rfi_thereshold;
  auto d_norm_avg_shared = transform_and_average(
      d_in, in_count, [](C c) { return srtb::norm(c); }, q);
  auto d_norm_avg = d_norm_avg_shared.get();
  q.parallel_for(sycl::range<1>{in_count}, [=](sycl::item<1> id) {
     const size_t i = id.get_id(0);
     const srtb::real norm_avg = (*d_norm_avg);
     const srtb::real val = srtb::norm(d_in[i]);
     if (val > thereshold * norm_avg) {
       d_in[i] = C(T(0), T(0));
     }
   }).wait();
}

}  // namespace spectrum
}  // namespace srtb

#endif  // __SRTB_SPECTRIM_RFI_MITIGAGION__
