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
#ifndef __SRTB_ALGORITHM_MAP_REDUCE_HPP__
#define __SRTB_ALGORITHM_MAP_REDUCE_HPP__

#include "srtb/commons.hpp"

namespace srtb {
namespace algorithm {

/**
 * @brief Apply func on input, then calculate reduced value of the transformed value.
 *        Just a simple wrapper of sycl APIs.
 * @return auto std::shared_pointer of size 1 on the device, containing the final result 
 *         i.e. reduced value of transformed data.
 */
template <typename DeviceInputAccessor, typename MapFunctor,
          typename ReduceFunctor>
inline auto map_reduce(DeviceInputAccessor d_in, size_t in_count,
                       MapFunctor map, ReduceFunctor reduce, sycl::queue& q) {
  using transformed_type = std::remove_reference_t<decltype(map(0, d_in[0]))>;
  auto d_reduced_shared =
      srtb::device_allocator.allocate_shared<transformed_type>(1);
  auto d_reduced = d_reduced_shared.get();
  q.fill<transformed_type>(d_reduced, transformed_type{0}, 1).wait();
  q.submit([&](sycl::handler& cgh) {
     auto reduction = sycl::reduction(d_reduced, reduce);
     cgh.parallel_for(sycl::range<1>{in_count}, reduction,
                      [=](sycl::id<1> id, auto& reduce) {
                        const size_t i = id;
                        reduce.combine(map(i, d_in[i]));
                      });
   }).wait();
  return d_reduced_shared;
}

/**
 * @brief Apply func on input, then calculate sum of the transformed value.
 * @return auto std::shared_pointer of size 1 on the device, containing the final result 
 *         i.e. sum of transformed data.
 */
template <typename DeviceInputAccessor, typename MapFunctor>
inline auto map_sum(DeviceInputAccessor d_in, size_t in_count, MapFunctor map,
                    sycl::queue& q) {
  using transformed_type = std::remove_reference_t<decltype(map(0, d_in[0]))>;
  return map_reduce(d_in, in_count, map, sycl::plus<transformed_type>(), q);
}

/**
 * @brief Apply func on input, then calculate average value of the transformed value.
 * @return auto std::shared_pointer of size 1 on the device, containing the final result 
 *         i.e. average value of transformed data.
 */
template <typename DeviceInputAccessor, typename MapFunctor>
inline auto map_average(DeviceInputAccessor d_in, size_t in_count,
                        MapFunctor map, sycl::queue& q) {
  auto d_sum_shared = map_sum(d_in, in_count, map, q);
  auto d_sum = d_sum_shared.get();
  auto d_avg = d_sum;  // same pointer, renaming for different meaning
  q.single_task([=] { (*d_avg) = (*d_sum) / in_count; }).wait();
  return d_sum_shared;  // now d_avg_shared
}

}  // namespace algorithm
}  // namespace srtb

#endif  // __SRTB_ALGORITHM_MAP_REDUCE_HPP__
