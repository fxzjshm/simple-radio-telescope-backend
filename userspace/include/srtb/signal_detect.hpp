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
#ifndef __SRTB_SIGNAL_DETECT__
#define __SRTB_SIGNAL_DETECT__

#include <cstddef>
#include <type_traits>

#include "srtb/sycl.hpp"
// --- divide line for clang-format
#include "srtb/algorithm/map_reduce.hpp"
#include "srtb/algorithm/multi_reduce.hpp"

namespace srtb {
namespace signal_detect {

/**
 * @brief trivial signal detection, using signal-to-noise ratio
 */
template <typename T, typename InputIterator>
inline auto count_signal(InputIterator d_in, size_t in_count, T threshold,
                         sycl::queue& q) -> size_t {
  // check value types, to avoid unexpected type casts
  static_assert(
      std::is_same_v<typename std::iterator_traits<InputIterator>::value_type,
                     T>);

  // if T (= float, usually) is not enough, use double
  using variance_sum_type = T;
  auto d_variance_squared_shared = srtb::algorithm::map_average(
      d_in, in_count,
      []([[maybe_unused]] size_t pos, T x) -> variance_sum_type {
        const variance_sum_type y = static_cast<variance_sum_type>(x);
        return y * y;
      },
      q);
  auto d_variance_squared = d_variance_squared_shared.get();
  auto d_variance_shared = srtb::device_allocator.allocate_shared<T>(1);
  auto d_variance = d_variance_shared.get();
  q.single_task([=]() {
     (*d_variance) = static_cast<T>(sycl::sqrt(*d_variance_squared));
   }).wait();

  auto d_signal_count_shared = srtb::algorithm::map_sum(
      d_in, in_count, /* map = */
      [=]([[maybe_unused]] size_t pos, srtb::real x) -> size_t {
        // also known as count_if
        if (x > threshold * (*d_variance)) {
          return size_t{1};
        } else {
          return size_t{0};
        }
      },
      q);
  size_t* d_signal_count = d_signal_count_shared.get();
  size_t h_signal_count;
  q.copy(d_signal_count, /* -> */ &h_signal_count, /* size = */ 1).wait();
  return h_signal_count;
}

}  // namespace signal_detect
}  // namespace srtb

#endif  // __SRTB_SIGNAL_DETECT__
