/*******************************************************************************
 * Copyright (c) 2024 fxzjshm
 * 21cma-make_beam is licensed under Mulan PubL v2.
 * You can use this software according to the terms and conditions of the Mulan PubL v2.
 * You may obtain a copy of Mulan PubL v2 at:
 *          http://license.coscl.org.cn/MulanPubL-2.0
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PubL v2 for more details.
 ******************************************************************************/

#pragma once
#ifndef __SRTB_21CMA_MAKE_BEAM_STATISTICS__
#define __SRTB_21CMA_MAKE_BEAM_STATISTICS__

#include <span>
#include <sycl/execution_policy>
// --
#include <sycl/algorithm/transform_reduce.hpp>

#include "srtb/sycl.hpp"
// ---
#include "assert.hpp"
#include "merge_sort.hpp"

namespace srtb::_21cma::make_beam {

/**
 * @brief Calculate median value of input array. 
 *        Input in d_temp1 but will be overwritten by sorted values.
 */
template <typename T>
[[nodiscard]]
inline auto median(std::span<T> d_temp1, std::span<T> d_temp2, sycl::queue& q) -> T {
  BOOST_ASSERT(d_temp1.size() == d_temp2.size());
  const auto n = d_temp1.size();
  sycl_stl::merge_sort_on_gpu(q, d_temp1.data(), d_temp1.data() + n, d_temp2.data(), std::less<T>());
  // d_temp1 should be sorted
  T median_val;
  q.copy(d_temp1.data() + (n / 2), &median_val, 1).wait();
  return median_val;
}

/**
 * @brief Calculate median value of input array. 
 *        Input in d_in and will not be overwritten, but needs 1 copy to d_temp1.
 *        After execution, d_temp1 is sorted version of d_in.
 */
template <typename T>
[[nodiscard]]
inline auto median(const std::span<T> d_in, std::span<T> d_temp1, std::span<T> d_temp2, sycl::queue& q) -> T {
  BOOST_ASSERT(d_in.size() == d_temp1.size());
  BOOST_ASSERT(d_in.size() == d_temp2.size());
  const auto n = d_in.size();
  q.copy(d_in.data(), d_temp1.data(), n).wait();
  return median(d_temp1, d_temp2);
}

/**
 * @brief get median & mean absolute deviation of d_in
 */
template <typename T>
[[nodiscard]]
inline auto median_absolute_deviation(std::span<T> d_in, std::span<T> d_temp1, std::span<T> d_temp2,
                                      sycl::queue& q) -> std::pair<T, T> {
  BOOST_ASSERT(d_in.size() == d_temp1.size());
  BOOST_ASSERT(d_in.size() == d_temp2.size());
  const auto n = d_in.size();
  q.copy(d_in.data(), d_temp1.data(), n).wait();
  const T in_median = median(d_temp1, d_temp2, q);
  // now d_temp1 is sorted version of d_in
  q.parallel_for(sycl::range<1>{n}, [=](sycl::item<1> id) {
     const auto i = id.get_id(0);
     d_temp1[i] = sycl::fabs(d_temp1[i] - in_median);
   }).wait();
  sycl_stl::merge_sort_on_gpu(q, d_temp1.data(), d_temp1.data() + n, d_temp2.data(), std::less<T>());
  // d_temp1 should be sorted again
  T mad;
  q.copy(d_temp1.data() + (n / 2), &mad, 1).wait();
  return std::pair{in_median, mad};
}

template <typename T>
[[nodiscard]]
inline auto standard_deviation(std::span<T> d_in, sycl::queue& q) -> std::pair<T, T> {
  sycl::sycl_execution_policy<> snp{q};
  const T sum_val =
      sycl::impl::transform_reduce(snp, d_in.data(), d_in.data() + d_in.size(), std::identity(), T{0}, std::plus());
  const T avg_val = sum_val / d_in.size();
  const T std2_val = sycl::impl::transform_reduce(
      snp, d_in.data(), d_in.data() + d_in.size(), [=](T x) { return (x - avg_val) * (x - avg_val); }, T{0},
      std::plus());
  const T std_val = sycl::sqrt(std2_val);
  return std::pair{avg_val, std_val};
}

}  // namespace srtb::_21cma::make_beam

#endif  // __SRTB_21CMA_MAKE_BEAM_GET_DELAYS__
