/******************************************************************************* 
 * Copyright (c) 2023 fxzjshm
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
#ifndef __SRTB_ALGORITHM_MJD_HPP__
#define __SRTB_ALGORITHM_MJD_HPP__

#include <chrono>

namespace srtb {
namespace algorithm {

/**
 * ref: https://en.cppreference.com/w/cpp/chrono/duration
 *      https://stackoverflow.com/questions/466321/convert-unix-timestamp-to-julian
 *      https://en.wikipedia.org/wiki/Julian_day
 */
template <typename Rep, typename Period>
inline auto unix_timestamp_to_mjd(std::chrono::duration<Rep, Period> d)
    -> double {
  auto d_in_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(d);
  return (d_in_ns.count() / (1.0e9 * 86400.0)) + 2440587.5 - 2400000.5;
}

}  // namespace algorithm
}  // namespace srtb

#endif  // __SRTB_ALGORITHM_MAP_IDENTITY_HPP__
