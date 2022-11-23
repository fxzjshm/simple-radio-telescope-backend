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
#ifndef __SRTB_ALGORITHM_MAP_IDENTITY_HPP__
#define __SRTB_ALGORITHM_MAP_IDENTITY_HPP__

#include <functional>

namespace srtb {
namespace algorithm {

/** @brief like std::identity(), but for map functions below. */
struct map_identity {
  template <typename T>
  [[nodiscard]] constexpr T&& operator()(size_t n, T&& x) const noexcept {
    (void)n;
    return std::forward<T>(x);
  }
};

}  // namespace algorithm
}  // namespace srtb

#endif  // __SRTB_ALGORITHM_MAP_IDENTITY_HPP__
