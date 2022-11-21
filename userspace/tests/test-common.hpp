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
#ifndef __SRTB_TEST_COMMONS__
#define __SRTB_TEST_COMMONS__

#include "srtb/commons.hpp"

#define SRTB_CHECK_TEST(tag, expr)                                    \
  SRTB_CHECK(expr, true, {                                            \
    throw std::runtime_error{tag #expr " at " __FILE__ ":" +          \
                             std::to_string(__LINE__) + " returns " + \
                             std::to_string(ret)};                    \
  })

template <typename Iterator1, typename Iterator2, typename T>
inline bool check_absolute_error(Iterator1 first1, Iterator1 last1,
                                 Iterator2 first2, T threshold) {
  {
    auto iter1 = first1;
    auto iter2 = first2;
    for (; iter1 != last1; ++iter1, ++iter2) {
      if (srtb::abs((*iter1) - (*iter2)) > threshold) {
        return false;
      }
    }
  }
  return true;
}

template <typename Iterator1, typename Iterator2, typename T>
inline bool check_relative_error(Iterator1 first1, Iterator1 last1,
                                 Iterator2 first2, T threshold) {
  {
    auto iter1 = first1;
    auto iter2 = first2;
    for (; iter1 != last1; ++iter1, ++iter2) {
      const auto val1 = *iter1, val2 = *iter2;
      const auto diff = srtb::abs(val1 - val2), avg = srtb::abs(val1 + val2);
      if (!(std::isfinite(diff) && std::isfinite(avg))) [[unlikely]] {
        return false;
      }
      if (std::fpclassify(avg) == FP_ZERO) [[unlikely]] {
        if (std::fpclassify(diff) != FP_ZERO) {
          return false;
        }
      }
      if (diff / avg > threshold) [[unlikely]] {
        return false;
      }
    }
  }
  return true;
}

#endif  // __SRTB_TEST_COMMONS__
