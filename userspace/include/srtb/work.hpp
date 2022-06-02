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
#ifndef __SRTB_WORK__
#define __SRTB_WORK__

namespace srtb {

/**
 * @brief This represents a work to be done and should be the same as `std::pair<T, size_t>`,
 *        created just because `std::pair` doesn't satisfy `boost::has_trivial_assign`.
 * 
 * @tparam T Type of the pointer of the work, e.g. void* for unpack and srtb::real* for FFT.
 */
template <typename T>
struct work {
  T ptr;
  size_t size;

  work() : ptr{nullptr}, size{0} {}

  work(T ptr_, size_t size_) : ptr{ptr_}, size{size_} {}
};

}  // namespace srtb

#endif  // __SRTB_WORK__