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
#ifndef __SRTB_MEMORY_STREAMBUF__
#define __SRTB_MEMORY_STREAMBUF__

#include <boost/asio/streambuf.hpp>

namespace srtb {
namespace memory {

/**
 * @brief wrapper for boost::asio::streambuf, just to get access to reserve()
 */
class streambuf : public boost::asio::streambuf {
 public:
  void reserve(std::size_t n) { return boost::asio::streambuf::reserve(n); }
};

}  // namespace memory
}  // namespace srtb

#endif  // __SRTB_MEMORY_STREAMBUF__
