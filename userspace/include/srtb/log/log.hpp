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
#ifndef __SRTB_LOG__
#define __SRTB_LOG__

#include <iostream>
#include <syncstream>

#include "srtb/global_variables.hpp"
#include "srtb/log/sync_ostream_wrapper.hpp"

// reference: hipSYCL logger at hipSYCL/common/debug.hpp
#define SRTB_LOG(level)                                  \
  if (static_cast<int>(level) <= srtb::config.log_level) \
  srtb::log::sync_stream_wrapper{std::cout} << srtb::log::get_log_prefix(level)

namespace srtb {

inline constexpr auto endl = '\n';

namespace log {

enum class levels : int {
  NONE = 0,
  ERROR = 1,
  WARNING = 2,
  INFO = 3,
  DEBUG = 4
};

inline constexpr auto get_log_prefix(const log::levels level) {
  switch (level) {
    case srtb::log::levels::ERROR:
      return "\033[1;31m[SRTB ERROR]\033[0m";  // bright red
    case srtb::log::levels::WARNING:
      return "\033[;35m[SRTB  WARN]\033[0m";  // magenta
    case srtb::log::levels::INFO:
      return "\033[;32m[SRTB  INFO]\033[0m";  // green
    case srtb::log::levels::DEBUG:
      return "\033[;36m[SRTB DEBUG]\033[0m";  // cyan
    case srtb::log::levels::NONE:
    default:
      return "";
  }
}

}  // namespace log
}  // namespace srtb

#define SRTB_LOGE SRTB_LOG(srtb::log::levels::ERROR)
#define SRTB_LOGW SRTB_LOG(srtb::log::levels::WARNING)
#define SRTB_LOGI SRTB_LOG(srtb::log::levels::INFO)
#define SRTB_LOGD SRTB_LOG(srtb::log::levels::DEBUG)

#endif  // __SRTB_LOG__