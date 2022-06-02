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
#ifndef __SRTB_LOGGER__
#define __SRTB_LOGGER__

#include <iostream>

#include "srtb/global_variables.hpp"

// reference: hipSYCL logger
#define SRTB_LOG(level)                     \
  if (level < srtb::config.log_debug_level) \
  std::cout << srtb::log::get_log_prefix(level)

namespace srtb {
namespace log {

enum debug_levels { NONE = 0, ERROR = 1, WARNING = 2, INFO = 3, DEBUG = 4 };

inline constexpr auto get_log_prefix(const debug_levels level) {
  switch (level) {
    case srtb::log::debug_levels::ERROR:
      return "[SRTB ERROR]";
    case srtb::log::debug_levels::WARNING:
      return "[SRTB  WARN]";
    case srtb::log::debug_levels::INFO:
      return "[SRTB  INFO]";
    case srtb::log::debug_levels::DEBUG:
      return "[SRTB DEBUG]";
    case srtb::log::debug_levels::NONE:
    default:
      return "";
  }
}

}  // namespace log
}  // namespace srtb

#define SRTB_LOGE SRTB_LOG(srtb::log::debug_levels::ERROR)
#define SRTB_LOGW SRTB_LOG(srtb::log::debug_levels::WARNING)
#define SRTB_LOGI SRTB_LOG(srtb::log::debug_levels::INFO)
#define SRTB_LOGD SRTB_LOG(srtb::log::debug_levels::DEBUG)

#endif  // __SRTB_LOGGER__