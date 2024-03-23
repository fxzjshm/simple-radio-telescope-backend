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

#include <chrono>
#include <iostream>
#include <sstream>

#include "srtb/log/sync_ostream_wrapper.hpp"

// reference: hipSYCL logger at hipSYCL/common/debug.hpp
#define SRTB_LOG(level)                                  \
  if (static_cast<int>(level) <= static_cast<int>(srtb::log::log_level)) \
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

/**
  * @brief Debug level for console log output.
  * @see srtb::log::levels
  */
inline srtb::log::levels log_level = srtb::log::levels::INFO;

/** @brief record start time of program, used in log to indicate relative time */
inline auto log_start_time = std::chrono::system_clock::now();

inline constexpr size_t PREFIX_BUFFER_LENGTH = 64ul;

inline std::string get_log_prefix(const log::levels level) {
  // TODO: std::string, std::string_view, char*, or else?
  //       when can we use std::format ?
  std::string prefix, suffix, tag;
  switch (level) {
    case srtb::log::levels::ERROR:
      prefix = "\033[1;31m";  // bright red
      tag = "E";
      break;
    case srtb::log::levels::WARNING:
      prefix = "\033[;35m";  // magenta
      tag = "W";
      break;
    case srtb::log::levels::INFO:
      prefix = "\033[;32m";  // green
      tag = "I";
      break;
    case srtb::log::levels::DEBUG:
      prefix = "\033[;36m";  // cyan
      tag = "D";
      break;
    case srtb::log::levels::NONE:
    default:
      return "";
  }
  suffix = "\033[0m";  // clear colour

  auto interval = std::chrono::system_clock::now() - srtb::log::log_start_time;
  double interval_sec = static_cast<double>(interval.count()) / 1e9;

  char str[srtb::log::PREFIX_BUFFER_LENGTH];
  std::snprintf(str, sizeof(str), "%s[%9.06f] %s%s:", prefix.c_str(),
                interval_sec, tag.c_str(), suffix.c_str());
  return std::string(str);
}

template <typename Container, typename U>
inline auto container_to_string(const Container& container, U delimiter)
    -> std::string {
  std::stringstream ss;
  bool is_first_one = true;
  for (auto value : container) {
    if (is_first_one) {
      is_first_one = false;
    } else {
      ss << delimiter;
    }
    ss << value;
  }
  return ss.str();
}

}  // namespace log
}  // namespace srtb

#define SRTB_LOGE SRTB_LOG(srtb::log::levels::ERROR)
#define SRTB_LOGW SRTB_LOG(srtb::log::levels::WARNING)
#define SRTB_LOGI SRTB_LOG(srtb::log::levels::INFO)
#define SRTB_LOGD SRTB_LOG(srtb::log::levels::DEBUG)

#endif  // __SRTB_LOG__
