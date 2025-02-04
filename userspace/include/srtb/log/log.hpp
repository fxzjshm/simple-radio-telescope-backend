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
#include <syncstream>

// reference: hipSYCL logger at hipSYCL/common/debug.hpp
#define SRTB_LOG(level)                                                      \
  if (static_cast<int>(level) <= static_cast<int>(srtb::log::current_level)) \
  std::osyncstream{std::cout} << srtb::log::get_log_prefix(level)

namespace srtb {

inline constexpr auto endl = '\n';

namespace log {

enum class level : int {
  NONE = 0,
  ERROR = 1,
  WARNING = 2,
  INFO = 3,
  DEBUG = 4
};

inline auto get_level_from_env_or(srtb::log::level default_level)
    -> srtb::log::level {
  srtb::log::level log_level = default_level;
  try {
    char* log_env = std::getenv("SRTB_LOG_LEVEL");
    if (log_env != nullptr) {
      log_level = static_cast<srtb::log::level>(std::stoi(log_env));
    }
  } catch (const std::invalid_argument& ignored) {
  }
  return log_level;
}

/**
  * @brief Log level for console output.
  * @see srtb::log::levels
  */
inline srtb::log::level current_level =
    get_level_from_env_or(srtb::log::level::INFO);

/** @brief record start time of program, used in log to indicate relative time */
inline auto log_start_time = std::chrono::system_clock::now();

inline constexpr size_t PREFIX_BUFFER_LENGTH = 64ul;

inline constexpr std::string bright_red = "\033[1;31m";
inline constexpr std::string magenta = "\033[;35m";
inline constexpr std::string green = "\033[;32m";
inline constexpr std::string cyan = "\033[;36m";
inline constexpr std::string clear_colour = "\033[0m";

inline std::string get_log_prefix(const log::level level) {
  std::string prefix, suffix, tag;
  switch (level) {
    case srtb::log::level::ERROR:
      prefix = bright_red;
      tag = "E";
      break;
    case srtb::log::level::WARNING:
      prefix = magenta;
      tag = "W";
      break;
    case srtb::log::level::INFO:
      prefix = green;
      tag = "I";
      break;
    case srtb::log::level::DEBUG:
      prefix = cyan;
      tag = "D";
      break;
    case srtb::log::level::NONE:
    default:
      return "";
  }
  suffix = clear_colour;

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

#define SRTB_LOGE SRTB_LOG(srtb::log::level::ERROR)
#define SRTB_LOGW SRTB_LOG(srtb::log::level::WARNING)
#define SRTB_LOGI SRTB_LOG(srtb::log::level::INFO)
#define SRTB_LOGD SRTB_LOG(srtb::log::level::DEBUG)

#endif  // __SRTB_LOG__
