#pragma once
#ifndef __SRTB_LOGGER__
#define __SRTB_LOGGER__

#include <iostream>

#include "commons.hpp"
#include "global_variables.hpp"

#define SRTB_LOG(level, content)                 \
  if (level > srtb::global_config.debug_level) { \
  } else                                         \
    std::cout << srtb::get_log_prefix(level)

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

#define SRTB_LOGE(content) SRTB_LOG(srtb::debug_levels::ERROR, content)
#define SRTB_LOGW(content) SRTB_LOG(srtb::debug_levels::WARNING, content)
#define SRTB_LOGI(content) SRTB_LOG(srtb::debug_levels::INFO, content)
#define SRTB_LOGD(content) SRTB_LOG(srtb::debug_levels::DEBUG, content)

#endif  // __SRTB_LOGGER__