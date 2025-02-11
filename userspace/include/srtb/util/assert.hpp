/*******************************************************************************
 * Copyright (c) 2024 fxzjshm
 * 21cma-make_beam is licensed under Mulan PubL v2.
 * You can use this software according to the terms and conditions of the Mulan PubL v2.
 * You may obtain a copy of Mulan PubL v2 at:
 *          http://license.coscl.org.cn/MulanPubL-2.0
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PubL v2 for more details.
 ******************************************************************************/

#pragma once
#ifndef SRTB_ASSERT
#define SRTB_ASSERT

#define BOOST_ENABLE_ASSERT_HANDLER
#include <boost/assert.hpp>
// ---
#include <stdexcept>
#include <string>

#include "srtb/log/log.hpp"
#include "srtb/util/termination_handler.hpp"

namespace srtb::log {

inline auto error_message(char const *expr, char const *function, char const *file, long line) -> std::string {
  return "Expression \"" + bright_red + expr + clear_colour + "\" evaluates to false in funtion \"" + function +
         "\" in file \"" + file + "\" line " + std::to_string(line) + clear_colour;
}

inline auto error_message(char const *expr, char const *msg, char const *function, char const *file,
                          long line) -> std::string {
  return "Message: \"" + bright_red + msg + clear_colour + "\". " + error_message(expr, function, file, line);
}

}  // namespace srtb

namespace boost {

inline void assertion_failed(char const *expr, char const *function, char const *file, long line) {
  auto err = srtb::log::error_message(expr, function, file, line);
  SRTB_LOGE << " [assertion_failed] " << err << srtb::endl;
  throw std::runtime_error{err};
}

inline void assertion_failed_msg(char const *expr, char const *msg, char const *function, char const *file, long line) {
  auto err = srtb::log::error_message(expr, msg, function, file, line);
  SRTB_LOGE << " [assertion_failed_msg] " << err << srtb::endl;
  throw std::runtime_error{err};
}

}  // namespace boost

#endif  // __SRTB_21CMA_MAKE_BEAM_ASSERT__
