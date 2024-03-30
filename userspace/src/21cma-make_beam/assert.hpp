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
#ifndef __SRTB_21CMA_MAKE_BEAM_ASSERT__
#define __SRTB_21CMA_MAKE_BEAM_ASSERT__

#define BOOST_ENABLE_ASSERT_HANDLER
#include <boost/assert.hpp>

#include "srtb/termination_handler.hpp"

namespace boost {

inline void assertion_failed(char const *expr, char const *function, char const *file, long line) {
  throw std::runtime_error{std::string{} + "Expression \"" + expr + "\" evaluates to false in funtion \"" + function +
                           "\" in file \"" + file + "\" line " + std::to_string(line)};
}

inline void assertion_failed_msg(char const *expr, char const *msg, char const *function, char const *file, long line) {
  throw std::runtime_error{std::string{} + "Expression \"" + expr + "\" evaluates to false in funtion \"" + function +
                           "\" in file \"" + file + "\" line " + std::to_string(line) + " with message \"" + msg +
                           "\""};
}
} // namespace boost

#endif // __SRTB_21CMA_MAKE_BEAM_ASSERT__
