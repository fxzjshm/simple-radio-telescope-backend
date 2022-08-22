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
#ifndef __SRTB_TERMINATION_HANDLER__
#define __SRTB_TERMINATION_HANDLER__

// ref: https://www.boost.org/doc/libs/1_80_0/doc/html/stacktrace/getting_started.html
//      https://stackoverflow.com/a/77336

#include <boost/stacktrace.hpp>
#include <csignal>
#include <cstdlib>    // std::abort
#include <exception>  // std::set_terminate
#include <iostream>   // std::cerr

#include "srtb/log/log.hpp"

namespace srtb {

inline void termination_handler() {
  try {
    std::cerr << srtb::log::get_log_prefix(srtb::log::levels::ERROR)
              << " [singal handler] "
              << "Backtrace:" << '\n'
              << boost::stacktrace::stacktrace();
  } catch (...) {
  }
  std::abort();
}

inline void signal_handler(int signal) {
  std::cerr << srtb::log::get_log_prefix(srtb::log::levels::ERROR)
            << " [singal handler] "
            << "Received signal " << signal << '\n';
  termination_handler();
}

class termination_handler_t {
 public:
  termination_handler_t() {
    std::set_terminate(&srtb::termination_handler);
    std::signal(SIGILL, srtb::signal_handler);
    std::signal(SIGFPE, srtb::signal_handler);
    std::signal(SIGSEGV, srtb::signal_handler);
  }
};

inline termination_handler_t termination_handler_v;

}  // namespace srtb

#endif  // __SRTB_TERMINATION_HANDLER__
