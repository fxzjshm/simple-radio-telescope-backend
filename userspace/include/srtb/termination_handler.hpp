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

// forward declarations
inline void termination_handler();
inline void signal_handler(int signal);

class termination_handler_t {
 public:
  std::terminate_handler original_terminate_handler;
  sighandler_t original_SIGILL_handler;
  sighandler_t original_SIGFPE_handler;
  sighandler_t original_SIGSEGV_handler;

  termination_handler_t() {
    original_terminate_handler = std::set_terminate(&srtb::termination_handler);
    // remember to set the behaviour in signal_handler
    original_SIGILL_handler = std::signal(SIGILL, srtb::signal_handler);
    original_SIGFPE_handler = std::signal(SIGFPE, srtb::signal_handler);
    original_SIGSEGV_handler = std::signal(SIGSEGV, srtb::signal_handler);
  }
};

inline termination_handler_t termination_handler_v;

inline void print_stacktrace() {
  try {
    SRTB_LOGE << " [termination handler] "
              << "Stacktrace:" << '\n'
              << boost::stacktrace::stacktrace() << '\n';
  } catch (...) {
  }
}

inline void termination_handler() {
  print_stacktrace();
  (*termination_handler_v.original_terminate_handler)();
}

constexpr std::string_view get_signal_name(int signal) {
  switch (signal) {
    case SIGTERM:
      return "SIGTERM";
    case SIGSEGV:
      return "SIGSEGV";
    case SIGINT:
      return "SIGINT";
    case SIGILL:
      return "SIGILL";
    case SIGABRT:
      return "SIGABRT";
    case SIGFPE:
      return "SIGFPE";
    default:
      return "";
  }
}

inline void signal_handler(int signal) {
  SRTB_LOGE << " [singal handler] "
            << "Received signal " << signal << ' ' << get_signal_name(signal)
            << '\n';
  print_stacktrace();
  sighandler_t next_handler;
  switch (signal) {
    case SIGSEGV:
      next_handler = termination_handler_v.original_SIGSEGV_handler;
      break;
    case SIGILL:
      next_handler = termination_handler_v.original_SIGILL_handler;
      break;
    case SIGFPE:
      next_handler = termination_handler_v.original_SIGFPE_handler;
      break;
    default:
      next_handler = SIG_DFL;
      break;
  }
  std::signal(signal, next_handler);
}

}  // namespace srtb

#endif  // __SRTB_TERMINATION_HANDLER__
