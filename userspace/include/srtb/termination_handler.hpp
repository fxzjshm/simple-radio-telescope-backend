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

// these variables and functions handle terminations and signals
// when the program will be abnormally closed.
// ref: https://www.boost.org/doc/libs/1_80_0/doc/html/stacktrace/getting_started.html
//      https://stackoverflow.com/a/77336

#include <array>
#include <boost/stacktrace.hpp>
#include <csignal>
#include <cstdlib>    // std::abort
#include <exception>  // std::set_terminate
#include <iostream>   // std::cerr
#include <map>

#include "srtb/log/log.hpp"

namespace srtb {

// forward declarations
inline void termination_handler();
inline void signal_handler(int signal);

class termination_handler_t {
 public:
  std::terminate_handler original_terminate_handler;
  std::map<int, sighandler_t> original_handlers;
  constexpr static std::array tracked_signals = {SIGTERM, SIGSEGV, SIGINT,
                                                 SIGILL,  SIGABRT, SIGFPE};

  /** @brief this constructor registers handlers to runtime */
  termination_handler_t() {
    original_terminate_handler = std::set_terminate(&srtb::termination_handler);
    for (auto tracked_signal : tracked_signals) {
      original_handlers[tracked_signal] =
          std::signal(tracked_signal, srtb::signal_handler);
    }
  }

  /** @brief revert signal handlers */
  ~termination_handler_t() {
    std::set_terminate(original_terminate_handler);
    for (auto tracked_signal : tracked_signals) {
      std::signal(tracked_signal, original_handlers[tracked_signal]);
    }
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
      return "<unknown>";
  }
}

inline void signal_handler(int signal) {
  SRTB_LOGE << " [singal handler] "
            << "Received signal " << signal << ' ' << get_signal_name(signal)
            << '\n';
  print_stacktrace();
  sighandler_t next_handler;
  if (termination_handler_v.original_handlers.contains(signal)) {
    next_handler = termination_handler_v.original_handlers[signal];
  } else {
    next_handler = SIG_DFL;
  }
  std::signal(signal, next_handler);
}

}  // namespace srtb

#endif  // __SRTB_TERMINATION_HANDLER__
