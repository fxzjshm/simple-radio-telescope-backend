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
#ifndef __SRTB_PIPELINE_PIPE__
#define __SRTB_PIPELINE_PIPE__

#include <functional>
#include <thread>
#include <type_traits>

#if __has_include(<pthread.h>)
#include <pthread.h>
#endif

#include "srtb/commons.hpp"

namespace srtb {
namespace pipeline {

/**
 * @brief Represents a pipe, which receives work from a work_queue,
 *        apply transform and then push the result to next one, on different threads.
 * @note functions required for derived classes to have: @c run_once_impl
 *       optional ones: @c setup_impl, @c teardown_impl , which constructs/deconstructs
 *                      required resources on the thread to be run
 * @note TODO: decouple work queue push and pop from @c run_once_impl
 * @tparam Derived derived class. ref: curiously recurring template pattern (CRTP)
 */
template <typename Derived>
class pipe {
  friend Derived;

 public:
  /**
   * @brief queue used in this pipeline.
   *        In some implementations @c sycl::queue is sequencial, in order to 
   *        overlap different stages different queues should be used in different
   *        pipes, which run on their own thread.
   */
  sycl::queue q;
  // other things should be defined on derived pipes.
  // TODO: should this class include in/out work queue, instead of using global work queues?

 public:
  /**
   * @brief Start working in a new thread
   * @return std::jthread the thread running on.
   */
  template <typename... Args>
  static std::jthread start(Args... args) {
    std::jthread jthread{[](std::stop_token stop_token, Args... args) {
                           Derived derived{args...};
                           derived.run(stop_token);
                         },
                         args...};
#if __has_include(<pthread.h>)
    const std::string thread_name = generate_thread_name();
    pthread_setname_np(jthread.native_handle(), thread_name.c_str());
#else
#warning not setting thread name of pipe (TODO)
#endif
    jthread.detach();
    return jthread;
  }

  /**
   * @brief Run on this thread. May block this thread.
   */
  void run(std::stop_token stop_token) {
    constexpr bool has_setup = requires() { sub().setup_impl(stop_token); };
    constexpr bool has_teardown =
        requires() { sub().teardown_impl(stop_token); };

    if constexpr (has_setup) {
      sub().setup_impl(stop_token);
    }
    srtb::pipeline::running_pipe_count++;
    while ((!stop_token.stop_possible()) ||
           (stop_token.stop_possible() && !stop_token.stop_requested()))
        [[likely]] {
      sub().run_once_impl(stop_token);
    }
    srtb::pipeline::running_pipe_count--;
    if constexpr (has_teardown) {
      sub().teardown_impl(stop_token);
    }
  }

  // functions run on the new thread:
  // setup() -> setup_impl(), if constructor doesn't apply here.
  // run_once() -> run_once_impl()
  // teardown() -> teardown_impl(), if destructor doesn't apply here.
  // some functions rely on thread_local variables, but constructor & destructor
  // run on main thread, so setup_impl() & teardown_impl() are needed.

 private:
  pipe(sycl::queue q_ = srtb::queue) : q{q_} {};

  ~pipe() {
    // wait until all operations in this pipe finish, then exit
    // otherwise if the sycl runtime exit before operations (like host to device memcpy) is done,
    // error may happen
    q.wait();
  }

  /**
   * @brief Shortcut for CRTP.
   */
  Derived& sub() { return static_cast<Derived&>(*this); }

  /**
   * @brief thread name of a pipe is type name of the pipe, for debugging
   * 
   * @return std::string 
   */
  static inline auto generate_thread_name() -> std::string {
    const auto full_type_name = srtb::type_name<Derived>();
    // example full type name:
    //   1) srtb::pipeline::xxx_pipe
    //   2) srtb::pipeline::xxx_pipe<some_template_parameter>
    //   3) xxx_pipe
    // need to deal with all of these
    constexpr std::string_view template_mark{"<"};
    constexpr std::string_view namespace_mark{"::"};
    const size_t template_mark_position = full_type_name.find(template_mark);
    const auto end = full_type_name.size();
    size_t start;
    if (template_mark_position != std::string::npos) {
      start = full_type_name.rfind(namespace_mark, template_mark_position);
    } else {
      start = full_type_name.rfind(namespace_mark, end);
    }
    start += namespace_mark.size();
    auto name = full_type_name;
    if (start != std::string::npos && start < end) {
      name = full_type_name.substr(start, (end - start));
    }

    // pthread restrict thread name length < 16 characters (not including \0 ?),
    // otherwise pthread_setname_np has no effect
    constexpr size_t thread_name_length_limit = 15;
    if (name.size() > thread_name_length_limit) {
      name = name.substr(0, thread_name_length_limit);
    }
    return std::string{name};
  }
};

inline void wait_for_notify(std::stop_token stop_token) {
  while (true) {
    bool b = true;  // expected
    constexpr bool desired = false;
    /*
    compare and exchange / compare and swap (CAS):
    ```
    synchronized {
      if (*this == expected){
        *this <- desired;
      } else {
        expected <- *this
      }
    }
    ```
    */
    srtb::pipeline::need_more_work.compare_exchange_weak(/* expected = */ b,
                                                         desired);
    // now b is actural value of srtb::pipeline::need_more_work
    if (b) {
      break;
    }
    if (stop_token.stop_requested()) [[unlikely]] {
      break;
    }
    std::this_thread::sleep_for(
        std::chrono::nanoseconds(srtb::config.thread_query_work_wait_time));
  }

  SRTB_LOGD << " [pipeline] "
            << "received notify." << srtb::endl;
}

inline void notify() {
  need_more_work = true;
  SRTB_LOGD << " [pipeline] "
            << "notified pipeline source for more work." << srtb::endl;
}

}  // namespace pipeline
}  // namespace srtb

#endif  // __SRTB_PIPELINE_PIPE__
