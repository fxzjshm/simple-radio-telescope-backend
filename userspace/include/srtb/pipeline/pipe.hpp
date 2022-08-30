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
  std::jthread start() {
    std::jthread jthread{
        [this](std::stop_token stop_token) { run(stop_token); }};
    jthread.detach();
    return jthread;
  }

  /**
   * @brief Run on this thread. May block this thread.
   */
  void run(std::stop_token stop_token) {
    constexpr bool has_setup = requires() { sub().setup_impl(); };
    constexpr bool has_teardown = requires() { sub().teardown_impl(); };

    if constexpr (has_setup) {
      sub().setup_impl();
    }
    while ((!stop_token.stop_possible()) ||
           (stop_token.stop_possible() && !stop_token.stop_requested()))
      [[likely]] { sub().run_once_impl(); }
    if constexpr (has_teardown) {
      sub().teardown_impl();
    }
  }

  // setup() -> setup_impl(), if constructor doesn't apply here.
  // run_once() -> run_once_impl()
  // teardown() -> teardown_impl(), if destructor doesn't apply here.

 private:
  pipe() {
    q = sycl::queue{srtb::queue.get_context(), srtb::queue.get_device()};
  }

  pipe(sycl::queue q_) : q{q_} {};

  /**
   * @brief Shortcut for CRTP.
   */
  Derived& sub() { return static_cast<Derived&>(*this); }
};

}  // namespace pipeline
}  // namespace srtb

#endif  // __SRTB_PIPELINE_PIPE__
