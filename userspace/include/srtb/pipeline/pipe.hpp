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

#include <boost/type_traits.hpp>
#include <functional>
#include <stop_token>
#include <thread>
#include <type_traits>

#if __has_include(<pthread.h>)
#include <pthread.h>
#endif

#include "srtb/commons.hpp"

namespace srtb {
namespace pipeline {

inline namespace detail {

/** @brief class name without namespace or template */
template <typename Type>
struct type_helper {
  inline auto class_name() -> std::string {
    const auto full_type_name = srtb::type_name<Type>();
    // example full type name:
    //   1) srtb::pipeline::xxx_pipe
    //   2) srtb::pipeline::xxx_pipe<some_template_parameter>
    // need to deal with all of these
    constexpr std::string_view template_mark{"<"};
    constexpr std::string_view namespace_mark{"::"};

    constexpr std::string_view blank_chars = " \n\t";
    const size_t full_type_name_end =
        full_type_name.find_last_not_of(blank_chars) + 1;

    const size_t template_mark_position = full_type_name.find(template_mark);
    size_t namespace_pos;
    if (template_mark_position != std::string::npos) {
      namespace_pos =
          full_type_name.rfind(namespace_mark, template_mark_position);
    } else {
      namespace_pos = full_type_name.rfind(namespace_mark, full_type_name_end);
    }

    size_t start;
    if (namespace_pos != std::string::npos) {
      start = namespace_pos + namespace_mark.size();
    } else {
      start = 0;
    }

    const size_t end = std::min(full_type_name_end, template_mark_position);

    auto name = full_type_name;
    if (0 <= start && start < end && end < full_type_name.size()) {
      name = full_type_name.substr(start, (end - start));
    }
    return std::string{name};
  }
};

/** @brief thread name of a pipe is type name of the pipe, for debugging */
template <typename Type>
inline auto generate_thread_name() -> std::string {
  auto name = type_helper<Type>{}.class_name();

  // pthread restrict thread name length < 16 characters (not including \0 ?),
  // otherwise pthread_setname_np has no effect
  constexpr size_t thread_name_length_limit = 15;
  if (name.size() > thread_name_length_limit) {
    name = name.substr(0, thread_name_length_limit);
  }
  return std::string{name};
}

inline bool is_running(std::stop_token stop_token) {
  return (!stop_token.stop_possible()) ||
         (stop_token.stop_possible() && !stop_token.stop_requested());
}

}  // namespace detail

/**
 * @brief Represents a pipe, which receives work from a work_queue,
 *        apply transform and then push the result to next one, on different threads.
 * @note TODO: decouple work queue push and pop from @c run_once_impl
 * @tparam PipeFunctor actural pipe, with signature pipe_functor(std::stop_token, in_work_type) -> out_work_type
 * @tparam InFunctor funtor to provide work for this pipe, with signature @c in_functor(std::stop_token) -> std::optional<in_work_type>
 * @tparam OutFunctor functor to store result of this pipe, with signature @c out_functor(std::stop_token, out_work_type) -> void
 */
template <typename PipeFunctor, typename InFunctor, typename OutFunctor>
class pipe {
 public:
  PipeFunctor pipe_functor;
  InFunctor in_functor;
  OutFunctor out_functor;
  // TODO: should this class include in/out work queue, instead of using global work queues?

 public:
  /**
   * @brief Run on this thread. May block this thread.
   */
  void run(std::stop_token stop_token) {
    const auto class_name = type_helper<PipeFunctor>{}.class_name();
    const auto tag = " [" + class_name + "] ";
    srtb::pipeline::running_pipe_count++;
    while (is_running(stop_token)) [[likely]] {
      std::optional opt_in_work = in_functor(stop_token);
      if (!is_running(stop_token)) [[unlikely]] {
        break;
      }
      SRTB_LOGD << tag << "popped work" << srtb::endl;
      auto in_work = opt_in_work.value();
      std::optional opt_out_work = pipe_functor(stop_token, in_work);
      if (!is_running(stop_token)) [[unlikely]] {
        break;
      }
      auto out_work = opt_out_work.value();
      out_functor(stop_token, out_work);
      SRTB_LOGD << tag << "pushed work" << srtb::endl;
    }
    srtb::pipeline::running_pipe_count--;
  }
};

template <typename Queue>
class queue_in_functor {
 protected:
  Queue& work_queue;
  using Work = typename Queue::work_type;

 public:
  explicit queue_in_functor(Queue& work_queue_) : work_queue{work_queue_} {}

  auto operator()(std::stop_token stop_token) {
    Work work;
    bool ret = work_queue.pop(work);
    if (!ret) [[unlikely]] {
      while (!ret) {
        if (stop_token.stop_requested()) [[unlikely]] {
          return std::optional<Work>{};
        }
        std::this_thread::sleep_for(
            std::chrono::nanoseconds(srtb::config.thread_query_work_wait_time));
        ret = work_queue.pop(work);
      }
    }
    return std::optional{work};
  }
};

template <typename Queue>
class queue_out_functor {
 protected:
  Queue& work_queue;
  using Work = typename Queue::work_type;

 public:
  explicit queue_out_functor(Queue& work_queue_) : work_queue{work_queue_} {}

  auto operator()(std::stop_token stop_token, Work work) {
    bool ret = work_queue.push(work);
    if (!ret) [[unlikely]] {
      while (!ret) {
        if (stop_token.stop_requested()) [[unlikely]] {
          return;
        }
        std::this_thread::sleep_for(
            std::chrono::nanoseconds(srtb::config.thread_query_work_wait_time));
        ret = work_queue.push(work);
      }
    }
  }
};

/**
  * @brief Start working in a new thread
  * @return std::jthread the thread running on.
  */
template <typename PipeFunctor, typename InFunctor, typename OutFunctor,
          typename... Args>
static std::jthread start_pipe(sycl::queue q, InFunctor in_functor,
                               OutFunctor out_functor, Args... args) {
  std::jthread jthread{
      [=](std::stop_token stop_token, Args... args) {
        // the pipe lives on its thread
        srtb::pipeline::pipe<PipeFunctor, InFunctor, OutFunctor> pipe{
            PipeFunctor{q, args...}, in_functor, out_functor};
        pipe.run(stop_token);
      },
      args...};
#if __has_include(<pthread.h>)
  const std::string thread_name = generate_thread_name<PipeFunctor>();
  pthread_setname_np(jthread.native_handle(), thread_name.c_str());
#else
#warning not setting thread name of pipe (TODO)
#endif
  return jthread;
}

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
