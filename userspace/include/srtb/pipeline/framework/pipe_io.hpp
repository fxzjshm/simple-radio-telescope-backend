/******************************************************************************* 
 * Copyright (c) 2023 fxzjshm
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
#ifndef __SRTB_PIPELINE_PIPE_IO__
#define __SRTB_PIPELINE_PIPE_IO__

#include <memory>
#include <stop_token>
#include <thread>

#include "srtb/commons.hpp"

namespace srtb {
namespace pipeline {

/** @brief This functor reads input work from work queue */
template <typename QueuePtr>
class queue_in_functor {
 protected:
  QueuePtr work_queue;
  using Work = typename std::pointer_traits<QueuePtr>::element_type::work_type;

 public:
  explicit queue_in_functor(QueuePtr work_queue_) : work_queue{work_queue_} {}

  auto operator()(std::stop_token stop_token) {
    Work work;
    bool ret = work_queue->pop(work);
    if (!ret) [[unlikely]] {
      while (!ret) {
        if (stop_token.stop_requested()) [[unlikely]] {
          return std::optional<Work>{};
        }
        std::this_thread::sleep_for(
            std::chrono::nanoseconds(srtb::config.thread_query_work_wait_time));
        ret = work_queue->pop(work);
      }
    }
    return std::optional{work};
  }
};

/** @brief This functor pushes output work to work queue */
template <typename QueuePtr>
class queue_out_functor {
 protected:
  QueuePtr work_queue;
  using Work = typename std::pointer_traits<QueuePtr>::element_type::work_type;

 public:
  explicit queue_out_functor(QueuePtr work_queue_) : work_queue{work_queue_} {}

  auto operator()(std::stop_token stop_token, Work work) {
    bool ret = work_queue->push(work);
    if (!ret) [[unlikely]] {
      while (!ret) {
        if (stop_token.stop_requested()) [[unlikely]] {
          return;
        }
        std::this_thread::sleep_for(
            std::chrono::nanoseconds(srtb::config.thread_query_work_wait_time));
        ret = work_queue->push(work);
      }
    }
  }
};

/** @brief This functor pushes output work to work queue, but only try once */
template <typename QueuePtr>
class loose_queue_out_functor {
 protected:
  QueuePtr work_queue;
  using Work = typename std::pointer_traits<QueuePtr>::element_type::work_type;

 public:
  explicit loose_queue_out_functor(QueuePtr work_queue_)
      : work_queue{work_queue_} {}

  auto operator()(std::stop_token stop_token, Work work) {
    if (!stop_token.stop_requested()) [[likely]] {
      work_queue->push(work);
    }
  }
};

/** @brief This functor copies work and give it to multiple out functors. */
template <typename... OutFunctors>
class multiple_out_functors_functor {
  // ref: https://stackoverflow.com/questions/54631547/using-stdapply-with-variadic-packs/54631721
 public:
  std::tuple<OutFunctors...> out_functors;

  explicit multiple_out_functors_functor(OutFunctors... out_functors_)
      : out_functors{out_functors_...} {}

  template <typename Work>
  auto operator()(std::stop_token stop_token, Work work) {
    std::apply(
        [=](auto&... out_functor) { (out_functor(stop_token, work), ...); },
        out_functors);
  }
};

/** 
 * @brief This functor iterates over container and extract all of them.
 *        Useful if a pipe input 1 work but output multiple works
 */
template <typename OutFunctor>
class multiple_works_out_functor {
  // ref: https://stackoverflow.com/questions/54631547/using-stdapply-with-variadic-packs/54631721
 public:
  OutFunctor out_functor;

#if defined(__clang__) && __clang_major__ <= 15
  explicit multiple_works_out_functor(OutFunctor out_functor_)
      : out_functor{out_functor_} {}
#endif

  template <typename WorkContainer>
  auto operator()(std::stop_token stop_token, WorkContainer work_container) {
    for (auto&& work : work_container) {
      if (stop_token.stop_requested()) {
        return;
      }
      out_functor(stop_token, work);
    }
  }
};

template <typename T = srtb::work::dummy_work>
class dummy_in_functor {
 public:
  auto operator()([[maybe_unused]] std::stop_token) {
    return std::optional{T{}};
  }
};

template <typename T = srtb::work::dummy_work>
class dummy_out_functor {
 public:
  auto operator()([[maybe_unused]] std::stop_token, [[maybe_unused]] T) {}
};

}  // namespace pipeline
}  // namespace srtb

#endif  // __SRTB_PIPELINE_PIPE_IO__
