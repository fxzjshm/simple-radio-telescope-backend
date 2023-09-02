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
#ifndef __SRTB_PIPELINE_COMPOSITE_PIPE__
#define __SRTB_PIPELINE_COMPOSITE_PIPE__

#include <stop_token>

#include "srtb/pipeline/framework/pipe.hpp"

namespace srtb {
namespace pipeline {

/**
 * @brief this pipe serializes multiple pipes
 */
template <typename Pipe1, typename... Pipes>
class composite_pipe {
 public:
  Pipe1 pipe_1;
  composite_pipe<Pipes...> pipes;

  // TODO: support arguments
  explicit composite_pipe(sycl::queue q_) : pipe_1{q_}, pipes{q_} {}

  auto operator()(std::stop_token stop_token, auto in_work) {
    std::optional opt_work = pipe_1(stop_token, in_work);
    using out_type = decltype(pipes(stop_token, opt_work.value()));
    if (opt_work) [[likely]] {
      return pipes(stop_token, opt_work.value());
    } else [[unlikely]] {
      return out_type{};
    }
  }
};

template <typename Pipe>
class composite_pipe<Pipe> : public Pipe {};

inline namespace detail {

template <typename Pipe1, typename... Pipes>
struct type_helper<composite_pipe<Pipe1, Pipes...> > {
  inline auto class_name() -> std::string {
    return type_helper<Pipe1>{}.class_name() + ", " +
           type_helper<Pipes...>{}.class_name();
  }
};

template <typename Pipe>
struct type_helper<composite_pipe<Pipe> > {
  inline auto class_name() -> std::string {
    return type_helper<Pipe>{}.class_name();
  }
};

}  // namespace detail

}  // namespace pipeline
}  // namespace srtb

#endif  // __SRTB_PIPELINE_COMPOSITE_PIPE__
