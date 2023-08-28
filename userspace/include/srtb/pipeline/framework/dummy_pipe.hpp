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
#ifndef __SRTB_PIPELINE_DUMMY_PIPE__
#define __SRTB_PIPELINE_DUMMY_PIPE__

#include <stop_token>

#include "srtb/work.hpp"

namespace srtb {
namespace pipeline {

/**
 * @brief this dummy pipe is a place holder to clean work queue.
 */
template <typename DummyWork = srtb::work::dummy_work>
class dummy_pipe {
 public:
  template <typename... Args>
  dummy_pipe([[maybe_unused]] Args... args){};

  template <typename Work>
  auto operator()([[maybe_unused]] std::stop_token stop_token,
                  [[maybe_unused]] Work work) {
    return std::optional{DummyWork{}};
  }
};

}  // namespace pipeline
}  // namespace srtb

#endif  // __SRTB_PIPELINE_DUMMY_PIPE__
