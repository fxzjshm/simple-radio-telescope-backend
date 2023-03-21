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
#ifndef __SRTB_PIPELINE_EXIT_HANDLER__
#define __SRTB_PIPELINE_EXIT_HANDLER__

#include <thread>
#include <vector>

#include "srtb/commons.hpp"

namespace srtb {
namespace pipeline {

/** 
 * @brief this function requests every thread to stop and wait for them
 *        when the program is about to (normally) exit.
 * @note extracted from srtb::gui::ExitHandler for no-GUI use
 * @see srtb::termination_handler for handler of unexpected exit
 */
inline void on_exit(std::vector<std::jthread> threads) {
  for (size_t i = 0; i < threads.size(); i++) {
    threads.at(i).request_stop();
  }
  size_t last_count = 0;
  while (srtb::pipeline::running_pipe_count != 0) {
    // sometimes program may stuck here due to deadlock,
    // but it's safe to Crrl+C (or SIGTERM, if Ctrl+C no effect)
    // at least better than nothing
    std::this_thread::sleep_for(
        std::chrono::nanoseconds(srtb::config.thread_query_work_wait_time));

    // deallocate memory early, in case some thread stucks
    const size_t current_count = srtb::pipeline::running_pipe_count;
    if (last_count != current_count) {
      srtb::device_allocator.deallocate_all_free_ptrs();
      srtb::host_allocator.deallocate_all_free_ptrs();
    }
  }
}

}  // namespace pipeline
}  // namespace srtb

#endif  // __SRTB_PIPELINE_EXIT_HANDLER__
