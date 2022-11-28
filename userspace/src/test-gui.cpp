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

#include <chrono>
#include <iostream>
#include <thread>

#include "srtb/commons.hpp"
#include "srtb/gui/gui.hpp"
#include "srtb/spectrum/simplify_spectrum.hpp"

// TODO: move to tests and auto skip if "headless"
auto start_gui_test() {
  return std::jthread{[](std::stop_token stop_token) {
    while (true) {
      const auto in_size = srtb::gui::spectrum::width;
      auto h_in_shared =
          srtb::host_allocator.allocate_shared<srtb::real>(in_size);
      auto h_in = h_in_shared.get();
      std::generate(h_in, h_in + in_size, []() {
        // cannot be narrowed using initializion list
        return static_cast<srtb::real>(std::rand()) /
               static_cast<srtb::real>(INT_MAX);
      });
      srtb::work::draw_spectrum_work draw_spectrum_work;
      draw_spectrum_work.ptr = h_in_shared;
      draw_spectrum_work.count = in_size;
      draw_spectrum_work.batch_size = 1;
      draw_spectrum_work.timestamp = 0;
      SRTB_PUSH_WORK_OR_RETURN(" [gui_test] ", srtb::draw_spectrum_queue,
                               draw_spectrum_work, stop_token);
      using namespace std::chrono_literals;
      std::this_thread::sleep_for(10ms);
      std::this_thread::yield();
    }
  }};
}

int main(int argc, char **argv) {
  std::jthread test_thread = start_gui_test();
  std::vector<std::jthread> threads;
  threads.push_back(std::move(test_thread));
  return srtb::gui::show_gui(argc, argv, std::move(threads));
}
