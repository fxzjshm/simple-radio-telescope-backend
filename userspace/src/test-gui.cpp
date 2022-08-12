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
#include "srtb/spectrum.hpp"

// TODO: move to tests and auto skip if "headless"
void start_gui_test() {
  std::jthread{[]() {
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
      srtb::work::draw_spectrum_work draw_spectrum_work{h_in_shared, in_size};
      SRTB_PUSH_WORK(" [gui_test] ", srtb::draw_spectrum_queue,
                     draw_spectrum_work);
      using namespace std::chrono_literals;
      std::this_thread::sleep_for(10ms);
      std::this_thread::yield();
    }
  }}.detach();
}

int main(int argc, char **argv) {
  start_gui_test();
  return srtb::gui::show_gui(argc, argv);
}
