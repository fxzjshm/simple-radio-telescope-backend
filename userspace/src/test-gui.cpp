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
#include <memory>
#include <thread>

#include "mdspan/mdspan.hpp"
// ---
#include "srtb/config.hpp"
#include "srtb/global_variables.hpp"
#include "srtb/gui/gui.hpp"
#include "srtb/memory/mem.hpp"
#include "srtb/pipeline/framework/pipe_io.hpp"
#include "srtb/spectrum/simplify_spectrum.hpp"
#include "srtb/util/assert.hpp"

int main(int argc, char **argv) {
  auto draw_spectrum_queue_2 = std::make_shared<srtb::work_queue<srtb::work::draw_spectrum_work_2>>();

  // cellular automaton
  std::jthread test_thread = std::jthread{[&draw_spectrum_queue_2](std::stop_token stop_token) {
    constexpr size_t k = 4;  // down sample
    const auto pixmap_width = srtb::config.gui_pixmap_width;
    const auto pixmap_height = srtb::config.gui_pixmap_width;
    const auto width = pixmap_width / k;
    const auto height = pixmap_height / k;

    auto h_in_shared = srtb::mem_allocate_shared<uint8_t>(&srtb::host_allocator, width * height);
    auto h_in = h_in_shared.get_span();
    auto d_map_1_shared = srtb::mem_allocate_shared<uint8_t>(&srtb::device_allocator, width * height);
    auto d_map_1 = d_map_1_shared.get_mdspan(height, width);
    auto d_map_2_shared = srtb::mem_allocate_shared<uint8_t>(&srtb::device_allocator, width * height);
    auto d_map_2 = d_map_2_shared.get_mdspan(height, width);
    auto d_pixmap_shared = srtb::mem_allocate_shared<uint32_t>(&srtb::device_allocator, pixmap_width * pixmap_height);
    auto d_pixmap = d_pixmap_shared.get_mdspan(pixmap_height, pixmap_width);
    auto h_pixmap_shared = srtb::mem_allocate_shared<uint32_t>(&srtb::host_allocator, pixmap_width * pixmap_height);

    std::generate(h_in.data(), h_in.data() + width * height, []() { return static_cast<uint8_t>(std::rand() & 1); });

    sycl::queue q;

    q.copy(h_in_shared.ptr.get(), /* -> */ d_map_1_shared.ptr.get(), h_in_shared.count).wait();

    auto oldMapBool = d_map_1;
    auto newMapBool = d_map_2;

    size_t frame_count = 0;

    while (true) {
      q.parallel_for(sycl::range<2>{width, height}, [=](sycl::item<2> id) {
         const auto x = id.get_id(0);
         const auto y = id.get_id(1);
         const auto xm1 = (x - 1) % width;
         const auto xp1 = (x + 1) % width;
         const auto ym1 = (y - 1) % height;
         const auto yp1 = (y + 1) % height;

         int8_t neighbourCount = 0;

         // @off
         // @formatter:off
         // clang-format off
         neighbourCount += oldMapBool[ym1, xm1];
         neighbourCount += oldMapBool[ym1,  x ];
         neighbourCount += oldMapBool[ym1, xp1];
         neighbourCount += oldMapBool[ y , xm1];
         neighbourCount += oldMapBool[ y , xp1];
         neighbourCount += oldMapBool[yp1, xm1];
         neighbourCount += oldMapBool[yp1,  x ];
         neighbourCount += oldMapBool[yp1, xp1];
         // @on
         // @formatter:on
         // clang-format on

         if (3 == neighbourCount)
           newMapBool[y, x] = 1;
         else if (2 == neighbourCount)
           newMapBool[y, x] = oldMapBool[y, x];
         else
           newMapBool[y, x] = 0;

         const uint32_t color = newMapBool[y, x] == 1 ? srtb::gui::color_1 : srtb::gui::color_0;
#pragma unroll
         for (size_t j = k * y; j < k * (y + 1); j++) {
#pragma unroll
           for (size_t i = k * x; i < k * (x + 1); i++) {
             d_pixmap[j, i] = color;
           }
         }
       }).wait();
      q.copy(d_pixmap_shared.ptr.get(), /* -> */ h_pixmap_shared.ptr.get(), d_pixmap_shared.count).wait();
      std::swap(oldMapBool, newMapBool);

      SRTB_LOGI << " [test-gui] " << "frame_count = " << frame_count << srtb::endl;

      srtb::work::draw_spectrum_work_2 draw_spectrum_work;
      draw_spectrum_work.ptr = h_pixmap_shared.ptr;
      draw_spectrum_work.data_stream_id = 0;
      draw_spectrum_work.width = pixmap_width;
      draw_spectrum_work.height = pixmap_height;
      srtb::pipeline::queue_out_functor{draw_spectrum_queue_2}(stop_token, draw_spectrum_work);

      using namespace std::chrono_literals;
      std::this_thread::sleep_for(250ms);
      std::this_thread::yield();

      if (stop_token.stop_requested()) [[unlikely]] {
        return;
      }

      frame_count += 1;
    }
  }};
  std::vector<std::jthread> threads;
  threads.push_back(std::move(test_thread));
  return srtb::gui::show_gui(argc, argv, draw_spectrum_queue_2);
}
