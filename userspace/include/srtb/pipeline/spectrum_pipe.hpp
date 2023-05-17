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
#ifndef __SRTB_PIPELINE_SPECTRUM_PIPE__
#define __SRTB_PIPELINE_SPECTRUM_PIPE__

#include "srtb/gui/spectrum_image_provider.hpp"
#include "srtb/pipeline/pipe.hpp"
#include "srtb/spectrum/simplify_spectrum.hpp"

namespace srtb {
namespace pipeline {

/**
 * @brief This temporary pipe reads FFT-ed data and adapt it to lines on GUI pixmap.
 */
class simplify_spectrum_pipe : public pipe<simplify_spectrum_pipe> {
  friend pipe<simplify_spectrum_pipe>;

 public:
  simplify_spectrum_pipe() = default;

 protected:
  void run_once_impl(std::stop_token stop_token) {
    srtb::work::simplify_spectrum_work simplify_spectrum_work;
    SRTB_POP_WORK_OR_RETURN(" [simplify spectrum pipe] ",
                            srtb::simplify_spectrum_queue,
                            simplify_spectrum_work, stop_token);

    const size_t in_count = simplify_spectrum_work.count;
    const size_t out_count = srtb::gui::spectrum::width;
    const size_t batch_size = simplify_spectrum_work.batch_size;
    const size_t total_out_count = out_count * batch_size;
    auto& d_in_shared = simplify_spectrum_work.ptr;
    auto d_out_shared =
        srtb::device_allocator.allocate_shared<srtb::real>(total_out_count);
    auto h_out_shared =
        srtb::host_allocator.allocate_shared<srtb::real>(total_out_count);
    auto d_in = d_in_shared.get();
    auto d_out = d_out_shared.get();
    auto h_out = h_out_shared.get();

    SRTB_LOGD << " [simplify spectrum pipe] "
              << " start simplifying" << srtb::endl;

    srtb::spectrum::simplify_spectrum_calculate_norm(d_in, in_count, d_out,
                                                     out_count, batch_size, q);
    d_in = nullptr;
    d_in_shared.reset();

    // normalization, so that it can be drawn onto image
    // choice 1: normalize all together using max value
    //srtb::spectrum::simplify_spectrum_normalize_with_max_value(
    //    d_out, total_out_count, 1, q);
    // choice 2: normalize per spectrum using max value
    //srtb::spectrum::simplify_spectrum_normalize_with_max_value(d_out, out_count,
    //                                                           batch_size, q);
    // choice 3: normalize all together using average value
    srtb::spectrum::simplify_spectrum_normalize_with_average_value(
        d_out, total_out_count, q);

    SRTB_LOGD << " [simplify spectrum pipe] "
              << " finished simplifying" << srtb::endl;

    q.copy(d_out, /* -> */ h_out, total_out_count).wait();
    d_out = nullptr;
    d_out_shared.reset();

    srtb::work::draw_spectrum_work draw_spectrum_work;
    draw_spectrum_work.ptr = h_out_shared;
    draw_spectrum_work.count = out_count;
    draw_spectrum_work.batch_size = batch_size;
    draw_spectrum_work.baseband_data = simplify_spectrum_work.baseband_data;
    draw_spectrum_work.timestamp = simplify_spectrum_work.timestamp;
    draw_spectrum_work.udp_packet_counter =
        simplify_spectrum_work.udp_packet_counter;
    // same as SRTB_PUSH_WORK_OR_RETURN(), but supressed logs
    {
      bool success = false;
      do {
        if (stop_token.stop_requested()) [[unlikely]] {
          return;
        }
        std::this_thread::sleep_for(
            std::chrono::nanoseconds(srtb::config.thread_query_work_wait_time));
        success = srtb::draw_spectrum_queue.push(draw_spectrum_work);
      } while (!success);
    }
  }
};

}  // namespace pipeline
}  // namespace srtb

#endif  // __SRTB_PIPELINE_SPECTRUM_PIPE__
