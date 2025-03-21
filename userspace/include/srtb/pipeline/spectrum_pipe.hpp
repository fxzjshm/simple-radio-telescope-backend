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

#include <boost/algorithm/string/replace.hpp>
#include <string>

#include "srtb/pipeline/framework/pipe.hpp"
#include "srtb/spectrum/simplify_spectrum.hpp"

namespace srtb {
namespace pipeline {

/**
 * @brief This temporary pipe reads FFT-ed data and adapt it to lines on GUI pixmap.
 */
class simplify_spectrum_pipe {
 protected:
  sycl::queue q;

 public:
  auto operator()([[maybe_unused]] std::stop_token stop_token,
                  srtb::work::simplify_spectrum_work simplify_spectrum_work) {
    const size_t in_count = simplify_spectrum_work.count;
    const size_t out_count = srtb::config.gui_pixmap_width;
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
    // using average value
    srtb::spectrum::simplify_spectrum_normalize_with_average_value(
        d_out, total_out_count, q);

    SRTB_LOGD << " [simplify spectrum pipe] "
              << " finished simplifying" << srtb::endl;

    q.copy(d_out, /* -> */ h_out, total_out_count).wait();
    d_out = nullptr;
    d_out_shared.reset();

    srtb::work::draw_spectrum_work draw_spectrum_work;
    draw_spectrum_work.move_parameter_from(std::move(simplify_spectrum_work));
    draw_spectrum_work.ptr = h_out_shared;
    draw_spectrum_work.count = out_count;
    draw_spectrum_work.batch_size = batch_size;

    return std::optional{draw_spectrum_work};
  }
};

/**
 * @brief This temporary pipe reads FFT-ed data and generate thumbnail of the spectrum to draw it on GUI pixmap.
 */
class simplify_spectrum_pipe_2 {
 public:
  sycl::queue q;

 public:
  auto operator()([[maybe_unused]] std::stop_token stop_token,
                  srtb::work::simplify_spectrum_work simplify_spectrum_work) {
    const size_t in_width = simplify_spectrum_work.count;
    const size_t in_height = simplify_spectrum_work.batch_size;
    const size_t out_width = srtb::config.gui_pixmap_width;
    const size_t out_height = srtb::config.gui_pixmap_height;
    const size_t total_out_count = out_width * out_height;
    auto& d_in_shared = simplify_spectrum_work.ptr;
    auto d_out_shared =
        srtb::device_allocator.allocate_shared<srtb::real>(total_out_count);
    auto d_in = d_in_shared.get();
    auto d_out = d_out_shared.get();

    SRTB_LOGD << " [simplify spectrum pipe] "
              << " start simplifying" << srtb::endl;

    srtb::spectrum::resample_spectrum_3(
        d_in, in_width, in_height,
        /* transform = */
        [](auto&& x) {
          // use log(norm(x)) if needed
          return srtb::norm(x);
        },
        d_out, out_width, out_height, q);
    d_in = nullptr;
    d_in_shared.reset();

    // normalization, so that it can be drawn onto image
    // using average value
    srtb::spectrum::simplify_spectrum_normalize_with_average_value(
        d_out, total_out_count, q);

    auto d_image_shared =
        srtb::device_allocator.allocate_shared<uint32_t>(total_out_count);
    auto d_image = d_image_shared.get();
    auto h_image_shared =
        srtb::host_allocator.allocate_shared<uint32_t>(total_out_count);
    auto h_image = h_image_shared.get();

    srtb::spectrum::generate_pixmap(d_out, d_image, out_width, out_height, srtb::gui::color_0, srtb::gui::color_1,
                                    srtb::gui::color_overflow, q);

    SRTB_LOGD << " [simplify spectrum pipe] "
              << " finished simplifying" << srtb::endl;

    q.copy(d_image, /* -> */ h_image, total_out_count).wait();
    d_out = nullptr;
    d_out_shared.reset();

    srtb::work::draw_spectrum_work_2 draw_spectrum_work{
        .ptr = h_image_shared,
        .data_stream_id = simplify_spectrum_work.data_stream_id,
        .width = out_width,
        .height = out_height};
    return std::optional{draw_spectrum_work};
  }
};

}  // namespace pipeline
}  // namespace srtb

#endif  // __SRTB_PIPELINE_SPECTRUM_PIPE__
