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
#include "srtb/spectrum.hpp"

namespace srtb {
namespace pipeline {

/**
 * @brief This temporary pipe reads FFT-ed data and adapt it to a line on GUI pixmap.
 */
class simplify_spectrum_pipe : public pipe<simplify_spectrum_pipe> {
  friend pipe<simplify_spectrum_pipe>;

 public:
  simplify_spectrum_pipe() {}

 protected:
  void run_once_impl() {
    const auto sum_count = srtb::config.simplify_spectrum_sum_count;
    const size_t out_count = srtb::gui::spectrum::width;
    auto d_sum_shared =
        srtb::device_allocator.allocate_shared<srtb::real>(out_count);
    auto d_sum = d_sum_shared.get();
    q.fill<srtb::real>(d_sum, srtb::real{0}, out_count).wait();
    auto h_out_shared =
        srtb::host_allocator.allocate_shared<srtb::real>(out_count);
    auto h_out = h_out_shared.get();

    SRTB_LOGD << " [simplify spectrum pipe] "
              << " start simplifying" << srtb::endl;
    srtb::work::simplify_spectrum_work simplify_spectrum_work;
    for (size_t i = 0; i < sum_count; i++) {
      SRTB_POP_WORK(" [simplify spectrum pipe] ", srtb::simplify_spectrum_queue,
                    simplify_spectrum_work);
      const size_t in_count = simplify_spectrum_work.count;
      auto d_in_shared = simplify_spectrum_work.ptr;
      auto d_in = d_in_shared.get();
      srtb::spectrum::simplify_spectrum_norm_and_sum(d_in, in_count, d_sum,
                                                     out_count, q);
    }
    srtb::spectrum::simplify_spectrum_normalize(d_sum, out_count, q);
    SRTB_LOGD << " [simplify spectrum pipe] "
              << " finished simplifying" << srtb::endl;

    q.copy(d_sum, /* -> */ h_out, out_count).wait();
    srtb::work::draw_spectrum_work draw_spectrum_work{h_out_shared, out_count};
    SRTB_PUSH_WORK(" [simplify spectrum pipe] ", srtb::draw_spectrum_queue,
                   draw_spectrum_work);
  }
};

}  // namespace pipeline
}  // namespace srtb

#endif  // __SRTB_PIPELINE_SPECTRUM_PIPE__
