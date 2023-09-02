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
#ifndef __SRTB_PIPELINE_DEDISPERSE_PIPE__
#define __SRTB_PIPELINE_DEDISPERSE_PIPE__

#include "srtb/coherent_dedispersion.hpp"
#include "srtb/pipeline/framework/pipe.hpp"

namespace srtb {
namespace pipeline {

/**
 * @brief codd = coherent dedispersion
 *        this pipe applies coherent dedispertion to FFT-ed data.
 */
class dedisperse_pipe {
 public:
  sycl::queue q;

  auto operator()([[maybe_unused]] std::stop_token stop_token,
                  srtb::work::dedisperse_work work) {
    const size_t N = work.count;
    const srtb::real df = srtb::config.baseband_bandwidth / N;
    auto& d_in_shared = work.ptr;
    auto d_in = d_in_shared.get();
    const srtb::real f_min = srtb::config.baseband_freq_low,
                     f_c = f_min + srtb::config.baseband_bandwidth;
    const srtb::real dm = srtb::config.dm;
    srtb::coherent_dedispersion::coherent_dedispertion(d_in, N, f_min, f_c, df,
                                                       dm, q);

    srtb::work::ifft_1d_c2c_work ifft_1d_c2c_work;
    ifft_1d_c2c_work.move_parameter_from(std::move(work));
    ifft_1d_c2c_work.ptr = d_in_shared;
    ifft_1d_c2c_work.count = N;
    return std::optional{ifft_1d_c2c_work};
  }
};

}  // namespace pipeline
}  // namespace srtb

#endif  // __SRTB_PIPELINE_DEDISPERSE_PIPE__
