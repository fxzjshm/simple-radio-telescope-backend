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
#ifndef __SRTB_PIPELINE_DEDISPERSE_AND_CHANNELIZE_PIPE__
#define __SRTB_PIPELINE_DEDISPERSE_AND_CHANNELIZE_PIPE__

#include "srtb/coherent_dedispersion.hpp"
#include "srtb/frequency_domain_filterbank.hpp"
#include "srtb/pipeline/pipe.hpp"

namespace srtb {
namespace pipeline {

/**
 * @brief codd = coherent dedispersion, fdfb = frequency domain filterbank
 *        this pipe applies coherent dedispertion and frequency domain filterbank
 *        to FFT-ed data.
 * @note the highest frequency channel is dropped
 * TODO: check this
 */
class dedisperse_and_channelize_pipe
    : public pipe<dedisperse_and_channelize_pipe> {
  friend pipe<dedisperse_and_channelize_pipe>;

 public:
  dedisperse_and_channelize_pipe() {}

 protected:
  void run_once_impl() {
    srtb::work::dedisperse_and_channelize_work work;
    SRTB_POP_WORK(" [dedisperse & channelize pipe] ",
                  srtb::dedisperse_and_channelize_queue, work);
    // drop the highest frequency point
    const size_t N = work.count - 1;
    // not using frequency domain filterbank now
    //const size_t M = work.channel_count;
    const size_t M = 1;
    // supported if N % M == 0;
    const size_t n = N / M;
    assert(n * M == N);
    // TODO: check this
    const srtb::real delta_f = work.baseband_sample_rate / N;
    auto d_in_shared = work.ptr;
    auto d_out_shared =
        srtb::device_allocator.allocate_shared<srtb::complex<srtb::real> >(N);
    auto d_in = d_in_shared.get();
    auto d_out = d_out_shared.get();
    srtb::coherent_dedispersion_and_frequency_domain_filterbank(
        d_in, d_out, work.baseband_freq_low, delta_f, work.dm, M, N, q);

    srtb::work::refft_1d_c2c_work refft_1d_c2c_work{{d_out_shared, n},
                                                    srtb::config.refft_length};
    SRTB_PUSH_WORK(" [dedisperse & channelize pipe] ", srtb::refft_1d_c2c_queue,
                   refft_1d_c2c_work);
  }
};

}  // namespace pipeline
}  // namespace srtb

#endif  // __SRTB_PIPELINE_DEDISPERSE_AND_CHANNELIZE_PIPE__
