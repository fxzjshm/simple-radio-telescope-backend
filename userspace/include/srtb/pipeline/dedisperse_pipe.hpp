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
#include "srtb/pipeline/pipe.hpp"

namespace srtb {
namespace pipeline {

#if 0

/**
 * @brief codd = coherent dedispersion, fdfb = frequency domain filterbank
 *        this pipe applies coherent dedispertion and frequency domain filterbank
 *        to FFT-ed data.
 * @note the highest frequency channel is dropped
 * @deprecated not used now
 */
class dedisperse_and_channelize_pipe
    : public pipe<dedisperse_and_channelize_pipe> {
  friend pipe<dedisperse_and_channelize_pipe>;

 public:
  dedisperse_and_channelize_pipe() {}

 protected:
  void run_once_impl(std::stop_token stop_token) {
    srtb::work::dedisperse_and_channelize_work work;
    SRTB_POP_WORK_OR_RETURN(" [dedisperse & channelize pipe] ",
                            srtb::dedisperse_and_channelize_queue, work,
                            stop_token);
    // drop the highest frequency point
    const size_t N = work.count - 1;
    const size_t M = work.channel_count;
    // supported if N % M == 0;
    const size_t L = N / M;
    assert(L * M == N);
    // TODO: check this
    // baseband_sample_rate is samples/second, delta_freq is in MHz
    // assume Nyquist sample rate here
    const srtb::real df =
        static_cast<srtb::real>(work.baseband_sample_rate) / 2 / N / 1e6;
    auto& d_in_shared = work.ptr;
    auto d_out_shared =
        srtb::device_allocator.allocate_shared<srtb::complex<srtb::real> >(N);
    auto d_in = d_in_shared.get();
    auto d_out = d_out_shared.get();
    const srtb::real f_min = work.baseband_freq_low,
                     f_c = f_min + srtb::config.baseband_bandwidth;
    //srtb::coherent_dedispersion_and_frequency_domain_filterbank(
    //    d_in, d_out, f_min, f_c, df, work.dm, M, N, q);

    srtb::coherent_dedispersion::coherent_dedispertion(d_in, N, f_min, f_c, df,
                                                       work.dm, q);
    srtb::frequency_domain_filterbank::frequency_domain_filterbank(d_in, d_out,
                                                                   N, M, q);

    d_in = nullptr;
    d_in_shared.reset();

    srtb::work::refft_1d_c2c_work refft_1d_c2c_work;
    refft_1d_c2c_work.ptr = d_out_shared;
    refft_1d_c2c_work.count = N;
    refft_1d_c2c_work.timestamp = work.timestamp;
    SRTB_PUSH_WORK_OR_RETURN(" [dedisperse & channelize pipe] ", srtb::refft_1d_c2c_queue,
                   refft_1d_c2c_work, stop_token);


    //srtb::work::signal_detect_work signal_detect_work;
    //signal_detect_work.ptr = d_out_shared;
    //signal_detect_work.count = M;
    //signal_detect_work.batch_size = L;
    //signal_detect_work.timestamp = work.timestamp;
    //SRTB_PUSH_WORK_OR_RETURN(" [dedisperse & channelize_pipe] ",
    //                         srtb::signal_detect_queue, signal_detect_work,
    //                         stop_token);
    
  }
};

#endif  // 0

/**
 * @brief codd = coherent dedispersion
 *        this pipe applies coherent dedispertion to FFT-ed data.
 * @note the highest frequency channel is dropped
 */
class dedisperse_pipe : public pipe<dedisperse_pipe> {
  friend pipe<dedisperse_pipe>;

 public:
  dedisperse_pipe() = default;

 protected:
  void run_once_impl(std::stop_token stop_token) {
    srtb::work::dedisperse_work work;
    SRTB_POP_WORK_OR_RETURN(" [dedisperse pipe] ", srtb::dedisperse_queue, work,
                            stop_token);
    // drop the highest frequency point
    // *Drop*  -- LinusTechTips
    const size_t N = work.count - 1;
    // TODO: check this
    // baseband_sample_rate is samples/second, delta_freq is in MHz
    // assume Nyquist sastatic_cast<srtb::real>mple rate here
    const srtb::real df =
        static_cast<srtb::real>(work.baseband_sample_rate) / 2 / N / 1e6;
    auto& d_in_shared = work.ptr;
    auto d_in = d_in_shared.get();
    const srtb::real f_min = work.baseband_freq_low,
                     f_c = f_min + srtb::config.baseband_bandwidth;
    srtb::coherent_dedispersion::coherent_dedispertion(d_in, N, f_min, f_c, df,
                                                       work.dm, q);

    // shortcut
    //srtb::work::simplify_spectrum_work simplify_spectrum_work;
    //simplify_spectrum_work.ptr = d_in_shared;
    //simplify_spectrum_work.count = N;
    //simplify_spectrum_work.batch_size = 1;
    //SRTB_PUSH_WORK_OR_RETURN(" [dedisperse pipe] ", srtb::simplify_spectrum_queue,
    //               simplify_spectrum_work, stop_token);

    srtb::work::ifft_1d_c2c_work ifft_1d_c2c_work;
    ifft_1d_c2c_work.ptr = d_in_shared;
    ifft_1d_c2c_work.count = N;
    ifft_1d_c2c_work.timestamp = work.timestamp;
    SRTB_PUSH_WORK_OR_RETURN(" [dedisperse pipe] ", srtb::ifft_1d_c2c_queue,
                             ifft_1d_c2c_work, stop_token);
  }
};

}  // namespace pipeline
}  // namespace srtb

#endif  // __SRTB_PIPELINE_DEDISPERSE_PIPE__
