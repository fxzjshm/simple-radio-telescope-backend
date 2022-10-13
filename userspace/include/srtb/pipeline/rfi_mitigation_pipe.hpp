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
#ifndef __SRTB_PIPELINE_RFI_MITIGATION_PIPE__
#define __SRTB_PIPELINE_RFI_MITIGATION_PIPE__

#include "srtb/pipeline/pipe.hpp"
#include "srtb/spectrum/rfi_mitigation.hpp"

namespace srtb {
namespace pipeline {

class rfi_mitigation_pipe : public pipe<rfi_mitigation_pipe> {
  friend pipe<rfi_mitigation_pipe>;

 protected:
  void run_once_impl() {
    srtb::work::rfi_mitigation_work rfi_mitigation_work;
    SRTB_POP_WORK(" [rfi mitigation pipe] ", srtb::rfi_mitigation_queue,
                  rfi_mitigation_work);

    auto d_in_shared = rfi_mitigation_work.ptr;
    auto d_in = d_in_shared.get();
    const size_t in_count = rfi_mitigation_work.count;

    srtb::spectrum::mitigate_rfi(d_in, in_count, q);

    // shortcut
    //srtb::work::simplify_spectrum_work simplify_spectrum_work;
    //simplify_spectrum_work.ptr = d_in_shared;
    //simplify_spectrum_work.count = in_count;
    //simplify_spectrum_work.batch_size = 1;
    //SRTB_PUSH_WORK(" [rfi mitigation pipe] ", srtb::simplify_spectrum_queue,
    //               simplify_spectrum_work);

    srtb::work::dedisperse_and_channelize_work out_work;
    out_work.ptr = d_in_shared;
    out_work.count = in_count;
    out_work.baseband_freq_low = srtb::config.baseband_freq_low;
    out_work.baseband_sample_rate = srtb::config.baseband_sample_rate;
    out_work.channel_count = srtb::config.ifft_channel_count;
    out_work.dm = srtb::config.dm;
    SRTB_PUSH_WORK(" [rfi mitigation pipe] ",
                   srtb::dedisperse_and_channelize_queue, out_work);
  }
};

}  // namespace pipeline
}  // namespace srtb

#endif  // __SRTB_PIPELINE_RFI_MITIGATION_PIPE__
