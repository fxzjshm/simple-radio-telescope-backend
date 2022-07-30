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
#ifndef __SRTB_PIPELINE_FFT_PIPE__
#define __SRTB_PIPELINE_FFT_PIPE__

#include "srtb/fft/fft.hpp"
#include "srtb/pipeline/pipe.hpp"

namespace srtb {
namespace pipeline {

/**
 * @brief This pipe reads from @c srtb::fft_1d_r2c_queue , perform FFT by calling
 *        @c srtb::fft::dispatch_1d_r2c , then push result to ( TODO: @c srtb::frequency_domain_filterbank_queue )
 */
class fft_1d_r2c_pipe : public pipe<fft_1d_r2c_pipe> {
 public:
  fft_1d_r2c_pipe() {
    srtb::fft::init_1d_r2c(srtb::fft::default_fft_1d_r2c_input_size(), q);
  }

 protected:
  void run_once_impl() {
    srtb::work::fft_1d_r2c_work fft_1d_r2c_work;
    SRTB_POP_WORK(" [fft 1d r2c pipe] ", srtb::fft_1d_r2c_queue,
                  fft_1d_r2c_work);
    size_t n = fft_1d_r2c_work.count;
    // reset FFT plan if mismatch
    if (srtb::fft::get_size_1d_r2c() != n) [[unlikely]] {
      srtb::fft::init_1d_r2c(n, q);
    }
    auto d_in_shared = fft_1d_r2c_work.ptr;
    auto d_out_shared =
        srtb::device_allocator.allocate_shared<srtb::complex<srtb::real> >(
            n / 2 + 1);
    srtb::fft::dispatch_1d_r2c(d_in_shared.get(), d_out_shared.get());
    // TODO: next pipe
  }
};

}  // namespace pipeline
}  // namespace srtb

#endif  // __SRTB_PIPELINE_FFT_PIPE__
