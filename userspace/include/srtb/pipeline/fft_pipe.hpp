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
  friend pipe<fft_1d_r2c_pipe>;

 protected:
  srtb::fft::fft_1d_dispatcher<srtb::fft::type::R2C_1D> dispatcher;

 public:
  fft_1d_r2c_pipe()
      : dispatcher{/* n = */
                   srtb::config.baseband_input_length * srtb::BITS_PER_BYTE /
                       srtb::config.baseband_input_bits,
                   /* batch_size = */ 1, q} {}

 protected:
  void run_once_impl() {
    srtb::work::fft_1d_r2c_work fft_1d_r2c_work;
    SRTB_POP_WORK(" [fft 1d r2c pipe] ", srtb::fft_1d_r2c_queue,
                  fft_1d_r2c_work);
    const size_t in_count = fft_1d_r2c_work.count;
    const size_t out_count = in_count / 2 + 1;
    // reset FFT plan if mismatch
    if (dispatcher.get_n() != in_count || dispatcher.get_batch_size() != 1)
        [[unlikely]] {
      dispatcher.set_size(in_count, 1);
    }
    auto d_in_shared = fft_1d_r2c_work.ptr;
    auto d_out_shared =
        srtb::device_allocator.allocate_shared<srtb::complex<srtb::real> >(
            out_count);
    dispatcher.process(d_in_shared.get(), d_out_shared.get());
    // TODO: next pipe
    // temporary work: spectrum analyzer
    srtb::work::simplify_spectrum_work simplify_spectrum_work{d_out_shared,
                                                              out_count};
    SRTB_PUSH_WORK(" [fft 1d r2c pipe] ", srtb::simplify_spectrum_queue,
                   simplify_spectrum_work);
  }
};

/**
 * @brief This pipe reads from @c srtb::fft_1d_r2c_queue , perform FFT by calling
 *        @c srtb::fft::dispatch_1d_r2c , then push result to ( TODO: @c srtb::frequency_domain_filterbank_queue )
 */
class ifft_1d_c2c_pipe : public pipe<ifft_1d_c2c_pipe> {
  friend pipe<ifft_1d_c2c_pipe>;

 protected:
  srtb::fft::fft_1d_dispatcher<srtb::fft::type::C2C_1D_BACKWARD> dispatcher;

 public:
  ifft_1d_c2c_pipe()
      : dispatcher{/* n = */
                   srtb::config.baseband_input_length * srtb::BITS_PER_BYTE /
                       srtb::config.baseband_input_bits,
                   /* batch_size = */ srtb::config.ifft_channel_count, q} {}

 protected:
  void run_once_impl() {
    srtb::work::ifft_1d_c2c_work ifft_1d_c2c_work;
    SRTB_POP_WORK(" [ifft 1d c2c pipe] ", srtb::ifft_1d_c2c_queue,
                  ifft_1d_c2c_work);
    const size_t count = ifft_1d_c2c_work.count;
    const size_t batch_size = ifft_1d_c2c_work.batch_size;
    // reset FFT plan if mismatch
    if (dispatcher.get_n() != count ||
        dispatcher.get_batch_size() != batch_size) [[unlikely]] {
      dispatcher.set_size(count, batch_size);
    }
    auto d_in_shared = ifft_1d_c2c_work.ptr;
    auto d_out_shared =
        srtb::device_allocator.allocate_shared<srtb::complex<srtb::real> >(
            count);
    dispatcher.process(d_in_shared.get(), d_out_shared.get());
    // TODO: next pipe
    // temporary work: spectrum analyzer
    srtb::work::simplify_spectrum_work simplify_spectrum_work{d_out_shared,
                                                              count};
    SRTB_PUSH_WORK(" [ifft 1d c2c pipe] ", srtb::simplify_spectrum_queue,
                   simplify_spectrum_work);
  }
};

}  // namespace pipeline
}  // namespace srtb

#endif  // __SRTB_PIPELINE_FFT_PIPE__
