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
#ifndef __SRTB_PIPELINE_UNPACK_PIPE__
#define __SRTB_PIPELINE_UNPACK_PIPE__

#include "srtb/fft/fft_window.hpp"
#include "srtb/pipeline/pipe.hpp"
#include "srtb/unpack.hpp"

namespace srtb {
namespace pipeline {

/**
 * @brief this pipe reads from @c srtb::unpack_queue, unpack and apply FFT window
 *        to input baseband data, then push work to @c srtb::fft_1d_r2c_queue
 */
class unpack_pipe : public pipe<unpack_pipe> {
  friend pipe<unpack_pipe>;

 protected:
  srtb::fft::fft_window_functor_manager<> window_functor_manager;

 public:
  unpack_pipe()
      : window_functor_manager{srtb::fft::default_window{},
                               /* n = */ srtb::config.baseband_input_length *
                                   srtb::BITS_PER_BYTE /
                                   srtb::config.baseband_input_bits,
                               q} {}

 protected:
  void run_once_impl() {
    srtb::work::unpack_work unpack_work;
    SRTB_POP_WORK(" [unpack pipe] ", srtb::unpack_queue, unpack_work);
    // data length after unpack
    size_t out_count = unpack_work.count * srtb::BITS_PER_BYTE /
                       unpack_work.baseband_input_bits;

    // re-construct fft_window_functor_manager if length mismatch
    if (out_count != window_functor_manager.functor.n) [[unlikely]] {
      SRTB_LOGW << " [unpack pipe] "
                << "re-construct fft_window_functor_manager of size "
                << out_count << srtb::endl;
      window_functor_manager = srtb::fft::fft_window_functor_manager{
          srtb::fft::default_window{}, out_count, q};
    }

    auto d_out_shared =
        srtb::device_allocator.allocate_shared<srtb::real>(out_count);
    auto d_in = unpack_work.ptr.get();
    auto d_out = d_out_shared.get();
    srtb::unpack::unpack(unpack_work.baseband_input_bits, d_in, d_out,
                         /* in_count =  */ unpack_work.count,
                         window_functor_manager.functor, q);
    srtb::work::fft_1d_r2c_work fft_1d_r2c_work;
    fft_1d_r2c_work.ptr = d_out_shared;
    fft_1d_r2c_work.count = out_count;
    SRTB_PUSH_WORK(" [unpack pipe] ", srtb::fft_1d_r2c_queue, fft_1d_r2c_work);
  }
};

}  // namespace pipeline
}  // namespace srtb

#endif  // __SRTB_PIPELINE_UNPACK_PIPE__
