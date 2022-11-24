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
  srtb::fft::fft_window_functor_manager<srtb::real, srtb::fft::default_window>
      window_functor_manager;

 public:
  unpack_pipe()
      : window_functor_manager{srtb::fft::default_window{},
                               /* n = */ srtb::config.baseband_input_count, q} {
  }

 protected:
  void run_once_impl() {
    srtb::work::unpack_work unpack_work;
    SRTB_POP_WORK(" [unpack pipe] ", srtb::unpack_queue, unpack_work);
    // data length after unpack
    const auto baseband_input_bits = unpack_work.baseband_input_bits;
    const size_t out_count =
        unpack_work.count * srtb::BITS_PER_BYTE / baseband_input_bits;

    // re-construct fft_window_functor_manager if length mismatch
    if (out_count != window_functor_manager.functor.n) [[unlikely]] {
      SRTB_LOGW << " [unpack pipe] "
                << "re-construct fft_window_functor_manager of size "
                << out_count << srtb::endl;
      window_functor_manager =
          srtb::fft::fft_window_functor_manager<srtb::real,
                                                srtb::fft::default_window>{
              srtb::fft::default_window{}, out_count, q};
    }

    auto& d_in_shared = unpack_work.ptr;
    // size += 2 because fft_pipe may operate in-place
    auto d_out_shared =
        srtb::device_allocator.allocate_shared<srtb::real>(out_count + 2);
    auto d_in = d_in_shared.get();
    auto d_out = d_out_shared.get();
    // runtime dispatch of different input bits
    if (baseband_input_bits <= srtb::BITS_PER_BYTE) {
      // 1, 2, 4, 8 -> std::byte
      srtb::unpack::unpack(baseband_input_bits, d_in, d_out, out_count,
                           window_functor_manager.functor, q);
    } else if (baseband_input_bits == sizeof(float) * srtb::BITS_PER_BYTE) {
      // 32 -> float/f32
      srtb::unpack::unpack(baseband_input_bits, reinterpret_cast<float*>(d_in),
                           d_out, out_count, window_functor_manager.functor, q);
    } else if (baseband_input_bits == sizeof(double) * srtb::BITS_PER_BYTE) {
      // 64 -> double/f64
      srtb::unpack::unpack(baseband_input_bits, reinterpret_cast<double*>(d_in),
                           d_out, out_count, window_functor_manager.functor, q);
    } else {
      throw std::runtime_error(
          "[unpack pipe] unsupported baseband_input_bits = " +
          std::to_string(baseband_input_bits));
    }
    d_in = nullptr;
    d_in_shared.reset();

    srtb::work::fft_1d_r2c_work fft_1d_r2c_work;
    fft_1d_r2c_work.ptr = d_out_shared;
    fft_1d_r2c_work.count = out_count;
    fft_1d_r2c_work.timestamp = unpack_work.timestamp;
    SRTB_PUSH_WORK(" [unpack pipe] ", srtb::fft_1d_r2c_queue, fft_1d_r2c_work);
  }
};

}  // namespace pipeline
}  // namespace srtb

#endif  // __SRTB_PIPELINE_UNPACK_PIPE__
