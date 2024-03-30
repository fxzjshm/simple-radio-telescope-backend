/******************************************************************************* 
 * Copyright (c) 2023 fxzjshm
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
#ifndef __SRTB_PIPELINE_FFT_R2C_POST_PROCESS_PIPE__
#define __SRTB_PIPELINE_FFT_R2C_POST_PROCESS_PIPE__

#include <cstddef>
#include <stop_token>

#include "21cma_genfil_work.hpp"
#include "srtb/fft/fft_1d_r2c_post_process.hpp"
#include "srtb/math.hpp"
#include "srtb/pipeline/framework/pipe.hpp"
#include "srtb/sycl.hpp"

namespace srtb {
namespace pipeline {

/**
 * @brief this pipe is the post processes procedure of batched 1D R2C FFT.
 * 
 * note: to operate in-place, all highest frequency points are dropped
 * 
 * ref: naive_fft::fft_1d_r2c <= https://www.cnblogs.com/liam-ji/p/11742941.html
 */
class fft_r2c_post_process_pipe {
 public:
  using in_work_type = srtb::work::fft_r2c_post_process_work;
  using out_work_type = srtb::work::work<std::shared_ptr<srtb::real> >;

  sycl::queue q;

  auto operator()([[maybe_unused]] std::stop_token stop_token,
                  in_work_type in_work) -> out_work_type {
    auto& d_in_shared = in_work.ptr;
    auto d_in = d_in_shared.get();

    const size_t N = in_work.count;
    const size_t batch_size = in_work.batch_size;

    srtb::fft::fft_1d_r2c_in_place_post_process(d_in, N, batch_size, q);

    // cut spectrum & reverse
    // snap1 frequency range: 0-200 MHz, but only 50-200 MHz has value
    // filterbank requires negative df
    const size_t out_nchans = N * 3 / 4;
    const size_t out_nsamps = batch_size;
    const size_t out_total_count = out_nchans * out_nsamps;
    auto d_out_shared =
        srtb::device_allocator.allocate_shared<srtb::real>(out_total_count);
    auto d_out = d_out_shared.get();

    q.parallel_for(sycl::range<2>{out_nchans, out_nsamps}, [=](sycl::item<2>
                                                                   id) {
       const size_t i = id.get_id(0);
       const size_t j = id.get_id(1);
       d_out[out_nchans * j + i] = srtb::norm(d_in[N * j + N - 1 - i]);
     }).wait();

    out_work_type out_work;
    out_work.move_parameter_from(std::move(in_work));
    out_work.ptr = d_out_shared;
    out_work.count = out_nchans;
    out_work.batch_size = out_nsamps;
    return out_work;
  }
};

}  // namespace pipeline
}  // namespace srtb

#endif  // __SRTB_PIPELINE_FFT_R2C_POST_PROCESS_PIPE__
