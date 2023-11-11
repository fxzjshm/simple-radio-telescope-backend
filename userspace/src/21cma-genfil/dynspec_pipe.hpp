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
#ifndef __SRTB_PIPELINE_COMPRSS_DYNSPEC_PIPE__
#define __SRTB_PIPELINE_COMPRSS_DYNSPEC_PIPE__

#include <boost/iterator/transform_iterator.hpp>
#include <cstddef>
#include <stop_token>

#include "srtb/algorithm/running_mean.hpp"
#include "srtb/fft/fft.hpp"
#include "srtb/math.hpp"
#include "srtb/pipeline/framework/pipe.hpp"
#include "srtb/sycl.hpp"

#include "21cma_genfil_work.hpp"

namespace srtb {
namespace pipeline {

/**
 * @brief this pipe compress dynamic spectrum to 1 bit.
 * @note spectrum has been reversed & cut in fft_r2c_post_process_pipe
 */
class dynspec_pipe {
 public:
  using in_work_type = srtb::work::dynspec_work;
  using out_work_type = srtb::work::work<std::shared_ptr<std::byte> >;
  constexpr static size_t windowsize = 4096 * 8;

 public:
  sycl::queue q;
  std::shared_ptr<srtb::real> d_ave_shared;
  size_t d_ave_size;

  explicit dynspec_pipe(sycl::queue q_)
      : q{q_}, d_ave_shared{nullptr}, d_ave_size{0} {}

  auto operator()([[maybe_unused]] std::stop_token stop_token,
                  in_work_type in_work) -> out_work_type {
    auto& d_in_shared = in_work.ptr;
    auto d_in = d_in_shared.get();

    // auto d_spec = boost::transform_iterator{
    //     d_in, [](srtb::complex<srtb::real> x) { return srtb::norm(x); }};
    auto d_spec = d_in;

    const size_t nchans = in_work.count;
    const size_t nsamps = in_work.batch_size;
    const size_t input_total_size = nchans * nsamps;

    auto d_filtered_shared =
        srtb::device_allocator.allocate_shared<uint8_t>(input_total_size);
    auto d_filtered = d_filtered_shared.get();

    if (nchans > d_ave_size) [[unlikely]] {
      d_ave_shared = srtb::device_allocator.allocate_shared<srtb::real>(nchans);
      d_ave_size = nchans;
      auto d_ave = d_ave_shared.get();
      srtb::algorithm::running_mean_init_average(
          d_spec, nsamps, nchans, d_filtered, windowsize, d_ave, q);
    }

    auto d_ave = d_ave_shared.get();
    srtb::algorithm::running_mean(d_spec, nsamps, nchans, d_filtered,
                                  windowsize, d_ave, q);

    // compress to 1 bit
    const size_t output_bytes_per_samp = nchans / srtb::BITS_PER_BYTE;
    const size_t output_total_bytes = output_bytes_per_samp * nsamps;
    auto d_compressed_shared =
        srtb::device_allocator.allocate_shared<std::byte>(output_total_bytes);
    auto d_compressed = d_compressed_shared.get();

    q.parallel_for(
         sycl::range<2>{output_bytes_per_samp, nsamps},
         [=](sycl::item<2> id) {
           const auto i = id.get_id(0);
           const auto j = id.get_id(1);

           std::byte x = std::byte{0};
#pragma unroll
           for (uint8_t k = 0; k < srtb::BITS_PER_BYTE; k++) {
             const std::byte in = std::byte{
                 d_filtered[j * nchans + srtb::BITS_PER_BYTE * i + k]};
#if __has_builtin(__builtin_assume)
             __builtin_assume(in == std::byte{0} || in == std::byte{1});
#endif
             // TODO: check bit order
             x |= std::byte{(in & std::byte{1})
                            << (srtb::BITS_PER_BYTE - 1 - k)};
           }

           d_compressed[j * output_bytes_per_samp + i] = x;
         })
        .wait();

    out_work_type out_work;
    out_work.move_parameter_from(std::move(in_work));
    out_work.ptr = std::move(d_compressed_shared);
    out_work.count = output_bytes_per_samp;
    out_work.batch_size = nsamps;
    return out_work;
  }
};

}  // namespace pipeline
}  // namespace srtb

#endif  // __SRTB_PIPELINE_COMPRSS_DYNSPEC_PIPE__
