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
#ifndef __SRTB_PIPELINE_SIGNAL_DETECT_PIPE__
#define __SRTB_PIPELINE_SIGNAL_DETECT_PIPE__

#include "srtb/commons.hpp"
#include "srtb/pipeline/pipe.hpp"
// --- divide line for clang-format
#include "srtb/algorithm/map_reduce.hpp"
#include "srtb/algorithm/multi_reduce.hpp"
#include "srtb/spectrum/rfi_mitigation.hpp"

namespace srtb {
namespace pipeline {

/**
 * @brief this pipe reads from refft-ed spectrum, sum it into time series,
 *        and detect if there's signal in it.
 *        If yes, notify someone to write the original baseband data;
 *        otherwise, just drop it.
 * TODO: separate this into 2 pipes ? 
 *       (one for sum into time series, one for actual signal detect)
 */
class signal_detect_pipe : public pipe<signal_detect_pipe> {
  friend pipe<signal_detect_pipe>;

 protected:
  void run_once_impl(std::stop_token stop_token) {
    srtb::work::signal_detect_work signal_detect_work;
    SRTB_POP_WORK_OR_RETURN(" [signal_detect_pipe] ", srtb::signal_detect_queue,
                            signal_detect_work, stop_token);

    auto& d_in_shared = signal_detect_work.ptr;
    auto d_in = d_in_shared.get();
    const size_t count_per_batch = signal_detect_work.count;
    const size_t batch_size = signal_detect_work.batch_size;

    srtb::spectrum::mitigate_rfi_spectural_kurtosis_method(
        d_in, count_per_batch, batch_size,
        srtb::config.mitigate_rfi_spectral_kurtosis_threshold, q);

    size_t zero_count = 0;
    // count masked channels
    {
      auto d_zero_count_shared = srtb::algorithm::map_reduce(
          d_in, count_per_batch, /* map = */
          []([[maybe_unused]] const size_t pos,
             const srtb::complex<srtb::real> x) {
            if (srtb::norm(x) == 0) {
              return size_t{1};
            } else {
              return size_t{0};
            }
          },
          sycl::plus<size_t>(), q);
      auto d_zero_count = d_zero_count_shared.get();
      q.copy(d_zero_count, &zero_count, 1).wait();
    }

    {
      // temporary work: spectrum analyzer
      srtb::work::simplify_spectrum_work simplify_spectrum_work;
      simplify_spectrum_work.ptr = d_in_shared;
      simplify_spectrum_work.count = count_per_batch;
      simplify_spectrum_work.batch_size = batch_size;
      simplify_spectrum_work.timestamp = signal_detect_work.timestamp;
      // just try once, in case GUI is stuck (e.g. when using X forwarding on SSH)
      srtb::simplify_spectrum_queue.push(simplify_spectrum_work);
    }

    // time series
    const size_t out_count = batch_size;
    auto d_out_shared =
        srtb::device_allocator.allocate_shared<srtb::real>(out_count);
    auto d_out = d_out_shared.get();

    constexpr auto map = []([[maybe_unused]] size_t pos,
                            srtb::complex<srtb::real> c) {
      return srtb::norm(c);
    };

    srtb::algorithm::multi_mapreduce(
        d_in, count_per_batch, batch_size, d_out, map,
        /* reduce = */ sycl::plus<srtb::real>(), q);

    d_in = nullptr;
    d_in_shared.reset();

    const srtb::real threshold = srtb::config.signal_detect_threshold;

    // remove baseline -- substract average
    // this is done first to avoid big float - big float when calculating variance
    // TODO: does baseline changes a lot in this time scale ? Is a linear approximation needed ?
    {
      auto d_average_shared = srtb::algorithm::map_average(
          d_out, out_count, srtb::algorithm::map_identity(), q);
      auto d_average = d_average_shared.get();
      q.parallel_for(sycl::range<1>{out_count}, [=](sycl::item<1> item) {
         const auto i = item.get_id(0);
         d_out[i] -= (*d_average);
       }).wait();
    }

    // TODO: matched filter

    // trivial signal detect
    {
      auto d_variance_squared_shared = srtb::algorithm::map_average(
          d_out, out_count,
          []([[maybe_unused]] size_t pos, srtb::real x) -> double {
            const double y = static_cast<double>(x);
            return y * y;
          },
          q);
      auto d_variance_squared = d_variance_squared_shared.get();
      auto d_variance_shared =
          srtb::device_allocator.allocate_shared<srtb::real>(1);
      auto d_variance = d_variance_shared.get();
      q.single_task([=]() {
         (*d_variance) =
             static_cast<srtb::real>(sycl::sqrt(*d_variance_squared));
       }).wait();

      auto d_signal_count_shared = srtb::algorithm::map_sum(
          d_out, out_count, /* map = */
          [=]([[maybe_unused]] size_t pos, srtb::real x) -> size_t {
            // also known as count_if
            if (srtb::abs(x) > threshold * (*d_variance)) {
              return size_t{1};
            } else {
              return size_t{0};
            }
          },
          q);
      size_t* d_signal_count = d_signal_count_shared.get();
      size_t h_signal_count;
      q.copy(d_signal_count, /* -> */ &h_signal_count, /* size = */ 1).wait();

      // if too many frequency channels are masked, result is often inaccurate
      const bool has_signal =
          ((zero_count < 0.9 * count_per_batch) && (h_signal_count > 0));
      srtb::work::signal_detect_result signal_detect_result{
          .timestamp = signal_detect_work.timestamp,
          .has_signal = has_signal,
          .time_series_ptr = std::shared_ptr<srtb::real>{},
          .time_series_length = 0};
      if (has_signal) {
        signal_detect_result.time_series_ptr = d_out_shared;
        signal_detect_result.time_series_length = out_count;
        SRTB_LOGI << " [signal_detect_pipe] " << h_signal_count
                  << " signal(s) detected!" << srtb::endl;
      } else {
        SRTB_LOGD << " [signal_detect_pipe] "
                  << "no signal detected." << srtb::endl;
      }
      SRTB_PUSH_WORK_OR_RETURN(" [signal_detect_pipe] ",
                               srtb::signal_detect_result_queue,
                               signal_detect_result, stop_token);
    }
  }
};

}  // namespace pipeline
}  // namespace srtb

#endif  // __SRTB_PIPELINE_SIGNAL_DETECT_PIPE__
