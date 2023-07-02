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

#include "sycl/execution_policy"
// --- divide line for clang-format
#include "srtb/commons.hpp"
#include "srtb/pipeline/pipe.hpp"
// --- divide line for clang-format
#include "srtb/algorithm/map_reduce.hpp"
#include "srtb/algorithm/multi_reduce.hpp"
#include "srtb/signal_detect.hpp"
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
  sycl::sycl_execution_policy<> execution_policy{q};

 protected:
  void run_once_impl(std::stop_token stop_token) {
    srtb::work::signal_detect_work signal_detect_work;
    SRTB_POP_WORK_OR_RETURN(" [signal_detect_pipe] ", srtb::signal_detect_queue,
                            signal_detect_work, stop_token);

    auto& d_in_shared = signal_detect_work.ptr;
    auto d_in = d_in_shared.get();
    const size_t count_per_batch = signal_detect_work.count;
    const size_t batch_size = signal_detect_work.batch_size;

    srtb::spectrum::mitigate_rfi_spectral_kurtosis_method(
        d_in, count_per_batch, batch_size,
        srtb::config.mitigate_rfi_spectral_kurtosis_threshold, q);

    SRTB_LOGD << " [signal_detect_pipe] "
              << "mitigate_rfi_spectral_kurtosis_method finished"
              << srtb::endl;

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

    if (SRTB_ENABLE_GUI && srtb::config.gui_enable) {
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
    const size_t time_series_count = batch_size;
    auto d_time_series_shared =
        srtb::device_allocator.allocate_shared<srtb::real>(time_series_count);
    auto d_time_series = d_time_series_shared.get();

    constexpr auto map = []([[maybe_unused]] size_t pos,
                            srtb::complex<srtb::real> c) {
      return srtb::norm(c);
    };

    srtb::algorithm::multi_mapreduce(
        d_in, count_per_batch, batch_size, d_time_series, map,
        /* reduce = */ sycl::plus<srtb::real>(), q);

    const srtb::real snr_threshold =
        srtb::config.signal_detect_signal_noise_threshold;

    // remove baseline -- substract average
    // this is done first to avoid big float - big float when calculating variance
    // TODO: does baseline changes a lot in this time scale ? Is a linear approximation needed ?
    {
      auto d_average_shared = srtb::algorithm::map_average(
          d_time_series, time_series_count, srtb::algorithm::map_identity(), q);
      auto d_average = d_average_shared.get();
      q.parallel_for(sycl::range<1>{time_series_count},
                     [=](sycl::item<1> item) {
                       const auto i = item.get_id(0);
                       d_time_series[i] -= (*d_average);
                     })
          .wait();
    }
    // time series is now available

    srtb::work::baseband_output_work baseband_output_work;
    baseband_output_work.ptr = nullptr;  // TODO
    baseband_output_work.count = 0;
    baseband_output_work.batch_size = 0;
    baseband_output_work.baseband_data =
        std::move(signal_detect_work.baseband_data);
    baseband_output_work.timestamp = signal_detect_work.timestamp;
    baseband_output_work.udp_packet_counter =
        signal_detect_work.udp_packet_counter;

    // if too many frequency channels are masked, result is often inaccurate
    if (zero_count <
        srtb::config.signal_detect_channel_threshold * count_per_batch) {
      // trivial signal detect on raw time series
      {
        const size_t h_signal_count = srtb::signal_detect::count_signal(
            d_time_series, time_series_count, snr_threshold, q);
        const bool has_signal = (h_signal_count > 0);
        if (has_signal) /* [[unlikely]] */ {
          // copy from device to host
          auto h_out_shared = srtb::host_allocator.allocate_shared<srtb::real>(
              time_series_count);
          auto h_out = h_out_shared.get();
          sycl::event event =
              q.copy(d_time_series, /* -> */ h_out, time_series_count);
          srtb::work::time_series_holder time_series_holder{
              .h_time_series = h_out_shared,
              .time_series_length = time_series_count,
              .boxcar_length = 1,
              .transfer_event = event};
          baseband_output_work.time_series.push_back(time_series_holder);
        }
      }

      // boxcar method
      // ref: heimdall
      {
        auto d_accumulated_unique_ptr =
            srtb::device_allocator.allocate_unique<srtb::real>(
                time_series_count);
        auto d_accumulated = d_accumulated_unique_ptr.get();
        sycl::impl::inclusive_scan(execution_policy, d_time_series,
                                   d_time_series + time_series_count,
                                   d_accumulated, srtb::real{0}, std::plus());
        // TODO: reuse d_time_series ?
        // this is reused for different boxcar_length
        auto d_boxcared_time_series_unique_ptr =
            srtb::device_allocator.allocate_unique<srtb::real>(
                time_series_count);
        auto d_boxcared_time_series = d_boxcared_time_series_unique_ptr.get();

        const size_t max_boxcar_length =
            srtb::config.signal_detect_max_boxcar_length;
        // TODO: async submit kernel ?
        for (size_t boxcar_length = 2; (boxcar_length <= max_boxcar_length &&
                                        boxcar_length < time_series_count);
             boxcar_length *= 2) {
          // compute boxcared time series
          const size_t boxcared_time_series_count =
              time_series_count - boxcar_length;
          q.parallel_for(sycl::range{boxcared_time_series_count},
                         [=](sycl::item<1> id) {
                           const size_t i = id.get_id(0);
                           d_boxcared_time_series[i] =
                               d_accumulated[i + boxcar_length] -
                               d_accumulated[i];
                         })
              .wait();

          // trivially detect signal, again
          const size_t h_signal_count = srtb::signal_detect::count_signal(
              d_boxcared_time_series, boxcared_time_series_count, snr_threshold,
              q);
          const bool has_signal = (h_signal_count > 0);
          if (has_signal) /* [[unlikely]] */ {
            // copy from device to host
            auto h_out_shared =
                srtb::host_allocator.allocate_shared<srtb::real>(
                    time_series_count);
            auto h_out = h_out_shared.get();
            sycl::event event = q.copy(d_boxcared_time_series, /* -> */ h_out,
                                       boxcared_time_series_count);
            srtb::work::time_series_holder time_series_holder{
                .h_time_series = h_out_shared,
                .time_series_length = boxcared_time_series_count,
                .boxcar_length = boxcar_length,
                .transfer_event = event};
            baseband_output_work.time_series.push_back(time_series_holder);
          }
        }
      }
    } else {
      // baseband_output_work.has_signal = false;
      // currently represented as time_series.size() == 0
      // that is, do nothing
    }

    const bool has_signal = (baseband_output_work.time_series.size() > 0);
    if (has_signal) {
      baseband_output_work.ptr = std::move(d_in_shared);
      baseband_output_work.count = signal_detect_work.count;
      baseband_output_work.batch_size = signal_detect_work.batch_size;
      SRTB_LOGI << " [signal_detect_pipe] "
                << " signal detected in "
                << baseband_output_work.time_series.size() << " time series"
                << srtb::endl;
    } else {
      SRTB_LOGD << " [signal_detect_pipe] "
                << "no signal detected" << srtb::endl;
    }
    SRTB_PUSH_WORK_OR_RETURN(" [signal_detect_pipe] ",
                             srtb::baseband_output_queue, baseband_output_work,
                             stop_token);
  }
};

}  // namespace pipeline
}  // namespace srtb

#endif  // __SRTB_PIPELINE_SIGNAL_DETECT_PIPE__
