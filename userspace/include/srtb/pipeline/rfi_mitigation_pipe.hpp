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

#include <string>
#include <utility>
#include <vector>

#include "srtb/pipeline/framework/pipe.hpp"
#include "srtb/spectrum/rfi_mitigation.hpp"

namespace srtb {
namespace pipeline {

/**
 * @brief this pipe mitigates radio frequency interferences (RFI)
 *        using average intensity of frequency channels and manually set ranges.
 * @note another RFI mitigation using spectral kurtosis is in signal_detect_pipe
 */
class rfi_mitigation_s1_pipe {
 protected:
  sycl::queue q;
  /** @brief frequency ranges that is manuallly set in config by user */
  std::string mitigate_rfi_freq_list;
  /** @brief frequency tanges parsed from the string above */
  std::vector<srtb::spectrum::rfi_range_type> rfi_ranges;

 public:
  rfi_mitigation_s1_pipe(sycl::queue q_) : q{q_} {}

  auto operator()([[maybe_unused]] std::stop_token stop_token,
                  srtb::work::rfi_mitigation_s1_work rfi_mitigation_s1_work) {
    auto d_in_shared = rfi_mitigation_s1_work.ptr;
    auto d_in = d_in_shared.get();
    const size_t in_count = rfi_mitigation_s1_work.count;

    // mitigate using average method & normalize
    {
      const srtb::real threshold =
          srtb::config.mitigate_rfi_average_method_threshold;
      auto d_norm_avg_shared = srtb::algorithm::map_average(
          d_in, in_count,
          /* map = */
          []([[maybe_unused]] size_t pos, const srtb::complex<srtb::real> c) {
            return srtb::norm(c);
          },
          q);
      auto d_norm_avg = d_norm_avg_shared.get();
      const srtb::real normalization_coefficient = std::pow(
          static_cast<srtb::real>(in_count) *
              static_cast<srtb::real>(in_count) /
              static_cast<srtb::real>(srtb::config.spectrum_channel_count),
          -0.5);
      q.parallel_for(sycl::range<1>{in_count}, [=](sycl::item<1> id) {
         const size_t i = id.get_id(0);
         const srtb::real norm_avg = (*d_norm_avg);
         const auto in = d_in[i];
         const srtb::real val = srtb::norm(in);
         constexpr auto zero = srtb::complex<srtb::real>{0, 0};
         if (val > threshold * norm_avg) {
           // zap this channel
           d_in[i] = zero;
         } else {
           // normalize
           d_in[i] = in * normalization_coefficient;
         }
       }).wait();
    }

    // manual zap channel
    {
      if (srtb::config.mitigate_rfi_freq_list != mitigate_rfi_freq_list)
          [[unlikely]] {
        mitigate_rfi_freq_list = srtb::config.mitigate_rfi_freq_list;
        rfi_ranges = srtb::spectrum::eval_rfi_ranges(mitigate_rfi_freq_list);
      }
      const auto baseband_freq_low = srtb::config.baseband_freq_low;
      const auto baseband_bandwidth = srtb::config.baseband_bandwidth;
      // TODO: in_count + 1 ? as highest freq point is not included in in_count
      srtb::spectrum::mitigate_rfi_manual(d_in, in_count, baseband_freq_low,
                                          baseband_bandwidth, rfi_ranges, q);
    }

    srtb::work::dedisperse_work dedisperse_work;
    dedisperse_work.move_parameter_from(std::move(rfi_mitigation_s1_work));
    dedisperse_work.ptr = d_in_shared;
    dedisperse_work.count = in_count;
    return std::optional{dedisperse_work};
  }
};

/**
 * @brief this pipe mitigates radio frequency interferences (RFI)
 *        using average intensity of frequency channels and manually set ranges.
 * @note another RFI mitigation using spectral kurtosis is in signal_detect_pipe
 */
class rfi_mitigation_s2_pipe {
 public:
  sycl::queue q;

  auto operator()([[maybe_unused]] std::stop_token stop_token,
                  srtb::work::rfi_mitigation_s2_work rfi_mitigation_s2_work) {
    auto d_in_shared = rfi_mitigation_s2_work.ptr;
    auto d_in = d_in_shared.get();
    const size_t time_sample_count = rfi_mitigation_s2_work.count;
    const size_t frequency_bin_count = rfi_mitigation_s2_work.batch_size;

    srtb::spectrum::mitigate_rfi_spectral_kurtosis_method_2(
        d_in, time_sample_count, frequency_bin_count,
        srtb::config.mitigate_rfi_spectral_kurtosis_threshold, q);

    srtb::work::signal_detect_work signal_detect_work;
    signal_detect_work.move_parameter_from(std::move(rfi_mitigation_s2_work));
    signal_detect_work.ptr = d_in_shared;
    signal_detect_work.count = time_sample_count;
    signal_detect_work.batch_size = frequency_bin_count;
    return std::optional{signal_detect_work};
  }
};

}  // namespace pipeline
}  // namespace srtb

#endif  // __SRTB_PIPELINE_RFI_MITIGATION_PIPE__
