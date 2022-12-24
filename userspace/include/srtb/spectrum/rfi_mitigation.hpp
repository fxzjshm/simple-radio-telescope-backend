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
#ifndef __SRTB_SPECTRIM_RFI_MITIGAGION__
#define __SRTB_SPECTRIM_RFI_MITIGAGION__

#include <boost/algorithm/string.hpp>
#include <utility>
#include <vector>

#include "srtb/commons.hpp"
// --- divide line for clang-format ---
#include "srtb/algorithm/map_reduce.hpp"

namespace srtb {
namespace spectrum {

/**
 * @brief Mitigate radio frequency interference (RFI),
 *        in this method RFI is determined by intensity of single frequency 
 *        compared to average intensity, so better for long time scale RFI
 * TODO: use "spectural kurtosis"
 * TODO: compute norm twice or once with temporary buffer for it ?
 */
template <typename T = srtb::real, typename C = srtb::complex<srtb::real>,
          typename DeviceComplexInputAccessor = C*>
inline void mitigate_rfi_average_method(DeviceComplexInputAccessor d_in,
                                        size_t in_count, srtb::real threshold,
                                        sycl::queue& q = srtb::queue) {
  auto d_norm_avg_shared = srtb::algorithm::map_average(
      d_in, in_count,
      []([[maybe_unused]] size_t pos, C c) { return srtb::norm(c); }, q);
  auto d_norm_avg = d_norm_avg_shared.get();
  q.parallel_for(sycl::range<1>{in_count}, [=](sycl::item<1> id) {
     const size_t i = id.get_id(0);
     const srtb::real norm_avg = (*d_norm_avg);
     const srtb::real val = srtb::norm(d_in[i]);
     if (val > threshold * norm_avg) {
       d_in[i] = C(T(0), T(0));
     }
   }).wait();
}

using rfi_range_type = std::pair<srtb::real, srtb::real>;

template <typename T = srtb::real>
inline auto eval_rfi_ranges(const std::string& mitigate_rfi_freq_list) {
  std::vector<rfi_range_type> rfi_ranges;

  std::vector<std::string> range_strings, number_strings;
  boost::split(range_strings, mitigate_rfi_freq_list, boost::is_any_of(","),
               boost::token_compress_on);
  rfi_ranges.reserve(range_strings.size());
  number_strings.reserve(2);
  for (auto str : range_strings) {
    number_strings.clear();
    boost::split(number_strings, str, boost::is_any_of("-"),
                 boost::token_compress_on);
    if (number_strings.size() != 2) [[unlikely]] {
      SRTB_LOGW << " [eval_rfi_ranges] "
                << "cannot parse \"" << str << "\"" << srtb::endl;
    } else {
      const srtb::real freq_1 =
          static_cast<srtb::real>(std::stod(number_strings[0]));
      const srtb::real freq_2 =
          static_cast<srtb::real>(std::stod(number_strings[1]));
      rfi_ranges.push_back(std::make_pair(freq_1, freq_2));
    }
  }
  return rfi_ranges;
}

/**
 * @brief Mitigate radio frequency interference (RFI),
 *        in this method RFI is determined manually
 * @param rfi_ranges frequency range of RFI, set manually
 * TODO: use "spectural kurtosis"
 */
template <typename T = srtb::real, typename C = srtb::complex<srtb::real>,
          typename DeviceComplexInputAccessor = C*>
inline void mitigate_rfi_manual(DeviceComplexInputAccessor d_in,
                                size_t in_count, srtb::real baseband_freq_low,
                                srtb::real baseband_bandwidth,
                                const std::vector<rfi_range_type>& rfi_ranges,
                                sycl::queue& q = srtb::queue) {
  const srtb::real baseband_freq_high = baseband_freq_low + baseband_bandwidth;
  std::vector<sycl::event> events{rfi_ranges.size()};
  for (size_t i = 0; i < rfi_ranges.size(); i++) {
    auto [rfi_freq_low, rfi_freq_high] = rfi_ranges.at(i);
    // correct order
    if (rfi_freq_low > rfi_freq_high) [[unlikely]] {
      std::swap(rfi_freq_low, rfi_freq_high);
    }

    // check range
    if (rfi_freq_low < baseband_freq_low) [[unlikely]] {
      SRTB_LOGW << " [mitigate_rfi_manual] "
                << "rfi_freq_low = " << rfi_freq_low
                << " < baseband_freq_low = " << baseband_freq_low << srtb::endl;
      rfi_freq_low = baseband_freq_low;
    }
    if (rfi_freq_high > baseband_freq_high) [[unlikely]] {
      SRTB_LOGW << " [mitigate_rfi_manual] "
                << "rfi_freq_high = " << rfi_freq_high
                << " > baseband_freq_high = " << baseband_freq_high
                << srtb::endl;
      rfi_freq_high = baseband_freq_high;
    }

    /*
    (e.g. in_count = 4)
           frequency bin:     0     1     2     3
                             /|\   / \   / \   /|\
                            / | \ /   \ /   \ / | \
                           /  |  X     X     X  |  \
                          /   | /|\   /|\   /|\ |   \
                         /    |/ | \ / | \ / | \|    \
                              |  |     |     |  | 
                              |--+-----+-----+--| 
      input (normalized):     0                 1
    */
    const size_t rfi_bin_index_low =
        static_cast<size_t>(std::round((rfi_freq_low - baseband_freq_low) /
                                       baseband_bandwidth * (in_count - 1)));
    const size_t rfi_bin_index_high =
        static_cast<size_t>(std::round((rfi_freq_high - baseband_freq_low) /
                                       baseband_bandwidth * (in_count - 1)));
    // inclusive, so +1
    const size_t bin_count = rfi_bin_index_high - rfi_bin_index_low + 1;
    if (0 < bin_count && bin_count <= in_count) {
      events.at(i) =
          q.parallel_for(sycl::range<1>{bin_count}, [=](sycl::item<1> id) {
            const auto index = id.get_id(0);
            d_in[rfi_bin_index_low + index] = C(T(0), T(0));
          });
    }
  }

  // wait until all kernels finished
  for (auto iter = events.rbegin(); iter != events.rend(); iter++) {
    (*iter).wait();
  }
}

}  // namespace spectrum
}  // namespace srtb

#endif  // __SRTB_SPECTRIM_RFI_MITIGAGION__
