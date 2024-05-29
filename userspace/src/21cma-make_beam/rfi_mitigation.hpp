/******************************************************************************* 
 * Copyright (c) 2022-2024 fxzjshm
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
#ifndef __SRTB_21CMA_MAKE_BEAM_RFI_MITIGAGION__
#define __SRTB_21CMA_MAKE_BEAM_RFI_MITIGAGION__

#include <boost/algorithm/string.hpp>
#include <cmath>
#include <utility>
#include <vector>

#include "mdspan/mdspan.hpp"
#include "srtb/log/log.hpp"
#include "srtb/math.hpp"

namespace srtb {
namespace _21cma {
namespace spectrum {

// ---------------- rfi mitigation using manual selected ranges ----------------

using rfi_range_type = std::pair<srtb::real, srtb::real>;

template <typename T = srtb::real>
inline auto eval_rfi_ranges(const std::string& mitigate_rfi_freq_list) {
  std::vector<rfi_range_type> rfi_ranges;

  std::vector<std::string> range_strings, number_strings;
  boost::split(range_strings, mitigate_rfi_freq_list, boost::is_any_of(","), boost::token_compress_on);
  rfi_ranges.reserve(range_strings.size());
  number_strings.reserve(2);
  for (auto str : range_strings) {
    number_strings.clear();
    boost::split(number_strings, str, boost::is_any_of("-"), boost::token_compress_on);
    if (number_strings.size() != 2) [[unlikely]] {
      SRTB_LOGW << " [eval_rfi_ranges] " << "cannot parse \"" << str << "\"" << srtb::endl;
    } else {
      const srtb::real freq_1 = static_cast<srtb::real>(std::stod(number_strings[0]));
      const srtb::real freq_2 = static_cast<srtb::real>(std::stod(number_strings[1]));
      rfi_ranges.push_back(std::make_pair(freq_1, freq_2));
    }
  }
  return rfi_ranges;
}

/**
 * @brief Mitigate radio frequency interference (RFI),
 *        in this method RFI is determined manually
 * @param rfi_ranges frequency range of RFI, set manually
 */
template <typename T = srtb::real, typename C = srtb::complex<srtb::real>, typename DeviceComplexInputAccessor = C*>
inline void mitigate_rfi_manual(DeviceComplexInputAccessor d_in, size_t in_count, srtb::real baseband_freq_low,
                                srtb::real baseband_bandwidth, const std::vector<rfi_range_type>& rfi_ranges,
                                sycl::queue& q) {
  const srtb::real baseband_freq_high = baseband_freq_low + baseband_bandwidth;
  const bool bandwidth_sign = std::signbit(baseband_bandwidth);

  std::vector<sycl::event> events{rfi_ranges.size()};
  for (size_t i = 0; i < rfi_ranges.size(); i++) {
    auto [rfi_freq_low, rfi_freq_high] = rfi_ranges.at(i);
    const srtb::real rfi_bandwidth = rfi_freq_high - rfi_freq_low;
    const bool rfi_bandwidth_sign = std::signbit(rfi_bandwidth);
    // correct order
    if (bandwidth_sign != rfi_bandwidth_sign) {
      // order of baseband frequency range and RFI range mismatch
      std::swap(rfi_freq_low, rfi_freq_high);
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
        static_cast<size_t>(std::round((rfi_freq_low - baseband_freq_low) / baseband_bandwidth * (in_count - 1)));
    const size_t rfi_bin_index_high =
        static_cast<size_t>(std::round((rfi_freq_high - baseband_freq_low) / baseband_bandwidth * (in_count - 1)));
    // inclusive, so +1
    const size_t bin_count = rfi_bin_index_high - rfi_bin_index_low + 1;
    // check range
    if (0 <= rfi_bin_index_low && rfi_bin_index_low <= rfi_bin_index_high && rfi_bin_index_high < in_count) [[likely]] {
      events.at(i) = q.parallel_for(sycl::range<1>{bin_count}, [=](sycl::item<1> id) {
        const auto index = id.get_id(0);
        d_in[rfi_bin_index_low + index] = C(T(0), T(0));
      });
    } else {
      SRTB_LOGW << " [mitigate_rfi_manual] " << "RFI frequency range is out of bounds: " << rfi_freq_low << " - "
                << rfi_freq_high << " MHz is out of baseband frequency range " << baseband_freq_low << " - "
                << baseband_freq_high << " MHz" << srtb::endl;
    }
  }

  // wait until all kernels finished
  for (auto iter = events.rbegin(); iter != events.rend(); iter++) {
    (*iter).wait();
  }
}

// ---------------- rfi mitigation using spectrum kutorsis ----------------

/**
 * @brief Mitigate radio frequency interference (RFI), using spectral kurtosis
 * @note modified to operate on intensity; also removed normalization
 * 
 *        frequency
 *       ---------->
 *     1111......1111 |
 *     1111......1111 | time
 *          ....      |
 *     1111......1111 v
 * 
 * ->  xxxx......xxxx
 * 
 * ref: Vrabie, Granjon, and Serviere (2003), Nita (2016), Taylor et al. (2018), and Jiang (2022)
 */
template <typename T = srtb::real>
inline void mitigate_rfi_spectral_kurtosis_method(Kokkos::mdspan<T, Kokkos::dextents<size_t, 2>> d_intensity,
                                                  T sk_threshold, sycl::queue& q) {
  // sometimes float is not enough, change to double if so
  using sum_real_t = T;

  // notice the difference of definition of spectral kurtosis (constant can be -1 or -2)
  // here -1 is picked, and constants are moved into threshold to reduce computation on device

  const size_t n_sample = d_intensity.extent(0);
  const size_t n_channel = d_intensity.extent(1);
  const size_t M = n_sample;
  const srtb::real M_ = M;
  srtb::real threshold_high = sk_threshold;
  srtb::real threshold_low = 2 - sk_threshold;
  if (threshold_low > threshold_high) {
    std::swap(threshold_low, threshold_high);
  }
  const srtb::real threshold_low_ = threshold_low * ((M_ - 1) / (M_ + 1)) + 1;
  const srtb::real threshold_high_ = threshold_high * ((M_ - 1) / (M_ + 1)) + 1;

  q.parallel_for(sycl::range<1>{n_channel}, [=](sycl::item<1> id) {
     const size_t i_channel = id.get_id(0);
     // s_n := \sum x^n
     sum_real_t s2 = 0, s4 = 0;
     for (size_t i_sample = 0; i_sample < M; i_sample++) {
       const sum_real_t x2 = d_intensity[i_sample, i_channel];
       const sum_real_t x4 = x2 * x2;
       s2 += x2;
       s4 += x4;
     }

     // for sk_, notice the comment of threshold above
     const srtb::real sk_ = M * (s4 / (s2 * s2));
     const bool zeroing = (sk_ > threshold_high_ || sk_ < threshold_low_);

     if (zeroing) {
       for (size_t i_sample = 0; i_sample < M; i_sample++) {
         d_intensity[i_sample, i_channel] = 0;
       }
     }
   }).wait();
}

}  // namespace spectrum
}  // namespace _21cma
}  // namespace srtb

#endif  // __SRTB_SPECTRIM_RFI_MITIGAGION__
