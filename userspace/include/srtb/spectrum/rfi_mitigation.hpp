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
#include "srtb/algorithm/map_identity.hpp"
#include "srtb/algorithm/map_reduce.hpp"
#include "srtb/algorithm/multi_reduce.hpp"

namespace srtb {
namespace spectrum {

// ---------------- rfi mitigation using average value ----------------

/**
 * @brief Mitigate radio frequency interference (RFI),
 *        in this method RFI is determined by intensity of single frequency 
 *        compared to average intensity, so better for long time scale RFI
 * @note not used because normalization fused in, see @c mitigate_rfi_pipe
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

// ---------------- rfi mitigation using manual selected ranges ----------------

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

// ---------------- rfi mitigation using spectrum kutorsis ----------------

/**
 * @brief Mitigate radio frequency interference (RFI), using spectral kurtosis
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
 * 
 * @note current implementation is not efficient if fft_bins is small
 * 
 * update: fused normalizations (disabled now, need further investigation)
 *         remove different amplification on different channels
 */
template <typename T = srtb::real, typename C = srtb::complex<srtb::real>,
          typename DeviceComplexInputAccessor = C*, bool normalization = false>
inline void mitigate_rfi_spectral_kurtosis_method(
    DeviceComplexInputAccessor d_in, size_t fft_bins, size_t time_counts,
    T sk_threshold, sycl::queue& q = srtb::queue) {
  // sometimes float is not enough, change to double if so
  using sum_real_t = T;
  // notice the difference of definition of spectral kurtosis (constant can be -1 or -2)
  // here -1 is picked, and constants are moved into threshold to reduce computation on device
  const size_t M = time_counts;
  const size_t total_size = fft_bins * time_counts;
  const srtb::real M_ = M;
  srtb::real threshold_high = sk_threshold;
  srtb::real threshold_low = 2 - sk_threshold;
  if (threshold_low > threshold_high) {
    std::swap(threshold_low, threshold_high);
  }
  const srtb::real threshold_low_ = threshold_low * ((M_ - 1) / (M_ + 1)) + 1;
  const srtb::real threshold_high_ = threshold_high * ((M_ - 1) / (M_ + 1)) + 1;

  auto d_sk_unique =
      srtb::device_allocator.allocate_unique<srtb::real>(fft_bins);
  auto d_sk_ = d_sk_unique.get();
  auto d_average_unique =
      srtb::device_allocator.allocate_unique<srtb::real>(fft_bins);
  auto d_average = d_average_unique.get();

  // TODO: further parallelize this, as fft_bins may only 1024 or even 256,
  //       but shader cores are 3840 / 8192 / 10752...
  q.parallel_for(sycl::range<1>{fft_bins}, [=](sycl::item<1> id) {
     const size_t j = id.get_id(0);
     // s_n := \sum x^n
     sum_real_t s1 = 0, s2 = 0, s4 = 0;
     for (size_t i = 0; i < M; i++) {
       const size_t index = i * fft_bins + j;
       SRTB_ASSERT_IN_KERNEL(index < fft_bins * time_counts);
       const C in = d_in[index];
       const sum_real_t x2 = srtb::norm(in);
       const sum_real_t x4 = x2 * x2;
       if constexpr (normalization) {
         const sum_real_t x1 = sycl::sqrt(x2);
         s1 += x1;
       }
       s2 += x2;
       s4 += x4;
     }
     // for sk_, notice the comment of threshold above
     const srtb::real sk_ = M * (s4 / (s2 * s2));
     const T average = s1 / M;
     d_sk_[j] = sk_;
     d_average[j] = average;
   }).wait();

  // d_average_threshold is to detect whether this channel is formerly
  // zapped manually
  // TODO: check this; better ways?
  auto d_average_intensity_shared = srtb::algorithm::map_average(
      d_average, fft_bins, srtb::algorithm::map_identity(), q);
  auto d_average_intensity = d_average_intensity_shared.get();
  q.single_task([=]() {
     // TODO: check this constant
     (*d_average_intensity) *= T{0.1};
   }).wait();
  auto d_average_threshold = d_average_intensity;

  q.parallel_for(sycl::range<1>(total_size), [=](sycl::item<1> id) {
     const size_t i = id.get_id(0);
     const size_t j = i - ((i / fft_bins) * fft_bins);
     const auto sk_ = d_sk_[j];
     const auto average = d_average[j];
     const bool zeroing = (sk_ > threshold_high_ || sk_ < threshold_low_);
     constexpr auto zero = C{T{0}, T{0}};

     const C in = d_in[i];
     if constexpr (normalization) {
       C out;
       if (!zeroing) {
         // other methods may make average == 0.0, e.g. manual zap channels
         if (average > (*d_average_threshold)) [[likely]] {
           out = in / average;
         } else {
           out = in;
         }
         SRTB_ASSERT_IN_KERNEL(out == out);
       } else {
         out = zero;
       }
       d_in[i] = out;
     } else {
       if (zeroing) {
         d_in[i] = zero;
       }
     }
   }).wait();
}

/**
 * @brief Mitigate radio frequency interference (RFI), using spectral kurtosis.
 * 
 * @note notice different interpretations of axis:
 *          time
 *       ---------->
 *     1111......1111 |                x
 *     1111......1111 | frequency  ->  x
 *          ....      |                .
 *     1111......1111 v                x
 * 
 * ref: Vrabie, Granjon, and Serviere (2003), Nita (2016), Taylor et al. (2018), and Jiang (2022)
 */
template <typename T = srtb::real, typename C = srtb::complex<srtb::real>,
          typename DeviceComplexInputAccessor = C*>
inline void mitigate_rfi_spectral_kurtosis_method_2(
    DeviceComplexInputAccessor d_in, size_t time_counts, size_t fft_bins,
    T sk_threshold, sycl::queue& q) {
  // notice the difference of definition of spectral kurtosis (constant can be -1 or -2)
  // here -1 is picked, and constants are moved into threshold to reduce computation on device
  const size_t M = time_counts;
  const size_t total_size = fft_bins * time_counts;
  const T M_ = M;
  T threshold_high = sk_threshold;
  T threshold_low = 2 - sk_threshold;
  if (threshold_low > threshold_high) {
    std::swap(threshold_low, threshold_high);
  }
  const T threshold_low_ = threshold_low * ((M_ - 1) / (M_ + 1)) + 1;
  const T threshold_high_ = threshold_high * ((M_ - 1) / (M_ + 1)) + 1;

  // <x^2, x^4>
  using power_pair = std::pair<T, T>;
  auto d_power_pair_unique =
      srtb::device_allocator.allocate_unique<power_pair>(fft_bins);
  auto d_power_pair = d_power_pair_unique.get();

  // sum all x^2, x^4
  srtb::algorithm::multi_mapreduce(
      d_in, time_counts, fft_bins, d_power_pair,
      /* map = */
      []([[maybe_unused]] auto pos, C x) {
        T x2 = srtb::norm(x), x4 = x2 * x2;
        return power_pair{x2, x4};
      },
      /* reduce = */
      [](power_pair a, power_pair b) {
        return power_pair{a.first + b.first, a.second + b.second};
      },
      q);

  q.parallel_for(sycl::range<1>(total_size), [=](sycl::item<1> id) {
     const size_t i = id.get_id(0);
     const size_t j = (i / time_counts);  // index on frequency axis
     //const size_t k = i - (j * time_counts);  // index on time axis
     const auto [s2, s4] = d_power_pair[j];
     const T sk_ = M * (s4 / (s2 * s2));
     const bool zeroing = (sk_ > threshold_high_ || sk_ < threshold_low_);
     constexpr auto zero = C{T{0}, T{0}};
     if (zeroing) {
       d_in[i] = zero;
     }
   }).wait();
}

}  // namespace spectrum
}  // namespace srtb

#endif  // __SRTB_SPECTRIM_RFI_MITIGAGION__
