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

#include "srtb/commons.hpp"
// -- divide line for clang-format --
#include "srtb/spectrum/rfi_mitigation.hpp"
#include "test-common.hpp"

#define SRTB_CHECK_TEST_RFI_MITIGATION(expr) \
  SRTB_CHECK_TEST("[test-rfi_mitigation] ", expr)

int main() {
  constexpr srtb::complex<srtb::real> one = srtb::complex<srtb::real>{1, 0};
  constexpr srtb::complex<srtb::real> zero = srtb::complex<srtb::real>{0, 0};
  const size_t n = 1500;

  sycl::queue q;

  std::string mitigate_rfi_freq_list = "11-12, 15-90, 233-235, 1176-1177";
  std::vector<srtb::spectrum::rfi_range_type> expected_ranges = {
      {11, 12}, {15, 90}, {233, 235}, {1176, 1177}};
  auto actural_ranges = srtb::spectrum::eval_rfi_ranges(mitigate_rfi_freq_list);
  SRTB_CHECK_TEST_RFI_MITIGATION(expected_ranges.size() ==
                                 actural_ranges.size());
  std::vector<srtb::real> expected_ranges_flatten, actural_ranges_flatten;
  for (auto pair : expected_ranges) {
    expected_ranges_flatten.push_back(pair.first);
    expected_ranges_flatten.push_back(pair.second);
  }
  for (auto pair : actural_ranges) {
    actural_ranges_flatten.push_back(pair.first);
    actural_ranges_flatten.push_back(pair.second);
  }
  SRTB_CHECK_TEST_RFI_MITIGATION(check_relative_error(
      expected_ranges_flatten.begin(), expected_ranges_flatten.end(),
      actural_ranges_flatten.begin(),
      std::numeric_limits<srtb::real>::epsilon()));

  auto d_in_shared =
      srtb::device_allocator.allocate_shared<srtb::complex<srtb::real> >(n);
  auto d_in = d_in_shared.get();

  q.fill(d_in, one, n).wait();
  srtb::spectrum::mitigate_rfi_manual(d_in, n, /* baseband_freq_low = */ 0,
                                      /* baseband_bandwidth =  */ n - 1,
                                      actural_ranges, q);
  auto h_out_shared =
      srtb::host_allocator.allocate_shared<srtb::complex<srtb::real> >(n);
  auto h_out = h_out_shared.get();
  q.copy(d_in, /* -> */ h_out, n).wait();
  for (size_t i = 0; i < n; i++) {
    auto expected = one;
    for (auto range : expected_ranges) {
      if (range.first <= i && i <= range.second) {
        expected = zero;
        break;
      }
    }
    auto actural = h_out[i];
    SRTB_CHECK_TEST_RFI_MITIGATION(expected == actural);
  }
  return 0;
}
