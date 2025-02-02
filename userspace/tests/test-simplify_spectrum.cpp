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

#include <array>
#include <vector>

#include "srtb/spectrum/simplify_spectrum.hpp"
#include "test-common.hpp"

#define SRTB_CHECK_TEST_SIMPLIFY_SPECTRUM(expr)                        \
  SRTB_CHECK(expr, true, {                                             \
    throw std::runtime_error{                                          \
        "[test-simplify_spectrum] " #expr " at " __FILE__ " line " +   \
        std::to_string(__LINE__) + " returns " + std::to_string(ret)}; \
  })

template <typename HostContainer1, typename HostContainer2>
bool test_simplify_spectrum(HostContainer1 h_in_original,
                            HostContainer2 h_out_expected,
                            sycl::queue& q) {
  std::vector<srtb::complex<srtb::real> > h_in;
  const size_t in_count = h_in_original.size(),
               out_count = h_out_expected.size();
  h_in.resize(in_count);
  for (size_t k = 0; k < in_count; k++) {
    h_in.at(k) = static_cast<srtb::complex<srtb::real> >(h_in_original.at(k));
  }

  auto d_in_shared =
      srtb::device_allocator.allocate_shared<srtb::complex<srtb::real> >(
          in_count);
  auto d_in = d_in_shared.get();
  auto d_out_shared =
      srtb::device_allocator.allocate_shared<srtb::real>(out_count);
  auto d_out = d_out_shared.get();
  q.fill<srtb::real>(d_out, srtb::real{0}, out_count).wait();
  q.copy(&h_in[0], /* -> */ d_in, in_count).wait();
  std::vector<srtb::real> h_out;
  h_out.resize(out_count);
  srtb::spectrum::simplify_spectrum_calculate_norm(d_in, in_count, d_out,
                                                   out_count, 1, q);
  srtb::spectrum::simplify_spectrum_normalize_with_average_value(d_out,
                                                                 out_count, q);
  q.copy(d_out, /* -> */ &h_out[0], out_count).wait();
  return check_absolute_error(h_out_expected.begin(), h_out_expected.end(),
                              h_out.begin(), srtb::spectrum::eps);
}

template <typename T, typename RealNumberContainer>
std::vector<srtb::complex<T> > form_complex_numbers(
    RealNumberContainer in) {
  const size_t in_count = in.size(), out_count = in_count / 2;
  std::vector<srtb::complex<T> > out;
  out.resize(out_count);
  for (size_t k = 0; k < out_count; k++) {
    out.at(k) = srtb::complex<T>{static_cast<T>(in[2 * k]),
                                 static_cast<T>(in[2 * k + 1])};
  }
  return out;
}

int main() {
  sycl::queue q;
  {
    //std::array h_in_intensity = {1.0, 3.0, 4.0, 0.0, 2.0};
    std::array h_in_original = {0.6,
                                0.8,
                                -0.8 * sycl::sqrt(3.0),
                                0.6 * sycl::sqrt(3.0),
                                -1.2,
                                -1.6,
                                0.0,
                                0.0,
                                0.0,
                                -sycl::sqrt(2.0)};
    // downsampled: {2.0, 5.0, 2.0/3.0}
    std::array h_out_expected = {9.0/20.0, 15.0/20.0, 6.0/20.0};
    SRTB_CHECK_TEST_SIMPLIFY_SPECTRUM(test_simplify_spectrum(
        form_complex_numbers<double>(h_in_original), h_out_expected, q));
  }
  {
    //std::array h_in_intensity = {17.0, 25.0};
    std::array h_in_original = {1.0, 4.0, 5.0, 0.0};
    // re-sampled: {34.0/3.0, 14.0, 50.0/3.0}
    std::array h_out_expected = {17.0/42.0, 21.0/42.0, 25.0/42.0};
    SRTB_CHECK_TEST_SIMPLIFY_SPECTRUM(test_simplify_spectrum(
        form_complex_numbers<double>(h_in_original), h_out_expected, q));
  }
  return 0;
}
