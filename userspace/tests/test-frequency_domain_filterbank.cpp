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
#include <iostream>
#include <random>
#include <source_location>
#include <vector>

#include "srtb/frequency_domain_filterbank.hpp"
#include "test-common.hpp"

// TODO: try `std::source_location::current();`
#define SRTB_CHECK_TEST_FREQUENCY_DOMAIN_FILTERBANK(expr)                      \
  SRTB_CHECK(expr, true, {                                                     \
    throw std::runtime_error{                                                  \
        "[test-frequency_domain_filterbank] " #expr " at " __FILE__ " line " + \
        std::to_string(__LINE__) + " returns " + std::to_string(ret)};         \
  })

template <typename T, typename C = srtb::complex<T> >
inline void coherent_dedispersion_and_frequency_domain_filterbank_host_ptr(
    C* h_in, C* h_out, const T f_min, const T delta_f, const T dm,
    const size_t M, const size_t N, sycl::queue& q) {
  auto d_in_shared = srtb::device_allocator.allocate_shared<C>(N);
  auto d_in = d_in_shared.get();
  auto d_out_shared = srtb::device_allocator.allocate_shared<C>(N);
  auto d_out = d_out_shared.get();
  q.copy(h_in, d_in, N).wait();
  srtb::coherent_dedispersion_and_frequency_domain_filterbank(
      d_in, d_out, f_min, delta_f, dm, M, N, q);
  q.copy(d_out, h_out, N).wait();
}

int main(int argc, char** argv) {
  (void)argc;
  (void)argv;
  sycl::queue q;
  srtb::real eps = 1e-6;

  {
    size_t N = 1 << 23;
    std::vector<srtb::complex<srtb::real> > h_in(N), h_out(N);
    std::mt19937 rng{233};
    std::generate(h_in.begin(), h_in.end(), [&]() {
      return srtb::complex<srtb::real>{
          static_cast<srtb::real>(static_cast<int>(rng())),
          static_cast<srtb::real>(static_cast<int>(rng()))};
    });
    coherent_dedispersion_and_frequency_domain_filterbank_host_ptr(
        &h_in[0], &h_out[0], srtb::real{1000.0}, srtb::real{1.0},
        srtb::real{0.0}, /* M := channel_count = */ 1, N, q);
    SRTB_CHECK_TEST_FREQUENCY_DOMAIN_FILTERBANK(
        check_absolute_error(h_in.begin(), h_in.end(), h_out.begin(), eps));
  }
  return 0;
}
