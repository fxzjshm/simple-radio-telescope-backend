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

#include "srtb/fft/fft_window.hpp"
#include "srtb/unpack.hpp"
#include "test-common.hpp"

#define SRTB_CHECK_TEST_FFT_WINDOW(expr)                                      \
  SRTB_CHECK(expr, true, {                                                    \
    throw std::runtime_error{"[test-fft_window] " #expr " at " __FILE__ ":" + \
                             std::to_string(__LINE__) + " returns " +         \
                             std::to_string(ret)};                            \
  })

void tiny_test() {
  double threshold = 1e-6;

  sycl::queue q;
  srtb::fft::window::hamming hamming_window;

  constexpr size_t n = 16;
  srtb::fft::fft_window_functor_manager hamming_window_functor_manager{
      hamming_window, n, q};
  srtb::fft::fft_window_functor hamming_window_functor =
      hamming_window_functor_manager.functor;

  // generated from numpy.hamming(16)
  // with a[i] = a0 - a1 * cos(2 * PI * i / (n - 1)), a0 = 0.54, a1 = 0.46
  std::array<srtb::real, n> expected = {
      0.08,       0.11976909, 0.23219992, 0.39785218, 0.58808309, 0.77,
      0.91214782, 0.9899479,  0.9899479,  0.91214782, 0.77,       0.58808309,
      0.39785218, 0.23219992, 0.11976909, 0.08};
  std::array<srtb::real, n> expected2;  // a0 = 25/46, a1 = 21/46
  for (size_t i = 0; i < n; i++) {
    const auto cos_value =
        (0.54 - expected[i]) / 0.46;  // == cos(2 * PI * i / (n - 1))
    expected2[i] = (25.0 / 46.0) - cos_value * (21.0 / 46.0);
  }

  {
    auto h_in_shared = srtb::host_allocator.allocate_shared<srtb::real>(n);
    auto d_in_shared = srtb::device_allocator.allocate_shared<srtb::real>(n);
    auto h_out_shared = srtb::host_allocator.allocate_shared<srtb::real>(n);
    auto d_out_shared = srtb::device_allocator.allocate_shared<srtb::real>(n);
    auto h_in = h_in_shared.get();
    auto d_in = d_in_shared.get();
    auto h_out = h_out_shared.get();
    auto d_out = d_out_shared.get();
    std::generate_n(h_in, n, []() { return srtb::real{1.0}; });
    q.copy(h_in, /* -> */ d_in, n).wait();
    q.parallel_for(sycl::range<1>(n), [=](sycl::item<1> id) {
       const auto i = id.get_id(0);
       d_out[i] = hamming_window_functor(i, d_in[i]);
     }).wait();
    q.copy(d_out, /* -> */ h_out, n).wait();
    for (size_t i = 0; i < n; i++) {
      std::cout << h_out[i] << ' ';
    }
    std::cout << std::endl;
    // difference betweed 0.54, 0.46 and 25/46, 21/46
    SRTB_CHECK_TEST_FFT_WINDOW(check_absolute_error(
        expected2.begin(), expected2.end(), h_out, threshold));

    // test using another definition
    {
      srtb::fft::window::cosine_sum_window<2> another_hamming_window{
          srtb::real{0.54}, srtb::real{0.46}};
      srtb::fft::fft_window_functor_manager<srtb::real>
          another_hamming_window_functor_manager{another_hamming_window, n, q};
      srtb::fft::fft_window_functor hamming_window_functor =
          another_hamming_window_functor_manager.functor;
      q.parallel_for(sycl::range<1>(n), [=](sycl::item<1> id) {
         const auto i = id.get_id(0);
         d_out[i] = hamming_window_functor(i, d_in[i]);
       }).wait();
      q.copy(d_out, /* -> */ h_out, n).wait();
      for (size_t i = 0; i < n; i++) {
        std::cout << h_out[i] << ' ';
      }
      std::cout << std::endl;
      SRTB_CHECK_TEST_FFT_WINDOW(
          check_absolute_error(h_out, h_out + n, expected.begin(), threshold));
    }
  }

  {
    constexpr size_t NBITS = 1;
    constexpr size_t in_count = 2,
                     out_count = in_count * srtb::BITS_PER_BYTE / NBITS;
    std::array<std::byte, in_count> h_in{std::byte{0b11111111},
                                         std::byte{0b11111111}};
    std::array<srtb::real, out_count> h_out;
    auto d_in_shared =
        srtb::device_allocator.allocate_shared<std::byte>(in_count);
    auto d_out_shared =
        srtb::device_allocator.allocate_shared<srtb::real>(out_count);
    auto d_in = d_in_shared.get();
    auto d_out = d_out_shared.get();
    q.copy(&h_in[0], /* -> */ d_in, in_count).wait();
    srtb::unpack::unpack<1, false>(d_in, d_out, in_count,
                                   hamming_window_functor, q);
    q.copy(d_out, &h_out[0], out_count).wait();
    SRTB_CHECK_TEST_FFT_WINDOW(out_count == n);
    for (size_t i = 0; i < n; i++) {
      std::cout << h_out[i] << ' ';
    }
    std::cout << std::endl;
    SRTB_CHECK_TEST_FFT_WINDOW(check_absolute_error(
        h_out.begin(), h_out.end(), expected2.begin(), threshold));
  }
}

int main(int argc, char** argv) {
  // TODO Auto-generated method stub
  (void)argc;
  (void)argv;
  tiny_test();
  //large_test();

  return 0;
}
