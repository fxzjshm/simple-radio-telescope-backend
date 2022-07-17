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

#include "srtb/unpack.hpp"

// TODO: try `std::source_location::current();`
#define SRTB_CHECK_TEST_UNPACK(expr)                                           \
  SRTB_CHECK(expr, true, {                                                     \
    throw std::runtime_error{"[test-unpack] " #expr " at " __FILE__ " line " + \
                             std::to_string(__LINE__) + " returns " +          \
                             std::to_string(ret)};                             \
  })

/**
 * @brief unpack bytes stream into floating-point numbers, for FFT
 * 
 * @tparam IN_NBITS bit width of one input number
 * @param in_count std::bytes count of in. Make sure [0, BITS_PER_BYTE / IN_NBITS * input_count) of out is accessible.
 */
template <int IN_NBITS, bool handwritten = false>
inline std::chrono::nanoseconds unpack_host_ptr(std::byte* h_in,
                                                srtb::real* h_out,
                                                size_t in_count,
                                                sycl::queue& q) {
  const size_t out_count = srtb::BITS_PER_BYTE / IN_NBITS * in_count;
  auto d_in_shared = srtb::device_allocator.allocate_smart<std::byte>(in_count);
  auto d_out_shared =
      srtb::device_allocator.allocate_smart<srtb::real>(out_count);
  auto d_in = d_in_shared.get();
  auto d_out = d_out_shared.get();
  q.copy(h_in, d_in, in_count).wait();
  q.copy(h_out, d_out, out_count).wait();
  auto unpack_start = std::chrono::system_clock::now();
  srtb::unpack::unpack<IN_NBITS, handwritten>(d_in, d_out, in_count, q);
  auto unpack_end = std::chrono::system_clock::now();
  q.copy(d_in, h_in, in_count).wait();
  q.copy(d_out, h_out, out_count).wait();
  return (unpack_end - unpack_start);
}

template <typename Iterator1, typename Iterator2, typename T>
inline bool check_absolute_error(Iterator1 first1, Iterator1 last1,
                                 Iterator2 first2, T threshold) {
  for (auto iter1 = first1, iter2 = first2; iter1 != last1; ++iter1, ++iter2) {
    if (((*iter1) - (*iter2)) > threshold) {
      return false;
    }
  }
  return true;
}

int main(int argc, char** argv) {
  using namespace srtb::unpack;
  double threshold = 1e-6;
  {
    {
      constexpr int bits = 1;
      const std::byte in{0b01100011};  // decided by coin
      std::array<srtb::real, srtb::BITS_PER_BYTE / bits> out1, out2;
      std::array expected = {0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0};
      SRTB_CHECK_TEST_UNPACK(expected.size() == out1.size());
      unpack_item<bits, /* handwritten = */ false>(&in, &out1[0], 0);
      unpack_item<bits, /* handwritten = */ true>(&in, &out2[0], 0);
      for (auto i : out1) {
        std::cout << i << ' ';
      }
      std::cout << std::endl;
      SRTB_CHECK_TEST_UNPACK(check_absolute_error(out1.begin(), out1.end(),
                                                  expected.begin(), threshold));
      SRTB_CHECK_TEST_UNPACK(check_absolute_error(out2.begin(), out2.end(),
                                                  expected.begin(), threshold));
    }
    {
      constexpr int bits = 2;
      const std::byte in{0b10110110};
      std::array<srtb::real, srtb::BITS_PER_BYTE / bits> out1, out2;
      std::array expected = {2.0, 3.0, 1.0, 2.0};
      SRTB_CHECK_TEST_UNPACK(expected.size() == out1.size());
      unpack_item<bits, /* handwritten = */ false>(&in, &out1[0], 0);
      unpack_item<bits, /* handwritten = */ true>(&in, &out2[0], 0);
      for (auto i : out1) {
        std::cout << i << ' ';
      }
      std::cout << std::endl;
      SRTB_CHECK_TEST_UNPACK(check_absolute_error(out1.begin(), out1.end(),
                                                  expected.begin(), threshold));
      SRTB_CHECK_TEST_UNPACK(check_absolute_error(out2.begin(), out2.end(),
                                                  expected.begin(), threshold));
    }
    {
      constexpr int bits = 4;
      const std::byte in{0b00001000};
      std::array<srtb::real, srtb::BITS_PER_BYTE / bits> out1, out2;
      std::array expected = {0.0, 8.0};
      SRTB_CHECK_TEST_UNPACK(expected.size() == out1.size());
      unpack_item<bits, /* handwritten = */ false>(&in, &out1[0], 0);
      unpack_item<bits, /* handwritten = */ true>(&in, &out2[0], 0);
      for (auto i : out1) {
        std::cout << i << ' ';
      }
      std::cout << std::endl;
      SRTB_CHECK_TEST_UNPACK(check_absolute_error(out1.begin(), out1.end(),
                                                  expected.begin(), threshold));
      SRTB_CHECK_TEST_UNPACK(check_absolute_error(out2.begin(), out2.end(),
                                                  expected.begin(), threshold));
    }
    {
      constexpr int bits = 8;
      const std::byte in{0b10011101};
      std::array<srtb::real, srtb::BITS_PER_BYTE / bits> out1, out2;
      std::array expected = {157.0};
      SRTB_CHECK_TEST_UNPACK(expected.size() == out1.size());
      unpack_item<bits, /* handwritten = */ false>(&in, &out1[0], 0);
      unpack_item<bits, /* handwritten = */ true>(&in, &out2[0], 0);
      for (auto i : out1) {
        std::cout << i << ' ';
      }
      std::cout << std::endl;
      SRTB_CHECK_TEST_UNPACK(check_absolute_error(out1.begin(), out1.end(),
                                                  expected.begin(), threshold));
      SRTB_CHECK_TEST_UNPACK(check_absolute_error(out2.begin(), out2.end(),
                                                  expected.begin(), threshold));
    }
  }

  sycl::queue q;
  {
    // not using uint32_t here as result may depend on big/little endian
    constexpr size_t in_count = 4;
    std::array<std::byte, in_count> in{
        std::byte{0b01100011}, std::byte{0b10110110}, std::byte{0b00001000},
        std::byte{0b10011101}};
    SRTB_CHECK_TEST_UNPACK(in_count == in.size());
    {
      constexpr int bits = 1;
      std::array<srtb::real, in_count * srtb::BITS_PER_BYTE / bits> out1, out2;
      std::array expected = {0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0,
                             1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0,
                             0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                             1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0};
      SRTB_CHECK_TEST_UNPACK(expected.size() == out1.size());
      unpack_host_ptr<bits, /* handwritten = */ false>(&in[0], &out1[0],
                                                       in.size(), q);
      unpack_host_ptr<bits, /* handwritten = */ true>(&in[0], &out2[0],
                                                      in.size(), q);
      SRTB_CHECK_TEST_UNPACK(check_absolute_error(out1.begin(), out1.end(),
                                                  expected.begin(), threshold));
      SRTB_CHECK_TEST_UNPACK(check_absolute_error(out2.begin(), out2.end(),
                                                  expected.begin(), threshold));
    }
    {
      constexpr int bits = 2;
      std::array<srtb::real, in_count * srtb::BITS_PER_BYTE / bits> out1, out2;
      std::array expected = {1.0, 2.0, 0.0, 3.0, 2.0, 3.0, 1.0, 2.0,
                             0.0, 0.0, 2.0, 0.0, 2.0, 1.0, 3.0, 1.0};
      SRTB_CHECK_TEST_UNPACK(expected.size() == out1.size());
      unpack_host_ptr<bits, /* handwritten = */ false>(&in[0], &out1[0],
                                                       in.size(), q);
      unpack_host_ptr<bits, /* handwritten = */ true>(&in[0], &out2[0],
                                                      in.size(), q);
      SRTB_CHECK_TEST_UNPACK(check_absolute_error(out1.begin(), out1.end(),
                                                  expected.begin(), threshold));
      SRTB_CHECK_TEST_UNPACK(check_absolute_error(out2.begin(), out2.end(),
                                                  expected.begin(), threshold));
    }
    {
      constexpr int bits = 4;
      std::array<srtb::real, in_count * srtb::BITS_PER_BYTE / bits> out1, out2;
      std::array expected = {6.0, 3.0, 11.0, 6.0, 0.0, 8.0, 9.0, 13.0};
      SRTB_CHECK_TEST_UNPACK(expected.size() == out1.size());
      unpack_host_ptr<bits, /* handwritten = */ false>(&in[0], &out1[0],
                                                       in.size(), q);
      unpack_host_ptr<bits, /* handwritten = */ true>(&in[0], &out2[0],
                                                      in.size(), q);
      SRTB_CHECK_TEST_UNPACK(check_absolute_error(out1.begin(), out1.end(),
                                                  expected.begin(), threshold));
      SRTB_CHECK_TEST_UNPACK(check_absolute_error(out2.begin(), out2.end(),
                                                  expected.begin(), threshold));
    }
    {
      constexpr int bits = 8;
      std::array<srtb::real, in_count * srtb::BITS_PER_BYTE / bits> out1, out2;
      std::array expected = {99.0, 182.0, 8.0, 157.0};
      SRTB_CHECK_TEST_UNPACK(expected.size() == out1.size());
      unpack_host_ptr<bits, /* handwritten = */ false>(&in[0], &out1[0],
                                                       in.size(), q);
      unpack_host_ptr<bits, /* handwritten = */ true>(&in[0], &out2[0],
                                                      in.size(), q);
      SRTB_CHECK_TEST_UNPACK(check_absolute_error(out1.begin(), out1.end(),
                                                  expected.begin(), threshold));
      SRTB_CHECK_TEST_UNPACK(check_absolute_error(out2.begin(), out2.end(),
                                                  expected.begin(), threshold));
    }
  }
  {
    std::vector<std::byte> in;
    std::vector<srtb::real> out1, out2;
    int test_count = 5;
    size_t in_count = static_cast<size_t>(1) << 16;
    if (argc > 1) {
      try {
        in_count = std::stoi(argv[1]);
      } catch (const std::invalid_argument& ignored) {
      }
    }
    if (argc > 2) {
      try {
        test_count = std::stoi(argv[2]);
      } catch (const std::invalid_argument& ignored) {
      }
    }
    in.resize(in_count);
    std::random_device rd;
    std::mt19937 rng{rd()};
    std::uniform_int_distribution dist{0, 0xFF};
    for (int t = 0; t < test_count; t++) {
      // random number generation is slow?
      std::generate(in.begin(), in.end(),
                    [&]() { return static_cast<std::byte>(dist(rng)); });
#define SRTB_TEST_UNPACK_3(bit)                                            \
  {                                                                        \
    constexpr int bits = bit;                                              \
    out1.resize(in_count* srtb::BITS_PER_BYTE / bits);                     \
    out2.resize(out1.size());                                              \
    unpack_host_ptr<bits, /* handwritten = */ false>(&in[0], &out1[0],     \
                                                     in.size(), q);        \
    unpack_host_ptr<bits, /* handwritten = */ true>(&in[0], &out2[0],      \
                                                    in.size(), q);         \
    SRTB_CHECK_TEST_UNPACK(check_absolute_error(out1.begin(), out1.end(),  \
                                                out2.begin(), threshold)); \
  }
      SRTB_TEST_UNPACK_3(1);
      SRTB_TEST_UNPACK_3(2);
      SRTB_TEST_UNPACK_3(4);
      SRTB_TEST_UNPACK_3(8);
#undef SRTB_TEST_UNPACK_3
    }
  }
  return 0;
}
