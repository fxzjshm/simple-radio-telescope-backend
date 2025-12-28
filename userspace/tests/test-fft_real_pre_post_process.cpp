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

#include <fftw3.h>

#include <random>
#include <ranges>

#include "srtb/fft/fft.hpp"
#include "srtb/fft/fft_1d_r2c_post_process.hpp"
#include "srtb/log/log.hpp"
#include "test-common.hpp"

#define SRTB_CHECK_TEST_FFT_REAL_PRE_POST_PROCESS(expr)                                      \
  SRTB_CHECK(expr, true, {                                                                   \
    throw std::runtime_error{"[test-fft_real_pre_post_process] " #expr " at " __FILE__ ":" + \
                             std::to_string(__LINE__) + " returns " + std::to_string(ret)};  \
  })

template <typename Container>
void print_container(Container container) {
  for (auto x : container) {
    std::cout << x << " ";
  }
  std::cout << std::endl;
}

void write_file(const char* file_name, void* ptr, size_t n_byte) {
  FILE* file = fopen(file_name, "wb");
  fwrite(ptr, n_byte, 1, file);
  fclose(file);
}

// Function to calculate standard deviation using C++20 ranges
// by Bing Chat
template <std::ranges::input_range Range>
double standard_deviation(const Range& data) {
    if (std::ranges::empty(data)) {
        throw std::invalid_argument("Data set cannot be empty.");
    }

    // Convert to double for precision
    auto size = static_cast<double>(std::ranges::distance(data));

    // Calculate mean
    double mean = std::accumulate(std::ranges::begin(data), std::ranges::end(data), 0.0) / size;

    // Calculate variance
    double variance = std::accumulate(
        std::ranges::begin(data), std::ranges::end(data), 0.0,
        [mean](double acc, auto value) {
            double diff = static_cast<double>(value) - mean;
            return acc + diff * diff;
        }
    ) / size; // For population SD; use (size - 1) for sample SD

    return std::sqrt(variance);
}

int main(int argc, char** argv) {
  int bit = 20;
  size_t batch_size = 5, test_count = 1;
  const int print_result_threshold = 32;
  if (argc > 1) {
    try {
      bit = std::stoi(argv[1]);
    } catch (const std::invalid_argument& ignored) {
      // bit should remain unchanged
    }
  }
  if (argc > 2) {
    try {
      batch_size = std::stoi(argv[2]);
    } catch (const std::invalid_argument& ignored) {
      // batch_size should remain unchanged
    }
  }
  if (argc > 3) {
    try {
      test_count = std::stoi(argv[3]);
    } catch (const std::invalid_argument& ignored) {
      // test_count should remain unchanged
    }
  }

  const size_t n = static_cast<size_t>(1) << bit;
  const size_t n_real = n;
  const size_t n_complex = n / 2 + 1;
  const size_t n_complex_in_place = n / 2;

  // 5% is a common error threshold in general physics experiments
  // another 5% since ignoring highest FFT point
  const srtb::real threshold = 0.1;
  SRTB_LOGI << " [test-fft_real_pre_post_process] "
            << "n = " << n << ", "
            << "batch_size = " << batch_size << srtb::endl;
  // SRTB_LOGW << " [test-fft_real_pre_post_process] "
  //           << "this test does not check since"

  const size_t total_size_real = n_real * batch_size;
  const size_t total_size_complex = n_complex * batch_size;
  const size_t total_size_complex_in_place = n_complex_in_place * batch_size;

  // init host data
  std::vector<srtb::real> h_in(total_size_real);
  std::vector<srtb::complex<srtb::real>> h_r2c(total_size_complex), h_r2c_in_place(total_size_complex_in_place);
  std::vector<srtb::real> h_c2r(total_size_real), h_c2r_in_place(total_size_real);
  std::mt19937 rng{233};
  std::generate(h_in.begin(), h_in.end(),
                [&]() { return static_cast<srtb::real>(static_cast<int>(rng())) / static_cast<srtb::real>(INT_MAX); });
  // for test
  for (size_t i_batch = 0; i_batch < batch_size; i_batch++) {
    std::iota(h_in.begin() + n_real * i_batch, h_in.begin() + n_real * (i_batch + 1), srtb::real{0});
  }

  std::vector<sycl::device> devices = sycl::device::get_devices();
  // test for every device
  for (auto device : devices) {
    // set up test environment
    sycl::queue q = sycl::queue{device};
    srtb::device_allocator =
#ifdef SRTB_USE_USM_SHARED_MEMORY
        srtb::memory::cached_allocator<
            sycl::usm_allocator<std::byte, sycl::usm::alloc::shared, srtb::MEMORY_ALIGNMENT>>{q};
#else
        srtb::memory::cached_allocator<srtb::memory::device_allocator<std::byte, srtb::MEMORY_ALIGNMENT>>{q};
#endif
    SRTB_LOGI << " [test-fft_real_pre_post_process] "
              << "device name = " << '\"' << device.get_info<sycl::info::device::name>() << '\"' << srtb::endl;
    {
      srtb::fft::fft_1d_dispatcher<srtb::fft::type::R2C_1D> dispatcher_r2c{n_real, batch_size, q};
      srtb::fft::fft_1d_dispatcher<srtb::fft::type::C2C_1D_FORWARD> dispatcher_c2c_forward{n_complex_in_place,
                                                                                           batch_size, q};
      srtb::fft::fft_1d_dispatcher<srtb::fft::type::C2R_1D> dispatcher_c2r{n_real, batch_size, q};
      srtb::fft::fft_1d_dispatcher<srtb::fft::type::C2C_1D_BACKWARD> dispatcher_c2c_backward{n_complex_in_place,
                                                                                             batch_size, q};
      auto d_real_shared = srtb::device_allocator.allocate_shared<srtb::real>(total_size_real);
      auto d_complex_shared = srtb::device_allocator.allocate_shared<srtb::complex<srtb::real>>(total_size_complex);
      auto d_complex_in_place_shared =
          srtb::device_allocator.allocate_shared<srtb::complex<srtb::real>>(total_size_complex_in_place);
      auto d_real = d_real_shared.get();
      auto d_complex = d_complex_shared.get();
      auto d_complex_in_place = d_complex_in_place_shared.get();
      for (size_t i = 0; i < test_count; i++) {
        q.copy(&h_in[0], d_real, total_size_real).wait();

        // 1. r2c vs. r2c in place (c2c + post process)
        dispatcher_r2c.process(d_real, d_complex);
        q.copy(d_complex, &h_r2c[0], total_size_complex).wait();
        dispatcher_c2c_forward.process(reinterpret_cast<srtb::complex<srtb::real>*>(d_real), d_complex_in_place);
        srtb::fft::fft_1d_r2c_in_place_post_process(d_complex_in_place, n_complex_in_place, batch_size, q);
        q.copy(d_complex_in_place, &h_r2c_in_place[0], total_size_complex_in_place).wait();
        if (n <= print_result_threshold) {
          std::cout << "h_in" << std::endl;
          print_container(h_in);
          std::cout << "h_r2c" << std::endl;
          print_container(h_r2c);
          std::cout << "h_r2c_in_place" << std::endl;
          print_container(h_r2c_in_place);
        }
        {
          write_file("h_in.bin", &h_in[0], sizeof(srtb::real) * total_size_real);
          write_file("h_r2c.bin", &h_r2c[0], sizeof(srtb::complex<srtb::real>) * total_size_complex);
          write_file("h_r2c_in_place.bin", &h_r2c_in_place[0],
                     sizeof(srtb::complex<srtb::real>) * total_size_complex_in_place);
        }
        // // check results
        // SRTB_CHECK_TEST_FFT_REAL_PRE_POST_PROCESS(
        //     check_relative_error(h_r2c_in_place.begin(), h_r2c_in_place.end(), h_r2c.begin(), threshold));

        // reset complex spectrum to standard
        q.parallel_for(sycl::range<2>(n_complex_in_place, batch_size), [=](sycl::item<2> id) {
           const size_t i = id.get_id(0);
           const size_t l = id.get_id(1);
           d_complex_in_place[l * n_complex_in_place + i] = d_complex[l * n_complex + i];
           if (i == 0) {
             d_complex[l * n_complex + n_complex - 1] = 0;
           }
         }).wait();

        // 2. c2r vs. c2r in place (pre process + c2c)
        dispatcher_c2r.process(d_complex, d_real);
        q.copy(d_real, &h_c2r[0], total_size_real).wait();
        srtb::fft::fft_1d_c2r_in_place_pre_process(d_complex_in_place, n_complex_in_place, batch_size, q);
        dispatcher_c2c_backward.process(d_complex_in_place, reinterpret_cast<srtb::complex<srtb::real>*>(d_real));
        q.copy(d_real, &h_c2r_in_place[0], total_size_real).wait();
        if (n <= print_result_threshold) {
          std::cout << "h_c2r" << std::endl;
          print_container(h_c2r);
          std::cout << "h_c2r_in_place" << std::endl;
          print_container(h_c2r_in_place);
        }
        {
          write_file("h_c2r.bin", &h_c2r[0], sizeof(srtb::real) * total_size_real);
          write_file("h_c2r_in_place.bin", &h_c2r_in_place[0], sizeof(srtb::real) * total_size_real);
        }
        // check results
        const srtb::real rms = standard_deviation(h_c2r);
        std::cout << "rms = " << rms << std::endl;
        SRTB_CHECK_TEST_FFT_REAL_PRE_POST_PROCESS(
            check_absolute_error(h_c2r.begin(), h_c2r.end(), h_c2r_in_place.begin(), threshold * rms));
      }
    }
  }
  return 0;
}