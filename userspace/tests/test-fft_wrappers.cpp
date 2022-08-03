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

#include "srtb/fft/fft.hpp"

#define SRTB_CHECK_TEST_FFT_WRAPPERS(expr)                             \
  SRTB_CHECK(expr, true, {                                             \
    throw std::runtime_error{                                          \
        "[test-fft_wrappers] " #expr " at " __FILE__ ":" +             \
        std::to_string(__LINE__) + " returns " + std::to_string(ret)}; \
  })

/**
 * use as a benchmark:
 * ```bash
 * export SYCL_DEVICE_FILTER=host  # for intel-llvm
 * export HIPSYCL_VISIBILITY_MASK=omp  # for hipSYCL
 * export SRTB_LOG_LEVEL=0
 * for bit in {0..26}; do ./test-fft_wrappers $bit; done
 * ```
 */
int main(int argc, char** argv) {
  int bit = 24, test_count = 3;
  if (argc > 1) {
    try {
      bit = std::stoi(argv[1]);
    } catch (const std::invalid_argument& ignored) {
      // bit should remain unchanged
    }
  }
  if (argc > 2) {
    try {
      test_count = std::stoi(argv[2]);
    } catch (const std::invalid_argument& ignored) {
      // test_count should remain unchanged
    }
  }

  try {
    char* log_env = std::getenv("SRTB_LOG_LEVEL");
    if (log_env == nullptr) {
      throw std::invalid_argument{""};
    }
    srtb::config.log_level = std::stoi(log_env);
  } catch (const std::invalid_argument& ignored) {
    // maybe unset. no, usually unset.
    srtb::config.log_level = static_cast<int>(srtb::log::levels::DEBUG);
  }
  size_t n = static_cast<size_t>(1) << bit;
  srtb::config.baseband_input_length =
      n * srtb::config.baseband_input_bits / srtb::BITS_PER_BYTE;
  SRTB_LOGD << " [test fft wrappers] "
            << "n = " << n << ", "
            << "fft_1d_r2c_input_size = "
            << srtb::fft::default_fft_1d_r2c_input_size() << srtb::endl;
  SRTB_CHECK_TEST_FFT_WRAPPERS(srtb::fft::default_fft_1d_r2c_input_size() == n);

  std::vector<sycl::device> devices = sycl::device::get_devices();
  for (auto device : devices) {
    // set up test environment
    srtb::queue = sycl::queue{device};
    srtb::device_allocator = std::move(
        srtb::memory::cached_allocator<
            srtb::memory::device_allocator<std::byte, srtb::MEMORY_ALIGNMENT> >{
            srtb::queue});
    {
      srtb::fft::init_1d_r2c();
      auto shared_in = srtb::device_allocator.allocate_shared<srtb::real>(n);
      auto shared_out =
          srtb::device_allocator.allocate_shared<srtb::complex<srtb::real> >(
              n / 2 + 1);
      auto in = shared_in.get();
      auto out = shared_out.get();
      for (int i = 0; i < test_count; i++) {
        auto start_time = std::chrono::system_clock::now();
        srtb::fft::dispatch_1d_r2c(in, out);
        auto end_time = std::chrono::system_clock::now();
        auto run_time = end_time - start_time;
        SRTB_LOGI << " [test fft wrappers] "
                  << "size = 2^" << bit << ", time = " << run_time.count()
                  << "ns, device name = " << '\"'
                  << device.get_info<sycl::info::device::name>() << '\"'
                  << srtb::endl;
        if (i == test_count - 1) {
          std::cerr << bit << " " << run_time.count() << srtb::endl;
        }
      }
    }

    {
      // TODO: batch 1D c2c FFT
    }
  }
  return 0;
}
