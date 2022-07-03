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

int main(int argc, char** argv) {
  int bit = 24;
  if (argc > 1) {
    try {
      bit = std::stoi(argv[1]);
    } catch (std::invalid_argument& ignored) {
      // bit should remain unchanged
    }
  }
  size_t n = 1 << bit;

  srtb::config.baseband_input_length =
      n * srtb::config.baseband_input_bits / srtb::BITS_PER_BYTE;
  assert(srtb::config.unpacked_input_count() == n);
  srtb::config.log_level = static_cast<int>(srtb::log::levels::DEBUG);

  std::vector<sycl::device> devices = sycl::device::get_devices();
  for (auto device : devices) {
    srtb::queue = sycl::queue{device};
    srtb::device_allocator = std::move(
        srtb::memory::cached_allocator<
            srtb::memory::device_allocator<std::byte, srtb::MEMORY_ALIGNMENT> >{
            srtb::queue});
    {
      srtb::fft::init_1d_r2c();
      auto in = srtb::device_allocator.allocate_smart<srtb::real>(n);
      auto out =
          srtb::device_allocator.allocate_smart<srtb::complex>(n / 2 + 1);
      auto start_time = std::chrono::system_clock::now();
      srtb::fft::dispatch_1d_r2c(in.get(), out.get());
      auto end_time = std::chrono::system_clock::now();
      auto run_time = end_time - start_time;
      SRTB_LOGI << " [test fft wrappers] "
                << "size = 2^" << bit << ", time = " << run_time.count()
                << "ns, device name = " << '\"'
                << device.get_info<sycl::info::device::name>() << '\"'
                << std::endl;
    }

    {
      // TODO: batch 1D c2c FFT
    }
  }
  return 0;
}
