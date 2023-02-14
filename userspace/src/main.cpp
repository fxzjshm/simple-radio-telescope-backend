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

// Qt's keywords like signals and slots are annoying
#ifndef QT_NO_KEYWORDS
#define QT_NO_KEYWORDS
#endif

#include <chrono>
#include <filesystem>
#include <iostream>
#include <vector>

#include "srtb/coherent_dedispersion.hpp"
#include "srtb/commons.hpp"
#include "srtb/frequency_domain_filterbank.hpp"
#include "srtb/gui/gui.hpp"
#include "srtb/io/udp_receiver.hpp"
#include "srtb/pipeline/baseband_output_pipe.hpp"
#include "srtb/pipeline/dedisperse_and_channelize_pipe.hpp"
#include "srtb/pipeline/fft_pipe.hpp"
#include "srtb/pipeline/read_file_pipe.hpp"
#include "srtb/pipeline/rfi_mitigation_pipe.hpp"
#include "srtb/pipeline/signal_detect_pipe.hpp"
#include "srtb/pipeline/spectrum_pipe.hpp"
#include "srtb/pipeline/udp_receiver_pipe.hpp"
#include "srtb/pipeline/unpack_pipe.hpp"
#include "srtb/program_options.hpp"
#include "srtb/spectrum/simplify_spectrum.hpp"

namespace srtb {
namespace main {

inline namespace detail {

/**
 * @brief allocate memory needed before pipeline start, used in `main()`
 * 
 * for device side, allocation is minimal;
 * for host side resonable spaces are used,
 * because it is observed that allocation of host pinned memory is very slow (~500ms for 1GB)
 *                         but allocation of device memory is quick
 *       & VRAM is usually limited.
 * 
 * this function is quite ugly, and should be updated once pipeline structure chnages.
 */
inline void allocate_memory_regions(size_t input_pipe_count) {
  // hold all pointers; using RAII
  std::vector<std::shared_ptr<std::byte> > ptrs;

  for (size_t k = 0; k < input_pipe_count; k++) {
    // host side udp receiver buffer, raw baseband data
    for (size_t i = 0; i < 5; i++) {
      ptrs.push_back(srtb::host_allocator.allocate_shared<std::byte>(
          sizeof(std::byte) * srtb::config.baseband_input_count *
          std::abs(srtb::config.baseband_input_bits) / srtb::BITS_PER_BYTE));
    }

    // device side raw baseband data, to be unpacked
    ptrs.push_back(srtb::device_allocator.allocate_shared<std::byte>(
        sizeof(std::byte) * srtb::config.baseband_input_count *
        std::abs(srtb::config.baseband_input_bits) / srtb::BITS_PER_BYTE));

    // device side unpacked baseband data / FFT-ed spectrum / STFT-ed waterfall (if in place)
    ptrs.push_back(srtb::device_allocator.allocate_shared<std::byte>(
        sizeof(srtb::complex<srtb::real>) *
        (srtb::config.baseband_input_count / 2 + 1)));

    // device side time series (original / accumulated / boxcar-ed)
    for (size_t i = 0; i < 3; i++) {
      ptrs.push_back(srtb::device_allocator.allocate_shared<std::byte>(
          sizeof(srtb::real) * srtb::config.baseband_input_count /
          srtb::config.refft_length / 2));
    }

    // host side time series for a segment of baseband
    for (size_t w = 1; w <= srtb::config.signal_detect_max_boxcar_length;
         w *= 2) {
      ptrs.push_back(srtb::host_allocator.allocate_shared<std::byte>(
          sizeof(srtb::real) * srtb::config.baseband_input_count /
          srtb::config.refft_length / 2));
    }

    // device side STFT buffer, for spectrural kurtosis & mean value
    for (size_t i = 0; i < 2; i++) {
      ptrs.push_back(srtb::device_allocator.allocate_shared<std::byte>(
          sizeof(srtb::complex<srtb::real>) * srtb::config.refft_length));
    }

    // host & device side waterfall
    ptrs.push_back(srtb::device_allocator.allocate_shared<std::byte>(
        sizeof(srtb::real) *
        (srtb::config.baseband_input_count - srtb::codd::nsamps_reserved()) /
        srtb::config.refft_length / 2 * srtb::gui::spectrum::width));
    for (size_t i = 0; i < 5; i++) {
      ptrs.push_back(srtb::host_allocator.allocate_shared<std::byte>(
          sizeof(srtb::real) *
          (srtb::config.baseband_input_count - srtb::codd::nsamps_reserved()) /
          srtb::config.refft_length / 2 * srtb::gui::spectrum::width));
    }

    // misc value holders
    for (size_t i = 0; i < 2; i++) {
      ptrs.push_back(srtb::device_allocator.allocate_shared<std::byte>(
          sizeof(srtb::real)));
      ptrs.push_back(srtb::device_allocator.allocate_shared<std::byte>(
          sizeof(srtb::complex<srtb::real>)));
    }
  }

  // ptrs.drop(); a.k.a. ~ptrs();
}

}  // namespace detail

int main(int argc, char** argv) {
  srtb::changed_configs = srtb::program_options::parse_arguments(
      argc, argv, std::string(srtb::config.config_file_name));
  srtb::program_options::apply_changed_configs(srtb::changed_configs,
                                               srtb::config);

// why always have to do something special for CUDA ???
#ifdef SRTB_ENABLE_CUDA_INTEROP
  cudaSetDeviceFlags(cudaDeviceScheduleYield);
#endif

  const size_t input_pipe_count =
      std::max(std::max(srtb::config.udp_receiver_sender_address.size(),
                        srtb::config.udp_receiver_sender_port.size()),
               size_t{1});

  allocate_memory_regions(input_pipe_count);

  // trigger JIT
  // some implementations may pack intermediate representation with executable binary
  // and just-in-time compile it into device code when launching first kernel
  {
    auto d_out_unique = srtb::device_allocator.allocate_unique<srtb::real>(1);
    auto d_out = d_out_unique.get();
    srtb::queue.single_task([=]() { (*d_out) = srtb::real{42}; }).wait();
  }

  // TODO std::jthread for other pipelines

  SRTB_LOGI << " [main] "
            << "device name = "
            << srtb::queue.get_device().get_info<sycl::info::device::name>()
            << srtb::endl;

  std::vector<std::jthread> input_thread;
  auto input_file_path = srtb::config.input_file_path;
  if (std::filesystem::exists(input_file_path)) {
    SRTB_LOGI << " [main] "
              << "Reading file " << input_file_path << srtb::endl;
    input_thread.push_back(srtb::pipeline::read_file_pipe::start());
  } else {
    if (input_file_path != "") {
      SRTB_LOGE << " [main] "
                << "Cannot read file " << input_file_path << srtb::endl;
      return EXIT_FAILURE;
    }
    SRTB_LOGI << " [main] "
              << "Receiving UDP packets" << srtb::endl;
    for (size_t i = 0; i < input_pipe_count; i++) {
      input_thread.push_back(
          srtb::pipeline::udp_receiver_pipe::start(/* id = */ i));
    }
  }

  std::jthread unpack_thread = srtb::pipeline::unpack_pipe::start();

  std::jthread fft_1d_r2c_thread = srtb::pipeline::fft_1d_r2c_pipe::start();

  std::jthread rfi_mitigation_thread =
      srtb::pipeline::rfi_mitigation_pipe::start();

  //srtb::pipeline::dedisperse_and_channelize_pipe::start();
  std::jthread dedisperse_thread = srtb::pipeline::dedisperse_pipe::start();

  std::jthread ifft_1d_c2c_thread = srtb::pipeline::ifft_1d_c2c_pipe::start();

  std::jthread refft_1d_c2c_thread = srtb::pipeline::refft_1d_c2c_pipe::start();

  std::jthread signal_detect_thread =
      srtb::pipeline::signal_detect_pipe::start();

  std::jthread baseband_output_thread;
  if (srtb::config.baseband_write_all) {
    SRTB_LOGW << " [main] "
              << "Writing all baseband data, take care of disk space!"
              << srtb::endl;
    baseband_output_thread = srtb::pipeline::baseband_output_pipe<
        /* continuous_write = */ true>::start();
  } else {
    baseband_output_thread = srtb::pipeline::baseband_output_pipe<
        /* continuous_write = */ false>::start();
  }

  std::jthread simplify_spectrum_thread =
      srtb::pipeline::simplify_spectrum_pipe::start();

  std::vector<std::jthread> threads;
  threads = std::move(input_thread);
  threads.push_back(std::move(unpack_thread));
  threads.push_back(std::move(fft_1d_r2c_thread));
  threads.push_back(std::move(rfi_mitigation_thread));
  threads.push_back(std::move(dedisperse_thread));
  threads.push_back(std::move(ifft_1d_c2c_thread));
  threads.push_back(std::move(refft_1d_c2c_thread));
  threads.push_back(std::move(signal_detect_thread));
  threads.push_back(std::move(baseband_output_thread));
  threads.push_back(std::move(simplify_spectrum_thread));

  srtb::pipeline::expected_running_pipe_count = threads.size();
  srtb::pipeline::expected_input_pipe_count = input_pipe_count;

  return srtb::gui::show_gui(argc, argv, std::move(threads));
}

}  // namespace main
}  // namespace srtb

int main(int argc, char** argv) { return srtb::main::main(argc, argv); }
