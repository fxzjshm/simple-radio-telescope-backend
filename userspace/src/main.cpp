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

int main(int argc, char** argv) {
  srtb::changed_configs = srtb::program_options::parse_arguments(
      argc, argv, std::string(srtb::config.config_file_name));
  srtb::program_options::apply_changed_configs(srtb::changed_configs,
                                               srtb::config);
  // TODO std::thread for other pipelines

  SRTB_LOGI << " [main] "
            << "device name = "
            << srtb::queue.get_device().get_info<sycl::info::device::name>()
            << srtb::endl;

  srtb::pipeline::udp_receiver_pipe udp_receiver_pipe;
  srtb::pipeline::read_file_pipe read_file_pipe;

  if (std::filesystem::exists(srtb::config.input_file_path)) {
    SRTB_LOGI << " [main] "
              << "Reading file " << srtb::config.input_file_path << srtb::endl;
    read_file_pipe.start();
  } else {
    SRTB_LOGI << " [main] "
              << "Receiving UDP packets" << srtb::endl;
    udp_receiver_pipe.start();
  }

  srtb::pipeline::unpack_pipe unpack_pipe;
  unpack_pipe.start();

  srtb::pipeline::fft_1d_r2c_pipe fft_1d_r2c_pipe;
  fft_1d_r2c_pipe.start();

  srtb::pipeline::rfi_mitigation_pipe rfi_mitigation_pipe;
  rfi_mitigation_pipe.start();

  //srtb::pipeline::dedisperse_and_channelize_pipe dedisperse_and_channelize_pipe;
  //dedisperse_and_channelize_pipe.start();
  srtb::pipeline::dedisperse_pipe dedisperse_pipe;
  dedisperse_pipe.start();

  srtb::pipeline::ifft_1d_c2c_pipe ifft_1d_c2c_pipe;
  ifft_1d_c2c_pipe.start();

  srtb::pipeline::refft_1d_c2c_pipe refft_1d_c2c_pipe;
  refft_1d_c2c_pipe.start();

  srtb::pipeline::signal_detect_pipe signal_detect_pipe;
  signal_detect_pipe.start();

  srtb::pipeline::baseband_output_pipe baseband_output_pipe;
  baseband_output_pipe.start();

  srtb::pipeline::simplify_spectrum_pipe simplify_spectrum_pipe;
  simplify_spectrum_pipe.start();

  return srtb::gui::show_gui(argc, argv);
}
