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

#pragma once
#ifndef __SRTB_PIPELINE_WRITE_FILE_PIPE__
#define __SRTB_PIPELINE_WRITE_FILE_PIPE__

#include <fstream>
#include <optional>

#include "srtb/coherent_dedispersion.hpp"
#include "srtb/commons.hpp"
#include "srtb/pipeline/framework/pipe.hpp"

namespace srtb {
namespace pipeline {

/**
 * @brief this pipe reads from baseband output queue 
 *        and unconditionally write all of works to one file.
 * @see srtb::pipeline::write_signal_pipe
 */
class write_file_pipe {
 protected:
  std::optional<std::ofstream> opt_file_output_stream;
  std::string file_path;
  sycl::queue q;

 public:
  write_file_pipe(sycl::queue q_) : q{q_} {}

  auto operator()([[maybe_unused]] std::stop_token stop_token,
                  srtb::work::write_file_work write_file_work) {
    // file name need time stamp, so cannot create early
    if (!opt_file_output_stream) [[unlikely]] {
      auto file_counter = write_file_work.udp_packet_counter;
      if (file_counter == write_file_work.no_udp_packet_counter) {
        file_counter = write_file_work.timestamp;
      }
      file_path = srtb::config.baseband_output_file_prefix +
                  std::to_string(file_counter) + ".bin";
      opt_file_output_stream.emplace(file_path.c_str(), std::ios::binary);
      if (!opt_file_output_stream || !opt_file_output_stream.value())
          [[unlikely]] {
        std::string err = "Cannot open file " + file_path;
        SRTB_LOGE << " [write_file_pipe] " << err << srtb::endl;
        throw std::runtime_error{err};
      }
    }

    auto& file_output_stream = opt_file_output_stream.value();

    const char* ptr = reinterpret_cast<char*>(
        write_file_work.baseband_data.baseband_ptr.get());
    const size_t baseband_input_count =
        write_file_work.baseband_data.baseband_input_bytes;
    size_t write_count;

    // reserved some samples for next round
    const size_t nsamps_reserved = srtb::codd::nsamps_reserved();
    const size_t nbytes_reserved = nsamps_reserved *
                                   srtb::config.baseband_input_bits /
                                   srtb::BITS_PER_BYTE;

    if (nbytes_reserved < baseband_input_count) {
      write_count = baseband_input_count - nbytes_reserved;
      SRTB_LOGD << " [write_file_pipe] "
                << "reserved " << nbytes_reserved << " bytes" << srtb::endl;
    } else {
      SRTB_LOGW << " [write_file_pipe] "
                << "baseband_input_count = " << baseband_input_count
                << " >= nbytes_reserved = " << nbytes_reserved << srtb::endl;
      write_count = baseband_input_count;
    }
    file_output_stream.write(
        ptr, write_count * sizeof(decltype(write_file_work.baseband_data
                                               .baseband_ptr)::element_type));
    if (!file_output_stream) [[unlikely]] {
      std::string err = "Cannot write to " + file_path;
      SRTB_LOGE << " [write_file_pipe] " << err << srtb::endl;
      throw std::runtime_error{err};
    }

    return std::optional{srtb::work::dummy_work{}};
  }
};

}  // namespace pipeline
}  // namespace srtb

#endif  // __SRTB_PIPELINE_WRITE_FILE_PIPE__
