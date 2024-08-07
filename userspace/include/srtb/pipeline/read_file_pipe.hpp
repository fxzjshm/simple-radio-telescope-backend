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
#ifndef __SRTB_PIPELINE_READ_FILE_PIPE__
#define __SRTB_PIPELINE_READ_FILE_PIPE__

#include <fstream>
#include <streambuf>

#include "srtb/coherent_dedispersion.hpp"
#include "srtb/commons.hpp"
#include "srtb/io/backend_registry.hpp"
#include "srtb/pipeline/framework/pipe.hpp"

namespace srtb {
namespace pipeline {

/**
 * @brief this pipe reads from baseband file and push into @c srtb::unpack_queue
 */
class read_file_pipe {
 protected:
  sycl::queue q;
  std::ifstream input_file_stream;
  std::streampos logical_file_pos;

 public:
  read_file_pipe(sycl::queue q_) : q{q_} {
    auto file_path = srtb::config.input_file_path;
    auto input_file_offset_bytes = srtb::config.input_file_offset_bytes;

    input_file_stream =
        std::ifstream{file_path, std::ifstream::in | std::ifstream::binary};

    input_file_stream.ignore(input_file_offset_bytes);

    // counting bytes read manually because position of file may stop at end of
    // file when reading to that, but that calculating position when reserving
    // sample needs continuous bytes counting, i.e.
    //         ......xxxxxxxxxxxx00000000
    //                          ^       ^
    //      position in file stream    logical end of this chunk of data
    //                    <-------------|
    //                     samples reserved because of coherent dedispersion
    logical_file_pos = input_file_offset_bytes;
  }

  auto operator()([[maybe_unused]] std::stop_token stop_token,
                  srtb::work::dummy_work) {
    if (input_file_stream) {
      // count per polarization
      auto baseband_input_count = srtb::config.baseband_input_count;
      auto baseband_input_bits = srtb::config.baseband_input_bits;
      const size_t data_stream_count =
          srtb::io::backend_registry::get_data_stream_count(
              srtb::config.baseband_format_type);

      const size_t time_sample_bytes = baseband_input_count *
                                       std::abs(baseband_input_bits) /
                                       BITS_PER_BYTE * data_stream_count;

      std::shared_ptr<char> h_in_shared =
          srtb::host_allocator.allocate_shared<char>(time_sample_bytes);
      char* h_in = h_in_shared.get();
      // parallel memset ?
      std::memset(h_in, 0, time_sample_bytes);
      input_file_stream.read(reinterpret_cast<char*>(h_in), time_sample_bytes);
      logical_file_pos += time_sample_bytes;

      std::shared_ptr<char> d_in_shared =
          srtb::device_allocator.allocate_shared<char>(time_sample_bytes);
      char* d_in = d_in_shared.get();
      q.copy(h_in, /* -> */ d_in, time_sample_bytes).wait();

      // reserved some samples for next round
      const size_t nsamps_reserved = srtb::codd::nsamps_reserved();
      const std::streamoff reserved_bytes = nsamps_reserved *
                                            std::abs(baseband_input_bits) /
                                            BITS_PER_BYTE * data_stream_count;
      if (static_cast<size_t>(reserved_bytes) < time_sample_bytes) {
        logical_file_pos -= reserved_bytes;
        input_file_stream.seekg(logical_file_pos);
        SRTB_LOGD << " [read_file] "
                  << "reserved " << reserved_bytes << " bytes" << srtb::endl;
      } else {
        SRTB_LOGW << " [read_file] "
                  << "time_sample_bytes = " << time_sample_bytes
                  << " >= reserved_bytes = " << reserved_bytes << srtb::endl;
      }

      const uint64_t timestamp =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count();

      srtb::work::copy_to_device_work copy_to_device_work;
      copy_to_device_work.ptr =
          std::reinterpret_pointer_cast<std::byte>(d_in_shared);
      copy_to_device_work.count = time_sample_bytes;
      copy_to_device_work.baseband_data = {
          std::reinterpret_pointer_cast<std::byte>(h_in_shared),
          time_sample_bytes};
      copy_to_device_work.timestamp = timestamp;
      copy_to_device_work.udp_packet_counter =
          copy_to_device_work.no_udp_packet_counter;
      copy_to_device_work.data_stream_id = 0;  // TODO: multiple file streams?
      return std::optional{copy_to_device_work};
    } else {
      // nothing to do ...
      auto file_path = srtb::config.input_file_path;
      SRTB_LOGI << " [read_file] " << file_path << " has been read"
                << srtb::endl;

      return std::optional<srtb::work::copy_to_device_work>{};
    }
  }
};

}  // namespace pipeline
}  // namespace srtb

#endif  // __SRTB_PIPELINE_READ_FILE_PIPE__
