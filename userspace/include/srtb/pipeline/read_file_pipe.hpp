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
#include "srtb/pipeline/pipe.hpp"

namespace srtb {
namespace pipeline {

/**
 * @brief this pipe reads from baseband file and push into @c srtb::unpack_queue
 */
class read_file_pipe : public pipe<read_file_pipe> {
  friend pipe<read_file_pipe>;

 public:
 protected:
  void setup_impl() {
    read_file(srtb::config.input_file_path, srtb::config.baseband_input_count,
              srtb::config.baseband_input_bits,
              srtb::config.input_file_offset_bytes, q);
  }

  /**
   * @brief reads binary file of give type T and unpack it and send it into the pipe.
   * 
   * @param file_path path of the file to read
   * @param q a sycl queue to use
   * @note TODO: share some part of the code with unpack pipe
   */
  void read_file(const std::string& file_path,
                 const size_t baseband_input_count,
                 const size_t baseband_input_bits,
                 const size_t input_file_offset_bytes, sycl::queue& q) {
    std::ifstream input_file_stream{file_path};
    const size_t time_sample_bytes =
        baseband_input_count * baseband_input_bits / BITS_PER_BYTE;

    input_file_stream.ignore(input_file_offset_bytes);
    // TODO: reserve data because of dedispersion
    while (input_file_stream) {
      std::shared_ptr<std::byte> h_in_shared =
          srtb::host_allocator.allocate_shared<std::byte>(time_sample_bytes);
      std::byte* h_in = h_in_shared.get();
      q.fill(h_in, std::byte(0), time_sample_bytes).wait();
      // why don't STL use std::byte ? maybe because std::byte is too new ...
      input_file_stream.read(reinterpret_cast<char*>(h_in), time_sample_bytes);

      std::shared_ptr<std::byte> d_in_shared =
          srtb::device_allocator.allocate_shared<std::byte>(time_sample_bytes);
      std::byte* d_in = d_in_shared.get();
      q.copy(h_in, /* -> */ d_in, time_sample_bytes).wait();

      srtb::work::unpack_work unpack_work;
      unpack_work.ptr = d_in_shared;
      unpack_work.count = time_sample_bytes;
      unpack_work.baseband_input_bits = baseband_input_bits;
      SRTB_PUSH_WORK(" [read_file] ", srtb::unpack_queue, unpack_work);

      srtb::pipeline::wait_for_notify();
    }
  }

  void run_once_impl() {
    // nothing to do ...
    // NOTE: here is 1000x sleep time, because thread_query_work_wait_time is of nanosecond
    std::this_thread::sleep_for(std::chrono::microseconds(srtb::config.thread_query_work_wait_time));
  }
};

}  // namespace pipeline
}  // namespace srtb

#endif  // __SRTB_PIPELINE_READ_FILE_PIPE__
