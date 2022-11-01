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
#ifndef __SRTB_PIPELINE_FILE_PIPE__
#define __SRTB_PIPELINE_FILE_PIPE__

#include "srtb/commons.hpp"
#include "srtb/pipeline/pipe.hpp"

namespace srtb {
namespace pipeline {

/**
 * @brief this pipe reads from baseband output pipe and write it to file.
 * TODO: write only if signal is detected.
 */
class baseband_output_pipe : public pipe<baseband_output_pipe> {
  friend pipe<baseband_output_pipe>;

 protected:
  std::ofstream file_output_stream;

 public:
  baseband_output_pipe() {}

 protected:
  void run_once_impl() {
    srtb::work::baseband_output_work baseband_output_work;
    SRTB_POP_WORK(" [baseband_output_pipe] ", srtb::baseband_output_queue,
                  baseband_output_work);
    // currently save into one file
    if (!file_output_stream) [[unlikely]] {
      std::string file_path = srtb::config.baseband_output_file_prefix +
                              std::to_string(baseband_output_work.timestamp) +
                              ".bin";
      file_output_stream = std::ofstream(file_path.c_str(), std::ios::binary);
    }

    const char* ptr = reinterpret_cast<char*>(baseband_output_work.ptr.get());
    size_t baseband_input_count = baseband_output_work.count;
    size_t write_count;

    // reserved some samples for next round
    const size_t nsamps_reserved = srtb::codd::nsamps_reserved();

    if (nsamps_reserved < baseband_input_count) {
      write_count = baseband_input_count - nsamps_reserved;
      SRTB_LOGD << " [baseband_output_pipe] "
                << "reserved " << nsamps_reserved << " samples" << srtb::endl;
    } else {
      SRTB_LOGW << " [baseband_output_pipe] "
                << "baseband_input_count = " << baseband_input_count
                << " >= nsamps_reserved = " << nsamps_reserved << srtb::endl;
      write_count = baseband_input_count;
    }
    file_output_stream.write(
        ptr,
        write_count * sizeof(decltype(baseband_output_work.ptr)::element_type));
  }
};  // namespace srtb

}  // namespace pipeline
}  // namespace srtb

#endif  // __SRTB_PIPELINE_FILE_PIPE__
