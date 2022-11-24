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

#include <fstream>

#include "srtb/commons.hpp"
#include "srtb/pipeline/pipe.hpp"

namespace srtb {
namespace pipeline {

/**
 * @brief this pipe reads from baseband output pipe and write it to file.
 */
template <bool continuous_write = srtb::write_all_baseband>
class baseband_output_pipe;

template <>
class baseband_output_pipe<true> : public pipe<baseband_output_pipe<true> > {
  friend pipe<baseband_output_pipe<true> >;

 protected:
  std::ofstream file_output_stream;

 public:
  baseband_output_pipe() {}

 protected:
  void run_once_impl() {
    srtb::work::baseband_output_work baseband_output_work;
    SRTB_POP_WORK(" [baseband_output_pipe] ", srtb::baseband_output_queue,
                  baseband_output_work);

    // since writing all baseband, ignore signal detect result
    while (!srtb::signal_detect_result_queue.empty()) {
      signal_detect_result_queue.pop();
    }

    if (!file_output_stream) {
      std::string file_path = srtb::config.baseband_output_file_prefix +
                              std::to_string(baseband_output_work.timestamp) +
                              ".bin";
      file_output_stream = std::ofstream(file_path.c_str(), std::ios::binary);
    }

    const char* ptr = reinterpret_cast<char*>(baseband_output_work.ptr.get());
    const size_t baseband_input_count = baseband_output_work.count;
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
};

template <>
class baseband_output_pipe<false> : public pipe<baseband_output_pipe<false> > {
  friend pipe<baseband_output_pipe<false> >;

 public:
  baseband_output_pipe() {}

 protected:
  void run_once_impl() {
    srtb::work::baseband_output_work baseband_output_work;
    srtb::work::signal_detect_result signal_detect_result;
    SRTB_POP_WORK(" [baseband_output_pipe] ", srtb::baseband_output_queue,
                  baseband_output_work);
    SRTB_POP_WORK(" [baseband_output_pipe] ", srtb::signal_detect_result_queue,
                  signal_detect_result);
    while (baseband_output_work.timestamp != signal_detect_result.timestamp)
      [[unlikely]] {
        if (baseband_output_work.timestamp < signal_detect_result.timestamp) {
          SRTB_LOGW << "[baseband_output_pipe] "
                    << "baseband_output_work.timestamp = "
                    << baseband_output_work.timestamp << " < "
                    << "signal_detect_result.timestamp = "
                    << signal_detect_result.timestamp << srtb::endl;
          SRTB_POP_WORK(" [baseband_output_pipe] ", srtb::baseband_output_queue,
                        baseband_output_work);
        } else if (baseband_output_work.timestamp >
                   signal_detect_result.timestamp) {
          // baseband_output_work.timestamp > signal_detect_result.timestamp
          SRTB_LOGW << "[baseband_output_pipe] "
                    << "baseband_output_work.timestamp = "
                    << baseband_output_work.timestamp << " > "
                    << "signal_detect_result.timestamp = "
                    << signal_detect_result.timestamp << srtb::endl;
          SRTB_POP_WORK(" [baseband_output_pipe] ",
                        srtb::signal_detect_result_queue, signal_detect_result);
        } else {
          SRTB_LOGE << "[baseband_output_pipe] "
                    << "Logic error. Something must be wrong." << srtb::endl;
        }
      }

    if (signal_detect_result.has_signal) {
      auto timestamp = baseband_output_work.timestamp;
      SRTB_LOGI << "[baseband_output_pipe] "
                << "Begin writing baseband data, timestamp = " << timestamp
                << srtb::endl;
      std::string file_path = srtb::config.baseband_output_file_prefix +
                              std::to_string(timestamp) + ".bin";
      std::ofstream file_output_stream{file_path.c_str(), std::ios::binary};

      const char* ptr = reinterpret_cast<char*>(baseband_output_work.ptr.get());
      const size_t baseband_input_count = baseband_output_work.count;
      const size_t write_count = baseband_input_count;

      file_output_stream.write(
          ptr, write_count *
                   sizeof(decltype(baseband_output_work.ptr)::element_type));
      file_output_stream.close();
      SRTB_LOGI << "[baseband_output_pipe] "
                << "Finished writing baseband data, timestamp = " << timestamp
                << srtb::endl;
    }
  }
};

}  // namespace pipeline
}  // namespace srtb

#endif  // __SRTB_PIPELINE_FILE_PIPE__
