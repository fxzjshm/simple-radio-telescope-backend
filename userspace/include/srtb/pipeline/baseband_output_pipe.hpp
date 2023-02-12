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

// https://stackoverflow.com/questions/11350878/how-can-i-determine-if-the-operating-system-is-posix-in-c/67627964#67627964
#if __has_include(<unistd.h>)
// System is posix-compliant
#include <unistd.h>
#else
// System is not posix-compliant
#endif

// https://stackoverflow.com/questions/676787/how-to-do-fsync-on-an-ofstream
#include <boost/iostreams/device/file_descriptor.hpp>
#include <boost/iostreams/stream.hpp>
#include <fstream>
#include <vector>

#include "matplotlibcpp.h"
#include "srtb/commons.hpp"
#include "srtb/pipeline/pipe.hpp"

namespace srtb {
namespace pipeline {

/**
 * @brief this pipe reads from baseband output pipe and write it to file.
 */
template <bool continuous_write = false>
class baseband_output_pipe;

template <>
class baseband_output_pipe</* continuous_write = */ true>
    : public pipe<baseband_output_pipe<true> > {
  friend pipe<baseband_output_pipe<true> >;

 protected:
  std::ofstream file_output_stream;

 public:
  baseband_output_pipe() {}

 protected:
  void run_once_impl(std::stop_token stop_token) {
    srtb::work::baseband_output_work baseband_output_work;
    SRTB_POP_WORK_OR_RETURN(" [baseband_output_pipe] ",
                            srtb::baseband_output_queue, baseband_output_work,
                            stop_token);

    // since writing all baseband, ignore signal detect result
    while (!srtb::signal_detect_result_queue.empty()) {
      signal_detect_result_queue.pop();
      // drop()
    }

    // file name need time stamp, so cannot create early
    if (!file_output_stream) [[unlikely]] {
      std::string file_path = srtb::config.baseband_output_file_prefix +
                              std::to_string(baseband_output_work.timestamp) +
                              ".bin";
      file_output_stream = std::ofstream(file_path.c_str(), std::ios::binary);
      if (!file_output_stream) [[unlikely]] {
        auto err = "Cannot open file " + file_path;
        SRTB_LOGE << " [baseband_output_pipe] " << err << srtb::endl;
        throw std::runtime_error{err};
      }
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

// --------------------------------------------------------------------

template <>
class baseband_output_pipe</* continuous_write = */ false>
    : public pipe<baseband_output_pipe<false> > {
  friend pipe<baseband_output_pipe<false> >;
  std::vector<srtb::real> time_series_buffer;

 public:
  baseband_output_pipe() {
    std::string check_file_path =
        srtb::config.baseband_output_file_prefix + "check.bin";
    try {
      boost::iostreams::stream<boost::iostreams::file_descriptor_sink>
          baseband_output_stream{check_file_path,
                                 BOOST_IOS::binary | BOOST_IOS::out};
    } catch (const boost::wrapexcept<std::ios_base::failure>& error) {
      SRTB_LOGE << " [baseband_output_pipe] "
                << "cannot open file " << check_file_path << srtb::endl;
      throw error;
    }
  }

 protected:
  void run_once_impl(std::stop_token stop_token) {
    srtb::work::baseband_output_work baseband_output_work;
    srtb::work::signal_detect_result signal_detect_result;
    SRTB_POP_WORK_OR_RETURN(" [baseband_output_pipe] ",
                            srtb::baseband_output_queue, baseband_output_work,
                            stop_token);
    SRTB_POP_WORK_OR_RETURN(" [baseband_output_pipe] ",
                            srtb::signal_detect_result_queue,
                            signal_detect_result, stop_token);
    while (baseband_output_work.timestamp != signal_detect_result.timestamp)
      [[unlikely]] {
        if (baseband_output_work.timestamp < signal_detect_result.timestamp) {
          SRTB_LOGW << " [baseband_output_pipe] "
                    << "baseband_output_work.timestamp = "
                    << baseband_output_work.timestamp << " < "
                    << "signal_detect_result.timestamp = "
                    << signal_detect_result.timestamp << srtb::endl;
          SRTB_POP_WORK_OR_RETURN(" [baseband_output_pipe] ",
                                  srtb::baseband_output_queue,
                                  baseband_output_work, stop_token);
        } else if (baseband_output_work.timestamp >
                   signal_detect_result.timestamp) {
          SRTB_LOGW << " [baseband_output_pipe] "
                    << "baseband_output_work.timestamp = "
                    << baseband_output_work.timestamp << " > "
                    << "signal_detect_result.timestamp = "
                    << signal_detect_result.timestamp << srtb::endl;
          SRTB_POP_WORK_OR_RETURN(" [baseband_output_pipe] ",
                                  srtb::signal_detect_result_queue,
                                  signal_detect_result, stop_token);
        } else {
          SRTB_LOGE << " [baseband_output_pipe] "
                    << "Logic error. Something must be wrong." << srtb::endl;
        }
      }

    const bool has_signal = (signal_detect_result.time_series.size() > 0);
    if (has_signal) {
      auto timestamp = baseband_output_work.timestamp;
      SRTB_LOGI << " [baseband_output_pipe] "
                << "Begin writing baseband data, timestamp = " << timestamp
                << srtb::endl;

      const std::string file_name_no_extension =
          srtb::config.baseband_output_file_prefix + std::to_string(timestamp);
      const std::string baseband_file_path = file_name_no_extension + ".bin";

      boost::iostreams::stream<boost::iostreams::file_descriptor_sink>
          baseband_output_stream{baseband_file_path,
                                 BOOST_IOS::binary | BOOST_IOS::out};

      // write original baseband data
      const char* baseband_ptr =
          reinterpret_cast<char*>(baseband_output_work.ptr.get());
      const size_t baseband_write_count = baseband_output_work.count;
      baseband_output_stream.write(
          baseband_ptr,
          baseband_write_count *
              sizeof(decltype(baseband_output_work.ptr)::element_type));
      baseband_output_stream.flush();

      // iterate over all time series, assumed with signal
      for (auto time_series_holder : signal_detect_result.time_series) {
        const auto boxcar_length = time_series_holder.boxcar_length;
        const std::string time_series_file_path =
            file_name_no_extension + "." + std::to_string(boxcar_length) +
            ".tim";
        const std::string time_series_picture_file_path =
            time_series_file_path + ".pdf";

        boost::iostreams::stream<boost::iostreams::file_descriptor_sink>
            time_series_output_stream{time_series_file_path,
                                      BOOST_IOS::binary | BOOST_IOS::out};

        // wait until data copy completed
        time_series_holder.transfer_event.wait();

        // write time series
        const auto time_series_ptr = time_series_holder.h_time_series.get();
        const size_t time_series_length = time_series_holder.time_series_length;
        time_series_output_stream.write(
            reinterpret_cast<const char*>(time_series_ptr),
            time_series_length *
                sizeof(
                    decltype(time_series_holder.h_time_series)::element_type));
        time_series_output_stream.flush();

        // draw time series using matplotlib cpp
        try {
          namespace plt = matplotlibcpp;
          time_series_buffer.resize(time_series_length);
          std::copy(time_series_ptr, time_series_ptr + time_series_length,
                    time_series_buffer.begin());
          plt::named_plot(time_series_file_path, time_series_buffer);
          plt::save(time_series_picture_file_path);
          plt::cla();
        } catch (const std::runtime_error& error) {
          SRTB_LOGW << " [baseband_output_pipe] "
                    << "Failed to plot time series: " << error.what()
                    << srtb::endl;
        }

        // check handle of time series data
        if (time_series_output_stream) [[likely]] {
#ifdef _POSIX_VERSION
          // sometimes file is not fully written, so force syncing it
          const int err = ::fdatasync(time_series_output_stream->handle());
          if (err != 0) [[unlikely]] {
            SRTB_LOGW << " [baseband_output_pipe] "
                      << "Failed to call ::fdatasync" << srtb::endl;
          }
#endif
        } else [[unlikely]] {
          SRTB_LOGW << " [baseband_output_pipe] "
                    << "Failed to write baseband data! timestamp = "
                    << timestamp << srtb::endl;
        }
      }

      // check handle of baseband data
      if (baseband_output_stream) [[likely]] {
#ifdef _POSIX_VERSION
        // sometimes file is not fully written, so force syncing it
        const int err = ::fdatasync(baseband_output_stream->handle());
        if (err != 0) [[unlikely]] {
          SRTB_LOGW << " [baseband_output_pipe] "
                    << "Failed to call ::fdatasync" << srtb::endl;
        }
#endif
        SRTB_LOGI << " [baseband_output_pipe] "
                  << "Finished writing baseband data, timestamp = " << timestamp
                  << srtb::endl;
      } else [[unlikely]] {
        SRTB_LOGW << " [baseband_output_pipe] "
                  << "Failed to write baseband data! timestamp = " << timestamp
                  << srtb::endl;
      }
    }
  }
};

}  // namespace pipeline
}  // namespace srtb

#endif  // __SRTB_PIPELINE_FILE_PIPE__
