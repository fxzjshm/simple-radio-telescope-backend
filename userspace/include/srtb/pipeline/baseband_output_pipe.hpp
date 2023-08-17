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
// -- divide line for clang-format --
#include <boost/asio/post.hpp>
#include <boost/asio/thread_pool.hpp>
#include <ctime>
#include <deque>
#include <fstream>
#include <optional>
#include <vector>

#include "cnpy.h"
#include "matplotlibcpp.h"
#include "srtb/commons.hpp"
#include "srtb/coherent_dedispersion.hpp"
#include "srtb/pipeline/framework/pipe.hpp"

namespace srtb {
namespace pipeline {

/**
 * @brief this pipe reads from baseband output pipe and write it to file.
 */
template <bool continuous_write = false>
class baseband_output_pipe;

template<>
class baseband_output_pipe</* continuous_write = */ true> {
 protected:
  std::optional<std::ofstream> opt_file_output_stream;
  std::string file_path;
  sycl::queue q;

 public:
  baseband_output_pipe(sycl::queue q_) : q{q_} {}

  auto operator()(std::stop_token stop_token,
                  srtb::work::baseband_output_work baseband_output_work) {
    //srtb::work::baseband_output_work baseband_output_work;
    //SRTB_POP_WORK_OR_RETURN(" [baseband_output_pipe] ",
    //                        srtb::baseband_output_queue, baseband_output_work,
    //                        stop_token);

    // file name need time stamp, so cannot create early
    if (!opt_file_output_stream) [[unlikely]] {
      auto file_counter = baseband_output_work.udp_packet_counter;
      if (file_counter == baseband_output_work.no_udp_packet_counter) {
        file_counter = baseband_output_work.timestamp;
      }
      file_path = srtb::config.baseband_output_file_prefix +
                  std::to_string(file_counter) + ".bin";
      opt_file_output_stream.emplace(file_path.c_str(), std::ios::binary);
      if (!opt_file_output_stream || !opt_file_output_stream.value())
          [[unlikely]] {
        std::string err = "Cannot open file " + file_path;
        SRTB_LOGE << " [baseband_output_pipe] " << err << srtb::endl;
        throw std::runtime_error{err};
      }
    }

    auto& file_output_stream = opt_file_output_stream.value();

    const char* ptr = reinterpret_cast<char*>(
        baseband_output_work.baseband_data.baseband_ptr.get());
    const size_t baseband_input_count =
        baseband_output_work.baseband_data.baseband_input_bytes;
    size_t write_count;

    // reserved some samples for next round
    const size_t nsamps_reserved = srtb::codd::nsamps_reserved();
    const size_t nbytes_reserved = nsamps_reserved *
                                   srtb::config.baseband_input_bits /
                                   srtb::BITS_PER_BYTE;

    if (nbytes_reserved < baseband_input_count) {
      write_count = baseband_input_count - nbytes_reserved;
      SRTB_LOGD << " [baseband_output_pipe] "
                << "reserved " << nbytes_reserved << " bytes" << srtb::endl;
    } else {
      SRTB_LOGW << " [baseband_output_pipe] "
                << "baseband_input_count = " << baseband_input_count
                << " >= nbytes_reserved = " << nbytes_reserved << srtb::endl;
      write_count = baseband_input_count;
    }
    file_output_stream.write(
        ptr, write_count * sizeof(decltype(baseband_output_work.baseband_data
                                               .baseband_ptr)::element_type));
    if (!file_output_stream) [[unlikely]] {
      std::string err = "Cannot write to " + file_path;
      SRTB_LOGE << " [baseband_output_pipe] " << err << srtb::endl;
      throw std::runtime_error{err};
    }

    srtb::pipeline::notify();
    return std::optional{srtb::work::dummy_work{}};
  }
};

// --------------------------------------------------------------------

template<>
class baseband_output_pipe</* continuous_write = */ false> {
  /** @brief local container of recent works with no signal detected (negative) */
  std::deque<srtb::work::baseband_output_work> recent_negative_works;
  /** @brief local container of timestamps of recent works with signal detected (positive) */
  std::deque<uint64_t> recent_positive_timestamps;

  /** @brief temporary memory lent to time_series_plot_thread_pool for using matplotlib-cpp */
  std::vector<srtb::real> time_series_buffer;
  boost::asio::thread_pool baseband_output_thread_pool;
  // CPython interpretor is not thread-safe, so only one thread can access it.
  // Actually just don't want to introduce another pipe...
  boost::asio::thread_pool time_series_plot_thread_pool{1};

  sycl::queue q;

 public:
  baseband_output_pipe(sycl::queue q_) : q{q_} {
    // check if directory is writable, also record start time
    std::string check_file_path = srtb::config.baseband_output_file_prefix +
                                  "begin_" + generate_time_tag();
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

  ~baseband_output_pipe() {
    // record end time
    // TODO: log file
    std::string check_file_path =
        srtb::config.baseband_output_file_prefix + "end_" + generate_time_tag();
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
  auto generate_time_tag() -> std::string {
    // modified from example from https://en.cppreference.com/w/cpp/chrono/c/strftime
    std::time_t time = std::time({});
    // '\0' should included in string literal
    char time_string[std::size("yyyymmdd_hhmmss")];
    std::strftime(std::data(time_string), std::size(time_string),
                  "%Y%m%d_%H%M%S", std::gmtime(&time));
    return std::string{time_string};
  }

  auto operator()(std::stop_token stop_token, srtb::work::baseband_output_work baseband_output_work) {
    //srtb::work::baseband_output_work baseband_output_work;
    //SRTB_POP_WORK_OR_RETURN(" [baseband_output_pipe] ",
    //                        srtb::baseband_output_queue, baseband_output_work,
    //                        stop_token);
    std::optional<srtb::work::baseband_output_work> opt_work_to_write;

    const bool has_signal = (baseband_output_work.time_series.size() > 0);
    const bool real_time_processing = (srtb::config.input_file_path == "");
    bool overlap_with_recent_positive = false;
    const double overlap_window = 0.45 * 1e9 *
                                  srtb::config.baseband_input_count /
                                  srtb::config.baseband_sample_rate;

    // clean outdated positive results
    while (real_time_processing && recent_positive_timestamps.size() > 0 &&
           static_cast<int64_t>(baseband_output_work.timestamp -
                                recent_positive_timestamps.front()) >
               5 * overlap_window) {
      recent_positive_timestamps.pop_front();
    }

    // 1) this work has signal
    if (has_signal) {
      recent_positive_timestamps.push_back(baseband_output_work.timestamp);
      opt_work_to_write = std::move(baseband_output_work);
    }

    // 2) sometimes signal is detected in only one polarization (which is strange)
    //    try writing signals if signal in another polarization is detected recently
    if ((!has_signal) && real_time_processing) {
      for (auto t : recent_positive_timestamps) {
        if (std::abs(static_cast<double>(static_cast<int64_t>(
                baseband_output_work.timestamp - t))) < overlap_window) {
          overlap_with_recent_positive = true;
          break;
        }
      }
      if (overlap_with_recent_positive) {
        opt_work_to_write = std::move(baseband_output_work);
      }
    }

    // send this to recent works
    if (!(has_signal || overlap_with_recent_positive) && real_time_processing) {
      recent_negative_works.push_back(std::move(baseband_output_work));
    }

    // 3) check if some recent negative works overlap with recent positive works
    if (real_time_processing && (!opt_work_to_write.has_value()) &&
        (recent_negative_works.size() > 0)) {
      srtb::work::baseband_output_work work_2 =
          std::move(recent_negative_works.front());
      recent_negative_works.pop_front();

      bool overlap_with_recent_positive_2 = false;
      for (auto t : recent_positive_timestamps) {
        if (std::abs(static_cast<double>(
                static_cast<int64_t>(work_2.timestamp - t))) < overlap_window) {
          overlap_with_recent_positive_2 = true;
          break;
        }
      }
      if (overlap_with_recent_positive_2) {
        opt_work_to_write = std::move(work_2);
      }
    }

    // write results
    if (opt_work_to_write.has_value()) {
      srtb::work::baseband_output_work work_to_write =
          std::move(opt_work_to_write.value());
      auto file_counter = work_to_write.udp_packet_counter;
      if (file_counter == work_to_write.no_udp_packet_counter) {
        file_counter = work_to_write.timestamp;
      }
      SRTB_LOGI << " [baseband_output_pipe] "
                << "Begin writing baseband data, file_counter = "
                << file_counter << srtb::endl;

      const std::string file_name_no_extension =
          srtb::config.baseband_output_file_prefix +
          std::to_string(file_counter);

      // write original baseband
      boost::asio::post(
          baseband_output_thread_pool,
          [=, baseband_data = std::move(work_to_write.baseband_data)]() {
            // write original baseband data
            const std::string baseband_file_path =
                file_name_no_extension + ".bin";
            boost::iostreams::stream<boost::iostreams::file_descriptor_sink>
                baseband_output_stream{baseband_file_path,
                                       BOOST_IOS::binary | BOOST_IOS::out};
            const char* baseband_ptr =
                reinterpret_cast<char*>(baseband_data.baseband_ptr.get());
            const size_t baseband_write_count =
                baseband_data.baseband_input_bytes;
            if (baseband_ptr) {
              baseband_output_stream.write(
                  baseband_ptr,
                  baseband_write_count *
                      sizeof(
                          decltype(baseband_data.baseband_ptr)::element_type));
            } else {
              SRTB_LOGE << " [baseband_output_pipe] "
                        << "baseband pointer not valid!" << srtb::endl;
            }
            baseband_output_stream.flush();

            // check handle of baseband data
            if (baseband_output_stream) [[likely]] {
          // sometimes file is not fully written, so force syncing it
#ifdef _POSIX_VERSION
              if constexpr (std::is_same_v<
                                decltype(baseband_output_stream->handle()),
                                int>) {
                const int err = ::fdatasync(baseband_output_stream->handle());
                if (err != 0) [[unlikely]] {
                  SRTB_LOGW << " [baseband_output_pipe] "
                            << "Failed to call ::fdatasync" << srtb::endl;
                }
              }
#endif
              SRTB_LOGI << " [baseband_output_pipe] "
                        << "Finished writing baseband data, file_counter = "
                        << file_counter << srtb::endl;
            } else [[unlikely]] {
              SRTB_LOGW << " [baseband_output_pipe] "
                        << "Failed to write baseband data! file_counter = "
                        << file_counter << srtb::endl;
            }
          });

      // write spectrum
      if (work_to_write.ptr.get()) [[likely]] {
        boost::asio::post(
            baseband_output_thread_pool,
            [this, file_name_no_extension,
             d_spectrum_shared = std::move(work_to_write.ptr),
             count = work_to_write.count,
             batch_size = work_to_write.batch_size]() {
              const size_t total_count = count * batch_size;
              auto h_spectrum_unique =
                  srtb::host_allocator
                      .allocate_unique<std::complex<srtb::real> >(total_count);
              auto h_spectrum = h_spectrum_unique.get();
              auto d_spectrum = d_spectrum_shared.get();
              static_assert(sizeof(srtb::complex<srtb::real>) ==
                            sizeof(std::complex<srtb::real>));
              q.copy(d_spectrum, /* -> */
                     reinterpret_cast<srtb::complex<srtb::real>*>(h_spectrum),
                     total_count)
                  .wait();
              const std::string spectrum_file_path =
                  file_name_no_extension + ".npy";
              // don't forget to check shape order...
              cnpy::npy_save(spectrum_file_path, h_spectrum,
                             std::vector<size_t>{count, batch_size});
            });
      }

      // write time series & plot
      boost::asio::post(
          time_series_plot_thread_pool,
          [=, this, time_series = std::move(work_to_write.time_series)]() {
            // iterate over all time series, assumed with signal
            for (auto time_series_holder : time_series) {
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
              const auto time_series_ptr =
                  time_series_holder.h_time_series.get();
              const size_t time_series_length =
                  time_series_holder.time_series_length;
              time_series_output_stream.write(
                  reinterpret_cast<const char*>(time_series_ptr),
                  time_series_length *
                      sizeof(decltype(time_series_holder
                                          .h_time_series)::element_type));
              time_series_output_stream.flush();

              // draw time series using matplotlib cpp
              try {
                namespace plt = matplotlibcpp;
                time_series_buffer.resize(time_series_length);
                std::copy(time_series_ptr, time_series_ptr + time_series_length,
                          time_series_buffer.begin());
                plt::backend("Agg");
                plt::named_plot(time_series_file_path, time_series_buffer);
                plt::save(time_series_picture_file_path);
                plt::cla();
              } catch (const std::runtime_error& error) {
                SRTB_LOGW << " [baseband_output_pipe] "
                          << "Failed to plot time series: " << error.what()
                          << srtb::endl;
              }
            }
          });
    }  // if (opt_work_to_write.has_value())

    srtb::pipeline::notify();
    return std::optional{srtb::work::dummy_work{}};
  }  // auto operator()
};

}  // namespace pipeline
}  // namespace srtb

#endif  // __SRTB_PIPELINE_FILE_PIPE__
