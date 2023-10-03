/******************************************************************************* 
 * Copyright (c) 2023 fxzjshm
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
#ifndef __SRTB_PIPELINE_WRTIE_MULTI_FILTERBANK_PIPE__
#define __SRTB_PIPELINE_WRTIE_MULTI_FILTERBANK_PIPE__

#include <boost/format.hpp>
#include <chrono>
#include <cstddef>
#include <fstream>
#include <stop_token>

#include "srtb/algorithm/mjd.hpp"
#include "srtb/io/sigproc_filterbank.hpp"
#include "srtb/pipeline/framework/pipe.hpp"
#include "srtb/sycl.hpp"

namespace srtb {

namespace work {

using write_multi_filterbank_work =
    srtb::work::work<std::shared_ptr<std::byte> >;

}

namespace pipeline {

/**
 * @brief this pipe compress dynamic spectrum to 1 bit.
 * @note spectrum has been reversed & cut in fft_r2c_post_process_pipe
 * TODO: in_work.data_stream_id
 */
class write_multi_filterbank_pipe {
 public:
  using in_work_type = srtb::work::write_multi_filterbank_work;
  using out_work_type = srtb::work::dummy_work;
  constexpr static size_t file_size_limit = 1 * (1ull << 30);  // 10 GiB

 public:
  sycl::queue q;
  std::optional<std::ofstream> opt_fout;
  size_t written_bytes;
  uint64_t file_number;
  std::string file_path;

  explicit write_multi_filterbank_pipe(sycl::queue q_)
      : q{q_}, written_bytes{0}, file_number{1} {}

  auto operator()([[maybe_unused]] std::stop_token stop_token,
                  in_work_type in_work) -> out_work_type {
    auto& d_in_shared = in_work.ptr;
    auto d_in = d_in_shared.get();

    const size_t count = in_work.count;
    const size_t batch_size = in_work.batch_size;
    const size_t total_bytes = count * batch_size;

    auto h_in_shared =
        srtb::host_allocator.allocate_shared<std::byte>(total_bytes);
    auto h_in = h_in_shared.get();
    auto event = q.copy(d_in, /* -> */ h_in, total_bytes);

    if (!opt_fout.has_value()) {
      // generate file name
      const auto timestamp = in_work.timestamp;
      const double mjd = srtb::algorithm::unix_timestamp_to_mjd(
          std::chrono::nanoseconds{timestamp});
      const std::string name = (boost::format("%.5f_%04d_%06d.fil") % mjd %
                                in_work.data_stream_id % file_number)
                                   .str();
      file_path = srtb::config.baseband_output_file_prefix + name;

      SRTB_LOGI << " [write_multi_filterbank_pipe] "
                << "Writing to " << name << srtb::endl;

      // create file handle
      if (std::filesystem::exists(file_path)) [[unlikely]] {
        throw std::runtime_error{
            "[write_multi_filterbank_pipe] file already exists: " + file_path};
      }
      opt_fout.emplace(file_path.c_str(), std::ios::binary);
      auto& fout = opt_fout.value();
      if (fout.bad()) [[unlikely]] {
        throw std::runtime_error{
            "[write_multi_filterbank_pipe] cannot write to " + file_path};
      }

      // write header
      // TODO: update these
      using namespace srtb::io::sigproc::filterbank_header;
      send(fout, "HEADER_START");
      send(fout, "telescope_id", int{233});
      send(fout, "machine_id", int{233});
      send(fout, "data_type", int{1});
      send(fout, "fch1",
           double{srtb::config.baseband_freq_low +
                  srtb::config.baseband_bandwidth});
      send(fout, "foff",
           double{srtb::config.baseband_bandwidth /
                  srtb::config.spectrum_channel_count});
      send(fout, "nchans", static_cast<int>(count * srtb::BITS_PER_BYTE));
      send(fout, "tsamp",
           double{1 / srtb::config.baseband_sample_rate *
                  srtb::config.spectrum_channel_count * 2});
      send(fout, "nbeams", int{1});
      send(fout, "ibeam", int{0});
      send(fout, "nbits", int{1});
      send(fout, "nifs", int{1});
      send(fout, "src_raj", double{0.0});
      send(fout, "src_dej", double{0.0});
      send(fout, "tstart", double{mjd});
      send(fout, "HEADER_END");
    }

    event.wait();

    auto& fout = opt_fout.value();

    fout.write(reinterpret_cast<char*>(h_in), total_bytes);

    if (!fout) [[unlikely]] {
      throw std::runtime_error{
          "[write_multi_filterbank_pipe] cannot write to " + file_path};
    }

    written_bytes += total_bytes;
    if (written_bytes >= file_size_limit) {
      // clean up
      fout.flush();
      fout.close();
      opt_fout.reset();
      written_bytes = 0;
      file_number++;
    }

    return srtb::work::dummy_work{};
  }
};

}  // namespace pipeline
}  // namespace srtb

#endif  // __SRTB_PIPELINE_WRTIE_MULTI_FILTERBANK_PIPE__
