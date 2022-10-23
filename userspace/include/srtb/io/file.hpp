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
#ifndef __SRTB_IO_FILE__
#define __SRTB_IO_FILE__

#include <fstream>
#include <streambuf>

#include "srtb/coherent_dedispersion.hpp"
#include "srtb/commons.hpp"
#include "srtb/fft/fft_window.hpp"
#include "srtb/unpack.hpp"

namespace srtb {
namespace io {
namespace file {

/**
 * @brief reads binary file of give type T and unpack it and send it into the pipe.
 * 
 * @tparam T type of the data in the given binary file to read
 * @param file_path path of the file to read
 * @param q a sycl queue to use
 * @note TODO: share some part of the code with unpack pipe
 */
template <typename T>
void read_file(const std::string& file_path, sycl::queue& q) {
  std::ifstream input_file_stream{file_path};
  const size_t baseband_input_count = srtb::config.baseband_input_count;
  const size_t baseband_input_bits = srtb::config.baseband_input_bits;
  const size_t time_sample_count = baseband_input_count;
  srtb::fft::fft_window_functor_manager<srtb::real, srtb::fft::default_window>
      window_functor_manager{srtb::fft::default_window{},
                             /* n = */ time_sample_count, q};

  input_file_stream.ignore(srtb::config.input_file_offset_bytes);
  // TODO: reserve data because of dedispersion
  while (input_file_stream) {
    std::shared_ptr<T> h_in_shared =
        srtb::host_allocator.allocate_shared<T>(time_sample_count);
    T* h_in = h_in_shared.get();
    q.fill(h_in, T(0), time_sample_count).wait();
    // why don't STL use std::byte ? maybe because std::byte is too new ...
    input_file_stream.read(reinterpret_cast<char*>(h_in),
                           sizeof(T) / sizeof(char) * time_sample_count);

    std::shared_ptr<T> d_in_shared =
        srtb::device_allocator.allocate_shared<T>(time_sample_count);
    T* d_in = d_in_shared.get();
    q.copy(h_in, /* -> */ d_in, time_sample_count).wait();

    srtb::work::unpack_work unpack_work;
    unpack_work.ptr = d_in_shared;
    unpack_work.count = time_sample_count * sizeof(T);
    unpack_work.baseband_input_bits = baseband_input_bits;
    SRTB_PUSH_WORK(" [read_file] ", srtb::unpack_queue, unpack_work);
  }
}

}  // namespace file
}  // namespace io
}  // namespace srtb

#endif  //  __SRTB_IO_FILE__
