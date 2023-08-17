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
#ifndef __SRTB_PIPELINE_COPY_TO_DEVICE_PIPE__
#define __SRTB_PIPELINE_COPY_TO_DEVICE_PIPE__

#include <cstddef>
#include <stop_token>

#include "srtb/pipeline/framework/pipe.hpp"
#include "srtb/sycl.hpp"

namespace srtb {
namespace pipeline {

/**
 * @brief this pipe copy data from host to device.
 * TODO: check if device has unified memory access
 */
class copy_to_device_pipe {
 public:
  sycl::queue q;

  auto operator()([[maybe_unused]] std::stop_token stop_token,
                  srtb::work::copy_to_device_work work) {
    auto& h_in_shared = work.baseband_data.baseband_ptr;
    auto h_in = h_in_shared.get();
    const size_t baseband_input_bytes = work.baseband_data.baseband_input_bytes;

    auto d_in_shared =
        srtb::device_allocator.allocate_shared<std::byte>(baseband_input_bytes);
    auto d_in = d_in_shared.get();

    q.copy(h_in, /* -> */ d_in, baseband_input_bytes).wait();

    srtb::work::unpack_work unpack_work;
    unpack_work.ptr = std::move(d_in_shared);
    unpack_work.count = baseband_input_bytes;
    unpack_work.baseband_data = std::move(work.baseband_data);
    unpack_work.timestamp = work.timestamp;
    unpack_work.udp_packet_counter = work.udp_packet_counter;
    unpack_work.baseband_input_bits = srtb::config.baseband_input_bits;
    return std::optional{unpack_work};
  }
};

}  // namespace pipeline
}  // namespace srtb

#endif  // __SRTB_PIPELINE_COPY_TO_DEVICE_PIPE__
