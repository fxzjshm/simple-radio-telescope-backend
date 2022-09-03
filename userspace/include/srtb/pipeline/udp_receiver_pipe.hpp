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
#ifndef __SRTB_PIPELINE_UDP_RECEIVER_PIPE__
#define __SRTB_PIPELINE_UDP_RECEIVER_PIPE__

#include <array>
#include <boost/asio.hpp>
#include <boost/bind.hpp>
#include <chrono>
#include <iostream>

#include "srtb/io/udp_receiver.hpp"
#include "srtb/pipeline/pipe.hpp"

namespace srtb {
namespace pipeline {

/**
 * @brief receive UDP data and transfer to unpack_work_queue.
 * @see @c srtb::io::udp_receiver::udp_receiver_worker
 * TODO: separate reserving samples for coherent dedispersion
 */
class udp_receiver_pipe : public pipe<udp_receiver_pipe> {
  friend pipe<udp_receiver_pipe>;

 protected:
  srtb::io::udp_receiver::udp_receiver_worker worker;

 public:
  udp_receiver_pipe(
      const std::string& sender_address =
          srtb::config.udp_receiver_sender_address,
      const unsigned short sender_port = srtb::config.udp_receiver_sender_port,
      const int udp_receiver_buffer_size =
          srtb::config.udp_receiver_buffer_size)
      : worker{sender_address, sender_port, udp_receiver_buffer_size} {}

 protected:
  void run_once_impl() {
    // this config should persist during one work push
    size_t baseband_input_length = srtb::config.baseband_input_length;
    SRTB_LOGD << " [udp receiver pipe] "
              << "start receiving" << srtb::endl;
    auto buffer = worker.receive(/* required_length = */ baseband_input_length);
    SRTB_LOGD << " [udp receiver pipe] "
              << "receive finished" << srtb::endl;

    auto time_before_push = std::chrono::system_clock::now();

    // flush input of baseband_input_length
    std::shared_ptr<std::byte> ptr =
        srtb::device_allocator.allocate_shared(baseband_input_length);
    q.memcpy(reinterpret_cast<void*>(ptr.get()), buffer.data(),
             baseband_input_length * sizeof(std::byte))
        .wait();
    srtb::work::unpack_work unpack_work{
        {ptr, /* size = */ baseband_input_length},
        srtb::config.baseband_input_bits};
    SRTB_PUSH_WORK(" [udp receiver pipe] ", srtb::unpack_queue, unpack_work);

    // reserved some samples for next round
    size_t nsamps_reserved = srtb::codd::nsamps_reserved();

    if (nsamps_reserved < baseband_input_length) {
      worker.consume(baseband_input_length - nsamps_reserved);
      SRTB_LOGD << " [udp receiver pipe] "
                << "reserved " << nsamps_reserved << " samples" << srtb::endl;
    } else {
      SRTB_LOGW << " [udp receiver pipe] "
                << "baseband_input_length = " << baseband_input_length
                << " >= nsamps_reserved = " << nsamps_reserved << srtb::endl;
      worker.consume(baseband_input_length);
    }

    auto time_after_push = std::chrono::system_clock::now();
    auto push_work_time = std::chrono::duration_cast<std::chrono::microseconds>(
                              time_after_push - time_before_push)
                              .count();
    SRTB_LOGD << " [udp receiver pipe] "
              << "push work time = " << push_work_time << " us" << srtb::endl;
  }
};

}  // namespace pipeline
}  // namespace srtb

#endif  // __SRTB_PIPELINE_UDP_RECEIVER_PIPE__
