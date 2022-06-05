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
#ifndef __SRTB_IO_UDP_RECEIVER__
#define __SRTB_IO_UDP_RECEIVER__

#include <array>
#include <boost/asio.hpp>
#include <boost/bind.hpp>
#include <chrono>
#include <iostream>
#include <thread>

#include "srtb/coherent_dedispersion.hpp"
#include "srtb/commons.hpp"

namespace srtb {
namespace io {

// TODO: maybe use lock free ring buffer, see https://ferrous-systems.com/blog/lock-free-ring-buffer/
namespace udp_receiver {

constexpr size_t UDP_MAX_SIZE = 1 << 16;

// ref: https://www.cnblogs.com/lidabo/p/8317296.html ,
//      https://stackoverflow.com/questions/37372993/boostasiostreambuf-how-to-reuse-buffer
inline void udp_receiver_worker(std::stop_token stop_token,
                                const std::string& sender_address,
                                const unsigned short& sender_port) {
  using boost::asio::ip::udp;
  udp::endpoint sender_endpoint{
      boost::asio::ip::address::from_string(sender_address), sender_port};
  boost::asio::io_service io_service;
  udp::socket socket{io_service, sender_endpoint};
  socket.set_option(boost::asio::ip::udp::socket::reuse_address{true});
  socket.set_option(boost::asio::socket_base::receive_buffer_size{
      srtb::config.udp_receiver_buffer_size});
  boost::asio::streambuf udp_streambuf;
  udp::endpoint ep2;
  sycl::queue q{srtb::queue.get_context(), srtb::queue.get_device()};

  while (!stop_token.stop_requested()) {
    // this config should persist during one work push
    size_t baseband_input_length = srtb::config.baseband_input_length;

    auto time_before_receive = std::chrono::system_clock::now();

    // wait until work size reached
    while (udp_streambuf.size() < baseband_input_length) {
      auto receive_buffer = udp_streambuf.prepare(UDP_MAX_SIZE);
      size_t len = socket.receive_from(receive_buffer, ep2);
      udp_streambuf.commit(len);
    }

    auto time_after_receive = std::chrono::system_clock::now();

    // flush input of baseband_input_length
    std::shared_ptr<std::byte> ptr =
        srtb::device_allocator.allocate_smart(baseband_input_length);
    q.memcpy(reinterpret_cast<void*>(ptr.get()), udp_streambuf.data().data(),
             baseband_input_length)
        .wait();
    bool ret =
        srtb::unpacker_queue.push(srtb::work{ptr, baseband_input_length});
    if (!ret) [[unlikely]] {
      SRTB_LOGE << " [udp_receiver] "
                << "Pushing work of size " << baseband_input_length
                << " to unpacker_queue failed!" << std::endl;
    } else {
      SRTB_LOGD << " [udp_receiver] "
                << "Pushed work of size " << baseband_input_length
                << " to unpacker_queue." << std::endl;
    }

    // reserved some samples for next round
    size_t nsamps_reserved = srtb::codd::nsamps_reserved();

    udp_streambuf.consume(baseband_input_length - nsamps_reserved);

    SRTB_LOGD << " [udp receiver] "
              << "reserved " << nsamps_reserved << " samples" << std::endl;

    auto time_after_push = std::chrono::system_clock::now();
    auto receive_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                            time_after_receive - time_before_receive)
                            .count();
    auto push_work_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                              time_after_push - time_after_receive)
                              .count();
    SRTB_LOGD << " [udp receiver] "
              << "recevice time = " << receive_time << " ms, "
              << "push work time = " << push_work_time << " ms" << std::endl;
  }
}

inline auto run_udp_receiver_worker() {
  std::jthread udp_receiver_thread{udp_receiver_worker,
                                   srtb::config.udp_receiver_sender_address,
                                   srtb::config.udp_receiver_sender_port};
  udp_receiver_thread.detach();
  return udp_receiver_thread;
}

}  // namespace udp_receiver
}  // namespace io
}  // namespace srtb

#endif  //  __SRTB_IO_UDP_RECEIVER__
