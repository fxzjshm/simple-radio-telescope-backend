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
#include <iostream>

#include "srtb/commons.hpp"

namespace srtb {
namespace io {

// TODO: maybe use lock free ring buffer, see https://ferrous-systems.com/blog/lock-free-ring-buffer/
namespace udp_receiver {

constexpr size_t UDP_MAX_SIZE = 1 << 16;

// ref: https://www.cnblogs.com/lidabo/p/8317296.html ,
//      https://stackoverflow.com/questions/37372993/boostasiostreambuf-how-to-reuse-buffer
inline void receiver_worker(const std::string& sender_address,
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
  while (1) {
    // this config should persist during one work push
    size_t baseband_input_length = srtb::config.baseband_input_length;

    // wait until work size reached
    while (udp_streambuf.size() < baseband_input_length) {
      auto receive_buffer = udp_streambuf.prepare(UDP_MAX_SIZE);
      size_t len = socket.receive_from(receive_buffer, ep2);
      udp_streambuf.commit(len);
    }

    // flush input of baseband_input_length
    void* ptr = srtb::device_allocator.allocate(baseband_input_length);
    queue.memcpy(ptr, udp_streambuf.data().data(), baseband_input_length)
        .wait();
    bool ret =
        srtb::unpacker_queue.push(srtb::work{ptr, baseband_input_length});
    if (!ret) [[unlikely]] {
      SRTB_LOGE << " [udp_receiver] Pushing work of size "
                << baseband_input_length << " to unpacker_queue failed!"
                << std::endl;
    } else {
      SRTB_LOGD << " [udp_receiver] Pushed work of size "
                << baseband_input_length << "to unpacker_queue." << std::endl;
    }
    udp_streambuf.consume(baseband_input_length); // TODO: take dedisperion into account
  }
}

}  // namespace udp_receiver
}  // namespace io
}  // namespace srtb

#endif  //  __SRTB_IO_UDP_RECEIVER__