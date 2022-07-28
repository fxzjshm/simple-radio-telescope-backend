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

namespace udp_receiver {

inline constexpr size_t UDP_MAX_SIZE = 1 << 16;

/**
 * @brief A source that receives UDP packet and unpack it.
 * 
 * TODO: maybe use lock free ring buffer, see https://ferrous-systems.com/blog/lock-free-ring-buffer/
 * ref: https://www.cnblogs.com/lidabo/p/8317296.html ,
 *      https://stackoverflow.com/questions/37372993/boostasiostreambuf-how-to-reuse-buffer
 */
class udp_receiver_worker {
 protected:
  boost::asio::ip::udp::endpoint sender_endpoint, ep2;
  boost::asio::io_service io_service;
  boost::asio::ip::udp::socket socket{io_service, sender_endpoint};
  boost::asio::streambuf udp_streambuf;

 public:
  udp_receiver_worker(const std::string& sender_address,
                      const unsigned short sender_port)
      : sender_endpoint{boost::asio::ip::address::from_string(sender_address),
                        sender_port},
        socket{io_service, sender_endpoint} {
    socket.set_option(boost::asio::ip::udp::socket::reuse_address{true});
    socket.set_option(boost::asio::socket_base::receive_buffer_size{
        srtb::config.udp_receiver_buffer_size});
  }

  /**
   * @brief Receive given number of bytes
   * @param required_length length in bytes
   * @return buffer of received data.
   * @note extra data received last time is preserved in @c udp_streambuf
   */
  auto receive(size_t required_length) {
    auto time_before_receive = std::chrono::system_clock::now();

    // wait until work size reached
    // TODO: unpack UDP data, check counter
    while (udp_streambuf.size() < required_length) {
      auto receive_buffer = udp_streambuf.prepare(UDP_MAX_SIZE);
      size_t len = socket.receive_from(receive_buffer, ep2);
      udp_streambuf.commit(len);
    }

    auto time_after_receive = std::chrono::system_clock::now();
    auto receive_time = std::chrono::duration_cast<std::chrono::microseconds>(
                            time_after_receive - time_before_receive)
                            .count();
    SRTB_LOGD << " [udp receiver worker] "
              << "recevice time = " << receive_time << " us." << std::endl;

    return udp_streambuf.data();
  }

  void consume(std::size_t n) { udp_streambuf.consume(n); }
};

}  // namespace udp_receiver
}  // namespace io
}  // namespace srtb

#endif  //  __SRTB_IO_UDP_RECEIVER__
