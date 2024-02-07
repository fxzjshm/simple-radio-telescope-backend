/******************************************************************************* 
 * Copyright (c) 2022-2023 fxzjshm
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
#ifndef __SRTB_IO_UDP_ASIO_PACKET_PROVIDER__
#define __SRTB_IO_UDP_ASIO_PACKET_PROVIDER__

#include <boost/asio/buffer.hpp>
#include <boost/asio/io_service.hpp>
#include <boost/asio/ip/udp.hpp>
#include <cstddef>
#include <span>
#include <string>

#include "srtb/log/log.hpp"

namespace srtb {
namespace io {
namespace udp {

/**
 * @brief Receive UDP packet using Asio
 */
class asio_packet_provider {
 protected:
  boost::asio::ip::udp::endpoint sender_endpoint, ep2;
  boost::asio::io_service io_service;
  boost::asio::ip::udp::socket socket;

 public:
  asio_packet_provider(const std::string& sender_address,
                       const unsigned short sender_port)
      : sender_endpoint{boost::asio::ip::address::from_string(sender_address),
                        sender_port},
        socket{io_service, sender_endpoint} {
    socket.set_option(boost::asio::ip::udp::socket::reuse_address{true});
    constexpr int socket_buffer_size = std::numeric_limits<int>::max();
    socket.set_option(
        boost::asio::socket_base::receive_buffer_size{socket_buffer_size});
  }

  /**
   * @brief Receive packet into given position
   * @param h_out (mutable) packet will be written to
   */
  auto receive(/* mutable */ std::span<std::byte> h_out) -> size_t {
    auto receive_buffer = boost::asio::buffer(h_out.data(), h_out.size());
    const size_t udp_packet_size = socket.receive_from(receive_buffer, ep2);
    return udp_packet_size;
  }

  /**
   * @brief Receive packet into given position (async version, return immediately)
   * @param h_out (mutable) packet will be written to
   */
  template <typename Callback>
  void receive_async(/* mutable */ std::span<std::byte> h_out,
                     Callback callback) {
    auto receive_buffer = boost::asio::buffer(h_out.data(), h_out.size());
    socket.async_receive_from(
        receive_buffer, ep2,
        [=](const boost::system::error_code& err, std::size_t read_bytes) {
          if (err) [[unlikely]] {
            SRTB_LOGE << " [asio_packet_provider] "
                      << "receiver callback error: " << err.to_string()
                      << srtb::endl;
          } else {
            callback(read_bytes);
          }
        });
  }

  // TODO: is this name appropriate?
  void run_eventloop() {
    io_service.run();
  }
};

}  // namespace udp
}  // namespace io
}  // namespace srtb

#endif  //  __SRTB_IO_UDP_ASIO_PACKET_PROVIDER__
