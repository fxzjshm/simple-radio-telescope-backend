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

#include <array>
#include <boost/asio/buffer.hpp>
#include <boost/asio/io_service.hpp>
#include <boost/asio/ip/udp.hpp>
#include <cstddef>
#include <span>
#include <string>

#include "srtb/io/udp/udp_common.hpp"

namespace srtb {
namespace io {
namespace udp {

/**
 * @brief Receive UDP packet using Asio
 */
class asio_packet_provider {
 protected:
  alignas(UDP_PACKET_ALIGNMENT) std::array<std::byte, UDP_MAX_SIZE> udp_buffer;

  boost::asio::ip::udp::endpoint receiver_endpoint, sender_endpoint;
  boost::asio::io_service io_service;
  boost::asio::ip::udp::socket socket;

 public:
  asio_packet_provider(std::string address, unsigned short port)
      : receiver_endpoint{boost::asio::ip::address::from_string(address), port}, socket{io_service, receiver_endpoint} {
    socket.set_option(boost::asio::ip::udp::socket::reuse_address{true});
    constexpr int socket_buffer_size = std::numeric_limits<int>::max();
    socket.set_option(
        boost::asio::socket_base::receive_buffer_size{socket_buffer_size});
  }

  /**
   * @brief Receive packet into given position
   * @param h_out (mutable) packet will be written to
   */
  auto receive() -> std::span<std::byte> {
    auto receive_buffer = boost::asio::buffer(udp_buffer.data(), udp_buffer.size());
    const size_t packet_size = socket.receive_from(receive_buffer, sender_endpoint);
    return std::span{udp_buffer.data(), packet_size};
  }
};

}  // namespace udp
}  // namespace io
}  // namespace srtb

#endif  //  __SRTB_IO_UDP_ASIO_PACKET_PROVIDER__
