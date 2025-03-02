/******************************************************************************* 
 * Copyright (c) 2025 fxzjshm
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
#ifndef __SRTB_IO_UDP_RECVFROM_PACKET_PROVIDER__
#define __SRTB_IO_UDP_RECVFROM_PACKET_PROVIDER__

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <array>
#include <cstring>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>

#include "srtb/io/udp/udp_common.hpp"

namespace srtb {
namespace io {
namespace udp {

/**
 * @brief Receive UDP packet using recv/recvfrom
 */
class recvfrom_packet_provider {
 protected:
  alignas(UDP_PACKET_ALIGNMENT) std::array<std::byte, UDP_MAX_SIZE> udp_buffer;
  socket_wrapper sock;

 public:
  recvfrom_packet_provider(std::string address, unsigned short port) {
    sockaddr_in servaddr = {};
    servaddr.sin_family = AF_INET;
    servaddr.sin_port = htons(port);
    servaddr.sin_addr.s_addr = inet_addr(address.c_str());
    const int bind_ret = bind(sock.sock, reinterpret_cast<sockaddr*>(&servaddr), sizeof(servaddr));
    if (bind_ret < 0) [[unlikely]] {
      std::string msg = std::string{"Bind to "} + address + ":" + std::to_string(int{port});
      throw std::runtime_error{msg + " failed: " + std::to_string(bind_ret)};
    }
    // maximize socket receive buffer size
    const int n = std::numeric_limits<int>::max();
    setsockopt(sock.sock, SOL_SOCKET, SO_RCVBUF, &n, sizeof(n));
  }

  /**
   * @brief Receive packet into given position
   * @param h_out (mutable) packet will be written to
   */
  auto receive() -> std::span<std::byte> {
#if 0  // recvfrom / recv
    sockaddr_in peer_addr;
    socklen_t peer_len;
    const ssize_t packet_size = recvfrom(
        sock.sock, udp_buffer.begin(), udp_buffer.size(), /* flag = */ 0,
        reinterpret_cast<sockaddr*>(&peer_addr), &peer_len);
#else
    const ssize_t packet_size = recv(sock.sock, udp_buffer.begin(), udp_buffer.size(), /* flag = */ 0);
#endif
    return std::span{udp_buffer.data(), static_cast<size_t>(packet_size)};
  }
};

}  // namespace udp
}  // namespace io
}  // namespace srtb

#endif  //  __SRTB_IO_UDP_RECVFROM_PACKET_PROVIDER__
