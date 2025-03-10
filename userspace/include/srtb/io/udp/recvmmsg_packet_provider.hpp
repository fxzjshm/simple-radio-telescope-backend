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
#ifndef SRTB_IO_UDP_RECVMMSG_PACKET_PROVIDER
#define SRTB_IO_UDP_RECVMMSG_PACKET_PROVIDER

#include <arpa/inet.h>
#include <linux/if_packet.h>
#include <linux/ip.h>
#include <linux/udp.h>
#include <netinet/in.h>
#include <sys/mman.h>
#include <sys/poll.h>
#include <sys/socket.h>
#include <sys/user.h>

#include <cerrno>
#include <cstddef>
#include <cstdlib>
#include <span>
#include <string>

#include "srtb/io/udp/udp_common.hpp"
#include "srtb/util/assert.hpp"

namespace srtb {
namespace io {
namespace udp {

// TODO: use values from config
inline unsigned int recvmmsg_packet_count = 128;

/**
 * @brief Receive UDP packet using recvmmsg
 */
class recvmmsg_packet_provider {
 public:
  /** Alignment: 2MB, huge page size */
  static constexpr size_t buffer_alignment = 1 << 21;

  socket_wrapper sock;
  // TODO: no suitable smart pointer found for aligned allocation
  std::byte* packet_buffer;
  std::vector<iovec> iovecs;
  std::vector<mmsghdr> msgs;
  std::vector<sockaddr_in> sock_from;

  /** These variables may change during receive(), i.e. interval variables of a state machine */
  struct mutable_var_t {
    /** index of packet in packet_view */
    int i_packet;
    /** count of available packets in packet_view */
    int n_packet;
  } mutable_var;

  recvmmsg_packet_provider(std::string address, unsigned short port) {
    {
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

    // allocate packet buffer & advise hugepage
    const auto packet_buffer_size = recvmmsg_packet_count * UDP_MAX_SIZE;
    packet_buffer = reinterpret_cast<std::byte*>(std::aligned_alloc(buffer_alignment, packet_buffer_size));
    madvise(packet_buffer, packet_buffer_size, MADV_HUGEPAGE);

    // prepare for recvmmsg
    iovecs.resize(recvmmsg_packet_count);
    msgs.resize(recvmmsg_packet_count);
    sock_from.resize(recvmmsg_packet_count);
    for (size_t i = 0; i < recvmmsg_packet_count; i++) {
      iovecs.at(i) = {.iov_base = packet_buffer + i * UDP_MAX_SIZE, .iov_len = UDP_MAX_SIZE};
      msgs.at(i) = {.msg_hdr = {.msg_name = &sock_from.at(i),
                                .msg_namelen = sizeof(decltype(sock_from[0])),
                                .msg_iov = &iovecs.at(i),
                                .msg_iovlen = 1,
                                .msg_control = {},
                                .msg_controllen = {},
                                .msg_flags = {}},
                    .msg_len = {}};
    }

    // clean internal states
    mutable_var.i_packet = 0;
    mutable_var.n_packet = 0;
  }

  ~recvmmsg_packet_provider() { std::free(packet_buffer); }

  /**
   * @brief Receive packet and return view of it
   */
  auto receive() -> std::span<std::byte> {
    if (mutable_var.i_packet == mutable_var.n_packet) {
      get_next_block();
    }
    // mutable_var should refreshed

    // update state of next packet
    const auto this_i_packet = mutable_var.i_packet;
    mutable_var.i_packet += 1;

    const auto ret = packet_buffer + this_i_packet * UDP_MAX_SIZE;
#if __has_builtin(__builtin_prefetch)
    __builtin_prefetch(ret, /* rw = read */ 0, /* locality = no */ 0);
#endif

    return std::span{ret, msgs[this_i_packet].msg_len};
  }

  void get_next_block() {
    int n_packet = recvmmsg(sock.sock, &msgs[0], recvmmsg_packet_count, 0, nullptr);

    BOOST_ASSERT(n_packet > 0);

    // read info from block
    mutable_var.i_packet = 0;
    mutable_var.n_packet = n_packet;
  }
};  // namespace udp

}  // namespace udp
}  // namespace io
}  // namespace srtb

#endif  //  SRTB_IO_UDP_RECVMMSG_PACKET_PROVIDER
