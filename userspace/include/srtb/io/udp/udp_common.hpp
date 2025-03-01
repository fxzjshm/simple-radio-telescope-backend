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
#ifndef __SRTB_IO_UDP_COMMON__
#define __SRTB_IO_UDP_COMMON__

#include <sys/socket.h>
#include <unistd.h>
#include <stdexcept>
#include <cstddef>

namespace srtb {
namespace io {
namespace udp {

inline constexpr size_t UDP_MAX_SIZE = 1 << 16;

struct socket_wrapper {
  int sock;

  explicit socket_wrapper() {
    sock = socket(PF_INET, SOCK_DGRAM, 0);
    if (sock < 0) {
      throw std::runtime_error{"udp socket creation error"};
    }
  }

  ~socket_wrapper() { close(sock); }
};

}  // namespace udp
}  // namespace io
}  // namespace srtb

#endif  //  __SRTB_IO_UDP_COMMON__
