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
#ifndef __SRTB_IO_UDP_PACKET_PARSER__
#define __SRTB_IO_UDP_PACKET_PARSER__

#include <cstddef>
#include <cstdint>
#include <span>
#include <tuple>

#include "srtb/commons.hpp"

namespace srtb {
namespace io {
namespace udp {

// parse packet to get: header size, counter, timestamp

/**
 * Target packet structure (x = 1 byte, in little endian):
 *    xxxxxxxx xxxxxxxxxxxx......xxxx
 *    |<--1->| |<------2---......-->|
 *   1. counter of UDP packets of type (u)int64_t, should be sequencially increasing if no packet is lost.
 *   2. real "baseband" data, typical length is 4096 bytes.
 */
class naocpsr_roach2_packet_parser {
 public:
  using counter_type = uint64_t;
  static inline constexpr size_t counter_size = sizeof(counter_type);
  static inline auto parse(std::span<std::byte> udp_packet_buffer) {
    // what if packet_size < counter_size ?
    counter_type received_counter = 0;
    // ref: https://stackoverflow.com/questions/12876361/reading-bytes-in-c
// in this way, endian problem should be solved, ... maybe.
#pragma unroll
    for (size_t i = size_t(0); i < counter_size; ++i) {
      received_counter |= (static_cast<counter_type>(udp_packet_buffer[i])
                           << (srtb::BITS_PER_BYTE * i));
    }
    return std::make_tuple(/* header_size = */ counter_size, received_counter,
                           /* timestamp = */ received_counter);
  }
};

class naocpsr_snap1_packet_parser : public naocpsr_roach2_packet_parser {
  // same parse()
  // polarization separation is done later in unpack
};

}  // namespace udp
}  // namespace io
}  // namespace srtb

#endif  //  __SRTB_IO_UDP_PACKET_PARSER__
