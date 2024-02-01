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

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <span>
#include <tuple>

#include "srtb/commons.hpp"
#include "srtb/io/vdif_header.hpp"

namespace srtb {
namespace io {
namespace udp {

// parse packet to get: header size, counter, timestamp

/**
 * @brief Format of ROACH 2 board used in NAOC-PSR
 * 
 * Target packet structure (x = 1 byte, in little endian, 8 + 4096 = 4104 bytes):
 *     xxxxxxxx xxxxxxxxxxxx......xxxx
 *     |<--1->| |<------2---......-->|
 *   1. counter of UDP packets of type (u)int64_t, should be sequencially increasing if no packet is lost.
 *   2. real "baseband" data, typical length is 4096 bytes, data type int8_t
 */
struct naocpsr_roach2_packet_parser {
  /** @brief number of polarizations in a single data stream. */
  constexpr static size_t data_stream_count = 1;
  using counter_type = uint64_t;
  static inline constexpr size_t counter_size = sizeof(counter_type);

  static inline constexpr auto parse(std::span<std::byte> udp_packet_buffer) {
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

/**
 * @brief Format of SNAP (1st Gen) board used in NAOC-PSR
 * 
 * Target packet structure (x = 1 byte, in little endian, 8 + 4096 = 4104 bytes):
 *     xxxxxxxx xx xx xx xx xx xx ...... xx xx
 *     |<--0->|  1  2  1  2  1  2 ......  1  2
 *   0. counter of UDP packets of type (u)int64_t, should be sequencially increasing if no packet is lost.
 *   1. 2. real "baseband" data for ADC 1 & 2, typical length is 2 * 2 * 1024 bytes, 
 *         two polarizations interleaved, each with 2 int8_t samples.
 */
struct naocpsr_snap1_packet_parser : public naocpsr_roach2_packet_parser {
  /** @brief number of polarizations in a single data stream. */
  constexpr static size_t data_stream_count = 2;

  // same parse()
  // polarization separation is done later in unpack
};

/**
 * @brief Format of a FPGA sampler board used in GZNU-PSR, output type 1
 * 
 * Target packet structure (x = 1 byte, in little endian, 32 + 32 + 8192 = 8256 bytes):
 *     xxxx......xxxx xxxx......xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx ...... xxxx xxxx xxxx xxxx
 *     |<----a----->| |<----b----->|  1    2    3    4    1    2    3    4   ......  1    2    3    4
 *   a. VDIF header (https://vlbi.org/vlbi-standards/vdif/), and word 6 & 7 forms a uint64_t counter (same as b)
 *   b. counter of UDP packets of type (u)int256_t, should be sequencially increasing if no packet is lost.
 *   1. 2. 3. 4. real "baseband" data for ADC 1, 2, 3 & 4, typical length is 4 * 4 * 512 bytes, 
 *         two polarizations interleaved, each with 2 int8_t samples.
 *   Note: size of a & b is 64 bytes in total because the UDP code they used can only push 64 bytes at once.
 */
struct gznupsr_a1_packet_parser {
  /** @brief number of polarizations in a single data stream; was 4 in original version */
  constexpr static size_t data_stream_count = 2;

  using counter_type = uint64_t;
  static constexpr size_t counter_size = sizeof(counter_type);
  // same size as a vdif header
  static constexpr size_t counter_2_size = 32;

  using vdif_header = srtb::io::vdif_header;
  using vdif_word = vdif_header::vdif_word;
  static constexpr auto vdif_word_size = vdif_header::vdif_word_size;
  static constexpr auto vdif_word_count = vdif_header::vdif_word_count;

  static constexpr size_t packet_header_size = 64;
  static_assert(vdif_word_size * vdif_word_count + counter_2_size ==
                packet_header_size);

  static inline constexpr auto parse(std::span<std::byte> udp_packet_buffer) {
    // what if packet_size < counter_size ?

    std::array<vdif_word, vdif_word_count> word;

    // ref: https://stackoverflow.com/questions/12876361/reading-bytes-in-c
// in this way, endian problem should be solved, ... maybe.
#pragma unroll
    for (size_t i = size_t{0}; i < word.size(); i++) {
      word[i] = 0;
#pragma unroll
      for (size_t j = size_t(0); j < vdif_word_size; j++) {
        word[i] |=
            (static_cast<vdif_word>(udp_packet_buffer[i * vdif_word_size + j])
             << (srtb::BITS_PER_BYTE * j));
      }
    }

    const counter_type received_counter =
        (static_cast<counter_type>(word[6])) |
        (static_cast<counter_type>(word[7])
         << (srtb::BITS_PER_BYTE * vdif_word_size));
    
    const vdif_header vh = std::bit_cast<vdif_header>(word);

    // TODO: timestamp
    return std::make_tuple(/* header_size = */ packet_header_size,
                           received_counter,
                           /* timestamp = */ received_counter);
  }
};

}  // namespace udp
}  // namespace io
}  // namespace srtb

#endif  //  __SRTB_IO_UDP_PACKET_PARSER__
