/******************************************************************************* 
 * Copyright (c) 2024 fxzjshm
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
#ifndef __SRTB_IO_BACKEND_REGISTRY__
#define __SRTB_IO_BACKEND_REGISTRY__

#include <climits>
#include <cstddef>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>

#include "srtb/io/vdif_header.hpp"

namespace srtb {
namespace io {
/* Hope for reflection. */
namespace backend_registry {

/**
 * @brief Just linear sequence of samples
 */
struct simple {
  constexpr static std::string_view name = "simple";
  constexpr static size_t data_stream_count = 1;
};

// for backend w/ UDP data stream, parse packet to get header size, counter, timestamp

/**
 * @brief ROACH 2 based backend used in NAOC PSR group.
 *
 * Target packet structure (x = 1 byte, in little endian, 8 + 4096 = 4104 bytes):
 *     xxxxxxxx xxxxxxxxxxxx......xxxx
 *     |<--1->| |<------2---......-->|
 *   1. counter of UDP packets of type (u)int64_t, should be sequencially increasing if no packet is lost.
 *   2. real "baseband" data, typical length is 4096 bytes, data type int8_t
 */
struct naocpsr_roach2 {
  constexpr static std::string_view name = "naocpsr_roach2";
  /** @brief number of polarizations in a single data stream. */
  constexpr static size_t data_stream_count = 1;
  using counter_type = uint64_t;
  static inline constexpr size_t counter_size = sizeof(counter_type);
  static inline constexpr size_t packet_header_size = counter_size;
  static inline constexpr size_t packet_payload_size = 4104;

  static inline constexpr auto parse_packet(std::span<std::byte> udp_packet_buffer) {
    // what if packet_size < counter_size ?
    counter_type received_counter = 0;
    // ref: https://stackoverflow.com/questions/12876361/reading-bytes-in-c
// in this way, endian problem should be solved, ... maybe.
#pragma unroll
    for (size_t i = size_t(0); i < counter_size; ++i) {
      received_counter |= (static_cast<counter_type>(udp_packet_buffer[i]) << (CHAR_BIT * i));
    }
    return std::make_tuple(received_counter, /* timestamp = */ received_counter);
  }
};

/**
 * @brief SNAP 1 based backend used in NAOC PSR group.
 * 
 * Target packet structure (x = 1 byte, in little endian, 8 + 4096 = 4104 bytes):
 *     xxxxxxxx xx xx xx xx xx xx ...... xx xx
 *     |<--0->|  1  2  1  2  1  2 ......  1  2
 *   0. counter of UDP packets of type (u)int64_t, should be sequencially increasing if no packet is lost.
 *   1. 2. real "baseband" data for ADC 1 & 2, typical length is 2 * 2 * 1024 bytes, 
 *         two polarizations interleaved, each with 2 int8_t samples.
 */
struct naocpsr_snap1 : public naocpsr_roach2 {
  constexpr static std::string_view name = "naocpsr_snap1";
  /** @brief number of polarizations in a single data stream. */
  constexpr static size_t data_stream_count = 2;

  // same parse_packet(), polarization separation is done later in unpack
};

/**
 * @brief ZCU 111 based backend used in GZNU PSR group.
 *
 * Target packet structure (x = 1 byte, in little endian, 32 + 32 + 8192 = 8256 bytes):
 *   Version 1:
 *     xxxx......xxxx xxxx......xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx ...... xxxx xxxx xxxx xxxx
 *     |<----a----->| |<----b----->|  1    2    3    4    1    2    3    4   ......  1    2    3    4
 *   Version 2 (current):
 *     xxxx......xxxx xxxx......xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx ...... xxxx xxxx xxxx xxxx
 *     |<----a----->| |<----b----->|  1    2    1    2    1    2    1    2   ......  1    2    1    2
 *   a. VDIF header (https://vlbi.org/vlbi-standards/vdif/), and word 6 & 7 forms a uint64_t counter (same as b)
 *   b. counter of UDP packets of type (u)int256_t, should be sequencially increasing if no packet is lost.
 *   1. 2. 3. 4. real "baseband" data for ADC 1, 2, 3 & 4, typical length is 4 * 4 * 512 bytes, 
 *         two polarizations interleaved, each with 2 int8_t samples.
 *   Note: size of a & b is 64 bytes in total because the UDP code they used can only push 64 bytes at once.
 */
struct gznupsr_a1 {
  constexpr static std::string_view name = "gznupsr_a1";
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

  static inline constexpr size_t packet_header_size = 64;
  static_assert(vdif_word_size * vdif_word_count + counter_2_size == packet_header_size);
  static inline constexpr size_t packet_payload_size = 8256;

  static inline constexpr auto parse_packet(std::span<std::byte> udp_packet_buffer) {
    // what if packet_size < counter_size ?

    std::array<vdif_word, vdif_word_count> word;

    // ref: https://stackoverflow.com/questions/12876361/reading-bytes-in-c
// in this way, endian problem should be solved, ... maybe.
#pragma unroll
    for (size_t i = size_t{0}; i < word.size(); i++) {
      word[i] = 0;
#pragma unroll
      for (size_t j = size_t(0); j < vdif_word_size; j++) {
        word[i] |= (static_cast<vdif_word>(udp_packet_buffer[i * vdif_word_size + j]) << (CHAR_BIT * j));
      }
    }

    const counter_type received_counter =
        (static_cast<counter_type>(word[6])) | (static_cast<counter_type>(word[7]) << (CHAR_BIT * vdif_word_size));

    const vdif_header vh = std::bit_cast<vdif_header>(word);

    // TODO: timestamp
    return std::make_tuple(received_counter, /* timestamp = */ received_counter);
  }
};

inline std::tuple<simple, naocpsr_roach2, naocpsr_snap1, gznupsr_a1> backends;

// helper function

inline auto get_data_stream_count(std::string_view backend_name) {
  std::optional<size_t> ret;
  auto f = [&](auto&& backend) {
    if (backend_name == backend.name) {
      ret = backend.data_stream_count;
    }
  };
  std::apply([&](auto&&... args) { ((f(args)), ...); }, backends);
  if (ret.has_value()) {
    return ret.value();
  } else {
    throw std::invalid_argument{"[backend_registry] Unknown backend name \"" +
                                std::string{backend_name} +
                                "\" when getting data stream count"};
  }
}

}  // namespace backend_registry
}  // namespace io
}  // namespace srtb

#endif  //  __SRTB_IO_BACKEND_REGISTRY__
