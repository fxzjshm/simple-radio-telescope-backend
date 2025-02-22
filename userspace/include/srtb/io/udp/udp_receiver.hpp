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

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <string>

#include "srtb/log/log.hpp"
#include "srtb/util/assert.hpp"

namespace srtb {
namespace io {
namespace udp {

inline constexpr size_t UDP_MAX_SIZE = 1 << 16;

/**
 * @brief A source that receives UDP packet and unpack it. 
 *        Removed auto-restart and reserve to make sure data stream is continuous.
 *        For original version, see git history of this file.
 * 
 * TODO: simultaneously receive 2 polars (i.e. 2 addresses) and *sync*
 * 
 * @see @c srtb::pipeline::udp_receiver_pipe
 */
template <typename PacketProvider, typename Backend>
class continuous_udp_receiver_worker {
 public:
  using packet_provider_t = PacketProvider;
  using backend_t = Backend;

 protected:
  PacketProvider packet_provider;
  Backend backend;

  /**
   * @brief buffer for receiving one UDP packet
   * @note not directly writing to output because packet counter is in the packet.
   */
  std::array<std::byte, UDP_MAX_SIZE> udp_packet_buffer;
  size_t udp_packet_buffer_pos = 0;
  size_t udp_packet_buffer_size = 0;

  size_t total_received_packet_count = 0;
  size_t total_lost_packet_count = 0;

  /** @brief this type should be able to hold all kinds of counters from different senders */
  using udp_packet_counter_type = uint64_t;
  static constexpr udp_packet_counter_type last_counter_initial_value =
      std::numeric_limits<udp_packet_counter_type>::max();
  /** 
   * @brief counter received from last packet
   */
  udp_packet_counter_type last_counter = last_counter_initial_value;

  size_t zeros_need_to_be_filled = 0;

 public:
  explicit continuous_udp_receiver_worker(const std::string& address, const unsigned short port)
      : packet_provider{address, port} {}

  /**
   * @brief Receive given number of bytes, with counter continuity already checked
   * @param required_length length in bytes
   * @return buffer of received data
   */
  auto receive(std::span<std::byte> data_buffer) {
    std::byte* data_buffer_ptr = data_buffer.data();
    const size_t data_buffer_capacity = data_buffer.size();
    size_t data_buffer_content_size = 0;
    size_t current_received_packet_count = 0;
    size_t current_lost_packet_count = 0;
    udp_packet_counter_type first_counter = 0;
    bool first_counter_set = false;

    // wait until work size reached
    while (data_buffer_content_size < data_buffer_capacity) {
      // fill zero if lost packet too much last time
      auto fill_zero = [&]() {
        const size_t data_buffer_available_length =
            data_buffer_capacity - data_buffer_content_size;
        const size_t zeros_to_be_filled =
            std::min(zeros_need_to_be_filled, data_buffer_available_length);
        std::fill_n(data_buffer_ptr + data_buffer_content_size,
                    zeros_to_be_filled, std::byte{0});
        zeros_need_to_be_filled -= zeros_to_be_filled;
        data_buffer_content_size += zeros_to_be_filled;
      };

      // copy received packet into buffer
      auto copy_packet = [&]() {
        const size_t data_buffer_available_length =
            data_buffer_capacity - data_buffer_content_size;
        const size_t data_to_be_copied_size =
            std::min(udp_packet_buffer_size - udp_packet_buffer_pos,
                     data_buffer_available_length);
        std::copy_n(udp_packet_buffer.begin() + udp_packet_buffer_pos,
                    data_to_be_copied_size, /* -> */
                    data_buffer_ptr + data_buffer_content_size);
        udp_packet_buffer_pos += data_to_be_copied_size;
        data_buffer_content_size += data_to_be_copied_size;
      };

      if (zeros_need_to_be_filled > 0) {
        fill_zero();
      } else if (udp_packet_buffer_pos < udp_packet_buffer_size) [[unlikely]] {
        copy_packet();
      } else [[likely]] {
        // receive packet
        udp_packet_buffer_size =
            packet_provider.receive(std::span<std::byte>{udp_packet_buffer});
        const auto [header_size, received_counter_, timestamp] =
            backend.parse_packet(std::span<std::byte>{udp_packet_buffer.begin(),
                                                     udp_packet_buffer_size});
        const size_t data_len = udp_packet_buffer_size - header_size;

        // remember to check whether this can hold all kinds of counters
        const udp_packet_counter_type received_counter = received_counter_;
        if (!first_counter_set) [[unlikely]] {
          first_counter = received_counter;
          first_counter_set = true;
        }

        size_t lost_packets_count = received_counter - last_counter - 1;
        if (last_counter ==
            last_counter_initial_value /* first counter may not be 0 */)
            [[unlikely]] {
          // this is first time that a packet is received, not really a packet lost
          lost_packets_count = 0;
        }
        current_lost_packet_count += lost_packets_count;
        current_received_packet_count++;
        BOOST_ASSERT(zeros_need_to_be_filled == 0);
        zeros_need_to_be_filled += data_len * lost_packets_count;
        fill_zero();
        last_counter = received_counter;

        udp_packet_buffer_pos = header_size;
        copy_packet();
      }
    }

    // packet counts & warning if lost
    total_lost_packet_count += current_lost_packet_count;
    total_received_packet_count += current_received_packet_count;
    if (current_lost_packet_count > 0) {
      const auto loss_rate =
          1.0 * total_lost_packet_count /
          (total_lost_packet_count + total_received_packet_count);
      SRTB_LOGW << " [udp receiver worker] "
                << "data loss detected: " << current_lost_packet_count
                << " packets this round. Filling these with zero. "
                << "overall loss rate: " << loss_rate << srtb::endl;
    }

    return std::make_pair(data_buffer, first_counter);
  }
};

}  // namespace udp
}  // namespace io
}  // namespace srtb

#endif  //  __SRTB_IO_UDP_RECEIVER__
