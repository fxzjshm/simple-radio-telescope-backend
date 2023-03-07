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
#include <boost/asio.hpp>
#include <chrono>
#include <cstring>
#include <iostream>
#include <limits>
#include <thread>
#include <vector>

#include "srtb/commons.hpp"
#include "srtb/memory/memcpy.hpp"

namespace srtb {
namespace io {
namespace udp_receiver {

inline constexpr size_t counter_bytes_count =
    sizeof(udp_packet_counter_type) / sizeof(std::byte);

/**
 * @brief A source that receives UDP packet and unpack it.
 * 
 * Target packet structure (x = 1 byte, in little endian):
 *    xxxxxxxx xxxxxxxxxxxx......xxxx
 *    |<--1->| |<------2---......-->|
 *   1. counter of UDP packets of type (u)int64_t, should be sequencially increasing if no packet is lost.
 *   2. real "baseband" data, typical length is 4096 bytes.
 * 
 * TODO: maybe use lock free ring buffer, see https://ferrous-systems.com/blog/lock-free-ring-buffer/
 *       or move ring buffer to GPU side, when RDMA method is considered.
 * 
 * TODO: simultaneously receive 2 polars (i.e. 2 addresses) and *sync*
 * 
 * @see @c srtb::pipeline::udp_receiver_pipe
 * 
 * ref: https://www.cnblogs.com/lidabo/p/8317296.html ,
 *      https://stackoverflow.com/questions/37372993/boostasiostreambuf-how-to-reuse-buffer
 */
class udp_receiver_worker {
 protected:
  boost::asio::ip::udp::endpoint sender_endpoint, ep2;
  boost::asio::io_service io_service;
  boost::asio::ip::udp::socket socket;

  /**
   * @brief buffer for receiving one UDP packet
   * @note not directly writing to output because packet counter is in the packet.
   */
  std::array<std::byte, UDP_MAX_SIZE> udp_packet_buffer;
  size_t udp_packet_buffer_pos = 0;
  size_t udp_packet_buffer_size = 0;

  size_t total_received_packet_count = 0;
  size_t total_lost_packet_count = 0;

  static constexpr udp_packet_counter_type last_counter_initial_value =
      static_cast<udp_packet_counter_type>(-1);
  /** 
   * @brief counter received from last packet
   */
  udp_packet_counter_type last_counter = last_counter_initial_value;

  size_t zeros_need_to_be_filled = 0;

  std::vector<std::byte> reserved_data_buffer;

 public:
  udp_receiver_worker(const std::string& sender_address,
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
   * @brief Receive given number of bytes, with counter continuity already checked
   * @param required_length length in bytes
   * @param reserved_length length of data reserved for next round
   * @return buffer of received data
   */
  auto receive(size_t required_length, size_t reserved_length) {
    auto time_before_receive = std::chrono::system_clock::now();

    std::shared_ptr<std::byte> data_buffer =
        srtb::host_allocator.allocate_shared<std::byte>(required_length);
    const auto data_buffer_ptr = data_buffer.get();
    const size_t data_buffer_capacity = required_length;
    size_t data_buffer_content_size = 0;
    size_t current_received_packet_count = 0;
    size_t current_lost_packet_count = 0;
    udp_packet_counter_type first_counter = 0;
    bool first_counter_set = false;

    // check if lost too many packets; if so, restart
    if (zeros_need_to_be_filled > data_buffer_capacity) [[unlikely]] {
      // discard reserved data
      reserved_data_buffer.clear();
      zeros_need_to_be_filled = zeros_need_to_be_filled % data_buffer_capacity;
      SRTB_LOGW << " [udp receiver worker] "
                << "too many packets lost, restart" << srtb::endl;
    } else [[likely]] {
      // copy reserved data to the beginning of current data buffer
      const size_t data_buffer_available_length =
          data_buffer_capacity - data_buffer_content_size;
      if (reserved_data_buffer.size() > data_buffer_available_length) {
        SRTB_LOGW << " [udp receiver worker] "
                  << "requested buffer too small, not reserving data"
                  << srtb::endl;
      } else {
        std::copy(reserved_data_buffer.begin(), reserved_data_buffer.end(),
                  /* -> */ data_buffer_ptr + data_buffer_content_size);
        data_buffer_content_size += reserved_data_buffer.size();
        reserved_data_buffer.clear();
      }
    }

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
        auto receive_buffer = boost::asio::buffer(udp_packet_buffer);
        udp_packet_buffer_size = socket.receive_from(receive_buffer, ep2);
        const size_t data_len = udp_packet_buffer_size - counter_bytes_count;

        // check counter
        udp_packet_counter_type received_counter = 0;
// ref: https://stackoverflow.com/questions/12876361/reading-bytes-in-c
// in this way, endian problem should be solved, ... maybe.
#pragma unroll
        for (size_t i = size_t(0); i < counter_bytes_count; ++i) {
          received_counter |=
              (static_cast<udp_packet_counter_type>(udp_packet_buffer[i])
               << (srtb::BITS_PER_BYTE * i));
        }
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
        assert(zeros_need_to_be_filled == 0);
        zeros_need_to_be_filled += data_len * lost_packets_count;
        fill_zero();
        last_counter = received_counter;

        udp_packet_buffer_pos = counter_bytes_count;
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

    // copy end of data buffer to reserved data for next round
    if (reserved_length < required_length) {
      reserved_data_buffer.resize(reserved_length);
      std::copy_n(data_buffer_ptr + data_buffer_capacity - reserved_length,
                  reserved_length, reserved_data_buffer.begin());
    } else {
      SRTB_LOGW << " [udp receiver worker] "
                << "reserved_length = " << reserved_length
                << " >= required_length = " << required_length << ", "
                << "not reserving data" << srtb::endl;
    }

    auto time_after_receive = std::chrono::system_clock::now();
    auto receive_time = std::chrono::duration_cast<std::chrono::microseconds>(
                            time_after_receive - time_before_receive)
                            .count();
    SRTB_LOGD << " [udp receiver worker] "
              << "recevice time = " << receive_time << " us." << srtb::endl;

    return std::make_pair(data_buffer, first_counter);
  }
};

}  // namespace udp_receiver
}  // namespace io
}  // namespace srtb

#endif  //  __SRTB_IO_UDP_RECEIVER__
