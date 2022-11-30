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
#include <thread>

#include "srtb/coherent_dedispersion.hpp"
#include "srtb/commons.hpp"

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
   */
  std::array<std::byte, UDP_MAX_SIZE> udp_packet_buffer;
  /**
   * @brief buffer for data storage, i.e. without counter 
   */
  boost::asio::streambuf data_buffer;
  static constexpr udp_packet_counter_type last_counter_initial_value =
      static_cast<udp_packet_counter_type>(-1);
  /** 
   * @brief counter received from last packet
   */
  udp_packet_counter_type last_counter = last_counter_initial_value;

 public:
  udp_receiver_worker(const std::string& sender_address,
                      const unsigned short sender_port,
                      const int udp_receiver_buffer_size)
      : sender_endpoint{boost::asio::ip::address::from_string(sender_address),
                        sender_port},
        socket{io_service, sender_endpoint} {
    socket.set_option(boost::asio::ip::udp::socket::reuse_address{true});
    socket.set_option(boost::asio::socket_base::receive_buffer_size{
        udp_receiver_buffer_size});
  }

  /**
   * @brief Receive given number of bytes, with counter continuity already checked
   * @param required_length length in bytes
   * @return buffer of received data.
   * @note extra data received last time is preserved in @c udp_streambuf
   */
  auto receive(size_t required_length) {
    auto time_before_receive = std::chrono::system_clock::now();
    size_t total_lost_packets_count = 0;

    // wait until work size reached
    // TODO: unpack UDP data, check counter
    while (data_buffer.size() < required_length) {
      auto receive_buffer = boost::asio::buffer(udp_packet_buffer);
      const size_t received_len = socket.receive_from(receive_buffer, ep2);
      const size_t data_len = received_len - counter_bytes_count;

      // check counter
      udp_packet_counter_type received_counter = 0;
      // ref: https://stackoverflow.com/questions/12876361/reading-bytes-in-c
      // in this way, endian problem should be solved, ... maybe.
      for (size_t i = size_t(0); i < counter_bytes_count; ++i) {
        received_counter |=
            (static_cast<udp_packet_counter_type>(udp_packet_buffer[i])
             << (srtb::BITS_PER_BYTE * i));
      }
      const auto lost_packets_count = received_counter - last_counter - 1;
      if (lost_packets_count != 0 &&
          last_counter !=
              last_counter_initial_value /* first counter may not be 0 */) {
        total_lost_packets_count += lost_packets_count;
        const auto fill_count = data_len * lost_packets_count;
        auto dest_buffer = data_buffer.prepare(fill_count);
        auto ptr = reinterpret_cast<std::byte*>(dest_buffer.data());
        std::fill(ptr, ptr + fill_count, std::byte{0});
        data_buffer.commit(fill_count);
      }
      last_counter = received_counter;

      auto dest_buffer = data_buffer.prepare(data_len);
      std::memcpy(dest_buffer.data(),
                  reinterpret_cast<std::byte*>(receive_buffer.data()) +
                      counter_bytes_count,
                  data_len);
      data_buffer.commit(data_len);
    }
    if (total_lost_packets_count > 0) {
      SRTB_LOGW << " [udp receiver worker] "
                << "data loss detected: " << total_lost_packets_count
                << " packets in total. Filling these with zero. " << srtb::endl;
    }

    auto time_after_receive = std::chrono::system_clock::now();
    auto receive_time = std::chrono::duration_cast<std::chrono::microseconds>(
                            time_after_receive - time_before_receive)
                            .count();
    SRTB_LOGD << " [udp receiver worker] "
              << "recevice time = " << receive_time << " us." << srtb::endl;

    return data_buffer.data();
  }

  void consume(std::size_t n) { data_buffer.consume(n); }
};

}  // namespace udp_receiver
}  // namespace io
}  // namespace srtb

#endif  //  __SRTB_IO_UDP_RECEIVER__
