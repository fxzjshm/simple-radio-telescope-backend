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
#include <boost/asio/buffer.hpp>
#include <boost/asio/io_service.hpp>
#include <boost/asio/ip/udp.hpp>
#include <boost/lockfree/spsc_queue.hpp>
#include <chrono>
#include <cstring>
#include <deque>
#include <iostream>
#include <limits>
#include <span>
#include <thread>
#include <vector>

#include "srtb/commons.hpp"
#include "srtb/thread_affinity.hpp"
#include "srtb/memory/dual_port_object_pool.hpp"

namespace srtb {
namespace io {
namespace udp {

template <typename T>
using packet_cache_t = srtb::memory::dual_port_object_pool<T>;

using packet_container_t = std::array<std::byte, UDP_MAX_SIZE>;
struct alignas(srtb::MEMORY_ALIGNMENT) packet_t {
  packet_container_t packet_countainer;
  size_t bytes_received;
};

/**
 * @brief A source that receives UDP packet and unpack it.
 * 
 * TODO: maybe use lock free ring buffer, see https://ferrous-systems.com/blog/lock-free-ring-buffer/
 *       or move ring buffer to GPU side, when RDMA method is considered.
 * 
 * TODO: simultaneously receive 2 polars (i.e. 2 addresses) and *sync*
 * 
 * @see @c srtb::pipeline::udp_receiver_pipe
 */
template <typename PacketProvider, typename PacketParser>
class udp_receiver_worker {
 protected:
  // TODO: this seems arbitary
  size_t initial_queue_size =
      static_cast<size_t>(std::numeric_limits<int>::max()) /
          PacketParser::data_size +
      1;
  packet_cache_t<packet_t> packet_cache;
  std::jthread packet_provider_thread;

  /**
   * @brief buffer for receiving one UDP packet
   * @note not directly writing to output because packet counter is in the packet.
   */
  packet_t* udp_packet_buffer = nullptr;
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

  std::vector<std::byte> reserved_data_buffer;

  bool can_restart;

  PacketProvider packet_provider;
  PacketParser packet_parser;

 public:
  udp_receiver_worker(const std::string& sender_address,
                      const unsigned short sender_port, bool can_restart_,
                      int32_t cpu_preferred)
      : packet_cache{initial_queue_size},
        can_restart{can_restart_},
        packet_provider{sender_address, sender_port} {
    srtb::thread_affinity::set_thread_affinity(cpu_preferred - 2);

    packet_provider_thread =
        std::jthread{[=, this](std::stop_token stop_token) {
          srtb::thread_affinity::set_thread_affinity(cpu_preferred);
          for (size_t i = 0; i < 2; i++) {
            do_async_receive();
          }
          packet_provider.run_eventloop();
        }};
  }

 protected:
  void do_async_receive() {
    packet_t* h_packet = packet_cache.get_or_allocate_free();
    packet_provider.receive_async(
        std::span<std::byte>{h_packet->packet_countainer},
        /* callback = */ [=, this](size_t read_bytes) {
          h_packet->bytes_received = read_bytes;
          packet_cache.push_received(h_packet);
          do_async_receive();
        });
  }

 public:
  ~udp_receiver_worker() { packet_provider_thread = std::jthread{}; }

  /**
   * @brief Receive given number of bytes, with counter continuity already checked
   * @param required_length length in bytes
   * @param reserved_length length of data reserved for next round
   * @return buffer of received data
   */
  auto receive(size_t required_length, size_t reserved_length) {
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
    if (can_restart && zeros_need_to_be_filled > data_buffer_capacity)
        [[unlikely]] {
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
        std::copy_n(udp_packet_buffer->packet_countainer.begin() +
                        udp_packet_buffer_pos,
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
        if (udp_packet_buffer) {
          packet_cache.put_free(udp_packet_buffer);
          udp_packet_buffer = nullptr;
        }
        // get received packet
        udp_packet_buffer = packet_cache.pop_received();
        udp_packet_buffer_size = udp_packet_buffer->bytes_received;
        const auto [header_size, received_counter_, timestamp] =
            packet_parser.parse(std::span<std::byte>{
                udp_packet_buffer->packet_countainer.begin(),
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
        assert(zeros_need_to_be_filled == 0);
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

    return std::make_pair(data_buffer, first_counter);
  }
};

}  // namespace udp
}  // namespace io
}  // namespace srtb

#endif  //  __SRTB_IO_UDP_RECEIVER__
