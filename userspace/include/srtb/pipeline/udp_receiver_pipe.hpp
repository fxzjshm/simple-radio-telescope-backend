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

#ifndef __SRTB_PIPELINE_UDP_RECEIVER_PIPE__
#define __SRTB_PIPELINE_UDP_RECEIVER_PIPE__

#include <memory>
#include <optional>

#include "srtb/global_variables.hpp"
#include "srtb/io/backend_registry.hpp"
#include "srtb/io/udp/asio_udp_packet_provider.hpp"
#include "srtb/io/udp/packet_mmap_v3_provider.hpp"
#include "srtb/io/udp/recvfrom_packet_provider.hpp"
#include "srtb/io/udp/recvmmsg_packet_provider.hpp"
#include "srtb/io/udp/udp_receiver.hpp"
#include "srtb/memory/mem.hpp"
#include "srtb/pipeline/framework/pipe.hpp"
#include "srtb/util/thread_affinity.hpp"

namespace srtb {
namespace pipeline {

/**
 * @brief receive UDP data and transfer to unpack_work_queue.
 * @see @c srtb::io::udp::udp_receiver_worker
 * TODO: add reserving samples for coherent dedispersion
 */
template <typename UDPReceiverWorker>
class udp_receiver_pipe {
 protected:
  sycl::queue q;
  std::optional<UDPReceiverWorker> opt_worker;
  /**
   * @brief identifier for different pipe instances
   * 
   * e.g. id == 0 and 1 for two polarizations, or 0-3 for 4 antennas.
   */
  size_t id;

 public:
  // init of worker is deferred because this pipe may not be used,
  // and failure of binding address will result in a error
  udp_receiver_pipe(sycl::queue q_, size_t id_ = 0) : q{q_}, id{id_} {
    std::string address;
    {
      const auto& addresses = srtb::config.udp_receiver_address;
      if (addresses.size() == 1) {
        // for argument broadcasting to all receivers
        address = addresses.at(0);
      } else if (id < addresses.size()) {
        address = addresses.at(id);
      } else {
        SRTB_LOGE << " [udp receiver pipe] "
                  << "id = " << id << ": "
                  << "no UDP address set" << srtb::endl;
        throw std::runtime_error{" [udp receiver pipe] no UDP address for id = " + std::to_string(id)};
      }
    }
    unsigned short port;
    {
      const auto& ports = srtb::config.udp_receiver_port;
      if (ports.size() == 1) {
        // for argument broadcasting to all receivers
        port = ports.at(0);
      } else if (id < ports.size()) {
        port = ports.at(id);
      } else {
        SRTB_LOGE << " [udp receiver pipe] "
                  << "id = " << id << ": "
                  << "no UDP port set" << srtb::endl;
        throw std::runtime_error{" [udp receiver pipe] no UDP port for id = " + std::to_string(id)};
      }
    }
    opt_worker.emplace(address, port);

    const auto& cpus_preferred = srtb::config.udp_receiver_cpu_preferred;
    if (0 <= id && id < cpus_preferred.size()) {
      const auto cpu_preferred = cpus_preferred.at(id);
      srtb::thread_affinity::set_thread_affinity(cpu_preferred);
    } else {
      // no preferred CPU is set, so not setting thread affinity
      SRTB_LOGW << " [udp receiver pipe] "
                << "id = " << id << ": "
                << "CPU affinity not set, performance may be degraded"
                << srtb::endl;
    }

    SRTB_LOGI << " [udp receiver pipe] "
              << "id = " << id << ": "
              << "start reading, address = " << address << ", "
              << "port = " << port << srtb::endl;
  }

  auto operator()([[maybe_unused]] std::stop_token stop_token,
                  srtb::work::dummy_work) {
    auto& worker = opt_worker.value();

    // this config should persist during one work push
    // count is per polarization
    const size_t baseband_input_count = srtb::config.baseband_input_count;
    const int baseband_input_bits = srtb::config.baseband_input_bits;
    const size_t baseband_input_bytes = baseband_input_count *
                                        std::abs(baseband_input_bits) /
                                        srtb::BITS_PER_BYTE;
    const size_t data_stream_count = UDPReceiverWorker::backend_t::data_stream_count;

    const size_t required_length = baseband_input_bytes * data_stream_count;
    auto h_mem = srtb::mem_allocate_shared<std::byte>(&srtb::host_allocator, required_length);

    SRTB_LOGD << " [udp receiver pipe] "
              << "id = " << id << ": "
              << "start receiving" << srtb::endl;
    auto first_counter = worker.receive(h_mem.get_span());
    SRTB_LOGD << " [udp receiver pipe] "
              << "id = " << id << ": "
              << "receive finished" << srtb::endl;

    auto time_before_push = std::chrono::system_clock::now();

    const uint64_t timestamp =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count();

    srtb::work::baseband_data_holder baseband_data{h_mem.ptr, required_length};
    srtb::work::copy_to_device_work copy_to_device_work;
    copy_to_device_work.ptr = nullptr;
    copy_to_device_work.count = 0;
    copy_to_device_work.batch_size = 0;
    copy_to_device_work.baseband_data = std::move(baseband_data);
    copy_to_device_work.timestamp = timestamp;
    copy_to_device_work.udp_packet_counter = first_counter;
    copy_to_device_work.data_stream_id = id;

    auto time_after_push = std::chrono::system_clock::now();
    auto push_work_time = std::chrono::duration_cast<std::chrono::microseconds>(
                              time_after_push - time_before_push)
                              .count();
    SRTB_LOGD << " [udp receiver pipe] "
              << "id = " << id << ": "
              << "push work time = " << push_work_time << " us" << srtb::endl;
    return std::optional{copy_to_device_work};
  }
};

template <typename... Args>
inline auto start_udp_receiver_pipe(std::string_view backend_name,
                                    Args... args) {
  using namespace srtb::io::backend_registry;
  using provider_t = srtb::io::udp::recvmmsg_packet_provider;

  if (backend_name == naocpsr_roach2::name) {
    using backend_t = naocpsr_roach2;
    using worker_t = srtb::io::udp::udp_receive_block_worker<provider_t, backend_t>;
    using pipe_t = udp_receiver_pipe<worker_t>;
    return start_pipe<pipe_t>(args...);
  }
  if (backend_name == naocpsr_snap1::name) {
    using backend_t = naocpsr_snap1;
    using worker_t = srtb::io::udp::udp_receive_block_worker<provider_t, backend_t>;
    using pipe_t = udp_receiver_pipe<worker_t>;
    return start_pipe<pipe_t>(args...);
  }
  if (backend_name == gznupsr_a1::name) {
    using backend_t = gznupsr_a1;
    using worker_t = srtb::io::udp::udp_receive_block_worker<provider_t, backend_t>;
    using pipe_t = udp_receiver_pipe<worker_t>;
    return start_pipe<pipe_t>(args...);
  }
  throw std::invalid_argument{
      "[start_udp_receiver_pipe] Unknown backend name: " +
      std::string{backend_name}};
}

}  // namespace pipeline
}  // namespace srtb

#endif  // __SRTB_PIPELINE_UDP_RECEIVER_PIPE__
