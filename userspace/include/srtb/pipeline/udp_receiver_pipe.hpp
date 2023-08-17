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

#include <optional>

#include "srtb/coherent_dedispersion.hpp"
#include "srtb/io/udp_receiver.hpp"
#include "srtb/pipeline/framework/pipe.hpp"
#include "srtb/thread_affinity.hpp"

namespace srtb {
namespace pipeline {

/**
 * @brief receive UDP data and transfer to unpack_work_queue.
 * @see @c srtb::io::udp_receiver::udp_receiver_worker
 * TODO: separate reserving samples for coherent dedispersion
 */
class udp_receiver_pipe {
 protected:
  sycl::queue q;
  std::optional<srtb::io::udp_receiver::udp_receiver_worker> opt_worker;
  /**
   * @brief identifier for different pipe instances
   * 
   * e.g. id == 0 and 1 for two polarizations
   */
  size_t id;

 public:
  // init of worker is deferred because this pipe may not be used,
  // and failure of binding address will result in a error
  udp_receiver_pipe(sycl::queue q_, size_t id_ = 0) : q{q_}, id{id_} {
    // wait until other pipes have set up
    while (srtb::pipeline::running_pipe_count !=
           srtb::pipeline::expected_running_pipe_count -
               srtb::pipeline::expected_input_pipe_count) {
      std::this_thread::sleep_for(
          std::chrono::nanoseconds(srtb::config.thread_query_work_wait_time));
    }

    std::string sender_address;
    {
      const auto& sender_addresses = srtb::config.udp_receiver_sender_address;
      if (sender_addresses.size() == 1) {
        sender_address = sender_addresses.at(0);
      } else if (id < sender_addresses.size()) {
        sender_address = sender_addresses.at(id);
      } else {
        SRTB_LOGE << " [udp receiver pipe] "
                  << "id = " << id << ": "
                  << "no UDP sender address set" << srtb::endl;
        throw std::runtime_error{
            " [udp receiver pipe] no UDP sender address for id = " +
            std::to_string(id)};
      }
    }
    unsigned short sender_port;
    {
      const auto& sender_ports = srtb::config.udp_receiver_sender_port;
      if (sender_ports.size() == 1) {
        sender_port = sender_ports.at(0);
      } else if (id < sender_ports.size()) {
        sender_port = sender_ports.at(id);
      } else {
        SRTB_LOGE << " [udp receiver pipe] "
                  << "id = " << id << ": "
                  << "no UDP sender port set" << srtb::endl;
        throw std::runtime_error{
            " [udp receiver pipe] no UDP sender port for id = " +
            std::to_string(id)};
      }
    }
    opt_worker.emplace(sender_address, sender_port);

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
              << "start reading, address = " << sender_address << ", "
              << "port = " << sender_port << srtb::endl;
  }

  auto operator()([[maybe_unused]] std::stop_token stop_token,
                  srtb::work::dummy_work) {
    auto& worker = opt_worker.value();

    // this config should persist during one work push
    const size_t baseband_input_count = srtb::config.baseband_input_count;
    const int baseband_input_bits = srtb::config.baseband_input_bits;
    const size_t baseband_input_bytes = baseband_input_count *
                                        std::abs(baseband_input_bits) /
                                        srtb::BITS_PER_BYTE;

    // reserved some samples for next round
    size_t nsamps_reserved = srtb::codd::nsamps_reserved();

    if (nsamps_reserved < baseband_input_count) {
      SRTB_LOGD << " [udp receiver pipe] "
                << "id = " << id << ": "
                << "reserved " << nsamps_reserved << " samples" << srtb::endl;
    } else {
      SRTB_LOGW << " [udp receiver pipe] "
                << "id = " << id << ": "
                << "baseband_input_count = " << baseband_input_count
                << " >= nsamps_reserved = " << nsamps_reserved << srtb::endl;
      nsamps_reserved = 0;
    }

    SRTB_LOGD << " [udp receiver pipe] "
              << "id = " << id << ": "
              << "start receiving" << srtb::endl;
    auto [h_ptr, first_counter] =
        worker.receive(/* required_length = */ baseband_input_bytes,
                       /* reserved_length = */ nsamps_reserved);
    SRTB_LOGD << " [udp receiver pipe] "
              << "id = " << id << ": "
              << "receive finished" << srtb::endl;

    auto time_before_push = std::chrono::system_clock::now();

    const uint64_t timestamp =
        std::chrono::system_clock::now().time_since_epoch().count();

    srtb::work::baseband_data_holder baseband_data{h_ptr, baseband_input_bytes};
    //if ((srtb::config.baseband_write_all && (!srtb::config.gui_enable))) {
    //// no need to process data if just recording baseband data & not showing spectrum
    //srtb::work::baseband_output_work baseband_output_work;
    //baseband_output_work.ptr = nullptr;
    //baseband_output_work.count = 0;
    //baseband_output_work.batch_size = 0;
    //baseband_output_work.baseband_data = std::move(baseband_data);
    //baseband_output_work.timestamp = timestamp;
    //baseband_output_work.udp_packet_counter = first_counter;
    //SRTB_PUSH_WORK_OR_RETURN(" [udp receiver pipe] ",
    //                         srtb::baseband_output_queue,
    //                         baseband_output_work, stop_token);
    //} else {
    //  // flush input of baseband_input_bytes
    //  std::shared_ptr<std::byte> d_ptr =
    //      srtb::device_allocator.allocate_shared<std::byte>(
    //          baseband_input_bytes);
    //  sycl::event host_to_devive_copy_event =
    //      q.copy(h_ptr.get(), /* -> */ d_ptr.get(), baseband_input_bytes);

    //  srtb::work::unpack_work unpack_work;
    //  unpack_work.ptr = d_ptr;
    //  unpack_work.count = baseband_input_bytes;
    //  unpack_work.baseband_data = std::move(baseband_data);
    //  unpack_work.baseband_input_bits = baseband_input_bits;
    //  unpack_work.timestamp = timestamp;
    //  unpack_work.udp_packet_counter = first_counter;
    //  unpack_work.copy_event = host_to_devive_copy_event;
    //  SRTB_PUSH_WORK_OR_RETURN(" [udp receiver pipe] ", srtb::unpack_queue,
    //                           unpack_work, stop_token);
    //}

    srtb::work::copy_to_device_work copy_to_device_work;
    copy_to_device_work.ptr = nullptr;
    copy_to_device_work.count = 0;
    copy_to_device_work.batch_size = 0;
    copy_to_device_work.baseband_data = std::move(baseband_data);
    copy_to_device_work.timestamp = timestamp;
    copy_to_device_work.udp_packet_counter = first_counter;

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

}  // namespace pipeline
}  // namespace srtb

#endif  // __SRTB_PIPELINE_UDP_RECEIVER_PIPE__
