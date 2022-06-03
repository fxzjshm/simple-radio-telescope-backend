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
#ifndef __SRTB_CONFIG__
#define __SRTB_CONFIG__

#include <boost/lockfree/queue.hpp>
#include <boost/lockfree/spsc_queue.hpp>

namespace srtb {

// ------ Compile time configuration ------

// TODO: maybe float on GPU?
typedef double real;

// TODO: check should use queue or spsc_queue here
template <typename... Args>
using work_queue = boost::lockfree::spsc_queue<Args...>;

// TODO: is this necessary or too large?
constexpr size_t MEMORY_ALIGNMENT = 64ul;

/**
 * @brief initial capacity of boost::lockfree::{queue, spsc_queue}
 */
constexpr size_t work_queue_initial_capacity = 64;

// ------ Runtime configuration ------

/**
 * @brief Runtime configuration.
 * @note module specific config names should prepend module name
 * @note named configs so that srtb::config is a variable
 * @see srtb::config in srtb/global_variables.hpp
 */
struct configs {
  /**
   * @brief Length of data to be transferred to GPU for once processing, in bytes.
   *        Should be power of 2 so that FFT and channelizing can work properly.
   */
  size_t baseband_input_length = 1 << 25;

  /**
   * @brief Lowerest frequency of received baseband signal, in MHz.
   */
  srtb::real baseband_freq_low = 1000.0;

  /**
   * @brief Band width of received baseband signal, in MHz.
   * 
   */
  srtb::real baseband_bandwidth = 500.0;

  /**
   * @brief Baseband sample rate, in samples / second
   */
  srtb::real baseband_sample_rate = 400 * 1e6;

  /**
   * @brief Target dispersion measurement for coherent dedispersion
   * TODO: DM search list for unknown source
   */
  srtb::real dm = 375;

  /**
    * @brief Buffer size of socket for receving udp packet.
    * @see srtb::io::udp_receiver
    */
  int udp_receiver_buffer_size = 1 << 24;

  /**
   * @brief Address to receive baseband UDP packets
   */
  std::string udp_receiver_sender_address = "10.0.1.2";

  /**
   * @brief Port to receive baseband UDP packets
   */
  unsigned short udp_receiver_sender_port = 12004;

  /**
    * @brief debug level for log
    * @see srtb::log::debug_levels
    */
  int log_debug_level = 4;
};

}  // namespace srtb

#endif  // __SRTB_CONFIG__