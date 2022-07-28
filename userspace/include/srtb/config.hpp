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

#include "srtb/sycl.hpp"

#ifdef SYCL_IMPLEMENTATION_ONEAPI
#ifndef SYCL_EXT_ONEAPI_COMPLEX
#define SYCL_EXT_ONEAPI_COMPLEX
#endif  // SYCL_EXT_ONEAPI_COMPLEX
#include <sycl/ext/oneapi/experimental/sycl_complex.hpp>
#else
#include <complex>
#endif  // SYCL_IMPLEMENTATION_ONEAPI

namespace srtb {

// ------ Compile time configuration ------

// TODO: maybe float on GPU?
typedef double real;
#ifdef SYCL_IMPLEMENTATION_ONEAPI
template <typename T>
using complex = sycl::ext::oneapi::experimental::complex<T>;
#else
template <typename T>
using complex = std::complex<T>;
#endif  // SYCL_IMPLEMENTATION_ONEAPI

// TODO: check should use queue or spsc_queue here
template <typename... Args>
using work_queue = boost::lockfree::spsc_queue<Args...>;

typedef uint64_t udp_packet_counter_type;

// TODO: is this necessary or too large?
inline constexpr size_t MEMORY_ALIGNMENT = 64ul;

inline constexpr size_t BITS_PER_BYTE = 8ul;

inline constexpr size_t UDP_MAX_SIZE = 1 << 16;

/**
 * @brief initial capacity of boost::lockfree::{queue, spsc_queue}
 */
inline constexpr size_t work_queue_initial_capacity = 64;

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
   * @brief Length of a single input data, used in unpack.
   *        TODO: 32 -> uint32 or float?
   */
  size_t baseband_input_bits = 8;

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
    * @see srtb::log::levels
    */
  /* srtb::log::levels */ int log_level = /* srtb::log::levels::DEBUG */ 4;

  inline size_t unpacked_input_count() {
    return baseband_input_length * BITS_PER_BYTE / baseband_input_bits;
  }

  std::string fft_fftw_wisdom_path = "srtb_fftw_wisdom.txt";
};

}  // namespace srtb

#endif  // __SRTB_CONFIG__
