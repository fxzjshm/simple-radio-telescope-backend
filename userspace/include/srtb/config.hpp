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
using real = double;

#ifdef SYCL_IMPLEMENTATION_ONEAPI
template <typename T>
using complex = sycl::ext::oneapi::experimental::complex<T>;
#else
template <typename T>
using complex = std::complex<T>;
#endif  // SYCL_IMPLEMENTATION_ONEAPI

/**
 * @brief initial capacity of srtb::work_queue
 */
inline constexpr size_t work_queue_capacity = 4;

// TODO: check should use spsc_queue or shared queue with mutex here

#ifdef SRTB_WORK_QUEUE_FIXED_SIZE
template <typename T>
using work_queue = boost::lockfree::spsc_queue<
    T, boost::lockfree::capacity<srtb::work_queue_capacity> >;
#else
template <typename T>
class work_queue : public boost::lockfree::spsc_queue<T> {
 public:
  using super_class = boost::lockfree::spsc_queue<T>;
  work_queue() : super_class{work_queue_capacity} {}
};
#endif

typedef uint64_t udp_packet_counter_type;

// FFT default window in srtb/fft/fft_window.hpp

// TODO: is this necessary or too large?
inline constexpr size_t MEMORY_ALIGNMENT = 64ul;

inline constexpr size_t BITS_PER_BYTE = 8ul;

inline constexpr size_t UDP_MAX_SIZE = 1 << 16;

inline constexpr size_t LOG_PREFIX_BUFFER_LENGTH = 64ul;

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
   * TODO: baseband_bandwidth and baseband_sample_rate which to use?
   */
  srtb::real baseband_bandwidth = 500.0;

  /**
   * @brief Baseband sample rate, in samples / second
   */
  srtb::real baseband_sample_rate = 1000 * 1e6;

  /**
   * @brief Target dispersion measurement for coherent dedispersion
   * TODO: DM search list for unknown source
   */
  srtb::real dm = 0;

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
   * @brief path to the binary file to be read as baseband input
   */
  std::string input_file_path = "";

  /**
   * @brief skip some data before reading in, usually avoids header
   */
  size_t input_file_offset_bytes = 0;

  /**
    * @brief debug level for log
    * @see srtb::log::levels
    */
  /* srtb::log::levels */ int log_level = /* srtb::log::levels::DEBUG */ 4;

  /**
   * @brief location to save fftw wisdom
   * @note TODO: change to char* if pure C ABI is needed.
   */
  std::string fft_fftw_wisdom_path = "srtb_fftw_wisdom.txt";

  /**
   * @brief location to save fftwf (float32) wisdom
   * @note TODO: change to char* if pure C ABI is needed.
   */
  std::string fft_fftwf_wisdom_path = "srtb_fftwf_wisdom.txt";

  /**
   * @brief temporary thereshold for RFI mitigation. Channels with signal stronger
   *        than this thereshold * average strength will be set to 0
   */
  srtb::real mitigate_rfi_thereshold = 2;

  /**
   * @brief sum some spectrum before drawing, to reduce CPU side pressure
   * TODO: re-implement
   */
  size_t spectrum_sum_count = 1;

  /**
   * @brief channel count / batch size when performing inverse FFT, also M in frequency domain filterbank
   * @note now set to 1 so frequency domain filterbank is not used.
   */
  size_t ifft_channel_count = 1;

  size_t refft_length = 1 << 12;

  /**
   * @brief Wait time in naneseconds for a thread to sleep if it fails to get work now.
   *        Trade off between CPU usage (most are wasted) and pipeline latency.
   */
  size_t thread_query_work_wait_time = 1000;
};

}  // namespace srtb

#endif  // __SRTB_CONFIG__
