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

#include <cstddef>  // for size_t
#include <string>
#include <string_view>

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

// option to use managed/unified memory as device memory
#define SRTB_USE_USM

/**
 * @brief initial capacity of srtb::work_queue
 */
inline constexpr size_t work_queue_capacity = 4;

// option to use fix the max size of work_queue or not
inline constexpr bool work_queue_fixed_size = false;

using udp_packet_counter_type = uint64_t;

// FFT default window in srtb/fft/fft_window.hpp
inline constexpr bool fft_window_precompute = false;
inline constexpr bool fft_operate_in_place = true;

inline constexpr bool write_all_baseband = false;

// TODO: is this necessary or too large?
inline constexpr size_t MEMORY_ALIGNMENT = 64ul;

inline constexpr size_t BITS_PER_BYTE = 8ul;

inline constexpr size_t UDP_MAX_SIZE = 1 << 16;

inline constexpr size_t LOG_PREFIX_BUFFER_LENGTH = 64ul;

// ------ Runtime configuration ------

/**
 * @brief Runtime configuration.
 * @note module specific config names should prepend module name
 * @note this struct is named configs so that srtb::config is a variable
 * @note remember to add program options parser in srtb/program_options.hpp
 *       if an option is added here.
 * @see srtb::config in srtb/global_variables.hpp
 */
struct configs {
  /**
   * @brief Path to config file to be used to read other configs.
   */
  std::string config_file_name = "srtb_config.cfg";

  /**
   * @brief Count of data to be transferred to GPU for once processing, in sample counts.
   *        Should be power of 2 so that FFT and channelizing can work properly.
   */
  size_t baseband_input_count = 1 << 28;

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
  int udp_receiver_buffer_size = static_cast<int>((1L << 31) - 1);

  /**
   * @brief Address to receive baseband UDP packets.
   */
  std::string udp_receiver_sender_address = "10.0.1.2";

  /**
   * @brief Port to receive baseband UDP packets.
   */
  unsigned short udp_receiver_sender_port = 12004;

  /**
   * @brief Path to the binary file to be read as baseband input.
   */
  std::string input_file_path = "";

  /**
   * @brief Skip some data before reading in, usually avoids header.
   */
  size_t input_file_offset_bytes = 0;

  /**
   * @brief Prefix of saved baseband data. Full name will be ${prefix}${counter}.bin
   */
  std::string baseband_output_file_prefix = "srtb_baseband_output_";

  /**
    * @brief Debug level for console log output.
    * @see srtb::log::levels
    */
  /* srtb::log::levels */ int log_level = /* srtb::log::levels::DEBUG */ 4;

  /**
   * @brief Location to save fftw wisdom.
   * @note TODO: change to char* if pure C ABI is needed.
   */
  std::string fft_fftw_wisdom_path = "srtb_fftw_wisdom.txt";

  /**
   * @brief Temporary thereshold for RFI mitigation. Channels with signal stronger
   *        than this thereshold * average strength will be set to 0
   */
  srtb::real mitigate_rfi_thereshold = 10;

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

  /**
   * @brief Length of FFT for re-constructing signals after coherent dedispersion,
   *        of complex numbers, so <= baseband_input_count / 2
   */
  size_t refft_length = 1 << 15;

  /**
   * @brief threshold for signal detect.
   *        If $ \exist x_i $ s.t. $$ x_i > \mu + k * \sigma $$,
   *        where $ x_i $ is a value of time series,
   *              $ \mu $ is its mean value, $ \sigma $ is variance,
   *              $ k $ is this threshold
   *        then it is thought a signal.
   */
  srtb::real signal_detect_threshold = 6;

  /**
   * @brief Wait time in naneseconds for a thread to sleep if it fails to get work now.
   *        Trade off between CPU usage (most are wasted) and pipeline latency.
   */
  size_t thread_query_work_wait_time = 1000;
};

}  // namespace srtb

#endif  // __SRTB_CONFIG__
