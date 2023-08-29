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
#include <limits>
#include <string>
#include <string_view>
#include <vector>

#include "srtb/math.hpp"

/**
 * @brief Shortcut for namespace of complex type.
 * @see srtb::complex
 */
namespace akira = _SYCL_CPLX_NAMESPACE;

namespace srtb {

// ------ Compile time configuration ------

// maybe double on CPU,
//       float on most GPUs, even most professional cards of SOME VENDOR.
using real = float;

/**
 * @brief In memory of Akira Complex, "My Guiding Star".
 */
template <typename T>
using complex = akira::complex<T>;

// option to use managed/unified memory as device memory
//#define SRTB_USE_USM_SHARED_MEMORY

// option to share work area between FFT plans
// may reduce VRAM usage but increase latency
#define SRTB_FFT_SHARE_WORK_AREA

/**
 * @brief initial capacity of srtb::work_queue
 */
inline constexpr size_t work_queue_capacity = 2;

// option to use fix the max size of work_queue or not
inline constexpr bool work_queue_fixed_size = true;

using udp_packet_counter_type = uint64_t;

// FFT default window in srtb/fft/fft_window.hpp
inline constexpr bool fft_window_precompute = false;
inline constexpr bool fft_operate_in_place = true;

/** @brief true: use sycl::reduction; false: use reduction from SYCL Parallel STL */
inline constexpr bool use_sycl_reduction = false;

/** @brief option to emulate fp64 using fp32 */
inline constexpr bool use_emulated_fp64 = false;

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
   *        negative value is signed integers
   *        TODO: 32 -> uint32 or float?
   */
  int baseband_input_bits = 8;

  /**
   * @brief Type of baseband format: 
   *          simple (just stream of samples from 1 source), 
   *          interleaved_samples_2 (interleaved 2 stream, each sample from one stream)
   */
  std::string baseband_format_type = "simple";

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
   * @brief if 1, baseband data affected by dispersion will be reserved for next segment, 
   *              i.e. segments will overlap, if possible; 
   *        if 0, baseband data will not overlap.
   */
  bool baseband_reserve_sample = true;

  /**
   * @brief Target dispersion measurement for coherent dedispersion
   * TODO: DM search list for unknown source
   */
  srtb::real dm = 0;

  /**
   * @brief Address to receive baseband UDP packets.
   */
  std::vector<std::string> udp_receiver_sender_address = {"10.0.1.2"};

  /**
   * @brief Port to receive baseband UDP packets.
   */
  std::vector<unsigned short> udp_receiver_sender_port = {12004};

  /**
   * @brief CPU core that UDP receiver should be bound to.
   */
  std::vector<unsigned int> udp_receiver_cpu_preferred = {0};

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
   * @brief if true, record all baseband into one file per polarization;
   *        if false, write only those with signal detected.
   */
  bool baseband_write_all = false;

  /**
    * @brief Debug level for console log output.
    * @see srtb::log::levels
    */
  /* srtb::log::levels */ int log_level = /* srtb::log::levels::INFO */ 3;

  /**
   * @brief Location to save fftw wisdom.
   * @note TODO: change to char* if pure C ABI is needed.
   */
  std::string fft_fftw_wisdom_path = "srtb_fftw_wisdom.txt";

  /**
   * @brief Temporary threshold for RFI mitigation. Frequency channels with 
   *        signal stronger than (this threshold * average strength) will be set to 0
   */
  srtb::real mitigate_rfi_average_method_threshold = 10;

  /**
   * @brief Frequency channels with spectral kurtosis larger than this threshold will be set to 0
   */
  srtb::real mitigate_rfi_spectral_kurtosis_threshold = 1.1;

  /**
   * @brief list of frequency pairs to zap/remove,
   *        format: 11-12, 15-90, 233-235, 1176-1177
   *                (arbitrary values)
   */
  std::string mitigate_rfi_freq_list = "";

  /**
   * @brief sum some spectrum before drawing, to reduce CPU side pressure
   * TODO: re-implement
   */
  size_t spectrum_sum_count = 1;

  /**
   * @brief Count of channels (complex numbers) in spectrum waterfall.
   */
  size_t spectrum_channel_count = 1 << 15;

  /**
   * @brief signal noise ratio threshold for signal detect,
   *        If $ \exist x_i $ s.t. $$ x_i > \mu + k * \sigma $$,
   *        where $ x_i $ is a value of time series,
   *              $ \mu $ is its mean value, $ \sigma $ is variance,
   *              $ k $ is this threshold
   *        then it is thought a signal.
   */
  srtb::real signal_detect_signal_noise_threshold = 6;

  /**
   * @brief threshold of ratio of non-zapped channels
   * 
   * if too many channels are zapped, result is often not correct
   */
  srtb::real signal_detect_channel_threshold = 0.9;

  /**
   * @brief max boxcar length for signal detect.
   */
  size_t signal_detect_max_boxcar_length = 1024;

  /**
   * @brief Wait time in naneseconds for a thread to sleep if it fails to get work now.
   *        Trade off between CPU usage (most are wasted) and pipeline latency.
   */
  size_t thread_query_work_wait_time = 1000;

  /**
   * @brief Runtime configuration to enable GUI
   */
  bool gui_enable = SRTB_ENABLE_GUI;

  /**
   * @brief Width of GUI spectrum pixmap
   */
  size_t gui_pixmap_width = 1920;

  /**
   * @brief Height of GUI spectrum pixmap
   */
  size_t gui_pixmap_height = 1080;
};

}  // namespace srtb

#endif  // __SRTB_CONFIG__
