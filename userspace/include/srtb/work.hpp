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
#ifndef __SRTB_WORK__
#define __SRTB_WORK__

#include <boost/lockfree/spsc_queue.hpp>
#include <memory>
#include <thread>
#include <vector>

#include "concurrentqueue.h"
#include "srtb/config.hpp"

namespace srtb {

// definition of work queue, a container of works to be processed.
// TODO: check should use spsc_queue or shared queue with mutex here
template <typename T, bool spsc = true,
          bool fixed_size = srtb::work_queue_fixed_size,
          size_t capacity = srtb::work_queue_capacity>
class work_queue;

template <typename T, size_t capacity>
class work_queue<T, true, true, capacity>
    : public boost::lockfree::spsc_queue<T,
                                         boost::lockfree::capacity<capacity> > {
 public:
  using work_type = T;
};

template <typename T, size_t initial_capacity>
class work_queue<T, true, false, initial_capacity>
    : public boost::lockfree::spsc_queue<T> {
 public:
  using work_type = T;
  using super_class = boost::lockfree::spsc_queue<T>;
  work_queue(size_t initial_capacity_ = initial_capacity)
      : super_class{initial_capacity_} {}
};

template <typename T, bool unused_1, size_t unused_2>
class work_queue<T, false, unused_1, unused_2>
    : public moodycamel::ConcurrentQueue<T> {
 public:
  using work_type = T;
  using super_class = moodycamel::ConcurrentQueue<T>;
  template <typename... Args>
  inline decltype(auto) push(Args&&... args) {
    return super_class::enqueue(std::forward<Args>(args)...);
  }

  template <typename... Args>
  inline decltype(auto) pop(Args&&... args) {
    return super_class::try_dequeue(std::forward<Args>(args)...);
  }

  inline size_t read_available() { return super_class::size_approx(); }

  inline bool empty() { return super_class::is_empty(); }
};

/**
 * @brief This namespace contains work types that defines the input of a pipe.
 *        Ideally all info needed to execute the pipeline should be written in a POD work class.
 * TODO: refactor work types
 */
namespace work {

/**
 * @brief Dummy work as place holder for work queue / pipe work transfer
 */
struct dummy_work {};

/**
 * @brief this holds ownership of original baseband data & its size
 */
struct baseband_data_holder {
  /** @brief contains baseband data without UDP packet counter in host memory. */
  std::shared_ptr<std::byte> baseband_ptr;
  size_t baseband_input_bytes;
};

/**
 * @brief This represents a work to be done and should be the same as `std::pair<T, size_t>`,
 *        created just because `std::pair` doesn't satisfy `boost::has_trivial_assign`,
 *        which is required for lockfree queue.
 * @tparam T Type of the pointer of the work, e.g. std::shared_ptr<std::byte> for unpack and std::shared_ptr<srtb::real> for r2c FFT.
 *         TODO: Maybe T = sycl::buffer<std::byte> if pointer isn't suitable for some backend in the future.
 */
template <typename T>
struct work {
  /**
   * @brief Pointer / Some pointer wrapper / sycl::buffer of data to be processed,
   *        refer to doc of speciallized work types for actual meaning of @c ptr;
   *        currently @c std::shared_ptr is used to reduce memory error.
   */
  T ptr;
  /**
   * @brief shape[0] of data in @c ptr, refer to specialized work types for detail
   * @see batch_size
   */
  size_t count;
  /**
   * @brief shape[1] of data in @c ptr, refer to specialized work types for detail
   * @see count
   */
  size_t batch_size;
  /**
   * @brief time stamp of these data, currectly 64-bit unix timestamp from server time
   */
  uint64_t timestamp;
  /**
   * @brief counter of correspond first new UDP packet, == no_udp_packet_counter means no value.
   */
  uint64_t udp_packet_counter;
  /**
   * @brief ID to identify which stream is this data belongs to
   */
  uint32_t data_stream_id;
  /**
   * @brief dummy value if no udp_packet_counter is available
   */
  static constexpr uint64_t no_udp_packet_counter = static_cast<uint64_t>(-1);
  /**
   * @brief original baseband input correspond to this work
   */
  baseband_data_holder baseband_data;

  template <typename U>
  inline void move_parameter_from(work<U>&& other) {
    timestamp = std::move(other.timestamp);
    udp_packet_counter = std::move(other.udp_packet_counter);
    data_stream_id = std::move(other.data_stream_id);
    baseband_data = std::move(other.baseband_data);
  }

  template <typename U>
  inline void copy_parameter_from(const work<U>& other) {
    timestamp = other.timestamp;
    udp_packet_counter = other.udp_packet_counter;
    data_stream_id = other.data_stream_id;
    baseband_data = other.baseband_data;
  }
};

/**
 * @brief copy input from host to device
 */
using copy_to_device_work = srtb::work::work<std::shared_ptr<std::byte> >;

/**
 * @brief contains a chunk of @c std::byte of size @c count, which is 
 *        baseband data and should be unpacked into @c srtb::real
 * @note count is count of std::byte, not of output time series,
 *       and should equal to `srtb::config.baseband_input_count * srtb::config.baseband_input_bits / srtb::BITS_PER_BYTE`
 */
using unpack_work = srtb::work::work<std::shared_ptr<std::byte> >;

/**
 * @brief contains a chunk of @c srtb::real that is to be FFT-ed.
 * @note real number of size @c n should be FFT-ed into @c n/2+1 @c srtb::complex<srtb::real> s,
 *       take care of memory allocation.
 */
using fft_1d_r2c_work = srtb::work::work<std::shared_ptr<srtb::real> >;

/**
 * @brief contains complex numbers to be FFT-ed
 */
using fft_1d_c2c_work =
    srtb::work::work<std::shared_ptr<srtb::complex<srtb::real> > >;

/**
 * @brief contains a block of @c srtb::complex<srtb::real> with radio interference
 *        to be cleared out
 * @note stage 1: whole data is frequency domain, i,e, batch_size = 1.
 */
using rfi_mitigation_s1_work =
    srtb::work::work<std::shared_ptr<srtb::complex<srtb::real> > >;

/**
 * @brief contains a piece of @c srtb::complex<srtb::real> to be coherently dedispersed
 */
using dedisperse_work =
    srtb::work::work<std::shared_ptr<srtb::complex<srtb::real> > >;

/**
 * @brief contains @c count of @c srtb::complex<srtb::real> in frequency domain
 *        to be inversed FFT-ed
 */
using ifft_1d_c2c_work =
    srtb::work::work<std::shared_ptr<srtb::complex<srtb::real> > >;

/**
 * @brief contains complex dedispersed baseband data of total length @c count
 *        to be reFFT-ed with length @c spectrum_channel_count to get spectrum with
 *        much higher time resolution.
 */
using refft_1d_c2c_work =
    srtb::work::work<std::shared_ptr<srtb::complex<srtb::real> > >;

/**
 * @brief contains complex dedispersed baseband data of total length @c count
 *        to be FFT-ed with length @c spectrum_channel_count to get spectrum with
 *        much higher time resolution.
 */
using watfft_1d_c2c_work =
    srtb::work::work<std::shared_ptr<srtb::complex<srtb::real> > >;

/**
 * @brief contains a block of @c srtb::complex<srtb::real> with radio interference
 *        to be cleared out
 * @note stage 2: work on dynamic spectrum, count: count in time axis, batch_size: count in frequency axis
 */
using rfi_mitigation_s2_work =
    srtb::work::work<std::shared_ptr<srtb::complex<srtb::real> > >;

/**
 * @brief contains @c srtb::complex<srtb::real> to be taken norm and summed into
 *        time series, then try tp find signal in it.
 */
using signal_detect_work =
    srtb::work::work<std::shared_ptr<srtb::complex<srtb::real> > >;

/**
 * @brief contains info for a time series of a spectrum
 */
struct time_series_holder {
  std::shared_ptr<srtb::real> h_time_series;
  std::shared_ptr<srtb::real> d_time_series;
  size_t time_series_length;
  /** @brief size of boxcar used to compute the time series; = 1 if not used */
  size_t boxcar_length;
  sycl::event transfer_event;
};

/**
 * @brief write baseband data to file without condition; @c ptr is reserved for future use
 */
using write_file_work =
    srtb::work::work<std::shared_ptr<srtb::complex<srtb::real> > >;

/**
 * @brief if there's signal, write baseband data to disk; @c ptr is correspond dynamic spectrum
 */
struct write_signal_work : public write_file_work {
  std::vector<time_series_holder> time_series;
};

/**
 * @brief contains @c srtb::complex<srtb::real> to be simplified into
 *        ~10^3 @c srtb::real to be displayed on GUI.
 *        Just like a software-defined-radio receiver.
 */
using simplify_spectrum_work =
    srtb::work::work<std::shared_ptr<srtb::complex<srtb::real> > >;

/**
 * @brief contains ~10^3 * @c batch_size of @c srtb::real to be summed and drawn
 *        to a line of a pixmap. @c ptr should be host pointer.
 */
using draw_spectrum_work = srtb::work::work<std::shared_ptr<srtb::real> >;

/**
 * @brief contains ARGB8888 @c uint32_t of width * height, to be drawn onto screen
 */
struct draw_spectrum_work_2 {
  std::shared_ptr<uint32_t> ptr;
  size_t width;
  size_t height;
};

// work queues are in global_variables.hpp

}  // namespace work
}  // namespace srtb

#endif  // __SRTB_WORK__
