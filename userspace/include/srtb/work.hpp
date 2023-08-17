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

#define SRTB_PUSH_WORK_OR_RETURN(tag, work_queue, work, stop_token)        \
  {                                                                        \
    bool ret = work_queue.push(work);                                      \
    if (!ret) [[unlikely]] {                                               \
      SRTB_LOGW << tag                                                     \
                << " Pipeline stuck: Pushing " #work " to " #work_queue    \
                   " failed! Retrying."                                    \
                << srtb::endl;                                             \
      while (!ret) {                                                       \
        if (stop_token.stop_requested()) [[unlikely]] {                    \
          return;                                                          \
        }                                                                  \
        std::this_thread::yield(); /* TODO: spin lock here? */             \
        std::this_thread::sleep_for(std::chrono::nanoseconds(              \
            srtb::config.thread_query_work_wait_time));                    \
        ret = work_queue.push(work);                                       \
      }                                                                    \
    }                                                                      \
    SRTB_LOGD << tag << " Pushed " #work " to " #work_queue << srtb::endl; \
  }

#define SRTB_POP_WORK_OR_RETURN(tag, work_queue, work, stop_token) \
  {                                                                \
    bool ret = work_queue.pop(work);                               \
    if (!ret) [[unlikely]] {                                       \
      while (!ret) {                                               \
        if (stop_token.stop_requested()) [[unlikely]] {            \
          return;                                                  \
        }                                                          \
        std::this_thread::yield(); /* TODO: spin lock here? */     \
        std::this_thread::sleep_for(std::chrono::nanoseconds(      \
            srtb::config.thread_query_work_wait_time));            \
        ret = work_queue.pop(work);                                \
      }                                                            \
    }                                                              \
  }

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
 */
namespace work {

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
   * @brief count of data in @c ptr; some may also have @c batch_size
   *        refer to specialized work types for detail
   */
  size_t count;
  /**
   * @brief time stamp of these data, currectly 64-bit unix timestamp from server time
   */
  uint64_t timestamp;
  /**
   * @brief counter of correspond first new UDP packet, == no_udp_packet_counter means no value.
   */
  uint64_t udp_packet_counter;
  /**
   * @brief dummy value if no udp_packet_counter is available
   */
  static constexpr uint64_t no_udp_packet_counter = static_cast<uint64_t>(-1);
  /**
   * @brief original baseband input correspond to this work
   */
  baseband_data_holder baseband_data;
};

/**
 * @brief contains a chunk of @c std::byte of size @c count, which is 
 *        baseband data and should be unpacked into @c srtb::real
 * @note count is count of std::byte, not of output time series,
 *       and should equal to `srtb::config.baseband_input_count * srtb::config.baseband_input_bits / srtb::BITS_PER_BYTE`
 */
struct unpack_work : public srtb::work::work<std::shared_ptr<std::byte> > {
  /**
   * @brief length of a single time sample in the input, come from @c srtb::config.baseband_input_bits
   *        currently 1, 2, 4 and 8 bit(s) baseband input is implemented,
   *        others will result in ... undefined behaviour.
   */
  int baseband_input_bits;
  /**
   * @brief let unpack_pipe to wait for host to device copy, so that
   *        udp receiver pipe may lose less packet.
   *        May be empty for other data sources that is not real-time.
   */
  sycl::event copy_event;
};

/**
 * @brief contains a chunk of @c srtb::real that is to be FFT-ed.
 * @note real number of size @c n should be FFT-ed into @c n/2+1 @c srtb::complex<srtb::real> s,
 *       take care of memory allocation.
 */
using fft_1d_r2c_work = srtb::work::work<std::shared_ptr<srtb::real> >;

/**
 * @brief comtains a block of @c srtb::complex<srtb::real> with radio interference
 *        to be cleared out
 */
using rfi_mitigation_work =
    srtb::work::work<std::shared_ptr<srtb::complex<srtb::real> > >;

/**
 * @brief contains a piece of @c srtb::complex<srtb::real> to be coherently dedispersed
 */
struct dedisperse_work
    : public srtb::work::work<std::shared_ptr<srtb::complex<srtb::real> > > {
  srtb::real dm;
  srtb::real baseband_freq_low;
  srtb::real baseband_sample_rate;
};

/**
 * @brief contains @c batch_size * @c count of @c srtb::complex<srtb::real>
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
 * @brief contains @c srtb::complex<srtb::real> to be simplified into
 *        ~10^3 @c srtb::real to be displayed on GUI.
 * @note temporary work, just do a software-defined-radio receiver job.
 */
struct simplify_spectrum_work
    : public srtb::work::work<std::shared_ptr<srtb::complex<srtb::real> > > {
  size_t batch_size;
};

/**
 * @brief contains ~10^3 * @c batch_size of @c srtb::real to be summed and drawn
 *        to a line of a pixmap. @c ptr should be host pointer.
 * @note temporary work, see above.
 */
struct draw_spectrum_work
    : public srtb::work::work<std::shared_ptr<srtb::real> > {
  size_t batch_size;
};

/**
 * @brief contains ARGB8888 @c uint32_t of width * height, to be drawn onto screen
 */
struct draw_spectrum_work_2 {
  std::shared_ptr<uint32_t> ptr;
  size_t width;
  size_t height;
};

/**
 * @brief contains @c srtb::complex<srtb::real> to be taken norm and summed into
 *        time series, then try tp find signal in it.
 */
using signal_detect_work = simplify_spectrum_work;

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
 * @brief write baseband data to disk; @c ptr is reserved for future use
 */
struct baseband_output_work
    : public srtb::work::work<std::shared_ptr<srtb::complex<srtb::real> > > {
  size_t batch_size;
  std::vector<time_series_holder> time_series;
};

// work queues are in global_variables.hpp

/**
 * @brief Dummy work as place holder for work queue / pipe work transfer
 */
struct dummy_work {};

}  // namespace work
}  // namespace srtb

#endif  // __SRTB_WORK__
