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
#include <thread>
#include <vector>
#include <memory>

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
template <typename T, bool fixed_size = srtb::work_queue_fixed_size,
          size_t capacity = srtb::work_queue_capacity>
class work_queue;

template <typename T, size_t capacity>
class work_queue<T, true, capacity>
    : public boost::lockfree::spsc_queue<T,
                                         boost::lockfree::capacity<capacity> > {
};

template <typename T, size_t initial_capacity>
class work_queue<T, false, initial_capacity>
    : public boost::lockfree::spsc_queue<T> {
 public:
  using super_class = boost::lockfree::spsc_queue<T>;
  work_queue(size_t initial_capacity_ = initial_capacity)
      : super_class{initial_capacity_} {}
};

/**
 * @brief This namespace contains work types that defines the input of a pipe.
 *        Ideally all info needed to execute the pipeline should be written in a POD work class.
 */
namespace work {

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
   *        currently @c std::shared_ptr is used to reduce memory error.
   */
  T ptr;
  /**
   * @brief count of data in @c ptr
   */
  size_t count;
  /**
   * @brief time stamp of these data, currectly 64-bit unix timestamp.
   */
  uint64_t timestamp;
  // TODO: udp_packet_counter_type counter;
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
  sycl::event host_to_device_copy_event;
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
 *        to be reFFT-ed with length @c refft_length to get spectrum with
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
 * @brief contains @c srtb::complex<srtb::real> to be taken norm and summed into
 *        time series, then try tp find signal in it.
 */
using signal_detect_work = simplify_spectrum_work;

/**
 * @brief contains baseband data without UDP packet counter in host memory.
 */
using baseband_output_work = srtb::work::work<std::shared_ptr<std::byte> >;

/**
 * @brief contains info for a time series of a spectrum
 */
struct time_series_holder {
  std::shared_ptr<srtb::real> h_time_series;
  size_t time_series_length;
  /** @brief size of boxcar used to compute the time series; = 1 if not used */
  size_t boxcar_length;
  sycl::event transfer_event;
};

/**
 * @brief contains timestamp of a work & signal related time-series.
 *        If it has signal, @c ptr is host pointer containing dedispersed baseband data
 *        Should be generated by @c signal_detect_pipe and received by @c baseband_output_pipe
 * @note time_series.size() == 0 means no signal
 * @see time_series_holder
 */
struct signal_detect_result {
  uint64_t timestamp;
  std::vector<time_series_holder> time_series;
};

// work queues are in global_variables.hpp

}  // namespace work
}  // namespace srtb

#endif  // __SRTB_WORK__
