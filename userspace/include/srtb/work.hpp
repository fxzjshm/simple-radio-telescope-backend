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

#define SRTB_PUSH_WORK(tag, work_queue, work)                                \
  {                                                                          \
    bool ret = work_queue.push(work);                                        \
    if (!ret) [[unlikely]] {                                                 \
      SRTB_LOGW << tag                                                       \
                << " Pushing " #work " to " #work_queue " failed! Retrying." \
                << srtb::endl;                                               \
      while (!ret) {                                                         \
        std::this_thread::yield(); /* TODO: spin lock here? */               \
        std::this_thread::sleep_for(std::chrono::nanoseconds(                \
            srtb::config.thread_query_work_wait_time));                      \
        ret = work_queue.push(work);                                         \
      }                                                                      \
      SRTB_LOGI << tag << " Pushed " #work " to " #work_queue << srtb::endl; \
    } else [[likely]] {                                                      \
      SRTB_LOGD << tag << " Pushed " #work " to " #work_queue << srtb::endl; \
    }                                                                        \
  }

#define SRTB_POP_WORK(tag, work_queue, work)                                   \
  {                                                                            \
    bool ret = work_queue.pop(work);                                           \
    if (!ret) [[unlikely]] {                                                   \
      SRTB_LOGD << tag                                                         \
                << " Popping " #work " from " #work_queue " failed! Retrying." \
                << srtb::endl;                                                 \
      while (!ret) {                                                           \
        std::this_thread::yield(); /* TODO: spin lock here? */                 \
        std::this_thread::sleep_for(std::chrono::nanoseconds(                  \
            srtb::config.thread_query_work_wait_time));                        \
        ret = work_queue.pop(work);                                            \
      }                                                                        \
    }                                                                          \
    SRTB_LOGD << tag << " Popped " #work " from " #work_queue << srtb::endl;   \
  }

namespace srtb {

// definition of work queue, a container of works to be processed.
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

/**
 * @brief This namespace contains work types that defines the input of a pipe.
 *        Ideally all info needed to execute the pipeline should be written in a POD work class.
 */
namespace work {

/**
 * @brief This represents a work to be done and should be the same as `std::pair<T, size_t>`,
 *        created just because `std::pair` doesn't satisfy `boost::has_trivial_assign`,
 *        which is required for lockfree queue.
 * @tparam T Type of the pointer of the work, e.g. std::shared_ptr<std::byte> for unpack and std::shared_ptr<srtb::real> for FFT.
 *         TODO: Maybe T = sycl::buffer<std::byte> if pointer isn't suitable for some backend in the future.
 */
template <typename T>
struct work {
  T ptr;
  /**
   * @brief count of data in @c ptr
   */
  size_t count;
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
  size_t baseband_input_bits;
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
 *        (and channelized if kernel fused)
 */
struct dedisperse_and_channelize_work
    : public srtb::work::work<std::shared_ptr<srtb::complex<srtb::real> > > {
  srtb::real dm;
  size_t channel_count;
  srtb::real baseband_freq_low;
  srtb::real baseband_sample_rate;
};

/**
 * @brief contains @c batch_size * @c count of @c srtb::complex<srtb::real>
 *        to be inversed FFT-ed
 * @note @c count is not total size
 */
struct ifft_1d_c2c_work
    : public srtb::work::work<std::shared_ptr<srtb::complex<srtb::real> > > {
  size_t batch_size;
};

/**
 * @brief contains complex FFT-ed (and dedispersed) data of total length @c count
 *        to be iFFT-ed and reFFT-ed with length @c refft_length to get much higher
 *        time resolution.
 */
struct refft_1d_c2c_work
    : public srtb::work::work<std::shared_ptr<srtb::complex<srtb::real> > > {
  size_t refft_length;
};

/**
 * @brief contains @c srtb::complex<srtb::real> to be simplified into
 *        ~10^3 @c srtb::real to be displayed on GUI.
 * @note temporary work, just do a software-defined-radio receiver job.
 */
using simplify_spectrum_work = ifft_1d_c2c_work;

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
 * @brief contains baseband data without UDP packet counter in host memory.
 *        @c counter contains counter of first UDP packet.
 * TODO: put this in a pool or first-in-first-out queue, and write data only if signal is detected.
 */
struct baseband_output_work : public srtb::work::work<std::shared_ptr<std::byte> > {
  /**
   * @brief time stamp these data
   */
  uint64_t timestamp;
  // TODO: udp_packet_counter_type counter;
};

// work queues are in global_variables.hpp

}  // namespace work
}  // namespace srtb

#endif  // __SRTB_WORK__
