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
#ifndef __SRTB_MEMORY_SYCL_RING_BUFFER__
#define __SRTB_MEMORY_SYCL_RING_BUFFER__

#include <memory>
#include <mutex>

#include "srtb/log/log.hpp"
#include "srtb/sycl.hpp"

// currently only used in single thread, so no need to use mutex now
// this flag is reserved for future use
// TODO: is std::atomic or some lockfree thing better ?
//#define SRTB_MEMORY_SYCL_RING_BUFFER_USE_MUTEX

namespace srtb {
namespace memory {

template <typename T, typename Deleter>
class sycl_ring_buffer {
 protected:
#ifdef SRTB_MEMORY_SYCL_RING_BUFFER_USE_MUTEX
  mutable std::mutex mutex;
#endif  // SRTB_MEMORY_SYCL_RING_BUFFER_USE_MUTEX
  sycl::queue q;
  std::unique_ptr<T, Deleter> buffer;
  /** @brief total length of memory region in @c buffer, in count of elements, not bytes */
  const size_t capacity;
  /** index of first element valid in ptr (i.e. inclusive), 0 <= head < capacity */
  size_t head;
  /** index of last element valid in ptr + 1 (i.e. exclusive), 0 <= tail <= capacity */
  size_t tail;

 public:
  explicit sycl_ring_buffer(std::unique_ptr<T, Deleter> ptr_, size_t capacity_,
                            sycl::queue q_)
      : q{q_}, buffer{std::move(ptr_)}, capacity{capacity_}, head{0}, tail{0} {}

  auto available_length() const -> size_t { return tail - head; }

  /**
   * @brief try to pop elements of @c request_length from buffer to @c ptr
   * 
   * @return true if success
   * @return false if failed to pop because not enough elements available
   */
  [[nodiscard]] bool pop(T* ptr, size_t request_length) {
#ifdef SRTB_MEMORY_SYCL_RING_BUFFER_USE_MUTEX
    std::lock_guard lock{mutex};
#endif  // SRTB_MEMORY_SYCL_RING_BUFFER_USE_MUTEX
    if (request_length <= available_length()) {
      if (ptr != nullptr) {
        q.copy(buffer.get() + head, /* -> */ ptr, request_length).wait();
      }
      head += request_length;
      return true;
    } else {
      return false;
    }
  }

  /**
   * @brief try to get raw pointer to head of the buffer, if enough is available;
   *        or @c nullptr if not enough.
   * @note actually this should return const T*, but since the program have to 
   *       do something on data before operating on these data, 
   *       e.g. apply FFT window, writable T* is allowed here.
   *       And perhaps const_cast is more ugly.
   *  
   * TODO: thread safety? data race?
   */
  [[nodiscard]] T* peek(size_t request_length) const {
#ifdef SRTB_MEMORY_SYCL_RING_BUFFER_USE_MUTEX
    std::lock_guard lock{mutex};
#endif  // SRTB_MEMORY_SYCL_RING_BUFFER_USE_MUTEX
    if (request_length <= available_length()) {
      return buffer.get() + head;
    } else {
      return nullptr;
    }
  }

  /**
   * @brief try to push elements of @c request_length at @c ptr into buffer
   * 
   * @return true if success
   * @return false if not enough space to push. 
   *         may also happen if data length in buffer is more than half of capacity
   */
  [[nodiscard]] bool push(T* ptr, size_t request_length) {
#ifdef SRTB_MEMORY_SYCL_RING_BUFFER_USE_MUTEX
    std::lock_guard lock{mutex};
#endif  // SRTB_MEMORY_SYCL_RING_BUFFER_USE_MUTEX
    const size_t current_length = available_length();
    if (current_length + request_length > capacity) {
      return false;
    }
    if (tail + request_length > capacity) {
      if (current_length <= head) {
        q.copy(buffer.get() + head, /* -> */ buffer.get(), current_length)
            .wait();
        head = 0;
        tail = current_length;
      } else {
        // TODO: memmove if available_length > head
        return false;
      }
    }

    if (tail + request_length > capacity) [[unlikely]] {
      SRTB_LOGW << " [sycl_ring_buffer] "
                << "logical error in push() ?" << srtb::endl;
      return false;
    } else {
      q.copy(ptr, /* -> */ buffer.get() + tail, request_length).wait();
      tail += request_length;
      return true;
    }
  }
};

}  // namespace memory
}  // namespace srtb

#endif  // __SRTB_MEMORY_SYCL_RING_BUFFER__
