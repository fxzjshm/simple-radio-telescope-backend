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

#include <algorithm>
#include <bit>
#include <cassert>
#include <memory>
#include <mutex>
#include <type_traits>

// currently only used in single thread, so no need to use mutex now
// this flag is reserved for future use
// TODO: is std::atomic or some lockfree thing better ?
//#define SRTB_MEMORY_HOST_RING_BUFFER_USE_MUTEX

namespace srtb {
namespace memory {

template <typename T>
class host_ring_buffer {
 protected:
#ifdef SRTB_MEMORY_HOST_RING_BUFFER_USE_MUTEX
  mutable std::mutex mutex;
#endif  // SRTB_MEMORY_HOST_RING_BUFFER_USE_MUTEX
  std::vector<T> buffer;
  /** index of first element valid in ptr (i.e. inclusive), 0 <= head < capacity, head == capacity -> empty */
  size_t head;
  /** index of last element valid in ptr + 1 (i.e. exclusive), 0 <= tail <= capacity */
  size_t tail;

 public:
  explicit host_ring_buffer(size_t init_capacity = 0) : head{0}, tail{0} {
    buffer.resize(init_capacity);
  }

  [[nodiscard]] auto size() const -> size_t {
    check_index();
    return tail - head;
  }

  [[nodiscard]] auto capacity() const -> size_t { return buffer.size(); }

  void reserve(size_t n) {
    check_index();
    buffer.resize(std::bit_ceil(n));
  }

  void check_index() const {
    assert(0 <= head);
    assert(head <= capacity());
    assert(0 <= tail);
    assert(tail <= capacity());
  }

  /**
   * @brief try to pop elements of @c request_length from buffer into @c output
   */
  template <typename OutputIterator>
  void pop(OutputIterator output, size_t request_length) {
#ifdef SRTB_MEMORY_HOST_RING_BUFFER_USE_MUTEX
    std::lock_guard lock{mutex};
#endif  // SRTB_MEMORY_HOST_RING_BUFFER_USE_MUTEX
    check_index();

    if (request_length <= size()) {
      bool output_valid = true;
      if constexpr (std::is_pointer_v<OutputIterator>) {
        if (output == nullptr) {
          output_valid = false;
        }
      }
      if constexpr (!std::is_null_pointer_v<OutputIterator>) {
        if (output_valid) {
          std::copy_n(buffer.begin() + head, request_length, output);
        }
      }
      head += request_length;
    } else {
      throw std::runtime_error{"[host_ring_buffer] pop: not enough items"};
    }
  }

  /**
   * @brief try to get raw pointer to head of the buffer, if enough is available;
   *        or @c nullptr if not enough.
   * @note actually this should return const T*, but since the program have to 
   *       do something on data before operating on these data, 
   *       e.g. apply FFT window, writable T* is allowed here.
   *       And perhaps const_cast is more ugly.
   * @note use pointer returned here after @c push() is forbidden,
   *       otherwise use-after-free may happen.
   *  
   * TODO: thread safety? data race?
   */
  [[nodiscard]] T* peek(size_t request_length) {
#ifdef SRTB_MEMORY_HOST_RING_BUFFER_USE_MUTEX
    std::lock_guard lock{mutex};
#endif  // SRTB_MEMORY_HOST_RING_BUFFER_USE_MUTEX
    check_index();

    if (request_length <= size()) {
      return &buffer[head];
    } else {
      throw std::runtime_error{"[host_ring_buffer] peek: not enough items"};
    }
  }

  /**
   * @brief prepare space for new elements of @c request_length
   */
  void prepare(size_t request_length) {
#ifdef SRTB_MEMORY_HOST_RING_BUFFER_USE_MUTEX
    std::lock_guard lock{mutex};
#endif  // SRTB_MEMORY_HOST_RING_BUFFER_USE_MUTEX
    check_index();
    prepare_impl(request_length);
  }

 protected:
  void prepare_impl(size_t request_length) {
    check_index();

    // 1) try move data to left of buffer
    if (tail + request_length > capacity()) {
      const size_t current_length = size();
      // copying to left, so use std::copy
      auto begin = buffer.begin();
      std::copy(begin + head, begin + tail, begin);
      head = 0;
      tail = current_length;
    }

    // 2) resize buffer
    if (tail + request_length > capacity()) {
      reserve(tail + request_length);
    }

    if (tail + request_length > capacity()) [[unlikely]] {
      throw std::runtime_error{" [sycl_ring_buffer] logical error in push() ?"};
    }
  }

 public:
  /**
   * @brief try to push elements of @c request_length at @c input into buffer
   */
  template <typename InputIterator>
  void push(InputIterator input, size_t request_length) {
#ifdef SRTB_MEMORY_HOST_RING_BUFFER_USE_MUTEX
    std::lock_guard lock{mutex};
#endif  // SRTB_MEMORY_HOST_RING_BUFFER_USE_MUTEX
    check_index();

    prepare_impl(request_length);
    std::copy_n(input, request_length, buffer.begin() + tail);
    tail += request_length;
  }

  /**
   * @brief try to fill elements of @c request_length into buffer
   */
  void fill(T x, size_t request_length) {
#ifdef SRTB_MEMORY_HOST_RING_BUFFER_USE_MUTEX
    std::lock_guard lock{mutex};
#endif  // SRTB_MEMORY_HOST_RING_BUFFER_USE_MUTEX
    check_index();

    prepare_impl(request_length);
    std::fill_n(buffer.begin() + tail, request_length, x);
    tail += request_length;
  }
};

}  // namespace memory
}  // namespace srtb

#endif  // __SRTB_MEMORY_SYCL_RING_BUFFER__
