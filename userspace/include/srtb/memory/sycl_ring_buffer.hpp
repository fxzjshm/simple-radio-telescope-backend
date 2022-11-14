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

namespace srtb {
namespace memory {

template <typename T>
class sycl_ring_buffer {
 protected:
  mutable std::mutex mutex;
  sycl::queue q;
  std::unique_ptr<T> buffer;
  /** @brief total length of memory region in @c buffer , in count of elements, not bytes */
  const size_t length;
  /** index of first element valid in ptr (i.e. inclusive), 0 <= head < length */
  size_t head;
  /** index of last element valid in ptr + 1 (i.e. exclusive), 0 <= tail <= length */
  size_t tail;

  auto available_length() const -> size_t { return tail - head; }

 public:
  explicit sycl_ring_buffer(std::unique_ptr<T> ptr_, size_t length_)
      : buffer{std::move(ptr_)}, length{length_}, head{0}, tail{0} {}

  [[nodiscard]] bool pop(T* ptr, size_t request_length) {
    std::lock_guard lock{mutex};
    if (request_length < available_length()) {
      if (ptr != nullptr) {
        q.copy(buffer.get() + head, /* -> */ ptr, request_length).wait();
      }
      head += request_length;
      return true;
    } else {
      return false;
    }
  }

  // TODO: thread safety? data race?
  [[nodiscard]] const T* peek(size_t request_length) {
    std::lock_guard lock{mutex};
    if (request_length < available_length()) {
      return buffer.get() + head;
    } else {
      return nullptr;
    }
  }

  [[nodiscard]] bool push(T* ptr, size_t request_length) {
    std::lock_guard lock{mutex};
    const size_t current_length = available_length();
    if (current_length + request_length > length) {
      return false;
    }
    if (tail + request_length > length) {
      if (current_length <= head) {
        q.copy(buffer.get() + head, /* -> */ buffer.get(), current_length).wait();
        head = 0;
        tail = current_length;
      } else {
        // TODO: memmove if available_length > head
        return false;
      }
    }

    if (tail + request_length > length) {
      SRTB_LOGW << " [sycl_ring_buffer] "
                << "logical error in push() ?" << srtb::endl;
    }

    q.copy(ptr, /* -> */ buffer.get() + tail, request_length);
    tail += request_length;
  }
};

}  // namespace memory
}  // namespace srtb

#endif  // __SRTB_MEMORY_SYCL_RING_BUFFER__
