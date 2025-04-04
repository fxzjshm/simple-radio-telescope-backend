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
#ifndef __SRTB_MEMORY_DEVICE_ALLOCATOR__
#define __SRTB_MEMORY_DEVICE_ALLOCATOR__

#include <exception>
#include <memory>
#include <type_traits>

#include "srtb/sycl.hpp"

namespace srtb {
namespace memory {

/**
 * @brief An allocator that allocates device memory.
 */
template <typename T, size_t align = sizeof(T)>
class device_allocator {
 public:
  template <typename U, size_t new_align = align>
  struct rebind {
    typedef device_allocator<U, new_align> other;
  };

  using value_type = T;

  explicit device_allocator(sycl::queue &queue_) : queue(queue_){};

  T* allocate(std::size_t num_elements) {
    T *ptr = sycl::aligned_alloc_device<T>(align, num_elements, queue);
    if (!ptr) [[unlikely]] {
      throw std::runtime_error("device_allocator: Allocation failed");
    }
    return ptr;
  }

  void deallocate(T* ptr, std::size_t size) {
    (void)size;
    if (ptr) sycl::free(ptr, queue);
  }

  template <typename U>  // for srtb::mem
    requires(std::is_same_v<T, U>)
  [[nodiscard]] std::shared_ptr<T> allocate_shared(size_t n) {
    T* ptr = allocate(n);
    auto deleter = [this, n](T* ptr) { deallocate(ptr, n); };
    return std::shared_ptr<T>{ptr, deleter};
  }

  template <typename U>  // for srtb::mem
    requires(std::is_same_v<T, U>)
  [[nodiscard]] std::unique_ptr<T> allocate_unique(size_t n) {
    T* ptr = allocate(n);
    auto deleter = [this, n](T* ptr) { deallocate(ptr, n); };
    return std::unique_ptr<T>{ptr, deleter};
  }

 private:
  sycl::queue queue;
};

}  // namespace memory
}  // namespace srtb

#endif  // __SRTB_MEMORY_DEVICE_ALLOCATOR__
