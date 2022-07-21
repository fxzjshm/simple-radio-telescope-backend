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
#ifndef __SRTB_MEMORY_CACHED_ALLOCATOR__
#define __SRTB_MEMORY_CACHED_ALLOCATOR__

#include <map>
#include <memory>
#include <mutex>
#include <type_traits>

#include "srtb/config.hpp"
#include "srtb/logger.hpp"

namespace srtb {
namespace memory {

/**
 * @brief An allocator that wraps the real allocator and caches allocation requests.
 * 
 * @tparam RealAllocator The real allocator to allocate new memory.
 *         This is a parameter in case different types of memory need to be allocated, 
 *         e.g. host memory, device memory, USM memory, etc.
 * @note TODO: @c std::mutex is used, but problem like dead locks need to be checked.
 * @note template template parameters seems not supporting Nontype template parameters...
 * @note TODO: read/write lock?
 */
template <typename RealAllocator>
class cached_allocator {
 public:
  using value_type = typename std::allocator_traits<RealAllocator>::value_type;
  using size_type = typename std::allocator_traits<RealAllocator>::size_type;
  using pointer = typename std::allocator_traits<RealAllocator>::pointer;

  template <typename... Args>
  explicit cached_allocator(Args... args) : allocator(args...) {
    p_mutex = std::make_shared<std::mutex>();
  };

  pointer allocate(size_type n) {
    std::lock_guard lock{*p_mutex};

    // find a memory region that is cached
    auto iter = free_ptrs.lower_bound(n);
    pointer ptr = nullptr;
    size_type ptr_size = 0;

    if (iter == free_ptrs.end()) {
      // not found, allocate a new one
      // pay attention to alignment
      if constexpr (srtb::MEMORY_ALIGNMENT != 0) {
        if (n > 0) [[likely]] {
          ptr_size =
              ((n - 1) / srtb::MEMORY_ALIGNMENT + 1) * srtb::MEMORY_ALIGNMENT;
        } else {
          ptr_size = srtb::MEMORY_ALIGNMENT;
        }
      } else {
        ptr_size = n;
      }
      ptr = allocator.allocate(ptr_size);
      if (ptr == nullptr) [[unlikely]] {
        throw std::runtime_error(
            std::string("Cached allocator: cannot allocate memory of size ") +
            std::to_string(ptr_size * sizeof(value_type)));
      }

      SRTB_LOGI << " [cached allocator] "
                << "allocated new memory of size "
                << ptr_size * sizeof(value_type) << std::endl;
    } else {
      // found. notice the real ptr size may be larger than requested n
      auto pair = *iter;
      ptr = pair.second;
      ptr_size = pair.first;
      free_ptrs.erase(iter);

      SRTB_LOGD << " [cached allocator] "
                << "cached allocation of memory of size "
                << ptr_size * sizeof(value_type) << std::endl;
    }

    used_ptrs.insert(std::make_pair(ptr, ptr_size));
    return ptr;
  }

  // TODO: check this
  template <typename U = value_type,
            typename = typename std::enable_if<std::is_convertible_v<
                typename std::remove_cv<pointer>::type, value_type*> >::type>
  std::shared_ptr<U> allocate_shared(size_type n_U) {
    using T = value_type;
    size_type n_T =
        (std::max(n_U, size_type{1}) * sizeof(U) - 1) / sizeof(T) + 1;
    assert(n_T * sizeof(T) >= n_U * sizeof(U));

    pointer ptr = allocate(n_T);
    return std::shared_ptr<U>{reinterpret_cast<U*>(ptr), [&](U* ptr) {
                                deallocate(reinterpret_cast<pointer>(ptr));
                              }};
  }

  void deallocate(pointer ptr) {
    std::lock_guard lock{*p_mutex};

    auto iter = used_ptrs.find(ptr);
    if (iter == used_ptrs.end()) [[unlikely]] {
      // something wrong here, check double free
      for (auto [ptr_size, ptr2] : free_ptrs) {
        if (ptr2 == ptr) {
          SRTB_LOGE << " [cached_allocator] "
                    << "double free of ptr (size "
                    << ptr_size * sizeof(value_type) << ") detected!"
                    << std::endl;
          return;
        }
      }
      // not a double free, but something more serious
      throw std::runtime_error(
          "Cached allocator: cannot handle unknown pointer.");
    }
    size_type ptr_size = (*iter).second;
    used_ptrs.erase(iter);
    free_ptrs.insert(std::make_pair(ptr_size, ptr));
    SRTB_LOGD << " [cached allocator] "
              << "take back memory of size " << ptr_size * sizeof(value_type)
              << " bytes" << std::endl;
  }

  void deallocate_all_free_ptrs() {
    std::lock_guard lock{*p_mutex};
    for (auto iter : free_ptrs) {
      size_type ptr_size = iter.first;
      pointer ptr = iter.second;
      allocator.deallocate(ptr, ptr_size);
      SRTB_LOGD << " [cached allocator] "
                << "deallocate memory of size " << ptr_size * sizeof(value_type)
                << " bytes" << std::endl;
    }
    free_ptrs.clear();
  }

  ~cached_allocator() {
    deallocate_all_free_ptrs();

    // TODO: should used_ptrs be cleared?
    if (used_ptrs.size() > 0) {
      SRTB_LOGW << " [cached allocator] " << used_ptrs.size()
                << " pointer(s) still in use!" << std::endl;
    }
  }

 protected:
  std::multimap<size_type, pointer> free_ptrs;
  std::map<pointer, size_type> used_ptrs;
  RealAllocator allocator;
  // TODO: is a shared_ptr here appropriate?
  std::shared_ptr<std::mutex> p_mutex;
};

}  // namespace memory
}  // namespace srtb

#endif  // __SRTB_MEMORY_CACHED_ALLOCATOR__
