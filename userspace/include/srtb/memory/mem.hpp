/*******************************************************************************
 * Copyright (c) 2024 fxzjshm
 * 21cma-make_beam is licensed under Mulan PubL v2.
 * You can use this software according to the terms and conditions of the Mulan PubL v2.
 * You may obtain a copy of Mulan PubL v2 at:
 *          http://license.coscl.org.cn/MulanPubL-2.0
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PubL v2 for more details.
 ******************************************************************************/

#pragma once
#ifndef SRTB_MEM
#define SRTB_MEM

#include <cstddef>
#include <span>

#include "mdspan/mdspan.hpp"
// ---
#if __has_include("srtb/util/assert.hpp")
#include "srtb/util/assert.hpp"
#else
#include <boost/assert.hpp>
#endif

namespace srtb {

/** @brief similar to std::span. but use smart pointer & owning this memory */
template <typename Pointer, typename SizeType = size_t>
struct mem {
  Pointer ptr;
  SizeType count;
  using Type = typename Pointer::element_type;

#if defined(__clang__) && __clang_major__ <= 15
  mem() : ptr(nullptr), count(0) {}
  mem(Pointer ptr_, SizeType count_) : ptr{ptr_}, count{count_} {};
#endif

  operator std::span<Type, std::dynamic_extent>() { return get_span(); }
  auto get_span() { return std::span{ptr.get(), count}; }

  template <typename... T>
  auto get_mdspan(T... sizes) {
    SizeType n = (1 * ... * sizes);
    BOOST_ASSERT(n == count);
    return Kokkos::mdspan{ptr.get(), sizes...};
  }
};

template <typename T, typename Allocator>
inline auto mem_allocate_shared(Allocator allocator, size_t count) {
  constexpr bool allocator_is_value = requires() { allocator.template allocate_shared<T>(count); };
  constexpr bool allocator_is_pointer = requires() { allocator->template allocate_shared<T>(count); };
  if constexpr (allocator_is_value) {
    return mem{allocator.template allocate_shared<T>(count), count};
  } else if constexpr (allocator_is_pointer) {
    return mem{allocator->template allocate_shared<T>(count), count};
  } else {
    static_assert(allocator_is_value || allocator_is_pointer);
  }
}

}  // namespace srtb

#endif  // SRTB_MEM
