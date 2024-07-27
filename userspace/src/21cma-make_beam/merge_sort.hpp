//  Distributed under the Boost Software License, Version 1.0.
//  See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt

#pragma once
#ifndef __SRTB_21CMA_MAKE_BEAM_MERGE_SORT__
#define __SRTB_21CMA_MAKE_BEAM_MERGE_SORT__

// merge_sort from Boost.Compute

#include <cstddef>
#include <iterator>
#include <sycl/sycl.hpp>
#include <type_traits>

namespace sycl_stl {

/** Calculate NdRange.
 * @brief Calculates an nd_range with a global size divisable by problemSize
 * @param problemSize : The problem size
 */
inline sycl::nd_range<1> calculateNdRange(sycl::queue& q, size_t problemSize) {
  const auto& d = q.get_device();
  const sycl::id<3> maxWorkItemSizes = d.template get_info<sycl::info::device::max_work_item_sizes<3> >();
  const auto localSize = std::min(
      problemSize, std::min(d.template get_info<sycl::info::device::max_work_group_size>(), maxWorkItemSizes[0]));

  size_t globalSize;
  if (problemSize % localSize == 0) {
    globalSize = problemSize;
  } else {
    globalSize = (problemSize / localSize + 1) * localSize;
  }

  return sycl::nd_range<1>{sycl::range<1>{globalSize}, sycl::range<1>{localSize}};
}

// reference: boost/compute/algorithm/detail/merge_sort_on_gpu.hpp
template <class InputIterator, class OutputIterator, class Compare>
inline void merge_blocks_on_gpu(sycl::queue& queue, InputIterator input_iterator, OutputIterator output_iterator,
                                Compare compare, const size_t count, const size_t block_size) {
  typedef typename std::iterator_traits<InputIterator>::value_type key_type;

  const auto ndRange = calculateNdRange(queue, count);

  auto f = [=](sycl::handler& h) {
    auto input = input_iterator;
    auto output = output_iterator;
    h.parallel_for(ndRange, [=](sycl::nd_item<1> id) {
      const size_t gid = id.get_global_id(0);
      if (gid >= count) {
        return;
      }
      const key_type my_key = input[gid];
      const size_t my_block_idx = gid / block_size;
      const bool my_block_idx_is_odd = my_block_idx & 0x1;
      const size_t other_block_idx = my_block_idx_is_odd ? my_block_idx - 1 : my_block_idx + 1;
      const size_t my_block_start = std::min(my_block_idx * block_size, count);
      const size_t my_block_end = std::min((my_block_idx + 1) * block_size, count);
      const size_t other_block_start = std::min(other_block_idx * block_size, count);
      const size_t other_block_end = std::min((other_block_idx + 1) * block_size, count);
      if (other_block_start == count) {
        output[gid] = my_key;
        return;
      }
      size_t left_idx = other_block_start;
      size_t right_idx = other_block_end;
      while (left_idx < right_idx) {
        size_t mid_idx = (left_idx + right_idx) / 2;
        key_type mid_key = input[mid_idx];
        bool smaller = compare(mid_key, my_key);
        left_idx = smaller ? mid_idx + 1 : left_idx;
        right_idx = smaller ? right_idx : mid_idx;
      }
      right_idx = other_block_end;
      if (my_block_idx_is_odd && left_idx != right_idx) {
        key_type upper_key = input[left_idx];
        while (!(compare(upper_key, my_key)) && !(compare(my_key, upper_key)) && left_idx < right_idx) {
          size_t mid_idx = (left_idx + right_idx) / 2;
          key_type mid_key = input[mid_idx];
          bool equal = !(compare(mid_key, my_key)) && !(compare(my_key, mid_key));
          left_idx = equal ? mid_idx + 1 : left_idx + 1;
          right_idx = equal ? right_idx : mid_idx;
          upper_key = input[left_idx];
        }
      }
      size_t offset = 0;
      offset += gid - my_block_start;
      offset += left_idx - other_block_start;
      offset += std::min(my_block_start, other_block_start);
      output[offset] = my_key;
    });
  };
  queue.submit(f).wait();
}

// reference: boost/compute/algorithm/detail/merge_sort_on_gpu.hpp
template <class Iterator, class Iterator2, class Compare>
inline void merge_sort_on_gpu(sycl::queue& queue, Iterator first, Iterator last, Iterator2 temp_keys, Compare compare) {
  typedef typename std::iterator_traits<Iterator>::value_type key_type;
  static_assert(std::is_same_v<key_type, typename std::iterator_traits<Iterator2>::value_type>);

  size_t count = std::distance(first, last);
  if (count < 2) {
    return;
  }

  size_t block_size = 1;

  bool result_in_temporary_buffer = false;

  for (; block_size < count; block_size *= 2) {
    result_in_temporary_buffer = !result_in_temporary_buffer;
    if (result_in_temporary_buffer) {
      merge_blocks_on_gpu(queue, first, temp_keys, compare, count, block_size);
    } else {
      merge_blocks_on_gpu(queue, temp_keys, first, compare, count, block_size);
    }
  }

  if (result_in_temporary_buffer) {
    // ::sycl::impl::copy(exec, temp_keys, temp_keys + count, first);
    queue
        .parallel_for(sycl::range<1>{count},
                      [=](sycl::item<1> id) {
                        const auto i = id.get_id(0);
                        first[i] = temp_keys[i];
                      })
        .wait();
  }
  queue.wait();
}

}  // namespace sycl_stl

#endif  // __SRTB_21CMA_MAKE_BEAM_MERGE_SORT__
