/**
 * This algorithm is directly edited from SyclParallelSTL's buffer_mapreduce
 * `sycl/algorithm/buffer_algorithms.hpp`
 * so ... don't know how this file is / should be licensed ...
 * original license header is attached below, as required.
 */

/* Copyright (c) 2015-2018 The Khronos Group Inc.

   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and/or associated documentation files (the
   "Materials"), to deal in the Materials without restriction, including
   without limitation the rights to use, copy, modify, merge, publish,
   distribute, sublicense, and/or sell copies of the Materials, and to
   permit persons to whom the Materials are furnished to do so, subject to
   the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Materials.

   MODIFICATIONS TO THIS FILE MAY MEAN IT NO LONGER ACCURATELY REFLECTS
   KHRONOS STANDARDS. THE UNMODIFIED, NORMATIVE VERSIONS OF KHRONOS
   SPECIFICATIONS AND HEADER INFORMATION ARE LOCATED AT
    https://www.khronos.org/registry/

  THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
  CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
  TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
  MATERIALS OR THE USE OR OTHER DEALINGS IN THE MATERIALS.

*/

#pragma once
#ifndef __SRTB_ALGORITHM_MULTI_REDUCE_HPP__
#define __SRTB_ALGORITHM_MULTI_REDUCE_HPP__

#include "srtb/sycl.hpp"
// -- divide line for clang-format --
#include "srtb/algorithm/map_identity.hpp"

namespace srtb {
namespace algorithm {

template <typename T>
inline T up_rounded_division(T x, T y) {
  return (x + (y - 1)) / y;
}

struct sycl_algorithm_descriptor {
  size_t size;
  size_t size_per_work_group;
  size_t size_per_work_item;
  size_t nb_work_group;
  size_t nb_work_item;
};

// base: sycl/algorithm/buffer_algorithms.hpp of SyclParallelSTL
//       https://github.com/KhronosGroup/SyclParallelSTL/blob/master/include/sycl/algorithm/buffer_algorithms.hpp
// ref:  https://stackoverflow.com/questions/17862078/reduce-matrix-rows-with-cuda
template <typename InputIterator, typename OutputIterator, typename Reduce,
          typename Map>
void multi_mapreduce(InputIterator input, size_t count_per_batch,
                     size_t batch_size, OutputIterator output, Map map,
                     Reduce reduce, sycl::queue& q) {
  using B = typename std::iterator_traits<OutputIterator>::value_type;

  const size_t total_count = count_per_batch * batch_size;
  const size_t size = total_count;
  const size_t sizeofB = sizeof(B);
  auto device = q.get_device();

  const sycl::id<3> max_work_item_sizes = device.get_info<
#if defined(__COMPUTECPP__)
      sycl::info::device::max_work_item_sizes
#else
      sycl::info::device::max_work_item_sizes<3>
#endif
      >();
  const auto max_work_item =
      std::min(device.get_info<sycl::info::device::max_work_group_size>(),
               max_work_item_sizes[0]);

  size_t local_mem_size = device.get_info<sycl::info::device::local_mem_size>();

  // edited: in this senario try to compute one batch using one work group
  const size_t nb_work_group = batch_size;

  // count of work items *in one work group*
  size_t nb_work_item =
      std::min(std::min(max_work_item, local_mem_size / sizeofB), size);

  //number of elements manipulated by each work_item
  size_t size_per_work_item =
      up_rounded_division(size, nb_work_item * nb_work_group);

  //number of elements manipulated by each work_group (except the last one)
  size_t size_per_work_group = size_per_work_item * nb_work_item;

  const sycl_algorithm_descriptor d{.size = total_count,
                                    .size_per_work_group = size_per_work_group,
                                    .size_per_work_item = size_per_work_item,
                                    .nb_work_group = nb_work_group,
                                    .nb_work_item = nb_work_item};

  /*
   * 'map' is not applied on init
   * 'map': A -> B
   * 'reduce': B * B -> B
   */

  q.submit([&](sycl::handler& cgh) {
     sycl::range<1> rg{d.nb_work_group};
     sycl::range<1> ri{d.nb_work_item};
     sycl::local_accessor<B, 1> sum{sycl::range<1>(d.nb_work_item), cgh};
     cgh.parallel_for(
         sycl::nd_range<1>(rg * ri, ri), [=](sycl::nd_item<1> nd_item) {
           size_t group_id = nd_item.get_group(0);
           size_t group_begin = group_id * d.size_per_work_group;
           size_t group_end =
               std::min((group_id + 1) * d.size_per_work_group, d.size);
           //assert(group_id < d.nb_work_group);
           //assert(group_begin < group_end); //< as we properly selected the
           //  number of work_group
           size_t local_id = nd_item.get_local_id(0);
           size_t local_pos = group_begin + local_id;
           if (local_pos < group_end) {
             //we peal the first iteration
             B acc = map(local_pos, input[local_pos]);
             for (size_t read = local_pos + d.nb_work_item; read < group_end;
                  read += d.nb_work_item) {
               acc = reduce(acc, map(read, input[read]));
             }
             sum[local_id] = acc;
           }

           nd_item.barrier(sycl::access::fence_space::local_space);

           // reduce_over_group, but SYCL 2020 doesn't support custom operations
           // so custom one is used
           if (local_id == 0) {
             B acc = sum[0];
             for (size_t i = 1;
                  i < std::min(d.nb_work_item, group_end - group_begin); i++) {
               acc = reduce(acc, sum[i]);
             }

             output[group_id] = acc;
           }
         });
   }).wait();
}

template <typename InputIterator, typename OutputIterator, typename Reduce>
void multi_reduce(InputIterator input, size_t count_per_batch,
                  size_t batch_size, OutputIterator output, Reduce reduce,
                  sycl::queue& q) {
  return multi_mapreduce(input, count_per_batch, batch_size, output,
                         srtb::algorithm::map_identity(), reduce, q);
}

}  // namespace algorithm
}  // namespace srtb

#endif  // __SRTB_ALGORITHM_MULTI_REDUCE_HPP__
