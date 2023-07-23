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
#ifndef __SRTB_SIMPLIFY_SPECTRUM__
#define __SRTB_SIMPLIFY_SPECTRUM__

#include "srtb/commons.hpp"
// --- divide line for clang-format ---
#include "srtb/algorithm/map_identity.hpp"
#include "srtb/algorithm/map_reduce.hpp"

namespace srtb {
namespace spectrum {

/** @brief a very small float number to test equivalence of two float numbers */
inline constexpr srtb::real eps = 1e-5;

/**
 * @brief average the norm of input complex numbers
 *        xxxxxxxxx xxxxxxxxx......xxxxxxxxx
 *        |-------| |-------|      |-------|
 *          \   /     \   /          \   /
 *         \bar{x}   \bar{x}        \bar{x}
 */
template <typename T = srtb::real, typename C = srtb::complex<srtb::real>,
          typename DeviceInputAccessor = C*, typename DeviceOutputAccessor = T*>
void simplify_spectrum_calculate_norm(DeviceInputAccessor d_in, size_t in_count,
                                      DeviceOutputAccessor d_out,
                                      size_t out_count, size_t batch_size = 1,
                                      sycl::queue& q = srtb::queue) {
  static_assert(sizeof(T) * 2 == sizeof(C));
  constexpr auto norm = [=](srtb::complex<srtb::real> c) {
    return srtb::norm(c);
  };

  const srtb::real in_count_real = static_cast<srtb::real>(in_count);
  const srtb::real out_count_real = static_cast<srtb::real>(out_count);
  // count of in data point that average into one out data point
  const srtb::real coverage = in_count_real / out_count_real;

  q.parallel_for(sycl::range<1>{out_count * batch_size}, [=](sycl::item<1> id) {
     const size_t idx = id.get_id(0);
     const size_t j = idx / out_count;
     const size_t in_offset = j * in_count;
     const size_t out_offset = j * out_count;
     const size_t i = idx - out_offset;
     SRTB_ASSERT_IN_KERNEL(i < out_count && j < batch_size);
     SRTB_ASSERT_IN_KERNEL(j * out_count + i == idx);

     srtb::real sum = 0;
     const srtb::real left_accurate = coverage * i,
                      right_accurate = coverage * (i + 1),
                      left_real = sycl::ceil(left_accurate),
                      right_real = sycl::floor(right_accurate);
     const size_t left = static_cast<size_t>(left_real),
                  right = static_cast<size_t>(right_real);
     SRTB_ASSERT_IN_KERNEL(left_real >= left_accurate);
     if (left_real - left_accurate > eps) [[likely]] {
       const size_t left_left = left - 1;
       // left_left == static_cast<size_t>(sycl::floor(left_real)),
       const size_t left_right = left;
       SRTB_ASSERT_IN_KERNEL(left_left < in_count);
       sum += (left_right - left_accurate) * norm(d_in[left_left + in_offset]);
     }
     for (size_t k = left; k < right; k++) {
       SRTB_ASSERT_IN_KERNEL(k < in_count);
       sum += norm(d_in[k + in_offset]);
     }
     SRTB_ASSERT_IN_KERNEL(right_accurate >= right_real);
     if (right_accurate - right_real > eps) [[likely]] {
       const size_t right_left = right;
       // const size_t right_right = right + 1;
       // right_right == static_cast<size_t>(sycl::ceil(right_real));
       SRTB_ASSERT_IN_KERNEL(right_left < in_count);
       sum +=
           (right_accurate - right_left) * norm(d_in[right_left + in_offset]);
     }
     d_out[i + out_offset] = sum;
   }).wait();
}

/**
 * @brief average the norm of input complex numbers on frequency axis,
 *        linear interpolation on time axis
 * @note this variant only uses two nearest data points on time axis in input region
 *       so output spectrum isn't smooth.
 * 
 * schmantic:
 * 
 *  .----------------------------------------------------------------------> t
 *  | x \           / x ... x \           / x ... ... x \           / x
 *  | x  \         /  x ... x  \         /  x ... ... x  \         /  x
 *  | x    \bar{x}    x ... x    \bar{x}    x ... ... x    \bar{x}    x
 *  | x  /         \  x ... x  /         \  x ... ... x  /         \  x
 *  | x /           \ x ... x /           \ x ... ... x /           \ x
 *  |
 *  | x \           / x ... x \           / x ... ... x \           / x
 *  | x  \         /  x ... x  \         /  x ... ... x  \         /  x
 *  | x    \bar{x}    x ... x    \bar{x}    x ... ... x    \bar{x}    x
 *  | x  /         \  x ... x  /         \  x ... ... x  /         \  x
 *  | x /           \ x ... x /           \ x ... ... x /           \ x
 *  |
 *  | .               . ... .               . ... ... .               .
 *  | .               . ... .               . ... ... .               .
 *  | .               . ... .               . ... ... .               .
 *  |
 *  | x \           / x ... x \           / x ... ... x \           / x
 *  | x  \         /  x ... x  \         /  x ... ... x  \         /  x
 *  | x    \bar{x}    x ... x    \bar{x}    x ... ... x    \bar{x}    x
 *  | x  /         \  x ... x  /         \  x ... ... x  /         \  x
 *  | x /           \ x ... x /           \ x ... ... x /           \ x
 *  |
 *  v
 *  f
 * 
 * @tparam DeviceInputAccessor e.g. device pointer complex*
 * @tparam DeviceOutputAccessor e.g. device pointer real*
 * @param d_in iterator of input complex spectrum, with time as x-axis and frequency as y-axis
 * @param in_width count in x-axis (time) of d_in
 * @param in_height count in y-axis (frequency) of d_in
 * @param d_out iterator of output intensity (real), axis same as d_in
 * @param out_width count in x-axis (time) of d_out
 * @param out_height count in y-axis (frequency) of d_out
 * @param q SYCL queue to submit kernel
 */
template <typename DeviceInputAccessor, typename DeviceOutputAccessor>
inline void resample_spectrum_1(DeviceInputAccessor d_in, size_t in_width,
                                size_t in_height, DeviceOutputAccessor d_out,
                                size_t out_width, size_t out_height,
                                sycl::queue& q) {
  using T = typename std::iterator_traits<DeviceOutputAccessor>::value_type;
  const srtb::real in_width_real = static_cast<srtb::real>(in_width);
  const srtb::real in_height_real = static_cast<srtb::real>(in_height);
  const srtb::real out_width_real = static_cast<srtb::real>(out_width);
  const srtb::real out_height_real = static_cast<srtb::real>(out_height);

  q.parallel_for(sycl::range<1>{out_width * out_height}, [=](sycl::item<1> id) {
     // working for pixel (x2, y2) on output pixmap
     const size_t idx = id.get_id(0);
     const size_t y2 = idx / out_width;
     const size_t x2 = idx - y2 * out_width;
     SRTB_ASSERT_IN_KERNEL(x2 < out_width && y2 < out_height);
     SRTB_ASSERT_IN_KERNEL(y2 * out_width + x2 == idx);

     // correspond position on input spectrum is (x1, y1)
     const srtb::real x1 =
         static_cast<srtb::real>(x2) / out_width_real * in_width_real;
     /*
         on input spectrum, t axis:

         ----|--------|------> t
           left   ^ right
                  |
                 x_1

         ("|" means integer index position)
       */
     const srtb::real left_real = sycl::floor(x1);
     const srtb::real right_real = left_real + 1;
     const size_t left_int = static_cast<size_t>(left_real);
     const size_t right_int = left_int + 1;
     const srtb::real left_portion = right_real - x1;
     const srtb::real right_portion = x1 - left_real;
     SRTB_ASSERT_IN_KERNEL(left_portion >= 0);
     SRTB_ASSERT_IN_KERNEL(right_portion >= 0);

     const auto sample = [=](size_t y) {
       return left_portion * srtb::norm(d_in[y * in_width + left_int]) +
              right_portion * srtb::norm(d_in[y * in_width + right_int]);
     };

     T sum = 0;
     /*
        on input spectrum, f axis:

        <--- up   |---------------- sum ------------------|          down --->
          --|--------|--------|---- ... ----|--------|--------|------> f
            ^     ^  ^                               ^    ^   ^
            |     |  |                               |    |   |
            |   up_accurate                          |  down_accurate
          up_up      |                      down == down_up   |
                 up == up_down                            down_down

        ("|" means integer index position)
      */
     const srtb::real up_accurate = y2 / out_height_real * in_height_real;
     const srtb::real down_accurate =
         (y2 + 1) / out_height_real * in_height_real;
     const srtb::real up_real = sycl::ceil(up_accurate);
     const srtb::real down_real = sycl::floor(down_accurate);
     const size_t up_int = static_cast<size_t>(up_real),
                  down_int = static_cast<size_t>(down_real);

     SRTB_ASSERT_IN_KERNEL(up_real >= up_accurate);
     if (up_real > up_accurate) [[likely]] {
       const size_t up_up = up_int - 1;
       // up_up == static_cast<size_t>(sycl::floor(up_real)),
       // const size_t up_down = up_int;
       SRTB_ASSERT_IN_KERNEL(up_up < in_height);
       sum += (up_real - up_accurate) * sample(up_up);
     }

     for (size_t y = up_int; y < down_int; y++) {
       SRTB_ASSERT_IN_KERNEL(y < in_height);
       sum += sample(y);
     }

     SRTB_ASSERT_IN_KERNEL(down_accurate >= down_real);
     if (down_accurate > down_real) [[likely]] {
       const size_t down_up = down_int;
       // const size_t down_down = down_int + 1;
       // down_down == static_cast<size_t>(sycl::ceil(down_real));
       SRTB_ASSERT_IN_KERNEL(down_up < in_height);
       sum += (down_accurate - down_real) * sample(down_up);
     }

     d_out[idx] = sum;
   }).wait();
}

/**
 * @brief average the norm of input complex numbers on frequency axis and time axis
 * @note this variant also average values on time axis, but it is expensive
 * 
 * schmantic:
 * 
 *  .----------------------------------------------------------------------> t
 *  | x \ ......... / x ... x \ ......... / x ... ... x \ ......... / x
 *  | x  \ ....... /  x ... x  \ ....... /  x ... ... x  \ ....... /  x
 *  | x    \bar{x}    x ... x    \bar{x}    x ... ... x    \bar{x}    x
 *  | x  / ....... \  x ... x  / ....... \  x ... ... x  / ....... \  x
 *  | x / ......... \ x ... x / ......... \ x ... ... x / ......... \ x
 *  |
 *  | x \ ......... / x ... x \ ......... / x ... ... x \ ......... / x
 *  | x  \ ....... /  x ... x  \ ....... /  x ... ... x  \ ....... /  x
 *  | x    \bar{x}    x ... x    \bar{x}    x ... ... x    \bar{x}    x
 *  | x  / ....... \  x ... x  / ....... \  x ... ... x  / ....... \  x
 *  | x / ......... \ x ... x / ......... \ x ... ... x / ......... \ x
 *  |
 *  | .               . ... .               . ... ... .               .
 *  | .               . ... .               . ... ... .               .
 *  | .               . ... .               . ... ... .               .
 *  |
 *  | x \ ......... / x ... x \ ......... / x ... ... x \ ......... / x
 *  | x  \ ....... /  x ... x  \ ....... /  x ... ... x  \ ....... /  x
 *  | x    \bar{x}    x ... x    \bar{x}    x ... ... x    \bar{x}    x
 *  | x  / ....... \  x ... x  / ....... \  x ... ... x  / ....... \  x
 *  | x / ......... \ x ... x / ......... \ x ... ... x / ......... \ x
 *  |
 *  v
 *  f
 * 
 * @tparam DeviceInputAccessor e.g. device pointer complex*
 * @tparam DeviceOutputAccessor e.g. device pointer real*
 * @param d_in iterator of input complex spectrum, with time as x-axis and frequency as y-axis
 * @param in_width count in x-axis (time) of d_in
 * @param in_height count in y-axis (frequency) of d_in
 * @param d_out iterator of output intensity (real), axis same as d_in
 * @param out_width count in x-axis (time) of d_out
 * @param out_height count in y-axis (frequency) of d_out
 * @param q SYCL queue to submit kernel
 */
template <typename DeviceInputAccessor, typename DeviceOutputAccessor>
inline void resample_spectrum_2(DeviceInputAccessor d_in, size_t in_width,
                                size_t in_height, DeviceOutputAccessor d_out,
                                size_t out_width, size_t out_height,
                                sycl::queue& q) {
  using T = typename std::iterator_traits<DeviceOutputAccessor>::value_type;
  const srtb::real in_width_real = static_cast<srtb::real>(in_width);
  const srtb::real in_height_real = static_cast<srtb::real>(in_height);
  const srtb::real out_width_real = static_cast<srtb::real>(out_width);
  const srtb::real out_height_real = static_cast<srtb::real>(out_height);

  q.parallel_for(sycl::range<1>{out_width * out_height}, [=](sycl::item<1> id) {
     // working for pixel (x2, y2) on output pixmap
     const size_t idx = id.get_id(0);
     const size_t y2 = idx / out_width;
     const size_t x2 = idx - y2 * out_width;
     SRTB_ASSERT_IN_KERNEL(x2 < out_width && y2 < out_height);
     SRTB_ASSERT_IN_KERNEL(y2 * out_width + x2 == idx);

     const srtb::real left_accurate = x2 / out_width_real * in_width_real;
     const srtb::real right_accurate =
         (x2 + 1) / out_width_real * in_width_real;
     const srtb::real left_real = sycl::ceil(left_accurate);
     const srtb::real right_real = sycl::floor(right_accurate);
     const size_t left_int = static_cast<size_t>(left_real);
     const size_t right_int = static_cast<size_t>(right_real);

     const auto sample = [=](size_t y) {
       T sum = 0;
       SRTB_ASSERT_IN_KERNEL(left_real >= left_accurate);
       if (left_real > left_accurate) [[likely]] {
         const size_t left_left = left_int - 1;
         // left_left == static_cast<size_t>(sycl::floor(left_real)),
         // const size_t left_right = left;
         SRTB_ASSERT_IN_KERNEL(left_left < in_width);
         sum += (left_real - left_accurate) *
                srtb::norm(d_in[y * in_width + left_left]);
       }
       for (size_t x = left_int; x < right_int; x++) {
         SRTB_ASSERT_IN_KERNEL(x < in_width);
         sum += srtb::norm(d_in[y * in_width + x]);
       }
       SRTB_ASSERT_IN_KERNEL(right_accurate >= right_real);
       if (right_accurate > right_real) [[likely]] {
         const size_t right_left = right_int;
         // const size_t right_right = right + 1;
         // right_right == static_cast<size_t>(sycl::ceil(right_real));
         SRTB_ASSERT_IN_KERNEL(right_left < in_width);
         sum += (right_accurate - right_left) *
                srtb::norm(d_in[y * in_width + right_left]);
       }
       return sum;
     };

     T sum = 0;
     /*
        on input spectrum, f axis:

        <--- up   |---------------- sum ------------------|          down --->
          --|--------|--------|---- ... ----|--------|--------|------> f
            ^     ^  ^                               ^    ^   ^
            |     |  |                               |    |   |
            |   up_accurate                          |  down_accurate
          up_up      |                      down == down_up   |
                 up == up_down                            down_down

        ("|" means integer index position)
      */
     const srtb::real up_accurate = y2 / out_height_real * in_height_real;
     const srtb::real down_accurate =
         (y2 + 1) / out_height_real * in_height_real;
     const srtb::real up_real = sycl::ceil(up_accurate);
     const srtb::real down_real = sycl::floor(down_accurate);
     const size_t up_int = static_cast<size_t>(up_real),
                  down_int = static_cast<size_t>(down_real);

     SRTB_ASSERT_IN_KERNEL(up_real >= up_accurate);
     if (up_real > up_accurate) [[likely]] {
       const size_t up_up = up_int - 1;
       // up_up == static_cast<size_t>(sycl::floor(up_real)),
       // const size_t up_down = up_int;
       SRTB_ASSERT_IN_KERNEL(up_up < in_height);
       sum += (up_real - up_accurate) * sample(up_up);
     }

     for (size_t y = up_int; y < down_int; y++) {
       SRTB_ASSERT_IN_KERNEL(y < in_height);
       sum += sample(y);
     }

     SRTB_ASSERT_IN_KERNEL(down_accurate >= down_real);
     if (down_accurate > down_real) [[likely]] {
       const size_t down_up = down_int;
       // const size_t down_down = down_int + 1;
       // down_down == static_cast<size_t>(sycl::ceil(down_real));
       SRTB_ASSERT_IN_KERNEL(down_up < in_height);
       sum += (down_accurate - down_real) * sample(down_up);
     }

     d_out[idx] = sum;
   }).wait();
}

/**
 * @brief average the norm of input complex numbers on frequency axis and time axis
 * @note this variant is based on resample_spectrum_2, thus also average values on time axis;
 *       but a work group for an output pixel instead of a work item, 
 *       which is intended for improving memory access pattern.
 * 
 * schmantic:
 * 
 *  .----------------------------------------------------------------------> t
 *  | x \ ......... / x ... x \ ......... / x ... ... x \ ......... / x
 *  | x  \ ....... /  x ... x  \ ....... /  x ... ... x  \ ....... /  x
 *  | x    \bar{x}    x ... x    \bar{x}    x ... ... x    \bar{x}    x
 *  | x  / ....... \  x ... x  / ....... \  x ... ... x  / ....... \  x
 *  | x / ......... \ x ... x / ......... \ x ... ... x / ......... \ x
 *  |
 *  | x \ ......... / x ... x \ ......... / x ... ... x \ ......... / x
 *  | x  \ ....... /  x ... x  \ ....... /  x ... ... x  \ ....... /  x
 *  | x    \bar{x}    x ... x    \bar{x}    x ... ... x    \bar{x}    x
 *  | x  / ....... \  x ... x  / ....... \  x ... ... x  / ....... \  x
 *  | x / ......... \ x ... x / ......... \ x ... ... x / ......... \ x
 *  |
 *  | .               . ... .               . ... ... .               .
 *  | .               . ... .               . ... ... .               .
 *  | .               . ... .               . ... ... .               .
 *  |
 *  | x \ ......... / x ... x \ ......... / x ... ... x \ ......... / x
 *  | x  \ ....... /  x ... x  \ ....... /  x ... ... x  \ ....... /  x
 *  | x    \bar{x}    x ... x    \bar{x}    x ... ... x    \bar{x}    x
 *  | x  / ....... \  x ... x  / ....... \  x ... ... x  / ....... \  x
 *  | x / ......... \ x ... x / ......... \ x ... ... x / ......... \ x
 *  |
 *  v
 *  f
 * 
 * @tparam DeviceInputAccessor e.g. device pointer complex*
 * @tparam DeviceOutputAccessor e.g. device pointer real*
 * @param d_in iterator of input complex spectrum, with time as x-axis and frequency as y-axis
 * @param in_width count in x-axis (time) of d_in
 * @param in_height count in y-axis (frequency) of d_in
 * @param d_out iterator of output intensity (real), axis same as d_in
 * @param out_width count in x-axis (time) of d_out
 * @param out_height count in y-axis (frequency) of d_out
 * @param q SYCL queue to submit kernel
 */
template <typename DeviceInputAccessor, typename DeviceOutputAccessor>
inline void resample_spectrum_3(DeviceInputAccessor d_in, size_t in_width,
                                size_t in_height, DeviceOutputAccessor d_out,
                                size_t out_width, size_t out_height,
                                sycl::queue& q) {
  using T = typename std::iterator_traits<DeviceOutputAccessor>::value_type;
  const srtb::real in_width_real = static_cast<srtb::real>(in_width);
  const srtb::real in_height_real = static_cast<srtb::real>(in_height);
  const srtb::real out_width_real = static_cast<srtb::real>(out_width);
  const srtb::real out_height_real = static_cast<srtb::real>(out_height);

  const size_t max_work_item_required =
      std::max(static_cast<size_t>(std::round(in_width_real / out_width_real)),
               size_t{1});
  // TODO: check if this is really optimal
  // Currently available info:
  //   NVIDIA GPUs have wrap size 32.
  //   AMD GPUs with GCN / CDNA? architecture have wave front size 64,
  //            with RDNA architecture have 32.
  //   [DATA EXPUNGED] 1st gen have wave front size 128?,
  //                   2nd gen have ... 32?
  //   amd64 CPUs with AVX512 -> 64 bytes
  // benchmark on AMD Radeon VII:
  //   auto: 31ms
  //   128: 25.5 25.9 25.7
  //   64: 16.637 16.971 16.496
  //   32: 19.889 19.401 (???)
  const size_t max_work_item_optimal = 64;

  // modified on the basis of bufffer_algorihms.hpp in SyclParallelSTL
  sycl::device device = q.get_device();
  const sycl::id<1> max_work_item_size_1 =
      device.get_info<sycl::info::device::max_work_item_sizes<1> >();
  const size_t max_work_item =
      std::min(device.get_info<sycl::info::device::max_work_group_size>(),
               max_work_item_size_1[0]);
  const size_t local_mem_size =
      device.get_info<sycl::info::device::local_mem_size>();
  constexpr size_t sizeofB = sizeof(T);
  const size_t nb_work_item =
      std::min(std::min(std::min(max_work_item, local_mem_size / sizeofB),
                        max_work_item_required),
               max_work_item_optimal);
  const size_t reduce_size = std::bit_ceil(nb_work_item) >> 1;
  SRTB_LOGD << " [resample_spectrum_3] "
            << "nb_work_item = " << nb_work_item << srtb::endl;

  q.submit([&](sycl::handler& cgh) {
     sycl::range<1> range_group{out_width * out_height};
     sycl::range<1> range_item{nb_work_item};
     // local memory area for partial sums
     sycl::accessor<T, /* dimensions = */ 1, sycl::access::mode::read_write,
                    sycl::access::target::local>
         sum{sycl::range<1>(nb_work_item), cgh, sycl::no_init};
     cgh.parallel_for(
         sycl::nd_range<1>(range_group * range_item, range_item),
         [=](sycl::nd_item<1> nd_item) {
           // this group / block works for pixel (x2, y2) on output pixmap
           const size_t group_idx = nd_item.get_group(0);
           const size_t y2 = group_idx / out_width;
           const size_t x2 = group_idx - y2 * out_width;
           SRTB_ASSERT_IN_KERNEL(x2 < out_width && y2 < out_height);
           SRTB_ASSERT_IN_KERNEL(y2 * out_width + x2 == group_idx);
           SRTB_ASSERT_IN_KERNEL(nd_item.get_local_range(0) == nb_work_item);
           const size_t local_idx = nd_item.get_local_id(0);

           // t axis: average
           /*
             on input spectrum, t axis:
 
             <--- left |---------------- sum ------------------|         right --->
               --|--------|--------|---- ... ----|--------|--------|------> f
                 ^     ^  ^                               ^    ^   ^
                 |     |  |                               |    |   |
                 | left_accurate                          |  right_accurate
             left_left    |                 right == right_left    |
                     left == left_right                       right_right
 
             ("|" means integer index position)
           */
           const srtb::real left_accurate = x2 / out_width_real * in_width_real;
           const srtb::real right_accurate =
               (x2 + 1) / out_width_real * in_width_real;
           const srtb::real left_real = sycl::ceil(left_accurate);
           const srtb::real right_real = sycl::floor(right_accurate);
           const size_t left_int = static_cast<size_t>(left_real);
           const size_t right_int = static_cast<size_t>(right_real);

           // instead of plain for-loop iteration over all input spectrum pixels
           // to form output one, a parallelization of size nb_work_item is used
           // here to improve efficiency
           const auto sample = [=](size_t y) {
             T partial_x_sum = 0;
             if (local_idx == 0) {
               SRTB_ASSERT_IN_KERNEL(left_real >= left_accurate);
               if (left_real > left_accurate) [[likely]] {
                 const size_t left_left = left_int - 1;
                 // left_left == static_cast<size_t>(sycl::floor(left_real)),
                 // const size_t left_right = left;
                 SRTB_ASSERT_IN_KERNEL(left_left < in_width);
                 partial_x_sum += (left_real - left_accurate) *
                                  srtb::norm(d_in[y * in_width + left_left]);
               }
             }
             for (size_t x = left_int + local_idx; x < right_int;
                  x += nb_work_item) {
               SRTB_ASSERT_IN_KERNEL(x < in_width);
               partial_x_sum += srtb::norm(d_in[y * in_width + x]);
             }
             if (local_idx == nb_work_item - 1) {
               SRTB_ASSERT_IN_KERNEL(right_accurate >= right_real);
               if (right_accurate > right_real) [[likely]] {
                 const size_t right_left = right_int;
                 // const size_t right_right = right + 1;
                 // right_right == static_cast<size_t>(sycl::ceil(right_real));
                 SRTB_ASSERT_IN_KERNEL(right_left < in_width);
                 partial_x_sum += (right_accurate - right_left) *
                                  srtb::norm(d_in[y * in_width + right_left]);
               }
             }
             return partial_x_sum;
           };

           srtb::real y_sum = 0;
           /*
            on input spectrum, f axis:

            <--- up   |---------------- sum ------------------|          down --->
              --|--------|--------|---- ... ----|--------|--------|------> f
                ^     ^  ^                               ^    ^   ^
                |     |  |                               |    |   |
                |   up_accurate                          |  down_accurate
              up_up      |                      down == down_up   |
                     up == up_down                            down_down

            ("|" means integer index position)
          */
           const srtb::real up_accurate = y2 / out_height_real * in_height_real;
           const srtb::real down_accurate =
               (y2 + 1) / out_height_real * in_height_real;
           const srtb::real up_real = sycl::ceil(up_accurate);
           const srtb::real down_real = sycl::floor(down_accurate);
           const size_t up_int = static_cast<size_t>(up_real),
                        down_int = static_cast<size_t>(down_real);

           SRTB_ASSERT_IN_KERNEL(up_real >= up_accurate);
           if (up_real > up_accurate) [[likely]] {
             const size_t up_up = up_int - 1;
             // up_up == static_cast<size_t>(sycl::floor(up_real)),
             // const size_t up_down = up_int;
             SRTB_ASSERT_IN_KERNEL(up_up < in_height);
             y_sum += (up_real - up_accurate) * sample(up_up);
           }

           for (size_t y = up_int; y < down_int; y++) {
             SRTB_ASSERT_IN_KERNEL(y < in_height);
             y_sum += sample(y);
           }

           SRTB_ASSERT_IN_KERNEL(down_accurate >= down_real);
           if (down_accurate > down_real) [[likely]] {
             const size_t down_up = down_int;
             // const size_t down_down = down_int + 1;
             // down_down == static_cast<size_t>(sycl::ceil(down_real));
             SRTB_ASSERT_IN_KERNEL(down_up < in_height);
             y_sum += (down_accurate - down_real) * sample(down_up);
           }
           sum[local_idx] = y_sum;

           nd_item.barrier(sycl::access::fence_space::local_space);

           // sum over work items, using reduce
           for (size_t offset = reduce_size; offset > 0; offset >>= 1) {
             if (local_idx + offset < nb_work_item) {
               sum[local_idx] += sum[local_idx + offset];
             }
             nd_item.barrier(sycl::access::fence_space::local_space);
           }
           if (local_idx == 0) {
             d_out[group_idx] = sum[0];
           }
         });
   }).wait();
}

/**
 * @brief normalize ( intended to scale values to [0, 1] )
 *        using average value in data.
 * @return average value
 */
template <typename T = srtb::real, typename DeviceInputAccessor = T*>
auto simplify_spectrum_normalize_with_average_value(
    DeviceInputAccessor d_in, size_t in_count, sycl::queue& q = srtb::queue)
    -> T {
  auto d_avg_val_shared = srtb::algorithm::map_average(
      d_in, in_count, srtb::algorithm::map_identity(), q);
  auto d_avg_val = d_avg_val_shared.get();
  T h_avg_val;
  q.copy(d_avg_val, /* -> */ &h_avg_val, 1).wait();
  const T coeff = T{1.0} / (h_avg_val * 2);
  if (h_avg_val > std::numeric_limits<T>::epsilon()) [[likely]] {
    q.parallel_for(sycl::range<1>{in_count}, [=](sycl::item<1> id) {
       const size_t i = id.get_id(0);
       d_in[i] *= coeff;
     }).wait();
  }
  return h_avg_val;
}

/**
 * @brief Extract components of uint32_t represent of ARGB
 * 
 * @param argb the value to be extracted
 * @return extracted chars, alpha, red, green, blue respectively
 */
inline constexpr auto getARGB(uint32_t argb)
    -> std::tuple<unsigned char, unsigned char, unsigned char, unsigned char> {
  unsigned char a = static_cast<unsigned char>((argb & 0xff000000) >> 24);
  unsigned char r = static_cast<unsigned char>((argb & 0x00ff0000) >> 16);
  unsigned char g = static_cast<unsigned char>((argb & 0x0000ff00) >> 8);
  unsigned char b = static_cast<unsigned char>((argb & 0x000000ff));
  return std::tuple{a, r, g, b};
}

/**
 * @brief combine A, R, G, B components into uint32_t represent of ARGB32
 * 
 * @param a value of the alpha channel
 * @param r value of the red channel
 * @param g value of the green channel
 * @param b value of the blue channel
 * @return uint32_t represent of ARGB32
 */
inline constexpr auto getARGB(unsigned char a, unsigned char r, unsigned char g,
                              unsigned char b) -> uint32_t {
  return (static_cast<uint32_t>(a) << 24) | (static_cast<uint32_t>(r) << 16) |
         (static_cast<uint32_t>(g) << 8) | (static_cast<uint32_t>(b));
}

/**
 * @brief This function maps input intensity (range [0, 1]) 
 *        to ARGB32 colors on pixmap (range argb_1 ~ argb_2, or argb_error if out of range)
 * 
 * @tparam InputIterator type of d_in, accessor on device, e.g. srtb::real*
 * @tparam OutputIterator type of d_out, accessor on device, e.g. uint32_t*
 * @tparam std::enable_if<
 *           std::is_same<
 *             typename std::decay<
 *               typename std::iterator_traits<OutputIterator>::value_type
 *             >::type, uint32_t
 *           >::value, void
 *         >::type 
 *         this function is enabled only if output type is uint32_t, which can represent ARGB32
 * 
 * @param d_in input intensity
 * @param d_out output pixmap
 * @param width width of both input and output 2D array
 * @param height height of both input and output array
 * @param argb_1 start of color range
 * @param argb_2 end of color range
 * @param argb_error color if input out of range
 * @param q the SYCL queue which to submit kernel to
 */
template <typename InputIterator, typename OutputIterator,
          typename = typename std::enable_if<
              std::is_same<typename std::decay<typename std::iterator_traits<
                               OutputIterator>::value_type>::type,
                           uint32_t>::value,
              void>::type>
inline void generate_pixmap(InputIterator d_in, OutputIterator d_out,
                            size_t width, size_t height, uint32_t argb_1,
                            uint32_t argb_2, uint32_t argb_error,
                            sycl::queue& q) {
  using T = typename std::iterator_traits<InputIterator>::value_type;
  const size_t total_count = width * height;
  const auto [a1, r1, g1, b1] = getARGB(argb_1);
  const auto [a2, r2, g2, b2] = getARGB(argb_2);
  const T a1f = a1, r1f = r1, g1f = g1, b1f = b1;
  const T a2f = a2, r2f = r2, g2f = g2, b2f = b2;

  q.parallel_for(sycl::range<1>{total_count}, [=](sycl::item<1> id) {
     const size_t i = id.get_id(0);
     const T in = d_in[i];
     uint32_t out;
     constexpr T zero = 0, one = 1;
     if (zero <= in && in <= one) {
       out = getARGB((one - in) * a1f + in * a2f, (one - in) * r1f + in * r2f,
                     (one - in) * g1f + in * g2f, (one - in) * b1f + in * b2f);
     } else {
       out = argb_error;
     }
     d_out[i] = out;
   }).wait();
}

}  // namespace spectrum
}  // namespace srtb

#endif  // __SRTB_SIMPLIFY_SPECTRUM__
