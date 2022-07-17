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
#ifndef __SRTB_UNPACK__
#define __SRTB_UNPACK__

#include <concepts>
#include <cstddef>
#include <iterator>
#include <type_traits>

#include "srtb/commons.hpp"

namespace srtb {
namespace unpack {

/**
 * @brief unpack in[x] to out[BITS_PER_BYTE / IN_NBITS * x]
 * 
 * @see srtb::unpack::unpack
 * TODO: does these the same after optimization?
 */
template <int IN_NBITS, bool handwritten, typename InputIterator,
          typename OutputIterator>
inline typename std::enable_if<(handwritten == false), void>::type unpack_item(
    InputIterator in, OutputIterator out, const size_t x) {
  static_assert(
      std::is_same_v<std::byte,
                     typename std::iterator_traits<InputIterator>::value_type>);
  typedef typename std::iterator_traits<OutputIterator>::value_type T;
  constexpr int count = BITS_PER_BYTE / IN_NBITS;  // count of numbers in a byte

  std::byte mask{0};

#pragma unroll
  for (int i = 0; i < IN_NBITS; i++) {
    mask <<= 1;
    mask |= std::byte{1};
  }

  mask <<= (BITS_PER_BYTE - IN_NBITS);

  const size_t out_offset = count * x;
  const std::byte in_val = in[x];

#pragma unroll
  for (int i = 0; i < count; i++) {
    const std::byte val_mask = mask >> (i * IN_NBITS);
    const std::byte out_val =
        ((in_val & val_mask) >> ((count - i - 1) * IN_NBITS));
    out[out_offset + i] = static_cast<T>(out_val);
  }
}

template <int IN_NBITS, bool handwritten, typename InputIterator,
          typename OutputIterator>
inline
    typename std::enable_if<(IN_NBITS == 1 && handwritten == true), void>::type
    unpack_item(InputIterator in, OutputIterator out, const size_t x) {
  static_assert(
      std::is_same_v<std::byte,
                     typename std::iterator_traits<InputIterator>::value_type>);
  typedef typename std::iterator_traits<OutputIterator>::value_type T;

  const std::byte in_val = in[x];
  out[8 * x + 0] = static_cast<T>((in_val & std::byte{0b10000000}) >> 7);
  out[8 * x + 1] = static_cast<T>((in_val & std::byte{0b01000000}) >> 6);
  out[8 * x + 2] = static_cast<T>((in_val & std::byte{0b00100000}) >> 5);
  out[8 * x + 3] = static_cast<T>((in_val & std::byte{0b00010000}) >> 4);
  out[8 * x + 4] = static_cast<T>((in_val & std::byte{0b00001000}) >> 3);
  out[8 * x + 5] = static_cast<T>((in_val & std::byte{0b00000100}) >> 2);
  out[8 * x + 6] = static_cast<T>((in_val & std::byte{0b00000010}) >> 1);
  out[8 * x + 7] = static_cast<T>((in_val & std::byte{0b00000001}) >> 0);
}

template <int IN_NBITS, bool handwritten, typename InputIterator,
          typename OutputIterator>
inline
    typename std::enable_if<(IN_NBITS == 2 && handwritten == true), void>::type
    unpack_item(InputIterator in, OutputIterator out, const size_t x) {
  static_assert(
      std::is_same_v<std::byte,
                     typename std::iterator_traits<InputIterator>::value_type>);
  typedef typename std::iterator_traits<OutputIterator>::value_type T;

  const std::byte in_val = in[x];
  out[4 * x + 0] = static_cast<T>((in_val & std::byte{0b11000000}) >> 6);
  out[4 * x + 1] = static_cast<T>((in_val & std::byte{0b00110000}) >> 4);
  out[4 * x + 2] = static_cast<T>((in_val & std::byte{0b00001100}) >> 2);
  out[4 * x + 3] = static_cast<T>((in_val & std::byte{0b00000011}) >> 0);
}

template <int IN_NBITS, bool handwritten, typename InputIterator,
          typename OutputIterator>
inline
    typename std::enable_if<(IN_NBITS == 4 && handwritten == true), void>::type
    unpack_item(InputIterator in, OutputIterator out, const size_t x) {
  static_assert(
      std::is_same_v<std::byte,
                     typename std::iterator_traits<InputIterator>::value_type>);
  typedef typename std::iterator_traits<OutputIterator>::value_type T;

  const std::byte in_val = in[x];
  out[2 * x + 0] = static_cast<T>((in_val & std::byte{0b11110000}) >> 4);
  out[2 * x + 1] = static_cast<T>((in_val & std::byte{0b00001111}) >> 0);
}

template <int IN_NBITS, bool handwritten, typename InputIterator,
          typename OutputIterator>
inline
    typename std::enable_if<(IN_NBITS == 8 && handwritten == true), void>::type
    unpack_item(InputIterator in, OutputIterator out, const size_t x) {
  static_assert(
      std::is_same_v<std::byte,
                     typename std::iterator_traits<InputIterator>::value_type>);
  typedef typename std::iterator_traits<OutputIterator>::value_type T;

  const std::byte in_val = in[x];
  out[x] = static_cast<T>(in_val);
}

/**
 * @brief unpack bytes stream into floating-point numbers, for FFT
 * 
 * @tparam IN_NBITS bit width of one input number
 * @param d_in iterator of std::byte
 * @param d_out iterator of output
 * @param in_count std::bytes count of in. Make sure [0, BITS_PER_BYTE / IN_NBITS * input_count) of out is accessible.
 */
template <int IN_NBITS, bool handwritten = false, typename InputIterator,
          typename OutputIterator>
inline void unpack(InputIterator d_in, OutputIterator d_out, size_t in_count,
                   sycl::queue& q) {
  q.parallel_for(sycl::range<1>(in_count), [=](sycl::item<1> id) {
     unpack_item<IN_NBITS, handwritten>(d_in, d_out, id.get_id(0));
   }).wait();
}

}  // namespace unpack
}  // namespace srtb

#endif  // __SRTB_UNPACK__
