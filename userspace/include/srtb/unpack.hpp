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
          typename OutputIterator, typename TransformFunctor>
inline typename std::enable_if<(handwritten == false), void>::type unpack_item(
    InputIterator in, OutputIterator out, const size_t x,
    TransformFunctor transform) {
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
    out[out_offset + i] = transform(out_offset + i, static_cast<T>(out_val));
  }
}

template <int IN_NBITS, bool handwritten, typename InputIterator,
          typename OutputIterator, typename TransformFunctor>
inline
    typename std::enable_if<(IN_NBITS == 1 && handwritten == true), void>::type
    unpack_item(InputIterator in, OutputIterator out, const size_t x,
                TransformFunctor transform) {
  static_assert(
      std::is_same_v<std::byte,
                     typename std::iterator_traits<InputIterator>::value_type>);
  typedef typename std::iterator_traits<OutputIterator>::value_type T;

  const std::byte in_val = in[x];
  const size_t offset = 8 * x;
  // clang-format off
  out[offset + 0] = transform(offset + 0, static_cast<T>((in_val & std::byte{0b10000000}) >> 7));
  out[offset + 1] = transform(offset + 1, static_cast<T>((in_val & std::byte{0b01000000}) >> 6));
  out[offset + 2] = transform(offset + 2, static_cast<T>((in_val & std::byte{0b00100000}) >> 5));
  out[offset + 3] = transform(offset + 3, static_cast<T>((in_val & std::byte{0b00010000}) >> 4));
  out[offset + 4] = transform(offset + 4, static_cast<T>((in_val & std::byte{0b00001000}) >> 3));
  out[offset + 5] = transform(offset + 5, static_cast<T>((in_val & std::byte{0b00000100}) >> 2));
  out[offset + 6] = transform(offset + 6, static_cast<T>((in_val & std::byte{0b00000010}) >> 1));
  out[offset + 7] = transform(offset + 7, static_cast<T>((in_val & std::byte{0b00000001}) >> 0));
  // clang-format on
}

template <int IN_NBITS, bool handwritten, typename InputIterator,
          typename OutputIterator, typename TransformFunctor>
inline
    typename std::enable_if<(IN_NBITS == 2 && handwritten == true), void>::type
    unpack_item(InputIterator in, OutputIterator out, const size_t x,
                TransformFunctor transform) {
  static_assert(
      std::is_same_v<std::byte,
                     typename std::iterator_traits<InputIterator>::value_type>);
  typedef typename std::iterator_traits<OutputIterator>::value_type T;

  const std::byte in_val = in[x];
  const size_t offset = 4 * x;
  // clang-format off
  out[offset + 0] = transform(offset + 0, static_cast<T>((in_val & std::byte{0b11000000}) >> 6));
  out[offset + 1] = transform(offset + 1, static_cast<T>((in_val & std::byte{0b00110000}) >> 4));
  out[offset + 2] = transform(offset + 2, static_cast<T>((in_val & std::byte{0b00001100}) >> 2));
  out[offset + 3] = transform(offset + 3, static_cast<T>((in_val & std::byte{0b00000011}) >> 0));
  // clang-format on
}

template <int IN_NBITS, bool handwritten, typename InputIterator,
          typename OutputIterator, typename TransformFunctor>
inline
    typename std::enable_if<(IN_NBITS == 4 && handwritten == true), void>::type
    unpack_item(InputIterator in, OutputIterator out, const size_t x,
                TransformFunctor transform) {
  static_assert(
      std::is_same_v<std::byte,
                     typename std::iterator_traits<InputIterator>::value_type>);
  typedef typename std::iterator_traits<OutputIterator>::value_type T;

  const std::byte in_val = in[x];
  const size_t offset = 2 * x;
  // clang-format off
  out[offset + 0] = transform(offset + 0, static_cast<T>((in_val & std::byte{0b11110000}) >> 4));
  out[offset + 1] = transform(offset + 1, static_cast<T>((in_val & std::byte{0b00001111}) >> 0));
  // clang-format on
}

template <int IN_NBITS, bool handwritten, typename InputIterator,
          typename OutputIterator, typename TransformFunctor>
inline
    typename std::enable_if<(IN_NBITS == 8 && handwritten == true), void>::type
    unpack_item(InputIterator in, OutputIterator out, const size_t x,
                TransformFunctor transform) {
  static_assert(
      std::is_same_v<std::byte,
                     typename std::iterator_traits<InputIterator>::value_type>);
  typedef typename std::iterator_traits<OutputIterator>::value_type T;

  const std::byte in_val = in[x];
  out[x] = transform(x, static_cast<T>(in_val));
}

/**
 * @brief unpack bytes stream into floating-point numbers, for FFT
 * 
 * @tparam IN_NBITS bit width of one input number
 * @param d_in iterator of std::byte
 * @param d_out iterator of output
 * @param in_count std::bytes count of in. Make sure [0, BITS_PER_BYTE / IN_NBITS * input_count) of out is accessible.
 * @param transform transform transformtor to be applied after unpacking, e.g. FFT window.
 *             it's operator() has the signature (size_t n, T val) -> T
 */
template <int IN_NBITS, bool handwritten = false, typename InputIterator,
          typename OutputIterator, typename TransformFunctor>
inline void unpack(InputIterator d_in, OutputIterator d_out, size_t in_count,
                   TransformFunctor transform, sycl::queue& q) {
  q.parallel_for(sycl::range<1>(in_count), [=](sycl::item<1> id) {
     unpack_item<IN_NBITS, handwritten>(d_in, d_out, id.get_id(0), transform);
   }).wait();
}

struct identity : std::identity {
  template <typename T>
  [[nodiscard]] constexpr T&& operator()(size_t n, T&& x) const noexcept {
    (void)n;
    return std::identity::operator()<T>(std::move(x));
  }
};

template <int IN_NBITS, bool handwritten = false, typename InputIterator,
          typename OutputIterator>
inline void unpack(InputIterator d_in, OutputIterator d_out, size_t in_count,
                   sycl::queue& q) {
  return unpack<IN_NBITS, handwritten>(d_in, d_out, in_count,
                                       srtb::unpack::identity(), q);
}

template <bool handwritten = false, typename InputIterator,
          typename OutputIterator, typename TransformFunctor>
inline void unpack(int in_nbits, InputIterator d_in, OutputIterator d_out,
                   size_t in_count, TransformFunctor transform,
                   sycl::queue& q) {
  switch (in_nbits) {
    case 1:
      return unpack<1, handwritten>(d_in, d_out, in_count, transform, q);
    case 2:
      return unpack<2, handwritten>(d_in, d_out, in_count, transform, q);
    case 4:
      return unpack<4, handwritten>(d_in, d_out, in_count, transform, q);
    case 8:
      return unpack<8, handwritten>(d_in, d_out, in_count, transform, q);
    default:
      throw std::runtime_error("unpack: unsupported in_nbits " +
                               std::to_string(in_nbits));
  }
}

template <bool handwritten = false, typename InputIterator,
          typename OutputIterator>
inline void unpack(int in_nbits, InputIterator d_in, OutputIterator d_out,
                   size_t in_count, sycl::queue& q) {
  return unpack<handwritten>(in_nbits, d_in, d_out, in_count,
                             srtb::unpack::identity(), q);
}

}  // namespace unpack
}  // namespace srtb

#endif  // __SRTB_UNPACK__
