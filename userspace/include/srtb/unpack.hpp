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
// -- divide line for clang-format --
#include "srtb/algorithm/map_identity.hpp"

namespace srtb {
/**
 * @brief unpack/type cast input to output.
 *        if input is std::byte and in_nbits = 1, 2, 4 then an "unpack" is performed,
 *        otherwise only type cast is done.
 * @note Additionally, @c transform functor is for kernel fusion, e.g. apply FFT window in unpack stage.
 */
namespace unpack {

/**
 * @brief unpack in[x] to out[BITS_PER_BYTE / IN_NBITS * x]
 *        enabled if IN_NBITS < 8, that is, really needs *unpack*.
 * 
 * @see srtb::unpack::unpack
 * TODO: are these the same after optimization?
 */
template <int IN_NBITS, bool handwritten, typename InputIterator,
          typename OutputIterator, typename TransformFunctor>
inline typename std::enable_if<
    (IN_NBITS < srtb::BITS_PER_BYTE && handwritten == false), void>::type
unpack_item(InputIterator in, OutputIterator out, const size_t x,
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

/** @brief not a unpack, just type cast and transform. */
template <int IN_NBITS, bool handwritten = false, typename InputIterator,
          typename OutputIterator, typename TransformFunctor>
inline
    typename std::enable_if<(IN_NBITS == sizeof(typename std::iterator_traits<
                                                InputIterator>::value_type) *
                                             srtb::BITS_PER_BYTE),
                            void>::type
    unpack_item(InputIterator in, OutputIterator out, const size_t x,
                TransformFunctor transform) {
  typedef typename std::iterator_traits<OutputIterator>::value_type out_type;

  const auto in_val = in[x];
  out[x] = transform(x, static_cast<out_type>(in_val));
}

/**
 * @brief unpack bytes stream into type needed. (for IN_BITS == 1, 2, 4)
 *        copy and cast input to needed type. (for IN_BITS >= 8)
 * 
 * @tparam IN_NBITS bit width of one input number
 * @param d_in iterator of std::byte ( for IN_NBITS == 1, 2, 4 )
 *                      or other ( for IN_NBITS == sizeof(input_type) )
 * @param d_out iterator of output
 * @param out_count count of output samples. Make sure [0, out_count) of out is accessible.
 * @param transform transform functor to be applied after unpacking, e.g. FFT window.
 *                  it's operator() has the signature (size_t n, T val) -> T
 * @param q the sycl queue to be used
 */
template <int IN_NBITS, bool handwritten = false, typename InputIterator,
          typename OutputIterator, typename TransformFunctor>
inline void unpack(InputIterator d_in, OutputIterator d_out,
                   const size_t out_count, TransformFunctor transform,
                   sycl::queue& q) {
  typedef typename std::iterator_traits<InputIterator>::value_type input_type;
  static_assert((std::is_same_v<std::byte, input_type> &&
                 IN_NBITS < srtb::BITS_PER_BYTE * sizeof(std::byte)) ||
                (srtb::BITS_PER_BYTE * sizeof(input_type) == IN_NBITS));

  size_t range_size;
  // here the parallel_for range differs.
  // 1) when in_nbits == 1, 2, 4, 8, one work item operates on 1 input std::byte,
  //    which write 8, 4, 2, 1 output(s), respectively
  // 2) when in_nbits == 8, 16, 32, 64 or whatever else,
  //    1 work item operates on only 1 input of some other type,
  //    which is not "unpack" but type casting to output type.
  // refer to device kernels above to see the difference.
  if constexpr (IN_NBITS < srtb::BITS_PER_BYTE) {
    range_size = out_count * IN_NBITS / srtb::BITS_PER_BYTE;
  } else {
    range_size = out_count;
  }
  q.parallel_for(sycl::range<1>(range_size), [=](sycl::item<1> id) {
     unpack_item<IN_NBITS, handwritten>(d_in, d_out, id.get_id(0), transform);
   }).wait();
}

/** @brief an overload that defaults functor to @c srtb::algorithm::map_identity */
template <int IN_NBITS, bool handwritten = false, typename InputIterator,
          typename OutputIterator>
inline void unpack(InputIterator d_in, OutputIterator d_out,
                   const size_t out_count, sycl::queue& q) {
  return unpack<IN_NBITS, handwritten>(d_in, d_out, out_count,
                                       srtb::algorithm::map_identity(), q);
}

// ----------------------------------------------------------------

/** 
 * @brief 2 basebands in one packet -> 2 segments of 1 baseband, 
 *        in "1" "2" "1" "2" mode
 */
template <int IN_NBITS, typename InputIterator, typename OutputIterator,
          typename TransformFunctor,
          typename = typename std::enable_if<
              (IN_NBITS == sizeof(typename std::iterator_traits<
                                  InputIterator>::value_type) *
                               srtb::BITS_PER_BYTE),
              void>::type>
inline void unpack_item(InputIterator in, OutputIterator out_1,
                        OutputIterator out_2, const size_t x,
                        TransformFunctor transform) {
  using out_type = typename std::iterator_traits<OutputIterator>::value_type;

  const auto in_val_1 = in[2 * x], in_val_2 = in[2 * x + 1];
  out_1[x] = transform(x, static_cast<out_type>(in_val_1));
  out_2[x] = transform(x, static_cast<out_type>(in_val_2));
}

template <int IN_NBITS, typename InputIterator, typename OutputIterator,
          typename TransformFunctor>
inline void unpack(InputIterator d_in, OutputIterator d_out_1,
                   OutputIterator d_out_2, const size_t out_count,
                   TransformFunctor transform, sycl::queue& q) {
  using input_type = typename std::iterator_traits<InputIterator>::value_type;
  static_assert((srtb::BITS_PER_BYTE * sizeof(input_type) == IN_NBITS));

  const size_t range_size = out_count;

  q.parallel_for(sycl::range<1>(range_size), [=](sycl::item<1> id) {
     unpack_item<IN_NBITS>(d_in, d_out_1, d_out_2, id.get_id(0), transform);
   }).wait();
}

// ----------------------------------------------------------------
// board specific unpack methods

/** 
 * @brief 2 basebands in one packet -> 2 segments of 1 baseband, 
 *        in "1" "1" "2" "2" mode
 */
template <typename InputIterator, typename OutputIterator,
          typename TransformFunctor>
inline void unpack_naocpsr_snap1_item(InputIterator in, OutputIterator out_1,
                                      OutputIterator out_2, const size_t x,
                                      TransformFunctor transform) {
  using out_type = typename std::iterator_traits<OutputIterator>::value_type;

  const auto in_val_1 = in[4 * x], in_val_2 = in[4 * x + 1],
             in_val_3 = in[4 * x + 2], in_val_4 = in[4 * x + 3];
  out_1[2 * x] = transform(2 * x, static_cast<out_type>(in_val_1));
  out_1[2 * x + 1] = transform(2 * x + 1, static_cast<out_type>(in_val_2));
  out_2[2 * x] = transform(2 * x, static_cast<out_type>(in_val_3));
  out_2[2 * x + 1] = transform(2 * x + 1, static_cast<out_type>(in_val_4));
}

template <typename InputIterator, typename OutputIterator,
          typename TransformFunctor>
inline void unpack_naocpsr_snap1(InputIterator d_in, OutputIterator d_out_1,
                                 OutputIterator d_out_2, const size_t out_count,
                                 TransformFunctor transform, sycl::queue& q) {
  using input_type = typename std::iterator_traits<InputIterator>::value_type;
  // output of snap1 should be int8
  static_assert((sizeof(input_type) == sizeof(int8_t)));

  const size_t range_size = out_count / 2;
  q.parallel_for(sycl::range<1>(range_size), [=](sycl::item<1> id) {
     unpack_naocpsr_snap1_item(d_in, d_out_1, d_out_2, id.get_id(0), transform);
   }).wait();
}

/** 
 * @brief 4 basebands in one packet -> 4 segments of 1 baseband
 * 
 *  xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx ...... xxxx xxxx xxxx xxxx
 *   1    2    3    4    1    2    3    4   ......  1    2    3    4
 */
template <typename InputIterator, typename OutputIterator,
          typename TransformFunctor>
inline constexpr void unpack_gznupsr_a1_item(InputIterator in,
                                             OutputIterator __restrict__ out_1,
                                             OutputIterator __restrict__ out_2,
                                             OutputIterator __restrict__ out_3,
                                             OutputIterator __restrict__ out_4,
                                             const size_t x,
                                             TransformFunctor transform) {
  using out_type = typename std::iterator_traits<OutputIterator>::value_type;
  constexpr size_t out_iter_count = 4;
  constexpr size_t sample_per_word = 4;
  const std::array<OutputIterator, out_iter_count> out = {out_1, out_2, out_3,
                                                          out_4};
  // store value on stack then copy to out,
  // so 4 (= sample_per_word) elements can be copied in one operation (load-store vectorization)
  std::array<out_type, out_iter_count * sample_per_word> out_val;
#pragma unroll
  for (size_t i = 0; i < out_iter_count; i++) {
#pragma unroll
    for (size_t j = 0; j < sample_per_word; j++) {
      out_val[i * sample_per_word + j] = transform(
          sample_per_word * x + j,
          static_cast<out_type>(in[out_iter_count * sample_per_word * x +
                                   i * sample_per_word + j] ^
                                0x80));
    }
  }

  // check with `roc-obj -d`: global_store_dwordx4
#pragma unroll
  for (size_t i = 0; i < out_iter_count; i++) {
#pragma unroll
    for (size_t j = 0; j < sample_per_word; j++) {
      out[i][sample_per_word * x + j] = out_val[i * sample_per_word + j];
    }
  }
}

/** 
 * @brief 4 basebands in one packet -> 4 segments of 1 baseband
 * 
 *  xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx ...... xxxx xxxx xxxx xxxx
 *   1    2    1    2    1    2    1    2   ......  1    2    1    2
 */
template <typename InputIterator, typename OutputIterator,
          typename TransformFunctor>
inline constexpr void unpack_gznupsr_a1_item(InputIterator in,
                                             OutputIterator __restrict__ out_1,
                                             OutputIterator __restrict__ out_2,
                                             const size_t x,
                                             TransformFunctor transform) {
  using out_type = typename std::iterator_traits<OutputIterator>::value_type;
  constexpr size_t out_iter_count = 2;
  constexpr size_t sample_per_word = 4;
  const std::array<OutputIterator, out_iter_count> out = {out_1, out_2};
  // store value on stack then copy to out,
  // so 4 (= sample_per_word) elements can be copied in one operation (load-store vectorization)
  std::array<out_type, out_iter_count * sample_per_word> out_val;
#pragma unroll
  for (size_t i = 0; i < out_iter_count; i++) {
#pragma unroll
    for (size_t j = 0; j < sample_per_word; j++) {
      out_val[i * sample_per_word + j] = transform(
          sample_per_word * x + j,
          static_cast<out_type>(in[out_iter_count * sample_per_word * x +
                                   i * sample_per_word + j]));
    }
  }

  // check with `roc-obj -d`: global_store_dwordx4
#pragma unroll
  for (size_t i = 0; i < out_iter_count; i++) {
#pragma unroll
    for (size_t j = 0; j < sample_per_word; j++) {
      out[i][sample_per_word * x + j] = out_val[i * sample_per_word + j];
    }
  }
}

template <typename InputIterator, typename OutputIterator,
          typename TransformFunctor>
inline void unpack_gznupsr_a1(InputIterator d_in, OutputIterator d_out_1,
                              OutputIterator d_out_2, OutputIterator d_out_3,
                              OutputIterator d_out_4, const size_t out_count,
                              TransformFunctor transform, sycl::queue& q) {
  using input_type = typename std::iterator_traits<InputIterator>::value_type;
  // output of snap1 should be int8
  static_assert((sizeof(input_type) == sizeof(int8_t)));
  constexpr size_t sample_per_word = 4;

  const size_t range_size = out_count / sample_per_word;
  q.parallel_for(sycl::range<1>(range_size), [=](sycl::item<1> id) {
     unpack_gznupsr_a1_item(d_in, d_out_1, d_out_2, d_out_3, d_out_4,
                            id.get_id(0), transform);
   }).wait();
}

template <typename InputIterator, typename OutputIterator,
          typename TransformFunctor>
inline void unpack_gznupsr_a1(InputIterator d_in, OutputIterator d_out_1,
                              OutputIterator d_out_2, const size_t out_count,
                              TransformFunctor transform, sycl::queue& q) {
  using input_type = typename std::iterator_traits<InputIterator>::value_type;
  // output of snap1 should be int8
  static_assert((sizeof(input_type) == sizeof(int8_t)));
  constexpr size_t sample_per_word = 4;

  const size_t range_size = out_count / sample_per_word;
  q.parallel_for(sycl::range<1>(range_size), [=](sycl::item<1> id) {
     unpack_gznupsr_a1_item(d_in, d_out_1, d_out_2, id.get_id(0), transform);
   }).wait();
}

// runtime dispatch moved to unpack_pipe because reinterpret_cast is not generally available
// for iterator of std::byte -> iterator of other types

}  // namespace unpack
}  // namespace srtb

#endif  // __SRTB_UNPACK__
