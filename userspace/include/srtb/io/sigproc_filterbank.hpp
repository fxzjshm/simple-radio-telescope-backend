/******************************************************************************* 
 * Copyright (c) 2023 fxzjshm
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
#ifndef __SRTB_IO_SIGPROC_FILTERBANK__
#define __SRTB_IO_SIGPROC_FILTERBANK__

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <type_traits>

namespace srtb {
namespace io {

/** 
 * @brief just another implementations of part of sigproc headers 
 * ref: sigproc/filterbank_header.c
 */
namespace sigproc {
namespace filterbank_header {

template <typename Stream, typename T>
inline void send(Stream& stream, const T& value)
  requires(std::is_floating_point_v<T> || std::is_integral_v<T>)
{
  constexpr int32_t size = sizeof(value);
  stream.write(reinterpret_cast<const char*>(&value), size);
}

template <typename Stream, typename T>
inline void send(Stream& stream, const T& value)
  requires(std::is_same_v<T, std::string>)
{
  const int32_t length = value.length();
  stream.write(reinterpret_cast<const char*>(&length), sizeof(length));
  stream.write(value.c_str(), length);
}

template <typename Stream>
inline void send(Stream& stream, const char* value) {
  return send(stream, std::string(value));
}

template <typename Stream, typename T, typename U>
inline void send(Stream& stream, const T& key, const U& value) {
  send(stream, key);
  send(stream, value);
}

/** @brief convert RA/Dec to sigproc hhmmss.s / ddmmss.m style */
template <typename T>
inline auto to_sigproc_dms(T x) -> T {
  const T sign = std::copysign(1.0, x);
  const T x_abs = std::abs(x);
  const int32_t x_d = static_cast<int32_t>(x_abs);
  const int32_t x_m = static_cast<int32_t>((x_abs - x_d) * 60);
  const T x_s = static_cast<T>(((x_abs - x_d) * 60 - x_m) * 60);
  return sign * (x_d * 10000 + x_m * 100 + x_s);
}

}  // namespace filterbank_header
}  // namespace sigproc

}  // namespace io
}  // namespace srtb

#endif  //  __SRTB_IO_SIGPROC_FILTERBANK__
