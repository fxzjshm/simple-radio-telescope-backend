/******************************************************************************* 
 * Copyright (c) 2024 fxzjshm
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
#ifndef __SRTB_IO_BACKEND_REGISTRY__
#define __SRTB_IO_BACKEND_REGISTRY__

#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>

#include "srtb/io/udp/packet_parser.hpp"

namespace srtb {
namespace io {
/* Hope for reflection. */
namespace backend_registry {

/**
 * @brief Just linear sequence of samples
 */
struct simple {
  constexpr static std::string_view name = "simple";
  constexpr static size_t data_stream_count = 1;
};

/**
 * @brief ROACH 2 based backend used in NAOC PSR group.
 */
struct naocpsr_roach2 {
  constexpr static std::string_view name = "naocpsr_roach2";
  using packet_parser = srtb::io::udp::naocpsr_roach2_packet_parser;
  constexpr static size_t data_stream_count = packet_parser::data_stream_count;
};

/**
 * @brief SNAP 1 based backend used in NAOC PSR group.
 */
struct naocpsr_snap1 {
  constexpr static std::string_view name = "naocpsr_snap1";
  using packet_parser = srtb::io::udp::naocpsr_snap1_packet_parser;
  constexpr static size_t data_stream_count = packet_parser::data_stream_count;
};

/**
 * @brief ZCU 111 based backend used in GZNU PSR group.
 */
struct gznupsr_a1 {
  constexpr static std::string_view name = "gznupsr_a1";
  using packet_parser = srtb::io::udp::gznupsr_a1_packet_parser;
  constexpr static size_t data_stream_count = packet_parser::data_stream_count;
};

std::tuple<simple, naocpsr_roach2, naocpsr_snap1, gznupsr_a1> backends;

inline auto get_data_stream_count(std::string_view backend_name) {
  std::optional<size_t> ret;
  auto f = [&](auto&& backend) {
    if (backend_name == backend.name) {
      ret = backend.data_stream_count;
    }
  };
  std::apply([&](auto&&... args) { ((f(args)), ...); }, backends);
  if (ret.has_value()) {
    return ret.value();
  } else {
    throw std::invalid_argument{"[backend_registry] Unknown backend name \"" +
                                std::string{backend_name} +
                                "\" when getting data stream count"};
  }
}

}  // namespace backend_registry
}  // namespace io
}  // namespace srtb

#endif  //  __SRTB_IO_BACKEND_REGISTRY__
