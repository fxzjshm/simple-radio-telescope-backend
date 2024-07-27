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
#ifndef __SRTB_21CMA_MAKE_BEAM_COMMON__
#define __SRTB_21CMA_MAKE_BEAM_COMMON__

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace srtb::_21cma::make_beam {

inline namespace detail {

struct relative_location_t {
  double E, N, H;
};

struct earth_location_t {
  double lon_deg, lat_deg, height;
};

struct sky_coord_t {
  double ra_hour, dec_deg;
};

enum observation_mode_t { TRACKING, DRIFTING };
enum beamform_mode_t { COHERENT, INCOHERENT };

inline std::string_view to_string(observation_mode_t o) {
  switch (o) {
    case TRACKING:
      return "TRACKING";
    case DRIFTING:
      return "DRIFTING";
  }
}

inline std::string_view to_string(beamform_mode_t o) {
  switch (o) {
    case COHERENT:
      return "COHERENT";
    case INCOHERENT:
      return "INCOHERENT";
  }
}

}  // namespace detail

inline constexpr size_t dada_dbdisk_file_header_size = 4096;
inline constexpr uint32_t station_per_udp_stream = 2;
inline constexpr double freq_min = 0;
inline constexpr double freq_max = 204.8 * 1e6;
inline constexpr uint64_t sample_rate = 2 * (freq_max - freq_min);
inline constexpr uint32_t second_in_day = 60 * 60 * 24;
inline constexpr double speed_of_light = 299792458;

}  // namespace srtb::_21cma::make_beam

#endif  // __SRTB_21CMA_MAKE_BEAM_COMMON__
