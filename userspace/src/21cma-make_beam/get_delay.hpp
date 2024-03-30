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
#ifndef __SRTB_21CMA_MAKE_BEAM_GET_DELAYS__
#define __SRTB_21CMA_MAKE_BEAM_GET_DELAYS__

#include <span>

#include "3rdparty/vcsbeam.hpp"
#include "assert.hpp"
#include "global_variables.hpp"

namespace srtb::_21cma::make_beam {

/** 
 * reference: MWA make_beam/get_delays_small.c 
 */
inline auto get_delay(sky_coord_t sky_coord, double obstime_mjd, const std::span<relative_location_t> location,
                      std::span<double> cable_delay) -> std::vector<double> {
  BOOST_ASSERT(location.size() == cable_delay.size());
  beam_geom bg;
  const double lon_rad = srtb::_21cma::make_beam::reference_point.lon_deg * D2R;
  calc_beam_geom(sky_coord.ra_hour, sky_coord.dec_deg, obstime_mjd, lon_rad, &bg);
  const size_t N = location.size();
  std::vector<double> total_delay;
  total_delay.resize(N);
  for (size_t i = 0; i < N; i++) {
    const auto _loc = location[i];
    total_delay.at(i) = _loc.E * bg.unit_E + _loc.N * bg.unit_N + _loc.H * bg.unit_H + cable_delay[i];
  }
  return total_delay;
}

}  // namespace srtb::_21cma::make_beam

#endif  // __SRTB_21CMA_MAKE_BEAM_GET_DELAYS__
