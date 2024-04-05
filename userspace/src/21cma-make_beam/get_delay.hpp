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

#include <cstddef>
#include <span>

#include "3rdparty/vcsbeam.hpp"
#include "assert.hpp"
#include "global_variables.hpp"
#include "mdspan/mdspan.hpp"
#include "srtb/math.hpp"

namespace srtb::_21cma::make_beam {

/** 
 * reference: MWA make_beam/get_delays_small.c 
 */
inline void get_delay(sky_coord_t sky_coord, double obstime_mjd, const std::span<relative_location_t> location,
                      const std::span<double> cable_delay, std::span<double> total_delay) {
  BOOST_ASSERT(location.size() == cable_delay.size());
  beam_geom bg;
  const double lon_rad = srtb::_21cma::make_beam::reference_point.lon_deg * D2R;
  const double lat_rad = srtb::_21cma::make_beam::reference_point.lat_deg * D2R;
  calc_beam_geom(sky_coord.ra_hour, sky_coord.dec_deg, obstime_mjd, lon_rad, lat_rad, &bg);
  const size_t N = location.size();
  for (size_t i = 0; i < N; i++) {
    const auto _loc = location[i];
    total_delay[i] = _loc.E * bg.unit_E + _loc.N * bg.unit_N + _loc.H * bg.unit_H + cable_delay[i];
  }
}

/**
 * @param d_total_delay d_total_delay[n_station]
 * @param d_weight d_weight[n_station, n_channel]
 */
inline void get_weight(std::span<double> d_total_delay, double f_min, double f_max,
                       Kokkos::mdspan<srtb::complex<srtb::real>, Kokkos::dextents<size_t, 2>> d_weight,
                       sycl::queue& q) {
  const size_t n_station = d_weight.extent(0);
  const size_t n_channel = d_weight.extent(1);
  BOOST_ASSERT(n_station == d_total_delay.size());
  q.parallel_for(sycl::range<2>{n_channel, n_station}, [=](sycl::item<2> id) {
     const auto i_channel = id.get_id(0);
     const auto i_station = id.get_id(1);
     const double freq = f_min + (f_max - f_min) / n_channel * i_channel;
     srtb::complex<srtb::real> factor;
     if (freq == 0) [[unlikely]] {
       factor = 0;
     } else {
       // TODO: weight of station & bandpass
       const double k = d_total_delay[i_station] / speed_of_light * freq;
       double k_int;
       const srtb::real k_frac = srtb::modf(k, &k_int);
       const double delta_phi = srtb::real{2 * M_PI} * k_frac;  // TODO: check the sign!
       srtb::real cos_delta_phi, sin_delta_phi;
       // &cos_delta_phi cannot be used here, not in specification
       sin_delta_phi = sycl::sincos(delta_phi, sycl::private_ptr<srtb::real>{&cos_delta_phi});
       factor = srtb::complex<srtb::real>(cos_delta_phi, sin_delta_phi);
     }
     d_weight[i_station, i_channel] = factor;
   }).wait();
}

}  // namespace srtb::_21cma::make_beam

#endif  // __SRTB_21CMA_MAKE_BEAM_GET_DELAYS__
