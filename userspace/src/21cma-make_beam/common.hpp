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

}  // namespace detail

}  // namespace srtb::_21cma::make_beam

#endif  // __SRTB_21CMA_MAKE_BEAM_COMMON__
