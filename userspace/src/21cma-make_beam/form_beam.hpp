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
#ifndef __SRTB_21CMA_MAKE_BEAM_FORM_BEAM__
#define __SRTB_21CMA_MAKE_BEAM_FORM_BEAM__

#include <cstddef>
#include <span>

#include "assert.hpp"
#include "mdspan/mdspan.hpp"
#include "srtb/math.hpp"

namespace srtb::_21cma::make_beam {

/**
 * @param d_in d_in[n_station, n_sample, n_channel]
 * @param d_weight d_weight[n_station, n_channel]
 * @param d_out d_out[n_sample, n_channel]
 */
inline void form_beam(Kokkos::mdspan<srtb::complex<srtb::real>, Kokkos::dextents<size_t, 3>> d_in,
                      Kokkos::mdspan<srtb::complex<srtb::real>, Kokkos::dextents<size_t, 2>> d_weight,
                      Kokkos::mdspan<srtb::complex<srtb::real>, Kokkos::dextents<size_t, 2>> d_out, sycl::queue &q) {
  const size_t n_station = d_in.extent(0);
  const size_t n_sample = d_in.extent(1);
  const size_t n_channel = d_in.extent(2);
  BOOST_ASSERT(n_station == d_weight.extent(0));
  BOOST_ASSERT(n_channel == d_weight.extent(1));
  BOOST_ASSERT(n_sample == d_out.extent(0));
  BOOST_ASSERT(n_channel == d_out.extent(1));

  q.parallel_for(sycl::range<2>{n_channel, n_sample}, [=](sycl::item<2> id) {
     const size_t i_channel = id.get_id(0);
     const size_t i_sample = id.get_id(1);
     srtb::complex<srtb::real> sum = {0, 0};
     for (size_t i_station = 0; i_station < n_station; i_station++) {
       sum += d_in[i_station, i_sample, i_channel] * d_weight[i_station, i_channel];
     }
     d_out[i_sample, i_channel] = sum;
   }).wait();
}

/**
 * @param d_in d_in[n_station, n_sample, n_channel]
 * @param d_weight d_weight[n_station, n_channel]
 * @param d_out d_out[n_sample, n_channel]
 */
inline void form_beam_incoh(Kokkos::mdspan<srtb::complex<srtb::real>, Kokkos::dextents<size_t, 3>> d_in,
                            Kokkos::mdspan<srtb::complex<srtb::real>, Kokkos::dextents<size_t, 2>> d_weight,
                            Kokkos::mdspan<srtb::complex<srtb::real>, Kokkos::dextents<size_t, 2>> d_out,
                            sycl::queue &q) {
  const size_t n_station = d_in.extent(0);
  const size_t n_sample = d_in.extent(1);
  const size_t n_channel = d_in.extent(2);
  BOOST_ASSERT(n_station == d_weight.extent(0));
  BOOST_ASSERT(n_channel == d_weight.extent(1));
  BOOST_ASSERT(n_sample == d_out.extent(0));
  BOOST_ASSERT(n_channel == d_out.extent(1));

  q.parallel_for(sycl::range<2>{n_channel, n_sample}, [=](sycl::item<2> id) {
     const size_t i_channel = id.get_id(0);
     const size_t i_sample = id.get_id(1);
     srtb::real sum = 0;
     for (size_t i_station = 0; i_station < n_station; i_station++) {
       sum += srtb::norm(d_in[i_station, i_sample, i_channel]) * srtb::norm(d_weight[i_station, i_channel]);
     }
     d_out[i_sample, i_channel] = sum;
   }).wait();
}

}  // namespace srtb::_21cma::make_beam

#endif  // __SRTB_21CMA_MAKE_BEAM_FORM_BEAM__
