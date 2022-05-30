/******************************************************************************* 
 * Copyright (c) 2022 fxzjshm
 * This software is licensed under Mulan PubL v2.
 * You can use this software according to the terms and conditions of the Mulan
 * PubL v2. You may obtain a copy of Mulan PubL v2 at:
 *          http://license.coscl.org.cn/MulanPubL-2.0
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
 * KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
 * NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE. See the
 * Mulan PubL v2 for more details.
 ******************************************************************************/

#pragma once
#ifndef __SRTB_COHERENT_DEDISPERSION__
#define __SRTB_COHERENT_DEDISPERSION__

#include <complex>

#include "commons.hpp"

namespace srtb {
namespace codd {

/**
 * @brief dispersion constant in "accurate" value
 * 
 * $\Delta t = D \mathrm{DM} / f^2$,
 * $D = {e^2 \over 2 \pi \epsilon_0 m_e c^2
 *    = 4.148808 \times 10^3 \mathrm{MHz^2 pc^{-1} cm^3 s}$
 * 
 * @note due to historical reason, in some software (tempo2, dspsr, etc.)
 *       $D = 1 / 2.41 \times 10^{-4} \mathrm{MHz^2 pc^{-1} cm^3 s}
 *          = 4.149378 \times 10^3 \mathrm{MHz^2 pc^{-1} cm^3 s}$
 *       (Jiang, 2022)
 */
constexpr srtb::real D = 4.148808e3;

/**
 * @brief coherent dedispersion of one frequency channel
 * @param input input buffer accessor, should be FFT-ed baseband signal
 * @param f_min actual frequency of input[0]
 * @param delta_f delta frequency between nearest two channels (input[i] and input[i-1])
 * @param dm disperse measurement, note: use "accurate" value
 * @param i operate on input[i]
 * @see srtb::codd::D
 */
template <typename T, typename C = std::complex<T>, typename Accessor>
inline void coherent_dedispertion_item(Accessor input, const T f_min,
                                       const T delta_f, const T dm,
                                       const size_t i) {
  // TODO: does pre-computing delta_phi saves time?
  const T f = f_min + delta_f * i;
  const T delta_phi = -2 * M_PI * D * dm / f;
  const C factor = C(sycl::cos(delta_phi), sycl::sin(delta_phi));
  input[i] *= factor;
}

/**
 * @brief coherent dedispersion of one frequency channel
 * @param input input buffer accessor, should be FFT-ed baseband signal
 * @param f_min actual frequency of input[0]
 * @param delta_f delta frequency between nearest two channels (input[i] and input[i-1])
 * @param dm disperse measurement, note: use "accurate" value
 * @see srtb::codd::D
 */
template <typename T, typename C = std::complex<T>, typename Accessor>
inline void coherent_dedispertion(Accessor input, const size_t length,
                                  const T f_min, const T f_max, const T dm,
                                  sycl::queue& q) {
  const T delta_f = (f_max - f_min) / (length - 1);
  q.parallel_for(sycl::range<1>(size), [=](sycl::item<1> id) {
    coherent_dedispertion_item(input, f_min, delta_f, dm, id.get_id(0));
  });
}

}  // namespace codd
}  // namespace srtb

#endif  //  __SRTB_COHERENT_DEDISPERSION__