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
#ifndef __SRTB_COHERENT_DEDISPERSION__
#define __SRTB_COHERENT_DEDISPERSION__

#include "srtb/commons.hpp"

namespace srtb {

/**
 * main reference: Jiang, 2022
 */
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
 * @brief delay time caused by dispersion relative to f_c, assuming f > f_c (for positive result).
 * For reference, the first FRB found has a delay time of 0.3 ~ 0.4 ms (Lorimer, et al. 2007).
 * The edge of dedispersed time series should be dropped as it is not reliable.
 * @see coherent_dedispersion_factor
 */
template <typename T>
inline T dispersion_delay_time(const T f, const T f_c, const T dm) {
  return -D * dm * (T(1.0) / (f * f) - T(1.0) / (f_c * f_c));
}

inline srtb::real max_delay_time() {
  return dispersion_delay_time(
      srtb::config.baseband_freq_low + srtb::config.baseband_bandwidth,
      srtb::config.baseband_freq_low, srtb::config.dm);
}

/** @brief count of samples to be reused in next round of baseband data submission,
  *        as dedispersed signal at two edges is not accurate
  * 
  * e.g. baseband_input_length = 20, max delayed samples = 5,
  *      x = not accurate signals after dedispersion
  *         ..............................
  *         |---  round  i  ---|
  *         xxxxx..........xxxxx
  *                   |---  round i+1 ---|
  *                   xxxxx..........xxxxx
  * hence nsamps_reserved = 2 * max delayed samples.
  * TODO: check this
  */
inline size_t nsamps_reserved() {
  return 2 * std::round(srtb::codd::max_delay_time() *
                        srtb::config.baseband_sample_rate);
}

/**
 * @brief Calculates the factor used in coherent dedispersion.
 * 
 * Consider light with frequency $\omega = 2 \pi f$, 
 * Delay time is $\Delta t = D {\mathrm{DM} \over f^2}$,
 * Delayed phase is $\Delta \phi = 2 \pi D {\mathrm{DM} \over f}$
 * 
 * @param f signal frequency in MHz
 * @param f_c reference signal frequency in MHz
 * @param dm dispersion measurement
 */
template <typename T, typename C = srtb::complex<T> >
inline C coherent_dedispersion_factor(const T f, const T f_c, const T dm) {
  // TODO: does pre-computing delta_phi saves time?
  const T delta_phi =
      2 * M_PI * D * dm * (T(1.0) / f - T(1.0) / f_c);  // TODO: check the sign!
  const C factor = C(sycl::cos(delta_phi), sycl::sin(delta_phi));
  return factor;
}

/**
 * @brief coherent dedispersion of one frequency channel
 * @param i operate on input[i]
 * @see srtb::codd::D
 * @see crtb::codd::coherent_dedispertion
 */
template <typename T, typename C = srtb::complex<T>, typename Accessor>
inline void coherent_dedispertion_item(Accessor input, const T f_min,
                                       const T delta_f, const T dm,
                                       const size_t i) {
  const T f = f_min + delta_f * i;
  input[i] *= coherent_dedispersion_factor(f, f_min, dm);
}

/**
 * @brief coherent dedispersion of frequency channels
 * @param input input buffer accessor, should be FFT-ed baseband signal
 * @param f_min actual frequency of input[0]
 * @param delta_f delta frequency between nearest two channels (input[i] and input[i-1])
 * @param dm disperse measurement, note: use "accurate" value
 * @see srtb::codd::D
 */
template <typename T, typename C = srtb::complex<T>, typename Accessor>
inline void coherent_dedispertion(Accessor input, const size_t length,
                                  const T f_min, const T f_max, const T dm,
                                  sycl::queue& q) {
  const T delta_f = (f_max - f_min) / (length - 1);
  q.parallel_for(sycl::range<1>(length), [=](sycl::item<1> id) {
    coherent_dedispertion_item(input, f_min, delta_f, dm, id.get_id(0));
  });
}

}  // namespace codd
}  // namespace srtb

#endif  //  __SRTB_COHERENT_DEDISPERSION__
