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
namespace coherent_dedispersion {

/**
 * @brief type to calculate phase delay of certain frequency channel because of
 *        coherent dedipsersion.
 * @note delta phase of lowerest and highest frequency can be up to 10^9,
 *       while delta phase of adjacent channels is 1~10
 */
using dedisp_real_t = double;
using dedisp_complex_t = srtb::complex<dedisp_real_t>;

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
inline constexpr dedisp_real_t D = 4.148808e3;

/**
 * @brief delay time caused by dispersion relative to f_c, assuming f > f_c (for positive result).
 * For reference, the first FRB found has a delay time of 0.3 ~ 0.4 ms (Lorimer, et al. 2007).
 * The edge of dedispersed time series should be dropped as it is not reliable.
 * @see coherent_dedispersion_factor
 */
template <typename T>
inline T dispersion_delay_time(const T f, const T f_c, const T dm) {
  return -D * dm *
         (dedisp_real_t(1.0) / (f * f) - dedisp_real_t(1.0) / (f_c * f_c));
}

inline srtb::real max_delay_time() {
  return dispersion_delay_time(
      srtb::config.baseband_freq_low + srtb::config.baseband_bandwidth,
      srtb::config.baseband_freq_low, srtb::config.dm);
}

/** @brief count of samples to be reused in next round of baseband data submission,
  *        as dedispersed signal at two edges is not accurate
  * 
  * e.g. baseband_input_count = 20, max delayed samples = 5,
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
  return 2 * std::round(srtb::coherent_dedispersion::max_delay_time() *
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
 * @param f_c reference signal frequency in MHz. subscription from (3.66) in Jiang (2022)
 * @param dm dispersion measurement
 * 
 * ref: C.G. Bassa et al, Enabling pulsar and fast transient searches using coherent dedispersion
 *      (arXiv:1607.00909), https://github.com/cbassa/cdmt/blob/master/cdmt.cu
 */
template <typename C = dedisp_complex_t>
inline C coherent_dedispersion_factor(const dedisp_real_t f,
                                      const dedisp_real_t f_c,
                                      const dedisp_real_t dm) noexcept {
  // TODO: does pre-computing delta_phi saves time?
  // TODO: check the sign!
  // phase delay
  // coefficient 1e6 is here because the unit of f is MHz, not Hz.
  // NOTE: this delta_phi may be up to 10^9, so the precision of float may not be sufficient here.
  const dedisp_real_t delta_phi =
      -2 * M_PI * D * 1e6 * dm *
      (dedisp_real_t(1.0) / (f * f) - dedisp_real_t(1.0) / (f_c * f_c)) * f;
  // another formula
  //const dedisp_real_t diff_f = f - f_c;
  //const dedisp_real_t delta_phi =
  //    -2 * M_PI * D * 1e6 * dm * ((diff_f * diff_f) / (f * f_c * f_c));
  const C factor = C(sycl::cos(delta_phi), sycl::sin(delta_phi));
  return factor;
}

/**
 * @brief coherent dedispersion of one frequency channel
 * @param i operate on input[i]
 * @param delta_f frequency difference between one channel and the next
 * @see srtb::codd::D
 * @see crtb::codd::coherent_dedispertion
 */
template <typename C = dedisp_complex_t, typename Accessor>
inline void coherent_dedispertion_item(Accessor input, Accessor output,
                                       const dedisp_real_t f_min,
                                       const dedisp_real_t delta_f,
                                       const dedisp_real_t dm,
                                       const sycl::item<1> id) {
  const size_t i = id.get_id(0);
  const size_t n = id.get_range(0);
  const dedisp_real_t freq = f_min + delta_f * i, f_c = f_min + delta_f * n;
  const C factor = coherent_dedispersion_factor(freq, f_c, dm);
  const C in = input[i];
  const C out = in * factor;
  output[i] = out;
}

/**
 * @brief coherent dedispersion of frequency channels
 * @param input input buffer accessor, should be FFT-ed baseband signal
 * @param output output buffer accessor
 * @param f_min actual frequency of input[0]
 * @param delta_f delta frequency between nearest two channels (input[i] and input[i-1])
 * @param dm disperse measurement, note: use "accurate" value
 * @see srtb::codd::D
 */
template <typename T, typename C = srtb::complex<T>, typename Accessor>
inline void coherent_dedispertion(Accessor input, Accessor output,
                                  const size_t length, const T f_min,
                                  const T delta_f, const T dm, sycl::queue& q) {
  q.parallel_for(sycl::range<1>(length), [=](sycl::item<1> id) {
     coherent_dedispertion_item(input, output, f_min, delta_f, dm, id);
   }).wait();
}

/**
 * @brief in-place version of the function above
 */
template <typename T, typename C = srtb::complex<T>, typename Accessor>
inline void coherent_dedispertion(Accessor input, const size_t length,
                                  const T f_min, const T delta_f, const T dm,
                                  sycl::queue& q) {
  return coherent_dedispertion<T, C, Accessor>(input, input, length, f_min,
                                               delta_f, dm, q);
}

}  // namespace coherent_dedispersion

namespace codd = coherent_dedispersion;

}  // namespace srtb

#endif  //  __SRTB_COHERENT_DEDISPERSION__
