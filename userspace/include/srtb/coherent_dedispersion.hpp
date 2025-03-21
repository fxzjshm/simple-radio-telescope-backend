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

#include <boost/config.hpp>

#include "dsmath_sycl.h"
#include "srtb/commons.hpp"

namespace srtb {

/**
 * main reference: Jiang, 2022
 */
namespace coherent_dedispersion {

inline namespace detail {

template <bool use_emulated_fp64>
struct dedispertion_real_trait;

template <>
struct dedispertion_real_trait<false> {
  using type = double;
};

template <>
struct dedispertion_real_trait<true> {
  using type = dsmath::df64;
};

}  // namespace detail

/**
 * @brief type to calculate phase delay of certain frequency channel because of
 *        coherent dedipsersion.
 * @note delta phase of lowerest and highest frequency can be up to 10^9,
 *       while delta phase of adjacent channels is 1~10
 */
using dedisp_real_t =
    detail::dedispertion_real_trait<srtb::use_emulated_fp64>::type;

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
 *       (J.C. Jiang, 2022)
 */
inline constexpr double D = 4.148808e3;

/**
 * @brief delay time caused by dispersion relative to f_c, assuming f > f_c (for positive result).
 * For reference, the first FRB found has a delay time of 0.3 ~ 0.4 ms (Lorimer, et al. 2007).
 * The edge of dedispersed time series should be dropped as it is not reliable.
 * @see coherent_dedispersion_factor
 */
template <typename T>
inline T dispersion_delay_time(const T f, const T f_c, const T dm) {
  return -D * dm * (1.0 / (f * f) - 1.0 / (f_c * f_c));
}

// TODO: not a pure function, move this
inline srtb::real max_delay_time() {
  return dispersion_delay_time(
      srtb::config.baseband_freq_low + srtb::config.baseband_bandwidth,
      srtb::config.baseband_freq_low, srtb::config.dm);
}

/** @brief count of real (not complex) time samples to be reused in next round
  *         of baseband data submission, as dedispersed signal at two edges 
  *         is not accurate
  * 
  * e.g. baseband_input_count = 20, max delayed samples = 5,
  *      x = not accurate signals after dedispersion
  *         ..............................
  *         |---  round  i  ---|
  *         xxxxx..........xxxxx
  *                   |---  round i+1 ---|
  *                   xxxxx..........xxxxx
  * hence nsamps_reserved = 2 * max delayed samples.
  * @note Additional requirement: accurate time samples (i.e. not reserved for next round)
  *       should be a multiple of 2 * spectrum_channel_count so that refft can be done in correct size.
  * TODO: not a pure function, move this
  */
inline auto nsamps_reserved() -> size_t {
  if (!srtb::config.baseband_reserve_sample) {
    return 0;
  }

  const size_t minimal_reserve_count =
      2 * std::round(srtb::coherent_dedispersion::max_delay_time() *
                     srtb::config.baseband_sample_rate);
  const size_t real_time_samples_per_bin =
      srtb::config.spectrum_channel_count * 2;
  const size_t baseband_input_count = srtb::config.baseband_input_count;
  const ssize_t refft_total_size =
      static_cast<ssize_t>(baseband_input_count - minimal_reserve_count) /
      real_time_samples_per_bin * real_time_samples_per_bin;
  const size_t nsamps_may_reserved = baseband_input_count - refft_total_size;
  if (refft_total_size > 0) {
    return nsamps_may_reserved;
  } else {
    SRTB_LOGW << " [nsamps_reserved] "
              << "nsamps_reserved = " << nsamps_may_reserved
              << " > baseband_input_count = " << baseband_input_count
              << srtb::endl;
    srtb::config.baseband_reserve_sample = false;
    return 0;
  }
}

/**
 * @brief v3: optimized from v2.1 using modf
 */
template <typename phase_real, typename T, typename C>
inline C phase_factor_v3(const phase_real f, const phase_real f_c, const phase_real dm) noexcept {
  // NOTE: this delta_phi may be up to 10^9, so precision of float may not be sufficient here.

  // coefficient 1e6 is here because the unit of f is MHz, not Hz.
  // explicitly constexpr, wish device compiler won't generate cl_khr_fp64 operations...
  constexpr phase_real D_ = D * 1e6;
  const phase_real delta_f = f - f_c;
  const phase_real k = D_ * dm / f * ((delta_f / f_c) * (delta_f / f_c));
  phase_real k_int;
  const T k_frac = srtb::modf(k, &k_int);
  const T delta_phi = -T{2 * M_PI} * k_frac;
  T cos_delta_phi, sin_delta_phi;
  // &cos_delta_phi cannot be used here, not in specification
  sin_delta_phi = sycl::sincos(delta_phi, sycl::private_ptr<T>{&cos_delta_phi});
  const C factor = C(cos_delta_phi, sin_delta_phi);
  return factor;
}

/** @brief v2.1: optimized from v2 using sincos */
template <typename phase_real, typename T, typename C>
inline C phase_factor_v2_1(const phase_real f, const phase_real f_c, const phase_real dm) noexcept {
  constexpr phase_real D_ = D * 1e6;
  const phase_real delta_f = f - f_c;
  const phase_real delta_phi = -T{2 * M_PI} * D * 1e6 * dm * ((delta_f * delta_f) / (f * f_c * f_c));
  phase_real cos_delta_phi, sin_delta_phi;
  // &cos_delta_phi cannot be used here, not in specification
  sin_delta_phi = sycl::sincos(delta_phi, sycl::decorated_private_ptr<phase_real>{&cos_delta_phi});
  const C factor = C(cos_delta_phi, sin_delta_phi);
  return factor;
}

/** @brief v2: J.C. Jiang (2022)  ->  Lorimer et al. (2004) Pulsar Handbook */
template <typename phase_real, typename T, typename C>
inline C phase_factor_v2(const phase_real f, const phase_real f_c, const phase_real dm) noexcept {
  constexpr phase_real D_ = D * 1e6;
  const phase_real delta_f = f - f_c;
  const phase_real delta_phi = -T{2 * M_PI} * D * 1e6 * dm * ((delta_f * delta_f) / (f * f_c * f_c));
  const C factor = C(sycl::cos(delta_phi), sycl::sin(delta_phi));
  return factor;
}

/** @brief v1: original formula */
template <typename phase_real, typename T, typename C>
inline C phase_factor_v1(const phase_real f, const phase_real f_c, const phase_real dm) noexcept {
  const phase_real delta_phi =
      -T{2 * M_PI} * D * 1e6 * dm * (phase_real(1.0) / (f * f) - phase_real(1.0) / (f_c * f_c)) * f;
  const C factor = C(sycl::cos(delta_phi), sycl::sin(delta_phi));
  return factor;
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
 * @tparam phase_real type internally used to calculate the delayed phase,
 *                    ususally float64 or df64
 * 
 * ref: Lorimer, D. R. and M. Kramer (2004). Handbook of Pulsar Astronomy. 
 * 
 * ref: C.G. Bassa et al, Enabling pulsar and fast transient searches using coherent dedispersion
 *      (arXiv:1607.00909), https://github.com/cbassa/cdmt/blob/master/cdmt.cu
 */
template <typename phase_real, typename T, typename C>
inline C phase_factor(const phase_real f, const phase_real f_c, const phase_real dm) noexcept {
  // TODO: does pre-computing delta_phi saves time?
  return phase_factor_v3<phase_real, T, C>(f, f_c, dm);
}

/**
 * @brief coherent dedispersion of frequency channels
 * 
 *       |----------------------------|
 *    f_min      ||         ^       f_max
 *               df        f_c
 * 
 * @param input input buffer accessor, should be FFT-ed baseband signal
 * @param output output buffer accessor
 * @param f_min actual frequency of input[0]
 * @param f_c reference frequency
 * @param df delta frequency between nearest two channels (input[i] and input[i-1])
 * @param dm disperse measurement, note: use "accurate" value
 * @see srtb::codd::D
 */
template <typename T, typename Accessor>
inline void coherent_dedispertion(Accessor input, Accessor output,
                                  const size_t length, const T f_min,
                                  const T f_c, const T df, const T dm,
                                  sycl::queue& q) {
  q.parallel_for(sycl::range<1>(length), [=](sycl::item<1> id) {
     using C = typename std::iterator_traits<Accessor>::value_type;
     const auto i = id.get_id(0);
     const dedisp_real_t f = dedisp_real_t{f_min} + dedisp_real_t{df} * static_cast<dedisp_real_t>(i);
     const C factor = phase_factor<dedisp_real_t, T, C>(f, dedisp_real_t{f_c}, dedisp_real_t{dm});
     const C in = input[i];
     const C out = in * factor;
     output[i] = out;
   }).wait();
}

/**
 * @brief in-place version of the function above
 */
template <typename T, typename Accessor>
inline void coherent_dedispertion(Accessor input, const size_t length,
                                  const T f_min, const T f_c, const T delta_f,
                                  const T dm, sycl::queue& q) {
  return coherent_dedispertion<T, Accessor>(input, input, length, f_min, f_c,
                                            delta_f, dm, q);
}

}  // namespace coherent_dedispersion

namespace codd = coherent_dedispersion;

}  // namespace srtb

#endif  //  __SRTB_COHERENT_DEDISPERSION__
