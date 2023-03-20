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
#ifndef __SRTB_FREQUENCY_DOMAIN_FILTERBANK__
#define __SRTB_FREQUENCY_DOMAIN_FILTERBANK__

#include "srtb/coherent_dedispersion.hpp"

namespace srtb {
namespace frequency_domain_filterbank {

/**
 * A frequency domain filterbank.
 * 
 * Reference: doi: 10.1109/ICOSP.2000.894459.
 * Chun Zhang and Zhihua Wang, "A fast frequency domain filter bank realization algorithm," WCC 2000 - ICSP 2000.
 * https://www.researchgate.net/publication/3881652_A_fast_frequency_domain_filter_bank_realization_algorithm 
 * 
 * @param d_in device pointer of input, which should be FFT-ed baseband of length N
 * @param d_out device pointer of output, which should be L * M matrix,
 *              where L = N/M will be down sampled time sample length after IFFT,
 *              and output[(m-1)*L] ~ output[m*L-1] is the m-th channel, 1 <= m <= M
 * @param N length of input
 * @param M channel count. If N % 2M != 0, behaviour is undefined/unknown.
 */
template <typename T>
inline void frequency_domain_filterbank(T* d_in, T* d_out, size_t N, size_t M,
                                        sycl::queue& q) {
  //std::vector<sycl::event> events(M * 2);
  const size_t L = N / M, L_2 = L / 2 /* == N / 2M */;
  //for (size_t m = 1; m <= M; ++m) {
  //  const auto event = q.copy(d_in + (m - 1) * L_2, d_out + (m - 1) * L, L_2);
  //  events.push_back(event);
  //}
  //for (size_t m = M; m >= 1; --m) {
  //  const auto event = q.copy(d_in + N - L * m, d_out + (m - 1) * L + L_2, L_2);
  //  events.push_back(event);
  //}
  ////q.wait(events);
  //for (auto event : events) {
  //  event.wait();
  //}

  // Alternative implementation
  q.parallel_for(sycl::range<1>{N}, [=](sycl::item<1> id) {
     const size_t i = id.get_id(0);
     const size_t m = i / L + 1;
     const size_t k = i - (m - 1) * L;
     SRTB_ASSERT_IN_KERNEL(1 <= m && m <= M);
     T val;
     if (0 <= k && k <= L_2 - 1) {
       val = d_in[k + L_2 * (m - 1)];
     } else {
       val = d_in[k + N - L * m];
     }
     //d_out[(m - 1) * L + k] = val;
     // transposed
     d_out[k * M + (m - 1)] = val;
   }).wait();
}

}  // namespace frequency_domain_filterbank

// TODO: kernel fusion
//       does this really runs faster? can index computations be reduced?
// TODO: RFI
template <typename T, typename C, typename Iterator>
inline void coherent_dedispersion_and_frequency_domain_filterbank_item(
    Iterator input, Iterator output, const T f_min, const T f_c, const T df,
    const T dm, const size_t M, const sycl::item<1>& id) {
  // frequency domain filterbank
  // calculating index, assuming N % (2*M) == 0.
  // some definitions should be found in the paper metioned above
  // data flow: input[i] ----(coherent dedispersion)----> output[j]
  const size_t j = id.get_id(0), N = id.get_range(0), L = N / M,
               L_2 = L / 2 /* == N / 2M */, m = j / L + 1, k = j - (m - 1) * L;
  SRTB_ASSERT_IN_KERNEL(N % (2 * M) == 0);
  SRTB_ASSERT_IN_KERNEL(0 <= k && k < L && k == j % L);
  size_t i;
  if (0 <= k && k <= L_2 - 1) {
    i = k + L_2 * (m - 1);
  } else {
    i = k + N - L * m;
  }
  SRTB_ASSERT_IN_KERNEL(0 <= i && i < N);
  if (M == 1) {
    SRTB_ASSERT_IN_KERNEL(i == j);
  }

  // coherent dedispersion
  // TODO: does pre-computing delta_phi saves time?
  const T f = f_min + df * i;
  const auto factor = srtb::codd::coherent_dedispersion_factor(f, f_c, dm);
  const C in = input[i], out = in * factor;
  output[j] = out;
}

/**
 * @brief fusion of coherent dedispersion and frequency domain filterbank
 * 
 * @tparam T real data type
 * @tparam C complex data type
 * @tparam Iterator type of the accessor to data, default to 
 * @param input input FFT data
 * @param output output dedispersed and channelized FFT data
 * @param f_min the frequency of the first channel of input data
 * @param df delta frequency between nearest two channels (input[i] and input[i-1])
 * @param dm disperse measurement
 * @param M segment count of output data, also the batch size of inverse FFT
 * @param N size of input, should be a power of 2
 * @param q the @c sycl::queue to be run on
 */
template <typename T, typename C = srtb::complex<T>, typename Iterator = C*>
inline void coherent_dedispersion_and_frequency_domain_filterbank(
    Iterator input, Iterator output, const T f_min, const T f_c, const T df,
    const T dm, const size_t M, const size_t N, sycl::queue& q) {
  q.parallel_for(sycl::range<1>{N}, [=](sycl::item<1> id) {
     coherent_dedispersion_and_frequency_domain_filterbank_item<T, C, Iterator>(
         input, output, f_min, f_c, df, dm, M, id);
   }).wait();
}

// TODO: fft windowing + ring buffer

namespace fdfb = frequency_domain_filterbank;

}  // namespace srtb

#endif  //  __SRTB_FREQUENCY_DOMAIN_FILTERBANK__
