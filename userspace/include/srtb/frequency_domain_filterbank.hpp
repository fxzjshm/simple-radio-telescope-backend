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

/**
 * A frequency domain filterbank.
 * 
 * Reference: doi: 10.1109/ICOSP.2000.894459.
 * Chun Zhang and Zhihua Wang, "A fast frequency domain filter bank realization algorithm," WCC 2000 - ICSP 2000.
 * https://www.researchgate.net/publication/3881652_A_fast_frequency_domain_filter_bank_realization_algorithm 
 * 
 * @param input device pointer of input, which should be FFT-ed baseband of length N
 * @param output device pointer of output, which should be L * M matrix,
 *               where L = N/M will be down sampled time sample length after IFFT,
 *               and output[(m-1)*L] ~ output[m*L-1] is the m-th channel, 1 <= m <= M
 * @param N length of input
 * @param M channel count. If N % 2M != 0, behaviour is undefined/unknown.
 */
template <typename T, typename C = srtb::complex<T> >
inline void frequency_domain_filterbank(T* input, T* output, size_t N, size_t M,
                                        sycl::queue& q) {
  std::vector<sycl::event> events(M * 2);
  const size_t L = N / M, L_2 = L / 2 /* == N / 2M */;
  for (size_t m = 1; m <= M; ++m) {
    const auto event =
        q.copy(input + (m - 1) * L_2, output + (m - 1) * L_2, L_2);
    events.push_back(event);
  }
  for (size_t m = M; m >= 1; --m) {
    const auto event = q.copy(input + N - L * m, output + (m - 1) * L + L_2);
    events.push_back(event);
  }
  //q.wait(events);
  for (auto event : events) {
    event.wait();
  }
}

// TODO: kernel fusion
//       does this really runs faster? can index computations be reduced?
// TODO: RFI
template <typename T, typename C = srtb::complex<T>, typename Iterator = T*>
inline void coherent_dedispersion_and_frequency_domain_filterbank(
    Iterator input, Iterator output, const T f_min, const T delta_f, const T dm,
    const size_t M, const sycl::item<1>& id) {
  // frequency domain filterbank
  // calculating index, assuming N % (2*M) == 0.
  // some definitions should be found in the paper metioned above
  // data flow: input[i] ----(coherent dedispersion)----> output[j]
  const size_t j = id.get_id(0), N = id.get_range(0), L = N / M,
               L_2 = L / 2 /* == N / 2M */, m = j / L + 1, k = j - (m - 1) * L;
  assert(0 <= k && k < L && k == j % L);
  size_t i;
  if (0 <= k && k <= L_2 - 1) {
    i = k + L_2 * (m - 1);
  } else {
    i = k + N - L * m;
  }

  // coherent dedispersion
  // TODO: does pre-computing delta_phi saves time?
  const T f = f_min + delta_f * i;
  output[j] = input[i] * coherent_dedispersion_factor(f, f_min, dm);
}

// TODO: fft windowing + ring buffer

#endif  //  __SRTB_FREQUENCY_DOMAIN_FILTERBANK__
