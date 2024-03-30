/******************************************************************************* 
 * Copyright (c) 2024 fxzjshm
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
#ifndef __SRTB_FFT_1D_R2C_POST_PROCESS__
#define __SRTB_FFT_1D_R2C_POST_PROCESS__

#include <cstddef>
#include <type_traits>

#include "srtb/math.hpp"
#include "srtb/sycl.hpp"

namespace srtb {
namespace fft {

/*
 * post processes procedure of batched 1D R2C FFT.
 * 
 * note: to operate in-place, all highest frequency points are dropped
 * 
 * ref: naive_fft::fft_1d_r2c <= https://www.cnblogs.com/liam-ji/p/11742941.html
 */
template <typename DeviceComplexInputIterator>
void fft_1d_r2c_in_place_post_process(DeviceComplexInputIterator d_in,
                                      size_t count, size_t batch_size,
                                      sycl::queue& q) {
  const size_t N = count;

  const auto H = d_in;
  // operate in-place
  const auto out = d_in;
  // drop highest: not N / 2 + 1 here
  q.parallel_for(sycl::range<2>{N / 2, batch_size}, [=](sycl::item<2> id) {
     using C = typename std::remove_cvref<decltype(d_in[0])>::type;
     using T = typename std::remove_cvref<decltype(d_in[0].real())>::type;

     const size_t k = id.get_id(0);
     const size_t l = id.get_id(1);
     const C H_k = H[l * N + k];
     const C H_N_k = ((k == 0) ? (H[l * N + 0]) : (H[l * N + N - k]));

     //const C H_k_conj = srtb::conj(H_k);
     const C H_N_k_conj = srtb::conj(H_N_k);
     const C F_k = (H_k + H_N_k_conj) / T{2};
     const C G_k = (H_k - H_N_k_conj) * (-C{0, 1} / T{2});
     //const C F_N_k = (H_N_k + H_k_conj) / T{2};
     //const C G_N_k = (H_N_k - H_k_conj) * (-C{0, 1} / T{2});
     const C F_N_k = srtb::conj(F_k);
     const C G_N_k = srtb::conj(G_k);

     //const T theta = -T{2.0 * M_PI} * k / n_real;
     const T theta_k = -T{M_PI} * k / N;
     T w_k_re, w_k_im;
     w_k_im = sycl::sincos(theta_k, sycl::private_ptr<T>{&w_k_re});
     const C w_k = C{w_k_re, w_k_im};
     //const T theta_N_k = -T{M_PI} * (N - k) / N;
     //const T w_N_k_re = sycl::cos(theta_N_k), w_N_k_im = sycl::sin(theta_N_k);
     //const C w_N_k = C{w_N_k_re, w_N_k_im};
     const C w_N_k = C{-w_k_re, w_k_im};

     const C X_k = F_k + G_k * w_k;
     const C X_N_k = F_N_k + G_N_k * w_N_k;
     out[l * N + k] = X_k;
     if (k != 0) {
       out[l * N + N - k] = X_N_k;
     }
     // can prove X_N also satisfies this formula
     // as F_0 and G_0 is real
   }).wait();
}

}  // namespace fft
}  // namespace srtb

#endif  // __SRTB_FFT_1D_R2C_POST_PROCESS__
