/******************************************************************************* 
 * Copyright (c) 2022-2023 fxzjshm
 * This software is licensed under Mulan PubL v2.
 * You can use this software according to the terms and conditions of the Mulan PubL v2.
 * You may obtain a copy of Mulan PubL v2 at:
 *          http://license.coscl.org.cn/MulanPubL-2.0
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PubL v2 for more details.
 ******************************************************************************/

#include <iostream>
#include <sycl/sycl.hpp>

//#define _SYCL_EXT_CPLX_FAST_MATH
#define _SYCL_CPLX_NAMESPACE sycl_cplx
#include "sycl_ext_complex.dp.hpp"
// ---
#include "naive_fft.dp.hpp"
// ---
#include "buffer_algorithms.dp.hpp"

/**
 * @brief basic demo for radio interference removal
 */
int main() {
  using real = float;
  using complex = _SYCL_CPLX_NAMESPACE::complex<real>;
  const int k = 10;
  const size_t N = 1 << k;
  const size_t n_real = N, n_complex = n_real / 2 + 1;

  sycl::queue q;
  std::cout << "device name = "
            << q.get_device().get_info<sycl::info::device::name>() << std::endl;

  real* d_in = sycl::malloc_device<real>(n_real, q);
  complex* d_spectrum = sycl::malloc_device<complex>(n_complex, q);
  std::cout << "d_in = " << (size_t)d_in << ", "
            << "d_spectrum = " << (size_t)d_spectrum << std::endl;

  // generate signal
  q.parallel_for(sycl::range{n_real}, [=](sycl::item<1> id) {
     const auto i = id.get_id(0);
     d_in[i] = sycl::sin(real{0.3} * i);
   }).wait();

  // naive FFT r2c
  naive_fft::fft_1d_r2c<real, complex>(k, d_in, d_spectrum, q);

  // find threshold and delete radio frequency interference
  //// sum for total spectrum power
  auto descriptor = sycl_pstl::impl::compute_mapreduce_descriptor(
      q.get_device(), /* size = */ n_complex, sizeof(complex));
  void* placeholder = nullptr;
  const real h_sum = sycl_pstl::impl::buffer_mapreduce(
      placeholder, q, d_spectrum, /* init = */ real{0}, descriptor,
      /* map = */
      [](size_t i, complex x) {
        (void)i;
        return _SYCL_CPLX_NAMESPACE::norm(x);
      },
      /* reduce = */
      [](real a, real b) { return a + b; });
  std::cout << "[test-setup] "
            << "input signal strength = " << h_sum << std::endl;
  const real h_average = h_sum / n_real;
  //// delete RFI && normalize
  q.parallel_for(sycl::range{n_complex}, [=](sycl::item<1> id) {
    const auto i = id.get_id(0);
    if (_SYCL_CPLX_NAMESPACE::norm(d_spectrum[i]) > real{1.5} * h_average) {
      d_spectrum[i] = 0;
    }
    d_spectrum[i] /= n_real;
  });

  // output signal
  auto d_out = d_spectrum;
  // in-place transformation
  naive_fft::fft_1d_c2c<real, complex>(k - 1, d_spectrum, d_out, -1, q);

  // output signal strength
  const real h_out_strength = sycl_pstl::impl::buffer_mapreduce(
      placeholder, q, d_spectrum, /* init = */ real{0}, descriptor,
      /* map = */
      [](size_t i, complex x) {
        (void)i;
        return _SYCL_CPLX_NAMESPACE::norm(x);
      },
      /* reduce = */
      [](real a, real b) { return a + b; });

  std::cout << "[test-setup] "
            << "output signal strength = " << h_out_strength << std::endl;

  sycl::free(d_in, q);
  d_in = nullptr;
  sycl::free(d_spectrum, q);
  d_spectrum = nullptr;
  return 0;
}
