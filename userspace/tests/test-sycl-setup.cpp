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

#include <chrono>
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
  const int k = 26;
  const size_t N = size_t{1} << k;
  const size_t n_real = N, n_complex = n_real / 2 + 1;

  sycl::queue q;
  std::cout << "device name = "
            << q.get_device().get_info<sycl::info::device::name>() << std::endl;

  real* d_in = sycl::malloc_device<real>(n_real, q);
  complex* d_spectrum = sycl::malloc_device<complex>(n_complex, q);
  std::cout << "d_in = " << reinterpret_cast<size_t>(d_in) << ", "
            << "d_spectrum = " << reinterpret_cast<size_t>(d_spectrum) << std::endl;

  // trigger JIT / GPU kernel load
  q.parallel_for(sycl::range{n_real}, [=](sycl::item<1> id) {
     const auto i = id.get_id(0);
     d_in[i] = 0;
   }).wait();

  std::chrono::steady_clock::time_point begin_generate =
      std::chrono::steady_clock::now();

  // generate signal
  q.parallel_for(sycl::range{n_real}, [=](sycl::item<1> id) {
     const auto i = id.get_id(0);
     d_in[i] = sycl::sin(real{0.3} * i);
   }).wait();

  std::chrono::steady_clock::time_point end_generate =
      std::chrono::steady_clock::now();

  // naive FFT r2c
  naive_fft::fft_1d_r2c<real, complex>(k, d_in, d_spectrum, q);

  std::chrono::steady_clock::time_point begin_rfi_mitigation =
      std::chrono::steady_clock::now();

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
  const real h_average = h_sum / n_real;
  //// delete RFI
  q.parallel_for(sycl::range{n_complex}, [=](sycl::item<1> id) {
    const auto i = id.get_id(0);
    if (_SYCL_CPLX_NAMESPACE::norm(d_spectrum[i]) > real{1.5} * h_average) {
      d_spectrum[i] = 0;
    }
  });

  std::chrono::steady_clock::time_point end_rfi_mitigation =
      std::chrono::steady_clock::now();

  //// output signal
  //auto d_out = d_spectrum;
  //// in-place transformation
  //naive_fft::fft_1d_c2c<real, complex>(k - 1, d_spectrum, d_out, -1, q);

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
            << "n_real = " << n_real << std::endl;
  std::cout << "[test-setup] "
            << "input signal strength = " << h_sum << std::endl;
  std::cout << "[test-setup] "
            << "h_average = " << h_average << std::endl;
  std::cout << "[test-setup] "
            << "output signal strength = " << h_out_strength << std::endl;
  std::cout << "[test-setup] generate time = "
            << std::chrono::duration_cast<std::chrono::nanoseconds>(
                   end_generate - begin_generate)
                   .count()
            << " ns" << std::endl;
  std::cout << "[test-setup] rfi mitigation 1 time = "
            << std::chrono::duration_cast<std::chrono::nanoseconds>(
                   end_rfi_mitigation - begin_rfi_mitigation)
                   .count()
            << " ns" << std::endl;

  sycl::free(d_in, q);
  d_in = nullptr;
  sycl::free(d_spectrum, q);
  d_spectrum = nullptr;
  return 0;
}
