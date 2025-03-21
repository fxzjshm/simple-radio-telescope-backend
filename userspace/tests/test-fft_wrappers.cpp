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

#include <fftw3.h>

#include <random>

#include "srtb/fft/fft.hpp"
#include "srtb/log/log.hpp"
#include "test-common.hpp"

#define SRTB_CHECK_TEST_FFT_WRAPPERS(expr)                             \
  SRTB_CHECK(expr, true, {                                             \
    throw std::runtime_error{                                          \
        "[test-fft_wrappers] " #expr " at " __FILE__ ":" +             \
        std::to_string(__LINE__) + " returns " + std::to_string(ret)}; \
  })

template <srtb::fft::type fft_type, typename C = srtb::complex<double> >
inline std::enable_if_t<(fft_type == srtb::fft::type::R2C_1D), void>
call_fftw_1d(size_t n, size_t batch_size, double* in, C* out) {
  static_assert(sizeof(C) == sizeof(fftw_complex));
  using T = double;
  const size_t n_real = n, n_complex = n / 2 + 1;
  auto tmp_in_shared = srtb::host_allocator.allocate_shared<T>(n_real);
  auto tmp_out_shared = srtb::host_allocator.allocate_shared<C>(n_complex);
  auto tmp_in = tmp_in_shared.get();
  auto tmp_out = tmp_out_shared.get();

  fftw_plan plan = fftw_plan_dft_r2c_1d(
      n, tmp_in, reinterpret_cast<fftw_complex*>(tmp_out), FFTW_ESTIMATE);

  for (size_t k = 0; k < batch_size; k++) {
    fftw_execute_dft_r2c(plan, in + k * n_real,
                         reinterpret_cast<fftw_complex*>(out) + k * n_complex);
  }
}

template <srtb::fft::type fft_type, typename C = srtb::complex<float> >
inline std::enable_if_t<(fft_type == srtb::fft::type::R2C_1D), void>
call_fftw_1d(size_t n, size_t batch_size, float* in, C* out) {
  static_assert(sizeof(C) == sizeof(fftwf_complex));
  using T = float;
  const size_t n_real = n, n_complex = n / 2 + 1;
  auto tmp_in_shared = srtb::host_allocator.allocate_shared<T>(n_real);
  auto tmp_out_shared = srtb::host_allocator.allocate_shared<C>(n_complex);
  auto tmp_in = tmp_in_shared.get();
  auto tmp_out = tmp_out_shared.get();

  fftwf_plan plan = fftwf_plan_dft_r2c_1d(
      n, tmp_in, reinterpret_cast<fftwf_complex*>(tmp_out), FFTW_ESTIMATE);

  for (size_t k = 0; k < batch_size; k++) {
    fftwf_execute_dft_r2c(
        plan, in + k * n_real,
        reinterpret_cast<fftwf_complex*>(out) + k * n_complex);
  }
}

/**
 * use as a benchmark:
 * ```bash
 * export SYCL_DEVICE_FILTER=...  # for Intel oneAPI DPC++, can be host, cuda, hip, opencl, ...
 * export ACPP_VISIBILITY_MASK=...  # for AdaptiveCpp, can be omp, cuda, hip, ...
 * export SRTB_LOG_LEVEL=0  # disable other output, or you can just read std::cerr
 * for bit in {0..26}; do ./test-fft_wrappers $bit; done
 * ```
 */
int main(int argc, char** argv) {
  int bit = 2;
  size_t batch_size = 2, test_count = 3;
  const int print_result_threshold = 32;
  if (argc > 1) {
    try {
      bit = std::stoi(argv[1]);
    } catch (const std::invalid_argument& ignored) {
      // bit should remain unchanged
    }
  }
  if (argc > 2) {
    try {
      batch_size = std::stoi(argv[2]);
    } catch (const std::invalid_argument& ignored) {
      // batch_size should remain unchanged
    }
  }
  if (argc > 3) {
    try {
      test_count = std::stoi(argv[3]);
    } catch (const std::invalid_argument& ignored) {
      // test_count should remain unchanged
    }
  }

  const size_t n = static_cast<size_t>(1) << bit, n_real = n,
               n_complex = n / 2 + 1;
  // assume operations related to an element gives 0.5 ulp error, and max tolerance is 5%, min = 1e-5
  // 1e-5 is a common threshold when comparing floating-point numbers; 
  // 5% is a common error threshold in general physics experiments
  srtb::real threshold =
      std::min(std::max(std::numeric_limits<srtb::real>::epsilon() * n / 2, srtb::real{1e-5}), srtb::real{0.05});
  srtb::config.baseband_input_count = n;
  size_t fft_1d_r2c_input_size = srtb::config.baseband_input_count;
  SRTB_LOGD << " [test fft wrappers] "
            << "n = " << n << ", "
            << "batch_size = " << batch_size << ", "
            << "fft_1d_r2c_input_size = " << fft_1d_r2c_input_size
            << srtb::endl;
  SRTB_CHECK_TEST_FFT_WRAPPERS(fft_1d_r2c_input_size == n);

  const size_t total_size_real = n_real * batch_size,
               total_size_complex = n_complex * batch_size;

  // init host data
  std::vector<srtb::real> h_in(total_size_real);
  std::vector<srtb::complex<srtb::real> > h_out(total_size_complex),
      h_out_fftw(total_size_complex);
  std::mt19937 rng{233};
  std::generate(h_in.begin(), h_in.end(), [&]() {
    return static_cast<srtb::real>(static_cast<int>(rng())) /
           static_cast<srtb::real>(INT_MAX);
  });
  SRTB_LOGD << " [test fft wrappers] "
            << "host data inited." << srtb::endl;

  call_fftw_1d<srtb::fft::type::R2C_1D>(n, batch_size, &h_in[0],
                                        &h_out_fftw[0]);
  SRTB_LOGD << " [test fft wrappers] "
            << "reference output computed." << srtb::endl;

  std::vector<sycl::device> devices = sycl::device::get_devices();
  // test for every device
  for (auto device : devices) {
    // set up test environment
    sycl::queue q = sycl::queue{device};
    srtb::device_allocator =
#ifdef SRTB_USE_USM_SHARED_MEMORY
        srtb::memory::cached_allocator<sycl::usm_allocator<
            std::byte, sycl::usm::alloc::shared, srtb::MEMORY_ALIGNMENT> >{
            q};
#else
        srtb::memory::cached_allocator<
            srtb::memory::device_allocator<std::byte, srtb::MEMORY_ALIGNMENT> >{
            q};
#endif
    SRTB_LOGI << " [test fft wrappers] "
              << "device name = " << '\"'
              << device.get_info<sycl::info::device::name>() << '\"'
              << srtb::endl;
    // R2C 1D test
    {
      srtb::fft::fft_1d_dispatcher<srtb::fft::type::R2C_1D> dispatcher{
          n_real, batch_size, q};
      auto d_in_shared =
          srtb::device_allocator.allocate_shared<srtb::real>(total_size_real);
      auto d_out_shared =
          srtb::device_allocator.allocate_shared<srtb::complex<srtb::real> >(
              total_size_complex);
      auto d_in = d_in_shared.get();
      auto d_out = d_out_shared.get();
      q.copy(&h_in[0], d_in, total_size_real).wait();
      for (size_t i = 0; i < test_count; i++) {
        auto start_time = std::chrono::system_clock::now();
        dispatcher.process(d_in, d_out);
        auto end_time = std::chrono::system_clock::now();
        auto run_time = end_time - start_time;
        SRTB_LOGI << " [test fft wrappers] "
                  << "size = 2^" << bit << ", "
                  << "batch_size = " << batch_size << ", "
                  << "time = " << run_time.count() << "ns" << srtb::endl;
        q.copy(d_out, &h_out[0], total_size_complex).wait();
        // print result if n is small enough, for debug only
        if (n < print_result_threshold) {
          std::cout << "Input: ";
          for (size_t k = 0; k < total_size_real; k++) {
            std::cout << h_in[k] << ' ';
          }
          std::cout << srtb::endl;
          std::cout << "Out: ";
          for (size_t k = 0; k < total_size_complex; k++) {
            std::cout << h_out[k] << ' ';
          }
          std::cout << srtb::endl;
          std::cout << "Ref: ";
          for (size_t k = 0; k < total_size_complex; k++) {
            std::cout << h_out_fftw[k] << ' ';
          }
          std::cout << srtb::endl;
        }

        // check results
        SRTB_CHECK_TEST_FFT_WRAPPERS(check_relative_error(
            h_out.begin(), h_out.end(), h_out_fftw.begin(), threshold));

        // print time, not using average time because of JIT
        if (i == test_count - 1) {
          std::cerr << bit << " " << run_time.count() << srtb::endl;
        }
      }
    }

    {
      // TODO: batch 1D c2c FFT
    }
  }
  return 0;
}
