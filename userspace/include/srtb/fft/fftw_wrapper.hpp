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
#ifndef __SRTB_FFTW_WRAPPER__
#define __SRTB_FFTW_WRAPPER__

#include <fftw3.h>

#include <concepts>
#include <thread>

#include "srtb/fft/fft_wrapper.hpp"
#include "srtb/global_variables.hpp"

namespace srtb {
namespace fft {

template <std::floating_point T, typename Complex = srtb::complex<T> >
class fftw_1d_r2c_wrapper;

/**
 * @brief This class inits fftw using RAII.
 * @c srtb::fft::global_fftw_initializer
 */
class fftw_initializer {
 public:
  fftw_initializer() { init_fftw(); }

  ~fftw_initializer() { deinit_fftw(); }

 protected:
  inline void load_fftw_wisdom() {
    fftw_import_system_wisdom();
    int ret = fftw_import_wisdom_from_filename(
        srtb::config.fft_fftw_wisdom_path.c_str());
    if (ret == 0) [[unlikely]] {
      SRTB_LOGW << " [fftw_wrapper] "
                << "load fftw wisdom failed!" << srtb::endl;
    }
  }

  inline void save_fftw_wisdom() {
    int ret = fftw_export_wisdom_to_filename(
        srtb::config.fft_fftw_wisdom_path.c_str());
    if (ret == 0) [[unlikely]] {
      SRTB_LOGW << " [fftw_wrapper] "
                << "save fftw wisdom failed!" << srtb::endl;
    }
  }

  inline void init_fftw() {
    int ret = fftw_init_threads();
    if (ret == 0) [[unlikely]] {
      throw std::runtime_error("[fft] init fftw failed!");
    }
    int n_threads = std::max(std::thread::hardware_concurrency(), 1u);
    SRTB_LOGD << " [init_fftw] "
              << "n_threads = " << n_threads << srtb::endl;
    fftw_plan_with_nthreads(n_threads);
    load_fftw_wisdom();
  }

  inline void deinit_fftw() { save_fftw_wisdom(); }
};

inline fftw_initializer global_fftw_initializer;

template <typename Complex>
class fftw_1d_r2c_wrapper<double, Complex>
    : public fft_wrapper<fftw_1d_r2c_wrapper, double, Complex> {
  friend fft_wrapper<fftw_1d_r2c_wrapper, double, Complex>;
  static_assert(sizeof(Complex) == sizeof(fftw_complex));

 protected:
  void create_impl(size_t n) {
    auto tmp_in = srtb::device_allocator.allocate_shared<double>(n);
    auto tmp_out = srtb::device_allocator.allocate_shared<Complex>(n / 2 + 1);

    // should be equivalent to this
    /*
    plan = fftw_plan_dft_r2c_1d(static_cast<int>(n), tmp_in.get(),
                               reinterpret_cast<fftw_complex*>(tmp_out.get()),
                               FFTW_PATIENT |  FFTW_DESTROY_INPUT);
    */

    fftw_iodim64 dims{.n = static_cast<ptrdiff_t>(n), .is = 1, .os = 1};
    fftw_iodim64 howmany_dims{.n = 1, .is = 1, .os = 1};
    plan = fftw_plan_guru64_dft_r2c(
        /* rank = */ 1, &dims,
        /* howmany_rank = */ 1, &howmany_dims,
        /* in = */ tmp_in.get(),
        /* out = */ reinterpret_cast<fftw_complex*>(tmp_out.get()),
        /* flags = */ FFTW_ESTIMATE | FFTW_DESTROY_INPUT);
    if (plan == nullptr) [[unlikely]] {
      throw std::runtime_error("[fftw_wrapper] fftw_plan create failed!");
    }
  }

  void destroy_impl() { fftw_destroy_plan(plan); }

  bool has_inited_impl() { return (plan != nullptr); }

  void process_impl(double* in, Complex* out) {
    fftw_execute_dft_r2c(plan, in, reinterpret_cast<fftw_complex*>(out));
  }

  void set_queue_impl(sycl::queue& queue) {
    // fftw runs on CPU, so no need to set a queue
    (void)queue;
  }

 private:
  fftw_plan plan;
};

}  // namespace fft
}  // namespace srtb

// ------------------------------------------------

namespace srtb {
namespace fft {

/**
 * @brief This class inits fftwf using RAII.
 * @c srtb::fft::global_fftwf_initializer
 */
class fftwf_initializer {
 public:
  fftwf_initializer() { init_fftwf(); }

  ~fftwf_initializer() { deinit_fftwf(); }

 protected:
  inline void load_fftwf_wisdom() {
    fftwf_import_system_wisdom();
    int ret = fftwf_import_wisdom_from_filename(
        srtb::config.fft_fftwf_wisdom_path.c_str());
    if (ret == 0) [[unlikely]] {
      SRTB_LOGW << " [fftwf_wrapper] "
                << "load fftwf wisdom failed!" << srtb::endl;
    }
  }

  inline void save_fftwf_wisdom() {
    int ret = fftwf_export_wisdom_to_filename(
        srtb::config.fft_fftwf_wisdom_path.c_str());
    if (ret == 0) [[unlikely]] {
      SRTB_LOGW << " [fftwf_wrapper] "
                << "save fftwf wisdom failed!" << srtb::endl;
    }
  }

  inline void init_fftwf() {
    int ret = fftwf_init_threads();
    if (ret == 0) [[unlikely]] {
      throw std::runtime_error("[fft] init fftwf failed!");
    }
    int n_threads = std::max(std::thread::hardware_concurrency(), 1u);
    SRTB_LOGD << " [init_fftwf] "
              << "n_threads = " << n_threads << srtb::endl;
    fftwf_plan_with_nthreads(n_threads);
    load_fftwf_wisdom();
  }

  inline void deinit_fftwf() { save_fftwf_wisdom(); }
};

inline fftwf_initializer global_fftwf_initializer;

template <typename Complex>
class fftw_1d_r2c_wrapper<float, Complex>
    : public fft_wrapper<fftw_1d_r2c_wrapper, float, Complex> {
  friend fft_wrapper<fftw_1d_r2c_wrapper, float, Complex>;
  static_assert(sizeof(Complex) == sizeof(fftwf_complex));

 protected:
  void create_impl(size_t n) {
    auto tmp_in = srtb::device_allocator.allocate_shared<float>(n);
    auto tmp_out = srtb::device_allocator.allocate_shared<Complex>(n / 2 + 1);

    // should be equivalent to this
    /*
    plan = fftwf_plan_dft_r2c_1d(static_cast<int>(n), tmp_in.get(),
                               reinterpret_cast<fftwf_complex*>(tmp_out.get()),
                               FFTW_PATIENT |  FFTW_DESTROY_INPUT);
    */

    fftwf_iodim64 dims{.n = static_cast<ptrdiff_t>(n), .is = 1, .os = 1};
    fftwf_iodim64 howmany_dims{.n = 1, .is = 1, .os = 1};
    plan = fftwf_plan_guru64_dft_r2c(
        /* rank = */ 1, &dims,
        /* howmany_rank = */ 1, &howmany_dims,
        /* in = */ tmp_in.get(),
        /* out = */ reinterpret_cast<fftwf_complex*>(tmp_out.get()),
        /* flags = */ FFTW_ESTIMATE | FFTW_DESTROY_INPUT);
    if (plan == nullptr) [[unlikely]] {
      throw std::runtime_error("[fftwf_wrapper] fftwf_plan create failed!");
    }
  }

  void destroy_impl() { fftwf_destroy_plan(plan); }

  bool has_inited_impl() { return (plan != nullptr); }

  void process_impl(float* in, Complex* out) {
    fftwf_execute_dft_r2c(plan, in, reinterpret_cast<fftwf_complex*>(out));
  }

  void set_queue_impl(sycl::queue& queue) {
    // fftwf runs on CPU, so no need to set a queue
    (void)queue;
  }

 private:
  fftwf_plan plan;
};

}  // namespace fft
}  // namespace srtb

#endif  //  __SRTB_FFTW_WRAPPER__
