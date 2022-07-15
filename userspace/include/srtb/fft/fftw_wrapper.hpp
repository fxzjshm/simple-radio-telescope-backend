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

inline void load_fftw_wisdom() {
  fftw_import_system_wisdom();
  int ret = fftw_import_wisdom_from_filename(
      srtb::config.fft_fftw_wisdom_path.c_str());
  if (ret == 0) [[unlikely]] {
    SRTB_LOGW << " [fftw_wrapper] "
              << "load fftw wisdom failed!" << std::endl;
  }
}

inline void save_fftw_wisdom() {
  int ret =
      fftw_export_wisdom_to_filename(srtb::config.fft_fftw_wisdom_path.c_str());
  if (ret == 0) [[unlikely]] {
    SRTB_LOGW << " [fftw_wrapper] "
              << "save fftw wisdom failed!" << std::endl;
  }
}

inline bool inited_fftw = false;

inline void init_fftw() {
  if (!inited_fftw) {
    int ret = fftw_init_threads();
    if (ret == 0) [[unlikely]] {
      throw std::runtime_error("[fft] init fftw failed!");
    }
    int n_threads = std::max(std::thread::hardware_concurrency(), 1u);
    SRTB_LOGD << " [init_fftw] "
              << "n_threads = " << n_threads << std::endl;
    fftw_plan_with_nthreads(n_threads);
    load_fftw_wisdom();
    inited_fftw = true;
  }
}

// TODO: enable multi thread

template <std::floating_point T, typename Complex = srtb::complex<T> >
class fftw_1d_r2c_wrapper;

template <typename Complex>
class fftw_1d_r2c_wrapper<double, Complex>
    : public fft_wrapper<fftw_1d_r2c_wrapper, double, Complex> {
  friend fft_wrapper<fftw_1d_r2c_wrapper, double, Complex>;

 protected:
  void create_impl(size_t n) {
    srtb::fft::init_fftw();

    auto tmp_in = srtb::device_allocator.allocate_smart<double>(n);
    auto tmp_out = srtb::device_allocator.allocate_smart<Complex>(n / 2 + 1);

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

  void destroy_impl() {
    save_fftw_wisdom();
    fftw_destroy_plan(plan);
  }

  bool has_inited_impl() { return (plan != nullptr); }

  void process_impl(double* in, Complex* out) {
    fftw_execute_dft_r2c(plan, in, reinterpret_cast<fftw_complex*>(out));
  }

  void set_queue_impl(sycl::queue& queue) {}

 private:
  fftw_plan plan;
};

}  // namespace fft
}  // namespace srtb

#endif  //  __SRTB_FFTW_WRAPPER__
