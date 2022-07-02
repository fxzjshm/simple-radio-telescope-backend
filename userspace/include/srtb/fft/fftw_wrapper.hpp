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

// TODO: enable multi thread

template <std::floating_point T, typename Complex = std::complex<T> >
class fftw_1d_r2c_wrapper;

template <typename Complex>
class fftw_1d_r2c_wrapper<double, Complex>
    : public fft_wrapper<fftw_1d_r2c_wrapper, double, Complex> {
 public:
  void create_impl(size_t n) {
    if (plan != nullptr) {
      fftw_destroy_plan(plan);
      plan = nullptr;
    }

    load_fftw_wisdom();

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

    if (plan != nullptr) [[likely]] {
      fftw_destroy_plan(plan);
    }
  }

  void process_impl(double* in, Complex* out) {
    fftw_execute_dft_r2c(plan, in, reinterpret_cast<fftw_complex*>(out));
  }

 private:
  fftw_plan plan;
};

}  // namespace fft
}  // namespace srtb

#endif  //  __SRTB_FFTW_WRAPPER__
