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
#include <mutex>
#include <thread>

#include "srtb/fft/fft_wrapper.hpp"
#include "srtb/global_variables.hpp"

namespace srtb {
namespace fft {

// ref: https://stackoverflow.com/questions/33106550/partial-specializations-of-templatized-alias-declarations
template <std::floating_point T>
struct fftw_traits;

template <>
struct fftw_traits<double> {
  using real = double;
  using complex = fftw_complex;
  using plan = fftw_plan;
  using iodim64 = fftw_iodim64;

  template <typename... Args>
  static inline decltype(auto) import_system_wisdom(Args&&... args) {
    return fftw_import_system_wisdom(std::forward<Args>(args)...);
  }

  template <typename... Args>
  static inline decltype(auto) import_wisdom_from_filename(Args&&... args) {
    return fftw_export_wisdom_to_filename(std::forward<Args>(args)...);
  }

  template <typename... Args>
  static inline decltype(auto) export_wisdom_to_filename(Args&&... args) {
    return fftw_export_wisdom_to_filename(std::forward<Args>(args)...);
  }

  template <typename... Args>
  static inline decltype(auto) init_threads(Args&&... args) {
    return fftw_init_threads(std::forward<Args>(args)...);
  }

  template <typename... Args>
  static inline decltype(auto) cleanup_threads(Args&&... args) {
    return fftw_cleanup_threads(std::forward<Args>(args)...);
  }

  template <typename... Args>
  static inline decltype(auto) plan_with_nthreads(Args&&... args) {
    return fftw_plan_with_nthreads(std::forward<Args>(args)...);
  }

  template <typename... Args>
  static inline decltype(auto) plan_dft_r2c_1d(Args&&... args) {
    return fftw_plan_dft_r2c_1d(std::forward<Args>(args)...);
  }

  template <typename... Args>
  static inline decltype(auto) plan_guru64_dft_r2c(Args&&... args) {
    return fftw_plan_guru64_dft_r2c(std::forward<Args>(args)...);
  }

  template <typename... Args>
  static inline decltype(auto) plan_guru64_dft_c2r(Args&&... args) {
    return fftw_plan_guru64_dft_c2r(std::forward<Args>(args)...);
  }

  template <typename... Args>
  static inline decltype(auto) plan_guru64_dft(Args&&... args) {
    return fftw_plan_guru64_dft(std::forward<Args>(args)...);
  }

  template <typename... Args>
  static inline decltype(auto) destroy_plan(Args&&... args) {
    return fftw_destroy_plan(std::forward<Args>(args)...);
  }

  template <typename... Args>
  static inline decltype(auto) execute_dft_r2c(Args&&... args) {
    return fftw_execute_dft_r2c(std::forward<Args>(args)...);
  }

  template <typename... Args>
  static inline decltype(auto) execute_dft(Args&&... args) {
    return fftw_execute_dft(std::forward<Args>(args)...);
  }

  template <typename... Args>
  static inline decltype(auto) execute_dft_c2r(Args&&... args) {
    return fftw_execute_dft_c2r(std::forward<Args>(args)...);
  }
};

template <>
struct fftw_traits<float> {
  using real = float;
  using complex = fftwf_complex;
  using plan = fftwf_plan;
  using iodim64 = fftwf_iodim64;

  template <typename... Args>
  static inline decltype(auto) import_system_wisdom(Args&&... args) {
    return fftwf_import_system_wisdom(std::forward<Args>(args)...);
  }

  template <typename... Args>
  static inline decltype(auto) import_wisdom_from_filename(Args&&... args) {
    return fftwf_export_wisdom_to_filename(std::forward<Args>(args)...);
  }

  template <typename... Args>
  static inline decltype(auto) export_wisdom_to_filename(Args&&... args) {
    return fftwf_export_wisdom_to_filename(std::forward<Args>(args)...);
  }

  template <typename... Args>
  static inline decltype(auto) init_threads(Args&&... args) {
    return fftwf_init_threads(std::forward<Args>(args)...);
  }

  template <typename... Args>
  static inline decltype(auto) cleanup_threads(Args&&... args) {
    return fftwf_cleanup_threads(std::forward<Args>(args)...);
  }

  template <typename... Args>
  static inline decltype(auto) plan_with_nthreads(Args&&... args) {
    return fftwf_plan_with_nthreads(std::forward<Args>(args)...);
  }

  template <typename... Args>
  static inline decltype(auto) plan_dft_r2c_1d(Args&&... args) {
    return fftwf_plan_dft_r2c_1d(std::forward<Args>(args)...);
  }

  template <typename... Args>
  static inline decltype(auto) plan_guru64_dft_r2c(Args&&... args) {
    return fftwf_plan_guru64_dft_r2c(std::forward<Args>(args)...);
  }

  template <typename... Args>
  static inline decltype(auto) plan_guru64_dft_c2r(Args&&... args) {
    return fftwf_plan_guru64_dft_c2r(std::forward<Args>(args)...);
  }

  template <typename... Args>
  static inline decltype(auto) plan_guru64_dft(Args&&... args) {
    return fftwf_plan_guru64_dft(std::forward<Args>(args)...);
  }

  template <typename... Args>
  static inline decltype(auto) destroy_plan(Args&&... args) {
    return fftwf_destroy_plan(std::forward<Args>(args)...);
  }

  template <typename... Args>
  static inline decltype(auto) execute_dft_r2c(Args&&... args) {
    return fftwf_execute_dft_r2c(std::forward<Args>(args)...);
  }

  template <typename... Args>
  static inline decltype(auto) execute_dft(Args&&... args) {
    return fftwf_execute_dft(std::forward<Args>(args)...);
  }

  template <typename... Args>
  static inline decltype(auto) execute_dft_c2r(Args&&... args) {
    return fftwf_execute_dft_c2r(std::forward<Args>(args)...);
  }
};

inline std::mutex fftw_mutex;

/**
 * @brief This class inits fftw using RAII.
 * @c srtb::fft::global_fftw_initializer
 */
template <std::floating_point T>
class fftw_initializer {
 public:
  fftw_initializer() { init_fftw(); }

  ~fftw_initializer() { deinit_fftw(); }

 protected:
  inline void load_fftw_wisdom() {
    fftw_traits<T>::import_system_wisdom();
    int ret = fftw_traits<T>::import_wisdom_from_filename(
        srtb::config.fft_fftw_wisdom_path.c_str());
    if (ret == 0) [[unlikely]] {
      SRTB_LOGW << " [fftw_wrapper] "
                << "load fftw wisdom failed!" << srtb::endl;
    }
  }

  inline void save_fftw_wisdom() {
    int ret = fftw_traits<T>::export_wisdom_to_filename(
        srtb::config.fft_fftw_wisdom_path.c_str());
    if (ret == 0) [[unlikely]] {
      SRTB_LOGW << " [fftw_wrapper] "
                << "save fftw wisdom failed!" << srtb::endl;
    }
  }

  inline void init_fftw() {
    int ret = fftw_traits<T>::init_threads();
    if (ret == 0) [[unlikely]] {
      throw std::runtime_error("[fft] init fftw failed!");
    }
    int n_threads = std::max(std::thread::hardware_concurrency(), 1u);
    SRTB_LOGD << " [init_fftw] "
              << "n_threads = " << n_threads << srtb::endl;
    fftw_traits<T>::plan_with_nthreads(n_threads);
    load_fftw_wisdom();
  }

  inline void deinit_fftw() {
    save_fftw_wisdom();
    fftw_traits<T>::cleanup_threads();
  }
};

inline fftw_initializer<srtb::real> global_fftw_initializer;

template <srtb::fft::type fft_type, std::floating_point T, typename C>
class fftw_1d_wrapper
    : public fft_wrapper<fftw_1d_wrapper<fft_type, T, C>, fft_type, T, C> {
 public:
  using super_class = fft_wrapper<fftw_1d_wrapper, fft_type, T, C>;
  friend super_class;
  using FFTW_complex = typename fftw_traits<T>::complex;
  static_assert(sizeof(C) == sizeof(FFTW_complex));
  using FFTW_iodim64 = typename fftw_traits<T>::iodim64;
  using FFTW_plan = typename fftw_traits<T>::plan;

 protected:
  FFTW_plan in_place_plan;
  FFTW_plan out_of_place_plan;

 public:
  fftw_1d_wrapper(size_t n, size_t batch_size, sycl::queue& queue)
      : super_class{n, batch_size, queue} {}

 protected:
  void create_impl(size_t n, size_t batch_size, sycl::queue& queue) {
    // fftw plan functions are not thread-safe
    std::lock_guard lock{srtb::fft::fftw_mutex};

    constexpr auto in_place_flags = FFTW_ESTIMATE;
    constexpr auto out_of_place_flags = FFTW_ESTIMATE | FFTW_DESTROY_INPUT;
    in_place_plan = nullptr;
    out_of_place_plan = nullptr;

    if constexpr (fft_type == srtb::fft::type::R2C_1D ||
                  fft_type == srtb::fft::type::C2R_1D) {
      const size_t n_real = n, n_complex = n / 2 + 1;
      const size_t total_size_real = n_real * batch_size,
                   total_size_complex = n_complex * batch_size;

      std::shared_ptr<C> tmp_complex =
          srtb::device_allocator.allocate_shared<C>(total_size_complex);
      std::shared_ptr<T> tmp_real =
          srtb::device_allocator.allocate_shared<T>(total_size_real);

      FFTW_iodim64 dims{.n = static_cast<ptrdiff_t>(n_real), .is = 1, .os = 1};

      if constexpr (fft_type == srtb::fft::type::R2C_1D) {
        // should be equivalent to this
        /*
        plan = fftw_traits<T>::plan_dft_r2c_1d(static_cast<int>(n), tmp_in.get(),
                                     reinterpret_cast<FFTW_complex*>(tmp_out.get()),
                                     flags);
        */
        T* in = tmp_real.get();
        FFTW_complex* out = reinterpret_cast<FFTW_complex*>(tmp_complex.get());
        FFTW_iodim64 howmany_dims{.n = static_cast<ptrdiff_t>(batch_size),
                                  .is = static_cast<ptrdiff_t>(n_real),
                                  .os = static_cast<ptrdiff_t>(n_complex)};
        in_place_plan = fftw_traits<T>::plan_guru64_dft_r2c(
            /* rank = */ 1, &dims,
            /* howmany_rank = */ 1, &howmany_dims, reinterpret_cast<T*>(out),
            out, in_place_flags);
        out_of_place_plan = fftw_traits<T>::plan_guru64_dft_r2c(
            /* rank = */ 1, &dims,
            /* howmany_rank = */ 1, &howmany_dims, in, out, out_of_place_flags);
      } else {
        // fft_type == srtb::fft::type::C2R_1D
        FFTW_complex* in = reinterpret_cast<FFTW_complex*>(tmp_complex.get());
        T* out = tmp_real.get();
        FFTW_iodim64 howmany_dims{.n = static_cast<ptrdiff_t>(batch_size),
                                  .is = static_cast<ptrdiff_t>(n_complex),
                                  .os = static_cast<ptrdiff_t>(n_real)};
        in_place_plan = fftw_traits<T>::plan_guru64_dft_c2r(
            /* rank = */ 1, &dims,
            /* howmany_rank = */ 1, &howmany_dims, in, reinterpret_cast<T*>(in),
            in_place_flags);
        out_of_place_plan = fftw_traits<T>::plan_guru64_dft_c2r(
            /* rank = */ 1, &dims,
            /* howmany_rank = */ 1, &howmany_dims, in, out, out_of_place_flags);
      }
    } else if constexpr (fft_type == srtb::fft::type::C2C_1D_FORWARD ||
                         fft_type == srtb::fft::type::C2C_1D_BACKWARD) {
      constexpr auto sign = (fft_type == srtb::fft::type::C2C_1D_BACKWARD)
                                ? FFTW_BACKWARD
                                : FFTW_FORWARD;
      const size_t total_size = n * batch_size;

      std::shared_ptr<C> tmp_in =
          srtb::device_allocator.allocate_shared<C>(total_size);
      std::shared_ptr<C> tmp_out =
          srtb::device_allocator.allocate_shared<C>(total_size);

      // should be equivalent to this
      /*
      plan = fftw_traits<T>::plan_dft_1d(static_cast<int>(n), tmp_in.get(), tmp_out.get(),
                               sign, flags);
      */
      FFTW_complex* in = reinterpret_cast<FFTW_complex*>(tmp_in.get());
      FFTW_complex* out = reinterpret_cast<FFTW_complex*>(tmp_out.get());
      FFTW_iodim64 dims{.n = static_cast<ptrdiff_t>(n), .is = 1, .os = 1};
      FFTW_iodim64 howmany_dims{.n = static_cast<ptrdiff_t>(batch_size),
                                .is = static_cast<ptrdiff_t>(n),
                                .os = static_cast<ptrdiff_t>(n)};
      in_place_plan = fftw_traits<T>::plan_guru64_dft(
          /* rank = */ 1, &dims,
          /* howmany_rank = */ 1, &howmany_dims, in, in, sign, in_place_flags);
      out_of_place_plan = fftw_traits<T>::plan_guru64_dft(
          /* rank = */ 1, &dims,
          /* howmany_rank = */ 1, &howmany_dims, in, out, sign,
          out_of_place_flags);
    } else {
      throw std::runtime_error("[fftw_wrapper] TODO");
    }
    if (in_place_plan == nullptr || out_of_place_plan == nullptr) [[unlikely]] {
      throw std::runtime_error("[fftw_wrapper] fftw_plan create failed!");
    }
    set_queue_impl(queue);
  }

  void destroy_impl() {
    std::lock_guard lock{srtb::fft::fftw_mutex};
    fftw_traits<T>::destroy_plan(in_place_plan);
    fftw_traits<T>::destroy_plan(out_of_place_plan);
    in_place_plan = nullptr;
    out_of_place_plan = nullptr;
  }

  bool has_inited_impl() {
    return (in_place_plan != nullptr && out_of_place_plan != nullptr);
  }

  // SFINAE ref: https://stackoverflow.com/a/50714150
  template <typename..., srtb::fft::type fft_type_ = fft_type,
            typename std::enable_if<(fft_type_ == srtb::fft::type::R2C_1D),
                                    int>::type = 0>
  void process_impl(T* in, C* out) {
    if (reinterpret_cast<void*>(in) == reinterpret_cast<void*>(out)) {
      fftw_traits<T>::execute_dft_r2c(in_place_plan, in,
                                      reinterpret_cast<FFTW_complex*>(out));
    } else {
      fftw_traits<T>::execute_dft_r2c(out_of_place_plan, in,
                                      reinterpret_cast<FFTW_complex*>(out));
    }
  }

  template <
      typename..., srtb::fft::type fft_type_ = fft_type,
      typename std::enable_if<(fft_type_ == srtb::fft::type::C2C_1D_FORWARD ||
                               fft_type_ == srtb::fft::type::C2C_1D_BACKWARD),
                              int>::type = 0>
  void process_impl(C* in, C* out) {
    if (reinterpret_cast<void*>(in) == reinterpret_cast<void*>(out)) {
      fftw_traits<T>::execute_dft(in_place_plan,
                                  reinterpret_cast<FFTW_complex*>(in),
                                  reinterpret_cast<FFTW_complex*>(out));
    } else {
      fftw_traits<T>::execute_dft(out_of_place_plan,
                                  reinterpret_cast<FFTW_complex*>(in),
                                  reinterpret_cast<FFTW_complex*>(out));
    }
  }

  template <typename..., srtb::fft::type fft_type_ = fft_type,
            typename std::enable_if<(fft_type_ == srtb::fft::type::C2R_1D),
                                    int>::type = 0>
  void process_impl(C* in, T* out) {
    if (reinterpret_cast<void*>(in) == reinterpret_cast<void*>(out)) {
      fftw_traits<T>::execute_dft_c2r(in_place_plan,
                                      reinterpret_cast<FFTW_complex*>(in), out);
    } else {
      fftw_traits<T>::execute_dft_c2r(out_of_place_plan,
                                      reinterpret_cast<FFTW_complex*>(in), out);
    }
  }

  void set_queue_impl(sycl::queue& queue) {
    // fftw runs on CPU, so no need to set a queue
    (void)queue;
  }
};

}  // namespace fft
}  // namespace srtb

#endif  //  __SRTB_FFTW_WRAPPER__
