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
#ifndef __SRTB_CUFFT_LIKE_WRAPPER__
#define __SRTB_CUFFT_LIKE_WRAPPER__

#include <concepts>
#include <mutex>

#include "srtb/fft/fft_wrapper.hpp"
#include "srtb/global_variables.hpp"

#define SRTB_CHECK_CUFFT_LIKE_WRAPPER(expr, expected)                      \
  SRTB_CHECK(expr, expected,                                               \
             throw std::runtime_error(                                     \
                 std::string("[cufft_like_wrapper] " #expr " returned ") + \
                 std::to_string(ret)););

#define SRTB_CHECK_CUFFT_LIKE(expr) \
  SRTB_CHECK_CUFFT_LIKE_WRAPPER(expr, trait::FFT_SUCCESS)

#define SRTB_CHECK_CUFFT_LIKE_API(expr) \
  SRTB_CHECK_CUFFT_LIKE_WRAPPER(expr, trait::API_SUCCESS)

namespace srtb {
namespace fft {

/** @brief backend APIs, used by @c cufft_like_1d_wrapper */
template <std::floating_point T, sycl::backend backend>
struct cufft_like_trait;

inline std::mutex cufft_like_mutex;
#ifdef SRTB_FFT_SHARE_WORK_AREA
// TODO: this seem ugly, re-design this
inline std::shared_ptr<std::byte> cufft_like_shared_work_area;
inline size_t cufft_like_shared_work_area_size = 0;
inline std::atomic<size_t> cufft_like_shared_work_area_users = 0;
#endif  // SRTB_FFT_SHARE_WORK_AREA

/** @brief common wrapper for different vendor's FFT libraries that has a cufft-like API design. */
template <sycl::backend backend, srtb::fft::type fft_type,
          std::floating_point T, typename C = srtb::complex<T> >
class cufft_like_1d_wrapper
    : public fft_wrapper<cufft_like_1d_wrapper<backend, fft_type, T, C>,
                         fft_type, T, C> {
 public:
  using trait = cufft_like_trait<T, backend>;
  using super_class =
      fft_wrapper<cufft_like_1d_wrapper<backend, fft_type, T, C>, fft_type, T,
                  C>;
  using fft_handle = typename trait::fft_handle;
  using stream_t = typename trait::stream_t;
  using real = typename trait::real;
  using complex = typename trait::complex;
  friend super_class;
  static_assert(std::is_same_v<T, real> && sizeof(C) == sizeof(complex));

 protected:
  fft_handle plan;
  size_t workSize;
  stream_t stream;

 public:
  cufft_like_1d_wrapper(size_t n, size_t batch_size, sycl::queue& queue)
      : super_class{n, batch_size, queue} {}

 protected:
  void create_impl(size_t n, size_t batch_size, sycl::queue& q) {
    std::lock_guard lock{srtb::fft::cufft_like_mutex};
    // should be equivalent to this
    /*
    plan = fftw_plan_dft_r2c_1d(static_cast<int>(n), tmp_in.get(),
                               reinterpret_cast<fftw_complex*>(tmp_out.get()),
                               FFTW_PATIENT |  FFTW_DESTROY_INPUT);
    */

    auto device = q.get_device();
    auto native_device = sycl::get_native<backend>(device);
    SRTB_CHECK_CUFFT_LIKE_API(trait::SetDevice(native_device));

    SRTB_CHECK_CUFFT_LIKE(trait::fftCreate(&plan));
#ifdef SRTB_FFT_SHARE_WORK_AREA
    SRTB_CHECK_CUFFT_LIKE(trait::fftSetAutoAllocation(plan, false));
#endif  // SRTB_FFT_SHARE_WORK_AREA
    constexpr auto native_fft_type = trait::get_native_fft_type(fft_type);

    // 64-bit version
    long long int n_ = static_cast<long long int>(n);
    long long int idist, odist;
    long long int inembed[1] = {1}, onembed[1] = {1};  // should have no effect
    if constexpr (fft_type == srtb::fft::type::C2C_1D_FORWARD ||
                  fft_type == srtb::fft::type::C2C_1D_BACKWARD) {
      const long long int n_complex = n_;
      idist = odist = n_complex;
    } else if constexpr (fft_type == srtb::fft::type::R2C_1D) {
      const long long int n_real = n_;
      const long long int n_complex = n_real / 2 + 1;
      idist = n_real;
      odist = n_complex;
    } else if constexpr (fft_type == srtb::fft::type::C2R_1D) {
      const long long int n_real = n_;
      const long long int n_complex = n_real / 2 + 1;
      idist = n_complex;
      odist = n_real;
    } else {
      throw std::runtime_error("[cufft_like_wrapper] create_impl: TODO");
    }

    auto ret_val = trait::fftMakePlanMany64(/* plan = */ plan,
                                            /* rank = */ 1,
                                            /* n = */ &n_,
                                            /* inembed = */ inembed,
                                            /* istride = */ 1,
                                            /* idist = */ idist,
                                            /* onembed = */ onembed,
                                            /* ostride = */ 1,
                                            /* odist = */ odist,
                                            /* type = */ native_fft_type,
                                            /* batch = */ batch_size,
                                            /* worksize = */ &workSize);
    if (ret_val != trait::FFT_SUCCESS && n < std::numeric_limits<int>::max()) {
      trait::fftDestroy(plan);
      // 32-bit version
      SRTB_CHECK_CUFFT_LIKE(trait::fftCreate(&plan));
      SRTB_CHECK_CUFFT_LIKE(trait::fftMakePlan1d(
          plan, static_cast<int>(n), native_fft_type, batch_size, &workSize));
    } else {
      SRTB_CHECK_CUFFT_LIKE(ret_val);
    }

    set_queue_impl(q);

#ifdef SRTB_FFT_SHARE_WORK_AREA
    const size_t required_work_area_bytes = workSize;
    if (required_work_area_bytes > cufft_like_shared_work_area_size) {
      // TODO: is this thread-safe ?
      cufft_like_shared_work_area_size = 0;
      cufft_like_shared_work_area.reset();
      cufft_like_shared_work_area =
          srtb::device_allocator.allocate_shared<std::byte>(
              required_work_area_bytes);
      cufft_like_shared_work_area_size = required_work_area_bytes;
    }
    cufft_like_shared_work_area_users++;
    // set this first, so internal buffer is destroyed
    // will re-set each execution in case it changes (TODO: does this affect performance ?)
    set_work_area_impl(
        reinterpret_cast<void*>(cufft_like_shared_work_area.get()));
#endif  // SRTB_FFT_SHARE_WORK_AREA

    SRTB_LOGI << " [cufft_like_wrapper] "
              << "plan finished. workSize = " << workSize << srtb::endl;
  }

  void destroy_impl() {
    std::lock_guard lock{srtb::fft::cufft_like_mutex};
    SRTB_CHECK_CUFFT_LIKE(trait::fftDestroy(plan));
#ifdef SRTB_FFT_SHARE_WORK_AREA
    cufft_like_shared_work_area_users--;
    if (cufft_like_shared_work_area_users == 0) {
      cufft_like_shared_work_area_size = 0;
      cufft_like_shared_work_area.reset();
    }
#endif  // SRTB_FFT_SHARE_WORK_AREA
  }

  // enable only if fft_type_ == srtb::fft::type::R2C_1D
  template <typename..., srtb::fft::type fft_type_ = fft_type,
            typename std::enable_if<(fft_type_ == srtb::fft::type::R2C_1D),
                                    int>::type = 0>
  void process_impl(T* in, C* out) {
#ifdef SRTB_FFT_SHARE_WORK_AREA
    std::lock_guard lock{srtb::fft::cufft_like_mutex};
    set_work_area_impl(
        reinterpret_cast<void*>(cufft_like_shared_work_area.get()));
#endif  // SRTB_FFT_SHARE_WORK_AREA
    SRTB_CHECK_CUFFT_LIKE(trait::fftExecR2C(
        (*this).plan, static_cast<real*>(in), reinterpret_cast<complex*>(out)));

    flush();
  }

  template <
      typename..., srtb::fft::type fft_type_ = fft_type,
      typename std::enable_if<(fft_type_ == srtb::fft::type::C2C_1D_FORWARD ||
                               fft_type_ == srtb::fft::type::C2C_1D_BACKWARD),
                              int>::type = 0>
  void process_impl(C* in, C* out) {
    constexpr auto direction = (fft_type == srtb::fft::type::C2C_1D_BACKWARD)
                                   ? trait::BACKWARD
                                   : trait::FORWARD;
#ifdef SRTB_FFT_SHARE_WORK_AREA
    std::lock_guard lock{srtb::fft::cufft_like_mutex};
    set_work_area_impl(
        reinterpret_cast<void*>(cufft_like_shared_work_area.get()));
#endif  // SRTB_FFT_SHARE_WORK_AREA
    SRTB_CHECK_CUFFT_LIKE(
        trait::fftExecC2C((*this).plan, reinterpret_cast<complex*>(in),
                          reinterpret_cast<complex*>(out), direction));

    flush();
  }

  template <typename..., srtb::fft::type fft_type_ = fft_type,
            typename std::enable_if<(fft_type_ == srtb::fft::type::C2R_1D),
                                    int>::type = 0>
  void process_impl(C* in, T* out) {
#ifdef SRTB_FFT_SHARE_WORK_AREA
    std::lock_guard lock{srtb::fft::cufft_like_mutex};
    set_work_area_impl(
        reinterpret_cast<void*>(cufft_like_shared_work_area.get()));
#endif  // SRTB_FFT_SHARE_WORK_AREA
    SRTB_CHECK_CUFFT_LIKE(trait::fftExecC2R(
        (*this).plan, reinterpret_cast<complex*>(in), static_cast<real*>(out)));
    flush();
  }

  bool has_inited_impl() {
    // invalid plan causes segmentation fault, so not using plan to check here.
    return true;
  }

  void set_queue_impl(sycl::queue& queue) {
    stream = srtb::backend::get_native_queue<backend, stream_t>(queue);
    // stream seems to be thread-local, so set it in this thread
    SRTB_CHECK_CUFFT_LIKE(trait::fftSetStream(plan, stream));
  }

  void set_work_area_impl(void* work_area) {
    SRTB_CHECK_CUFFT_LIKE(trait::fftSetWorkArea(plan, work_area));
  }

  void flush() {
    SRTB_CHECK_CUFFT_LIKE_API(trait::StreamSynchronize((*this).stream));
  }
};

}  // namespace fft
}  // namespace srtb

#endif  //  __SRTB_CUFFT_LIKE_WRAPPER__
