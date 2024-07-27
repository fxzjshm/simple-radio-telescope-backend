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
#ifndef __SRTB_FFT__
#define __SRTB_FFT__

#include <exception>
#include <optional>

#include "srtb/sycl.hpp"
#include "srtb/fft/fft_wrapper.hpp"
#ifdef SRTB_HAS_FFTW
#include "srtb/fft/fftw_wrapper.hpp"
#else
#warning FFTW not found, performance of CPU FFT transformations may be degraded
#endif  // SRTB_HAS_FFTW
#ifdef SRTB_ENABLE_CUDA_INTEROP
#include "srtb/fft/cufft_wrapper.hpp"
#endif  // SRTB_ENABLE_CUDA_INTEROP
#ifdef SRTB_ENABLE_ROCM_INTEROP
#include "srtb/fft/hipfft_wrapper.hpp"
#endif  // SRTB_ENABLE_ROCM_INTEROP
#ifdef SRTB_ENABLE_MUSA_INTEROP
#include "srtb/fft/mufft_wrapper.hpp"
#endif  // SRTB_ENABLE_MUSA_INTEROP
#include "srtb/fft/naive_fft_wrapper.hpp"

#ifdef SRTB_HAS_FFTW
#define SRTB_IF_HAS_FFTW(...) __VA_ARGS__
#else
#define SRTB_IF_HAS_FFTW(...)
#endif  // SRTB_HAS_FFTW

#define SRTB_CHECK_FFT(expr)                                          \
  SRTB_CHECK(expr, true, {                                            \
    throw std::runtime_error{"[fft] " #expr " at " __FILE__ ":" +     \
                             std::to_string(__LINE__) + " returns " + \
                             std::to_string(ret)};                    \
  })

namespace srtb {
namespace fft {

template <srtb::fft::type fft_type, typename T = srtb::real,
          typename C = srtb::complex<T> >
class fft_1d_dispatcher {
 protected:
  sycl::queue q;
#ifdef SRTB_ENABLE_CUDA_INTEROP
  std::optional<cufft_1d_wrapper<fft_type, T, C> > cufft_1d_wrapper_instance;
#endif  // SRTB_ENABLE_CUDA_INTEROP
#ifdef SRTB_ENABLE_ROCM_INTEROP
  std::optional<hipfft_1d_wrapper<fft_type, T, C> > hipfft_1d_wrapper_instance;
#endif  // SRTB_ENABLE_ROCM_INTEROP
#ifdef SRTB_ENABLE_MUSA_INTEROP
  std::optional<mufft_1d_wrapper<fft_type, T, C> > mufft_1d_wrapper_instance;
#endif  // SRTB_ENABLE_MUSA_INTEROP
#ifdef SRTB_HAS_FFTW
  std::optional<fftw_1d_wrapper<fft_type, T, C> > fftw_1d_wrapper_instance;
#endif  // SRTB_HAS_FFTW
  naive_fft_1d_wrapper<fft_type, T, C> naive_fft_1d_wrapper_instance;

 public:
  /**
  * @brief Construct a new fft 1d dispatcher object
  * @param n length of one FFT operation
  * @param batch_size = howmany
  * @param q_ the sycl queue that operaions will run on
  * @note total_size := n * batch_size
  */
  fft_1d_dispatcher(size_t n, size_t batch_size, sycl::queue& q_)
      : q{q_}, naive_fft_1d_wrapper_instance{n, batch_size, q} {
    auto device = q.get_device();
    SRTB_IF_ENABLED_CUDA_INTEROP({
      if (device.get_backend() == srtb::backend::cuda) [[likely]] {
        cufft_1d_wrapper_instance.emplace(n, batch_size, q);
        SRTB_CHECK_FFT(cufft_1d_wrapper_instance.has_value());
        return;
      }
    });

    SRTB_IF_ENABLED_ROCM_INTEROP({
      if (device.get_backend() == srtb::backend::rocm) [[likely]] {
        hipfft_1d_wrapper_instance.emplace(n, batch_size, q);
        SRTB_CHECK_FFT(hipfft_1d_wrapper_instance.has_value());
        return;
      }
    });

    SRTB_IF_ENABLED_MUSA_INTEROP({
      if (device.get_backend() == srtb::backend::musa) [[likely]] {
        mufft_1d_wrapper_instance.emplace(n, batch_size, q);
        SRTB_CHECK_FFT(mufft_1d_wrapper_instance.has_value());
        return;
      }
    });

    SRTB_IF_HAS_FFTW({
      if (device.is_cpu()) {
        fftw_1d_wrapper_instance.emplace(n, batch_size, q);
        SRTB_CHECK_FFT(fftw_1d_wrapper_instance.has_value());
        return;
      }
    });
  }

#define SRTB_FFT_DISPATCH(func, ...)                                 \
  {                                                                  \
    SRTB_IF_ENABLED_CUDA_INTEROP({                                   \
      if (cufft_1d_wrapper_instance.has_value()) [[likely]] {        \
        return cufft_1d_wrapper_instance.value().func(__VA_ARGS__);  \
      }                                                              \
    });                                                              \
                                                                     \
    SRTB_IF_ENABLED_ROCM_INTEROP({                                   \
      if (hipfft_1d_wrapper_instance.has_value()) [[likely]] {       \
        return hipfft_1d_wrapper_instance.value().func(__VA_ARGS__); \
      }                                                              \
    });                                                              \
                                                                     \
    SRTB_IF_ENABLED_MUSA_INTEROP({                                   \
      if (mufft_1d_wrapper_instance.has_value()) [[likely]] {        \
        return mufft_1d_wrapper_instance.value().func(__VA_ARGS__);  \
      }                                                              \
    });                                                              \
                                                                     \
    SRTB_IF_HAS_FFTW({                                               \
      if (fftw_1d_wrapper_instance.has_value()) {                    \
        return fftw_1d_wrapper_instance.value().func(__VA_ARGS__);   \
      }                                                              \
    });                                                              \
                                                                     \
    return naive_fft_1d_wrapper_instance.func(__VA_ARGS__);          \
  }

  template <typename InputIterator, typename OutputIterator>
  void process(InputIterator in, OutputIterator out) {
    SRTB_FFT_DISPATCH(process, in, out);
  }

  void set_size(size_t n, size_t batch_size) {
    SRTB_FFT_DISPATCH(set_size, n, batch_size);
  }

  size_t get_n() const { SRTB_FFT_DISPATCH(get_n); }

  size_t get_batch_size() const { SRTB_FFT_DISPATCH(get_batch_size); }

#undef SRTB_FFT_DISPATCH
};

}  // namespace fft
}  // namespace srtb

#endif  //  __SRTB_FFT__
