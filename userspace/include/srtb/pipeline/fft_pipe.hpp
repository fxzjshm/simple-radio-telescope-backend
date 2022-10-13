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
#ifndef __SRTB_PIPELINE_FFT_PIPE__
#define __SRTB_PIPELINE_FFT_PIPE__

#include "srtb/fft/fft.hpp"
#include "srtb/fft/fft_window.hpp"
#include "srtb/pipeline/pipe.hpp"
#include "srtb/spectrum/rfi_mitigation.hpp"

namespace srtb {
namespace pipeline {

/**
 * @brief This pipe reads from @c srtb::fft_1d_r2c_queue , perform FFT by calling
 *        @c srtb::fft::dispatch_1d_r2c , then push result to ( TODO: @c srtb::frequency_domain_filterbank_queue )
 */
class fft_1d_r2c_pipe : public pipe<fft_1d_r2c_pipe> {
  friend pipe<fft_1d_r2c_pipe>;

 protected:
  std::optional<srtb::fft::fft_1d_dispatcher<srtb::fft::type::R2C_1D> >
      opt_dispatcher;

 public:
  fft_1d_r2c_pipe() = default;

 protected:
  void setup_impl() {
    opt_dispatcher.emplace(/* n = */
                           srtb::config.baseband_input_length *
                               srtb::BITS_PER_BYTE /
                               srtb::config.baseband_input_bits,
                           /* batch_size = */ 1, q);
  }

  void run_once_impl() {
    // assume opt_dispatcher has value
    auto& dispatcher = opt_dispatcher.value();
    srtb::work::fft_1d_r2c_work fft_1d_r2c_work;
    SRTB_POP_WORK(" [fft 1d r2c pipe] ", srtb::fft_1d_r2c_queue,
                  fft_1d_r2c_work);
    const size_t in_count = fft_1d_r2c_work.count;
    const size_t out_count = in_count / 2 + 1;
    // reset FFT plan if mismatch
    if (dispatcher.get_n() != in_count || dispatcher.get_batch_size() != 1)
        [[unlikely]] {
      dispatcher.set_size(in_count, 1);
    }
    auto& d_in_shared = fft_1d_r2c_work.ptr;
    auto d_out_shared =
        srtb::device_allocator.allocate_shared<srtb::complex<srtb::real> >(
            out_count);
    dispatcher.process(d_in_shared.get(), d_out_shared.get());

    // TODO: RF detect pipe

    d_in_shared.reset();
    srtb::work::rfi_mitigation_work out_work;
    out_work.ptr = d_out_shared;
    out_work.count = out_count;
    SRTB_PUSH_WORK(" [fft 1d r2c pipe] ", srtb::rfi_mitigation_queue, out_work);
  }
};

/**
 * @brief This pipe reads from @c srtb::fft_1d_r2c_queue , perform FFT by calling
 *        @c srtb::fft::dispatch_1d_r2c , then push result to ( TODO: @c srtb::frequency_domain_filterbank_queue )
 */
class ifft_1d_c2c_pipe : public pipe<ifft_1d_c2c_pipe> {
  friend pipe<ifft_1d_c2c_pipe>;

 protected:
  srtb::fft::fft_1d_dispatcher<srtb::fft::type::C2C_1D_BACKWARD> dispatcher;

 public:
  ifft_1d_c2c_pipe()
      : dispatcher{/* n = */
                   srtb::config.baseband_input_length * srtb::BITS_PER_BYTE /
                       srtb::config.baseband_input_bits,
                   /* batch_size = */ srtb::config.ifft_channel_count, q} {}

 protected:
  void run_once_impl() {
    srtb::work::ifft_1d_c2c_work ifft_1d_c2c_work;
    SRTB_POP_WORK(" [ifft 1d c2c pipe] ", srtb::ifft_1d_c2c_queue,
                  ifft_1d_c2c_work);
    const size_t count = ifft_1d_c2c_work.count;
    const size_t batch_size = ifft_1d_c2c_work.batch_size;
    // reset FFT plan if mismatch
    if (dispatcher.get_n() != count ||
        dispatcher.get_batch_size() != batch_size) [[unlikely]] {
      dispatcher.set_size(count, batch_size);
    }
    auto& d_in_shared = ifft_1d_c2c_work.ptr;
    auto d_out_shared =
        srtb::device_allocator.allocate_shared<srtb::complex<srtb::real> >(
            count);
    dispatcher.process(d_in_shared.get(), d_out_shared.get());
    // TODO: next pipe
  }
};

/**
 * @brief This pipe reads coherently dedispersed data from @c srtb::refft_1d_c2c_queue ,
 *                  performs iFFT and shorter FFT to get high time resolution,
 *             then push to @c srtb::simplify_spectrum_queue
 */
class refft_1d_c2c_pipe : public pipe<refft_1d_c2c_pipe> {
  friend pipe<refft_1d_c2c_pipe>;

 protected:
  std::optional<srtb::fft::fft_1d_dispatcher<srtb::fft::type::C2C_1D_BACKWARD> >
      opt_ifft_dispatcher;
  std::optional<srtb::fft::fft_window_functor_manager<> >
      opt_ifft_window_functor_manager;
  std::optional<srtb::fft::fft_1d_dispatcher<srtb::fft::type::C2C_1D_FORWARD> >
      opt_refft_dispatcher;
  std::optional<srtb::fft::fft_window_functor_manager<> >
      opt_refft_window_functor_manager;

 public:
  refft_1d_c2c_pipe() = default;

 protected:
  void setup_impl() {
    // divided by 2 because input is real number but here is complex
    const size_t input_count = srtb::config.baseband_input_length *
                               srtb::BITS_PER_BYTE /
                               srtb::config.baseband_input_bits / 2;
    opt_ifft_dispatcher.emplace(/* n = */ input_count, /* batch_size = */ 1, q);
    opt_ifft_window_functor_manager.emplace(srtb::fft::default_window{},
                                            /* n = */ input_count, q);
    size_t refft_length = srtb::config.refft_length;
    size_t refft_batch_size = input_count / refft_length;
    if (refft_batch_size == 0) [[unlikely]] {
      SRTB_LOGW << " [refft 1d c2c pipe] "
                << "refft_length too large! Set to input length now.";
      refft_length = input_count;
      refft_batch_size = 1;
    }
    opt_refft_dispatcher.emplace(/* n = */ refft_length,
                                 /* batch_size = */ refft_batch_size, q);
    opt_refft_window_functor_manager.emplace(srtb::fft::default_window{},
                                             /* n = */ refft_length, q);
  }

  void run_once_impl() {
    auto& ifft_dispatcher = opt_ifft_dispatcher.value();
    auto& refft_dispatcher = opt_refft_dispatcher.value();

    srtb::work::refft_1d_c2c_work refft_1d_c2c_work;
    SRTB_POP_WORK(" [refft 1d c2c pipe] ", srtb::refft_1d_c2c_queue,
                  refft_1d_c2c_work);
    const size_t input_count = refft_1d_c2c_work.count;
    const size_t refft_length =
        std::min(refft_1d_c2c_work.refft_length, input_count);
    const size_t refft_batch_size = input_count / refft_length;
    const size_t refft_total_size = refft_length * refft_batch_size;

    // reset FFT plan if mismatch
    if (ifft_dispatcher.get_n() != input_count ||
        ifft_dispatcher.get_batch_size() != 1) [[unlikely]] {
      ifft_dispatcher.set_size(input_count, 1);
      opt_ifft_window_functor_manager.emplace(srtb::fft::default_window{},
                                              /* n = */ input_count, q);
    }
    if (refft_dispatcher.get_n() != refft_length ||
        refft_dispatcher.get_batch_size() != refft_batch_size) [[unlikely]] {
      refft_dispatcher.set_size(refft_length, refft_batch_size);
      opt_refft_window_functor_manager.emplace(srtb::fft::default_window{},
                                               /* n = */ refft_length, q);
    }

    auto& d_in_shared = refft_1d_c2c_work.ptr;
    auto d_tmp_shared =
        srtb::device_allocator.allocate_shared<srtb::complex<srtb::real> >(
            input_count);
    auto d_in = d_in_shared.get();
    auto d_tmp = d_tmp_shared.get();

    ifft_dispatcher.process(d_in, d_tmp);
    d_in = nullptr;
    d_in_shared.reset();

    // de-apply window function of size input_count and re-apply it of size refft_length
    if constexpr (!std::is_same_v<srtb::fft::default_window,
                                  srtb::fft::window::rectangle<srtb::real> >) {
      auto ifft_window =
          opt_ifft_window_functor_manager.value().get_coefficients().get();
      auto refft_window =
          opt_refft_window_functor_manager.value().get_coefficients().get();
      q.parallel_for(sycl::range<1>{input_count}, [=](sycl::item<1> id) {
         const auto i = id.get_id(0);
         const auto j = i - ((i / refft_length) * refft_length);
         SRTB_ASSERT_IN_KERNEL(j < refft_length);
         const auto x = d_tmp[i];
         d_tmp[i] = x / ifft_window[i] * refft_window[j];
       }).wait();
    }

    auto d_out_shared =
        srtb::device_allocator.allocate_shared<srtb::complex<srtb::real> >(
            input_count);
    auto d_out = d_out_shared.get();
    refft_dispatcher.process(d_tmp, d_out);
    d_tmp = nullptr;
    d_tmp_shared.reset();

    // TODO: why is it needed here? does this bury deeper problems?
    srtb::spectrum::mitigate_rfi(d_out, refft_total_size, q);

    // temporary work: spectrum analyzer
    srtb::work::simplify_spectrum_work simplify_spectrum_work;
    simplify_spectrum_work.ptr = d_out_shared;
    simplify_spectrum_work.count = refft_length;
    simplify_spectrum_work.batch_size = refft_batch_size;
    SRTB_PUSH_WORK(" [refft 1d c2c pipe] ", srtb::simplify_spectrum_queue,
                   simplify_spectrum_work);
  }
};

}  // namespace pipeline
}  // namespace srtb

#endif  // __SRTB_PIPELINE_FFT_PIPE__
