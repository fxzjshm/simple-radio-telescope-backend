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

#include "srtb/commons.hpp"
#include "srtb/fft/fft.hpp"
#include "srtb/fft/fft_window.hpp"
#include "srtb/pipeline/framework/pipe.hpp"
#include "srtb/spectrum/rfi_mitigation.hpp"

namespace srtb {
namespace pipeline {

/**
 * @brief This pipe reads from @c srtb::fft_1d_r2c_queue , perform FFT by calling
 *        @c srtb::fft::dispatch_1d_r2c , then push result to @c srtb::rfi_mitigation_queue
 */
class fft_1d_r2c_pipe {
 protected:
  sycl::queue q;
  std::optional<srtb::fft::fft_1d_dispatcher<srtb::fft::type::R2C_1D> >
      opt_dispatcher;

 public:
  fft_1d_r2c_pipe(sycl::queue q_) : q{q_} {
    opt_dispatcher.emplace(/* n = */ srtb::config.baseband_input_count,
                           /* batch_size = */ 1, q);
  }

  auto operator()(std::stop_token stop_token,
                  srtb::work::fft_1d_r2c_work fft_1d_r2c_work) {
    // assume opt_dispatcher has value
    auto& dispatcher = opt_dispatcher.value();
    //srtb::work::fft_1d_r2c_work fft_1d_r2c_work;
    //SRTB_POP_WORK_OR_RETURN(" [fft 1d r2c pipe] ", srtb::fft_1d_r2c_queue,
    //                        fft_1d_r2c_work, stop_token);
    const size_t in_count = fft_1d_r2c_work.count;
    const size_t out_count = in_count / 2 + 1;
    // reset FFT plan if mismatch
    if (dispatcher.get_n() != in_count || dispatcher.get_batch_size() != 1)
        [[unlikely]] {
      dispatcher.set_size(in_count, 1);
    }
    auto& d_in_shared = fft_1d_r2c_work.ptr;
    std::shared_ptr<srtb::complex<srtb::real> > d_out_shared;
    if constexpr (srtb::fft_operate_in_place) {
      d_out_shared = std::reinterpret_pointer_cast<srtb::complex<srtb::real> >(
          d_in_shared);
    } else {
      d_out_shared =
          srtb::device_allocator.allocate_shared<srtb::complex<srtb::real> >(
              out_count);
    }
    dispatcher.process(d_in_shared.get(), d_out_shared.get());

    // TODO: RF detect pipe

    // no need to check if srtb::fft_operate_in_place here
    // because std::reinterpret_pointer_cast() "share ownership with the initial value of r"
    d_in_shared.reset();

    srtb::work::rfi_mitigation_work rfi_mitigation_work;
    rfi_mitigation_work.ptr = d_out_shared;
    // drop the highest frequency point
    // *Drop*  -- LinusTechTips
    rfi_mitigation_work.count = out_count - 1;
    rfi_mitigation_work.baseband_data =
        std::move(fft_1d_r2c_work.baseband_data);
    rfi_mitigation_work.timestamp = fft_1d_r2c_work.timestamp;
    rfi_mitigation_work.udp_packet_counter = fft_1d_r2c_work.udp_packet_counter;
    //SRTB_PUSH_WORK_OR_RETURN(" [fft 1d r2c pipe] ", srtb::rfi_mitigation_queue,
    //                         rfi_mitigation_work, stop_token);
    return std::optional{rfi_mitigation_work};
  }
};

// ----------------------------------------------------------------

/**
 * @brief This pipe reads from @c srtb::fft_1d_r2c_queue , perform FFT by calling
 *        @c srtb::fft::dispatch_1d_r2c , then push result to ( TODO: @c srtb::frequency_domain_filterbank_queue )
 */
class ifft_1d_c2c_pipe {
 protected:
  sycl::queue q;
  std::optional<srtb::fft::fft_1d_dispatcher<srtb::fft::type::C2C_1D_BACKWARD> >
      //std::optional<
      //    srtb::fft::naive_fft_1d_wrapper<srtb::fft::type::C2C_1D_BACKWARD,
      //                                    srtb::real, srtb::complex<srtb::real> > >
      opt_ifft_dispatcher;
  std::optional<srtb::fft::fft_window_functor_manager<
      srtb::real, srtb::fft::default_window> >
      opt_ifft_window_functor_manager;

 public:
  ifft_1d_c2c_pipe(sycl::queue q_) : q{q_} {
    // divided by 2 because baseband input is real number, but here is complex
    const size_t input_count = srtb::config.baseband_input_count / 2;
    opt_ifft_dispatcher.emplace(/* n = */ input_count, /* batch_size = */ 1, q);
    opt_ifft_window_functor_manager.emplace(srtb::fft::default_window{},
                                            /* n = */ input_count, q);
  }

  auto run_once_impl(std::stop_token stop_token,
                     srtb::work::ifft_1d_c2c_work ifft_1d_c2c_work) {
    auto& ifft_dispatcher = opt_ifft_dispatcher.value();

    //srtb::work::ifft_1d_c2c_work ifft_1d_c2c_work;
    //SRTB_POP_WORK_OR_RETURN(" [ifft 1d c2c pipe] ", srtb::ifft_1d_c2c_queue,
    //                        ifft_1d_c2c_work, stop_token);
    const size_t input_count = ifft_1d_c2c_work.count;

    // reset FFT plan if mismatch
    if (ifft_dispatcher.get_n() != input_count ||
        ifft_dispatcher.get_batch_size() != 1) [[unlikely]] {
      ifft_dispatcher.set_size(input_count, 1);
      opt_ifft_window_functor_manager.emplace(srtb::fft::default_window{},
                                              /* n = */ input_count, q);
    }

    auto& d_in_shared = ifft_1d_c2c_work.ptr;
    std::shared_ptr<srtb::complex<srtb::real> > d_out_shared;
    if constexpr (srtb::fft_operate_in_place) {
      d_out_shared = d_in_shared;
    } else {
      d_out_shared =
          srtb::device_allocator.allocate_shared<srtb::complex<srtb::real> >(
              input_count);
    }
    auto d_in = d_in_shared.get();
    auto d_tmp = d_out_shared.get();

    ifft_dispatcher.process(d_in, d_tmp);
    d_in = nullptr;
    d_in_shared.reset();

    // de-apply window function of size input_count
    if constexpr (!std::is_same_v<srtb::fft::default_window,
                                  srtb::fft::window::rectangle<srtb::real> >) {
      auto ifft_window =
          opt_ifft_window_functor_manager.value().get_coefficients();
      q.parallel_for(sycl::range<1>{input_count}, [=](sycl::item<1> id) {
         const auto i = id.get_id(0);
         const auto x = d_tmp[i];
         d_tmp[i] = x / ifft_window[i];
       }).wait();
    }

    const auto nsamps_reserved_real =
        srtb::coherent_dedispersion::nsamps_reserved();
    const auto nsamps_reserved_complex = nsamps_reserved_real / 2;
    size_t output_count;
    if (nsamps_reserved_complex < input_count) {
      output_count = input_count - nsamps_reserved_complex;
      SRTB_LOGD << " [ifft 1d c2c pipe] "
                << "reserved " << nsamps_reserved_complex
                << " complex time samples." << srtb::endl;
    } else {
      SRTB_LOGW << " [ifft 1d c2c pipe] "
                << "nsamps_reserved_complex = " << nsamps_reserved_complex
                << " >= input_count = " << input_count << srtb::endl;
      output_count = input_count;
    }

    srtb::work::refft_1d_c2c_work refft_1d_c2c_work;
    refft_1d_c2c_work.ptr = d_out_shared;
    refft_1d_c2c_work.count = output_count;
    refft_1d_c2c_work.baseband_data = std::move(ifft_1d_c2c_work.baseband_data);
    refft_1d_c2c_work.timestamp = ifft_1d_c2c_work.timestamp;
    refft_1d_c2c_work.udp_packet_counter = ifft_1d_c2c_work.udp_packet_counter;
    //SRTB_PUSH_WORK_OR_RETURN(" [ifft 1d c2c pipe] ", srtb::refft_1d_c2c_queue,
    //                         refft_1d_c2c_work, stop_token);
    return std::optional{refft_1d_c2c_work};
  }
};

// ----------------------------------------------------------------

/**
 * @brief This pipe reads coherently dedispersed time data from @c srtb::refft_1d_c2c_queue ,
 *                  performs shorter FFTs to get high time resolution,
 *             then push to @c srtb::simplify_spectrum_queue
 * 
 * @note  In re-FFTed complex data, data stream with same frequency channel is 
 *        already time domain samples of this frequency channel.
 *        (seems common sense, but really didn't know before)
 */
class refft_1d_c2c_pipe {
 protected:
  sycl::queue q;
  std::optional<srtb::fft::fft_1d_dispatcher<srtb::fft::type::C2C_1D_FORWARD> >
      //std::optional<
      //    srtb::fft::naive_fft_1d_wrapper<srtb::fft::type::C2C_1D_FORWARD,
      //                                    srtb::real, srtb::complex<srtb::real> > >
      opt_refft_dispatcher;
  std::optional<srtb::fft::fft_window_functor_manager<
      srtb::real, srtb::fft::default_window> >
      opt_refft_window_functor_manager;

 public:
  refft_1d_c2c_pipe(sycl::queue q_) : q{q_} {
    const auto nsamps_reserved_real =
        srtb::coherent_dedispersion::nsamps_reserved();
    const auto nsamps_reserved_complex = nsamps_reserved_real / 2;
    // divided by 2 because baseband input is real number, but here is complex
    const size_t baseband_input_count_complex =
        srtb::config.baseband_input_count / 2;
    size_t input_count;
    if (baseband_input_count_complex <= nsamps_reserved_complex) [[unlikely]] {
      SRTB_LOGW << " [refft 1d c2c pipe] "
                << "baseband_input_count_complex = "
                << baseband_input_count_complex << " <= "
                << "nsamps_reserved_complex = " << nsamps_reserved_complex
                << srtb::endl;
      input_count = baseband_input_count_complex;
    } else {
      input_count = baseband_input_count_complex - nsamps_reserved_complex;
    }
    size_t refft_length = srtb::config.spectrum_channel_count;
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

  auto operator()(std::stop_token stop_token,
                  srtb::work::refft_1d_c2c_work refft_1d_c2c_work) {
    //srtb::work::refft_1d_c2c_work refft_1d_c2c_work;
    //SRTB_POP_WORK_OR_RETURN(" [refft 1d c2c pipe] ", srtb::refft_1d_c2c_queue,
    //                        refft_1d_c2c_work, stop_token);
    const size_t input_count = refft_1d_c2c_work.count;

    auto& refft_dispatcher = opt_refft_dispatcher.value();
    const size_t refft_length =
        std::min(srtb::config.spectrum_channel_count, input_count);
    const size_t refft_batch_size = input_count / refft_length;

    // reset FFT plan if mismatch
    if (refft_dispatcher.get_n() != refft_length ||
        refft_dispatcher.get_batch_size() != refft_batch_size) [[unlikely]] {
      refft_dispatcher.set_size(refft_length, refft_batch_size);
      opt_refft_window_functor_manager.emplace(srtb::fft::default_window{},
                                               /* n = */ refft_length, q);
    }

    auto d_in_shared = refft_1d_c2c_work.ptr;
    auto d_in = d_in_shared.get();

    // re-apply window function of size refft_length
    if constexpr (!std::is_same_v<srtb::fft::default_window,
                                  srtb::fft::window::rectangle<srtb::real> >) {
      auto refft_window =
          opt_refft_window_functor_manager.value().get_coefficients();
      q.parallel_for(sycl::range<1>{input_count}, [=](sycl::item<1> id) {
         const auto i = id.get_id(0);
         const auto j = i - ((i / refft_length) * refft_length);
         SRTB_ASSERT_IN_KERNEL(j < refft_length);
         const auto x = d_in[i];
         d_in[i] = x * refft_window[j];
       }).wait();
    }

    std::shared_ptr<srtb::complex<srtb::real> > d_out_shared;
    if constexpr (srtb::fft_operate_in_place) {
      d_out_shared = d_in_shared;
    } else {
      d_out_shared =
          srtb::device_allocator.allocate_shared<srtb::complex<srtb::real> >(
              input_count);
    }

    auto d_out = d_out_shared.get();
    refft_dispatcher.process(d_in, d_out);

    srtb::work::signal_detect_work signal_detect_work;
    signal_detect_work.ptr = d_out_shared;
    signal_detect_work.count = refft_length;
    signal_detect_work.batch_size = refft_batch_size;
    signal_detect_work.baseband_data =
        std::move(refft_1d_c2c_work.baseband_data);
    signal_detect_work.timestamp = refft_1d_c2c_work.timestamp;
    signal_detect_work.udp_packet_counter =
        refft_1d_c2c_work.udp_packet_counter;
    //SRTB_PUSH_WORK_OR_RETURN(" [refft 1d c2c pipe] ", srtb::signal_detect_queue,
    //                         signal_detect_work, stop_token);
    return std::optional{signal_detect_work};
  }
};

/**
 * @brief This pipe generates time-frequency spectrum from dedispersed spectrum.
 * May used to substitute ifft_pipe & refft_pipe, although a transpose may required.
 * @deprecated Not used now because dedispersed baseband may used for other purposes in the future.
 */
class watfft_1d_c2c_pipe {
 protected:
  sycl::queue q;
  std::optional<srtb::fft::fft_1d_dispatcher<srtb::fft::type::C2C_1D_BACKWARD> >
      opt_watfft_dispatcher;
  std::optional<srtb::fft::fft_window_functor_manager<
      srtb::real, srtb::fft::default_window> >
      opt_watfft_window_functor_manager;

 public:
  watfft_1d_c2c_pipe(sycl::queue q_) : q{q_} {
    // divided by 2 because baseband input is real number, but here is complex
    const size_t baseband_input_count_complex =
        srtb::config.baseband_input_count / 2;
    size_t watfft_batch_size = srtb::config.spectrum_channel_count;
    size_t watfft_length = baseband_input_count_complex / watfft_batch_size;
    if (watfft_length == 0) [[unlikely]] {
      SRTB_LOGW << " [watfft 1d c2c pipe] "
                << "watfft_batch_size too large! Set to input length now.";
      watfft_batch_size = baseband_input_count_complex;
      watfft_length = 1;
    }
    opt_watfft_dispatcher.emplace(/* n = */ watfft_length,
                                  /* batch_size = */ watfft_batch_size, q);
    opt_watfft_window_functor_manager.emplace(srtb::fft::default_window{},
                                              /* n = */ watfft_length, q);
  }

  auto operator()(std::stop_token stop_token,
                  srtb::work::ifft_1d_c2c_work ifft_1d_c2c_work) {
    srtb::work::ifft_1d_c2c_work ifft_1d_c2c_work;
    SRTB_POP_WORK_OR_RETURN(" [watfft 1d c2c pipe] ", srtb::ifft_1d_c2c_queue,
                            ifft_1d_c2c_work, stop_token);
    const size_t input_count = ifft_1d_c2c_work.count;

    auto& watfft_dispatcher = opt_watfft_dispatcher.value();
    const size_t watfft_batch_size =
        std::min(srtb::config.spectrum_channel_count, input_count);
    const size_t watfft_length = input_count / watfft_batch_size;

    // reset FFT plan if mismatch
    if (watfft_dispatcher.get_n() != watfft_length ||
        watfft_dispatcher.get_batch_size() != watfft_batch_size) [[unlikely]] {
      watfft_dispatcher.set_size(watfft_length, watfft_batch_size);
      opt_watfft_window_functor_manager.emplace(srtb::fft::default_window{},
                                                /* n = */ watfft_length, q);
    }

    auto& d_in_shared = ifft_1d_c2c_work.ptr;
    auto d_in = d_in_shared.get();

    std::shared_ptr<srtb::complex<srtb::real> > d_out_shared;
    if constexpr (srtb::fft_operate_in_place) {
      d_out_shared = d_in_shared;
    } else {
      d_out_shared =
          srtb::device_allocator.allocate_shared<srtb::complex<srtb::real> >(
              input_count);
    }

    auto d_out = d_out_shared.get();
    watfft_dispatcher.process(d_in, d_out);

    // de-apply window function of size watfft_length
    if constexpr (!std::is_same_v<srtb::fft::default_window,
                                  srtb::fft::window::rectangle<srtb::real> >) {
      auto watfft_window =
          opt_watfft_window_functor_manager.value().get_coefficients();
      q.parallel_for(sycl::range<1>{input_count}, [=](sycl::item<1> id) {
         const auto i = id.get_id(0);
         const auto channel_id = i / watfft_length;
         const auto j = i - (channel_id * watfft_length);
         SRTB_ASSERT_IN_KERNEL(j < watfft_length);
         const auto x = d_in[i];
         d_in[i] = x / watfft_window[j];
       }).wait();
    }

    // TODO: nsamps_reserved
    //const auto nsamps_reserved_real =
    //    srtb::coherent_dedispersion::nsamps_reserved();
    //const auto nsamps_reserved_complex = nsamps_reserved_real / 2;

    srtb::work::signal_detect_work signal_detect_work;
    signal_detect_work.ptr = d_out_shared;
    signal_detect_work.count = watfft_length;
    signal_detect_work.batch_size = watfft_batch_size;
    signal_detect_work.timestamp = ifft_1d_c2c_work.timestamp;
    signal_detect_work.udp_packet_counter = ifft_1d_c2c_work.udp_packet_counter;
    signal_detect_work.baseband_data =
        std::move(ifft_1d_c2c_work.baseband_data);
    //SRTB_PUSH_WORK_OR_RETURN(" [watfft 1d c2c pipe] ",
    //                         srtb::signal_detect_queue, signal_detect_work,
    //                         stop_token);
    return std::optional{signal_detect_work};
  }
};

}  // namespace pipeline
}  // namespace srtb

#endif  // __SRTB_PIPELINE_FFT_PIPE__
