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

#if SRTB_ENABLE_GUI
// Qt's keywords like signals and slots are annoying
#ifndef QT_NO_KEYWORDS
#define QT_NO_KEYWORDS
#endif  // QT_NO_KEYWORDS
#endif  // SRTB_ENABLE_GUI

#include <chrono>
#include <filesystem>
#include <iostream>
#include <vector>

#include "srtb/coherent_dedispersion.hpp"
#include "srtb/commons.hpp"
#include "srtb/pipeline/copy_to_device_pipe.hpp"
#include "srtb/pipeline/dedisperse_pipe.hpp"
#include "srtb/pipeline/fft_pipe.hpp"
#include "srtb/pipeline/framework/composite_pipe.hpp"
#include "srtb/pipeline/framework/dummy_pipe.hpp"
#include "srtb/pipeline/framework/exit_handler.hpp"
#include "srtb/pipeline/framework/pipe.hpp"
#include "srtb/pipeline/framework/pipe_io.hpp"
#include "srtb/pipeline/read_file_pipe.hpp"
#include "srtb/pipeline/rfi_mitigation_pipe.hpp"
#include "srtb/pipeline/signal_detect_pipe.hpp"
#include "srtb/pipeline/spectrum_pipe.hpp"
#include "srtb/pipeline/udp_receiver_pipe.hpp"
#include "srtb/pipeline/unpack_pipe.hpp"
#include "srtb/pipeline/write_file_pipe.hpp"
#include "srtb/pipeline/write_signal_pipe.hpp"
#include "srtb/program_options.hpp"

#if SRTB_ENABLE_GUI
#include "srtb/gui/gui.hpp"
#endif  // SRTB_ENABLE_GUI

namespace srtb {
namespace main {

inline namespace detail {

/**
 * @brief allocate some host-side pinned memory needed before pipeline start, used in `main()`
 * 
 * It is observed that allocation of host pinned memory is very slow (0.5s - 5s for 1GB)
 * 
 * this function is quite ugly, and should be updated once pipeline structure changes.
 */
inline void allocate_memory_regions(size_t input_pipe_count) {
  const auto data_stream_count =
      srtb::io::backend_registry::get_data_stream_count(
          srtb::config.baseband_format_type);
  // hold all pointers; using RAII
  std::vector<std::shared_ptr<std::byte> > ptrs;

  for (size_t k = 0; k < input_pipe_count; k++) {
    // host side udp receiver buffer, raw baseband data
    for (size_t i = 0; i < 5; i++) {
      ptrs.push_back(srtb::host_allocator.allocate_shared<std::byte>(
          data_stream_count * sizeof(std::byte) *
          srtb::config.baseband_input_count *
          std::abs(srtb::config.baseband_input_bits) / srtb::BITS_PER_BYTE));
    }

    // host side buffer to write processed spectrum
    for (size_t i = 0; i < 2 * data_stream_count; i++) {
      ptrs.push_back(srtb::host_allocator.allocate_shared<std::byte>(
          sizeof(srtb::complex<srtb::real>) *
          (srtb::config.baseband_input_count / 2 + 1)));
    }
  }
}

}  // namespace detail

int main(int argc, char** argv) {
  srtb::changed_configs = srtb::program_options::parse_arguments(
      argc, argv, std::string(srtb::config.config_file_name));
  srtb::program_options::apply_changed_configs(srtb::changed_configs,
                                               srtb::config);

// why always have to do something special for CUDA ???
#ifdef SRTB_ENABLE_CUDA_INTEROP
  cudaSetDeviceFlags(cudaDeviceScheduleYield);
#endif  // SRTB_ENABLE_CUDA_INTEROP

  sycl::queue q;
  srtb::host_allocator = decltype(srtb::host_allocator){q};
  srtb::device_allocator = decltype(srtb::device_allocator){q};

  SRTB_LOGI << " [main] "
            << "device name = "
            << q.get_device().get_info<sycl::info::device::name>()
            << srtb::endl;

  {
    const auto nsamps_reserved = srtb::codd::nsamps_reserved();
    SRTB_LOGI << " [main] "
              << "delay time = " << srtb::codd::max_delay_time() << ", "
              << "nsamps_reserved = " << nsamps_reserved << srtb::endl;
  }

  // trigger JIT
  // some implementations may pack intermediate representation with executable binary
  // and just-in-time compile it into device code when launching first kernel
  {
    auto d_out_unique = srtb::device_allocator.allocate_unique<srtb::real>(1);
    auto d_out = d_out_unique.get();
    q.single_task([=]() { (*d_out) = srtb::real{42}; }).wait();
  }

  // init matplotlib
  {
    namespace plt = matplotlibcpp;
    // no GUI is used to show plot, so using Agg backend
    // some backend may crash because Py_Finalize is called too late
    // ref: https://github.com/lava/matplotlib-cpp/issues/248
    plt::backend("Agg");
    // plot something here to trigger creation of some resources
    // so that it can be released later in this thread
    std::vector<srtb::real> points(srtb::config.baseband_input_count /
                                   srtb::config.spectrum_channel_count / 2);
    plt::plot(points);
    plt::cla();
  }

  // work queues
  srtb::work_queue<srtb::work::copy_to_device_work, /* spsc = */ false>
      copy_to_device_queue;
  srtb::work_queue<srtb::work::unpack_work, /* spsc = */ false> unpack_queue;
  srtb::work_queue<srtb::work::fft_1d_r2c_work> fft_1d_r2c_queue;
  srtb::work_queue<srtb::work::rfi_mitigation_s1_work> rfi_mitigation_s1_queue;
  srtb::work_queue<srtb::work::dedisperse_work> dedisperse_queue;
  srtb::work_queue<srtb::work::ifft_1d_c2c_work> ifft_1d_c2c_queue;
  //srtb::work_queue<srtb::work::refft_1d_c2c_work> refft_1d_c2c_queue;
  srtb::work_queue<srtb::work::rfi_mitigation_s2_work> rfi_mitigation_s2_queue;
  srtb::work_queue<srtb::work::simplify_spectrum_work> simplify_spectrum_queue;
  //srtb::work_queue<srtb::work::draw_spectrum_work> draw_spectrum_queue;
  srtb::work_queue<srtb::work::draw_spectrum_work_2> draw_spectrum_queue_2;
  srtb::work_queue<srtb::work::signal_detect_work> signal_detect_queue;
  srtb::work_queue<srtb::work::write_signal_work> baseband_output_queue;

  /** 
   * @brief count of works in a pipeline.
   * This work count is utilized for these functions:
   * * pipeline should exit only after all works are done, especially when reading files
   * * when reading file without GUI, pipeline should exit after end of file
   * * when reading file, there should be only one file in pipeline, to reduce VRAM usage
   */
  std::shared_ptr<std::atomic<int32_t> > work_in_pipeline_count =
      std::make_shared<std::atomic<int32_t> >(0);
  auto increase_work_count = [work_in_pipeline_count](
                                 [[maybe_unused]] std::stop_token stop_token,
                                 [[maybe_unused]] auto work) {
    // take in account that unpack_pipe may generate more than 1 works per input work
    (*work_in_pipeline_count) +=
        srtb::io::backend_registry::get_data_stream_count(
            srtb::config.baseband_format_type);
    assert((*work_in_pipeline_count) >= 0);
  };
  auto decrease_work_count = [work_in_pipeline_count](
                                 [[maybe_unused]] std::stop_token stop_token,
                                 [[maybe_unused]] auto work) {
    (*work_in_pipeline_count)--;
    assert((*work_in_pipeline_count) >= 0);
  };

  // setup threads for pipes
  using namespace srtb::pipeline;

  std::jthread copy_to_device_thread =
      srtb::pipeline::start_pipe<copy_to_device_pipe>(
          q, queue_in_functor{copy_to_device_queue},
          queue_out_functor{unpack_queue});

  std::jthread unpack_thread = srtb::pipeline::start_unpack_pipe(
      srtb::config.baseband_format_type, q, queue_in_functor{unpack_queue},
      queue_out_functor{fft_1d_r2c_queue});

  std::jthread fft_1d_r2c_thread = srtb::pipeline::start_pipe<fft_1d_r2c_pipe>(
      q, queue_in_functor{fft_1d_r2c_queue},
      queue_out_functor{rfi_mitigation_s1_queue});

  std::jthread rfi_mitigation_s1_thread =
      srtb::pipeline::start_pipe<rfi_mitigation_s1_pipe>(
          q, queue_in_functor{rfi_mitigation_s1_queue},
          queue_out_functor{dedisperse_queue});

  std::jthread dedisperse_thread = srtb::pipeline::start_pipe<dedisperse_pipe>(
      q, queue_in_functor{dedisperse_queue},
      queue_out_functor{ifft_1d_c2c_queue});

  //std::jthread ifft_1d_c2c_thread =
  //    srtb::pipeline::start_pipe<ifft_1d_c2c_pipe>(
  //        q,
  //        queue_in_functor{ifft_1d_c2c_queue},
  //        queue_out_functor{refft_1d_c2c_queue});

  //std::jthread refft_1d_c2c_thread =
  //    srtb::pipeline::start_pipe<refft_1d_c2c_pipe>(
  //        q,
  //        queue_in_functor{refft_1d_c2c_queue},
  //        queue_out_functor{signal_detect_queue});

  std::jthread watfft_1d_c2c_thread =
      srtb::pipeline::start_pipe<watfft_1d_c2c_pipe>(
          q, queue_in_functor{ifft_1d_c2c_queue},
          queue_out_functor{rfi_mitigation_s2_queue});

  //std::jthread signal_detect_thread =
  //    srtb::pipeline::start_pipe<signal_detect_pipe>(
  //        q,
  //        queue_in_functor{srtb::signal_detect_queue},
  //        multiple_out_functors_functor{
  //            queue_out_functor{srtb::baseband_output_queue},
  //            queue_out_functor{
  //                srtb::simplify_spectrum_queue}});

  std::jthread rfi_mitigation_s2_thread =
      srtb::pipeline::start_pipe<rfi_mitigation_s2_pipe>(
          q, queue_in_functor{rfi_mitigation_s2_queue},
          multiple_out_functors_functor{
              queue_out_functor{signal_detect_queue},
              loose_queue_out_functor{simplify_spectrum_queue}});

  std::jthread signal_detect_thread =
      srtb::pipeline::start_pipe<signal_detect_pipe_2>(
          q, queue_in_functor{signal_detect_queue},
          queue_out_functor{baseband_output_queue});

  std::jthread baseband_output_thread;
  if (srtb::config.baseband_write_all) {
    SRTB_LOGW << " [main] "
              << "Writing all baseband data, take care of disk space!"
              << srtb::endl;
    srtb::config.udp_receiver_can_restart = false;
    baseband_output_thread = srtb::pipeline::start_pipe<write_file_pipe>(
        q, queue_in_functor{baseband_output_queue}, decrease_work_count);
  } else {
    // catch & write mode ("piggybank" ?)
    srtb::config.udp_receiver_can_restart = true;
    baseband_output_thread = srtb::pipeline::start_pipe<write_signal_pipe>(
        q, queue_in_functor{baseband_output_queue}, decrease_work_count);
  }

  std::jthread simplify_spectrum_thread;
  if (SRTB_ENABLE_GUI && srtb::config.gui_enable) {
    //simplify_spectrum_thread =
    //    srtb::pipeline::start_pipe<simplify_spectrum_pipe>(
    //        q,
    //        queue_in_functor{srtb::simplify_spectrum_queue},
    //        queue_out_functor{srtb::draw_spectrum_queue});
    simplify_spectrum_thread =
        srtb::pipeline::start_pipe<simplify_spectrum_pipe_2>(
            q, queue_in_functor{simplify_spectrum_queue},
            queue_out_functor{draw_spectrum_queue_2});
  } else {
    simplify_spectrum_thread = srtb::pipeline::start_pipe<dummy_pipe<> >(
        q, queue_in_functor{simplify_spectrum_queue},
        dummy_out_functor<srtb::work::dummy_work>{});
  }

  // TODO: maybe multiple file input or something else
  size_t input_pipe_count =
      std::max(std::max(srtb::config.udp_receiver_sender_address.size(),
                        srtb::config.udp_receiver_sender_port.size()),
               size_t{1});

  std::vector<std::jthread> input_thread;
  auto input_file_path = srtb::config.input_file_path;
  if (std::filesystem::exists(input_file_path)) {
    SRTB_LOGI << " [main] "
              << "Reading file " << input_file_path << srtb::endl;
    input_thread.push_back(srtb::pipeline::start_pipe<read_file_pipe>(
        q,
        // wait until last work finishes
        [work_in_pipeline_count](std::stop_token stop_token) {
          while (
              (!(stop_token.stop_possible() && stop_token.stop_requested())) &&
              (*work_in_pipeline_count) != 0) {
            std::this_thread::sleep_for(std::chrono::nanoseconds(
                srtb::config.thread_query_work_wait_time));
          }
          if (stop_token.stop_requested()) {
            return std::optional<srtb::work::dummy_work>{};
          }
          return std::optional{srtb::work::dummy_work{}};
        },
        multiple_out_functors_functor{queue_out_functor{copy_to_device_queue},
                                      increase_work_count}));
    input_pipe_count = 1;
  } else {
    if (input_file_path != "") {
      throw std::runtime_error{" [main] Cannot read file " + input_file_path};
    }
    SRTB_LOGI << " [main] "
              << "Receiving UDP packets" << srtb::endl;
    allocate_memory_regions(input_pipe_count);
    for (size_t i = 0; i < input_pipe_count; i++) {
      input_thread.push_back(srtb::pipeline::start_udp_receiver_pipe(
          srtb::config.baseband_format_type, q,
          // don't wait for last work
          [work_in_pipeline_count](
              [[maybe_unused]] std::stop_token stop_token) {
            (*work_in_pipeline_count)++;
            return std::optional{srtb::work::dummy_work{}};
          },
          multiple_out_functors_functor{queue_out_functor{copy_to_device_queue},
                                        increase_work_count},
          /* id = */ i));
    }
  }

  std::vector<std::jthread> threads;
  threads = std::move(input_thread);
  threads.push_back(std::move(copy_to_device_thread));
  threads.push_back(std::move(unpack_thread));
  threads.push_back(std::move(fft_1d_r2c_thread));
  threads.push_back(std::move(rfi_mitigation_s1_thread));
  threads.push_back(std::move(dedisperse_thread));
  //threads.push_back(std::move(ifft_1d_c2c_thread));
  //threads.push_back(std::move(refft_1d_c2c_thread));
  threads.push_back(std::move(watfft_1d_c2c_thread));
  threads.push_back(std::move(rfi_mitigation_s2_thread));
  threads.push_back(std::move(signal_detect_thread));
  threads.push_back(std::move(baseband_output_thread));
  threads.push_back(std::move(simplify_spectrum_thread));

  int return_value;

#if SRTB_ENABLE_GUI
  if (srtb::config.gui_enable) {
    return_value = srtb::gui::show_gui(argc, argv, draw_spectrum_queue_2);
  } else {
#endif  // SRTB_ENABLE_GUI

    // wait for first work
    while ((*work_in_pipeline_count) == 0) {
      std::this_thread::sleep_for(
          std::chrono::nanoseconds(srtb::config.thread_query_work_wait_time));
    }

    // TODO: assume threads at beginning are work producers
    for (size_t i = 0; i < input_pipe_count; i++) {
      if (threads.at(i).joinable()) {
        threads.at(i).join();
      }
    }

    // wait for last work
    while ((*work_in_pipeline_count) != 0) {
      std::this_thread::sleep_for(
          std::chrono::nanoseconds(srtb::config.thread_query_work_wait_time));
    }

    return_value = EXIT_SUCCESS;

#if SRTB_ENABLE_GUI
  }
#endif  // SRTB_ENABLE_GUI

  srtb::pipeline::on_exit(std::move(threads));

  SRTB_LOGI << " [main] "
            << "Exiting." << srtb::endl;

  // de-init matplotlibcpp
  // https://stackoverflow.com/questions/67533541/py-finalize-resulting-in-segmentation-fault-for-python-3-9-but-not-for-python
  matplotlibcpp::detail::_interpreter::kill();

  return return_value;
}

}  // namespace main
}  // namespace srtb

int main(int argc, char** argv) { return srtb::main::main(argc, argv); }
