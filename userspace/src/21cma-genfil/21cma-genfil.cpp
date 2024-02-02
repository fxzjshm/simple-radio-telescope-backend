/******************************************************************************* 
 * Copyright (c) 2023 fxzjshm
 * This software is licensed under Mulan PubL v2.
 * You can use this software according to the terms and conditions of the Mulan PubL v2.
 * You may obtain a copy of Mulan PubL v2 at:
 *          http://license.coscl.org.cn/MulanPubL-2.0
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PubL v2 for more details.
 ******************************************************************************/

#include <cstdlib>
#include <optional>

#include "srtb/commons.hpp"
#include "srtb/pipeline/copy_to_device_pipe.hpp"
#include "srtb/pipeline/fft_pipe.hpp"
#include "srtb/pipeline/framework/composite_pipe.hpp"
#include "srtb/pipeline/framework/pipe.hpp"
#include "srtb/pipeline/framework/pipe_io.hpp"
#include "srtb/pipeline/udp_receiver_pipe.hpp"
#include "srtb/pipeline/unpack_pipe.hpp"
#include "srtb/program_options.hpp"

// ---

#include "21cma_genfil_work.hpp"
#include "dynspec_pipe.hpp"
#include "fft_r2c_post_process_pipe.hpp"
#include "write_multi_filterbank_pipe.hpp"

namespace srtb {

namespace work {

using genfil_21cma_cast_work = fft_1d_r2c_work;

}

namespace pipeline {

class genfil_21cma_cast_pipe {
 public:
  genfil_21cma_cast_pipe() = default;
  genfil_21cma_cast_pipe([[maybe_unused]] sycl::queue q) {}

  auto operator()([[maybe_unused]] std::stop_token stop_token,
                  srtb::work::genfil_21cma_cast_work genfil_21cma_cast_work) {
    srtb::work::fft_1d_c2c_work fft_1d_c2c_work;
    fft_1d_c2c_work.move_parameter_from(std::move(genfil_21cma_cast_work));
    const size_t nchan = srtb::config.spectrum_channel_count;
    const size_t nsamps = genfil_21cma_cast_work.count / nchan / 2;
    fft_1d_c2c_work.ptr =
        std::reinterpret_pointer_cast<srtb::complex<srtb::real> >(
            genfil_21cma_cast_work.ptr);
    // implicitly *2 in size here, as reinterpret casted to complex
    fft_1d_c2c_work.count = nchan;
    fft_1d_c2c_work.batch_size = nsamps;
    return fft_1d_c2c_work;
  }
};

}  // namespace pipeline

namespace main {

/**
 * @brief This program receives baseband from UDP port, generate dynamic spectrum,
 *        compress it and write to filterbank.
 *        TODO: need test with real data
 */
int genfil_21cma(int argc, char** argv) {
  srtb::config.udp_receiver_can_restart = false;
  srtb::config.baseband_format_type = "naocpsr_snap1";
  srtb::changed_configs = srtb::program_options::parse_arguments(
      argc, argv, std::string(srtb::config.config_file_name));
  srtb::program_options::apply_changed_configs(srtb::changed_configs,
                                               srtb::config);

  sycl::queue q;
  SRTB_LOGI << " [main] "
            << "device name = "
            << q.get_device().get_info<sycl::info::device::name>()
            << srtb::endl;

  srtb::work_queue<srtb::work::copy_to_device_work> copy_to_device_queue;
  srtb::work_queue<srtb::work::genfil_21cma_cast_work> genfil_21cma_cast_queue;
  srtb::work_queue<srtb::work::dynspec_work> dynspec_queue;
  srtb::work_queue<srtb::work::write_multi_filterbank_work>
      write_multi_filterbank_queue;

  using namespace srtb::pipeline;
  std::jthread udp_receiver_thread =
      srtb::pipeline::start_pipe<udp_receiver_pipe>(
          q, dummy_in_functor{}, queue_out_functor{copy_to_device_queue});

  std::jthread unpack_thread = srtb::pipeline::start_pipe<
      composite_pipe<copy_to_device_pipe, unpack_interleaved_samples_2_pipe> >(
      q, queue_in_functor{copy_to_device_queue},
      multiple_works_out_functor{queue_out_functor{genfil_21cma_cast_queue}});

  std::jthread fft_thread = srtb::pipeline::start_pipe<composite_pipe<
      genfil_21cma_cast_pipe, fft_1d_c2c_pipe, fft_r2c_post_process_pipe> >(
      q, queue_in_functor{genfil_21cma_cast_queue},
      queue_out_functor{dynspec_queue});

  std::jthread dynspec_thread = srtb::pipeline::start_pipe<dynspec_pipe>(
      q, queue_in_functor{dynspec_queue},
      queue_out_functor{write_multi_filterbank_queue});

  std::jthread write_multi_filterbank_thread =
      srtb::pipeline::start_pipe<write_multi_filterbank_pipe>(
          q, queue_in_functor{write_multi_filterbank_queue},
          dummy_out_functor{});

  while (true) {
    std::this_thread::sleep_for(
        std::chrono::nanoseconds(srtb::config.thread_query_work_wait_time));
  }

  return EXIT_SUCCESS;
}

}  // namespace main
}  // namespace srtb

int main(int argc, char** argv) { return srtb::main::genfil_21cma(argc, argv); }
