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
#include "srtb/pipeline/framework/composite_pipe.hpp"
#include "srtb/pipeline/framework/pipe.hpp"
#include "srtb/pipeline/framework/pipe_io.hpp"
#include "srtb/pipeline/udp_receiver_pipe.hpp"
#include "srtb/pipeline/write_file_pipe.hpp"
#include "srtb/program_options.hpp"

namespace srtb {

namespace work {

using baseband_receiver_cast_work =
    srtb::work::work<std::shared_ptr<std::byte>>;

}

namespace pipeline {

class baseband_receiver_cast_pipe {
 public:
  baseband_receiver_cast_pipe() = default;
  baseband_receiver_cast_pipe([[maybe_unused]] sycl::queue q) {}

  auto operator()([[maybe_unused]] std::stop_token stop_token,
                  srtb::work::baseband_receiver_cast_work in_work) {
    srtb::work::write_file_work out_work;
    out_work.move_parameter_from(std::move(in_work));
    // don't need to care about other things
    return out_work;
  }
};

}  // namespace pipeline

namespace main {

/**
 * @brief This program receives baseband from UDP port and write to single file
 *        TODO: need test with real data
 */
int baseband_receiver(int argc, char** argv) {
  srtb::config.udp_receiver_can_restart = true;
  srtb::changed_configs = srtb::program_options::parse_arguments(
      argc, argv, std::string(srtb::config.config_file_name));
  srtb::program_options::apply_changed_configs(srtb::changed_configs,
                                               srtb::config);

  sycl::queue q;

  // type mismatch; manual cast required
  srtb::work_queue<srtb::work::copy_to_device_work> write_file_queue;

  using namespace srtb::pipeline;
  std::jthread udp_receiver_thread =
      srtb::pipeline::start_pipe<udp_receiver_pipe>(
          q, dummy_in_functor{}, queue_out_functor{write_file_queue});
  std::jthread baseband_output_thread = srtb::pipeline::start_pipe<
      composite_pipe<baseband_receiver_cast_pipe, write_file_pipe>>(
      q, queue_in_functor{write_file_queue}, dummy_out_functor{});

  return EXIT_SUCCESS;
}

}  // namespace main
}  // namespace srtb

int main(int argc, char** argv) {
  return srtb::main::baseband_receiver(argc, argv);
}
