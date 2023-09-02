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
#include "srtb/pipeline/framework/pipe.hpp"
#include "srtb/pipeline/framework/pipe_io.hpp"
#include "srtb/pipeline/udp_receiver_pipe.hpp"
#include "srtb/pipeline/write_file_pipe.hpp"
#include "srtb/program_options.hpp"

namespace srtb {
namespace main {

/**
 * @brief This program receives baseband from UDP port and write to single file
 *        TODO: need test with real data
 */
int baseband_receiver(int argc, char** argv) {
  srtb::changed_configs = srtb::program_options::parse_arguments(
      argc, argv, std::string(srtb::config.config_file_name));
  srtb::program_options::apply_changed_configs(srtb::changed_configs,
                                               srtb::config);

  sycl::queue q;
  srtb::work_queue<srtb::work::baseband_output_work> baseband_output_queue;
  std::jthread udp_receiver_thread =
      srtb::pipeline::start_pipe<srtb::pipeline::udp_receiver_pipe>(
          q, srtb::pipeline::dummy_in_functor{},
          srtb::pipeline::queue_out_functor{baseband_output_queue});
  std::jthread baseband_output_thread =
      srtb::pipeline::start_pipe<srtb::pipeline::write_file_pipe>(
          q, srtb::pipeline::queue_in_functor{baseband_output_queue},
          srtb::pipeline::dummy_out_functor{});
  return EXIT_SUCCESS;
}

}  // namespace main
}  // namespace srtb

int main(int argc, char** argv) {
  return srtb::main::baseband_receiver(argc, argv);
}
