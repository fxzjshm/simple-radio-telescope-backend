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
#include "srtb/pipeline/baseband_output_pipe.hpp"
#include "srtb/pipeline/pipe.hpp"
#include "srtb/pipeline/udp_receiver_pipe.hpp"
#include "srtb/program_options.hpp"

namespace srtb {
namespace main {

int baseband_receiver(int argc, char** argv) {
  srtb::changed_configs = srtb::program_options::parse_arguments(
      argc, argv, std::string(srtb::config.config_file_name));
  srtb::program_options::apply_changed_configs(srtb::changed_configs,
                                               srtb::config);

  sycl::queue q = srtb::queue;
  std::jthread udp_receiver_thread =
      srtb::pipeline::start_pipe<srtb::pipeline::udp_receiver_pipe>(
          q,
          []([[maybe_unused]] std::stop_token) { return std::optional{true}; },
          srtb::pipeline::queue_out_functor{srtb::baseband_output_queue});
  std::jthread baseband_output_thread =
      srtb::pipeline::start_pipe<srtb::pipeline::baseband_output_pipe<true> >(
          q, srtb::pipeline::queue_in_functor{srtb::baseband_output_queue},
          []([[maybe_unused]] std::stop_token, [[maybe_unused]] bool) {});
  return EXIT_SUCCESS;
}

}  // namespace main
}  // namespace srtb

int main(int argc, char** argv) {
  return srtb::main::baseband_receiver(argc, argv);
}
