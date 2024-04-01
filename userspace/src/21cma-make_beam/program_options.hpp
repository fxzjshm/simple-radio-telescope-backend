/*******************************************************************************
 * Copyright (c) 2024 fxzjshm
 * 21cma-make_beam is licensed under Mulan PubL v2.
 * You can use this software according to the terms and conditions of the Mulan PubL v2.
 * You may obtain a copy of Mulan PubL v2 at:
 *          http://license.coscl.org.cn/MulanPubL-2.0
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PubL v2 for more details.
 ******************************************************************************/

#pragma once
#ifndef __SRTB_21CMA_MAKE_BEAM_PROGRAM_OPTIONS__
#define __SRTB_21CMA_MAKE_BEAM_PROGRAM_OPTIONS__

#include <string>
#include <vector>

#include "common.hpp"

namespace srtb::_21cma::make_beam {

struct config {
  enum observation_mode_t { TRACKING, DRIFTING } observation_mode;

  std::vector<std::vector<std::string>> baseband_file_list;
  std::vector<sky_coord_t> pointing;
  double start_mjd;
};

namespace program_options {

auto parse(int argc, char** argv) -> config;

}  // namespace program_options

}  // namespace srtb::_21cma::make_beam

#endif  // __SRTB_21CMA_MAKE_BEAM_GLOBAL_VARIABLES__
