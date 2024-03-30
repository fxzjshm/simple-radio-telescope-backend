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

#include "program_options.hpp"

#include <boost/program_options.hpp>
#include <cstdlib>
#include <fstream>

#include "exprgrammar.h"
#include "srtb/log/log.hpp"

namespace srtb::_21cma::make_beam {

inline namespace detail {

auto read_lines_in_file(std::string file_path) -> std::vector<std::string> {
  std::ifstream file_stream{file_path};
  BOOST_ASSERT_MSG(file_stream, ("[read_lines_in_file] Cannot read meta_file_list: " + file_path).c_str());

  std::vector<std::string> lines;
  {
    std::string line;
    while (std::getline(file_stream, line)) {
      if ((!line.starts_with('#')) && line.size() > 0) {
        lines.push_back(std::move(line));
      }
    }
  }

  return lines;
}

}  // namespace detail

namespace program_options {

inline auto parse_number(const std::string& expression) -> double {
  const char* iter = expression.c_str();
  const char* end = iter + expression.size();
  return exprgrammar_parse_double(iter, end);
}

auto parse_cmdline(int argc, char** argv) {
  namespace po = boost::program_options;
  po::options_description general_option("General Options");
  // clang-format off
/*
  template:
  ("config_name", po::value<std::string>(),
   "decsription")
*/
  general_option.add_options()
    ("log_level", po::value<int>()->default_value(static_cast<int>(srtb::log::levels::INFO)),
     "Debug level for console log output. ")
    ("P,pointings", po::value<std::string>(),
     "File containing RA DEC, format: \"hh:mm:ss.s dd:mm:ss.s\". ")
    ("b,begin", po::value<std::string>(),
     "Begin time in UTC, format: \"yyyy-mm-dd_hh-mm-ss\". ")
    ("drifting",
     "If set, use drifting scan observation mode. ")
    ("meta_file_list", po::value<std::string>(),
     "File containing file of file lists. ")
  ;
  // clang-format on

  boost::program_options::variables_map vm;
  boost::program_options::store(boost::program_options::command_line_parser(argc, argv).options(general_option).run(),
                                vm);

  if (vm.count("help")) {
    SRTB_LOGI << " [program_options] " << "Command line options:" << srtb::endl << general_option << srtb::endl;
    std::exit(EXIT_SUCCESS);
  }
  return vm;
}

auto set_config(boost::program_options::variables_map vm) {
  srtb::_21cma::make_beam::config cfg;

  srtb::log::log_level = vm["log_level"].as<srtb::log::levels>();
  SRTB_LOGI << " [program_options] " << "log_level" << " = " << srtb::log::log_level << srtb::endl;

  if (vm.count("drifting")) {
    cfg.observation_mode = srtb::_21cma::make_beam::config::observation_mode_t::DRIFTING;
  } else {
    cfg.observation_mode = srtb::_21cma::make_beam::config::observation_mode_t::TRACKING;
  }

  // read file lists
  {
    std::vector<std::vector<std::string>> file_list;
    std::string meta_file_list = vm["meta_file_list"].as<std::string>();
    SRTB_LOGI << "[21cma-make_beam] " << "Reading file list metadata: " << meta_file_list << srtb::endl;
    std::vector<std::string> file_lists = read_lines_in_file(meta_file_list);
    for (size_t i = 0; i < file_lists.size(); i++) {
      SRTB_LOGI << "[21cma-make_beam] " << "Reading file list: " << file_lists[i] << srtb::endl;
      std::vector<std::string> files = read_lines_in_file(file_lists[i]);
      file_list.push_back(files);
      BOOST_ASSERT_MSG(
          files.size() == file_list[0].size(),
          ("[21cma-make_beam] [main] file count mismatch: " + file_lists[i] + "(" + std::to_string(files.size()) + ")" +
           ", " + file_lists[0] + "(" + std::to_string(file_list[0].size()) + ")")
              .c_str());
    }
    cfg.baseband_file_list = std::move(file_list);
  }
  return cfg;
}

auto parse(int argc, char** argv) -> config {
  auto vm = parse_cmdline(argc, argv);
  config cfg = set_config(vm);
  return cfg;
}

}  // namespace program_options

}  // namespace srtb::_21cma::make_beam
