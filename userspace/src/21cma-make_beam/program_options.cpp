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

#include <boost/algorithm/string/trim.hpp>
#include <boost/program_options.hpp>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "3rdparty/jdate_clock.hpp"
#include "assert.hpp"
#include "common.hpp"
#include "exprgrammar.h"
#include "srtb/log/log.hpp"

namespace srtb::_21cma::make_beam {

inline namespace detail {

auto read_lines_in_file(std::string file_path) -> std::vector<std::string> {
  std::ifstream file_stream{file_path};
  BOOST_ASSERT_MSG(file_stream, ("[read_lines_in_file] Cannot read " + file_path).c_str());

  std::vector<std::string> lines;
  {
    std::string line;
    while (std::getline(file_stream, line)) {
      if ((!(line.starts_with("#") || line.starts_with("//"))) && line.size() > 0) {
        lines.push_back(std::move(line));
      }
    }
  }

  return lines;
}

inline auto parse_number(const std::string& expression) -> double {
  const char* iter = expression.c_str();
  const char* end = iter + expression.size();
  return exprgrammar_parse_double(iter, end);
}

auto parse_ra_dec(std::string s) -> sky_coord_t {
  sky_coord_t sky_coord;
  std::regex regex{R"(^(\d{2}):(\d{2}):(\d{2}(\.\d+)?)(_|\s+)(\+|-)?(\d{2}):(\d{2}):(\d{2}(\.\d+)?)$)"};
  std::smatch match_result;
  bool match_success = std::regex_match(s, match_result, regex);
  if (!match_success) [[unlikely]] {
    throw std::invalid_argument("Cannot parse RA Dec: " + s);
  }

  // match_result[0 is whole string result
  const double ra_h = std::stod(match_result[1]);
  const double ra_m = std::stod(match_result[2]);
  const double ra_s = std::stod(match_result[3]);
  // match[4]: ra_s fraction part
  // match[5]: delimiter
  const double dec_sgn = (match_result[6] == "-") ? -1 : +1;
  const double dec_deg = std::stod(match_result[7]);
  const double dec_min = std::stod(match_result[8]);
  const double dec_sec = std::stod(match_result[9]);
  // match[10]: dec_sec fraction part
  // match[11:13]: ???

  if (!(0 <= ra_h && ra_h < 24)) [[unlikely]] {
    throw std::invalid_argument("RA hour out of range: " + s);
  }
  if (!(0 <= ra_m && ra_m < 60)) [[unlikely]] {
    throw std::invalid_argument("RA minute out of range: " + s);
  }
  if (!(0 <= ra_s && ra_s < 60)) [[unlikely]] {
    throw std::invalid_argument("RA second out of range: " + s);
  }
  if (!(0 <= dec_deg && dec_deg <= 90)) [[unlikely]] {
    throw std::invalid_argument("Dec degree out of range: " + s);
  }
  if (!(0 <= dec_min && dec_min < 60)) [[unlikely]] {
    throw std::invalid_argument("Dec minute out of range: " + s);
  }
  if (!(0 <= dec_sec && dec_sec < 60)) [[unlikely]] {
    throw std::invalid_argument("Dec second out of range: " + s);
  }

  sky_coord.ra_hour = ra_h + (ra_m + (ra_s / 60)) / 60;
  sky_coord.dec_deg = dec_sgn * (dec_deg + (dec_min + (dec_sec / 60)) / 60);

  return sky_coord;
}

}  // namespace detail

namespace program_options {

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
    ("pointings,P", po::value<std::string>(),
     "File containing RA DEC, format: \"hh:mm:ss.s dd:mm:ss.s\". ")
    ("begin,b", po::value<std::string>(),
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

  // set log level
  {
    srtb::log::log_level = static_cast<srtb::log::levels>(vm["log_level"].as<int>());
    SRTB_LOGI << " [program_options] " << "log_level" << " = " << static_cast<int>(srtb::log::log_level) << srtb::endl;
  }

  // set observation_mode
  {
    if (vm.count("drifting")) {
      cfg.observation_mode = srtb::_21cma::make_beam::config::observation_mode_t::DRIFTING;
      SRTB_LOGI << " [program_options] " << "observation_mode" << " = " << "DRIFTING" << srtb::endl;

    } else {
      cfg.observation_mode = srtb::_21cma::make_beam::config::observation_mode_t::TRACKING;
      SRTB_LOGI << " [program_options] " << "observation_mode" << " = " << "TRACKING" << srtb::endl;
    }
  }

  // read file lists
  {
    std::vector<std::vector<std::string>> file_list;
    std::string meta_file_list = vm["meta_file_list"].as<std::string>();
    SRTB_LOGI << " [21cma-make_beam] " << "Reading file list metadata: " << meta_file_list << srtb::endl;
    std::vector<std::string> file_lists = read_lines_in_file(meta_file_list);
    auto base_folder = std::filesystem::absolute(std::filesystem::path{meta_file_list}).parent_path();
    BOOST_ASSERT(std::filesystem::exists(base_folder));
    for (size_t i = 0; i < file_lists.size(); i++) {
      std::filesystem::path path = base_folder / file_lists[i];
      SRTB_LOGI << " [21cma-make_beam] " << "Reading file list: " << path.string() << srtb::endl;
      std::vector<std::string> files = read_lines_in_file(path);
      file_list.push_back(files);
      BOOST_ASSERT_MSG(
          files.size() == file_list[0].size(),
          ("[21cma-make_beam] [main] file count mismatch: " + path.string() + "(" + std::to_string(files.size()) + ")" +
           ", " + file_lists[0] + "(" + std::to_string(file_list[0].size()) + ")")
              .c_str());
    }
    std::vector<std::future<void>> future{file_lists.size()};

    cfg.baseband_file_list = std::move(file_list);
  }

  // read pointing file
  {
    std::vector<std::string> pointing_str = read_lines_in_file(vm["pointings"].as<std::string>());
    std::vector<sky_coord_t> pointing{pointing_str.size()};
    std::stringstream ss;
    ss << " [program_options] " << "Pointings: " << "{";
    for (size_t i = 0; i < pointing_str.size(); i++) {
      std::string str = pointing_str[i];
      boost::algorithm::trim(str);
      pointing.at(i) = parse_ra_dec(str);
      ss << "{" << pointing.at(i).ra_hour << ", " << pointing.at(i).dec_deg << "}" << ", ";
    }
    ss << "}" << srtb::endl;
    cfg.pointing = std::move(pointing);
    SRTB_LOGI << ss.str();
  }

  // read begin time
  {
    constexpr auto date_format = "%Y-%m-%d_%H-%M-%S";
    const std::string begin_str = vm["begin"].as<std::string>();
    std::stringstream ss;
    ss << begin_str;
    std::tm tm;
    ss >> std::get_time(&tm, date_format);
    time_t time = timegm(&tm);
    std::chrono::time_point<std::chrono::system_clock, std::chrono::duration<double>> sys_tp =
        std::chrono::system_clock::from_time_t(time);
    auto jdate_tp = jdate_clock::from_sys(sys_tp);
    constexpr uint32_t seconds_in_day = 60 * 60 * 24;
    const double jd = jdate_tp.time_since_epoch().count() / seconds_in_day;
    const double mjd = jd - 2400000.5;
    cfg.start_mjd = mjd;
    SRTB_LOGI << " [program_options] " << "Input time = " << begin_str << ", MJD = " << mjd << srtb::endl;
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
