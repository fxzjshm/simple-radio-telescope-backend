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
#ifndef __SRTB_PROGRAM_OPTIONS__
#define __SRTB_PROGRAM_OPTIONS__

#include <boost/program_options.hpp>
#include <filesystem>
#include <string>

#include "srtb/config.hpp"
#include "srtb/log/log.hpp"
#include "suzerain/exprgrammar.hpp"

namespace srtb {
namespace program_options {

/**
 * @brief Read configs from command line & file.
 * @note sync configs & descriptions here and in srtb::configs
 * @return std::map<std::string, std::string> key-value pairs of changed options
 */
[[nodiscard]] inline auto parse_arguments(
    int argc, char** argv, const std::string& default_config_file_name)
    -> std::map<std::string, std::string> {
  // TODO parse arguments
  // here values are stored in std::string to be evaluated later.
  boost::program_options::options_description general_option("General Options"),
      baseband_option("Baseband Options"),
      data_io_option("Data Input/Output Options"),
      udp_receiver_options("UDP Receiver Options"),
      file_io_options("File Input/Output Options"),
      operation_option("Operation Options"),
      cmd_only_options("Command Line Only Options"),
      in_file_options("Options available in config file"),
      all_option("Options");
  /* clang-format off */
    /*
      ("config_name", boost::program_options::value<std::string>(),
       "decsription")
    */
    cmd_only_options.add_options()
      ("help,h", "Show help message")
      ("config_file_name", boost::program_options::value<std::string>(),
       "Path to config file to be used to read other configs.")
    ;
    general_option.add_options()
      ("log_level", boost::program_options::value<std::string>(),
       "Debug level for console log output.")
      ("thread_query_work_wait_time", boost::program_options::value<std::string>(),
       "Wait time in naneseconds for a thread to sleep if it fails to get work. "
       "Trade off between CPU usage (most are wasted) and pipeline latency.")
    ;
    baseband_option.add_options()
      ("baseband_input_count", boost::program_options::value<std::string>(),
       "Count of data to be transferred to GPU for once processing, in sample counts. "
       "Should be power of 2 so that FFT and channelizing can work properly.")
      ("baseband_input_bits", boost::program_options::value<std::string>(),
       "Length of a single input data, used in unpack.")
      ("baseband_freq_low", boost::program_options::value<std::string>(),
       "Lowerest frequency of received baseband signal, in MHz.")
      ("baseband_bandwidth", boost::program_options::value<std::string>(),
       "Band width of received baseband signal, in MHz.")
      ("baseband_sample_rate", boost::program_options::value<std::string>(),
       "Baseband sample rate, in samples / second."
       "Should be 2 * baseband_bandwidth (* 1e6 because of unit) if Nyquist rate.")
    ;
    udp_receiver_options.add_options()
      ("udp_receiver_buffer_size", boost::program_options::value<std::string>(),
       "Buffer size of socket for receving udp packet.")
      ("udp_receiver_sender_address", boost::program_options::value<std::string>(),
       "Address to receive baseband UDP packets")
      ("udp_receiver_sender_port", boost::program_options::value<std::string>(),
       "Port to receive baseband UDP packets")
      ("udp_receiver_cpu_preferred", boost::program_options::value<std::string>(),
       "CPU core that UDP receiver should be bound to.")
    ;
    file_io_options.add_options()
      ("input_file_path", boost::program_options::value<std::string>(),
       "Path to the binary file to be read as baseband input.")
      ("input_file_offset_bytes", boost::program_options::value<std::string>(),
       "Skip some data before reading in, usually avoids header")
      ("baseband_output_file_prefix", boost::program_options::value<std::string>(),
       "Prefix of saved baseband data. Full name will be ${prefix}_${counter}.bin")
      ("baseband_write_all", boost::program_options::value<std::string>(),
       "if true, record all baseband into one file per polarization; "
       "if false, write only those with signal detected.")
    ;
    operation_option.add_options()
      ("dm,dedisperse_measurement", boost::program_options::value<std::string>(),
       "Target dispersion measurement for coherent dedispersion.")
      ("fft_fftw_wisdom_path", boost::program_options::value<std::string>(),
       "Location to save fftw wisdom.")
      ("mitigate_rfi_threshold", boost::program_options::value<std::string>(),
       "Temporary threshold for RFI mitigation. Channels with signal stronger "
       "than this threshold * average strength will be set to 0 .")
      ("mitigate_rfi_freq_list", boost::program_options::value<std::string>(),
       "list of frequency pairs to zap/remove, "
       "format: 11-12, 15-90, 233-235, 1176-1177 (arbitary values)")
      ("refft_length", boost::program_options::value<std::string>(),
       "Length of FFT for re-constructing signals after coherent dedispersion, "
       "of complex numbers, so refft_length <= baseband_input_count / 2")
      ("signal_detect_threshold", boost::program_options::value<std::string>(),
       "threshold for signal detect, target signal / noise ratio")
    ;
  /* clang-format on */
  data_io_option.add(udp_receiver_options).add(file_io_options);
  in_file_options.add(general_option)
      .add(baseband_option)
      .add(data_io_option)
      .add(operation_option);
  all_option.add(cmd_only_options).add(in_file_options);

  // ref: https://www.boost.org/doc/libs/1_80_0/libs/program_options/example/multiple_sources.cpp
  // here: command line > config file > default config
  // the first read config is used, so read command line first
  boost::program_options::variables_map vm;
  boost::program_options::store(
      boost::program_options::command_line_parser(argc, argv)
          .options(all_option)
          .run(),
      vm);
  std::string config_file_name;
  if (vm.contains("config_file_name")) {
    config_file_name = vm["config_file_name"].as<std::string>();
  } else {
    config_file_name = default_config_file_name;
  }
  if (std::filesystem::exists(config_file_name)) {
    boost::program_options::notify(vm);
    boost::program_options::store(
        boost::program_options::parse_config_file(config_file_name.c_str(),
                                                  in_file_options),
        vm);
    boost::program_options::notify(vm);
  } else {
    SRTB_LOGW << " [program_options] "
              << "config file " << config_file_name << " (absolute path "
              << std::filesystem::absolute(config_file_name) << ") not found."
              << srtb::endl;
  }

  if (vm.count("help")) {
    SRTB_LOGI << " [program_options] "
              << "Command line options:" << srtb::endl;
    srtb::log::sync_stream_wrapper{std::cout} << all_option << srtb::endl;
    std::exit(0);
  }

  std::map<std::string, std::string> changed_configs;
  for (auto item : vm) {
    changed_configs[item.first] = item.second.as<std::string>();
  }
  return changed_configs;
}

// assume numeric literals in config files are double
// but this may not work for very large integers.
using numeric_literal_type = double;

template <typename T = numeric_literal_type>
inline auto parse(const std::string& expression) {
  numeric_literal_type value;
  std::string::const_iterator iter = expression.begin();
  std::string::const_iterator end = expression.end();
  suzerain::exprgrammar::parse(iter, end, value);
  return static_cast<T>(value);
}

inline void evaluate_and_apply_changed_config(const std::string& name,
                                              const std::string& value,
                                              srtb::configs& config) {
// TODO: this seems ugly. better approach?
#define SRTB_PARSE(target_name)                                     \
  if (name == #target_name) {                                       \
    const decltype(config.target_name) parsed_value = parse(value); \
    SRTB_LOGI << " [program_options] " << #target_name << " = "     \
              << parsed_value << srtb::endl;                        \
    config.target_name = parsed_value;                              \
  } else
#define SRTB_ASSIGN(target_name)                                         \
  if (name == #target_name) {                                            \
    SRTB_LOGI << " [program_options] " << #target_name << " = " << value \
              << srtb::endl;                                             \
    config.target_name = value;                                          \
  } else

  ;  // <- for clang-format
  SRTB_PARSE(baseband_input_count)
  SRTB_PARSE(baseband_input_bits)
  SRTB_PARSE(baseband_freq_low)
  SRTB_PARSE(baseband_bandwidth)
  SRTB_PARSE(baseband_sample_rate)
  SRTB_PARSE(dm)
  SRTB_PARSE(udp_receiver_buffer_size)
  SRTB_ASSIGN(udp_receiver_sender_address)
  SRTB_PARSE(udp_receiver_sender_port)
  SRTB_PARSE(udp_receiver_cpu_preferred)
  SRTB_ASSIGN(input_file_path)
  SRTB_PARSE(input_file_offset_bytes)
  SRTB_ASSIGN(baseband_output_file_prefix)
  SRTB_PARSE(baseband_write_all)
  SRTB_PARSE(log_level)
  SRTB_ASSIGN(fft_fftw_wisdom_path)
  SRTB_PARSE(mitigate_rfi_threshold)
  SRTB_ASSIGN(mitigate_rfi_freq_list)
  SRTB_PARSE(refft_length)
  SRTB_PARSE(signal_detect_threshold)
  SRTB_PARSE(thread_query_work_wait_time)
  /* else */ {
    SRTB_LOGW << " [program_options] "
              << "Unrecognized config: name = " << '\"' << name << '\"' << ", "
              << "value = " << '\"' << value << '\"' << srtb::endl;
  }
#undef SRTB_PARSE
#undef SRTB_ASSIGN
}

inline void apply_changed_configs(
    std::map<std::string, std::string>& changed_configs,
    srtb::configs& config) {
  for (auto item : changed_configs) {
    evaluate_and_apply_changed_config(item.first, item.second, config);
  }
}

}  // namespace program_options
}  // namespace srtb

#endif  // __SRTB_PROGRAM_OPTIONS__
