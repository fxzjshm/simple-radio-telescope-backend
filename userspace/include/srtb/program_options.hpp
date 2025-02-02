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

#include <boost/algorithm/string.hpp>
#include <boost/program_options.hpp>
#include <filesystem>
#include <string>

#include "exprgrammar.h"
#include "srtb/config.hpp"
#include "srtb/log/log.hpp"

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
      cfg_file_options("Options available in config file"),
      all_option("Options");
  /* clang-format off */
    /*
      template:
      ("config_name", boost::program_options::value<std::string>(),
       "decsription")
    */
    cmd_only_options.add_options()
      ("help,h", "Show help message")
      ("config_file_name", boost::program_options::value<std::string>(),
       "Path to config file to be used to read other configs. ")
    ;
    general_option.add_options()
      ("log_level", boost::program_options::value<std::string>(),
       "Debug level for console log output. ")
      ("thread_query_work_wait_time", boost::program_options::value<std::string>(),
       "Wait time in naneseconds for a thread to sleep if it fails to get work. "
       "Trade off between CPU usage (most are wasted) and pipeline latency. ")
#if SRTB_ENABLE_GUI
       ("gui_enable", boost::program_options::value<std::string>(),
        "Runtime configuration to enable GUI")
       ("gui_pixmap_width", boost::program_options::value<std::string>(),
        "Width of GUI spectrum pixmap")
       ("gui_pixmap_height", boost::program_options::value<std::string>(),
        "Height of GUI spectrum pixmap")
#endif
    ;
    baseband_option.add_options()
      ("baseband_input_count", boost::program_options::value<std::string>(),
       "Count of data to be transferred to GPU for once processing, in sample counts. "
       "Should be power of 2 so that FFT and channelizing can work properly. ")
      ("baseband_input_bits", boost::program_options::value<std::string>(),
       "Length of a single input data, used in unpack. "
       "Negative value is signed integers. "
       "Currently supported: 1(uint1), 2(uint2), 4(uint4), 8(uint8), -8(int8), 32(float), 64(double)")
      ("baseband_format_type", boost::program_options::value<std::string>(),
       "Type of baseband format: "
       "simple, naocpsr_roach2, naocpsr_snap1, gznupsr_a1")
      ("baseband_freq_low", boost::program_options::value<std::string>(),
       "Lowerest frequency of received baseband signal, in MHz. ")
      ("baseband_bandwidth", boost::program_options::value<std::string>(),
       "Band width of received baseband signal, in MHz. ")
      ("baseband_sample_rate", boost::program_options::value<std::string>(),
       "Baseband sample rate, in samples / second. "
       "Should be 2 * baseband_bandwidth (* 1e6 because of unit) if Nyquist rate. ")
      ("baseband_reserve_sample", boost::program_options::value<std::string>(),
       "if 1, baseband data affected by dispersion will be reserved for next segment, "
       "i.e. segments will overlap, if possible; "
       "if 0, baseband data will not overlap.")
    ;
    udp_receiver_options.add_options()
      ("udp_receiver_sender_address", boost::program_options::value<std::string>(),
       "Address(es) to receive baseband UDP packets")
      ("udp_receiver_sender_port", boost::program_options::value<std::string>(),
       "Port(s) to receive baseband UDP packets")
      ("udp_receiver_cpu_preferred", boost::program_options::value<std::string>(),
       "CPU core that UDP receiver should be bound to. ")
    ;
    file_io_options.add_options()
      ("input_file_path", boost::program_options::value<std::string>(),
       "Path to the binary file to be read as baseband input. ")
      ("input_file_offset_bytes", boost::program_options::value<std::string>(),
       "Skip some data before reading in, usually avoids header")
      ("baseband_output_file_prefix", boost::program_options::value<std::string>(),
       "Prefix of saved baseband data. Full name will be ${prefix}${counter}.bin")
      ("baseband_write_all", boost::program_options::value<std::string>(),
       "if 1, record all baseband into one file per polarization; "
       "if 0, write only those with signal detected. ")
    ;
    operation_option.add_options()
      ("dm,dedisperse_measurement", boost::program_options::value<std::string>(),
       "Target dispersion measurement for coherent dedispersion. ")
      ("fft_fftw_wisdom_path", boost::program_options::value<std::string>(),
       "Location to save fftw wisdom. ")
      ("mitigate_rfi_average_method_threshold", boost::program_options::value<std::string>(),
       "Temporary threshold for RFI mitigation. Frequency channels with signal "
       "stronger than (this threshold * average strength) will be set to 0")
      ("mitigate_rfi_spectral_kurtosis_threshold", boost::program_options::value<std::string>(),
       "Frequency channels with spectral kurtosis larger than this threshold will be set to 0")
      ("mitigate_rfi_freq_list", boost::program_options::value<std::string>(),
       "list of frequency pairs to zap/remove, "
       "format: 11-12, 15-90, 233-235, 1176-1177 (arbitrary values)")
      ("spectrum_channel_count", boost::program_options::value<std::string>(),
       "Count of channels (complex numbers) in spectrum waterfall. "
       "Time resolution for one bin is 2 * spectrum_channel_count / baseband_sample_rate")
      ("signal_detect_signal_noise_threshold", boost::program_options::value<std::string>(),
       "threshold for signal detect, target signal / noise ratio")
      ("signal_detect_channel_threshold", boost::program_options::value<std::string>(),
       "threshold of ratio of non-zapped channels. "
       "if too many channels are zapped, result is often not correct")
      ("signal_detect_max_boxcar_length", boost::program_options::value<std::string>(),
       "max boxcar length for signal detect")
    ;
  /* clang-format on */
  data_io_option.add(udp_receiver_options).add(file_io_options);
  cfg_file_options.add(general_option)
      .add(baseband_option)
      .add(data_io_option)
      .add(operation_option);
  all_option.add(cmd_only_options).add(cfg_file_options);

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
    SRTB_LOGI << " [program_options] "
              << "using config file " << config_file_name << " (absolute path "
              << std::filesystem::absolute(config_file_name) << ")"
              << srtb::endl;
    boost::program_options::notify(vm);
    boost::program_options::store(
        boost::program_options::parse_config_file(config_file_name.c_str(),
                                                  cfg_file_options),
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
              << "Command line options:" << srtb::endl
              << all_option << srtb::endl;
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
inline auto parse(const std::string& expression) -> double {
  const char* iter = expression.c_str();
  const char* end = iter + expression.size();
  return exprgrammar_parse_double(iter, end);
}

inline void evaluate_and_apply_changed_config(const std::string& name,
                                              const std::string& value,
                                              srtb::configs& config) {
// TODO: this seems ugly. better approach?
#define SRTB_PARSE(target_name)                                              \
  if (name == #target_name) {                                                \
    using target_type = decltype(config.target_name);                        \
    const target_type parsed_value = static_cast<target_type>(parse(value)); \
    SRTB_LOGI << " [program_options] " << #target_name << " = "              \
              << parsed_value << srtb::endl;                                 \
    config.target_name = parsed_value;                                       \
  } else

#define SRTB_ASSIGN(target_name)                                         \
  if (name == #target_name) {                                            \
    SRTB_LOGI << " [program_options] " << #target_name << " = " << value \
              << srtb::endl;                                             \
    config.target_name = value;                                          \
  } else

#define SRTB_SPLIT_PARSE(target_name, delimiter)                           \
  if (name == #target_name) {                                              \
    using target_type = typename decltype(config.target_name)::value_type; \
    std::vector<std::string> sub_strings;                                  \
    boost::split(sub_strings, value, boost::is_any_of(delimiter),          \
                 boost::token_compress_on);                                \
    std::vector<target_type> sub_values;                                   \
    for (auto sub_string : sub_strings) {                                  \
      const target_type sub_value =                                        \
          static_cast<target_type>(parse(sub_string));                     \
      sub_values.push_back(sub_value);                                     \
    }                                                                      \
    SRTB_LOGI << " [program_options] " << #target_name << " = "            \
              << srtb::log::container_to_string(sub_values, delimiter)     \
              << srtb::endl;                                               \
    config.target_name = sub_values;                                       \
  } else

#define SRTB_SPLIT_ASSIGN(target_name, delimiter)                       \
  if (name == #target_name) {                                           \
    std::vector<std::string> sub_strings;                               \
    boost::split(sub_strings, value, boost::is_any_of(delimiter),       \
                 boost::token_compress_on);                             \
    SRTB_LOGI << " [program_options] " << #target_name << " = "         \
              << srtb::log::container_to_string(sub_strings, delimiter) \
              << srtb::endl;                                            \
    config.target_name = sub_strings;                                   \
  } else

  ;  // <- for clang-format
  SRTB_PARSE(baseband_input_count)
  SRTB_PARSE(baseband_input_bits)
  SRTB_ASSIGN(baseband_format_type)
  SRTB_PARSE(baseband_freq_low)
  SRTB_PARSE(baseband_bandwidth)
  SRTB_PARSE(baseband_sample_rate)
  SRTB_PARSE(baseband_reserve_sample)
  SRTB_PARSE(dm)
  SRTB_SPLIT_ASSIGN(udp_receiver_sender_address, /* delimiter = */ ",")
  SRTB_SPLIT_PARSE(udp_receiver_sender_port, /* delimiter = */ ",")
  SRTB_SPLIT_PARSE(udp_receiver_cpu_preferred, /* delimiter = */ ",")
  SRTB_ASSIGN(input_file_path)
  SRTB_PARSE(input_file_offset_bytes)
  SRTB_ASSIGN(baseband_output_file_prefix)
  SRTB_PARSE(baseband_write_all)
  SRTB_ASSIGN(fft_fftw_wisdom_path)
  SRTB_PARSE(mitigate_rfi_average_method_threshold)
  SRTB_PARSE(mitigate_rfi_spectral_kurtosis_threshold)
  SRTB_ASSIGN(mitigate_rfi_freq_list)
  SRTB_PARSE(spectrum_channel_count)
  SRTB_PARSE(signal_detect_signal_noise_threshold)
  SRTB_PARSE(signal_detect_channel_threshold)
  SRTB_PARSE(signal_detect_max_boxcar_length)
  SRTB_PARSE(thread_query_work_wait_time)
  SRTB_PARSE(gui_enable)
  SRTB_PARSE(gui_pixmap_width)
  SRTB_PARSE(gui_pixmap_height)
  /* else */ if (name == "config_file_name") {
    // has been processed earlier
  } else if (name == "log_level") {
    using target_type = decltype(srtb::log::log_level);
    const target_type parsed_value = static_cast<target_type>(parse(value));
    SRTB_LOGI << " [program_options] "
              << "log_level"
              << " = " << static_cast<int>(parsed_value) << srtb::endl;
    srtb::log::log_level = parsed_value;
  } else {
    SRTB_LOGW << " [program_options] "
              << "Unrecognized config: name = " << '\"' << name << '\"' << ", "
              << "value = " << '\"' << value << '\"'
              << ", check option list at " __FILE__ ": " << __LINE__
              << srtb::endl;
  }

#undef SRTB_PARSE
#undef SRTB_ASSIGN
#undef SRTB_SPLIT_PARSE
#undef SRTB_SPLIT_ASSIGN
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
