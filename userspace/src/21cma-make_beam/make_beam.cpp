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

#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <map>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "3rdparty/multi_file_reader.hpp"
#include "assert.hpp"
#include "common.hpp"
#include "form_beam.hpp"
#include "get_delay.hpp"
#include "global_variables.hpp"
#include "mdspan/mdspan.hpp"
#include "program_options.hpp"
#include "srtb/algorithm/map_identity.hpp"
#include "srtb/fft/fft.hpp"
#include "srtb/fft/fft_1d_r2c_post_process.hpp"
#include "srtb/log/log.hpp"
#include "srtb/math.hpp"
#include "srtb/memory/cached_allocator.hpp"
#include "srtb/memory/sycl_device_allocator.hpp"
#include "srtb/termination_handler.hpp"
#include "srtb/unpack.hpp"

namespace srtb::_21cma::make_beam {

inline namespace detail {

/** @brief similar to std::span. but use smart pointer & owning this memory */
template <typename Pointer, typename SizeType = size_t>
struct mem {
  Pointer ptr;
  SizeType count;
  using Type = Pointer::element_type;

  operator std::span<Type, std::dynamic_extent>() { return std::span{ptr.get(), count}; }

  template <typename... T>
  auto get_mdspan(T... sizes) {
    SizeType n = (1 * ... * sizes);
    BOOST_ASSERT(n == count);
    return Kokkos::mdspan{ptr.get(), sizes...};
  }
};

}  // namespace detail

auto main(int argc, char **argv) -> int {
  const auto cfg = srtb::_21cma::make_beam::program_options::parse(argc, argv);
  auto file_list = cfg.baseband_file_list;
  auto station_whitelist = cfg.station_whitelist;

  // sizes used
  const size_t n_ifstream = file_list.size();
  const size_t n_station = station_per_udp_stream * n_ifstream;
  const size_t n_channel = static_cast<size_t>(cfg.n_channel);
  const size_t n_sample = static_cast<size_t>(cfg.n_sample);
  const size_t n_buffer_complex = n_channel * n_sample * station_per_udp_stream;
  const size_t n_buffer_real = n_buffer_complex * 2;
  const size_t n_pointing = cfg.pointing.size();

  const double start_mjd = cfg.start_mjd;
  const auto observation_mode = cfg.observation_mode;

  // setup SYCL environment
  sycl::queue q;
  SRTB_LOGI << " [21cma-make_beam] " << "device name = " << q.get_device().get_info<sycl::info::device::name>()
            << srtb::endl;

  srtb::memory::cached_allocator<sycl::usm_allocator<std::byte, sycl::usm::alloc::host, srtb::MEMORY_ALIGNMENT>>
      host_allocator{q};

#ifdef SRTB_USE_USM_SHARED_MEMORY
  srtb::memory::cached_allocator<sycl::usm_allocator<std::byte, sycl::usm::alloc::shared, srtb::MEMORY_ALIGNMENT>>
      device_allocator{q};
#else
  srtb::memory::cached_allocator<srtb::memory::device_allocator<std::byte, srtb::MEMORY_ALIGNMENT>> device_allocator{q};
#endif

  // setup file reader
  std::vector<std::shared_ptr<MultiFileReader>> reader;
  reader.resize(n_ifstream);
  for (size_t i_ifstream = 0; i_ifstream < n_ifstream; i_ifstream++) {
    reader.at(i_ifstream) = std::make_shared<MultiFileReader>(file_list.at(i_ifstream), dada_dbdisk_file_header_size);
    //  reader.at(i) = MultiFileReader{file_list.at(i), dada_dbdisk_file_header_size};
  }

  // allocate memory regions used
  std::vector<mem<std::shared_ptr<int8_t>, size_t>> h_in{n_ifstream};
  {
    for (size_t i = 0; i < h_in.size(); i++) {
      h_in.at(i) = mem{host_allocator.allocate_shared<int8_t>(n_buffer_real), n_buffer_real};
    }
  }
  std::vector<mem<std::shared_ptr<int8_t>, size_t>> d_in{h_in.size()};
  for (size_t i = 0; i < d_in.size(); i++) {
    d_in.at(i) = mem{device_allocator.allocate_shared<int8_t>(n_buffer_real), n_buffer_real};
  }

  const size_t operate_buffer_count = n_station * n_channel * n_sample * 2;
  mem<std::shared_ptr<srtb::real>, size_t> d_buffer_real = {
      device_allocator.allocate_shared<srtb::real>(operate_buffer_count), operate_buffer_count};
  mem<std::shared_ptr<srtb::complex<srtb::real>>, size_t> d_buffer_complex = {
      std::reinterpret_pointer_cast<srtb::complex<srtb::real>>(d_buffer_real.ptr), d_buffer_real.count / 2};

  mem<std::shared_ptr<double>, size_t> h_delay = {host_allocator.allocate_shared<double>(n_station), n_station};
  mem<std::shared_ptr<double>, size_t> d_delay = {device_allocator.allocate_shared<double>(n_station), n_station};

  mem<std::shared_ptr<srtb::complex<srtb::real>>, size_t> d_weight = {
      device_allocator.allocate_shared<srtb::complex<srtb::real>>(n_station * n_channel), n_station * n_channel};
  mem<std::shared_ptr<srtb::complex<srtb::real>>, size_t> d_out = {
      device_allocator.allocate_shared<srtb::complex<srtb::real>>(n_sample * n_channel), n_sample * n_channel};
  mem<std::shared_ptr<srtb::complex<srtb::real>>, size_t> h_out = {
      host_allocator.allocate_shared<srtb::complex<srtb::real>>(n_sample * n_channel), n_sample * n_channel};

  std::vector<std::ofstream> fout{n_pointing};
  BOOST_ASSERT(n_pointing == cfg.out_path.size());
  for (size_t i_pointing = 0; i_pointing < n_pointing; i_pointing++) {
    fout.at(i_pointing) = std::ofstream{cfg.out_path.at(i_pointing)};
  }

  // station info
  std::vector<relative_location_t> station_location(n_station);
  std::vector<double> station_cable_delay(n_station);
  for (size_t i_station = 0; i_station < n_station; i_station++) {
    const auto &station = station_map.at(i_station);
    station_location.at(i_station) = antenna_location[station];
    station_cable_delay.at(i_station) = cable_delay_table[station];
  }

  std::vector<std::future<void>> read_future{n_ifstream};
  std::vector<std::future<void>> copy_unpack_future{n_ifstream};

  srtb::fft::fft_1d_dispatcher<srtb::fft::type::C2C_1D_FORWARD> fft_dispatcher{n_channel, n_sample * n_station, q};

  std::atomic_size_t read_byte = 0;
  std::atomic_uint64_t n_eof_reader = 0;

  auto request_read = [&](size_t i_ifstream) {
    read_future.at(i_ifstream) = std::async(
        std::launch::async,
        [i_ifstream, reader = reader[i_ifstream], h_in = h_in.at(i_ifstream), &n_eof_reader, &read_byte]() {
          const auto read_byte_this_round = reader->read(reinterpret_cast<char *>(h_in.ptr.get()), h_in.count);
          if (read_byte_this_round == 0) [[unlikely]] {
            n_eof_reader++;
          }
          if (i_ifstream == 0) {
            read_byte += read_byte_this_round;
          }
        });
  };

  for (size_t i_ifstream = 0; i_ifstream < n_ifstream; i_ifstream++) {
    request_read(i_ifstream);
  }

  // main loop
  while (n_eof_reader == 0) {
    // copy and unpack
    auto d_unpack = d_buffer_real.get_mdspan(n_station, n_channel * n_sample * 2);
    BOOST_ASSERT(d_unpack.size() == d_buffer_real.count);
    for (size_t i_ifstream = 0; i_ifstream < n_ifstream; i_ifstream++) {
      copy_unpack_future.at(i_ifstream) = std::async(
          std::launch::async,
          [read_ftr = &read_future.at(i_ifstream), h_in = h_in.at(i_ifstream), d_in = d_in.at(i_ifstream),
           d_unpack_1 = Kokkos::submdspan(d_unpack, station_per_udp_stream * i_ifstream, Kokkos::full_extent),
           d_unpack_2 = Kokkos::submdspan(d_unpack, station_per_udp_stream * i_ifstream + 1, Kokkos::full_extent),
           _q = &q] {
            thread_local auto q = sycl::queue{*_q};
            BOOST_ASSERT(h_in.count == d_in.count);
            // wait for read
            read_ftr->wait();
            q.copy(h_in.ptr.get(), d_in.ptr.get(), h_in.count).wait();

            BOOST_ASSERT(d_unpack_1.size() == d_unpack_2.size());
            BOOST_ASSERT(d_unpack_1.size() * 2 == d_in.count);
            srtb::unpack::unpack_naocpsr_snap1(d_in.ptr.get(), d_unpack_1.data_handle(), d_unpack_2.data_handle(),
                                               d_unpack_1.size(), srtb::algorithm::map_identity{}, q);
          });
    }
    for (size_t i_ifstream = 0; i_ifstream < n_ifstream; i_ifstream++) {
      copy_unpack_future.at(i_ifstream).wait();
      request_read(i_ifstream);
    }

    // FFT
    auto d_fft = d_buffer_complex.get_mdspan(n_station, n_sample, n_channel);
    BOOST_ASSERT(d_fft.size() == d_buffer_complex.count);
    fft_dispatcher.process(d_fft.data_handle(), d_fft.data_handle());
    srtb::fft::fft_1d_r2c_in_place_post_process(d_fft.data_handle(), n_channel, n_station * n_sample, q);

    // Beamform
    const auto read_byte_ = read_byte.load();
    double obstime_mjd_;
    switch (observation_mode) {
      case observation_mode_t::TRACKING:
        obstime_mjd_ =
            start_mjd + static_cast<double>(read_byte_) / station_per_udp_stream / sample_rate / second_in_day;
        break;
      case observation_mode_t::DRIFTING:
        obstime_mjd_ = start_mjd;
        break;
    }
    const double obstime_mjd = obstime_mjd_;
    SRTB_LOGI << " [21cma-make_beam] " << "read_byte = " << read_byte_ << ", " << "obstime_mjd = " << std::to_string(obstime_mjd)
              << srtb::endl;

    auto d_form_beam_in = d_fft;
    for (size_t i_pointing = 0; i_pointing < n_pointing; i_pointing++) {
      const auto pointing = cfg.pointing.at(i_pointing);
      get_delay(pointing, obstime_mjd, station_location, station_cable_delay, h_delay);
      BOOST_ASSERT(h_delay.count == d_delay.count);
      q.copy(h_delay.ptr.get(), d_delay.ptr.get(), h_delay.count).wait();

      auto d_weight_ = d_weight.get_mdspan(n_station, n_channel);
      get_weight(d_delay, freq_min, freq_max, d_weight_, q);
      for (size_t i_station = 0; i_station < n_station; i_station++) {
        if (std::find(station_whitelist.begin(), station_whitelist.end(), station_map[i_station]) ==
            station_whitelist.end()) {
          // station not found in whitelist
          auto d_weight_i = Kokkos::submdspan(d_weight_, i_station, Kokkos::full_extent);
          BOOST_ASSERT(d_weight_i.size() == n_channel);
          q.fill(d_weight_i.data_handle(), srtb::complex<srtb::real>{0, 0}, d_weight_i.size()).wait();
        }
      }

      auto d_out_ = d_out.get_mdspan(n_sample, n_channel);
      srtb::_21cma::make_beam::form_beam(d_form_beam_in, d_weight_, d_out_, q);
      BOOST_ASSERT(d_out.count == h_out.count);
      q.copy(d_out.ptr.get(), h_out.ptr.get(), d_out.count).wait();
      fout.at(i_pointing).write(reinterpret_cast<char *>(h_out.ptr.get()), h_out.count);
    }
  }

  if (n_eof_reader != n_ifstream) {
    SRTB_LOGE << " [21cma-make_beam] " << "n_eof_reader = " << std::to_string(n_eof_reader.load()) << ", "
              << "n_ifstream = " << n_ifstream << srtb::endl;
    return EXIT_FAILURE;
  } else {
    return EXIT_SUCCESS;
  }
}

}  // namespace srtb::_21cma::make_beam

int main(int argc, char **argv) { return srtb::_21cma::make_beam::main(argc, argv); }
