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
#include <stdexcept>
#include <string>
#include <utility>

#include "3rdparty/multi_file_reader.hpp"
#include "assert.hpp"
#include "form_beam.hpp"
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

template <typename Pointer, typename SizeType = size_t>
struct mem {
  Pointer ptr;
  SizeType count;
};

}  // namespace detail

auto main(int argc, char **argv) -> int {
  const auto cfg = srtb::_21cma::make_beam::program_options::parse(argc, argv);
  auto &file_list = cfg.baseband_file_list;

  // sizes used
  const size_t n_ifstream = file_list.size();
  const size_t n_station = station_per_udp_stream * n_ifstream;
  const size_t n_channel = static_cast<size_t>(cfg.n_channel);
  const size_t n_sample = static_cast<size_t>(cfg.n_sample);
  // const size_t n_buffer_complex = n_channel * n_sample * station_per_udp_stream;
  // const size_t n_buffer_real = n_buffer_complex * 2;

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

  std::vector<std::shared_ptr<MultiFileReader>> reader;
  reader.resize(n_ifstream);
  for (size_t i_ifstream = 0; i_ifstream < n_ifstream; i_ifstream++) {
    reader.at(i_ifstream) = std::make_shared<MultiFileReader>(file_list.at(i_ifstream), dada_dbdisk_file_header_size);
    //  reader.at(i) = MultiFileReader{file_list.at(i), dada_dbdisk_file_header_size};
  }

  // allocate memory regions used
  std::vector<mem<std::shared_ptr<std::byte>, size_t>> h_in{n_ifstream};
  {
    const size_t n_buffer_complex = n_channel * n_sample * station_per_udp_stream;
    const size_t n_buffer_real = n_buffer_complex * 2;
    for (size_t i = 0; i < h_in.size(); i++) {
      h_in.at(i) = mem{host_allocator.allocate_shared<std::byte>(n_buffer_real), n_buffer_real};
    }
  }
  std::vector<mem<std::shared_ptr<std::byte>, size_t>> d_in{h_in.size()};
  for (size_t i = 0; i < d_in.size(); i++) {
    const auto n_buffer_real = h_in.at(i).count;
    d_in.at(i) = mem{device_allocator.allocate_shared(n_buffer_real), n_buffer_real};
  }

  const size_t operate_buffer_count = n_station * n_channel * n_sample * 2;
  mem<std::shared_ptr<srtb::real>, size_t> d_buffer_real = {
      device_allocator.allocate_shared<srtb::real>(operate_buffer_count), operate_buffer_count};
  mem<std::shared_ptr<srtb::complex<srtb::real>>, size_t> d_buffer_complex = {
      std::reinterpret_pointer_cast<srtb::complex<srtb::real>>(d_buffer_real.ptr), d_buffer_real.count / 2};

  std::vector<std::future<void>> task_future{n_ifstream};
  // std::vector<sycl::event> event{d_unpacked.size()};

  srtb::fft::fft_1d_dispatcher<srtb::fft::type::C2C_1D_FORWARD> fft_dispatcher{n_channel, n_sample * n_station, q};

  std::atomic_uint64_t n_eof_reader = 0;

  // main loop
  while (n_eof_reader == 0) {
    // read, copy and unpack
    Kokkos::mdspan<srtb::real, Kokkos::dextents<size_t, 2>> d_unpack{d_buffer_real.ptr.get(), n_station,
                                                                     n_channel * n_sample * 2};
    BOOST_ASSERT(d_unpack.size() == d_buffer_real.count);
    for (size_t i_ifstream = 0; i_ifstream < n_ifstream; i_ifstream++) {
      task_future.at(i_ifstream) = std::async(
          std::launch::async,
          [reader = reader[i_ifstream], h_in = h_in.at(i_ifstream), d_in = d_in.at(i_ifstream),
           d_unpack_1 = Kokkos::submdspan(d_unpack, station_per_udp_stream * i_ifstream, Kokkos::full_extent),
           d_unpack_2 = Kokkos::submdspan(d_unpack, station_per_udp_stream * i_ifstream + 1, Kokkos::full_extent),
           _q = &q, &n_eof_reader] {
            const auto read_byte = reader->read(reinterpret_cast<char *>(h_in.ptr.get()), h_in.count);
            if (read_byte == 0) {
              n_eof_reader++;
            }

            thread_local auto q = sycl::queue{*_q};
            BOOST_ASSERT(h_in.count == d_in.count);
            q.copy(h_in.ptr.get(), d_in.ptr.get(), h_in.count).wait();

            BOOST_ASSERT(d_unpack_1.size() == d_unpack_2.size());
            srtb::unpack::unpack_naocpsr_snap1(d_in.ptr.get(), d_unpack_1.data_handle(), d_unpack_2.data_handle(),
                                               d_unpack_1.size(), srtb::algorithm::map_identity{}, q);
          });
    }
    for (auto &task : task_future) {
      task.wait();
    }

    // FFT
    Kokkos::mdspan<srtb::complex<srtb::real>, Kokkos::dextents<size_t, 3>> d_fft{d_buffer_complex.ptr.get(), n_station,
                                                                                 n_sample, n_channel};
    BOOST_ASSERT(d_fft.size() == d_buffer_complex.count);
    fft_dispatcher.process(d_fft.data_handle(), d_fft.data_handle());
    srtb::fft::fft_1d_r2c_in_place_post_process(d_fft.data_handle(), n_channel, n_station * n_sample, q);
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
