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

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <map>
#include <memory>
#include <string>
#include <utility>

#include "assert.hpp"
#include "form_beam.hpp"
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

std::string meta_file_list = "meta_file_list.txt";

size_t expected_baseband_buffer_size = 819200000;
constexpr size_t dada_dbdisk_file_header_size = 4096;
constexpr size_t station_per_in_file_stream = 2;
size_t expected_baseband_file_size = expected_baseband_buffer_size + dada_dbdisk_file_header_size;

size_t n_channel = 1 << 15;  // complex numbers

auto main(int argc, char **argv) -> int {
  auto cfg = srtb::_21cma::make_beam::program_options::parse(argc, argv);
  auto file_list = cfg.baseband_file_list;

  // sizes used
  const size_t n_in_file_stream = file_list.size();
  const size_t n_station = station_per_in_file_stream * n_in_file_stream;
  const size_t n_buffer_real = expected_baseband_buffer_size / station_per_in_file_stream;
  const size_t n_buffer_complex = n_buffer_real / 2;
  const size_t n_sample = n_buffer_complex / n_channel;

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

  // allocate memory regions used
  std::vector<mem<std::shared_ptr<std::byte>, size_t>> h_in{n_in_file_stream};
  for (size_t i = 0; i < h_in.size(); i++) {
    h_in.at(i) = mem{host_allocator.allocate_shared(expected_baseband_buffer_size), expected_baseband_buffer_size};
  }

  std::vector<mem<std::shared_ptr<std::byte>, size_t>> d_in{h_in.size()};
  for (size_t i = 0; i < d_in.size(); i++) {
    const auto size_in_block = h_in.at(i).count;
    d_in.at(i) = mem{device_allocator.allocate_shared(size_in_block), size_in_block};
  }

  const size_t operate_buffer_count = n_in_file_stream * expected_baseband_buffer_size;
  mem<std::shared_ptr<srtb::real>, size_t> d_buffer_real = {
      device_allocator.allocate_shared<srtb::real>(operate_buffer_count), operate_buffer_count};
  mem<std::shared_ptr<srtb::complex<srtb::real>>, size_t> d_buffer_complex = {
      std::reinterpret_pointer_cast<srtb::complex<srtb::real>>(d_buffer_real.ptr), d_buffer_real.count / 2};

  std::vector<std::future<void>> task_future{n_in_file_stream};
  // std::vector<sycl::event> event{d_unpacked.size()};

  srtb::fft::fft_1d_dispatcher<srtb::fft::type::C2C_1D_FORWARD> fft_dispatcher{n_channel, n_sample * n_station, q};

  for (size_t i_file = 0; i_file < file_list[0].size(); i_file++) {
    // read, copy and unpack
    Kokkos::mdspan<srtb::real, Kokkos::dextents<size_t, 2>> d_unpack{d_buffer_real.ptr.get(), n_station, n_buffer_real};
    BOOST_ASSERT(d_unpack.size() == d_buffer_real.count);
    for (size_t i_in_file_stream = 0; i_in_file_stream < n_in_file_stream; i_in_file_stream++) {
      const auto file_path = std::filesystem::path{file_list[i_in_file_stream][i_file]};
      task_future.at(i_in_file_stream) = std::async(
          std::launch::async,
          [file_path, h_in = h_in.at(i_in_file_stream), d_in = d_in.at(i_in_file_stream),
           d_unpack_1 = Kokkos::submdspan(d_unpack, station_per_in_file_stream * i_in_file_stream, Kokkos::full_extent),
           d_unpack_2 =
               Kokkos::submdspan(d_unpack, station_per_in_file_stream * i_in_file_stream + 1, Kokkos::full_extent),
           _q = q] {
            BOOST_ASSERT_MSG(std::filesystem::exists(file_path), (file_path.string() + "not found").c_str());
            BOOST_ASSERT_MSG(std::filesystem::file_size(file_path) == expected_baseband_file_size,
                             ("File size not expected, expected " + std::to_string(expected_baseband_file_size) +
                              ", actural " + std::to_string(std::filesystem::file_size(file_path)))
                                 .c_str());

            std::ifstream in_handle{file_path, std::ios::in | std::ios::binary};
            in_handle.ignore(dada_dbdisk_file_header_size);
            in_handle.read(reinterpret_cast<char *>(h_in.ptr.get()), h_in.count);

            BOOST_ASSERT(h_in.count == d_in.count);
            auto q = _q;
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

  return EXIT_SUCCESS;
}

}  // namespace srtb::_21cma::make_beam

int main(int argc, char **argv) { return srtb::_21cma::make_beam::main(argc, argv); }
