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

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
// -- divide line for clang-format --
#include <boost/iostreams/device/file_descriptor.hpp>
#include <boost/iostreams/stream.hpp>

#define SRTB_USE_USM_SHARED_MEMORY

#include "srtb/commons.hpp"
// -- divide line for clang-format --
#include "srtb/fft/fft.hpp"
#include "srtb/unpack.hpp"

/**
 * Basic "baseband" correlator
 * ref: https://casper.astro.berkeley.edu/astrobaki/index.php/Correlators
 *      in a word, \hat{f \star g}(\omega) = \hat{f}(\omega) \hat{g}^{*}(\omega)
 */
int main() {
  // parameters
  std::string in_file_1 = "pol_1.bin";
  std::string in_file_2 = "pol_2.bin";
  std::string out_file_path = "/dev/shm/corr.bin";
  boost::iostreams::stream<boost::iostreams::file_descriptor_sink>
      output_stream{out_file_path, BOOST_IOS::binary | BOOST_IOS::out};
  sycl::queue q;

  // input handles & size
  SRTB_LOGI << " [correlator] "
            << "reading " << std::filesystem::absolute(in_file_1) << srtb::endl;
  SRTB_LOGI << " [correlator] "
            << "reading " << std::filesystem::absolute(in_file_2) << srtb::endl;
  std::ifstream input_file_stream_1{in_file_1,
                                    std::ifstream::in | std::ifstream::binary};
  std::ifstream input_file_stream_2{in_file_2,
                                    std::ifstream::in | std::ifstream::binary};
  const size_t size_1 = std::filesystem::file_size(in_file_1);
  const size_t size_2 = std::filesystem::file_size(in_file_2);
  const size_t input_size = std::min(size_1, size_2);
  const size_t complex_count = input_size / 2;
  const size_t real_count = complex_count * 2;
  // numbers should be normalized, otherwise will exceed max value of float
  const srtb::real norm_coeff = std::pow(input_size, -1.5);

  // read file
  auto h_in_1_shared =
      srtb::host_allocator.allocate_shared<std::byte>(real_count);
  auto h_in_2_shared =
      srtb::host_allocator.allocate_shared<std::byte>(real_count);
  std::byte* h_in_1 = h_in_1_shared.get();
  std::byte* h_in_2 = h_in_2_shared.get();
  std::memset(h_in_1, 0, real_count);
  std::memset(h_in_2, 0, real_count);
  input_file_stream_1.read(reinterpret_cast<char*>(h_in_1), input_size);
  input_file_stream_2.read(reinterpret_cast<char*>(h_in_2), input_size);

  // copy to device
  auto d_in_1_shared =
      srtb::device_allocator.allocate_shared<std::byte>(real_count);
  auto d_in_2_shared =
      srtb::device_allocator.allocate_shared<std::byte>(real_count);
  std::byte* d_in_1 = d_in_1_shared.get();
  std::byte* d_in_2 = d_in_2_shared.get();
  sycl::event evt_1 = q.copy(h_in_1, d_in_1, real_count);
  sycl::event evt_2 = q.copy(h_in_2, d_in_2, real_count);
  evt_2.wait();
  evt_1.wait();

  // unpack
  // size +2 for r2c fft, dropped later
  auto d_unpacked_1_shared =
      srtb::device_allocator.allocate_shared<srtb::real>(real_count + 2);
  auto d_unpacked_2_shared =
      srtb::device_allocator.allocate_shared<srtb::real>(real_count + 2);
  srtb::real* d_unpacked_1 = d_unpacked_1_shared.get();
  srtb::real* d_unpacked_2 = d_unpacked_2_shared.get();
  srtb::unpack::unpack<8>(d_in_1, d_unpacked_1, real_count, q);
  srtb::unpack::unpack<8>(d_in_2, d_unpacked_2, real_count, q);

  // fft r2c inplace
  srtb::complex<srtb::real>* d_ffted_1 =
      reinterpret_cast<srtb::complex<srtb::real>*>(d_unpacked_1);
  srtb::complex<srtb::real>* d_ffted_2 =
      reinterpret_cast<srtb::complex<srtb::real>*>(d_unpacked_2);
  {
    srtb::fft::fft_1d_dispatcher<srtb::fft::type::R2C_1D, srtb::real,
                                 srtb::complex<srtb::real> >
        r2c_dispatcher{real_count, 1, q};
    r2c_dispatcher.process(d_unpacked_1, d_ffted_1);
    r2c_dispatcher.process(d_unpacked_2, d_ffted_2);
  }

  // correlation
  //auto d_corr_ffted_shared =
  //    srtb::device_allocator.allocate_shared<srtb::complex<srtb::real> >(
  //        complex_count);
  //srtb::complex<srtb::real>* d_corr_ffted = d_corr_ffted_shared.get();
  srtb::complex<srtb::real>* d_corr_ffted = d_ffted_1;
  // (1)
  q.parallel_for(sycl::range{complex_count}, [=](sycl::item<1> id) {
     const auto i = id.get_id(0);
     d_corr_ffted[i] = (norm_coeff * d_ffted_1[i]) * srtb::conj(d_ffted_2[i]);
   }).wait();
  d_in_1 = nullptr;
  d_in_2 = nullptr;
  d_in_1_shared.reset();
  d_in_2_shared.reset();
  // (2)
  srtb::complex<srtb::real>* d_corr = d_corr_ffted;
  {
    srtb::fft::fft_1d_dispatcher<srtb::fft::type::C2C_1D_BACKWARD, srtb::real,
                                 srtb::complex<srtb::real> >
        c2c_dispatcher{complex_count, 1, q};
    // one complex number is silently dropped
    c2c_dispatcher.process(d_corr_ffted, d_corr);
  }
  // (3)
  auto d_out_shared =
      srtb::device_allocator.allocate_shared<srtb::real>(complex_count);
  srtb::real* d_out = d_out_shared.get();
  q.parallel_for(sycl::range{complex_count}, [=](sycl::item<1> id) {
     const auto i = id.get_id(0);
     d_out[i] = srtb::abs(d_corr[i]);
   }).wait();

  // copy back
  std::vector<srtb::real> h_out;
  h_out.resize(complex_count);
  q.copy(d_out, &h_out[0], complex_count).wait();

  // write
  output_stream.write(reinterpret_cast<char*>(&h_out[0]),
                      complex_count * sizeof(srtb::real));

  return EXIT_SUCCESS;
}
