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

#include <boost/iterator/counting_iterator.hpp>
#include <fstream>

#include "srtb/algorithm/map_reduce.hpp"
#include "srtb/coherent_dedispersion.hpp"
#include "test-common.hpp"

#define SRTB_CHECK_TEST_DF64(expr)                                      \
  SRTB_CHECK(expr, true, {                                              \
    throw std::runtime_error{"[test-df64] " #expr " at " __FILE__ ":" + \
                             std::to_string(__LINE__) + " returns " +   \
                             std::to_string(ret)};                      \
  })

int main() {
  bool write_out = false;

  const size_t N = (size_t{1} << 20);
  const srtb::real f_min = 1000, f_max = 1500, df = (f_max - f_min) / N;
  const srtb::real dm = 56.778;
  const srtb::real eps = 1e-5;

  auto h_double_out_shared =
      srtb::host_allocator.allocate_unique<srtb::complex<srtb::real> >(N);
  auto h_double_out = h_double_out_shared.get();
  auto d_double_out_shared =
      srtb::device_allocator.allocate_unique<srtb::complex<srtb::real> >(N);
  auto d_double_out = d_double_out_shared.get();

  auto h_dsmath_out_shared =
      srtb::host_allocator.allocate_unique<srtb::complex<srtb::real> >(N);
  auto h_dsmath_out = h_dsmath_out_shared.get();
  auto d_dsmath_out_shared =
      srtb::device_allocator.allocate_unique<srtb::complex<srtb::real> >(N);
  auto d_dsmath_out = d_dsmath_out_shared.get();

  sycl::queue q;
  q.parallel_for(sycl::range<1>(N), [=](sycl::item<1> id) {
     auto i = id.get_id(0);
     srtb::real f = f_min + df * i;
     d_double_out[i] =
         srtb::codd::coherent_dedispersion_factor<double, srtb::real,
                                                  srtb::complex<srtb::real> >(
             f, f_max, dm);
     d_dsmath_out[i] =
         srtb::codd::coherent_dedispersion_factor<dsmath::df64, srtb::real,
                                                  srtb::complex<srtb::real> >(
             f, f_max, dm);
   }).wait();

  if (write_out) {
    std::string double_file_name = "dedisp_double.bin";
    std::string dsmath_file_name = "dedisp_dsmath.bin";

    FILE *double_file = fopen(double_file_name.c_str(), "wb");
    FILE *dsmath_file = fopen(dsmath_file_name.c_str(), "wb");

    q.copy(d_double_out, /* -> */ h_double_out, N).wait();
    q.copy(d_dsmath_out, /* -> */ h_dsmath_out, N).wait();
    SRTB_LOGI << " [test-coherent_dedispersion] "
              << "writing to " << double_file_name << srtb::endl;
    fwrite(h_double_out, sizeof(srtb::complex<srtb::real>), N, double_file);
    fclose(double_file);
    SRTB_LOGI << " [test-coherent_dedispersion] "
              << "writing to " << dsmath_file_name << srtb::endl;
    fwrite(h_dsmath_out, sizeof(srtb::complex<srtb::real>), N, dsmath_file);
    fclose(dsmath_file);
  }

  std::shared_ptr<size_t> d_error_count_shared = srtb::algorithm::map_sum(
      boost::iterators::make_counting_iterator(size_t{0}), N,
      [=]([[maybe_unused]] size_t pos, size_t i) {
        return ((srtb::norm(d_double_out[i] - d_dsmath_out[i]) > eps)
                    ? size_t{1}
                    : size_t{0});
      },
      q);
  size_t h_error_count;
  q.copy(d_error_count_shared.get(), /* -> */ &h_error_count, 1).wait();
  if (h_error_count) {
    throw std::runtime_error{"Detected double and dsmath::df64 mismatch: " +
                             std::to_string(h_error_count)};
  }

  return 0;
}
