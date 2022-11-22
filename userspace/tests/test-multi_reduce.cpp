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

#include "srtb/commons.hpp"
// -- divide line for clang-format --
#include "srtb/algorithm/multi_reduce.hpp"
#include "test-common.hpp"

#define SRTB_CHECK_TEST_MULTI_REDUCE(expr) \
  SRTB_CHECK_TEST("[test-multi_reduce] ", expr)

int main(int argc, char** argv) {
  size_t count_per_batch = 4309;
  size_t batch_size = 11037;

  if (argc > 2) {
    try {
      count_per_batch = std::stoi(argv[1]);
      batch_size = std::stoi(argv[2]);
    } catch (const std::invalid_argument& ignored) {
      // should remain unchanged
    }
  }

  const size_t total_count = count_per_batch * batch_size;

  using T = float;
  auto map = [=](size_t, T x) { return sycl::sin(x); };
  auto reduce = sycl::plus<T>();

  std::vector<T> h_in;
  h_in.resize(total_count);
  auto d_in_shared = srtb::device_allocator.allocate_shared<T>(total_count);
  auto d_out_shared = srtb::device_allocator.allocate_shared<T>(batch_size);
  auto d_in = d_in_shared.get();
  auto d_out = d_out_shared.get();
  sycl::queue q = srtb::queue;

  std::generate(h_in.begin(), h_in.end(),
                []() { return static_cast<T>(std::rand() % 256); });
  q.copy(&h_in[0], /* -> */ d_in, total_count).wait();
  srtb::algorithm::multi_mapreduce(d_in, count_per_batch, batch_size, d_out,
                                   map, reduce, q);
  std::vector<T> h_out, h_ref;
  h_out.resize(batch_size);
  h_ref.resize(batch_size);
  q.copy(d_out, /* -> */ &h_out[0], batch_size).wait();

  {
    auto in = &h_in[0];
    auto out = &h_ref[0];
    for (size_t i = 0; i < batch_size; i++) {
      const size_t offset = i * count_per_batch;
      T acc = map(offset + 0, in[offset + 0]);
      for (size_t j = 1; j < count_per_batch; j++) {
        acc = reduce(acc, map(offset + j, in[offset + j]));
      }
      out[i] = acc;
    }
  }

  const auto err = std::min(
      std::numeric_limits<T>::epsilon() * count_per_batch * 2, T{0.05});
  check_relative_error(h_out.begin(), h_out.end(), h_ref.begin(), err);

  return 0;
}
