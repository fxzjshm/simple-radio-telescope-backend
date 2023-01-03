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
#include "srtb/memory/host_ring_buffer.hpp"
#include "test-common.hpp"

#define SRTB_CHECK_TEST_HOST_RING_BUFFER(expr) \
  SRTB_CHECK_TEST("[test-host_ring_buffer] ", expr)

int main() {
  constexpr size_t buffer_length = 1 << 22;
  using data_type = uint64_t;

  srtb::memory::host_ring_buffer<data_type> ring_buffer;

  const size_t mem_length = buffer_length;
  auto mem_shared = srtb::host_allocator.allocate_shared<data_type>(mem_length);
  auto mem = mem_shared.get();
  std::generate(mem, mem + mem_length,
                []() { return static_cast<data_type>(std::rand()); });
  const size_t tmp_length = buffer_length / 2;
  auto tmp_shared = srtb::host_allocator.allocate_shared<data_type>(tmp_length);
  auto tmp = tmp_shared.get();

  ring_buffer.push(mem, mem_length);
  ring_buffer.pop(tmp, tmp_length);
  SRTB_CHECK_TEST_HOST_RING_BUFFER(
      check_absolute_error(tmp, tmp + tmp_length, mem, data_type{0}));

  ring_buffer.push(tmp, tmp_length);

  ring_buffer.pop(nullptr, mem_length - tmp_length);
  ring_buffer.pop(tmp, tmp_length);
  SRTB_CHECK_TEST_HOST_RING_BUFFER(
      check_absolute_error(tmp, tmp + tmp_length, mem, data_type{0}));

  return 0;
}
