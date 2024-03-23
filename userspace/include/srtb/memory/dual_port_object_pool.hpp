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
#ifndef __SRTB_MEMORY_DUAL_PORT_OBJECT_POOL__
#define __SRTB_MEMORY_DUAL_PORT_OBJECT_POOL__

#include <boost/lockfree/spsc_queue.hpp>
#include <deque>

namespace srtb {
namespace memory {

template <typename T>
class dual_port_object_pool {
 protected:
  boost::lockfree::spsc_queue<T*> received_packet_queue;
  boost::lockfree::spsc_queue<T*> free_packet_queue;
  std::deque<std::unique_ptr<T> > packet_owner;

 public:
  explicit dual_port_object_pool(size_t queue_size)
      : received_packet_queue{queue_size}, free_packet_queue{queue_size} {
    for (size_t i = 0; i < queue_size / 2; i++) {
      free_packet_queue.push(allocate_free());
    }
  }

  auto allocate_free() -> T* {
    auto h_packet_unique = std::make_unique<T>();
    T* h_packet = h_packet_unique.get();
    packet_owner.push_back(std::move(h_packet_unique));
    return h_packet;
  }

  auto get_or_allocate_free() -> T* {
    T* h_packet = nullptr;
    const bool pop_success = free_packet_queue.pop(h_packet);
    if (!pop_success) {
      h_packet = allocate_free();
    }
    return h_packet;
  }

  void put_free(T* h_packet) {
    bool push_success = false;
    while (!push_success) {
      push_success = free_packet_queue.push(h_packet);
    }
  }

  void push_received(T* h_packet) {
    bool push_success = false;
    while (!push_success) {
      push_success = received_packet_queue.push(h_packet);
    }
  }

  auto pop_received() -> T* {
    T* ret = nullptr;
    bool pop_success = false;
    while (!pop_success) {
      pop_success = received_packet_queue.pop(ret);
    }
    return ret;
  }
};

}  // namespace memory
}  // namespace srtb

#endif  // __SRTB_MEMORY_DUAL_PORT_OBJECT_POOL__
