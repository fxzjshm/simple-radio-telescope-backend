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
#ifndef __SRTB_GLOBAL_VARIABLES__
#define __SRTB_GLOBAL_VARIABLES__

#include "srtb/config.hpp"
#include "srtb/memory/cached_allocator.hpp"
#include "srtb/memory/sycl_device_allocator.hpp"
#include "srtb/sycl.hpp"
#include "srtb/work.hpp"

namespace srtb {

inline srtb::configs config;

inline sycl::queue queue;

inline srtb::memory::cached_allocator<
    std::byte, sycl::usm_allocator<std::byte, sycl::usm::alloc::host,
                                   srtb::MEMORY_ALIGNMENT> >
    host_allocator{queue};

inline srtb::memory::cached_allocator<
    std::byte,
    srtb::memory::device_allocator<std::byte, srtb::MEMORY_ALIGNMENT> >
    device_allocator{queue};

inline srtb::work_queue<srtb::work<void*> > unpacker_queue{
    srtb::work_queue_initial_capacity};

}  // namespace srtb

#endif  // __SRTB_GLOBAL_VARIABLES__