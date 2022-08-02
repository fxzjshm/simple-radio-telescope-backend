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
#include "srtb/sycl.hpp"

namespace srtb {

inline srtb::configs config;

inline sycl::queue queue;

}  // namespace srtb

#include "srtb/memory/cached_allocator.hpp"
#include "srtb/memory/sycl_device_allocator.hpp"
#include "srtb/work.hpp"

namespace srtb {

inline srtb::memory::cached_allocator<sycl::usm_allocator<
    std::byte, sycl::usm::alloc::host, srtb::MEMORY_ALIGNMENT> >
    host_allocator{queue};

inline srtb::memory::cached_allocator<
    srtb::memory::device_allocator<std::byte, srtb::MEMORY_ALIGNMENT> >
    device_allocator{queue};

inline srtb::work_queue<srtb::work::unpack_work> unpack_queue{
    srtb::work_queue_initial_capacity};
inline srtb::work_queue<srtb::work::fft_1d_r2c_work> fft_1d_r2c_queue{
    srtb::work_queue_initial_capacity};
inline srtb::work_queue<srtb::work::simplify_spectrum_work>
    simplify_spectrum_queue{srtb::work_queue_initial_capacity};
inline srtb::work_queue<srtb::work::draw_spectrum_work> draw_spectrum_queue{
    srtb::work_queue_initial_capacity};

}  // namespace srtb

// FFT dispatch in srtb/fft/fft.hpp due to forward declaration

#endif  // __SRTB_GLOBAL_VARIABLES__
