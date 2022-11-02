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

/*
 * This file contains most of global variables used in this program.
 * Some global variables need extra dependency, so not written here.
 */

#include <condition_variable>
#include <mutex>
#include <thread>

#include "srtb/config.hpp"
#include "srtb/sycl.hpp"

namespace srtb {


// configs

// TODO: add `volatile` here?
//   * config shouldn't be changed frequently, force loading it from memory
//     would cause too much cache miss;
//   * however, are there optimizations related to this that are unexpected?
inline srtb::configs config;

/** @brief names and expressions of changed items of @c srtb::config */
inline std::map<std::string, std::string> changed_configs;


/** @brief default queue for all operations if no queue is specified */
inline sycl::queue queue;

/** @brief record start time of program, used in log to indicate relative time */
inline auto program_start_time = std::chrono::system_clock::now();

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

inline srtb::work_queue<srtb::work::unpack_work> unpack_queue;
inline srtb::work_queue<srtb::work::fft_1d_r2c_work> fft_1d_r2c_queue;
inline srtb::work_queue<srtb::work::rfi_mitigation_work> rfi_mitigation_queue;
inline srtb::work_queue<srtb::work::dedisperse_and_channelize_work>
    dedisperse_and_channelize_queue;
inline srtb::work_queue<srtb::work::ifft_1d_c2c_work> ifft_1d_c2c_queue;
inline srtb::work_queue<srtb::work::refft_1d_c2c_work> refft_1d_c2c_queue;
inline srtb::work_queue<srtb::work::simplify_spectrum_work>
    simplify_spectrum_queue;
inline srtb::work_queue<srtb::work::draw_spectrum_work> draw_spectrum_queue;
inline srtb::work_queue<srtb::work::baseband_output_work> baseband_output_queue;

namespace pipeline {

// currently used for end of pipeline to send a signal to start of the pipeline.
inline std::mutex pipeline_mutex;
inline std::condition_variable pipeline_cv;
inline bool one_work_just_finished = false;

}  // namespace pipeline

// termination_handler_v in termination_handler.hpp because termination_handler needs it.

// fftw initializer in srtb/fft/fftw_wrapper.hpp due to forward declearation

// color map holder in srtb/gui/spectrum_image_provider.hpp due to unnecessary Qt dependencies for tests

}  // namespace srtb

#endif  // __SRTB_GLOBAL_VARIABLES__
