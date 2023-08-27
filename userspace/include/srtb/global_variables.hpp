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

#include <atomic>
#include <condition_variable>
#include <map>
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

// memory

inline srtb::memory::cached_allocator<sycl::usm_allocator<
    std::byte, sycl::usm::alloc::host, srtb::MEMORY_ALIGNMENT> >
    host_allocator{queue};

#ifdef SRTB_USE_USM_SHARED_MEMORY
inline srtb::memory::cached_allocator<sycl::usm_allocator<
    std::byte, sycl::usm::alloc::shared, srtb::MEMORY_ALIGNMENT> >
    device_allocator{queue};
#else
inline srtb::memory::cached_allocator<
    srtb::memory::device_allocator<std::byte, srtb::MEMORY_ALIGNMENT> >
    device_allocator{queue};
#endif

// work queues
// TODO: global work queues seem ugly (no extensibility), better ideas ?
inline srtb::work_queue<srtb::work::copy_to_device_work, /* spsc = */ false>
    copy_to_device_queue;
inline srtb::work_queue<srtb::work::unpack_work, /* spsc = */ false>
    unpack_queue;
inline srtb::work_queue<srtb::work::fft_1d_r2c_work> fft_1d_r2c_queue;
inline srtb::work_queue<srtb::work::rfi_mitigation_s1_work>
    rfi_mitigation_s1_queue;
inline srtb::work_queue<srtb::work::dedisperse_work> dedisperse_queue;
inline srtb::work_queue<srtb::work::ifft_1d_c2c_work> ifft_1d_c2c_queue;
inline srtb::work_queue<srtb::work::refft_1d_c2c_work> refft_1d_c2c_queue;
inline srtb::work_queue<srtb::work::rfi_mitigation_s2_work>
    rfi_mitigation_s2_queue;
inline srtb::work_queue<srtb::work::simplify_spectrum_work>
    simplify_spectrum_queue;
inline srtb::work_queue<srtb::work::draw_spectrum_work> draw_spectrum_queue;
inline srtb::work_queue<srtb::work::draw_spectrum_work_2> draw_spectrum_queue_2;
inline srtb::work_queue<srtb::work::signal_detect_work> signal_detect_queue;
inline srtb::work_queue<srtb::work::baseband_output_work> baseband_output_queue;

namespace pipeline {

inline std::atomic<size_t> running_pipe_count = 0;
/**
 * @brief expected running pipe count, set in main(), 
 *        used by input pipes (signal sources) to determine when to start
 * @note see expected_input_pipe_count
 */
inline size_t expected_running_pipe_count = -233;
/** 
 * @brief expected input pipe (udp_receiver, file_reader, etc.) count
 * 
 * @note initial value must be larger than expected_running_pipe_count because
 *       there is data race between started pipe's thread & main() 
 *       (main() set these values after pipe has started but number here is
 *        count of started pipes, so...... )
 *       set initial expected_input_pipe_count larger so initially
 *           expected_running_pipe_count - expected_input_pipe_count
 *       is a large arbitrary number.
 */
inline size_t expected_input_pipe_count = -42;

/**
 * @brief used for end of pipeline to send a signal to start of the pipeline,
 * @note currently enabled only when input source is a file
 */
inline std::atomic<bool> need_more_work = false;

/**
 * @brief used for start of pipeline to send a signal to main thread
 * @note currently enabled only when input source is a file
 */
inline std::atomic<bool> no_more_work = false;

}  // namespace pipeline

// termination_handler_v in termination_handler.hpp because termination_handler needs it.

// fftw initializer in srtb/fft/fftw_wrapper.hpp due to forward declearation

// color map holder in srtb/gui/spectrum_image_provider.hpp due to unnecessary Qt dependencies for tests

}  // namespace srtb

#endif  // __SRTB_GLOBAL_VARIABLES__
