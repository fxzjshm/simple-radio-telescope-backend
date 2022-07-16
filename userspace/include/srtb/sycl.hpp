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
#ifndef __SRTB_SYCL__
#define __SRTB_SYCL__

/**
 * @brief SYCL headers and platform-specific things
 */

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

// TODO: platform specific things
// clang-format off
#if defined(SYCL_IMPLEMENTATION_ONEAPI)
    // maybe related: Compile-time definitions to detect used SYCL targets
    //                https://github.com/intel/llvm/issues/5562
    #if defined(SYCL_EXT_ONEAPI_BACKEND_CUDA) && defined(SRTB_ENABLE_CUDA) && SRTB_ENABLE_CUDA
        #define SRTB_ENABLE_CUDA_INTEROP 1
    #elif defined(SYCL_EXT_ONEAPI_BACKEND_HIP) && defined(SRTB_ENABLE_ROCM) && SRTB_ENABLE_ROCM
        #define SRTB_ENABLE_ROCM_INTEROP 1
    #else
        #warning "Detected OneAPI but no known backend. (TODO)"
    #endif  // defined(SYCL_EXT_ONEAPI_BACKEND_CUDA) or defined(SYCL_EXT_ONEAPI_BACKEND_HIP)
#elif defined(__HIPSYCL__)
    #if defined(__HIPSYCL_ENABLE_CUDA_TARGET__)
        #define SRTB_ENABLE_CUDA_INTEROP 1
    #endif
    #if defined(__HIPSYCL_ENABLE_HIP_TARGET__)
        #define SRTB_ENABLE_ROCM_INTEROP 1
    #endif
    #if defined(__HIPSYCL_ENABLE_OMPHOST_TARGET__)
        // no need to define introp macro
    #endif
#else
    #warning "Unknown SYCL backend"
#endif
// clang-format on

#ifdef SRTB_ENABLE_CUDA_INTEROP
#define SRTB_IF_ENABLED_CUDA_INTEROP(...) __VA_ARGS__
#else
#define SRTB_IF_ENABLED_CUDA_INTEROP(...)
#endif  // SRTB_ENABLE_CUDA_INTEROP

#ifdef SRTB_ENABLE_ROCM_INTEROP
#define SRTB_IF_ENABLED_ROCM_INTEROP(...) __VA_ARGS__
#else
#define SRTB_IF_ENABLED_ROCM_INTEROP(...)
#endif  // SRTB_ENABLE_ROCM_INTEROP

#endif  // __SRTB_SYCL__
