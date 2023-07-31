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
    #endif  // defined(SYCL_EXT_ONEAPI_BACKEND_CUDA) or defined(SYCL_EXT_ONEAPI_BACKEND_HIP)
#elif defined(__HIPSYCL__)
    #if defined(__HIPSYCL_ENABLE_CUDA_TARGET__)
        #define SRTB_ENABLE_CUDA_INTEROP 1
    #endif
    #if defined(__HIPSYCL_ENABLE_HIP_TARGET__)
        #define SRTB_ENABLE_ROCM_INTEROP 1
    #endif
    #if defined(SRTB_ENABLE_MUSA)
        #define SRTB_ENABLE_MUSA_INTEROP 1
    #endif
    #if defined(__HIPSYCL_ENABLE_OMPHOST_TARGET__)
        // no need to define introp macro
    #endif
#else
    #warning "Unknown SYCL backend, interop disabled"
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

#ifdef SRTB_ENABLE_MUSA_INTEROP
#define SRTB_IF_ENABLED_MUSA_INTEROP(...) __VA_ARGS__
#else
#define SRTB_IF_ENABLED_MUSA_INTEROP(...)
#endif  // SRTB_ENABLE_MUSA_INTEROP

// fix for sycl::get_native<sycl::backend::ext_oneapi_hip>(device)
// https://github.com/intel/llvm/pull/7145
#if defined(SYCL_IMPLEMENTATION_ONEAPI) && defined(SRTB_ENABLE_ROCM_INTEROP)
#include <sycl/ext/oneapi/backend/hip.hpp>
#endif

namespace srtb {
namespace backend {

// clang-format off
#if defined(SYCL_IMPLEMENTATION_ONEAPI)
    #if defined(SRTB_ENABLE_CUDA_INTEROP)
        inline constexpr sycl::backend cuda = sycl::backend::ext_oneapi_cuda;
    #endif
    #if defined(SRTB_ENABLE_ROCM_INTEROP)
        inline constexpr sycl::backend rocm = sycl::backend::ext_oneapi_hip;
    #endif
#elif defined(__HIPSYCL__)
    #if defined(SRTB_ENABLE_CUDA_INTEROP)
        inline constexpr sycl::backend cuda = sycl::backend::cuda;
    #endif
    #if defined(SRTB_ENABLE_ROCM_INTEROP)
        inline constexpr sycl::backend rocm = sycl::backend::hip;
    #endif
    #if defined(SRTB_ENABLE_MUSA_INTEROP)
        inline constexpr sycl::backend musa = sycl::backend::musa;
    #endif
    inline constexpr sycl::backend cpu = sycl::backend::omp;
#else
    #warning "Unknown SYCL backend"
#endif
// clang-format on

}  // namespace backend
}  // namespace srtb

// if conflict between BOOST_NOINLINE and HIP is encountered, upgrade boost to 1.80+
// ref: https://github.com/boostorg/config/issues/392

#endif  // __SRTB_SYCL__
