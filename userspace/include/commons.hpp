/******************************************************************************* 
 * Copyright (c) 2022 fxzjshm
 * This software is licensed under Mulan PubL v2.
 * You can use this software according to the terms and conditions of the Mulan
 * PubL v2. You may obtain a copy of Mulan PubL v2 at:
 *          http://license.coscl.org.cn/MulanPubL-2.0
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
 * KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
 * NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE. See the
 * Mulan PubL v2 for more details.
 ******************************************************************************/

#pragma once
#ifndef __SRTB_COMMONS__
#define __SRTB_COMMONS__

/**
 * This file should contain commonly included headers, and forward
 * declaration if needed.
 */

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

#include "config.hpp"

#include <boost/lockfree/spsc_queue.hpp>

// TODO: platform specific things
//#if defined()
//#define __SRTB_CUDA__
//#endif

namespace srtb {

// TODO: maybe float on GPU?
typedef double real;

// TODO: check should use queue or spsc_queue here
template <typename... Args>
using queue = boost::lockfree::spsc_queue<Args>;

}  // namespace srtb

#include "global_variables.hpp"
#include "logger.hpp"

#endif  // __SRTB_COMMONS__