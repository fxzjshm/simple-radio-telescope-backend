/******************************************************************************* 
 * Copyright (c) 2023 fxzjshm
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
#ifndef __SRTB_WORK_21CMA_GENFIL_WORK__
#define __SRTB_WORK_21CMA_GENFIL_WORK__

#include <cstddef>

#include "srtb/work.hpp"

namespace srtb {

namespace work {

using fft_r2c_post_process_work = fft_1d_c2c_work;
using dynspec_work = srtb::work::work<std::shared_ptr<srtb::real> >;
using write_multi_filterbank_work =
    srtb::work::work<std::shared_ptr<std::byte> >;

}  // namespace work

}  // namespace srtb

#endif  // __SRTB_WORK_21CMA_GENFIL_WORK__
