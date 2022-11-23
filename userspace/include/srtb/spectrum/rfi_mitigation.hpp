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
#ifndef __SRTB_SPECTRIM_RFI_MITIGAGION__
#define __SRTB_SPECTRIM_RFI_MITIGAGION__

#include "srtb/commons.hpp"
// --- divide line for clang-format ---
#include "srtb/algorithm/map_reduce.hpp"

namespace srtb {
namespace spectrum {

/**
 * @brief Mitigate RFI by setting narrow band interference to 0 directly
 * TODO: use "spectural kurtosis"
 * TODO: compute norm twice or once with temporary buffer for it ?
 */
template <typename T = srtb::real, typename C = srtb::complex<srtb::real>,
          typename DeviceComplexInputAccessor = C*>
void mitigate_rfi(DeviceComplexInputAccessor d_in, size_t in_count,
                  sycl::queue& q = srtb::queue) {
  const srtb::real thereshold = srtb::config.mitigate_rfi_thereshold;
  auto d_norm_avg_shared = srtb::algorithm::map_average(
      d_in, in_count,
      []([[maybe_unused]] size_t pos, C c) { return srtb::norm(c); }, q);
  auto d_norm_avg = d_norm_avg_shared.get();
  q.parallel_for(sycl::range<1>{in_count}, [=](sycl::item<1> id) {
     const size_t i = id.get_id(0);
     const srtb::real norm_avg = (*d_norm_avg);
     const srtb::real val = srtb::norm(d_in[i]);
     if (val > thereshold * norm_avg) {
       d_in[i] = C(T(0), T(0));
     }
   }).wait();
}

}  // namespace spectrum
}  // namespace srtb

#endif  // __SRTB_SPECTRIM_RFI_MITIGAGION__
