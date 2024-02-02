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
#ifndef __SRTB_IO_VDIF_HEADER__
#define __SRTB_IO_VDIF_HEADER__

#include <cstddef>
#include <cstdint>

namespace srtb {
namespace io {

/**
 * ref: https://vlbi.org/vlbi-standards/vdif/
 * 
 * use std::bit_cast to convert to this.
 */
struct vdif_header {
  using vdif_word = uint32_t;
  static inline constexpr size_t vdif_word_size = sizeof(vdif_word);
  static inline constexpr size_t vdif_word_count = 8;

  uint32_t seconds_from_ref_epoch : 30;
  uint32_t legacy_mode : 1;
  uint32_t invalid_data : 1;

  uint32_t data_frame_count_in_second : 24;
  uint32_t reference_epoch : 6;
  uint32_t unassigned : 2;

  uint32_t data_frame_length : 24;
  uint32_t log2_channels : 5;
  uint32_t vdif_version : 3;

  uint32_t station_id : 16;
  uint32_t thread_id : 10;
  uint32_t bits_per_sample_minus_1 : 5;
  uint32_t data_type : 1;

  uint32_t extended_user_data_1 : 24;
  uint32_t extended_data_version : 8;

  uint32_t extended_user_data_2;

  uint32_t extended_user_data_3;

  uint32_t extended_user_data_4;
};

static_assert(sizeof(vdif_header) ==
              vdif_header::vdif_word_size * vdif_header::vdif_word_count);

}  // namespace io
}  // namespace srtb

#endif  //  __SRTB_IO_VDIF_HEADER__
