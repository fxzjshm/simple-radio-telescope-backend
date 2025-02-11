/******************************************************************************* 
 * Copyright (c) 2024 fxzjshm
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
#ifndef SRTB_TYPE_NAME
#define SRTB_TYPE_NAME

#if defined(__clang__) || defined(__GNUC__) || defined(_MSC_VER)
#define SRTB_HAS_CONSTEXPR_UNMANGLED_TYPE_NAME
#include "unmangled_type_name.hpp"
#else
#include <boost/core/demangle.hpp>
#endif

namespace srtb {

#ifdef SRTB_HAS_CONSTEXPR_UNMANGLED_TYPE_NAME
template <typename T>
static inline constexpr auto type_name() {
  return unmangled_type_name::type_name<T>();
}
#else
template <typename T>
static inline auto type_name() {
  return boost::core::demangle(typeid(T).name());
}
#endif  // SRTB_HAS_CONSTEXPR_UNMANGLED_TYPE_NAME

}  // namespace srtb

#endif  // SRTB_TYPE_NAME
