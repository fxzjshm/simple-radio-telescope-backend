/******************************************************************************* 
 * Copyright (c) 2025 fxzjshm
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
#ifndef SRTB_SYCL_DEVICE_COPYABLE
#define SRTB_SYCL_DEVICE_COPYABLE

#include <boost/iterator/counting_iterator.hpp>
#include <boost/iterator/permutation_iterator.hpp>

#include "srtb/sycl.hpp"

#if defined(SYCL_DEVICE_COPYABLE) && SYCL_DEVICE_COPYABLE
// patch for foreign iterators
template <typename T>
struct sycl::is_device_copyable<boost::iterators::counting_iterator<T>> : std::true_type {};

template <class ElementIterator, class IndexIterator>
struct sycl::is_device_copyable<boost::iterators::permutation_iterator<ElementIterator, IndexIterator>>
    : std::true_type {};
#endif

#endif  // SRTB_SYCL_DEVICE_COPYABLE
