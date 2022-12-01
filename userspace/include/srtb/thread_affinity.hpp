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
#ifndef __SRTB_THREAD_AFFINITY__
#define __SRTB_THREAD_AFFINITY__

#include <hwloc.h>

#include "srtb/commons.hpp"

namespace srtb {
namespace thread_affinity {

/**
 * @brief Set thread affinity of current thread to @c target_cpu
 * 
 * adapted from https://github.com/open-mpi/hwloc/blob/master/doc/examples/cpuset%2Bbitmap%2Bcpubind.c
 * ref: https://hwloc.readthedocs.io/en/v2.4/group__hwlocality__bitmap.html
 */
int set_thread_affinity(unsigned int target_cpu) {
  hwloc_topology_t topology = nullptr;
  hwloc_bitmap_t set2 = nullptr;
  hwloc_obj_t obj;
  int err = 0;

  do {
    /* create a topology */
    err = hwloc_topology_init(&topology);
    if (err < 0) {
      SRTB_LOGW << "[thread_affinity] "
                << "failed to initialize the topology" << srtb::endl;
      break;
    }
    err = hwloc_topology_load(topology);
    if (err < 0) {
      SRTB_LOGW << "[thread_affinity] "
                << "failed to load the topology" << srtb::endl;
      break;
    }

    /* retrieve the single PU where the current thread actually runs within this process binding */
    set2 = hwloc_bitmap_alloc();
    if (!set2) {
      SRTB_LOGW << "[thread_affinity] "
                << "failed to allocate a bitmap" << srtb::endl;
      break;
    }
    err = hwloc_get_last_cpu_location(topology, set2, HWLOC_CPUBIND_THREAD);
    if (err < 0) {
      SRTB_LOGW << "[thread_affinity] "
                << "failed to get last cpu location" << srtb::endl;
      break;
    }

    unsigned int i;
    /* print the logical number of the PU where that thread runs */
    /* extract the PU OS index from the bitmap */
    i = hwloc_bitmap_first(set2);
    obj = hwloc_get_pu_obj_by_os_index(topology, i);
    SRTB_LOGI << "[thread_affinity] "
              << "thread is now running on PU logical index "
              << obj->logical_index << " (OS/physical index " << i << ")"
              << srtb::endl;

    /* migrate this single thread to where other PUs within the current binding */
    hwloc_bitmap_only(set2, target_cpu);
    hwloc_bitmap_singlify(set2);
    err = hwloc_set_cpubind(topology, set2, HWLOC_CPUBIND_THREAD);
    if (err < 0) {
      SRTB_LOGW << "[thread_affinity] "
                << "failed to set thread binding" << srtb::endl;
      break;
    }
    /* reprint the PU where that thread runs */
    err = hwloc_get_last_cpu_location(topology, set2, HWLOC_CPUBIND_THREAD);
    if (err < 0) {
      SRTB_LOGW << "[thread_affinity] "
                << "failed to get last cpu location" << srtb::endl;
      break;
    }
    /* print the logical number of the PU where that thread runs */
    /* extract the PU OS index from the bitmap */
    i = hwloc_bitmap_first(set2);
    obj = hwloc_get_pu_obj_by_os_index(topology, i);
    SRTB_LOGI << "[thread_affinity] "
              << "thread is running on PU logical index " << obj->logical_index
              << " (OS/physical index " << i << ")" << srtb::endl;
  } while (0);

  if (set2 != nullptr) {
    hwloc_bitmap_free(set2);
  }
  if (topology != nullptr) {
    hwloc_topology_destroy(topology);
  }

  if (err < 0) {
    return EXIT_FAILURE;
  } else {
    return EXIT_SUCCESS;
  }
}

}  // namespace thread_affinity
}  // namespace srtb

#endif  // __SRTB_THREAD_AFFINITY__
