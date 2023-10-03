/**
 * SYCL implementation of https://github.com/NAOC-pulsar/datacompression/blob/admin3/runningmean/runningmean.c
 */

#pragma once
#ifndef __SRTB_ALGORITHM_RUNNING_MEAN_HPP__
#define __SRTB_ALGORITHM_RUNNING_MEAN_HPP__

#include <type_traits>

#include "srtb/sycl.hpp"

namespace srtb {
namespace algorithm {

/**
 * @brief Return 1 bit average spectrum of input data
 * @note channel reverse not done here; init average values using running_mean_init_average
 * 
 *     nchan
 *   <--------->
 *   xxx ... xxx | ^
 *   xxx ... xxx | | windowsize
 *   ...     ... | v
 *   xxx ... xxx | nsamp
 *   xxx ... xxx v
 * 
 */
template <typename InputIterator, typename OutputIterator,
          typename AverageValueIterator>
inline void running_mean(InputIterator data, size_t nsamp, size_t nchan,
                         OutputIterator outdata, size_t windowsize,
                         AverageValueIterator ave, sycl::queue& q) {
  using input_t = typename std::iterator_traits<InputIterator>::value_type;
  using output_t = typename std::iterator_traits<InputIterator>::value_type;
  using avgval_t =
      typename std::iterator_traits<AverageValueIterator>::value_type;

  q.parallel_for(sycl::range<1>{nchan}, [=](sycl::item<1> id) {
     const auto j = id.get_id(0);
     avgval_t ave_j = ave[j];
     for (size_t i = windowsize; i < nsamp; i++) {
       const input_t head = data[(i - windowsize) * nchan + j];
       const input_t tail = data[i * nchan + j];
       outdata[(i - windowsize) * nchan + j] =
           static_cast<output_t>(head > ave_j);
       ave_j += ((avgval_t)tail - (avgval_t)head) / windowsize;
     }
     for (size_t i = 0; i < windowsize; i++) {
       const input_t head = data[(nsamp + i - windowsize) * nchan + j];
       const input_t tail = data[(nsamp - i - 1) * nchan + j];
       outdata[(i + nsamp - windowsize) * nchan + j] =
           static_cast<output_t>(head > ave_j);
       ave_j += ((avgval_t)tail - (avgval_t)head) / windowsize;
     }
     ave[j] = ave_j;
   }).wait();
}

template <typename InputIterator, typename OutputIterator,
          typename AverageValueIterator>
inline void running_mean_init_average(
    InputIterator data, [[maybe_unused]] size_t nsamp, size_t nchan,
    [[maybe_unused]] OutputIterator outdata, size_t windowsize,
    AverageValueIterator ave, sycl::queue& q) {
  using avgval_t =
      typename std::iterator_traits<AverageValueIterator>::value_type;

  q.parallel_for(sycl::range<1>{nchan}, [=](sycl::item<1> id) {
     const auto j = id.get_id(0);
     avgval_t sum = 0;
     for (size_t k = 0; k < windowsize; k++) {
       sum += data[k * nchan + j];
     }
     ave[j] = sum / windowsize;
   }).wait();
}

}  // namespace algorithm
}  // namespace srtb

#endif  // __SRTB_ALGORITHM_RUNNING_MEAN_HPP__
