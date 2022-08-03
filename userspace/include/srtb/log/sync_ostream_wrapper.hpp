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
#ifndef __SRTB_LOG_SYNC_OSTREAM_WRAPPER__
#define __SRTB_LOG_SYNC_OSTREAM_WRAPPER__

#include <functional>
#include <mutex>

namespace srtb {
namespace log {

inline std::mutex log_mutex;

/**
 * @brief Just a std::osyncstream, used because _GLIBCXX_USE_CXX11_ABI is forced to be 0 on CentOS 7
 */
template <typename StreamType>
class sync_stream_wrapper {
 public:
  sync_stream_wrapper(StreamType& stream)
      : _stream{std::ref(stream)}, _lock{log_mutex} {}

  template <typename T>
  sync_stream_wrapper& operator<<(const T& t) {
    _stream.get() << t;
    return *this;
  }

 private:
  std::reference_wrapper<StreamType> _stream;
  std::lock_guard<std::mutex> _lock;
};

}  // namespace log
}  // namespace srtb

#endif  // __SRTB_LOG_SYNC_OSTREAM_WRAPPER__
