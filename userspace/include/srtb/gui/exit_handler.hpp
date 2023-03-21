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
#ifndef __SRTB_GUI_EXIT_HANDLER__
#define __SRTB_GUI_EXIT_HANDLER__

#include <thread>
#include <vector>

#include "srtb/commons.hpp"
#include "srtb/pipeline/exit_handler.hpp"

// Qt related things
// some macro defined in Qt may conflict with some STL class, especially C++11 or newer ones
// so put these last
#include <QObject>

namespace srtb {
namespace gui {

/** 
 * @brief this class requests every thread to stop and wait for them
 *        when the program is about to (normally) exit.
 * @note Q_OBJECT does not support template class, so std::array is not used here.
 * @note moved to srtb::pipeline::on_exit() for no-GUI usage
 * @see srtb::termination_handler for handler of unexpected exit
 */
class ExitHandler : public QObject {
  Q_OBJECT

 public:
  std::vector<std::jthread> threads;

 public:
  explicit ExitHandler(std::vector<std::jthread> threads_)
      : threads{std::move(threads_)} {}

 public Q_SLOTS:
  void onExit() {
    srtb::pipeline::on_exit(std::move(threads));
  }
};

}  // namespace gui
}  // namespace srtb

#endif  // __SRTB_GUI_EXIT_HANDLER__
