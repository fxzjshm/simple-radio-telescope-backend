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
#ifndef __SRTB_GUI__
#define __SRTB_GUI__

#include <thread>
#include <vector>

// Qt related
#include <QGuiApplication>
#include <QPointer>
#include <QQmlApplicationEngine>
#include <QQmlComponent>
#include <QQuickWindow>

#include "srtb/gui/spectrum_image_provider.hpp"

namespace srtb {
namespace gui {

// QML related things in src/main.qml, which is treated as a .cpp file.

inline int show_gui(int argc, char **argv, auto &draw_spectrum_queue_2) {
#if QT_VERSION < QT_VERSION_CHECK(6, 0, 0)
  QCoreApplication::setAttribute(Qt::AA_EnableHighDpiScaling);
#endif
  QGuiApplication app(argc, argv);

  QQmlApplicationEngine engine;
  // an explicit operator new here because "The QQmlEngine takes ownership of provider."
  // otherwise a double free is going to happen
  QPointer spectrum_image_provider_ptr{
      new srtb::gui::spectrum::SimpleSpectrumImageProvider{
          /* parent = */ nullptr, draw_spectrum_queue_2}};
  engine.addImageProvider(QLatin1String("spectrum-image-provider"),
                          spectrum_image_provider_ptr);
  const QUrl url(QStringLiteral("qrc:/main.qml"));
  QObject::connect(
      &engine, &QQmlApplicationEngine::objectCreated, &app,
      [url, spectrum_image_provider_ptr](QObject *obj, const QUrl &objUrl) {
        SRTB_LOGD << " [gui] " << objUrl.toString().toStdString() << srtb::endl;
        if (url == objUrl) {
          if (obj) [[likely]] {
            spectrum_image_provider_ptr->parent = obj;
          } else {
            SRTB_LOGE << " [gui] "
                      << "load main window failed!" << srtb::endl;
            QCoreApplication::exit(-1);
          }
        }
      },
      Qt::QueuedConnection);
  engine.load(url);

  return app.exec();
}

}  // namespace gui
}  // namespace srtb

#endif  // __SRTB_GUI__
