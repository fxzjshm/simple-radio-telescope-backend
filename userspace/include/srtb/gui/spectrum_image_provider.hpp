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
#ifndef __SRTB_GUI_SPECTRUM_IMAGE_PROVIDER__
#define __SRTB_GUI_SPECTRUM_IMAGE_PROVIDER__

#include "srtb/commons.hpp"

// Qt related
#include <QImage>
#include <QObject>
#include <QPainter>
#include <QQuickImageProvider>
#include <QThread>

namespace srtb {
namespace gui {
namespace spectrum {

// TODO: where to put the pixmap?
inline constexpr int width = 3840;
inline constexpr int height = 2160;
inline QColor color = Qt::cyan;

/**
 * @brief draw processed spectrum data onto the pixmap
 * 
 * ref: https://github.com/Daguerreo/QML-ImageProvider
 *      Gqrx's plotter: https://github.com/gqrx-sdr/gqrx/blob/master/src/qtgui/plotter.cpp
 *      LMMS's spectrum analyzer plugin: https://github.com/LMMS/lmms/tree/master/plugins/SpectrumAnalyzer
 */

class SpectrumImageProvider : public QObject, public QQuickImageProvider {
  Q_OBJECT

 private:
  QPixmap _pixmap;
  int spectrum_update_counter = 0;

 public:
  explicit SpectrumImageProvider(QObject* parent = nullptr)
      : QObject{parent},
        QQuickImageProvider{QQuickImageProvider::Pixmap},
        _pixmap{width, height} {
    _pixmap.fill(color);
  }

 public slots:
  void update_pixmap() {
    srtb::work::draw_spectrum_work draw_spectrum_work;
    SRTB_LOGD << " [SpectrumImageProvider] "
              << "updating pixmap" << srtb::endl;
    while (srtb::draw_spectrum_queue.pop(draw_spectrum_work) != false) {
      SRTB_LOGD << " [SpectrumImageProvider] "
                << "drawing pixmap" << srtb::endl;
      // draw
      _pixmap.scroll(/* dx = */ 0, /* dy = */ 1, /* x = */ 0, /* y = */ 0,
                     width, height);
      auto ptr = draw_spectrum_work.ptr;
      size_t len =
          sycl::min(draw_spectrum_work.count, static_cast<size_t>(width));
      QPainter painter{&_pixmap};
      // draw new line of fft data at top of waterfall bitmap  -- from Gqrx
      for (size_t i = 0; i < len; i++) {
        QColor local_color = color;
        local_color.setAlphaF(static_cast<qreal>(ptr.get()[i]));
        painter.setPen(color);
        painter.drawPoint(i, 0);
      }
    }
  }

 public:
  // ref: https://stackoverflow.com/questions/45755655/how-to-correctly-use-qt-qml-image-provider
  void trigger_update(QObject* object) {
    // object should be main window
    QMetaObject::invokeMethod(object, "update_spectrum",
                              Q_ARG(QVariant, spectrum_update_counter));
    spectrum_update_counter++;
    SRTB_LOGD << " [SpectrumImageProvider] "
              << "trigger update, spectrum_update_counter = "
              << spectrum_update_counter << srtb::endl;
    //QThread::sleep(1.0/srtb::config.gui_fps);
  }

  QPixmap requestPixmap(const QString& id, QSize* size,
                        const QSize& requestedSize) override {
    SRTB_LOGD << " [SpectrumImageProvider] "
              << "requestPixmap called with id " << id.toStdString()
              << srtb::endl;
    if (size) {
      *size = QSize(width, height);
    }
    QPixmap pixmap_scaled = _pixmap.scaled(
        requestedSize.width() > 0 ? requestedSize.width() : width,
        requestedSize.height() > 0 ? requestedSize.height() : height);
    return pixmap_scaled;
  }
};

}  // namespace spectrum

class TriggerUpdateThread : public QThread {
  Q_OBJECT

 private:
  QObject* _object;
  int spectrum_update_counter = 0;

 public:
  TriggerUpdateThread(QObject* object)
      : _object{object}, spectrum_update_counter{0} {}

  virtual void run() {
    QMetaObject::invokeMethod(_object, "update_spectrum",
                              Q_ARG(QVariant, spectrum_update_counter));
    spectrum_update_counter++;
    SRTB_LOGD << " [TriggerUpdateThread] "
              << "spectrum_update_counter = " << spectrum_update_counter
              << srtb::endl;
    //QThread::sleep(1.0/srtb::config.gui_fps);
  }
};

}  // namespace gui
}  // namespace srtb

#endif  // __SRTB_GUI_SPECTRUM_IMAGE_PROVIDER__
