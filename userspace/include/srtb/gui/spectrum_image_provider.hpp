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
inline constexpr int max_draw_update_count = 100;
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
  QPixmap pixmap;
  int spectrum_update_counter = 0;

 public:
  explicit SpectrumImageProvider(QObject* parent = nullptr)
      : QObject{parent},
        QQuickImageProvider{QQuickImageProvider::Pixmap},
        pixmap{width, height} {
    //pixmap.fill(color);
  }

 public slots:
  void update_pixmap() {
    srtb::work::draw_spectrum_work draw_spectrum_work;
    size_t update_count = 0;
    while (srtb::draw_spectrum_queue.pop(draw_spectrum_work) != false) {
      // draw new line of fft data at top of waterfall bitmap  -- from Gqrx
      const size_t count = draw_spectrum_work.count;
      const size_t batch_size = draw_spectrum_work.batch_size;
      pixmap.scroll(/* dx = */ 0, /* dy = */ batch_size, /* x = */ 0,
                    /* y = */ 0, width, height);
      auto ptr = draw_spectrum_work.ptr.get();
      size_t x_max = sycl::min(count, static_cast<size_t>(width));
      size_t y_max = sycl::min(batch_size, static_cast<size_t>(height));
      //SRTB_LOGD << " [SpectrumImageProvider] "
      //          << "drawing pixmap, len = " << x_max << srtb::endl;
      QPainter painter{&pixmap};
      for (size_t j = 0; j < y_max; j++) {
        for (size_t i = 0; i < x_max; i++) {
          //SRTB_LOGD << " [SpectrumImageProvider] " << "ptr[" << i << "] = " << ptr[i] << srtb::endl;
          assert(i + j * count < count * batch_size);

          qreal h, s, v, a;
          color.getHsvF(&h, &s, &v, &a);
          v *= static_cast<qreal>(ptr[i + j * count]);
          QColor local_color = QColor::fromHsvF(h, s, v, a);

          painter.setPen(local_color);
          painter.drawPoint(i, y_max - j - 1);
        }
      }
      update_count++;
      if (update_count > max_draw_update_count) {
        SRTB_LOGW << " [SpectrumImageProvider] "
                  << "update count limit exceeded, abort updating. "
                  << "Maybe system load is too high." << srtb::endl;
        break;
      }
    }
    SRTB_LOGD << " [SpectrumImageProvider] "
              << "updated pixmap, count = " << update_count << srtb::endl;
  }

 public:
  // ref: https://stackoverflow.com/questions/45755655/how-to-correctly-use-qt-qml-image-provider
  void trigger_update(QObject* object) {
    // object should be main window
    QMetaObject::invokeMethod(object, "update_spectrum",
                              Q_ARG(QVariant, spectrum_update_counter));
    spectrum_update_counter++;
    //SRTB_LOGD << " [SpectrumImageProvider] "
    //          << "trigger update, spectrum_update_counter = "
    //          << spectrum_update_counter << srtb::endl;
  }

  QPixmap requestPixmap(const QString& id, QSize* size,
                        const QSize& requestedSize) override {
    //SRTB_LOGD << " [SpectrumImageProvider] "
    //          << "requestPixmap called with id " << id.toStdString()
    //          << srtb::endl;
    (void)id;
    if (size) {
      *size = QSize(width, height);
    }
    QPixmap pixmap_scaled = pixmap.scaled(
        requestedSize.width() > 0 ? requestedSize.width() : width,
        requestedSize.height() > 0 ? requestedSize.height() : height);
    return pixmap_scaled;
  }
};

}  // namespace spectrum
}  // namespace gui
}  // namespace srtb

#endif  // __SRTB_GUI_SPECTRUM_IMAGE_PROVIDER__
