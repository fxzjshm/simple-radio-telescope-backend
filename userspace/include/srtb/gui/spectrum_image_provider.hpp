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

#include <thread>

#include "srtb/commons.hpp"

// Qt related
#include <QImage>
#include <QObject>
#include <QPainter>
#include <QPointer>
#include <QQuickImageProvider>
#include <QThread>

namespace srtb {
namespace gui {
namespace spectrum {

/** @brief common 8-bit color has 2^8 == 256 values*/
inline constexpr size_t color_value_count = 1 << 8;

/** @brief cached colors to be used, so that don't need to construct temporary objects frequently */
class color_map_holder_t {
 protected:
  std::array<QColor, color_value_count> color_map;
  QColor error_color;

 public:
  color_map_holder_t(QColor color = Qt::cyan,
                     QColor error_color = QColor{255, 127, 127}) {
    set_color(color, error_color);
  }

  void set_color(QColor color, QColor error_color_) {
    int h, s, v, a;
    color.getHsv(&h, &s, &v, &a);
    for (size_t i = 0; i < color_map.size(); i++) {
      const int v2 = static_cast<int>(i);
      color_map[i] = QColor::fromHsv(h, s, v2, a);
    }
    error_color = error_color_;
  }

  auto operator[](size_t i) const -> const QColor& {
    if (i == color_map.size()) [[unlikely]] {
      i -= 1;
    }
    if (0 <= i && i < color_map.size()) [[likely]] {
      return color_map.at(i);
    } else {
      return error_color;
    }
  }
};

inline color_map_holder_t color_map_holder;

/**
 * @brief This class determines how many lines should be drawn at one UI update
 *        for SpectrumImageProvider. 
 * @note This part is split out because the scheduler may be changed frequently
 */
class request_size_scheduler {
 protected:
  size_t request_size = 1;

 public:
  void set_last_size_too_few(bool is_too_few) {
    size_t next_request_size;
    // just a simple thought inspired by 3n+1 problem :)
    if (is_too_few) {
      next_request_size = 3 * request_size + 1;
    } else {
      next_request_size = request_size / 2;
    }
    if (next_request_size == 0) [[unlikely]] {
      next_request_size = 1;
    }
    request_size = next_request_size;
  }

  size_t get_next_request_size() const noexcept {
    //return 1;
    return request_size;
  }
};

/**
 * @brief contains ~10^3 * @c batch_size of @c srtb::real to be drawn to 
 *        @c batch_size lines of a pixmap. @c ptr should be host pointer.
 * @note the work range is ptr.get() + offset ~ ptr.get() + offset + batch_size * count
 */
struct draw_spectrum_lines_work : public srtb::work::draw_spectrum_work {
  size_t offset;
};

/**
 * @brief This class holds a @c srtb::work::draw_spectrum_work and split it into
 *        size that @c SpectrumImageProvider needs (corespond to @c request_size_scheduler )
 * 
 */
class draw_spectrum_work_holder {
 protected:
  srtb::work::draw_spectrum_work draw_spectrum_work;
  /** 
   * @brief correspond to @c batch_size of @c draw_spectrum_work ,
   *        @c work_counter of @c batch_size have done.
   */
  size_t work_counter;

 public:
  explicit draw_spectrum_work_holder() : work_counter{0} {
    // make sure get_one_work do not crash on first run
    draw_spectrum_work.batch_size = 0;
  }

  /**
   * @brief try to get work of @c n_lines_requested lines of the pixmap, 
   *        to be summed or directly drawn.
   * @param draw_lines_work see @c draw_spectrum_lines_work 
   * @param n_lines_requested lines needed.
   * @return @c true if successfully get work and @c work changed, 
   *         @c false if no work available and @c work unchanged.
   *         same as lockfree queue's pop().
   */
  bool get_lines(
      srtb::work_queue<srtb::work::draw_spectrum_work>& draw_spectrum_queue,
      draw_spectrum_lines_work& draw_lines_work,
      const size_t n_lines_requested = 1) {
    if (draw_spectrum_work.batch_size > 0 &&
        work_counter > draw_spectrum_work.batch_size) [[unlikely]] {
      throw std::runtime_error{
          "Unexpected work_counter = " + std::to_string(work_counter) +
          ", batch_size = " + std::to_string(draw_spectrum_work.batch_size)};
    }
    if (n_lines_requested == 0) [[unlikely]] {
      throw std::runtime_error{"Unexpected n_lines_requested = " +
                               std::to_string(n_lines_requested)};
    }
    if (work_counter == draw_spectrum_work.batch_size) {
      // last received work done, get new set of works
      bool ret = draw_spectrum_queue.pop(draw_spectrum_work);
      if (!ret) {
        // no work available in work queue.
        return false;
      }
      work_counter = 0;
    }
    const size_t this_work_batch_size = std::min(
        n_lines_requested, draw_spectrum_work.batch_size - work_counter);
    draw_lines_work.ptr = draw_spectrum_work.ptr;
    draw_lines_work.count = draw_spectrum_work.count;
    draw_lines_work.offset = work_counter * draw_spectrum_work.count;
    draw_lines_work.batch_size = this_work_batch_size;

    work_counter += this_work_batch_size;
    return true;
  }
};

/**
 * @brief draw processed spectrum data onto the pixmap, as a scrolling waterfall
 * 
 * ref: https://github.com/Daguerreo/QML-ImageProvider
 *      Gqrx's plotter: https://github.com/gqrx-sdr/gqrx/blob/master/src/qtgui/plotter.cpp
 *      LMMS's spectrum analyzer plugin: https://github.com/LMMS/lmms/tree/master/plugins/SpectrumAnalyzer
 */
class SpectrumImageProvider : public QObject, public QQuickImageProvider {
  Q_OBJECT

 protected:
  QPixmap pixmap;
  int spectrum_update_counter = 0;
  request_size_scheduler request_size_scheduler;
  draw_spectrum_work_holder work_holder;
  srtb::work_queue<srtb::work::draw_spectrum_work>& draw_spectrum_queue;

 public:
  explicit SpectrumImageProvider(
      QObject* parent,
      srtb::work_queue<srtb::work::draw_spectrum_work>& draw_spectrum_queue_)
      : QObject{parent},
        QQuickImageProvider{QQuickImageProvider::Pixmap},
        pixmap{static_cast<int>(srtb::config.gui_pixmap_width),
               static_cast<int>(srtb::config.gui_pixmap_height)},
        draw_spectrum_queue{draw_spectrum_queue_} {}

 public Q_SLOTS:
  void update_pixmap() {
    const size_t requested_lines_count =
        request_size_scheduler.get_next_request_size();
    size_t requested_lines_count_remained = requested_lines_count;
    draw_spectrum_lines_work draw_lines_work;
    // TODO: optimize memory allocation here
    std::vector<draw_spectrum_lines_work> works;
    bool ret = true;
    while (requested_lines_count_remained > 0) {
      ret = work_holder.get_lines(draw_spectrum_queue, draw_lines_work,
                                  requested_lines_count_remained);
      if (ret) {
        requested_lines_count_remained -= draw_lines_work.batch_size;
        works.push_back(draw_lines_work);
      } else {
        // no work available
        break;
      }
    }
    if (!ret) {
      // no work available now, request less next time
      request_size_scheduler.set_last_size_too_few(false);
    } else {
      // still some work in queue, request more next time
      request_size_scheduler.set_last_size_too_few(true);
    }

    if (works.size() == 0) [[unlikely]] {
      return;
    }

    // now `works` contains all lines needed to draw this update
    // note that at most `height` (of pixmap) lines needed to draw, skip unused lines.
    size_t k = works.size() - 1, lines_to_draw = 0;
    while (1) {
      if (lines_to_draw + works[k].batch_size <
          static_cast<size_t>(pixmap.height())) {
        lines_to_draw += works[k].batch_size;
        if (k == 0) {
          break;
        } else {
          k--;
        }
      } else {
        works[k].offset +=
            works[k].count *
            (lines_to_draw + works[k].batch_size - pixmap.height());
        works[k].batch_size = pixmap.height() - lines_to_draw;
        lines_to_draw = pixmap.height();
        break;
      }
    }
    // draw new line of fft data at top of waterfall bitmap
    pixmap.scroll(/* dx = */ 0, /* dy = */ lines_to_draw, /* x = */ 0,
                  /* y = */ 0, pixmap.width(), pixmap.height());
    QPainter painter{&pixmap};
    for (; k < works.size(); k++) {
      const size_t count = works[k].count;
      const size_t batch_size = works[k].batch_size;
      const size_t offset = works[k].offset;
      auto ptr = works[k].ptr.get();
      size_t x_max = sycl::min(count, static_cast<size_t>(pixmap.width()));
      assert(batch_size <= static_cast<size_t>(pixmap.height()));
      //SRTB_LOGD << " [SpectrumImageProvider] "
      //          << "drawing pixmap, len = " << x_max << srtb::endl;
      for (size_t j = 0; j < batch_size; j++) {
        for (size_t i = 0; i < x_max; i++) {
          //SRTB_LOGD << " [SpectrumImageProvider] " << "ptr[" << i << "] = " << ptr[i] << srtb::endl;
          assert(i + j * count < count * batch_size);

          // 'v2' is new value of 'v' in HSV form
          int v2 =
              static_cast<int>(color_value_count * ptr[offset + j * count + i]);
          // invalid v2 values should be handled properly in operator[]
          QColor local_color = color_map_holder[v2];

          painter.setPen(local_color);
          painter.drawPoint(i, lines_to_draw - j - 1);
        }
      }
      lines_to_draw -= batch_size;
    }
    //SRTB_LOGD << " [SpectrumImageProvider] "
    //          << "updated pixmap, count = " << update_count << srtb::endl;
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
      *size = QSize(pixmap.width(), pixmap.height());
    }
    QPixmap pixmap_scaled;
    if (requestedSize.width() == pixmap.width() &&
        requestedSize.height() == pixmap.height()) {
      pixmap_scaled = pixmap;
    } else {
      pixmap_scaled = pixmap.scaled(
          requestedSize.width() > 0 ? requestedSize.width() : pixmap.width(),
          requestedSize.height() > 0 ? requestedSize.height()
                                     : pixmap.height());
    }
    return pixmap_scaled;
  }
};

// ------------------------------------------------------------------------

/**
 * @brief draw processed spectrum data onto the pixmap
 */
class SimpleSpectrumImageProvider : public QObject, public QQuickImageProvider {
  Q_OBJECT

 public:
  QImage image;
  std::shared_ptr<uint32_t> image_data;
  // currently alpha channel of all color used are 0xFF
  static constexpr QImage::Format image_format = QImage::Format_RGB32;
  int spectrum_update_counter = 0;
  srtb::work_queue<srtb::work::draw_spectrum_work_2>& draw_spectrum_queue_2;
  std::jthread checker_thread;
  QPointer<QObject> parent;

 public:
  explicit SimpleSpectrumImageProvider(
      QObject* parent_, srtb::work_queue<srtb::work::draw_spectrum_work_2>&
                            draw_spectrum_queue_2_)
      : QObject{parent_},
        QQuickImageProvider{QQuickImageProvider::Pixmap},
        image{static_cast<int>(srtb::config.gui_pixmap_width),
              static_cast<int>(srtb::config.gui_pixmap_height), image_format},
        draw_spectrum_queue_2{draw_spectrum_queue_2_},
        parent{parent_} {
    checker_thread = std::jthread{[this](std::stop_token stop_token) {
      while (!stop_token.stop_requested()) [[likely]] {
        srtb::work::draw_spectrum_work_2 draw_spectrum_work;
        const bool got_work = draw_spectrum_queue_2.pop(draw_spectrum_work);
        if (got_work) {
          Q_EMIT new_pixmap_available(draw_spectrum_work);
        }
        std::this_thread::sleep_for(
            std::chrono::nanoseconds(srtb::config.thread_query_work_wait_time));
      }
    }};

    QObject::connect(
        this, &SimpleSpectrumImageProvider::new_pixmap_available,
        [this](srtb::work::draw_spectrum_work_2 draw_spectrum_work) {
          SRTB_LOGD << " [gui] "
                    << "update_pixmap " << srtb::endl;
          update_pixmap(draw_spectrum_work);
          trigger_update();
        });
  }

 Q_SIGNALS:
  void new_pixmap_available(
      srtb::work::draw_spectrum_work_2 draw_spectrum_work);

 public Q_SLOTS:
  /** @return true if actually updated, false if not. */
  void update_pixmap(srtb::work::draw_spectrum_work_2 draw_spectrum_work) {
    const int width = draw_spectrum_work.width;
    const int height = draw_spectrum_work.height;
    if (image.width() != width || image.height() != height) [[unlikely]] {
      SRTB_LOGW << " [SimpleSpectrumImageProvider] "
                << "resizing image from " << image.width() << " x "
                << image.height() << " to " << width << " x " << height
                << srtb::endl;
      image = QImage{width, height, image_format};
    }

    auto h_image_shared = draw_spectrum_work.ptr;
    uint32_t* h_image = h_image_shared.get();
    image =
        QImage{reinterpret_cast<uchar*>(h_image), width, height, image_format};
    // TODO: check lifetime of old pixmap data
    image_data = h_image_shared;

    SRTB_LOGD << " [SimpleSpectrumImageProvider] "
              << "updated pixmap, counter = " << spectrum_update_counter
              << srtb::endl;
  }

 public:
  // ref: https://stackoverflow.com/questions/45755655/how-to-correctly-use-qt-qml-image-provider
  void trigger_update() {
    if (parent) {  // object should be main window
      QMetaObject::invokeMethod(parent, "update_spectrum",
                                Q_ARG(QVariant, spectrum_update_counter));
      spectrum_update_counter++;
      SRTB_LOGD << " [SimpleSpectrumImageProvider] "
                << "trigger update, spectrum_update_counter = "
                << spectrum_update_counter << srtb::endl;
    }
  }

  QPixmap requestPixmap(const QString& id, QSize* size,
                        const QSize& requestedSize) override {
    SRTB_LOGD << " [SimpleSpectrumImageProvider] "
              << "requestImage called with id " << id.toStdString()
              << srtb::endl;
    (void)id;
    if (size) {
      *size = QSize(image.width(), image.height());
    }
    QImage image_scaled = image.scaled(
        requestedSize.width() > 0 ? requestedSize.width() : image.width(),
        requestedSize.height() > 0 ? requestedSize.height() : image.height());
    return QPixmap::fromImage(image_scaled);
  }
};

}  // namespace spectrum
}  // namespace gui
}  // namespace srtb

#endif  // __SRTB_GUI_SPECTRUM_IMAGE_PROVIDER__
