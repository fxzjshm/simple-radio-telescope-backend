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
inline constexpr int width = 640;
inline constexpr int height = 480;
inline constexpr int max_draw_update_count = 2;

/** @brief common 8-bit color has 2^8 == 256 values*/
inline constexpr size_t color_value_count = 1 << 8;

/** @brief cached colors to be used, so that don't need to construct temporary objects frequently */
class color_map_holder_t {
 protected:
  std::array<QColor, color_value_count> color_map;

 public:
  color_map_holder_t(QColor color = Qt::cyan) { set_color(color); }

  void set_color(QColor color) {
    int h, s, v, a;
    color.getHsv(&h, &s, &v, &a);
    for (size_t i = 0; i < color_map.size(); i++) {
      const int v2 = static_cast<int>(i);
      color_map[i] = QColor::fromHsv(h, s, v2, a);
    }
  }

  auto operator[](size_t i) const -> const QColor& { return color_map.at(i); }
};

inline color_map_holder_t color_map_holder;

/**
 * @brief This class determines how many lines should be drawn at one UI update
 *        for SpectrumImageProvider. 
 * @note This part is split out because the scheduler may be changed frequently
 */
class request_size_scheduler {
 protected:
  size_t next_request_size = 1;

 public:
  void set_last_size_too_few(bool is_too_few) {
    if (is_too_few) {
      next_request_size++;
    } else {
      next_request_size--;
    }
  }

  size_t get_next_request_size() const noexcept {
    //return 1;
    return next_request_size;
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
  bool get_lines(draw_spectrum_lines_work& draw_lines_work,
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
      bool ret = srtb::draw_spectrum_queue.pop(draw_spectrum_work);
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
 * @brief draw processed spectrum data onto the pixmap
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

 public:
  explicit SpectrumImageProvider(QObject* parent = nullptr)
      : QObject{parent},
        QQuickImageProvider{QQuickImageProvider::Pixmap},
        pixmap{width, height} {
    //pixmap.fill(color);
  }

 public slots:
  void update_pixmap() {
    const size_t requested_lines_count =
        request_size_scheduler.get_next_request_size();
    size_t requested_lines_count_remained = requested_lines_count;
    draw_spectrum_lines_work draw_lines_work;
    // TODO: optimize memory allocation here
    std::vector<draw_spectrum_lines_work> works;
    bool ret = true;
    while (requested_lines_count_remained > 0) {
      ret = work_holder.get_lines(draw_lines_work,
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
      if (lines_to_draw + works[k].batch_size < height) {
        lines_to_draw += works[k].batch_size;
        if (k == 0) {
          break;
        } else {
          k--;
        }
      } else {
        works[k].offset +=
            works[k].count * (lines_to_draw + works[k].batch_size - height);
        works[k].batch_size = height - lines_to_draw;
        lines_to_draw = height;
        break;
      }
    }
    // draw new line of fft data at top of waterfall bitmap  -- from Gqrx
    pixmap.scroll(/* dx = */ 0, /* dy = */ lines_to_draw, /* x = */ 0,
                  /* y = */ 0, width, height);
    QPainter painter{&pixmap};
    for (; k < works.size(); k++) {
      const size_t count = works[k].count;
      const size_t batch_size = works[k].batch_size;
      const size_t offset = works[k].offset;
      auto ptr = works[k].ptr.get();
      size_t x_max = sycl::min(count, static_cast<size_t>(width));
      assert(batch_size <= static_cast<size_t>(height));
      //SRTB_LOGD << " [SpectrumImageProvider] "
      //          << "drawing pixmap, len = " << x_max << srtb::endl;
      for (size_t j = 0; j < batch_size; j++) {
        for (size_t i = 0; i < x_max; i++) {
          //SRTB_LOGD << " [SpectrumImageProvider] " << "ptr[" << i << "] = " << ptr[i] << srtb::endl;
          assert(i + j * count < count * batch_size);

          // 'v2' is new value of 'v' in HSV form
          size_t v2 = static_cast<size_t>(color_value_count *
                                          ptr[offset + j * count + i]);
          if (v2 >= color_value_count) [[unlikely]] {
            v2 = color_value_count - 1;
          }
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
