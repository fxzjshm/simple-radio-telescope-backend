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
#ifndef __SRTB_PIPELINE_UNPACK_PIPE__
#define __SRTB_PIPELINE_UNPACK_PIPE__

#include <array>

#include "srtb/fft/fft_window.hpp"
#include "srtb/pipeline/framework/pipe.hpp"
#include "srtb/pipeline/framework/pipe_io.hpp"
#include "srtb/unpack.hpp"

namespace srtb {
namespace pipeline {

/**
 * @brief this pipe reads from @c srtb::unpack_queue, unpack and apply FFT window
 *        to input baseband data, then push work to @c srtb::fft_1d_r2c_queue
 * @note this trivial unpack pipe unpacks 1 data source to 1 polarization,
 *       correspond baseband_format_type = "simple"
 */
class unpack_pipe {
 protected:
  sycl::queue q;
  srtb::fft::fft_window_functor_manager<srtb::real, srtb::fft::default_window>
      window_functor_manager;

 public:
  unpack_pipe(sycl::queue q_)
      : q{q_},
        window_functor_manager{srtb::fft::default_window{},
                               /* n = */ srtb::config.baseband_input_count, q} {
  }

  auto operator()([[maybe_unused]] std::stop_token stop_token,
                  srtb::work::unpack_work unpack_work) {
    // data length after unpack
    const int baseband_input_bits = srtb::config.baseband_input_bits;
    const size_t out_count =
        unpack_work.count * srtb::BITS_PER_BYTE / std::abs(baseband_input_bits);

    // re-construct fft_window_functor_manager if length mismatch
    if (out_count != window_functor_manager.functor.n) [[unlikely]] {
      SRTB_LOGW << " [unpack pipe] "
                << "re-construct fft_window_functor_manager of size "
                << out_count << srtb::endl;
      window_functor_manager =
          srtb::fft::fft_window_functor_manager<srtb::real,
                                                srtb::fft::default_window>{
              srtb::fft::default_window{}, out_count, q};
    }

    auto& d_in_shared = unpack_work.ptr;
    // size += 2 because fft_pipe may operate in-place
    auto d_out_shared =
        srtb::device_allocator.allocate_shared<srtb::real>(out_count + 2);
    auto d_in = d_in_shared.get();
    auto d_out = d_out_shared.get();
    // runtime dispatch of different input bits
    // TODO: baseband_input_bits = -4
    if (baseband_input_bits == 1) {
      // 1 -> std::byte
      srtb::unpack::unpack<1>(d_in, d_out, out_count,
                              window_functor_manager.functor, q);
    } else if (baseband_input_bits == 2) {
      // 2 -> std::byte
      srtb::unpack::unpack<2>(d_in, d_out, out_count,
                              window_functor_manager.functor, q);
    } else if (baseband_input_bits == 4) {
      // 4 -> std::byte
      srtb::unpack::unpack<4>(d_in, d_out, out_count,
                              window_functor_manager.functor, q);
    } else if (baseband_input_bits == sizeof(uint8_t) * srtb::BITS_PER_BYTE) {
      // 8 -> uint8_t / unsigned char / u8
      using T = uint8_t;
      srtb::unpack::unpack<sizeof(T) * srtb::BITS_PER_BYTE>(
          reinterpret_cast<T*>(d_in), d_out, out_count,
          window_functor_manager.functor, q);
    } else if (baseband_input_bits ==
               -int{sizeof(int8_t) * srtb::BITS_PER_BYTE}) {
      // -8 -> int8_t / signed char / i8
      using T = int8_t;
      srtb::unpack::unpack<sizeof(T) * srtb::BITS_PER_BYTE>(
          reinterpret_cast<T*>(d_in), d_out, out_count,
          window_functor_manager.functor, q);
    } else if (baseband_input_bits ==
               int{sizeof(uint16_t) * srtb::BITS_PER_BYTE}) {
      // 16 -> uint16_t / unsigned short int / u16
      using T = uint16_t;
      srtb::unpack::unpack<sizeof(T) * srtb::BITS_PER_BYTE>(
          reinterpret_cast<T*>(d_in), d_out, out_count,
          window_functor_manager.functor, q);
    } else if (baseband_input_bits ==
               -int{sizeof(int16_t) * srtb::BITS_PER_BYTE}) {
      // -16 -> int16_t / signed short int / i16
      using T = int16_t;
      srtb::unpack::unpack<sizeof(T) * srtb::BITS_PER_BYTE>(
          reinterpret_cast<T*>(d_in), d_out, out_count,
          window_functor_manager.functor, q);
    } else if (baseband_input_bits == sizeof(float) * srtb::BITS_PER_BYTE) {
      // 32 -> float / f32
      using T = float;
      srtb::unpack::unpack<sizeof(T) * srtb::BITS_PER_BYTE>(
          reinterpret_cast<T*>(d_in), d_out, out_count,
          window_functor_manager.functor, q);
    } else if (baseband_input_bits == sizeof(double) * srtb::BITS_PER_BYTE) {
      // 64 -> double / f64
      using T = double;
      srtb::unpack::unpack<sizeof(T) * srtb::BITS_PER_BYTE>(
          reinterpret_cast<T*>(d_in), d_out, out_count,
          window_functor_manager.functor, q);
    } else {
      throw std::runtime_error(
          "[unpack pipe] unsupported baseband_input_bits = " +
          std::to_string(baseband_input_bits));
    }
    d_in = nullptr;
    d_in_shared.reset();

    srtb::work::fft_1d_r2c_work fft_1d_r2c_work;
    fft_1d_r2c_work.move_parameter_from(std::move(unpack_work));
    fft_1d_r2c_work.ptr = d_out_shared;
    fft_1d_r2c_work.count = out_count;
    return std::optional{fft_1d_r2c_work};
  }
};

/**
 * @brief this pipe reads from @c srtb::unpack_queue, unpack and apply FFT window
 *        to input baseband data, then push work to @c srtb::fft_1d_r2c_queue.
 * @note this pipe unpacks interleaved 2 polarizations into 2 segments, each contains 1 polariztion.
 *       pay attention that when 1 work is eaten by this pipe, 2 works are given out.
 *       correspond baseband_format_type = "interleaved_samples_2"
 */
class unpack_interleaved_samples_2_pipe {
 public:
  using in_work_type = srtb::work::unpack_work;
  using out_work_type = srtb::work::fft_1d_r2c_work;

 protected:
  sycl::queue q;
  srtb::fft::fft_window_functor_manager<srtb::real, srtb::fft::default_window>
      window_functor_manager;

 public:
  unpack_interleaved_samples_2_pipe(sycl::queue q_)
      : q{q_},
        window_functor_manager{srtb::fft::default_window{},
                               /* n = */ srtb::config.baseband_input_count, q} {
  }

  auto operator()([[maybe_unused]] std::stop_token stop_token,
                  srtb::work::unpack_work unpack_work) {
    const int baseband_input_bits = srtb::config.baseband_input_bits;
    // out_count is count in each d_out, so / 2 here
    const size_t out_count = unpack_work.count * srtb::BITS_PER_BYTE /
                             std::abs(baseband_input_bits) / 2;

    // re-construct fft_window_functor_manager if length mismatch
    if (out_count != window_functor_manager.functor.n) [[unlikely]] {
      SRTB_LOGW << " [unpack pipe] "
                << "re-construct fft_window_functor_manager of size "
                << out_count << srtb::endl;
      window_functor_manager =
          srtb::fft::fft_window_functor_manager<srtb::real,
                                                srtb::fft::default_window>{
              srtb::fft::default_window{}, out_count, q};
    }

    auto& d_in_shared = unpack_work.ptr;
    // size += 2 because fft_pipe may operate in-place
    auto d_out_1_shared =
        srtb::device_allocator.allocate_shared<srtb::real>(out_count + 2);
    auto d_out_2_shared =
        srtb::device_allocator.allocate_shared<srtb::real>(out_count + 2);
    auto d_in = d_in_shared.get();
    auto d_out_1 = d_out_1_shared.get();
    auto d_out_2 = d_out_2_shared.get();

    // runtime dispatch of different input bits
    // TODO: baseband_input_bits = 1, 2, 4
    if (baseband_input_bits == sizeof(uint8_t) * srtb::BITS_PER_BYTE) {
      // 8 -> uint8_t / unsigned char / u8
      using T = uint8_t;
      srtb::unpack::unpack<sizeof(T) * srtb::BITS_PER_BYTE>(
          reinterpret_cast<T*>(d_in), d_out_1, d_out_2, out_count,
          window_functor_manager.functor, q);

    } else if (baseband_input_bits ==
               -int{sizeof(int8_t) * srtb::BITS_PER_BYTE}) {
      // -8 -> int8_t / signed char / i8
      using T = int8_t;
      // std::string.contains (c++23) :(
      if (srtb::config.baseband_format_type.find("naocpsr_snap1") !=
          std::string::npos) {
        srtb::unpack::unpack_naocpsr_snap1(reinterpret_cast<T*>(d_in), d_out_1,
                                           d_out_2, out_count,
                                           window_functor_manager.functor, q);
      } else {
        srtb::unpack::unpack<sizeof(T) * srtb::BITS_PER_BYTE>(
            reinterpret_cast<T*>(d_in), d_out_1, d_out_2, out_count,
            window_functor_manager.functor, q);
      }
    } else if (baseband_input_bits ==
               int{sizeof(uint16_t) * srtb::BITS_PER_BYTE}) {
      // 16 -> uint16_t / unsigned short int / u16
      using T = uint16_t;
      srtb::unpack::unpack<sizeof(T) * srtb::BITS_PER_BYTE>(
          reinterpret_cast<T*>(d_in), d_out_1, d_out_2, out_count,
          window_functor_manager.functor, q);
    } else if (baseband_input_bits ==
               -int{sizeof(int16_t) * srtb::BITS_PER_BYTE}) {
      // -16 -> int16_t / signed short int / i16
      using T = int16_t;
      srtb::unpack::unpack<sizeof(T) * srtb::BITS_PER_BYTE>(
          reinterpret_cast<T*>(d_in), d_out_1, d_out_2, out_count,
          window_functor_manager.functor, q);
    } else if (baseband_input_bits == sizeof(float) * srtb::BITS_PER_BYTE) {
      // 32 -> float / f32
      using T = float;
      srtb::unpack::unpack<sizeof(T) * srtb::BITS_PER_BYTE>(
          reinterpret_cast<T*>(d_in), d_out_1, d_out_2, out_count,
          window_functor_manager.functor, q);
    } else if (baseband_input_bits == sizeof(double) * srtb::BITS_PER_BYTE) {
      // 64 -> double / f64
      using T = double;
      srtb::unpack::unpack<sizeof(T) * srtb::BITS_PER_BYTE>(
          reinterpret_cast<T*>(d_in), d_out_1, d_out_2, out_count,
          window_functor_manager.functor, q);
    } else {
      throw std::runtime_error(
          "[unpack_2pol_interleave_pipe] unsupported baseband_input_bits = " +
          std::to_string(baseband_input_bits));
    }
    d_in = nullptr;
    d_in_shared.reset();

    srtb::work::fft_1d_r2c_work fft_1d_r2c_work_1, fft_1d_r2c_work_2;
    fft_1d_r2c_work_1.copy_parameter_from(unpack_work);
    fft_1d_r2c_work_1.data_stream_id = unpack_work.data_stream_id * 2;
    fft_1d_r2c_work_1.ptr = d_out_1_shared;
    fft_1d_r2c_work_1.count = out_count;
    fft_1d_r2c_work_2.copy_parameter_from(unpack_work);
    fft_1d_r2c_work_1.data_stream_id = unpack_work.data_stream_id * 2 + 1;
    fft_1d_r2c_work_2.ptr = d_out_2_shared;
    fft_1d_r2c_work_2.count = out_count;
    return std::optional{std::array{fft_1d_r2c_work_1, fft_1d_r2c_work_2}};
  }
};

class unpack_gznupsr_a1_pipe {
 public:
  using in_work_type = srtb::work::unpack_work;
  using out_work_type = srtb::work::fft_1d_r2c_work;

 protected:
  sycl::queue q;
  srtb::fft::fft_window_functor_manager<srtb::real, srtb::fft::default_window>
      window_functor_manager;

 public:
  unpack_gznupsr_a1_pipe(sycl::queue q_)
      : q{q_},
        window_functor_manager{srtb::fft::default_window{},
                               /* n = */ srtb::config.baseband_input_count, q} {
  }

  auto operator()([[maybe_unused]] std::stop_token stop_token,
                  srtb::work::unpack_work unpack_work) {
    const int baseband_input_bits = 8;
    // out_count is count in each d_out, so / 4 here
    const size_t out_count = unpack_work.count * srtb::BITS_PER_BYTE /
                             std::abs(baseband_input_bits) / 4;

    // re-construct fft_window_functor_manager if length mismatch
    if (out_count != window_functor_manager.functor.n) [[unlikely]] {
      SRTB_LOGW << " [unpack pipe] "
                << "re-construct fft_window_functor_manager of size "
                << out_count << srtb::endl;
      window_functor_manager =
          srtb::fft::fft_window_functor_manager<srtb::real,
                                                srtb::fft::default_window>{
              srtb::fft::default_window{}, out_count, q};
    }

    auto& d_in_shared = unpack_work.ptr;
    auto d_in = d_in_shared.get();

    std::array<std::shared_ptr<srtb::real>, 4> d_out_shared;
    std::array<srtb::real*, 4> d_out;
    for (size_t i = 0; i < d_out_shared.size(); i++) {
      // size += 2 because fft_pipe may operate in-place
      d_out_shared[i] =
          srtb::device_allocator.allocate_shared<srtb::real>(out_count + 2);
      d_out[i] = d_out_shared[i].get();
    }

    srtb::unpack::unpack_gznupsr_a1(reinterpret_cast<int8_t*>(d_in), d_out[0],
                                    d_out[1], d_out[2], d_out[3], out_count,
                                    window_functor_manager.functor, q);

    d_in = nullptr;
    d_in_shared.reset();

    std::array<srtb::work::fft_1d_r2c_work, 4> fft_1d_r2c_work;
    for (size_t i = 0; i < fft_1d_r2c_work.size(); i++) {
      fft_1d_r2c_work[i].copy_parameter_from(unpack_work);
      fft_1d_r2c_work[i].data_stream_id = 4 * unpack_work.data_stream_id + i;
      fft_1d_r2c_work[i].ptr = d_out_shared[i];
      fft_1d_r2c_work[i].count = out_count;
    }
    return std::optional{fft_1d_r2c_work};
  }
};

class unpack_gznupsr_a1_v2_1_pipe {
 public:
  using in_work_type = srtb::work::unpack_work;
  using out_work_type = srtb::work::fft_1d_r2c_work;

 protected:
  sycl::queue q;
  srtb::fft::fft_window_functor_manager<srtb::real, srtb::fft::default_window>
      window_functor_manager;

 public:
  unpack_gznupsr_a1_v2_1_pipe(sycl::queue q_)
      : q{q_},
        window_functor_manager{srtb::fft::default_window{},
                               /* n = */ srtb::config.baseband_input_count, q} {
  }

  auto operator()([[maybe_unused]] std::stop_token stop_token,
                  srtb::work::unpack_work unpack_work) {
    const int baseband_input_bits = 8;
    // out_count is count in each d_out, so / 4 here
    const size_t out_count = unpack_work.count * srtb::BITS_PER_BYTE /
                             std::abs(baseband_input_bits) / 2;

    // re-construct fft_window_functor_manager if length mismatch
    if (out_count != window_functor_manager.functor.n) [[unlikely]] {
      SRTB_LOGW << " [unpack pipe] "
                << "re-construct fft_window_functor_manager of size "
                << out_count << srtb::endl;
      window_functor_manager =
          srtb::fft::fft_window_functor_manager<srtb::real,
                                                srtb::fft::default_window>{
              srtb::fft::default_window{}, out_count, q};
    }

    auto& d_in_shared = unpack_work.ptr;
    auto d_in = d_in_shared.get();

    std::array<std::shared_ptr<srtb::real>, 2> d_out_shared;
    std::array<srtb::real*, 2> d_out;
    for (size_t i = 0; i < d_out_shared.size(); i++) {
      // size += 2 because fft_pipe may operate in-place
      d_out_shared[i] =
          srtb::device_allocator.allocate_shared<srtb::real>(out_count + 2);
      d_out[i] = d_out_shared[i].get();
    }

    srtb::unpack::unpack_gznupsr_a1(reinterpret_cast<int8_t*>(d_in), d_out[0],
                                    d_out[1], out_count,
                                    window_functor_manager.functor, q);

    d_in = nullptr;
    d_in_shared.reset();

    std::array<srtb::work::fft_1d_r2c_work, 2> fft_1d_r2c_work;
    for (size_t i = 0; i < fft_1d_r2c_work.size(); i++) {
      fft_1d_r2c_work[i].copy_parameter_from(unpack_work);
      fft_1d_r2c_work[i].data_stream_id = 2 * unpack_work.data_stream_id + i;
      fft_1d_r2c_work[i].ptr = d_out_shared[i];
      fft_1d_r2c_work[i].count = out_count;
    }
    return std::optional{fft_1d_r2c_work};
  }
};

template <typename InFunctor, typename OutFunctor, typename... Args>
inline auto start_unpack_pipe(std::string_view backend_name, sycl::queue q,
                              InFunctor in_functor, OutFunctor out_functor,
                              Args... args) {
  using namespace srtb::io::backend_registry;

  if (backend_name == naocpsr_roach2::name) {
    return start_pipe<unpack_pipe>(q, in_functor, out_functor, args...);
  }
  if (backend_name == naocpsr_snap1::name) {
    return start_pipe<unpack_interleaved_samples_2_pipe>(
        q, in_functor, multiple_works_out_functor{out_functor}, args...);
  }
  if (backend_name == gznupsr_a1::name) {
    return start_pipe<unpack_gznupsr_a1_v2_1_pipe>(
        q, in_functor, multiple_works_out_functor{out_functor}, args...);
  }
  throw std::invalid_argument{"[start_unpack_pipe] Unknown backend name: " +
                              std::string{backend_name}};
}

}  // namespace pipeline
}  // namespace srtb

#endif  // __SRTB_PIPELINE_UNPACK_PIPE__
