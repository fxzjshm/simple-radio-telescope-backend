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

#include <atomic>
#include <boost/asio.hpp>
#include <chrono>
#include <future>

#include "srtb/commons.hpp"
#include "srtb/io/udp_receiver.hpp"
#include "srtb/pipeline/udp_receiver_pipe.hpp"

#define SRTB_CHECK_TEST_UDP_RECEIVER(expr)                             \
  SRTB_CHECK(expr, true, {                                             \
    throw std::runtime_error{                                          \
        "[test-udp_receiver] " #expr " at " __FILE__ ":" +             \
        std::to_string(__LINE__) + " returns " + std::to_string(ret)}; \
  })

std::atomic<bool> g_test_passes = false;

int main() {
  // init enivronment
  std::string address = "127.0.0.1";
  unsigned short port = 23333;
  srtb::config.baseband_sample_rate = 4;  // arbitrary
  srtb::config.udp_receiver_sender_address = address;
  srtb::config.udp_receiver_sender_port = port;
  // below code mixed bytes and count, so should work only if input bits == 8.
  srtb::config.baseband_input_bits = 8;
  srtb::config.log_level = static_cast<int>(srtb::log::levels::DEBUG);
  srtb::config.dm = 375;
  size_t nsamps_reserved = srtb::codd::nsamps_reserved();
  srtb::config.baseband_input_count = nsamps_reserved * 4;  // arbitrary
  size_t data_size = nsamps_reserved * 128;                 // arbitrary
  size_t n_segments = (data_size - nsamps_reserved) /
                      (srtb::config.baseband_input_count - nsamps_reserved);
  g_test_passes = false;

  SRTB_LOGI << " [test-udp_receiver] "
            << "nsamp_reserved = " << nsamps_reserved << ", "
            << "baseband_input_count = " << srtb::config.baseband_input_count
            << srtb::endl;

  // set up watchdog
  // sometimes the whole program stuck at somewhere (maybe sending/receiving UDP packet?)
  // so shut this down if test not passed in 1 minute
  std::jthread watchdog_thread{[]() {
    for (size_t i = 0; i < 60; i++) {
      if (g_test_passes) [[unlikely]] {
        break;
      } else {
        std::this_thread::sleep_for(std::chrono::seconds{1});
      }
    }
    if (!g_test_passes) {
      std::cerr << " [test-udp_receiver] "
                << " [watchdog] Time out." << std::endl;
      std::abort();
    }
  }};

  // set up receiver
  srtb::pipeline::udp_receiver_pipe udp_receiver_pipe;
  auto udp_receiver_thread = udp_receiver_pipe.start();
  auto wait_time_short = std::chrono::milliseconds(200);
  auto wait_time_long = std::chrono::milliseconds(500);
  std::this_thread::sleep_for(wait_time_long);

  // set up server
  // https://www.boost.org/doc/libs/1_79_0/doc/html/boost_asio/example/cpp11/multicast/sender.cpp
  using boost::asio::ip::udp;
  boost::asio::io_service io_service;
  udp::endpoint ep1{boost::asio::ip::address::from_string(address), port};
  udp::socket server_socket{io_service, ep1.protocol()};

  // prepare data
  std::vector<std::byte> data{data_size};
  std::generate(data.begin(), data.end(),
                []() { return static_cast<std::byte>(std::rand() & 0xFF); });

  // send data
  {
    size_t sent_data_size = 0;
    srtb::udp_packet_counter_type counter = 0;
    std::vector<std::byte> temp_buffer;
    constexpr auto counter_bytes_count =
        srtb::io::udp_receiver::counter_bytes_count;
    while (sent_data_size < data_size) {
      size_t send_data_size =
          std::min(std::rand() % (nsamps_reserved * 7) /* arbitrary */,
                   data_size - sent_data_size);
      size_t len = send_data_size + counter_bytes_count;
      temp_buffer.resize(len);
      // construct a packet with a counter
      for (size_t i = 0; i < counter_bytes_count; ++i) {
        temp_buffer.at(i) =
            std::byte((counter >> (srtb::BITS_PER_BYTE * i)) & 0xFF);
      }
      for (size_t j = 0; j < send_data_size; j++) {
        temp_buffer.at(counter_bytes_count + j) = data.at(sent_data_size + j);
      }
      server_socket.send_to(boost::asio::buffer(temp_buffer), ep1);
      SRTB_LOGD << " [test-udp_receiver] "
                << "sent data length = " << send_data_size << ", "
                << " counter = " << counter << srtb::endl;
      counter++;
      sent_data_size += send_data_size;
    }
    SRTB_CHECK_TEST_UDP_RECEIVER(sent_data_size == data_size);
  }
  SRTB_LOGI << " [test-udp_receiver] "
            << "data sent" << srtb::endl;

  // wait to be processed
  // https://stackoverflow.com/a/51850018/5269168
  auto status =
      std::async(std::launch::async, [&]() {
        while (srtb::unpack_queue.read_available() < n_segments) {
          std::this_thread::yield();
        }
        std::this_thread::sleep_for(wait_time_short);
        if (srtb::unpack_queue.read_available() > n_segments) {
          SRTB_CHECK_TEST_UDP_RECEIVER(
              false && "segment count in unpack queue isn't expected");
        }
      }).wait_for(wait_time_long);
  switch (status) {
    case std::future_status::deferred:
      SRTB_CHECK_TEST_UDP_RECEIVER(
          false && "... should never happen with std::launch::async");
      break;
    case std::future_status::ready:
      //...
      break;
    case std::future_status::timeout:
      SRTB_CHECK_TEST_UDP_RECEIVER(
          false && "segment count in unpack queue isn't expected");
      break;
  }

  // per byte check of work
  srtb::work::unpack_work unpack_work;
  size_t counter = 0, index = 0,
         length = srtb::config.baseband_input_count *
                  srtb::config.baseband_input_bits;
  std::shared_ptr<std::byte> host_mem =
      srtb::host_allocator.allocate_shared(length);
  auto start_time = std::chrono::system_clock::now();
  bool time_out = false;
  while (true) {
    // wait enough time, in case number of work in unpacker_queue isn't expected.
    while (!srtb::unpack_queue.read_available()) {
      auto now = std::chrono::system_clock::now();
      if (now - start_time > wait_time_long) {
        time_out = true;
        break;
      }
    }
    if (time_out) {
      break;
    }
    srtb::unpack_queue.pop(unpack_work);
    SRTB_CHECK_TEST_UDP_RECEIVER(length == unpack_work.count);
    srtb::queue.copy(unpack_work.ptr.get(), host_mem.get(), length).wait();
    counter++;
    for (size_t i = 0; i < length; ++i, ++index) {
      SRTB_CHECK_TEST_UDP_RECEIVER(index < data.size());
      SRTB_CHECK_TEST_UDP_RECEIVER(host_mem.get()[i] == data.at(index));
    }
    index -= nsamps_reserved;
    SRTB_LOGD << " [test-udp_receiver] "
              << "one segment checked." << srtb::endl;
  }
  SRTB_CHECK_TEST_UDP_RECEIVER(counter == n_segments);
  g_test_passes = true;

  return 0;
}
