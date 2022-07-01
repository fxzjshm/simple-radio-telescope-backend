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

#include <boost/asio.hpp>
#include <chrono>
#include <future>

#include "srtb/global_variables.hpp"
#include "srtb/io/udp_receiver.hpp"

int main() {
  // init enivronment
  std::string address = "127.0.0.1";
  unsigned short port = 23333;
  srtb::config.baseband_sample_rate = 4;
  srtb::config.udp_receiver_sender_address = address;
  srtb::config.udp_receiver_sender_port = port;
  srtb::config.log_level = static_cast<int>(srtb::log::levels::DEBUG);
  size_t nsamps_reserved = srtb::codd::nsamps_reserved();
  srtb::config.baseband_input_length = nsamps_reserved * 4;
  size_t data_size = nsamps_reserved * 128;
  size_t n_segments = (data_size - nsamps_reserved) /
                      (srtb::config.baseband_input_length - nsamps_reserved);

  SRTB_LOGI << " [test-udp_receiver] "
            << "nsamp_reserved = " << nsamps_reserved << ", "
            << "baseband_input_length = " << srtb::config.baseband_input_length
            << std::endl;

  // set up receiver
  auto udp_receiver_thread = srtb::io::udp_receiver::run_udp_receiver_worker();
  std::this_thread::sleep_for(std::chrono::seconds(1));

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
  server_socket.send_to(boost::asio::buffer(data), ep1);
  SRTB_LOGI << " [test-udp_receiver] "
            << "data sent" << std::endl;

  // wait to be processed
  // https://stackoverflow.com/a/51850018/5269168
  int time_out = 10;
  auto status = std::async(std::launch::async, [&]() {
                  auto start = std::chrono::system_clock::now();
                  while (srtb::unpacker_queue.read_available() < n_segments) {
                    std::this_thread::yield();
                  }
                  if (srtb::unpacker_queue.read_available() > n_segments) {
                    throw std::runtime_error(
                        "segment count in unpack queue isn't expected");
                    std::exit(-1);
                  }
                }).wait_for(std::chrono::seconds{time_out});
  switch (status) {
    case std::future_status::deferred:
      //... should never happen with std::launch::async
      break;
    case std::future_status::ready:
      //...
      break;
    case std::future_status::timeout:
      throw std::runtime_error("segment count in unpack queue isn't expected");
      break;
  }

  // per byte check of work
  srtb::unpacker_work_type unpacker_work;
  size_t counter = 0, index = 0, length = srtb::config.baseband_input_length;
  std::shared_ptr<std::byte> host_mem =
      srtb::host_allocator.allocate_smart(length);
  while (srtb::unpacker_queue.read_available()) {
    bool ret;
    do {
      ret = srtb::unpacker_queue.pop(unpacker_work);
    } while (ret == false);
    srtb::queue.copy(unpacker_work.ptr.get(), host_mem.get(), length).wait();
    for (size_t i = 0; i < length; ++i, ++index) {
      assert(index < data.size());
      assert(host_mem.get()[i] == data[index]);
    }
    index -= nsamps_reserved;
  }

  std::exit(0);
  return 0;
}
