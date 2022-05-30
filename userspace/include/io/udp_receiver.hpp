#pragma once
#ifndef __SRTB_IO_UDP_RECEIVER__
#define __SRTB_IO_UDP_RECEIVER__

#include <boost/asio.hpp>
#include <boost/bind.hpp>
#include <array>
#include <iostream>

namespace srtb {
namespace io {
namespace udp_receiver{

constexpr size_t SRTB_RECEIVER_BUFFER_SIZE = 1 << 20;

inline void receiver_worker(const std::string& sender_address, const unsigned short& sender_port) {
    using boost::asio::ip::udp;
    udp::endpoint sender_endpoint{boost::asio::ip::address::from_string(sender_address), sender_port};
    boost::asio::io_service io_service;
    udp::socket socket{io_service, sender_endpoint};

    udp::endpoint ep2;
    std::array<char, SRTB_RECEIVER_BUFFER_SIZE> receive_buffer;
    size_t len = socket.receive_from(boost::asio::buffer(receive_buffer), ep2);
    std::cout << "Received data length = " << len << std::endl;
}

} // namespace udp_receiver
} // namespace io
} // namespace srtb

#endif //  __SRTB_IO_UDP_RECEIVER__