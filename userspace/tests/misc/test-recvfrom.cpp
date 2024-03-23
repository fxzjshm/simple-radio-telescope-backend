#include <arpa/inet.h>
#include <liburing.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <array>
#include <cstring>
#include <stdexcept>
#include <string>

#include "srtb/io/udp/packet_parser.hpp"
#include "srtb/termination_handler.hpp"
#include "srtb/thread_affinity.hpp"

uint16_t MYPORT = 60002;
std::string SERVERIP = "10.17.16.12";
size_t cpu_preferred = 0;
constexpr size_t max_cnt = 1 << 19;

std::array<std::byte, 65536> out_buffer;

struct udpdk_udp_socket {
  int sock;

  explicit udpdk_udp_socket() {
    sock = socket(PF_INET, SOCK_DGRAM, 0);
    if (sock < 0) {
      throw std::runtime_error{"udp socket creation error"};
    }
  }

  ~udpdk_udp_socket() { close(sock); }
};

int main(int argc, char** argv) {
  udpdk_udp_socket sock;

  if (argc >= 2) {
    try {
      SERVERIP = argv[1];
    } catch (...) {
    }
  }

  if (argc >= 3) {
    try {
      MYPORT = std::stoi(argv[2]);
    } catch (...) {
    }
  }

  if (argc >= 4) {
    try {
      cpu_preferred = std::stoll(argv[3]);
    } catch (...) {
    }
  }

  srtb::thread_affinity::set_thread_affinity(cpu_preferred);

  {
    std::string msg =
        std::string{"Bind to "} + SERVERIP + ":" + std::to_string(int{MYPORT});
    SRTB_LOGI << msg << srtb::endl;
    sockaddr_in servaddr = {};
    servaddr.sin_family = AF_INET;
    servaddr.sin_port = htons(MYPORT);
    servaddr.sin_addr.s_addr = inet_addr(SERVERIP.c_str());
    const int bind_ret = bind(sock.sock, reinterpret_cast<sockaddr*>(&servaddr),
                              sizeof(servaddr));
    if (bind_ret < 0) {
      throw std::runtime_error{msg + " failed: " + std::to_string(bind_ret)};
    }
  }

  std::array<std::byte, 65536> udp_buffer;
  constexpr uint64_t unset_counter = -1;
  uint64_t last_counter = unset_counter;
  uint64_t lost_packet = 0, total_packet = 0;
  for (size_t i = 0; true; i++) {
#if 0  // recvfrom / recv
    sockaddr_in peer_addr;
    socklen_t peer_len;
    const ssize_t packet_size = recvfrom(
        sock.sock, udp_buffer.begin(), udp_buffer.size(), /* flag = */ 0,
        reinterpret_cast<sockaddr*>(&peer_addr), &peer_len);
#else
    const ssize_t packet_size =
        recv(sock.sock, udp_buffer.begin(), udp_buffer.size(), /* flag = */ 0);
#endif
    if (packet_size <= 0) {
      throw std::runtime_error{"recv(from) returned " +
                               std::to_string(packet_size)};
    }

    auto [packet_header_size, counter, timestamp] =
        srtb::io::udp::gznupsr_a1_packet_parser::parse(
            std::span{udp_buffer.begin(), static_cast<size_t>(packet_size)});
    if (last_counter != unset_counter) [[likely]] {
      lost_packet += counter - last_counter - 1;
      total_packet += counter - last_counter;
    } else {
      total_packet += 1;
    }
    last_counter = counter;

    std::copy(udp_buffer.begin(), udp_buffer.begin() + packet_size,
              out_buffer.begin());

    if (i % max_cnt == 0) {
      SRTB_LOGI << "lost_packet = " << lost_packet << ", "
                << "total_packet = " << total_packet << ", "
                << "rate = "
                << static_cast<double>(lost_packet) /
                       static_cast<double>(total_packet)
                << ", "
                << "last_counter = " << last_counter << srtb::endl;
    }
  }

  return 0;
}
