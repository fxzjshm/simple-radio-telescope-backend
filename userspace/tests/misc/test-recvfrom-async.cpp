#include <arpa/inet.h>
#include <liburing.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <array>
#include <cstring>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <thread>

#include "srtb/io/udp/packet_parser.hpp"
#include "srtb/memory/dual_port_object_pool.hpp"
#include "srtb/termination_handler.hpp"
#include "srtb/thread_affinity.hpp"

uint16_t MYPORT = 60001;
std::string SERVERIP = "10.17.16.12";
size_t cpu_preferred = 126;
constexpr size_t max_cnt = 1 << 18;
unsigned int uring_entries_len = 1 << 17;

constexpr size_t UDP_MAX_SIZE = 65536;
std::array<std::byte, UDP_MAX_SIZE> out_buffer;

using packet_container_t = std::array<std::byte, UDP_MAX_SIZE>;
struct alignas(64) packet_t {
  packet_container_t packet_countainer;
  size_t bytes_received;
};

srtb::memory::dual_port_object_pool<packet_t> dp_pool =
    srtb::memory::dual_port_object_pool<packet_t>{size_t{uring_entries_len}};

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

  std::jthread consumer_jthread = std::jthread{[](std::stop_token stop_token) {
    while (!stop_token.stop_requested()) [[likely]] {
      packet_t* pkt = dp_pool.pop_received();
      std::copy(pkt->packet_countainer.begin(),
                pkt->packet_countainer.begin() + pkt->bytes_received,
                out_buffer.begin());
      dp_pool.put_free(pkt);
    }
  }};

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

  constexpr uint64_t unset_counter = -1;
  uint64_t last_counter = unset_counter;
  uint64_t lost_packet = 0, total_packet = 0;
  for (size_t i = 0; true; i++) {
    packet_t* pkt = dp_pool.get_or_allocate_free();
#if 0  // recvfrom / recv
    sockaddr_in peer_addr;
    socklen_t peer_len;
    const ssize_t packet_size = recvfrom(
        sock.sock, udp_buffer.begin(), udp_buffer.size(), /* flag = */ 0,
        reinterpret_cast<sockaddr*>(&peer_addr), &peer_len);
#else
    const ssize_t packet_size =
        recv(sock.sock, pkt->packet_countainer.begin(),
             pkt->packet_countainer.size(), /* flag = */ 0);
#endif
    if (packet_size <= 0) {
      throw std::runtime_error{"recv(from) returned " +
                               std::to_string(packet_size)};
    }
    pkt->bytes_received = packet_size;

    auto [packet_header_size, counter, timestamp] =
        srtb::io::udp::gznupsr_a1_packet_parser::parse(
            std::span{pkt->packet_countainer.begin(), pkt->bytes_received});
    if (last_counter != unset_counter) [[likely]] {
      lost_packet += counter - last_counter - 1;
      total_packet += counter - last_counter;
    } else {
      total_packet += 1;
    }
    last_counter = counter;

    dp_pool.push_received(pkt);

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
