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
size_t cpu_preferred = 0, cpu_preferred_sq_poll = 0;
constexpr size_t max_cnt = 1 << 18;
unsigned int uring_entries_len = 4096;

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

  if (argc >= 5) {
    try {
      cpu_preferred_sq_poll = std::stoi(argv[4]);
    } catch (...) {
    }
  }

    if (argc >= 6) {
    try {
      uring_entries_len = std::stoi(argv[5]);
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
    if (bind_ret < 0) [[unlikely]] {
      throw std::runtime_error{msg + " failed: " + std::to_string(bind_ret)};
    }
  }

  // https://wjwh.eu/posts/2021-10-01-no-syscall-server-iouring.html
  io_uring ring;
  io_uring_params params = {};
#if 1  // sq poll switch
  params.flags |= IORING_SETUP_SQPOLL;
  // params.flags |= IORING_SETUP_SQ_AFF;
  // params.flags |= IORING_SETUP_SINGLE_ISSUER;
  params.sq_thread_cpu = cpu_preferred_sq_poll;
  params.sq_thread_idle = 120000;  // 2 minutes in ms
#endif
  int init_ret = io_uring_queue_init_params(uring_entries_len, &ring, &params);
  if (init_ret) [[unlikely]] {
    throw std::runtime_error{"io_uring_queue_init_params: " +
                             std::to_string(init_ret)};
  }

  std::array<std::byte, 65536> udp_buffer;
  constexpr uint64_t unset_counter = -1;
  uint64_t last_counter = unset_counter;
  uint64_t lost_packet = 0, total_packet = 0;
  for (size_t i = 0; i<=(10*max_cnt); i++) {
    // https://zhuanlan.zhihu.com/p/639112789?utm_id=0
    // https://github.com/axboe/liburing/issues/521#issue-1134862376
    io_uring_sqe* sqe = io_uring_get_sqe(&ring);
    io_uring_prep_recv(sqe, sock.sock, udp_buffer.begin(), udp_buffer.size(),
                       /* flag = */ 0);
    int submit_ret = io_uring_submit(&ring);
    if (submit_ret < 0) [[unlikely]] {
      throw std::runtime_error{"io_uring_submit: " + std::to_string(submit_ret)};
    }

    io_uring_cqe* cqe;
    int cqe_ret;
#if 1  // CQE poll / wait switch
    do {
      cqe_ret = io_uring_peek_cqe(&ring, &cqe);
    } while (cqe_ret);
#else
    cqe_ret = io_uring_wait_cqe(&ring, &cqe);
    if (cqe_ret < 0) {
      throw std::runtime_error{"io_uring_wait_cqe: " + std::to_string(cqe_ret)};
    }
#endif
    const ssize_t packet_size = cqe->res;
    if (packet_size <= 0) [[unlikely]] {
      throw std::runtime_error{"io_uring recv returned " +
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

    io_uring_cqe_seen(&ring, cqe);

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
