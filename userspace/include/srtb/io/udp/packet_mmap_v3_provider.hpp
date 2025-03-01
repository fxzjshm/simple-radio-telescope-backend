/******************************************************************************* 
 * Copyright (c) 2025 fxzjshm
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
#ifndef __SRTB_IO_UDP_PACKET_MMAP_V3_PROVIDER__
#define __SRTB_IO_UDP_PACKET_MMAP_V3_PROVIDER__

#include <linux/if_packet.h>
#include <linux/ip.h>
#include <linux/udp.h>
#include <netinet/in.h>
#include <sys/mman.h>
#include <sys/poll.h>
#include <sys/user.h>

#include <atomic>
#include <cerrno>
#include <cstddef>
#include <span>
#include <string>

#include "srtb/io/udp/recvfrom_packet_provider.hpp"
#include "srtb/util/assert.hpp"

#define SRTB_SOCKET_CHECK(expr)                                                                            \
  do {                                                                                                     \
    const int ret = expr;                                                                                  \
    if (ret < 0) [[unlikely]] {                                                                            \
      throw std::runtime_error{                                                                            \
          (#expr " returned " + std::to_string(ret) + ", " + "errno = " + std::to_string(errno)).c_str()}; \
    }                                                                                                      \
  } while (0);

namespace srtb {
namespace io {
namespace udp {

// TODO: use values from config
inline unsigned int tp_block_size = 1u << 31;
inline unsigned int tp_block_nr = 8;
inline unsigned int tp_frame_size = 8192;

/**
 * @brief Receive UDP packet using packet mmap v3
 *
 * ref: https://www.kernel.org/doc/html/latest/networking/packet_mmap.html
 *      https://github.com/david-macmahon/hashpipe
 *      https://github.com/SparkePei/UWB_HASHPIPE
 *      https://csulrong.github.io/blogs/2022/03/10/linux-afpacket/
 *      https://gist.github.com/adubovikov/e4eb7898ab411d7b7d531d9796d2c51d
 *
 * @deprecated TODO: Not correctly implemented: should work on PF_PACKET level, but if it can only operate on one interface, why don't I use DPDK / XDP ?
 */
class [[deprecated(
    "TODO: Not correctly implemented: should work on PF_PACKET level, but if it can only operate on one interface, why "
    "don't I use DPDK / XDP ?")]] packet_mmap_v3_provider : recvfrom_packet_provider {
 protected:
  tpacket_req3 req;
  std::byte* ring;
  pollfd pfd;

  /** These variables may change during receive(), i.e. interval variables of a state machine */
  struct mutable_var_t {
    /** block index that currently at; n_block == tp_block_nr, block_size == tp_block_size */
    unsigned int i_block;
    /** packet mmap v3 blcok descriptor */
    tpacket_block_desc* desc;
    /** index of packet in i_block */
    uint32_t i_packet;
    /** count of packet in i_block */
    uint32_t n_packet;
    /** point to header of this packet */
    tpacket3_hdr* packet_ptr;
  } mutable_var;

 public:
  packet_mmap_v3_provider(std::string address, unsigned short port)
      : recvfrom_packet_provider{address, port}  // <- TODO: this is wrong, should work on PF_PACKET level, thus provide interface name
  {
    int fd = sock.sock;

    // set to v3
    int v = TPACKET_V3;
    SRTB_SOCKET_CHECK(setsockopt(fd, SOL_PACKET, PACKET_VERSION, &v, sizeof(v)));

    req.tp_block_size = tp_block_size;
    req.tp_block_nr = tp_block_nr;
    req.tp_frame_size = tp_frame_size;
    req.tp_frame_nr = req.tp_block_size / req.tp_frame_size * req.tp_block_nr;
    setsockopt(fd, SOL_PACKET, PACKET_RX_RING, &req, sizeof(req));

    // mmap ring
    unsigned int total_size = req.tp_block_size * req.tp_block_nr;
    ring = reinterpret_cast<std::byte*>(
        mmap(NULL, total_size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_LOCKED | MAP_HUGETLB, fd, 0));
    if (ring == reinterpret_cast<void*>(-1)) [[unlikely]] {
      throw std::runtime_error{"packet ring mmap failed with errno: " + std::to_string(errno)};
    }

    // setup polling
    pfd.fd = fd;
    pfd.events = POLLIN | POLLERR;
    pfd.revents = 0;

    // clean interval states
    mutable_var.i_block = req.tp_block_nr - 1;
    mutable_var.desc = nullptr;
    mutable_var.i_packet = 0;
    mutable_var.n_packet = 0;
    mutable_var.packet_ptr = nullptr;
  }

  /**
   * @brief Receive packet into given position
   * @param h_out (mutable) packet will be written to
   */
  auto receive() -> std::span<std::byte> {
    if (mutable_var.i_packet == mutable_var.n_packet) {
      get_next_block();
    }
    // mutable_var should refreshed

    // wait for packet
    // TODO: this should not happen, check it
    while (!(mutable_var.packet_ptr->tp_status & TP_STATUS_USER)) [[unlikely]] {
      // just wait
    }
    // packet ready, parse header
    // ref: https://github.com/david-macmahon/hashpipe/blob/44b432af3c88ff6ccd224f526ac2b06674af04a8/src/hashpipe_pktsock.h#L33
    const auto p = mutable_var.packet_ptr;
    iphdr* const ip_hdr = reinterpret_cast<iphdr*>(p + p->tp_net);
    // TODO: assuming a UDP packet
    BOOST_ASSERT(ip_hdr->protocol == IPPROTO_UDP);
    udphdr* const udp_hdr = reinterpret_cast<udphdr*>(ip_hdr + 1);
    std::byte* const udp_data = reinterpret_cast<std::byte*>(udp_hdr + 1);

    // update state of next packet
    mutable_var.packet_ptr += mutable_var.packet_ptr->tp_next_offset;
    mutable_var.i_packet += 1;
#if __has_builtin(__builtin_prefetch)
    __builtin_prefetch(mutable_var.packet_ptr, /* rw = read */ 0, /* locality = no */ 0);
#endif

    return std::span{udp_data, udp_hdr->len};
  }

  void get_next_block() {
    // close this block, if applicable
    if (mutable_var.desc) [[likely]] {
      mutable_var.desc->hdr.bh1.block_status = TP_STATUS_KERNEL;
      // __sync_synchronize()
      std::atomic_thread_fence(std::memory_order_seq_cst);

      // get location of next block
      mutable_var.i_block = (mutable_var.i_block + 1) % req.tp_block_nr;
    } else {
      // at init
      mutable_var.i_block = 0;
    }

    mutable_var.desc = reinterpret_cast<tpacket_block_desc*>(ring + mutable_var.i_block * req.tp_block_size);

    // wait for data
    while (!(mutable_var.desc->hdr.bh1.block_status & TP_STATUS_USER)) {
      poll(&pfd, /* nfds = */ 1, /* timeout = */ -1);
    }

    // read info from block
    mutable_var.i_packet = 0;
    mutable_var.n_packet = mutable_var.desc->hdr.bh1.num_pkts;
    mutable_var.packet_ptr =
        reinterpret_cast<tpacket3_hdr*>(mutable_var.desc + mutable_var.desc->hdr.bh1.offset_to_first_pkt);
  }
};

}  // namespace udp
}  // namespace io
}  // namespace srtb

#endif  //  __SRTB_IO_UDP_PACKET_MMAP_V3_PROVIDER__
