#include <fftw3.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>

#include "srtb/commons.hpp"
#include "srtb/fft/naive_fft.hpp"

// Adapted from https://www.cnblogs.com/CaCO3/p/15996732.html by CaO
// original post is licensed under CC BY-NC-SA
#define reg
using Complex = srtb::complex<double>;
// naive_fft: f -> g, fftw: f -> h, sequence fft: f -> f
std::vector<Complex> f, g, h;
const auto PI = M_PI;
int siz = 1;
int bit;

std::vector<int> rev;
void fft(Complex* arr,
         int tp) {  //tp 代表变换的类型，如果 tp = 1 代表 FFT，tp = -1 代表 IFFT
  for (reg int i(0); i < siz; ++i)
    rev[i] =
        (rev[i >> 1] >> 1) |
        ((i & 1) << (bit -
                     1));  //预处理二进制反转，思考一下为什么这样能够行得通 QwQ

  for (reg int i(0); i < siz; ++i) {
    if (i < rev[i]) std::swap(arr[i], arr[rev[i]]);  //蝴蝶变换
  }

  for (reg int mid(1); mid < siz; mid <<= 1) {  //枚举每次合并区间的半长度
    Complex Wn = Complex(cos(PI / mid), -tp * sin(PI / mid));

    for (reg int i(0); i < siz; i += (mid << 1)) {  //枚举每个区间的左端点
      Complex w = Complex(1, 0);
      for (reg int j(0); j < mid; ++j) {  //FFT 代值
        Complex A1 = arr[i + j], A2 = w * arr[i + j + mid];
        arr[i + j] = A1 + A2, arr[i + j + mid] = A1 - A2;
        w = w * Wn;
      }
    }
  }
}

int main(int argc, char** argv) {
  sycl::queue q;

  const int manual_thereshold = 10;

  bit = 20;
  if (argc > 1) {
    try {
      bit = std::stoi(argv[1]);
    } catch (const std::invalid_argument& ignored) {
      // bit should remain unchanged
    }
  }
  siz = 1 << bit;
  f.resize(siz + 5);
  g.resize(siz + 5);
  h.resize(siz + 5);
  rev.resize(siz + 5);
  rev[0] = 0;

  {
    int ret = fftw_init_threads();
    if (ret == 0) [[unlikely]] {
      throw std::runtime_error("[fft] init fftw failed!");
    }
    int n_threads = std::max(std::thread::hardware_concurrency(), 1u);
    fftw_plan_with_nthreads(n_threads);
  }

  auto fftw_plan_start = std::chrono::system_clock::now();
  fftw_plan p = fftw_plan_dft_1d(siz, reinterpret_cast<fftw_complex*>(&f[0]),
                                 reinterpret_cast<fftw_complex*>(&h[0]),
                                 FFTW_FORWARD, FFTW_ESTIMATE);
  auto fftw_plan_end = std::chrono::system_clock::now();

  if (siz < manual_thereshold) {
    for (int i = 0; i < siz; i++) {
      int x;
      std::cin >> x;
      f[i].real(x);
    }
  } else {
    double x, y;
    for (int i = 0; i < siz; i++) {
      x = double(std::rand()) / (0xcafebabe);
      y = double(std::rand()) / (0xbeefcace);
      f[i].real(x);
      f[i].imag(y);
    }
  }

  auto naive_fft_start = std::chrono::system_clock::now(),
       naive_fft_end = std::chrono::system_clock::now();
  {
    auto h_f = &f[0], h_g = &g[0];
    auto d_f_shared =
             srtb::device_allocator.allocate_shared<std::remove_const_t<
                 std::remove_reference_t<decltype(f.front())> > >(f.size()),
         d_g_shared =
             srtb::device_allocator.allocate_shared<std::remove_const_t<
                 std::remove_reference_t<decltype(g.front())> > >(g.size());
    auto d_f = d_f_shared.get(), d_g = d_g_shared.get();
    q.copy(h_f, d_f, f.size()).wait();
    naive_fft_start = std::chrono::system_clock::now();
    naive_fft::fft_1d_c2c<double>(bit, q, d_f, d_g, 1);
    naive_fft_end = std::chrono::system_clock::now();
    q.copy(d_g, h_g, g.size()).wait();
  }

  auto fftw_execute_start = std::chrono::system_clock::now();
  fftw_execute(p);
  auto fftw_execute_end = std::chrono::system_clock::now();

  auto sequencial_fft_start = std::chrono::system_clock::now();
  fft(&f[0], 1);
  auto sequencial_fft_end = std::chrono::system_clock::now();

  auto naive_fft_time = naive_fft_end - naive_fft_start,
       sequencial_fft_time = sequencial_fft_end - sequencial_fft_start,
       fftw_plan_time = fftw_plan_end - fftw_plan_start,
       fftw_execute_time = fftw_execute_end - fftw_execute_start;
  std::cout << "naive_fft " << naive_fft_time.count() << "ns, "
            << "sequence " << sequencial_fft_time.count() << "ns, "
            << "fftw plan " << fftw_plan_time.count() << "ns, "
            << "fftw execute " << fftw_execute_time.count() << "ns. "
            << std::endl;

  if (siz < manual_thereshold) {
    for (int i = 0; i < siz; i++) {
      std::cout << f[i] << ' ';
    }
    std::cout << std::endl;
    for (int i = 0; i < siz; i++) {
      std::cout << g[i] << ' ';
    }
    std::cout << std::endl;
    for (int i = 0; i < siz; i++) {
      std::cout << h[i] << ' ';
    }
    std::cout << std::endl;
  }
  for (int i = 0; i < siz; i++) {
    auto df = (f[i] - g[i]) / f[i], dh = (h[i] - g[i]) / h[i];
    if (srtb::abs(df) > 1e-5 || srtb::abs(dh) > 1e-5) {
      std::cerr << "Difference at i = " << i << ", "
                << "f[i] = " << f[i] << ", "
                << "g[i] = " << g[i] << ", "
                << "h[i] = " << h[i] << ". " << std::endl;
      exit(-1);
    }
  }
  std::cout << "naive/sequencial = "
            << double(naive_fft_time.count()) /
                   double(sequencial_fft_time.count())
            << std::endl;

  fftw_cleanup_threads();

  return 0;
}
