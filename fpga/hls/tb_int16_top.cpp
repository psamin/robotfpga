// The int16 HLS top itself (float_top.cpp with -DFSTREAM_ARITH=16, ap_int / hls::stream types, as synthesized)
// compiled with g++ against the Vitis HLS headers: what C simulation would run, without the Vitis csim flow
// (broken on this install). One float_top() call on a case's DDR images (pack_float.py --tests-int16: packed.bin,
// xmem.bin) must give the bits of the host-typed C model (TB_DUMP_Y output of tb_float int16 mode, untiled) and
// pass the testbench tolerance against y_ref.
//   g++ -std=c++14 -O2 -D__VITIS_HLS__ -DFSTREAM_ARITH=16 -ffp-contract=off -I<Vitis_HLS>/include
//       -Wl,--stack,268435456 -o tb_int16_top tb_int16_top.cpp float_top.cpp
//   tb_int16_top CASE_DIR [Y_HOST.bin]
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "float_linear.h"

using fstream::IoWord;
using fstream::WPort;

int float_top(const WPort* w0, const WPort* w1, const WPort* w2, const WPort* w3, const IoWord* xmem, IoWord* ymem,
              int M, int K, int N);

template <class T>
static std::vector<T> slurp(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    fprintf(stderr, "cannot open %s\n", path.c_str());
    exit(2);
  }
  std::vector<char> raw((std::istreambuf_iterator<char>(f)), {});
  std::vector<T> out(raw.size() / sizeof(T));
  memcpy(out.data(), raw.data(), out.size() * sizeof(T));
  return out;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s CASE_DIR [Y_HOST.bin]\n", argv[0]);
    return 2;
  }
  const std::string d = argv[1];
  int Mp, K, N, M;
  char kind[16] = {0};
  FILE* f = fopen((d + "/dims.txt").c_str(), "r");
  if (!f || fscanf(f, "%d %d %d %d %15s", &Mp, &K, &N, &M, kind) != 5 || strcmp(kind, "int16") || K > 4096) {
    fprintf(stderr, "%s: not a single-chunk int16 case\n", d.c_str());
    return 2;
  }
  fclose(f);
  const int np = (N + 31) / 32 * 32, groups = Mp / 32;
  auto wraw = slurp<uint8_t>(d + "/packed.bin");
  auto xraw = slurp<uint8_t>(d + "/xmem.bin");
  // the HLS word types are ap_uint<128>: copy the DDR images into arrays of them (little-endian lanes)
  std::vector<WPort> w(wraw.size() / 16);
  for (size_t i = 0; i < w.size(); i++)
    for (int b = 0; b < 16; b++) w[i].range(8 * b + 7, 8 * b) = wraw[16 * i + b];
  std::vector<IoWord> x(xraw.size() / 16);
  for (size_t i = 0; i < x.size(); i++)
    for (int b = 0; b < 16; b++) x[i].range(8 * b + 7, 8 * b) = xraw[16 * i + b];
  std::vector<IoWord> y((size_t)groups * np * 32 / 4);
  const size_t pw = (size_t)groups * K;
  const int st = float_top(w.data(), w.data() + pw, w.data() + 2 * pw, w.data() + 3 * pw, x.data(), y.data(), Mp, K, N);
  std::vector<float> yt(y.size() * 4);  // Y[g][np][32]
  for (size_t i = 0; i < y.size(); i++)
    for (int j = 0; j < 4; j++) {
      const uint32_t u = (uint32_t)y[i].range(32 * j + 31, 32 * j);
      memcpy(&yt[4 * i + j], &u, 4);
    }
  auto ref = slurp<double>(d + "/y_ref.bin");
  auto sa = slurp<double>(d + "/sumabs.bin");
  double worst = 0;
  for (int n = 0; n < N; n++)
    for (int m = 0; m < M; m++) {
      const size_t i = (size_t)n * Mp + m;
      const double e = fabs((double)yt[((size_t)(m / 32) * np + n) * 32 + m % 32] - ref[i]) / (sa[i] + 1e-30);
      if (e > worst) worst = e;
    }
  bool ok = st == 0 && worst <= 1e-6;
  std::string host = "not compared";
  if (argc > 2) {
    auto yh = slurp<float>(argv[2]);  // tb_float int16 mode: y[N][Mp]
    size_t diff = 0;
    for (int n = 0; n < N; n++)
      for (int m = 0; m < Mp; m++) {
        const float a = yt[((size_t)(m / 32) * np + n) * 32 + m % 32], b = yh[(size_t)n * Mp + m];
        if (memcmp(&a, &b, 4)) diff++;
      }
    host = diff ? "MISMATCH (" + std::to_string(diff) + " values)" : "bit-identical";
    ok = ok && diff == 0;
  }
  printf("%s: HLS-typed float_top M=%d K=%d N=%d: status %d, max err/sum|xw| %.3g, vs host-typed C model %s\n",
         ok ? "PASS" : "FAIL", Mp, K, N, st, worst, host.c_str());
  return ok ? 0 : 1;
}
