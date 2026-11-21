// Shared-library wrapper around the HLS float linear core, so a whole PyTorch model can run with
// every linear layer computed by the exact engine code (policy/smolvla_fpga.py).
//   g++ -O3 -fopenmp -shared -fPIC -o libfloat_linear.so float_shim.cpp
// Work is split into (token tile, block of output groups) tasks across OpenMP threads, each calling
// the unmodified core on a slice of the weight arrays with its own buffers.
//
// Packed weight layout (pack_float.py): for each K chunk c (K_MAX wide), for each port p,
// groups g, k in chunk -> one WordBF. M is already padded to a multiple of 32 by the packer.
// Chunk outputs are summed in chunk order, as fpga/board/runtime.py does on the ARM.
#include <stdlib.h>
#include <string.h>

#include <vector>

#include "float_linear.h"

using namespace fstream;

static const int GBLOCK = 4;  // output groups (of 32 channels) per task

extern "C" int linear_bf16(const uint16_t* w_packed, const float* x, float* y, int M, int K, int N) {
  if (M % LANES || K % F4) return 1;
  const int chunks = (K + K_MAX - 1) / K_MAX;
  const int groups = M / LANES;
  memset(y, 0, sizeof(float) * (size_t)N * M);
  int status = 0;
  const WPort* base = (const WPort*)w_packed;
  size_t off = 0;
  for (int c = 0; c < chunks; c++) {
    const int k0 = c * K_MAX, kc = (K - k0) < K_MAX ? (K - k0) : K_MAX;
    const size_t port_words = (size_t)groups * kc;
    const WPort* wp[PORTS];
    for (int p = 0; p < PORTS; p++) wp[p] = base + off + p * port_words;
    off += PORTS * port_words;
    std::vector<float> xc((size_t)N * kc);
    for (int n = 0; n < N; n++) memcpy(&xc[(size_t)n * kc], x + (size_t)n * K + k0, sizeof(float) * kc);
    const int tiles = (N + NT - 1) / NT, gblocks = (groups + GBLOCK - 1) / GBLOCK;
#pragma omp parallel
    {
      float(*xbuf)[NB][K_MAX] = (float(*)[NB][K_MAX])malloc(sizeof(float) * TP * NB * K_MAX);
      std::vector<float> yt((size_t)NT * GBLOCK * LANES);
#pragma omp for schedule(dynamic) collapse(2)
      for (int t = 0; t < tiles; t++) {
        for (int b = 0; b < gblocks; b++) {
          const int n0 = t * NT, nt = (N - n0) < NT ? (N - n0) : NT;
          const int g0 = b * GBLOCK, gn = (groups - g0) < GBLOCK ? (groups - g0) : GBLOCK;
          const size_t woff = (size_t)g0 * kc;
          const int ms = gn * LANES;
          int st = linear_bf16_core(wp[0] + woff, wp[1] + woff, wp[2] + woff, wp[3] + woff,
                                    (const IoWord*)&xc[(size_t)n0 * kc], (IoWord*)yt.data(), ms, kc, nt, xbuf);
          if (st) status = st;
          // yt is Y_tiled[gn][NT][LANES] (one tile): un-tile into the row-major output
          for (int g = 0; g < gn; g++)
            for (int n = 0; n < nt; n++)
              for (int l = 0; l < LANES; l++)
                y[(size_t)(n0 + n) * M + (g0 + g) * LANES + l] += yt[((size_t)g * NT + n) * LANES + l];
        }
      }
      free(xbuf);
    }
  }
  return status;
}

#if defined(FSTREAM_ARITH) && FSTREAM_ARITH == 16
// int16 engine (int16_linear.h), built as build/libfloat_linear_int16.so with -DFSTREAM_ARITH=16.
#include "int16_linear.h"

// The host side of one engine call, in C++: the xmem image of the DRIVER CONTRACT for K chunk [k0, k0 + kc).
static std::vector<IoWord> xmem_image(const float* s_row, const int16_t* xq, const float* s_tok, int M, int K, int N,
                                      int k0, int kc) {
  const int np = (N + fstream16::TB - 1) / fstream16::TB * fstream16::TB;
  const size_t words = (size_t)M / F4 + (size_t)np / F4 + (size_t)np / 8 * kc;
  std::vector<IoWord> img(words);
  float* f = (float*)img.data();
  memcpy(f, s_row, sizeof(float) * M);
  for (int n = 0; n < np; n++) f[M + n] = n < N ? s_tok[n] : 1.0f;
  int16_t* xi = (int16_t*)(f + M + np);  // [block][k][32 tokens]
  for (int b = 0; b < np / fstream16::TB; b++)
    for (int k = 0; k < kc; k++)
      for (int a = 0; a < fstream16::TB; a++) {
        const int n = b * fstream16::TB + a;
        xi[((size_t)b * kc + k) * fstream16::TB + a] = n < N ? xq[(size_t)n * K + k0 + k] : 0;
      }
  return img;
}

// y[N][M] = sum over K chunks of float(Xq Wq^T) * s_row * s_tok. w_packed: pack_float.pack(Wq bits) (M padded
// to 32, s_row padded with 1.0); xq [N][K] int16; s_tok [N] (scales over the whole row of K).
// Tasks = blocks of output groups across OpenMP threads; each runs the unmodified core on all tokens (every
// tile, so the double-buffered X path is exercised) with xmem shifted so its s_row slice starts at word 0.
extern "C" int linear_int16(const uint16_t* w_packed, const float* s_row, const int16_t* xq, const float* s_tok,
                            float* y, int M, int K, int N) {
  if (M % LANES || K < 1 || N < 1) return 1;
  const int chunks = (K + K_MAX - 1) / K_MAX;
  const int groups = M / LANES;
  const int np = (N + fstream16::TB - 1) / fstream16::TB * fstream16::TB;
  memset(y, 0, sizeof(float) * (size_t)N * M);
  int status = 0;
  const WPort* base = (const WPort*)w_packed;
  size_t off = 0;
  for (int c = 0; c < chunks; c++) {
    const int k0 = c * K_MAX, kc = (K - k0) < K_MAX ? (K - k0) : K_MAX;
    const size_t port_words = (size_t)groups * kc;
    const WPort* wp[PORTS];
    for (int p = 0; p < PORTS; p++) wp[p] = base + off + p * port_words;
    off += PORTS * port_words;
    const std::vector<IoWord> img = xmem_image(s_row, xq, s_tok, M, K, N, k0, kc);
    const int gblocks = (groups + GBLOCK - 1) / GBLOCK;
    const int off_stok = M / F4, off_x = (M + np) / F4;
#pragma omp parallel for schedule(dynamic)
    for (int bk = 0; bk < gblocks; bk++) {
      const int g0 = bk * GBLOCK, gn = (groups - g0) < GBLOCK ? (groups - g0) : GBLOCK;
      const size_t woff = (size_t)g0 * kc;
      std::vector<float> yt((size_t)gn * np * LANES);
      const int sh = g0 * LANES / F4;  // xmem shifted so that s_row[32 g0] is word 0
      int st = fstream16::linear_int16_core(wp[0] + woff, wp[1] + woff, wp[2] + woff, wp[3] + woff,
                                            img.data() + sh, (IoWord*)yt.data(), gn * LANES, kc, N,
                                            off_stok - sh, off_x - sh);
      if (st) {
#pragma omp critical
        status = st;
      }
      for (int g = 0; g < gn; g++)  // yt is Y[gn][np][32]: un-tile, add this chunk in chunk order
        for (int n = 0; n < N; n++)
          for (int l = 0; l < LANES; l++)
            y[(size_t)n * M + (g0 + g) * LANES + l] += yt[((size_t)g * np + n) * LANES + l];
    }
  }
  return status;
}

// C model counters of the last single-threaded call (linear_int16_raw): compute16 main-loop clocks, clocks with a
// MAC step, and the idle clocks waiting for X, weights or the shadow. v[5]; reset != 0 clears them afterwards.
extern "C" void linear_int16_sim_stats(long* v, int reset) {
  v[0] = fstream16::sim_stats.clocks;
  v[1] = fstream16::sim_stats.macs;
  v[2] = fstream16::sim_stats.wait_x;
  v[3] = fstream16::sim_stats.wait_w;
  v[4] = fstream16::sim_stats.wait_cap;
  if (reset) fstream16::sim_stats = fstream16::SimStats();
}

// Exactly one engine call on host buffers in the DRIVER CONTRACT layout (K <= 4096): w_packed is pack()'s output
// for this K (the four port arrays back to back), xmem the image of pack_float.pack_xmem_int16, ytiled receives
// Y[M/32][Np][32] FP32. Returns the engine's status (1 = rejected).
extern "C" int linear_int16_raw(const uint16_t* w_packed, const void* xmem, void* ytiled, int M, int K, int N) {
  if (M % LANES || K < 1 || K > K_MAX || N < 1) return 1;
  const int groups = M / LANES;
  const int np = (N + fstream16::TB - 1) / fstream16::TB * fstream16::TB;
  const WPort* w = (const WPort*)w_packed;
  const size_t pw = (size_t)groups * K;
  return fstream16::linear_int16_core(w, w + pw, w + 2 * pw, w + 3 * pw, (const IoWord*)xmem, (IoWord*)ytiled, M, K,
                                      N, M / F4, (M + np) / F4);
}
#endif
