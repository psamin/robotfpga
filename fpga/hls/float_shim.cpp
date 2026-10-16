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
          for (int n = 0; n < nt; n++)
            for (int m = 0; m < ms; m++) y[(size_t)(n0 + n) * M + g0 * LANES + m] += yt[(size_t)n * ms + m];
        }
      }
      free(xbuf);
    }
  }
  return status;
}
