// Full-precision linear engine for SmolVLA on the KR260 (issue #148).
//
// Y[n][m] = sum_k X[n][k] * W[m][k]. W lives in DDR as BF16; X, Y and all arithmetic are FP32.
// Design rules from docs/fpga-ddr-streaming-references.md:
//  - Batch-1 transformer layers are weight-bandwidth bound, so weights stream over all four
//    128-bit HP ports in parallel (one m_axi bundle each, [Li2025-KV260LLM]: 16-18 GB/s measured on
//    the K26 at 300 MHz). Each port carries 8 output channels per word, so one cycle delivers k for
//    32 output channels ("lanes"). The four ports are read in one loop, so HLS issues four bursts
//    at once.
//  - Weights are pre-tiled in DDR in the order they are consumed: port p, group g, k ascending,
//    so every load is one flat increasing burst [AMD-UG1399-burst].
//  - Ping-pong tiles: the group loop is an HLS DATAFLOW region, so the next 32-channel group streams
//    in while the current one computes [Zhang2015-Roofline]. NT tokens of X stay on chip, so each
//    weight is reused NT times.
//  - FP32 adds take several cycles, so consecutive iterations of the MAC loop must not touch the same
//    sum. The loop runs over (k, token block) with the block innermost: every (token, lane) keeps
//    its partial sums in a small RAM indexed by block and k mod PS, and the same entry comes back
//    only after NB * PS >= 16 cycles. No register muxes, no duplicated adders (a runtime k % PS index
//    into partitioned registers cost II = 7 and 515 adders; unrolling k by PS made HLS build 8x the
//    units). Summation order is fixed (partial p sums k = p mod PS in k order, then partials in
//    order), so results are deterministic.
//  - X and Y move over the io port as 128-bit words (4 FP32), so K must be a multiple of 4.
//  - Under Vitis HLS the DDR words are ap_uint<128> (its burst inference needs that); the host
//    build uses plain structs with the same bit layout.
// Bias, softmax, norms, activations and RoPE stay on the ARM. Reductions longer than K_MAX are split
// by the host into K_MAX chunks whose outputs it adds; output widths are padded to 32 by the host.
//
// Plain C++: the same core builds with Vitis HLS (float_top.cpp), g++ (tb_float.cpp) and as a
// shared library (float_shim.cpp) for running the whole model in PyTorch.
#pragma once
#include <stdint.h>
#include <string.h>

#ifdef NT
#undef NT  // Vitis HLS C simulation on Windows passes -DNT to the compiler (csim.mk)
#endif
#ifdef __VITIS_HLS__
#include <ap_int.h>
#endif

namespace fstream {

constexpr int PORTS = 4;            // HP0-HP3
constexpr int FL = 8;               // BF16 weights (output channels) per 128-bit word
constexpr int LANES = PORTS * FL;   // 32 output channels per group
constexpr int K_MAX = 4096;         // longest reduction per call
constexpr int NT = 32;              // tokens kept on chip per pass
#ifndef FSTREAM_TP
#define FSTREAM_TP 2  // run_float_hls.tcl overrides it: FSTREAM_TP=4 vitis_hls -f run_float_hls.tcl
#endif
constexpr int TP = FSTREAM_TP;      // tokens computed in parallel (TP * LANES FP32 MACs per cycle), power of 2
constexpr int NB = NT / TP;         // token blocks per pass
constexpr int PS = NB >= 8 ? 2 : 16 / NB;  // partial sums per (token, lane): NB * PS >= 16 cycles apart
constexpr int ACC_DIST = NB * PS;   // iterations between two updates of the same partial sum
constexpr int F4 = 4;               // FP32 values per 128-bit io word

static_assert(TP > 0 && (TP & (TP - 1)) == 0 && NT % TP == 0, "TP must be a power of 2 dividing NT");
static_assert(ACC_DIST >= 16, "partial sums must be revisited after the FP32 add latency");

#ifdef __VITIS_HLS__
typedef ap_uint<128> WPort;   // one 128-bit word on a weight port: 8 BF16, lane l in bits [16l+15:16l]
typedef ap_uint<128> IoWord;  // one 128-bit word on the io port: 4 FP32, value j in bits [32j+31:32j]

static inline uint16_t w_lane(const WPort& w, int l) {
#pragma HLS INLINE
  return (uint16_t)w.range(16 * l + 15, 16 * l);
}
static inline float io_lane(const IoWord& w, int j) {
#pragma HLS INLINE
  uint32_t u = w.range(32 * j + 31, 32 * j);
  float f;
  memcpy(&f, &u, sizeof f);
  return f;
}
static inline void io_set(IoWord& w, int j, float f) {
#pragma HLS INLINE
  uint32_t u;
  memcpy(&u, &f, sizeof u);
  w.range(32 * j + 31, 32 * j) = u;
}
#else
struct WPort {  // same bit layout as the HLS word (little-endian lanes)
  uint16_t v[FL];
};
struct IoWord {
  float v[F4];
};
static inline uint16_t w_lane(const WPort& w, int l) { return w.v[l]; }
static inline float io_lane(const IoWord& w, int j) { return w.v[j]; }
static inline void io_set(IoWord& w, int j, float f) { w.v[j] = f; }
#endif

static inline float bf16_to_f32(uint16_t b) {
#pragma HLS INLINE
  uint32_t u = (uint32_t)b << 16;  // BF16 is the top half of an FP32: widening is exact
  float f;
  memcpy(&f, &u, sizeof f);
  return f;
}

// One group of 32 output channels: K words per port at flat increasing addresses, the four ports in
// one loop so the four bursts run at once.
static void load_group(const WPort* w0, const WPort* w1, const WPort* w2, const WPort* w3, int g, int k_len,
                       WPort tile[PORTS][K_MAX]) {
  const size_t off = (size_t)g * k_len;
load:
  for (int k = 0; k < k_len; k++) {
#pragma HLS PIPELINE II = 1
    tile[0][k] = w0[off + k];
    tile[1][k] = w1[off + k];
    tile[2][k] = w2[off + k];
    tile[3][k] = w3[off + k];
  }
}

// One group against the NT tokens on chip. xbuf[a][b][k] holds token b * TP + a, so the TP tokens of
// a block sit in TP different memories (one read each per cycle). All NB blocks are always computed
// (padding rows are zero) so the partial-sum spacing never shrinks.
static void compute_group(const WPort tile[PORTS][K_MAX], const float xbuf[TP][NB][K_MAX], IoWord* ymem, int n0,
                          int nt, int g, int K, int M) {
  float acc[TP][LANES][ACC_DIST];
#pragma HLS ARRAY_PARTITION variable = acc complete dim = 1
#pragma HLS ARRAY_PARTITION variable = acc complete dim = 2
#pragma HLS BIND_STORAGE variable = acc type = ram_2p impl = lutram
  int k = 0, b = 0;
mac:
  for (int it = 0; it < K * NB; it++) {
#pragma HLS PIPELINE II = 1
#pragma HLS DEPENDENCE variable = acc inter distance = ACC_DIST true
    float w[LANES];
    for (int q = 0; q < PORTS; q++) {
#pragma HLS UNROLL
      const WPort word = tile[q][k];
      for (int l = 0; l < FL; l++) {
#pragma HLS UNROLL
        w[q * FL + l] = bf16_to_f32(w_lane(word, l));
      }
    }
    const int idx = b * PS + (k & (PS - 1));
    const bool first = k < PS;  // first visit of this partial sum: start from zero
    for (int a = 0; a < TP; a++) {
#pragma HLS UNROLL
      const float x = xbuf[a][b][k];
      for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
        const float cur = first ? 0.0f : acc[a][l][idx];
        acc[a][l][idx] = cur + x * w[l];
      }
    }
    if (b == NB - 1) {
      b = 0;
      k++;
    } else {
      b++;
    }
  }
store:
  for (int bb = 0; bb < NB; bb++) {
    for (int a = 0; a < TP; a++) {
      if (bb * TP + a < nt) {
      store_row:
        for (int l4 = 0; l4 < LANES / F4; l4++) {
#pragma HLS PIPELINE II = 1
          IoWord o;
          for (int j = 0; j < F4; j++) {
#pragma HLS UNROLL
            float s = acc[a][l4 * F4 + j][bb * PS];
            for (int p = 1; p < PS; p++) s += acc[a][l4 * F4 + j][bb * PS + p];
            io_set(o, j, s);
          }
          ymem[(size_t)(n0 + bb * TP + a) * (M / F4) + g * (LANES / F4) + l4] = o;
        }
      }
    }
  }
}

// All groups against one pass of tokens. The loop is a DATAFLOW region: HLS double-buffers `tile`
// and loads group g + 1 while group g computes. Its bound and the scalars it uses are arguments,
// as Vitis HLS requires for a dataflow loop.
static void run_tile(const WPort* w0, const WPort* w1, const WPort* w2, const WPort* w3, IoWord* ymem,
                     const float xbuf[TP][NB][K_MAX], int groups, int n0, int nt, int K, int M) {
groups_loop:
  for (int g = 0; g < groups; g++) {
#pragma HLS DATAFLOW
    WPort tile[PORTS][K_MAX];
#pragma HLS ARRAY_PARTITION variable = tile complete dim = 1
#pragma HLS BIND_STORAGE variable = tile type = ram_2p impl = uram
    load_group(w0, w1, w2, w3, g, K, tile);
    compute_group(tile, xbuf, ymem, n0, nt, g, K, M);
  }
}

// Core with a caller-owned X buffer (static in the HLS top, per-thread in the shim).
// w0..w3: the four port arrays, each [M / LANES groups][K] words. xmem: N x K FP32, ymem: N x M FP32,
// both as 128-bit words (K and M multiples of 4; M a multiple of 32).
static int linear_bf16_core(const WPort* w0, const WPort* w1, const WPort* w2, const WPort* w3, const IoWord* xmem,
                            IoWord* ymem, int M, int K, int N, float xbuf[TP][NB][K_MAX]) {
  if (M % LANES || K < 1 || K > K_MAX || K % F4 || N < 1) return 1;
  const int groups = M / LANES;
  const int k4 = K / F4;
tiles:
  for (int n0 = 0; n0 < N; n0 += NT) {
    const int nt = (N - n0) < NT ? (N - n0) : NT;
  load_x:
    for (int n = 0; n < NT; n++) {
      for (int i = 0; i < k4; i++) {
#pragma HLS PIPELINE II = 1
        IoWord wx;
        if (n < nt) {
          wx = xmem[(size_t)(n0 + n) * k4 + i];
        } else {
          for (int j = 0; j < F4; j++) io_set(wx, j, 0.0f);  // pad the last pass with zero tokens
        }
        for (int j = 0; j < F4; j++) {
#pragma HLS UNROLL
          xbuf[n % TP][n / TP][i * F4 + j] = io_lane(wx, j);
        }
      }
    }
    run_tile(w0, w1, w2, w3, ymem, xbuf, groups, n0, nt, K, M);
  }
  return 0;
}

}  // namespace fstream
