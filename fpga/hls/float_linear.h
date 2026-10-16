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
//  - FP32 adds take several cycles: every (token, lane) keeps PS interleaved partial sums (k mod PS).
//    The reduction loop steps PS values of k per iteration (II = PS, one weight word per port per
//    cycle) so each partial sum has a fixed index; a runtime k % PS index made Vitis HLS mux the
//    registers and serialize the adds (II = 7, 515 adders at TP = 2). Summation order is fixed
//    (partials in k order, then partials 0..PS-1), so results are deterministic.
//  - X and Y move over the io port as 128-bit words (4 FP32), so K must be a multiple of 4.
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
constexpr int PS = 8;               // interleaved partial sums, covers the FP32 add latency
constexpr int F4 = 4;               // FP32 values per 128-bit io word

static_assert(TP > 0 && (TP & (TP - 1)) == 0 && NT % TP == 0, "TP must be a power of 2 dividing NT");

struct WordBF {  // one 128-bit DDR word on one weight port
  uint16_t v[FL];
};

struct WordF4 {  // one 128-bit word on the io port: 4 FP32 activations
  float v[F4];
};

static inline float bf16_to_f32(uint16_t b) {
#pragma HLS INLINE
  uint32_t u = (uint32_t)b << 16;  // BF16 is the top half of an FP32: widening is exact
  float f;
  memcpy(&f, &u, sizeof f);
  return f;
}

// One group of 32 output channels: K words per port at flat increasing addresses, the four ports in
// one loop so the four bursts run at once.
static void load_group(const WordBF* w0, const WordBF* w1, const WordBF* w2, const WordBF* w3, int g, int k_len,
                       WordBF tile[PORTS][K_MAX]) {
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
// a block sit in TP different memories (one read each per cycle).
static void compute_group(const WordBF tile[PORTS][K_MAX], const float xbuf[TP][NB][K_MAX], WordF4* ymem,
                          int n0, int nt, int g, int K, int M) {
token_blocks:
  for (int b = 0; b * TP < nt; b++) {
    float acc[TP][LANES][PS];
#pragma HLS ARRAY_PARTITION variable = acc complete dim = 0
    for (int a = 0; a < TP; a++)
      for (int l = 0; l < LANES; l++)
        for (int p = 0; p < PS; p++) acc[a][l][p] = 0.0f;
  red:
    for (int k0 = 0; k0 < K; k0 += PS) {
#pragma HLS PIPELINE II = PS
      for (int p = 0; p < PS; p++) {
#pragma HLS UNROLL
        const int k = k0 + p;
        if (k < K) {
          float w[LANES];
          for (int q = 0; q < PORTS; q++) {
#pragma HLS UNROLL
            const WordBF word = tile[q][k];
            for (int l = 0; l < FL; l++) {
#pragma HLS UNROLL
              w[q * FL + l] = bf16_to_f32(word.v[l]);
            }
          }
          for (int a = 0; a < TP; a++) {
#pragma HLS UNROLL
            const float x = xbuf[a][b][k];
            for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
              acc[a][l][p] += x * w[l];
            }
          }
        }
      }
    }
  store:
    for (int a = 0; a < TP; a++) {
      if (b * TP + a < nt) {
      store_row:
        for (int l4 = 0; l4 < LANES / F4; l4++) {
#pragma HLS PIPELINE II = 1
          WordF4 o;
          for (int j = 0; j < F4; j++) {
#pragma HLS UNROLL
            float s = 0.0f;
            for (int p = 0; p < PS; p++) s += acc[a][l4 * F4 + j][p];
            o.v[j] = s;
          }
          ymem[(size_t)(n0 + b * TP + a) * (M / F4) + g * (LANES / F4) + l4] = o;
        }
      }
    }
  }
}

// Core with a caller-owned X buffer (static in the HLS top, per-thread in the shim).
// w0..w3: the four port arrays, each [M / LANES groups][K] words. xmem: N x K FP32, ymem: N x M FP32,
// both as 128-bit words (K and M multiples of 4; M a multiple of 32).
static int linear_bf16_core(const WordBF* w0, const WordBF* w1, const WordBF* w2, const WordBF* w3,
                            const WordF4* xmem, WordF4* ymem, int M, int K, int N, float xbuf[TP][NB][K_MAX]) {
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
        WordF4 wx;
        if (n < nt) {
          wx = xmem[(size_t)(n0 + n) * k4 + i];
        } else {
          for (int j = 0; j < F4; j++) wx.v[j] = 0.0f;  // pad the last pass with zero tokens
        }
        for (int j = 0; j < F4; j++) {
#pragma HLS UNROLL
          xbuf[n % TP][n / TP][i * F4 + j] = wx.v[j];
        }
      }
    }
  groups_loop:
    for (int g = 0; g < groups; g++) {
#pragma HLS DATAFLOW
      WordBF tile[PORTS][K_MAX];
#pragma HLS AGGREGATE compact = bit variable = tile
#pragma HLS ARRAY_PARTITION variable = tile complete dim = 1
      load_group(w0, w1, w2, w3, g, K, tile);
      compute_group(tile, xbuf, ymem, n0, nt, g, K, M);
    }
  }
  return 0;
}

}  // namespace fstream
