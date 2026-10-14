// Full-precision linear engine for SmolVLA on the KR260 (issue #148).
//
// Y[n][m] = sum_k X[n][k] * W[m][k]. W lives in DDR as BF16; X, Y and all arithmetic are FP32.
// Design rules from plans/fpga-ddr-streaming-references.md:
//  - Batch-1 transformer layers are weight-bandwidth bound, so weights stream over all four
//    128-bit HP ports in parallel (one m_axi bundle each, [Li2025-KV260LLM]: 16-18 GB/s measured on
//    the K26 at 300 MHz). Each port carries 8 output channels per word, so one cycle delivers k for
//    32 output channels ("lanes").
//  - Weights are pre-tiled in DDR in the order they are consumed: port p, group g, k ascending,
//    so every load is one flat increasing burst [AMD-UG1399-burst].
//  - Ping-pong tiles: the next 32-channel group streams in while the current one computes
//    [Zhang2015-Roofline]. NT tokens of X stay on chip, so each weight is reused NT times.
//  - FP32 adds take several cycles: every (token, lane) keeps PS interleaved partial sums (k mod PS)
//    so the reduction loop still pipelines at II = 1. Summation order is fixed (partials in k order,
//    then partials 0..PS-1), so results are deterministic.
// Bias, softmax, norms, activations and RoPE stay on the ARM. Reductions longer than K_MAX are split
// by the host into K_MAX chunks whose outputs it adds; output widths are padded to 32 by the host.
//
// Plain C++: the same core builds with Vitis HLS (float_top.cpp), g++ (tb_float.cpp) and as a
// shared library (float_shim.cpp) for running the whole model in PyTorch.
#pragma once
#include <stdint.h>
#include <string.h>

namespace fstream {

constexpr int PORTS = 4;            // HP0-HP3
constexpr int FL = 8;               // BF16 weights (output channels) per 128-bit word
constexpr int LANES = PORTS * FL;   // 32 output channels per group
constexpr int K_MAX = 4096;         // longest reduction per call
constexpr int NT = 32;              // tokens kept on chip per pass
constexpr int TP = 2;               // tokens computed in parallel (TP * LANES FP32 MACs per cycle)
constexpr int PS = 8;               // interleaved partial sums, covers the FP32 add latency

struct WordBF {  // one 128-bit DDR word on one port
  uint16_t v[FL];
};

static inline float bf16_to_f32(uint16_t b) {
#pragma HLS INLINE
  uint32_t u = (uint32_t)b << 16;  // BF16 is the top half of an FP32: widening is exact
  float f;
  memcpy(&f, &u, sizeof f);
  return f;
}

// One port's share of a group: K words, flat increasing addresses (a single long burst).
static void load_port(const WordBF* wp, size_t off, int k_len, WordBF dst[K_MAX]) {
  for (int k = 0; k < k_len; k++) {
#pragma HLS PIPELINE II = 1
    dst[k] = wp[off + k];
  }
}

static void load_group(const WordBF* w0, const WordBF* w1, const WordBF* w2, const WordBF* w3, int g, int k_len,
                       WordBF tile[PORTS][K_MAX]) {
  const size_t off = (size_t)g * k_len;
  // Four independent bundles: HLS issues the four bursts in parallel.
  load_port(w0, off, k_len, tile[0]);
  load_port(w1, off, k_len, tile[1]);
  load_port(w2, off, k_len, tile[2]);
  load_port(w3, off, k_len, tile[3]);
}

// Core with caller-owned buffers (static in the HLS top, per-thread in the shim).
// w0..w3: the four port arrays, each [M / LANES groups][K] words.
static int linear_bf16_core(const WordBF* w0, const WordBF* w1, const WordBF* w2, const WordBF* w3,
                            const float* xmem, float* ymem, int M, int K, int N, float xbuf[NT][K_MAX],
                            WordBF tile[2][PORTS][K_MAX]) {
  if (M % LANES || K < 1 || K > K_MAX || N < 1) return 1;
  const int groups = M / LANES;
tiles:
  for (int n0 = 0; n0 < N; n0 += NT) {
    const int nt = (N - n0) < NT ? (N - n0) : NT;
  load_x:
    for (int i = 0; i < nt * K; i++) {
#pragma HLS PIPELINE II = 1
      xbuf[i / K][i % K] = xmem[(size_t)(n0 + i / K) * K + i % K];
    }
    for (int i = nt; i < NT; i++)
      for (int k = 0; k < K; k++) xbuf[i][k] = 0.0f;
    load_group(w0, w1, w2, w3, 0, K, tile[0]);
  groups_loop:
    for (int g = 0; g < groups; g++) {
      if (g + 1 < groups) load_group(w0, w1, w2, w3, g + 1, K, tile[(g + 1) & 1]);
      WordBF(*t)[K_MAX] = tile[g & 1];
    token_blocks:
      for (int nb = 0; nb < nt; nb += TP) {
        float acc[TP][LANES][PS];
#pragma HLS ARRAY_PARTITION variable = acc complete dim = 0
        for (int a = 0; a < TP; a++)
          for (int l = 0; l < LANES; l++)
            for (int p = 0; p < PS; p++) acc[a][l][p] = 0.0f;
      red:
        for (int k = 0; k < K; k++) {
#pragma HLS PIPELINE II = 1
#pragma HLS DEPENDENCE variable = acc inter distance = 8 true
          const int p = k % PS;
          float w[LANES];
          for (int q = 0; q < PORTS; q++) {
#pragma HLS UNROLL
            for (int l = 0; l < FL; l++) {
#pragma HLS UNROLL
              w[q * FL + l] = bf16_to_f32(t[q][k].v[l]);
            }
          }
          for (int a = 0; a < TP; a++) {
#pragma HLS UNROLL
            const float x = xbuf[nb + a][k];
            for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
              acc[a][l][p] += x * w[l];
            }
          }
        }
        for (int a = 0; a < TP && nb + a < nt; a++) {
          for (int l = 0; l < LANES; l++) {
            float s = 0.0f;
            for (int p = 0; p < PS; p++) s += acc[a][l][p];
            ymem[(size_t)(n0 + nb + a) * M + g * LANES + l] = s;
          }
        }
      }
    }
  }
  return 0;
}

}  // namespace fstream
