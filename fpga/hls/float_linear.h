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
//  - One pass over a tile of NT tokens is a three-process DATAFLOW region: load streams the weights
//    of all groups through FIFOs, compute consumes them, store writes the results. The weight loads
//    and the result stores therefore overlap the MACs [Zhang2015-Roofline]. (An earlier version ran
//    load and compute per group with a ping-pong tile; measured on the board, the load and the
//    32 single-row stores were serialized with the MACs, ~K + 2,500 extra cycles per group.) NT
//    tokens of X stay on chip, so each weight is reused NT times.
//  - FP32 adds take several cycles, so consecutive iterations of the MAC loop must not touch the same
//    sum. The loop runs over (k, token block) with the block innermost: every (token, lane) keeps
//    its partial sums in a small RAM indexed by block and k mod PS, and the same entry comes back
//    only after NB * PS >= 16 cycles. No register muxes, no duplicated adders (a runtime k % PS index
//    into partitioned registers cost II = 7 and 515 adders; unrolling k by PS made HLS build 8x the
//    units). Summation order is fixed (partial p sums k = p mod PS in k order, then partials in
//    order), so results are deterministic.
//  - The BF16 x FP32 product could be one DSP: the 8 x 24-bit significand product is exact, so rounding
//    it to nearest even gives the IEEE FP32 product bit for bit (bf16_mul below, fpga/hls/tb_mul.cpp).
//    Not used: the Xilinx multiplier core gives the same bits with far fewer LUTs; see mac_mul.
//  - X moves over the io port as 128-bit words (4 FP32), so K must be a multiple of 4. Y is written
//    group-major: Y_tiled[g][n][l] for group g (32 lanes), token n of all NT-token tiles (padded), so
//    one tile-group is a single 4 KB burst. The host un-tiles it.
// Bias, norms, activations and RoPE stay on the ARM. Reductions longer than K_MAX are split by the
// host into K_MAX chunks whose outputs it adds; output widths are padded to 32 by the host.
// A second operation, selected by the top's `mode` register, is the row softmax (softmax_rows below):
// it shares the io port and the register file with the GEMM and touches none of its logic.
//
// Plain C++: the same core builds with Vitis HLS (float_top.cpp), g++ (tb_float.cpp, tb_softmax.cpp)
// and as a shared library (float_shim.cpp) for running the whole model in PyTorch.
#pragma once
#include <stdint.h>
#include <string.h>

#ifdef NT
#undef NT  // Vitis HLS C simulation on Windows passes -DNT to the compiler (csim.mk)
#endif
#ifdef __VITIS_HLS__
#include <ap_int.h>
#include <hls_math.h>
#include <hls_stream.h>
#else
#include <math.h>

#include <vector>
#ifdef FSTREAM_FPO_EXP
#include <hls_fpo.h>  // the bit-accurate model of the Xilinx FP exp core, see smx_exp
#endif
#endif

namespace fstream {

constexpr int PORTS = 4;            // HP0-HP3
constexpr int FL = 8;               // BF16 weights (output channels) per 128-bit word
constexpr int LANES = PORTS * FL;   // 32 output channels per group
constexpr int K_MAX = 4096;         // longest reduction per call
constexpr int NT = 32;              // tokens kept on chip per pass
#ifndef FSTREAM_TP
#define FSTREAM_TP 2  // run_float_hls.tcl overrides it: FSTREAM_TP=8 vitis_hls -f run_float_hls.tcl
#endif
constexpr int TP = FSTREAM_TP;      // tokens computed in parallel (TP * LANES FP32 MACs per cycle), power of 2
constexpr int NB = NT / TP;         // token blocks per pass
constexpr int PS = NB >= 8 ? 2 : 16 / NB;  // partial sums per (token, lane): NB * PS >= 16 cycles apart
constexpr int ACC_DIST = NB * PS;   // iterations between two updates of the same partial sum
constexpr int F4 = 4;               // FP32 values per 128-bit io word
constexpr int YW = NT * LANES / F4; // io words per tile-group of Y (256 = one 4 KB burst)
constexpr int WFIFO = 512;          // weight words buffered per port between load and compute
constexpr int YFIFO = 2 * YW;       // result words buffered between compute and store

static_assert(TP > 0 && (TP & (TP - 1)) == 0 && NT % TP == 0, "TP must be a power of 2 dividing NT");
static_assert(ACC_DIST >= 16, "partial sums must be revisited after the FP32 add latency");

#ifdef __VITIS_HLS__
typedef ap_uint<128> WPort;   // one 128-bit word on a weight port: 8 BF16, lane l in bits [16l+15:16l]
typedef ap_uint<128> IoWord;  // one 128-bit word on the io port: 4 FP32, value j in bits [32j+31:32j]
template <class T>
using Fifo = hls::stream<T>;

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
template <class T>
class Fifo {  // the host runs the three processes one after another, so this is just a queue
 public:
  void write(const T& v) { q.push_back(v); }
  T read() { return q[r++]; }

 private:
  std::vector<T> q;
  size_t r = 0;
};
#endif

static inline float bf16_to_f32(uint16_t b) {
#pragma HLS INLINE
  uint32_t u = (uint32_t)b << 16;  // BF16 is the top half of an FP32: widening is exact
  float f;
  memcpy(&f, &u, sizeof f);
  return f;
}

// BF16 x FP32 product, bit for bit the IEEE round-to-nearest-even FP32 product (the result the FP32
// multiplier it replaces gives), from one 8 x 24-bit integer multiply: that product is exact in 32 bits,
// so rounding it to 24 is the whole job. One DSP per MAC instead of three. Denormal inputs and results
// are flushed to signed zero, as the Xilinx FP32 multiplier does; NaN propagates. fpga/hls/tb_mul.cpp
// checks it against the native multiply on every BF16 value and 40 M random pairs.
static inline float bf16_mul(float x, uint16_t w) {
#pragma HLS INLINE
  uint32_t xb;
  memcpy(&xb, &x, sizeof xb);
  const uint32_t ex = (xb >> 23) & 0xFF, ew = ((uint32_t)w >> 7) & 0xFF;
  const uint32_t mx = xb & 0x7FFFFF, mw = (uint32_t)w & 0x7F;
  const uint32_t sign = ((xb >> 31) ^ ((uint32_t)w >> 15)) << 31;
  const uint32_t sigx = mx | 0x800000;  // 24 bits
  const uint32_t sigw = mw | 0x80;      // 8 bits
#ifdef __VITIS_HLS__
  const ap_uint<32> prod = ap_uint<24>(sigx) * ap_uint<8>(sigw);
#pragma HLS BIND_OP variable = prod op = mul impl = dsp
  const uint32_t p = prod;
#else
  const uint32_t p = sigx * sigw;  // < 2^32
#endif
  const bool hi = (p >> 31) & 1;                      // product in [2, 4): one more bit to drop
  const uint32_t mant = hi ? (p >> 8) : (p >> 7);     // 24 bits with the hidden one
  const uint32_t rem = hi ? (p & 0xFF) : (p & 0x7F);  // dropped bits
  const uint32_t half = hi ? 0x80 : 0x40;
  const bool up = rem > half || (rem == half && (mant & 1));
  const uint32_t rounded = mant + (up ? 1 : 0);  // 2^24 when the mantissa was all ones: carry
  const int e = (int)ex + (int)ew - 127 + (hi ? 1 : 0) + (int)(rounded >> 24);
  const uint32_t frac = rounded & 0x7FFFFF;  // 0 after a carry, as it should be
  const bool nan = (ex == 0xFF && mx) || (ew == 0xFF && mw) || (ex == 0xFF && ew == 0) || (ew == 0xFF && ex == 0);
  uint32_t out;
  if (nan) {
    out = 0x7FC00000;
  } else if (ex == 0xFF || ew == 0xFF || e >= 0xFF) {
    out = sign | 0x7F800000;
  } else if (ex == 0 || ew == 0 || e <= 0) {
    out = sign;
  } else {
    out = sign | ((uint32_t)e << 23) | frac;
  }
  float f;
  memcpy(&f, &out, sizeof f);
  return f;
}

// The product used in the MAC loop: the FP32 multiply, which Vitis HLS maps to its multiplier core with
// FSTREAM_FMUL_IMPL = maxdsp (3 DSP, ~120 LUT; the shipped bitstreams), fulldsp or meddsp (1 DSP, ~300
// LUT). FSTREAM_BF16_MUL uses bf16_mul instead: the same bits (denormals flush either way), but Vitis
// HLS estimates it at ~400 LUT per MAC, and this part runs out of LUTs before DSPs.
#define FSTREAM_PRAGMA_(x) _Pragma(#x)
#define FSTREAM_PRAGMA(x) FSTREAM_PRAGMA_(x)
#ifndef FSTREAM_FMUL_IMPL
#define FSTREAM_FMUL_IMPL maxdsp
#endif
static inline float mac_mul(float x, uint16_t w) {
#pragma HLS INLINE
#ifdef FSTREAM_BF16_MUL
  return bf16_mul(x, w);
#else
  float p = x * bf16_to_f32(w);
  FSTREAM_PRAGMA(HLS BIND_OP variable = p op = fmul impl = FSTREAM_FMUL_IMPL)
  return p;
#endif
}

// Process 1: every group's K words per port, flat increasing addresses, the four ports in one loop so
// the four bursts run at once. Runs ahead of compute by up to WFIFO words.
static void load_all(const WPort* w0, const WPort* w1, const WPort* w2, const WPort* w3, int words,
                     Fifo<WPort>& s0, Fifo<WPort>& s1, Fifo<WPort>& s2, Fifo<WPort>& s3) {
load:
  for (int i = 0; i < words; i++) {
#pragma HLS PIPELINE II = 1
    s0.write(w0[i]);
    s1.write(w1[i]);
    s2.write(w2[i]);
    s3.write(w3[i]);
  }
}

// Process 2: every group against the NT tokens on chip. xbuf[a][b][k] holds token b * TP + a, so the
// TP tokens of a block sit in TP different memories (one read each per cycle). All NB blocks are
// always computed (padding rows are zero) so the partial-sum spacing never shrinks. One weight word
// per port is read when a new k starts and serves the NB blocks. After the MACs of a group, its
// NT x 32 sums go out as YW io words, row by row.
static void compute_all(Fifo<WPort>& s0, Fifo<WPort>& s1, Fifo<WPort>& s2, Fifo<WPort>& s3,
                        const float xbuf[TP][NB][K_MAX], int groups, int K, Fifo<IoWord>& ys) {
  float acc[TP][LANES][ACC_DIST];
#pragma HLS ARRAY_PARTITION variable = acc complete dim = 1
#pragma HLS ARRAY_PARTITION variable = acc complete dim = 2
#pragma HLS BIND_STORAGE variable = acc type = ram_2p impl = lutram
groups_loop:
  for (int g = 0; g < groups; g++) {
    WPort word[PORTS];
#pragma HLS ARRAY_PARTITION variable = word complete
    int k = 0, b = 0;
  mac:
    for (int it = 0; it < K * NB; it++) {
#pragma HLS PIPELINE II = 1
#pragma HLS DEPENDENCE variable = acc inter distance = ACC_DIST true
      if (b == 0) {
        word[0] = s0.read();
        word[1] = s1.read();
        word[2] = s2.read();
        word[3] = s3.read();
      }
      uint16_t w[LANES];
      for (int q = 0; q < PORTS; q++) {
#pragma HLS UNROLL
        for (int l = 0; l < FL; l++) {
#pragma HLS UNROLL
          w[q * FL + l] = w_lane(word[q], l);
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
          acc[a][l][idx] = cur + mac_mul(x, w[l]);
        }
      }
      if (b == NB - 1) {
        b = 0;
        k++;
      } else {
        b++;
      }
    }
  emit:
    for (int r = 0; r < NT; r++) {  // token r of the tile = block r / TP, slot r % TP
      for (int l4 = 0; l4 < LANES / F4; l4++) {
#pragma HLS PIPELINE II = 1
        const int bb = r / TP, a = r % TP;
        IoWord o;
        for (int j = 0; j < F4; j++) {
#pragma HLS UNROLL
          float s = acc[a][l4 * F4 + j][bb * PS];
          for (int p = 1; p < PS; p++) s += acc[a][l4 * F4 + j][bb * PS + p];
          io_set(o, j, s);
        }
        ys.write(o);
      }
    }
  }
}

// Process 3: one 4 KB burst per group: Y_tiled[g][tile * NT + r][lanes] for the NT rows of this tile.
static void store_all(Fifo<IoWord>& ys, IoWord* ymem, int groups, int ntiles, int tile) {
store:
  for (int g = 0; g < groups; g++) {
    const size_t base = ((size_t)g * ntiles + tile) * YW;
  store_burst:
    for (int i = 0; i < YW; i++) {
#pragma HLS PIPELINE II = 1
      ymem[base + i] = ys.read();
    }
  }
}

// One pass of NT tokens against all groups: load, compute and store run concurrently.
static void run_tile(const WPort* w0, const WPort* w1, const WPort* w2, const WPort* w3, IoWord* ymem,
                     const float xbuf[TP][NB][K_MAX], int groups, int ntiles, int tile, int K) {
#pragma HLS DATAFLOW
  Fifo<WPort> s0, s1, s2, s3;
  Fifo<IoWord> ys;
#pragma HLS STREAM variable = s0 depth = WFIFO
#pragma HLS STREAM variable = s1 depth = WFIFO
#pragma HLS STREAM variable = s2 depth = WFIFO
#pragma HLS STREAM variable = s3 depth = WFIFO
#pragma HLS STREAM variable = ys depth = YFIFO
  load_all(w0, w1, w2, w3, groups * K, s0, s1, s2, s3);
  compute_all(s0, s1, s2, s3, xbuf, groups, K, ys);
  store_all(ys, ymem, groups, ntiles, tile);
}

// Core with a caller-owned X buffer (static in the HLS top, per-thread in the shim).
// w0..w3: the four port arrays, each [M / LANES groups][K] words. xmem: N x K FP32 as 128-bit words
// (K a multiple of 4). ymem: Y_tiled[M / LANES][ntiles * NT][LANES] FP32 as 128-bit words, where
// ntiles = ceil(N / NT); rows past N hold the zero padding tokens' sums. M a multiple of 32.
static int linear_bf16_core(const WPort* w0, const WPort* w1, const WPort* w2, const WPort* w3, const IoWord* xmem,
                            IoWord* ymem, int M, int K, int N, float xbuf[TP][NB][K_MAX]) {
  if (M % LANES || K < 1 || K > K_MAX || K % F4 || N < 1) return 1;
  const int groups = M / LANES;
  const int k4 = K / F4;
  const int ntiles = (N + NT - 1) / NT;
tiles:
  for (int t = 0; t < ntiles; t++) {
    const int n0 = t * NT;
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
    run_tile(w0, w1, w2, w3, ymem, xbuf, groups, ntiles, t, K);
  }
  return 0;
}

// ---------------------------------------------------------------------------------------------------
// Mode 1: row softmax. xmem holds N rows of L FP32 values (L a multiple of 4, L <= SMX_L_MAX), ymem
// receives softmax(row) in FP32, same layout. The recipe is the one of torch's CPU kernel
// (_vec_softmax_lastdim in aten/src/ATen/native/cpu/SoftMaxKernel.cpp): m = max_i x_i;
// e_i = exp(x_i - m); s = sum_i e_i; inv = 1 / s; y_i = e_i * inv (one reciprocal per row and a
// multiply per element, not a division per element). What differs from torch, all inside FP32
// rounding: exp is the Xilinx floating-point exp core (smx_exp, max 1.1 ulp measured) instead of
// SLEEF's expf_u10; s is summed as 4 lanes x SMX_P partials in a fixed tree (torch: Vec::size()
// lanes, then a shuffle tree); e_i below FLT_MIN flush to 0 (the core has no subnormal outputs).
// The max is exact (an order-preserving integer key, so no FP compare sits in a loop-carried path).
// Every loop is II = 1 and 4 elements wide: one 128-bit io word per cycle.
//
// Four dataflow processes, each a loop over the rows, joined by FIFOs:
//   smx_read_max    reads the rows as one sequential burst stream, forwards the words, sends each
//                   row's max when its last word is in
//   smx_exp_sum     e = exp(x - m) for the 4 lanes of a word, the 4 added in a small tree, the word's
//                   sum accumulated into one of SMX_P partial sums (the ACC_DIST idea of the MAC loop:
//                   a partial is revisited SMX_P >= 16 cycles later, after the adder latency); forwards
//                   the e words; after a row it sends the SMX_P partials as SMX_P / 4 io words
//   smx_reduce      adds the partials in a fixed order (deterministic) on four adders, one division per
//                   row (a 63-adder tree, which HLS builds for a plain unrolled reduction, cost 126 DSP)
//   smx_scale_store multiplies the e words by inv and writes the rows as one burst stream
// The word FIFOs hold a whole row (SMX_W_MAX words), because a row's max (and its sum) is only known
// after its last word, so the consumer can only start when the producer has pushed the entire row.
// With L = 1024 a FIFO holds 4 rows, so reads, exp and writes of neighbouring rows overlap.
constexpr int SMX_L_MAX = K_MAX;            // values per row (the K register holds L, N the rows)
constexpr int SMX_W_MAX = SMX_L_MAX / F4;   // io words per row
constexpr int SMX_P = 16;                   // partial sums per row: >= the FP32 adder latency (11)
constexpr int SMX_PW = SMX_P / F4;          // io words carrying a row's partials
constexpr int SMX_FIFO = SMX_W_MAX;         // words buffered between the processes (>= one whole row)
static_assert((SMX_P & (SMX_P - 1)) == 0 && SMX_P >= 16, "SMX_P: power of 2, past the FP32 add latency");

// The exponential: Vitis HLS maps hls::exp(float) to the Xilinx Floating-Point Operator exp core
// (hls_exp_float.cpp: HLS_FPO_EXPF). On the host, FSTREAM_FPO_EXP selects that core's bit-accurate
// C model (tb_softmax.cpp says how to link it), otherwise the C library's expf.
static inline float smx_exp(float x) {
#pragma HLS INLINE
#ifdef __VITIS_HLS__
  return hls::exp(x);
#elif defined(FSTREAM_FPO_EXP)
  return HLS_FPO_EXPF(x);
#else
  return expf(x);
#endif
}

// Order-preserving unsigned key of an FP32: key(a) > key(b) iff a > b (with -0 < +0, NaN above +inf),
// so the row max is an integer compare per lane and the loop-carried path has no FP compare.
static inline uint32_t smx_key(float f) {
#pragma HLS INLINE
  uint32_t u;
  memcpy(&u, &f, sizeof u);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
static inline float smx_unkey(uint32_t k) {
#pragma HLS INLINE
  const uint32_t u = (k & 0x80000000u) ? (k & 0x7FFFFFFFu) : ~k;
  float f;
  memcpy(&f, &u, sizeof f);
  return f;
}

// Process 1: the words of all rows in one sequential stream (one burst stream over total = rows * L4
// words); the running max of each lane is an integer key; the row max goes out with the row's last word.
static void smx_read_max(const IoWord* xmem, int total, int L4, Fifo<IoWord>& xs, Fifo<float>& ms) {
  int i = 0;
  uint32_t kmax[F4];
#pragma HLS ARRAY_PARTITION variable = kmax complete
read_max:
  for (int t = 0; t < total; t++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 3145728 avg = 3145728
    const IoWord w = xmem[t];
    xs.write(w);
    for (int j = 0; j < F4; j++) {
#pragma HLS UNROLL
      const uint32_t k = smx_key(io_lane(w, j));
      kmax[j] = (i == 0 || k > kmax[j]) ? k : kmax[j];
    }
    if (i == L4 - 1) {
      const uint32_t k01 = kmax[0] > kmax[1] ? kmax[0] : kmax[1];
      const uint32_t k23 = kmax[2] > kmax[3] ? kmax[2] : kmax[3];
      ms.write(smx_unkey(k01 > k23 ? k01 : k23));
      i = 0;
    } else {
      i++;
    }
  }
}

// Process 2: per row, e = exp(x - m), the word sum t = (e0 + e1) + (e2 + e3) and the partial sums;
// then the partials as SMX_PW words. Partial p is the sum of t over the words i == p mod SMX_P, in
// word order; a row shorter than SMX_P words leaves the partials p >= L4 untouched, so those go out
// as zero.
static void smx_exp_sum(Fifo<IoWord>& xs, Fifo<float>& ms, int rows, int L4, Fifo<IoWord>& es,
                        Fifo<IoWord>& ps) {
  float acc[SMX_P];
#pragma HLS BIND_STORAGE variable = acc type = ram_2p impl = lutram
rows_exp:
  for (int r = 0; r < rows; r++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 12288 avg = 12288
    const float m = ms.read();
  exp_sum:
    for (int i = 0; i < L4; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 1024 avg = 256
#pragma HLS DEPENDENCE variable = acc inter distance = SMX_P true
      const IoWord w = xs.read();
      const int idx = i & (SMX_P - 1);
      const bool first = i < SMX_P;  // first visit of this partial: start from zero
      IoWord o;
      float e[F4];
      for (int j = 0; j < F4; j++) {
#pragma HLS UNROLL
        e[j] = smx_exp(io_lane(w, j) - m);
        io_set(o, j, e[j]);
      }
      const float t = (e[0] + e[1]) + (e[2] + e[3]);
      const float cur = first ? 0.0f : acc[idx];
      acc[idx] = cur + t;
      es.write(o);
    }
  emit_partials:
    for (int p = 0; p < SMX_PW; p++) {
#pragma HLS PIPELINE II = 1
      IoWord q;
      for (int j = 0; j < F4; j++) {
#pragma HLS UNROLL
        io_set(q, j, p * F4 + j < L4 ? acc[p * F4 + j] : 0.0f);
      }
      ps.write(q);
    }
  }
}

// Process 3: per row, s = the sum of the SMX_P partials in a fixed order and inv = 1 / s (an IEEE
// division, as torch computes 1 / sum). Four chains of SMX_PW dependent adds run side by side on four
// adders, then the chains are added; the two sub-loops keep HLS from flattening the row loop, which
// would instantiate one adder per add.
static void smx_reduce(Fifo<IoWord>& ps, int rows, Fifo<float>& ss) {
rows_reduce:
  for (int r = 0; r < rows; r++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 12288 avg = 12288
    float v[SMX_P];
#pragma HLS ARRAY_PARTITION variable = v complete
  gather:
    for (int p = 0; p < SMX_PW; p++) {
#pragma HLS PIPELINE II = 1
      const IoWord q = ps.read();
      for (int j = 0; j < F4; j++) {
#pragma HLS UNROLL
        v[p * F4 + j] = io_lane(q, j);
      }
    }
    float a[F4];
#pragma HLS ARRAY_PARTITION variable = a complete
    for (int j = 0; j < F4; j++) {
#pragma HLS UNROLL
      a[j] = v[j];
    }
  chains:
    for (int p = 1; p < SMX_PW; p++) {
      for (int j = 0; j < F4; j++) {
#pragma HLS UNROLL
        a[j] += v[p * F4 + j];
      }
    }
    const float s = (a[0] + a[1]) + (a[2] + a[3]);
    ss.write(1.0f / s);
  }
}

// Process 4: y = e * inv, the rows as one sequential burst stream of total = rows * L4 writes.
static void smx_scale_store(Fifo<IoWord>& es, Fifo<float>& ss, int total, int L4, IoWord* ymem) {
  int i = 0;
  float inv = 0.0f;
scale_store:
  for (int t = 0; t < total; t++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 3145728 avg = 3145728
    if (i == 0) inv = ss.read();
    const IoWord e = es.read();
    IoWord o;
    for (int j = 0; j < F4; j++) {
#pragma HLS UNROLL
      io_set(o, j, io_lane(e, j) * inv);
    }
    ymem[t] = o;
    i = (i == L4 - 1) ? 0 : i + 1;
  }
}

static void smx_rows_dataflow(const IoWord* xmem, IoWord* ymem, int rows, int L4, int total) {
#pragma HLS DATAFLOW
  Fifo<IoWord> xs, es, ps;
  Fifo<float> ms, ss;
#pragma HLS STREAM variable = xs depth = SMX_FIFO
#pragma HLS STREAM variable = es depth = SMX_FIFO
#pragma HLS STREAM variable = ps depth = 4 * SMX_PW
#pragma HLS STREAM variable = ms depth = 8
#pragma HLS STREAM variable = ss depth = 8
  smx_read_max(xmem, total, L4, xs, ms);
  smx_exp_sum(xs, ms, rows, L4, es, ps);
  smx_reduce(ps, rows, ss);
  smx_scale_store(es, ss, total, L4, ymem);
}

// Mode 1 entry: softmax over each of `rows` rows of L FP32 values, xmem -> ymem (row-major, 128-bit
// words, rows * L / 4 words each). L a multiple of 4, 1 <= L <= SMX_L_MAX, rows >= 1, rows * L / 4 < 2^31.
static int softmax_rows(const IoWord* xmem, IoWord* ymem, int rows, int L) {
  if (rows < 1 || L < 1 || L > SMX_L_MAX || L % F4) return 1;
  const int L4 = L / F4;
  smx_rows_dataflow(xmem, ymem, rows, L4, rows * L4);
  return 0;
}

}  // namespace fstream
