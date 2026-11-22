// NORM mode (opcode 0x30) of float_top under the integration contract 1.0 (fpga/kernels/CONTRACT.md):
// LayerNorm / RMSNorm of FP32 rows -> int16 rows + per-row scales (F_QOUT, the contract quantizer) or FP32 rows.
// Header only; the same code builds in Vitis HLS and in g++ (-I<Vitis_HLS>/include -ffp-contract=off).
// nr_emul.py is its bit-exact model (tb_nr.cpp checks whole DDR images byte for byte).
//
// What this kernel contributes to the row unit (CONTRACT.md 4.1: "norm_residual: statistics + affine only") is the
// NORM front end, nr::norm_front: a token stream in (the backbone reader's qz::XT tokens, one 128-bit word = 4 FP32
// of a row each) and the same token stream out with every word replaced by its z = norm(y) word. It sits in the
// front-end slot of the row unit's backbone (fpga/kernels/quantize/quant_mode.h), between qz_rd and qz_amx:
//
//     qz_rd -> [nr::norm_front] -> qz_amx -> qz_ss -> qz_sr -> qz_qt -> qz_wb -> qz_wr
//
// For any other opcode (FC.on = false) every token passes through unchanged, so the slot can stay in the path of
// QUANT_X / QUANT_W / QUANT_WT / COPY. The reader, absmax, shared quantizer (quant_int16.h) and writer are the
// backbone's, unchanged: NORM with F_QOUT is QUANT_X rows (p1 = 0) of z, NORM without F_QOUT is COPY rows of z.
//
// nr::norm_mode below is that composition for the NORM opcode alone, the mode function the sequencer's dispatch
// switch calls (one call site, INLINE off) after ctr::validate_hw(c) == ME_OK:
//     case ctr::OP_NORM: r = nr::norm_mode(c, hp, xmem, ymem); break;
// In the merged row unit the backbone's row_plan maps OP_NORM to KX (F_QOUT) or KC (else) and its row_df calls
// norm_front in the slot; then norm_mode is no longer needed (README "Integration").
// Ports: rows A are read only on io (xmem, by qz_rd); gamma / beta (host constants) on one hp port, read only, once
// per command; every write goes out on io (ymem, by qz_wr). No pointer argument, no register, no m_axi adapter.
//
// Record (CONTRACT.md 3.3): K = values per row (8..1024, K % 8 == 0), N rows per item, B items;
//   A  FP32 rows   row n of item b at A.addr + b A.stride + n A.pitch
//   E  gamma [K] FP32 (shared by all items), beta [K] at E.addr + 4 K (F_LN)
//   C  F_QOUT: int16 rows (K / 8 words), rows [N, NP) written as zeros; else FP32 rows z (K / 4 words)
//   F  F_QOUT: scales, row n at F.addr + b F.stride + 4 n; pad4(NP) floats, pad lanes 1.0
//   p0 eps (FP32 bits), p4 scale shift (stored s 2^p4, zero rows 1.0), p5 NP (0 = N)
//   flags F_QOUT, F_LN (F_RESID / F_SUM and the records layout are retired: the residual add is the GEMM epilogue)
// Addresses: 64-bit in the record, 40-bit in the datapath (validate_hw rejects slot addresses >= 2^40; the
// backbone's word indices are 36 bits, rows and items advance by adding pitches and strides).
//
// Arithmetic (nr_emul.py has the same definition, CONTRACT.md 2.2 "NORM front end"):
//   stats    block floating point with a running exponent, one pass: per 4-value word, E = max(E, max exponent
//            field of the word); when E grows by d, S := rnd(S / 2^d), Q := rnd(Q / 2^(2d)) (round half up);
//            Y_i = round_half_away(m_i 2^(1 + e_i - E)) (|Y_i| < 2^25), S += Y_i, Q += Y_i^2 (exact integers).
//            Without a renormalization (E reached by the row's first word, e.g. every constant row) S and Q are
//            the exact sums of the block-fixed values.
//   scalars  mean = rn(S / K) 2^(E-151) (LN, else 0), var = rn(num / K^2) 2^(2 (E-151)), num = max(K Q - S^2, 0)
//            (LN) or K Q (RMS): exact integer products and bit-serial division, correctly rounded, so a constant
//            LayerNorm row gives y - mean = 0 and z = beta. Below 2^-126 -> +0, above the FP32 range -> +inf.
//   rstd     ve = max(fl(var + eps), 2^-126) (an IEEE RNE add in integers, inputs flushed to zero);
//            rstd = rn(1 / sqrt(ve)), correctly rounded (bit-serial 2^76 / M then integer sqrt); 0 if ve = inf
//   z        ((y - mean) * rstd) * g + b, FP32 cores (RMSNorm: mean = 0, b = 0)
//   q-out    the backbone's shared quantizer (quantize/quant_int16.h: s = fl32(max(amax, 1e-30) / 32767), zero rows 1.0)
// Every FP32 core operation flushes denormal inputs and results to zero with the sign kept (the Xilinx cores,
// contract 2.3); the g++ build models that with ctr::ref::ftz, so the host C model is the hardware.
//
// Front-end processes (nested DATAFLOW region; one token = one 128-bit word = 4 FP32 a clock in each flat loop):
//   fe_stats   running E, Y_i, S, Q (4 squares a clock); every token goes on to f2 (BRAM row FIFO)  -> f2, fs
//   fe_scal    K Q - S^2 (shift-add), then mean and var by two bit-serial dividers (~167 clocks a LN row,
//              ~127 an RMS row)                                                                    -> fv
//   fe_rsqrt   ve and rstd in integers (~127 clocks a row)                                         -> fm
//   fe_affine  gamma | beta into one dual-port BRAM once per command; z = ((y - mean) rstd) g + b  -> out
// Steady state: max(K/4, A words read, C words written) clocks a row, as long as fe_scal and fe_rsqrt keep up:
// io-bound for LN rows of K >= 672 and RMS rows of K >= 512 (all SmolVLA rows: vision LN 768, LM RMS 960, expert
// RMS 720); shorter rows are scalar-bound.
#pragma once
#include <ap_int.h>
#include <hls_stream.h>
#include <stdint.h>
#ifdef NR_DFSIM
#include <thread>
#endif

#include "contract.h"
#include "quant_mode.h"  // the row unit's backbone (-I../quantize): qz::RP, qz::XT, qz::row_plan, qz_rd .. qz_wr

#ifndef NR_FADD_IMPL
#define NR_FADD_IMPL fulldsp  // FP32 add / sub cores of the affine stage: fulldsp | fabric
#endif
#ifndef NR_FMUL_IMPL
#define NR_FMUL_IMPL fulldsp  // FP32 multipliers of the affine stage: maxdsp | fulldsp | meddsp | fabric
#endif
#ifndef NR_SQ_IMPL
#define NR_SQ_IMPL dsp  // the 4 exact 25 x 25 squares of the statistics: dsp | fabric
#endif

namespace nr {

typedef qz::word_t word_t;  // ap_uint<128>: the io and hp word
typedef qz::wa_t waddr_t;   // 128-bit word index = byte address >> 4 (36 bits: byte addresses < 2^40)

constexpr int L = 4;                   // lanes: FP32 values per io word
constexpr int KMAX = ctr::NORM_K_MAX;  // 1024
constexpr int KWMAX = KMAX / L;        // 256 words in the longest row
#ifndef NR_RFD
#define NR_RFD 512
#endif
constexpr int RFD = NR_RFD;  // f2 row FIFO (tokens): >= K/4 (one row) or the region deadlocks (fe_stats passes a
                             // whole row before its statistic; tb -DNR_DFSIM: 240 hangs at K = 1024, 256 runs);
                             // + the scalar latency (~300 clocks) for full rate
#ifndef NR_D_XZ
#define NR_D_XZ 16  // norm_front -> qz_amx tokens (as the backbone's rd -> amx)
#endif
constexpr uint32_t FLT_MIN_BITS = 0x00800000u;
constexpr uint32_t INF_BITS = 0x7F800000u;

#ifdef NR_DFSIM  // host deadlock model: every process its own thread, every stream bounded (dfsim/hls_stream.h)
#define NR_RUN(call) dfsim::run([&] { call; })
#define NR_JOIN() dfsim::join_all()
#else
#define NR_RUN(call) call
#define NR_JOIN()
#endif

#ifndef __SYNTHESIS__
extern long g_leftover;  // host only: streams left non-empty at the end of a call (tb_nr.cpp defines it)
#endif

// The front end's view of one command (STABLE in the region; every process reads only what it needs).
struct FC {
  bool on;          // a NORM command; else every token passes through unchanged
  bool ln;          // F_LN
  ap_uint<11> K;    // values per row (8..1024)
  ap_uint<9> KW;    // K / 4: words (tokens) of a row
  ap_uint<21> DV;   // divisor of var: K^2 (LN) or K (RMS: var = Q / K)
  ap_uint<32> eps;  // FP32 bits (p0)
  waddr_t e_w;      // gamma (word index on the hp port); beta at e_w + KW
};

struct Stat {
  ap_int<38> S;   // |S| < 2^35 + 2^8
  ap_uint<62> Q;  // < 2^60 + 2^8
  ap_uint<8> emax;
  bool end;  // the command's last row
};
struct MV {
  uint32_t mean, var;
  bool end;
};
struct MR {
  float mean, rstd;
  bool end;
};

// ------------------------------------------------------------------------------------------ FP32 cores
static inline float fp_mul(float a, float b) {
#pragma HLS INLINE
#ifdef __SYNTHESIS__
  float r = a * b;
#pragma HLS BIND_OP variable = r op = fmul impl = NR_FMUL_IMPL
  return r;
#else
  return ctr::ref::fmul_hw(a, b);
#endif
}
static inline float fp_add(float a, float b) {
#pragma HLS INLINE
#ifdef __SYNTHESIS__
  float r = a + b;
#pragma HLS BIND_OP variable = r op = fadd impl = NR_FADD_IMPL
  return r;
#else
  return ctr::ref::fadd_hw(a, b);
#endif
}
static inline float fp_sub(float a, float b) {
#pragma HLS INLINE
#ifdef __SYNTHESIS__
  float r = a - b;
#pragma HLS BIND_OP variable = r op = fsub impl = NR_FADD_IMPL
  return r;
#else
  return ctr::ref::fadd_hw(a, -b);  // a - b == a + (-b) exactly; ftz(-b) = -ftz(b)
#endif
}

static inline float lane_f(const word_t& w, int l) {
#pragma HLS INLINE
  return ctr::u2f((uint32_t)w.range(32 * l + 31, 32 * l));
}
static inline void set_lane_f(word_t& w, int l, float f) {
#pragma HLS INLINE
  w.range(32 * l + 31, 32 * l) = ctr::f2u(f);
}

// ------------------------------------------------------------------------------------------ statistics
// |Y| of one value against the row's running exponent field e_run (>= its own e), 25 bits
static inline ap_uint<25> yfix_mag(uint32_t b, ap_uint<8> e_run) {
#pragma HLS INLINE
  const ap_uint<8> e = (b >> 23) & 0xFF;
  const ap_uint<24> m = e == 0 ? ap_uint<24>(0) : ap_uint<24>(0x800000u | (b & 0x7FFFFFu));
  // t = sh + 1 = e_run - e in 0..255; w = floor(4 m / 2^t) = floor(m 2^(1 - sh)); |Y| = (w + 1) >> 1 rounds
  // m / 2^sh half away from zero (sh = -1: 2 m, sh = 0: m exactly; t >= 27: 0)
  const ap_uint<8> t = e_run - e;
  const ap_uint<26> m4 = ap_uint<26>(m) << 2;
  const ap_uint<26> w = t > 26 ? ap_uint<26>(0) : ap_uint<26>(m4 >> ap_uint<5>(t));
  return ap_uint<25>((ap_uint<27>(w) + 1) >> 1);
}
// rnd(x / 2^s) + a, rnd = round half up: rnd(x / 2^s) = floor((x + 2^(s-1)) / 2^s) = ((x >> (s-1)) + 1) >> 1 for
// s >= 1, x for s = 0. One add: floor((y + 1) / 2) + a = floor((y + 1 + 2a) / 2) for integers y, a, so with
// y = (2x) >> s (= x >> (s-1) for s >= 1, 2x for s = 0: one shifter) and c = (s >= 1): result = (y + (2a | c)) >> 1.
// clr: the row's first word (x counts as 0). This is the statistics' loop-carried path: one shifter, one adder.
template <int W, int WA>
static inline ap_int<W> racc_s(ap_int<W> x, ap_uint<9> s, ap_int<WA> a, bool clr) {
#pragma HLS INLINE
  const ap_int<W + 1> x2 = ap_int<W + 1>(x) << 1;
  const ap_int<W + 1> y = clr ? ap_int<W + 1>(0) : ap_int<W + 1>(x2 >> (s > W ? ap_uint<9>(W) : s));  // 0 or -1 past W
  const ap_int<W + 2> a2 = (ap_int<W + 2>(a) << 1) | ap_int<W + 2>(!clr && s != 0);
  return ap_int<W>((ap_int<W + 2>(y) + a2) >> 1);
}
template <int W, int WA>
static inline ap_uint<W> racc_u(ap_uint<W> x, ap_uint<10> s, ap_uint<WA> a, bool clr) {
#pragma HLS INLINE
  const ap_uint<W + 1> x2 = ap_uint<W + 1>(x) << 1;
  const ap_uint<W + 1> y = (clr || s > W) ? ap_uint<W + 1>(0) : ap_uint<W + 1>(x2 >> s);
  const ap_uint<W + 2> a2 = (ap_uint<W + 2>(a) << 1) | ap_uint<W + 2>(!clr && s != 0);
  return ap_uint<W>((ap_uint<W + 2>(y) + a2) >> 1);
}

// A pipeline register: identity in C, one clock of latency in hardware (the recipe of Vitis HLS's hls::reg,
// include/utils/x_hls_utils.h). It keeps the statistics' feed-forward adder trees out of the loop-carried cycle.
template <class T>
static T nr_reg(T d) {
#pragma HLS PIPELINE II = 1
#pragma HLS LATENCY min = 1 max = 1
#pragma HLS INLINE off
  return d;
}

// ------------------------------------------------------------------------------------------ integer scalars
// FP32 bits of rn(n / d) 2^k (half to even; biased exponent <= 0 -> +0, >= 255 -> +inf, sign neg), 1 <= d < 2^21.
// Long division with a remainder of the divisor's width: the dividend n (72 bits, MSB first, then zeros) enters
// 1 bit a clock; the quotient bit of step i weighs 2^(72 - i). From the leading quotient bit (step t0) on, 25 more
// are captured; later quotient bits and the final remainder form the sticky. n / d > 2^-21 puts t0 <= 93, so
// 118 steps always finish (no normalization of n). p_scal runs two of these side by side.
constexpr int DIV_STEPS = 118;
template <int RW>
struct Div {
  ap_uint<72> x;
  ap_uint<RW> r;
  ap_uint<26> q;
  ap_uint<5> cnt;
  ap_uint<7> t0;
  bool started, st;
};
template <int RW>
static inline void div_init(Div<RW>& v, ap_uint<72> n) {
#pragma HLS INLINE
  v.x = n;
  v.r = 0;
  v.q = 0;
  v.cnt = 0;
  v.t0 = 0;
  v.started = false;
  v.st = false;
}
template <int RW>
static inline void div_step(Div<RW>& v, ap_uint<RW> d, ap_uint<7> i) {
#pragma HLS INLINE
  const ap_uint<RW + 1> r2 = (ap_uint<RW + 1>(v.r) << 1) | ap_uint<RW + 1>(v.x[71]);
  v.x = v.x << 1;
  const bool qb = r2 >= d;
  v.r = qb ? ap_uint<RW>(r2 - d) : ap_uint<RW>(r2);
  if (!v.started) {
    if (qb) {
      v.started = true;
      v.t0 = i;
      v.q = 1;
      v.cnt = 1;
    }
  } else if (v.cnt < 26) {
    v.q = (v.q << 1) | ap_uint<26>(qb);
    v.cnt++;
  } else {
    v.st = v.st || qb;
  }
}
template <int RW>
static inline uint32_t div_pack(const Div<RW>& v, bool zero, int k, bool neg) {
#pragma HLS INLINE
  if (zero) return 0;
  ap_uint<25> m = v.q >> 2;
  const bool rb = v.q[1];
  const bool st = v.q[0] || v.st || v.r != 0;  // every dividend bit was consumed (118 > 72 steps)
  int ex = 72 - (int)v.t0;                     // n / d in [2^ex, 2^(ex+1))
  if (rb && (st || m[0])) m += 1;
  if (m[24]) {
    m = ap_uint<25>(1) << 23;
    ex += 1;
  }
  const int field = ex + k + 127;
  const uint32_t sign = neg ? 0x80000000u : 0u;
  if (field <= 0) return 0;
  if (field >= 255) return sign | INF_BITS;
  return sign | ((uint32_t)field << 23) | ((uint32_t)m & 0x7FFFFFu);
}
// The whole division (host unit tests; p_scal interleaves two of them in one loop).
template <int RW>
static uint32_t div_rn(ap_uint<72> n, ap_uint<RW> d, int k, bool neg) {
  Div<RW> v;
  div_init(v, n);
  for (int i = 1; i <= DIV_STEPS; i++) div_step(v, d, ap_uint<7>(i));
  return div_pack(v, n == 0, k, neg);
}

// fl(a + b) of two non-negative FP32 (bits), IEEE round half to even, denormal inputs flushed to +0, +inf kept.
// The smaller operand is aligned 1 bit a clock (27 steps, its sticky collected on the way): no barrel shifter.
static inline uint32_t fadd_pos(uint32_t a, uint32_t b) {
#pragma HLS INLINE
  if (((a >> 23) & 0xFF) == 0) a = 0;
  if (((b >> 23) & 0xFF) == 0) b = 0;
  const uint32_t X = a > b ? a : b, Y = a > b ? b : a;
  if (Y == 0 || (X >> 23) >= 255) return X;
  ap_uint<9> ex = X >> 23;
  const ap_uint<27> MX = ap_uint<27>(0x800000u | (X & 0x7FFFFFu)) << 3;
  ap_uint<27> MY = ap_uint<27>(0x800000u | (Y & 0x7FFFFFu)) << 3;
  ap_uint<8> dd = ap_uint<8>(X >> 23) - ap_uint<8>(Y >> 23);
  bool st = false;
align:
  for (int i = 0; i < 27; i++) {
#pragma HLS PIPELINE II = 1
    if (dd != 0) {
      st = st || MY[0];
      MY = MY >> 1;
      dd--;
    }
  }
  ap_uint<28> sum = ap_uint<28>(MX) + ap_uint<28>(MY | ap_uint<27>(st));
  if (sum[27]) {
    sum = (sum >> 1) | ap_uint<28>(sum[0]);
    ex += 1;
  }
  ap_uint<25> m = sum >> 3;
  const bool g = sum[2], rs = sum[1] || sum[0];
  if (g && (rs || m[0])) m += 1;
  if (m[24]) {
    m = ap_uint<25>(1) << 23;
    ex += 1;
  }
  if (ex >= 255) return INF_BITS;
  return ((uint32_t)ex << 23) | ((uint32_t)m & 0x7FFFFFu);
}

// rn(1 / sqrt(v)) of a positive normal FP32 v (bits), correctly rounded. v = M 2^e (M the 24-bit significand);
// M' = M or 2 M so that e' is even; D = floor(2^76 / M') (bit-serial, 54 steps, 26-bit remainder), R = isqrt(D)
// (restoring, 27 steps); R in (2^25.5, 2^26.5] carries 2 or 3 bits below the 24 kept; the sticky is set by a
// non-zero division or square-root remainder. 1 / sqrt(v) = R_exact 2^(-38 - e'/2) is always normal.
static uint32_t rsqrt_cr(uint32_t vb) {
#pragma HLS INLINE
  const int f = (vb >> 23) & 0xFF;
  const int e = f - 150;
  const bool odd = e & 1;
  const ap_uint<25> M2 = odd ? ap_uint<25>(ap_uint<25>(0x800000u | (vb & 0x7FFFFFu)) << 1)
                             : ap_uint<25>(0x800000u | (vb & 0x7FFFFFu));
  const int e2 = odd ? e - 1 : e;
  ap_uint<26> r = ap_uint<26>(1) << 22;  // the dividend's prefix 2^22 < M': quotient bits 76..54 are 0
  ap_uint<54> D = 0;
rq:
  for (int k = 0; k < 54; k++) {
#pragma HLS PIPELINE II = 1
    const ap_uint<27> r2 = ap_uint<27>(r) << 1;
    const bool qb = r2 >= M2;
    r = qb ? ap_uint<26>(r2 - M2) : ap_uint<26>(r2);
    D = (D << 1) | ap_uint<54>(qb);
  }
  const bool st1 = r != 0;
  // restoring square root, 2 bits of D a step (MSB pair first): rem <= 2 R, so rr <= 8 R + 3 < 2^30
  ap_uint<31> rem = 0;
  ap_uint<27> R = 0;
rs:
  for (int i = 0; i < 27; i++) {
#pragma HLS PIPELINE II = 1
    const ap_uint<31> rr = (rem << 2) | ap_uint<31>(D(53, 52));
    D = D << 2;
    const ap_uint<31> trial = (ap_uint<31>(R) << 2) | 1;
    const bool ge = rr >= trial;
    rem = ge ? ap_uint<31>(rr - trial) : rr;
    R = (R << 1) | ap_uint<27>(ge);
  }
  const bool st0 = st1 || rem != 0;
  const bool top = R[26];
  ap_uint<25> m = top ? ap_uint<25>(R >> 3) : ap_uint<25>(R >> 2);
  const bool rb = top ? R[2] : R[1];
  const bool st = st0 || (top ? (R[1] || R[0]) : R[0]);
  int field = (top ? 26 : 25) - 38 - e2 / 2 + 127;
  if (rb && (st || m[0])) m += 1;
  if (m[24]) {
    m = ap_uint<25>(1) << 23;
    field += 1;
  }
  return ((uint32_t)field << 23) | ((uint32_t)m & 0x7FFFFFu);
}

// ------------------------------------------------------------------------------------------ NORM front end
// Running exponent + exact block-fixed sums, 4 values a clock; every token goes on to f2 (the row FIFO the affine
// stage reads once the row's mean / rstd are known). Off (another opcode): tokens pass, no statistics are sent.
static void fe_stats(const FC c, hls::stream<qz::XT>& xs, hls::stream<qz::XT>& f2, hls::stream<Stat>& fs) {
  ap_uint<8> E = 0;
  ap_int<38> S = 0;
  ap_uint<62> Q = 0;
  ap_uint<9> jr = 0;  // word of the row
acc:
  for (;;) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 2 max = 131072 avg = 20000
    const qz::XT t = xs.read();
    const bool dat = t.fwd;  // a row word (the reader's shift-back overlap tokens have fwd = 0: passed, not counted)
    const word_t w = t.w;
    ap_uint<8> m = 0;
    for (int l = 0; l < L; l++) {
#pragma HLS UNROLL
      const ap_uint<8> e = w.range(32 * l + 30, 32 * l + 23);
      m = e > m ? e : m;
    }
    const bool first = dat && jr == 0;
    const ap_uint<8> E1 = (dat && (first || m > E)) ? m : E;
    const ap_uint<9> d = first ? ap_uint<9>(0) : ap_uint<9>(E1 - E);  // 0 for a non-data token
    // sum of +-|Y|: a negative lane adds ~|Y| (its bits inverted) and 1 (counted in nneg): -x = ~x + 1
    ap_int<28> s4 = 0;
    ap_uint<3> nneg = 0;
    ap_uint<52> q4 = 0;
    for (int l = 0; l < L; l++) {
#pragma HLS UNROLL
      const uint32_t bits = (uint32_t)w.range(32 * l + 31, 32 * l);
      const ap_uint<25> mag = yfix_mag(bits, E1);
      const bool neg = bits >> 31;
      const ap_int<27> sm = ap_int<27>(ap_uint<27>(mag) ^ (neg ? ap_uint<27>(0x7FFFFFF) : ap_uint<27>(0)));
      s4 += sm;
      nneg += neg;
      ap_uint<50> sq = mag * mag;
#pragma HLS BIND_OP variable = sq op = mul impl = NR_SQ_IMPL
      q4 += sq;
    }
    s4 += nneg;
    if (!dat) {  // contributes nothing: S1 = S, Q1 = Q (zeroed before the register, off the loop-carried path)
      s4 = 0;
      q4 = 0;
    }
    // the row's sums: the exponent step d, s4 and q4 of this word are registered first (nr_reg), so the S / Q
    // recurrence is one shift and one add a clock
    ap_uint<90> pk = (ap_uint<1>(first), d, ap_uint<28>(s4), q4);
    pk = nr_reg(pk);
    const bool first_r = pk[89];
    const ap_uint<9> d_r = pk(88, 80);
    const ap_int<28> s4_r = ap_int<28>(ap_uint<28>(pk(79, 52)));
    const ap_uint<52> q4_r = pk(51, 0);
    const ap_int<38> S1 = racc_s<38, 28>(S, d_r, s4_r, first_r);
    const ap_uint<62> Q1 = racc_u<62, 52>(Q, ap_uint<10>(ap_uint<10>(d_r) << 1), q4_r, first_r);
    S = S1;
    Q = Q1;
    E = E1;
    f2.write(t);
    if (!dat) {
    } else if (jr == c.KW - 1) {
      if (c.on) {
        Stat st;
        st.S = S1;
        st.Q = Q1;
        st.emax = E1;
        st.end = t.end;
        fs.write(st);
      }
      jr = 0;
    } else {
      jr++;
    }
    if (t.end) break;
  }
}

// num = max(K Q - S^2, 0) (LN: one signed shift-add accumulator, 37 clocks) or Q (RMS, divided by K instead of
// K^2: the same rational); var = rn(num / DV) 2^(2 k1), mean = rn(|S| / K) 2^k1 (LN): two bit-serial dividers side
// by side, 118 clocks. About 167 clocks a LayerNorm row, 127 an RMSNorm row.
static void fe_scal(const FC c, hls::stream<Stat>& fs, hls::stream<MV>& fv) {
  if (!c.on) return;
rows:
  for (;;) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 1024 avg = 100
#pragma HLS PIPELINE off
    const Stat st = fs.read();
    const bool sneg = st.S < 0;
    const ap_uint<37> Sa = sneg ? ap_uint<37>(-st.S) : ap_uint<37>(st.S);
    ap_uint<72> num = st.Q;
    if (c.ln) {
      ap_int<74> acc = 0;
      ap_uint<72> qs = st.Q, ss = Sa;
    mul:
      for (int i = 0; i < 37; i++) {
#pragma HLS PIPELINE II = 1
        const ap_uint<72> pa = (i < 11 && c.K[i < 11 ? i : 0]) ? qs : ap_uint<72>(0);
        const ap_uint<72> pb = Sa[i] ? ss : ap_uint<72>(0);
        acc = acc + ap_int<74>(pa) - ap_int<74>(pb);
        qs = qs << 1;
        ss = ss << 1;
      }
      num = acc < 0 ? ap_uint<72>(0) : ap_uint<72>(acc);
    }
    Div<21> dv;
    Div<11> dm;
    div_init(dv, num);
    div_init(dm, ap_uint<72>(Sa));
  dvd:
    for (int i = 1; i <= DIV_STEPS; i++) {
#pragma HLS PIPELINE II = 1
      div_step(dv, c.DV, ap_uint<7>(i));
      div_step(dm, ap_uint<11>(c.K), ap_uint<7>(i));
    }
    const int k1 = (int)st.emax - 151;
    MV o;
    o.var = div_pack(dv, num == 0, 2 * k1, false);
    o.mean = c.ln ? div_pack(dm, Sa == 0, k1, sneg) : 0u;
    o.end = st.end;
    fv.write(o);
    if (st.end) break;
  }
}

static void fe_rsqrt(const FC c, hls::stream<MV>& fv, hls::stream<MR>& fm) {
  if (!c.on) return;
rows:
  for (;;) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 1024 avg = 100
#pragma HLS PIPELINE off
    const MV mv = fv.read();
    uint32_t vb = fadd_pos(mv.var, (uint32_t)c.eps);
    if (vb < FLT_MIN_BITS) vb = FLT_MIN_BITS;  // var = 0 with a flushed (denormal) eps
    MR o;
    o.mean = ctr::u2f(mv.mean);
    o.rstd = vb >= INF_BITS ? 0.0f : ctr::u2f(rsqrt_cr(vb));
    o.end = mv.end;
    fm.write(o);
    if (mv.end) break;
  }
}

// gamma | beta from hp into one dual-port BRAM once per command (gamma at [0, KW), beta at [KWMAX, KWMAX + KW)),
// then z = ((y - mean) rstd) g + b, 4 lanes a clock; the token's flags pass unchanged.
// -DNR_KFOLD (area measurement only, NOT contract 1.0: E is ignored): z = (y - mean) rstd, gamma / beta folded into
// the next linear by the packer (README "Area"); removes 2 of the 4 FP32 cores a lane and the table.
static void fe_affine(const word_t* hp, const FC c, hls::stream<qz::XT>& f2, hls::stream<MR>& fm,
                      hls::stream<qz::XT>& out) {
#ifndef NR_KFOLD
  word_t gb[2 * KWMAX];
#ifdef NR_URAM
#pragma HLS BIND_STORAGE variable = gb type = ram_t2p impl = uram
#else
#pragma HLS BIND_STORAGE variable = gb type = ram_t2p impl = bram
#endif
  if (c.on) {
    // one burst: gamma (KW words), then beta (KW words, F_LN) right behind it
    const ap_uint<10> nload = c.ln ? ap_uint<10>(2 * c.KW) : ap_uint<10>(c.KW);
    ap_uint<9> jl = 0;
    ap_uint<1> second = 0;
  load_gb:
    for (ap_uint<10> j = 0; j < nload; j++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 2 max = 512 avg = 200
      gb[(second, ap_uint<8>(jl))] = hp[c.e_w + j];
      if (jl == c.KW - 1) {
        jl = 0;
        second = 1;
      } else {
        jl++;
      }
    }
  }
#else
  (void)hp;
#endif
  MR mr;
  mr.mean = 0.0f;
  mr.rstd = 0.0f;
  mr.end = false;
  ap_uint<9> jr = 0;
nz:
  for (;;) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 2 max = 131072 avg = 20000
    qz::XT t = f2.read();
    const bool dat = t.fwd;  // row words only; other tokens pass unchanged
    if (c.on && dat && jr == 0) mr = fm.read();
    if (c.on && dat) {
#ifndef NR_KFOLD
      const word_t g = gb[(ap_uint<1>(0), ap_uint<8>(jr))];
      const word_t bb = c.ln ? gb[(ap_uint<1>(1), ap_uint<8>(jr))] : word_t(0);
#endif
      word_t wz;
      for (int l = 0; l < L; l++) {
#pragma HLS UNROLL
        float v = fp_sub(lane_f(t.w, l), mr.mean);
        v = fp_mul(v, mr.rstd);
#ifndef NR_KFOLD
        v = fp_mul(v, lane_f(g, l));
        v = fp_add(v, lane_f(bb, l));
#endif
        set_lane_f(wz, l, v);
      }
      t.w = wz;
    }
    out.write(t);
    if (dat) jr = jr == c.KW - 1 ? ap_uint<9>(0) : ap_uint<9>(jr + 1);
    if (t.end) break;
  }
}

// The NORM front end: tokens in (the row words of every row in row order, marked fwd = 1, K/4 per row; tokens with
// fwd = 0, e.g. the reader's shift-back overlap, pass unchanged and are not counted; the command's last token
// flagged end), the same tokens out with z words. One process of the row unit's DATAFLOW region (a nested region). Sizes: f2 (RFD tokens,
// BRAM), fs / fv / fm (4 entries each), the gamma | beta table (2 KWMAX words, BRAM).
static void norm_front(const word_t* hp, const FC c, hls::stream<qz::XT>& xs, hls::stream<qz::XT>& out) {
#pragma HLS DATAFLOW
#pragma HLS STABLE variable = c
  hls::stream<qz::XT> f2("nr_f2");
  hls::stream<Stat> fs("nr_fs");
  hls::stream<MV> fv("nr_fv");
  hls::stream<MR> fm("nr_fm");
#pragma HLS STREAM variable = f2 depth = RFD
#pragma HLS STREAM variable = fs depth = 4
#pragma HLS STREAM variable = fv depth = 4
#pragma HLS STREAM variable = fm depth = 4
#ifdef NR_URAM
#pragma HLS BIND_STORAGE variable = f2 type = fifo impl = uram
#else
#pragma HLS BIND_STORAGE variable = f2 type = fifo impl = bram
#endif
  NR_RUN(fe_stats(c, xs, f2, fs));
  NR_RUN(fe_scal(c, fs, fv));
  NR_RUN(fe_rsqrt(c, fv, fm));
  NR_RUN(fe_affine(hp, c, f2, fm, out));
  NR_JOIN();
#ifndef __SYNTHESIS__
  g_leftover += !f2.empty() + !fs.empty() + !fv.empty() + !fm.empty();
#endif
}

// The front end's command view. Any opcode: on = (OP_NORM); assumes validate_hw(c) == ME_OK for a NORM.
static inline FC front_plan(const ctr::Cmd& c) {
#pragma HLS INLINE
  FC f;
  const uint32_t K = c.K & 0x7FF;
  f.on = c.op == ctr::OP_NORM;
  f.ln = (c.flags & ctr::F_LN) != 0;
  f.K = K;
  f.KW = K >> 2;
  f.DV = f.ln ? ap_uint<21>(ap_uint<11>(K) * ap_uint<11>(K)) : ap_uint<21>(K);
  f.eps = c.p[ctr::P_EPS];
  f.e_w = waddr_t(c.s[ctr::SLOT_E].addr >> 4);
  return f;
}

// ------------------------------------------------------------------------------------------ the NORM mode
// The row unit's backbone with the NORM front end in its slot, for the NORM opcode only (the standalone mode; the
// stream names, depths and storage are the backbone's row_df, quantize/quant_mode.h).
static void norm_df(const word_t* hp, const word_t* xmem, word_t* ymem, const qz::RP p, const FC f) {
#pragma HLS DATAFLOW
#pragma HLS STABLE variable = p
#pragma HLS STABLE variable = f
  hls::stream<qz::XT> xs("qz_xs");
  hls::stream<qz::XT> xz("nr_xz");
  hls::stream<qz::AM> am("qz_am");
  hls::stream<word_t> d("qz_d");
  hls::stream<qz::SS> ss("qz_ss");
  hls::stream<qz::ST> sf("qz_sf");
  hls::stream<word_t> ow("qz_ow");
  hls::stream<ap_uint<32>> sq("qz_sq");
  hls::stream<qz::UH> hd("qz_hd");
  hls::stream<word_t> dd("qz_dd");
#pragma HLS STREAM variable = xs depth = QZ_D_XS
#pragma HLS STREAM variable = xz depth = NR_D_XZ
#pragma HLS STREAM variable = am depth = QZ_D_AM
#pragma HLS STREAM variable = d depth = QZ_D_D
#pragma HLS STREAM variable = ss depth = QZ_D_SS
#pragma HLS STREAM variable = sf depth = QZ_D_SF
#pragma HLS STREAM variable = ow depth = QZ_D_OW
#pragma HLS STREAM variable = sq depth = QZ_D_SQ
#pragma HLS STREAM variable = hd depth = QZ_D_HD
#pragma HLS STREAM variable = dd depth = QZ_D_DD
#pragma HLS BIND_STORAGE variable = d type = fifo impl = bram
#pragma HLS BIND_STORAGE variable = am type = fifo impl = lutram
#pragma HLS BIND_STORAGE variable = sq type = fifo impl = lutram
#pragma HLS BIND_STORAGE variable = ow type = fifo impl = lutram
#pragma HLS BIND_STORAGE variable = xs type = fifo impl = lutram
#pragma HLS BIND_STORAGE variable = xz type = fifo impl = lutram
#pragma HLS BIND_STORAGE variable = dd type = fifo impl = lutram
  NR_RUN(qz::qz_rd(xmem, p, xs));
  NR_RUN(norm_front(hp, f, xs, xz));
  NR_RUN(qz::qz_amx(p, xz, am, d));
  NR_RUN(qz::qz_ss(p, am, ss));
  NR_RUN(qz::qz_sr(p, ss, sf));
  NR_RUN(qz::qz_qt(p, d, sf, ow, sq));
  NR_RUN(qz::qz_wb(p, ow, sq, hd, dd));
  NR_RUN(qz::qz_wr(ymem, p, hd, dd));
  NR_JOIN();
#ifndef __SYNTHESIS__
  g_leftover += !xs.empty() + !xz.empty() + !am.empty() + !d.empty() + !ss.empty() + !sf.empty() + !ow.empty() +
                !sq.empty() + !hd.empty() + !dd.empty();
#endif
}

// The mode function. Returns ME_OK, or ME_DIMS without touching memory for a command its counters could not hold
// (validate_hw already rejects every such record; this guard only keeps the region from hanging when called
// without it). hp: the hp port gamma / beta sit behind (its pointer register is 0: absolute addresses); xmem /
// ymem: the io bundle. Returns after the last write response (R-ORD): the region ends when qz_wr's last write is
// acknowledged.
static uint32_t norm_mode(const ctr::Cmd& cmd, const word_t* hp, const word_t* xmem, word_t* ymem) {
#pragma HLS INLINE off
  const uint32_t K = cmd.K, N = cmd.N, B = cmd.B;
  const bool qout = cmd.flags & ctr::F_QOUT;
  const uint32_t NP = cmd.p[ctr::P_NP] ? cmd.p[ctr::P_NP] : N;
  if (cmd.op != ctr::OP_NORM || K < 8 || K > (uint32_t)KMAX || (K & 7) || N < 1 || N > ctr::N_MAX || B < 1 ||
      B > ctr::B_MAX || NP < N || NP > ctr::N_MAX || (!qout && NP != N) || cmd.p[ctr::P_SEG])
    return ctr::ME_DIMS;
  // the backbone's plan: NORM with F_QOUT = QUANT_X rows (one scale per row), without = COPY rows
  ctr::Cmd b = cmd;
  b.op = qout ? ctr::OP_QUANT_X : ctr::OP_COPY;
  qz::RP p;
  if (qz::row_plan(b, p) != ctr::ME_OK) return ctr::ME_DIMS;
  norm_df(hp, xmem, ymem, p, front_plan(cmd));
  return ctr::ME_OK;
}

}  // namespace nr
