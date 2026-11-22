// NORM mode (opcode 0x30) of float_top under the integration contract 1.0 (fpga/kernels/CONTRACT.md):
// LayerNorm / RMSNorm of FP32 rows -> int16 rows + per-row scales (F_QOUT, the contract quantizer) or FP32 rows.
// Header only; the same code builds in Vitis HLS and in g++ (-I<Vitis_HLS>/include -ffp-contract=off).
// nr_emul.py is its bit-exact model (tb_nr.cpp checks whole DDR images byte for byte).
//
// Call from the sequencer's dispatch switch (one call site, INLINE off), after ctr::validate_hw(c) == ME_OK:
//     case ctr::OP_NORM: r = nr::norm_mode(c, w2, xmem, ymem); break;
// Ports: rows A are read only on io (xmem); gamma / beta (host constants) on one hp port (read only); every
// write goes out on io (ymem). No pointer argument, no register, no m_axi adapter of its own.
//
// Record (CONTRACT.md 3.3): K = values per row (8..1024, K % 8 == 0), N rows per item, B items;
//   A  FP32 rows   row n of item b at A.addr + b A.stride + n A.pitch
//   E  gamma [K] FP32 (shared by all items), beta [K] at E.addr + 4 K (F_LN)
//   C  F_QOUT: int16 rows (K / 8 words), rows [N, NP) written as zeros; else FP32 rows z (K / 4 words)
//   F  F_QOUT: scales, row n at F.addr + b F.stride + 4 n; pad4(NP) floats, pad lanes 1.0
//   p0 eps (FP32 bits), p4 scale shift (stored s 2^p4, zero rows 1.0), p5 NP (0 = N)
//   flags F_QOUT, F_LN (F_RESID / F_SUM and the records layout are retired: the residual add is the GEMM epilogue)
// Addresses: 64-bit in the record, 40-bit here (validate_hw rejects slot addresses >= 2^40). Row and item offsets
// are sums of pitches / strides in 36-bit word indices, never a 32-bit product.
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
//   q-out    the shared quantizer quantize/quant_int16.h (s = fl32(max(amax, 1e-30) / 32767), zero rows 1.0)
// Every FP32 core operation flushes denormal inputs and results to zero with the sign kept (the Xilinx cores,
// contract 2.3); the g++ build models that with ctr::ref::ftz, so the host C model is the hardware.
//
// DATAFLOW processes (one 128-bit word = 4 FP32 a clock in each flat loop):
//   ---- row-unit common parts: stand-ins for the shared reader / quantizer / writer (CONTRACT.md 4.1, quantize/)
//   p_rd     A rows over io: one burst per item when the row gap is at most half a row (gap words dropped),
//            else one burst per row                                                          -> fa
//   ---- NORM front end: what the row unit takes from this kernel (nr::front_* below, word stream in and out)
//   p_stats  running E, Y_i, S, Q (4 squares a clock); the row goes on to f2 (BRAM row FIFO)  -> f2, fs
//   p_scal   K Q - S^2 (shift-add), then mean and var by two small bit-serial dividers (~160 clocks a LN row,
//            ~125 an RMS row)                                                                -> fv
//   p_rsqrt  ve and rstd in integers (~120 clocks a row)                                     -> fm
//   p_affine gamma | beta into one dual-port BRAM once per command; z = ((y - mean) rstd) g + b -> fz
//   ---- row-unit common parts (stand-ins)
//   p_qamax  row FIFO + per-row absmax                                                       -> fq, fam
//   p_qscal  qz::row_scale (1 reciprocal bit a clock)                                        -> frs
//   p_quant  4 x qz::quant_elem a clock (or FP32 pass-through)                               -> fo, fsc
//   p_wr     C rows (one burst per <= 256-row chunk when C rows are contiguous, else per row), zero rows
//            [N, NP), then the chunk's scales in one burst (pad lanes 1.0)
// Steady state: max(K/4, A words read, C words written) clocks a row, as long as p_scal and p_rsqrt keep up
// (~160 LN / ~125 RMS and ~120 clocks a row): io-bound for LN rows of K >= 640 and RMS rows of K >= 512 (all
// SmolVLA rows: vision LN 768, LM RMS 960, expert RMS 720); shorter rows are scalar-bound.
#pragma once
#include <ap_int.h>
#include <hls_stream.h>
#include <stdint.h>

#include "contract.h"
#include "quant_int16.h"

#ifdef NR_DFSIM
#include "nr_dfsim.h"
#endif

#ifndef NR_FADD_IMPL
#define NR_FADD_IMPL fulldsp  // FP32 add / sub cores of the normalize stage: fulldsp | fabric
#endif
#ifndef NR_FMUL_IMPL
#define NR_FMUL_IMPL fulldsp  // FP32 multipliers of the normalize stage: maxdsp | fulldsp | meddsp | fabric
#endif
#ifndef NR_SQ_IMPL
#define NR_SQ_IMPL dsp  // the 4 exact 25 x 25 squares of the statistics: dsp | fabric
#endif

namespace nr {

typedef ap_uint<128> word_t;
typedef ap_uint<40> baddr_t;  // byte address (validate_hw: < 2^40)
typedef ap_uint<36> waddr_t;  // 128-bit word index = byte address >> 4

constexpr int L = 4;                   // lanes: FP32 values per io word
constexpr int KMAX = ctr::NORM_K_MAX;  // 1024
constexpr int KWMAX = KMAX / L;        // 256 words in the longest row
#ifndef NR_RFD
#define NR_RFD 512
#endif
constexpr int RFD = NR_RFD;  // row FIFOs (words): >= K/4 (one row) or the region deadlocks (p_stats / p_qamax hold
                             // a whole row before its statistic; tb -DNR_DFSIM: 240 hangs at K = 1024, 256 runs);
                             // + the scalar latency (~300 clocks) for full rate
constexpr int SCH = 256;     // writer chunk (rows): its scales are buffered, then written in one burst
constexpr uint32_t FLT_MIN_BITS = 0x00800000u;
constexpr uint32_t INF_BITS = 0x7F800000u;

#ifdef NR_DFSIM
template <class T>
using strm = dfsim::stream<T>;
#define NR_DECL(T, name, D) dfsim::stream<T> name(D, #name)
#define NR_RUN(call) dfsim::run([&] { call; })
#define NR_JOIN() dfsim::join_all()
#else
template <class T>
using strm = hls::stream<T>;
#define NR_DECL(T, name, D) hls::stream<T, D> name
#define NR_RUN(call) call
#define NR_JOIN()
#endif

#ifndef __SYNTHESIS__
extern long g_leftover;  // host only: streams left non-empty at the end of a call (tb_nr.cpp defines it)
#endif

// One command, decoded once by norm_mode (STABLE in the region; every process reads only what it needs).
struct Cfg {
  ap_uint<11> K;
  ap_uint<9> KW;    // K / 4: words of an FP32 row
  ap_uint<9> CW;    // words of an output row: K / 8 (F_QOUT) or K / 4
  ap_uint<17> N;    // rows per item (1..65535)
  ap_uint<17> NP;   // rows written per item (F_QOUT: N..65535; else N)
  ap_uint<17> B;    // items (1..65535)
  ap_uint<25> NKW;  // N * KW: words of an item (< 2^24)
  bool ln, qout;
  ap_uint<32> eps;  // FP32 bits
  ap_int<5> shift;  // p4 (-8..8)
  ap_uint<21> DV;   // divisor of var: K^2 (LN) or K (RMS: var = Q / K)
  // A (io read)
  waddr_t a_w;
  ap_uint<28> a_pw, a_sw;  // pitch, stride in words
  bool a_flat;
  ap_uint<26> a_len;  // flat: (N - 1) a_pw + KW words a burst
  // E (hp read)
  waddr_t e_w;
  // C, F (io write)
  waddr_t c_w, f_w;
  ap_uint<28> c_pw, c_sw, f_sw;
  bool c_flat;
};

struct Stat {
  ap_int<38> S;   // |S| < 2^35 + 2^8
  ap_uint<62> Q;  // < 2^60 + 2^8
  ap_uint<8> emax;
};
struct MV {
  uint32_t mean, var;
};
struct MR {
  float mean, rstd;
};
struct QS {  // per-row quantizer state
  qz::RowScale rs;
  qz::abits_t a;  // row absmax (stored_scale: zero rows 1.0, non-finite rows the absmax)
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
// rnd(x / 2^s) = floor((x + 2^(s-1)) / 2^s) for s >= 1 (round half up), x for s = 0; = ((x >> (s-1)) + 1) >> 1
template <int W>
static inline ap_int<W> rshr_s(ap_int<W> x, ap_uint<9> s) {
#pragma HLS INLINE
  if (s == 0) return x;
  const ap_uint<9> s1 = s - 1;
  const ap_int<W> t = x >> (s1 > W - 1 ? ap_uint<9>(W - 1) : s1);  // >> W-1 of a W-bit signed is 0 or -1
  return (ap_int<W + 1>(t) + 1) >> 1;
}
template <int W>
static inline ap_uint<W> rshr_u(ap_uint<W> x, ap_uint<9> s) {
#pragma HLS INLINE
  if (s == 0) return x;
  const ap_uint<9> s1 = s - 1;
  const ap_uint<W> t = s1 >= W ? ap_uint<W>(0) : ap_uint<W>(x >> s1);
  return ap_uint<W>((ap_uint<W + 1>(t) + 1) >> 1);
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

// ------------------------------------------------------------------------------------------ common: reader
// One read burst of len words from word src + off; keeps the words whose position in a pw-word row is < KW.
static void rd_seg(const word_t* src, waddr_t off, ap_uint<26> len, ap_uint<28> pw, ap_uint<9> KW,
                   strm<word_t>& fo) {
#pragma HLS INLINE off
  ap_uint<28> jr = 0;
rd:
  for (ap_uint<26> j = 0; j < len; j++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 180 max = 131072 avg = 12000
    const word_t w = src[off + j];
    if (jr < KW) fo.write(w);
    jr = jr == pw - 1 ? ap_uint<28>(0) : ap_uint<28>(jr + 1);
  }
}

static void p_rd(const word_t* xmem, Cfg c, strm<word_t>& fa) {
  // flat: one segment per item over (N - 1) pitch + KW words; else N segments of KW words (one per row)
  const ap_uint<17> nseg = c.a_flat ? ap_uint<17>(1) : c.N;
  const ap_uint<26> len = c.a_flat ? c.a_len : ap_uint<26>(c.KW);
  const ap_uint<28> pw = c.a_flat ? c.a_pw : ap_uint<28>(c.KW);
  waddr_t ib = c.a_w;
items:
  for (ap_uint<17> b = 0; b < c.B; b++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 2 avg = 1
    waddr_t rb = ib;
  segs:
    for (ap_uint<17> n = 0; n < nseg; n++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 512 avg = 1
#pragma HLS LOOP_FLATTEN off
      rd_seg(xmem, rb, len, pw, c.KW, fa);
      rb += c.a_pw;
    }
    ib += c.a_sw;
  }
}

// ------------------------------------------------------------------------------------------ NORM front end
// Running exponent + exact block-fixed sums, 4 values a clock; every word goes on to f2 (the row FIFO the affine
// stage reads once the row's mean / rstd are known).
static void p_stats(Cfg c, strm<word_t>& fa, strm<word_t>& f2, strm<Stat>& fs) {
items:
  for (ap_uint<17> b = 0; b < c.B; b++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 2 avg = 1
    ap_uint<8> E = 0;
    ap_int<38> S = 0;
    ap_uint<62> Q = 0;
    ap_uint<9> jr = 0;
  acc:
    for (ap_uint<25> j = 0; j < c.NKW; j++) {
#pragma HLS LOOP_FLATTEN off
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 9000 max = 131072 avg = 20000
      const word_t w = fa.read();
      ap_uint<8> m = 0;
      for (int l = 0; l < L; l++) {
#pragma HLS UNROLL
        const ap_uint<8> e = w.range(32 * l + 30, 32 * l + 23);
        m = e > m ? e : m;
      }
      const bool first = jr == 0;
      const ap_uint<8> E1 = (first || m > E) ? m : E;
      const ap_uint<9> d = first ? ap_uint<9>(0) : ap_uint<9>(E1 - E);
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
      const ap_int<38> S0 = first ? ap_int<38>(0) : rshr_s<38>(S, d);
      const ap_uint<62> Q0 = first ? ap_uint<62>(0) : rshr_u<62>(Q, ap_uint<9>(d << 1));
      const ap_int<38> S1 = S0 + s4;
      const ap_uint<62> Q1 = Q0 + q4;
      S = S1;
      Q = Q1;
      E = E1;
      f2.write(w);
      if (jr == c.KW - 1) {
        Stat st;
        st.S = S1;
        st.Q = Q1;
        st.emax = E1;
        fs.write(st);
        jr = 0;
      } else {
        jr++;
      }
    }
  }
}

// num = max(K Q - S^2, 0) (LN: one signed shift-add accumulator, 37 clocks) or Q (RMS, divided by K instead of
// K^2: the same rational); var = rn(num / DV) 2^(2 k1), mean = rn(|S| / K) 2^k1 (LN): two bit-serial dividers side
// by side, 118 clocks. About 160 clocks a LayerNorm row, 125 an RMSNorm row.
static void p_scal(Cfg c, strm<Stat>& fs, strm<MV>& fv) {
items:
  for (ap_uint<17> b = 0; b < c.B; b++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 2 avg = 1
  rows:
    for (ap_uint<17> r = 0; r < c.N; r++) {
#pragma HLS LOOP_TRIPCOUNT min = 50 max = 512 avg = 100
#pragma HLS LOOP_FLATTEN off
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
      fv.write(o);
    }
  }
}

static void p_rsqrt(Cfg c, strm<MV>& fv, strm<MR>& fm) {
items:
  for (ap_uint<17> b = 0; b < c.B; b++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 2 avg = 1
  rows:
    for (ap_uint<17> r = 0; r < c.N; r++) {
#pragma HLS LOOP_TRIPCOUNT min = 50 max = 512 avg = 100
#pragma HLS LOOP_FLATTEN off
#pragma HLS PIPELINE off
      const MV mv = fv.read();
      uint32_t vb = fadd_pos(mv.var, (uint32_t)c.eps);
      if (vb < FLT_MIN_BITS) vb = FLT_MIN_BITS;  // var = 0 with a flushed (denormal) eps
      MR o;
      o.mean = ctr::u2f(mv.mean);
      o.rstd = vb >= INF_BITS ? 0.0f : ctr::u2f(rsqrt_cr(vb));
      fm.write(o);
    }
  }
}

// gamma | beta from hp into one dual-port BRAM once per command (gamma at [0, KW), beta at [KWMAX, KWMAX + KW)),
// then z = ((y - mean) rstd) g + b, 4 lanes a clock.
// -DNR_KFOLD (area measurement only, NOT contract 1.0: E is ignored): z = (y - mean) rstd, gamma / beta folded into
// the next linear by the packer (README "Area"); removes 2 of the 4 FP32 cores a lane and the table.
static void p_affine(const word_t* hp, Cfg c, strm<word_t>& f2, strm<MR>& fm, strm<word_t>& fz) {
#ifndef NR_KFOLD
  word_t gb[2 * KWMAX];
#ifdef NR_URAM
#pragma HLS BIND_STORAGE variable = gb type = ram_t2p impl = uram
#else
#pragma HLS BIND_STORAGE variable = gb type = ram_t2p impl = bram
#endif
  // one burst: gamma (KW words), then beta (KW words, F_LN) right behind it
  const ap_uint<10> nload = c.ln ? ap_uint<10>(2 * c.KW) : ap_uint<10>(c.KW);
  ap_uint<9> jl = 0;
  ap_uint<1> second = 0;
load_gb:
  for (ap_uint<10> j = 0; j < nload; j++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 180 max = 512 avg = 200
    gb[(second, ap_uint<8>(jl))] = hp[c.e_w + j];
    if (jl == c.KW - 1) {
      jl = 0;
      second = 1;
    } else {
      jl++;
    }
  }
#else
  (void)hp;
#endif
items:
  for (ap_uint<17> b = 0; b < c.B; b++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 2 avg = 1
    MR mr;
    mr.mean = 0.0f;
    mr.rstd = 0.0f;
    ap_uint<8> jr = 0;
  nz:
    for (ap_uint<25> j = 0; j < c.NKW; j++) {
#pragma HLS LOOP_FLATTEN off
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 9000 max = 131072 avg = 20000
      if (jr == 0) mr = fm.read();
      const word_t w = f2.read();
#ifndef NR_KFOLD
      const word_t g = gb[(ap_uint<1>(0), jr)];
      const word_t bb = c.ln ? gb[(ap_uint<1>(1), jr)] : word_t(0);
#endif
      word_t wz;
      for (int l = 0; l < L; l++) {
#pragma HLS UNROLL
        float t = fp_sub(lane_f(w, l), mr.mean);
        t = fp_mul(t, mr.rstd);
#ifndef NR_KFOLD
        t = fp_mul(t, lane_f(g, l));
        t = fp_add(t, lane_f(bb, l));
#endif
        set_lane_f(wz, l, t);
      }
      fz.write(wz);
      jr = jr == c.KW - 1 ? ap_uint<8>(0) : ap_uint<8>(jr + 1);
    }
  }
}

// ------------------------------------------------------------------------------------------ common: quantizer
static void p_qamax(Cfg c, strm<word_t>& fz, strm<word_t>& fq, strm<qz::abits_t>& fam) {
items:
  for (ap_uint<17> b = 0; b < c.B; b++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 2 avg = 1
    qz::abits_t a = 0;
    ap_uint<9> jr = 0;
  w:
    for (ap_uint<25> j = 0; j < c.NKW; j++) {
#pragma HLS LOOP_FLATTEN off
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 9000 max = 131072 avg = 20000
      const word_t w = fz.read();
      const qz::abits_t m = qz::word_amax(w);
      const qz::abits_t a1 = (jr == 0 || m > a) ? m : a;
      a = a1;
      fq.write(w);
      if (jr == c.KW - 1) {
        if (c.qout) fam.write(a1);
        jr = 0;
      } else {
        jr++;
      }
    }
  }
}

static void p_qscal(Cfg c, strm<qz::abits_t>& fam, strm<QS>& frs) {
  if (!c.qout) return;
items:
  for (ap_uint<17> b = 0; b < c.B; b++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 2 avg = 1
  rows:
    for (ap_uint<17> r = 0; r < c.N; r++) {
#pragma HLS LOOP_TRIPCOUNT min = 50 max = 512 avg = 100
#pragma HLS LOOP_FLATTEN off
#pragma HLS PIPELINE off
      const qz::abits_t a = fam.read();
      QS o;
      o.rs = qz::row_scale(a);
      o.a = a;
      frs.write(o);
    }
  }
}

static void p_quant(Cfg c, strm<word_t>& fq, strm<QS>& frs, strm<word_t>& fo, strm<uint32_t>& fsc) {
items:
  for (ap_uint<17> b = 0; b < c.B; b++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 2 avg = 1
    QS qs;
    qs.rs = qz::row_scale_from_s(ap_uint<32>(qz::ONE_BITS));
    qs.a = 0;
    ap_uint<64> lo = 0;
    ap_uint<9> jr = 0;
  q:
    for (ap_uint<25> j = 0; j < c.NKW; j++) {
#pragma HLS LOOP_FLATTEN off
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 9000 max = 131072 avg = 20000
      if (c.qout && jr == 0) qs = frs.read();
      const word_t w = fq.read();
      ap_uint<64> h;
      for (int l = 0; l < L; l++) {
#pragma HLS UNROLL
        h.range(16 * l + 15, 16 * l) = ap_uint<16>(qz::quant_elem(qz::word_f32(w, l), qs.rs));
      }
      if (!c.qout) {
        fo.write(w);
      } else if (jr[0]) {
        word_t o;
        o.range(63, 0) = lo;
        o.range(127, 64) = h;
        fo.write(o);
      } else {
        lo = h;
      }
      if (jr == c.KW - 1) {
        if (c.qout) fsc.write((uint32_t)qz::stored_scale(qs.a, qs.rs.s_bits, c.shift));
        jr = 0;
      } else {
        jr++;
      }
    }
  }
}

// ------------------------------------------------------------------------------------------ common: writer
// Flat: nr rows of CW words from word off; rows >= nv are zero rows (written here, not read from fo). At each row
// end the row's scale goes into sbuf.
static void wr_flat(word_t* io, waddr_t off, ap_uint<9> nr, ap_uint<9> nv, ap_uint<9> CW, bool qout,
                    strm<word_t>& fo, strm<uint32_t>& fsc, uint32_t sbuf[SCH]) {
#pragma HLS INLINE off
  const ap_uint<17> len = nr * CW;
  ap_uint<9> jr = 0, n = 0;
wr:
  for (ap_uint<17> j = 0; j < len; j++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 4500 max = 65536 avg = 20000
    const bool real = n < nv;
    io[off + j] = real ? fo.read() : word_t(0);
    if (jr == CW - 1) {
      if (qout) sbuf[n] = real ? fsc.read() : qz::ONE_BITS;
      n++;
      jr = 0;
    } else {
      jr++;
    }
  }
}

static void wr_row(word_t* io, waddr_t off, ap_uint<9> CW, bool real, strm<word_t>& fo) {
#pragma HLS INLINE off
wr:
  for (ap_uint<9> j = 0; j < CW; j++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 90 max = 256 avg = 120
    io[off + j] = real ? fo.read() : word_t(0);
  }
}

// nw scale words from sbuf (4 a word); lanes at or past nr are padding (1.0)
static void wr_scales(word_t* io, waddr_t off, ap_uint<9> nr, const uint32_t sbuf[SCH]) {
#pragma HLS INLINE off
  const ap_uint<7> nw = (nr + 3) >> 2;
wf:
  for (ap_uint<7> w = 0; w < nw; w++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 13 max = 64 avg = 20
    word_t sw;
    for (int l = 0; l < 4; l++) {
#pragma HLS UNROLL
      const ap_uint<9> i = 4 * w + l;
      sw.range(32 * l + 31, 32 * l) = i < nr ? sbuf[i] : qz::ONE_BITS;
    }
    io[off + w] = sw;
  }
}

static void p_wr(word_t* io, Cfg c, strm<word_t>& fo, strm<uint32_t>& fsc) {
  uint32_t sbuf[SCH];
#pragma HLS ARRAY_PARTITION variable = sbuf cyclic factor = 4
#pragma HLS BIND_STORAGE variable = sbuf type = ram_2p impl = lutram
  waddr_t ic = c.c_w, iF = c.f_w;
items:
  for (ap_uint<17> b = 0; b < c.B; b++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 2 avg = 1
    waddr_t rc = ic, rf = iF;
    ap_uint<17> n0 = 0;
  chunks:
    while (n0 < c.NP) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 2 avg = 1
      const ap_uint<17> left = c.NP - n0;
      const ap_uint<9> nr = left < SCH ? ap_uint<9>(left) : ap_uint<9>(SCH);
      const ap_uint<17> vleft = c.N > n0 ? ap_uint<17>(c.N - n0) : ap_uint<17>(0);
      const ap_uint<9> nv = vleft < nr ? ap_uint<9>(vleft) : nr;  // real (not zero-padding) rows of the chunk
      if (c.c_flat) {
        wr_flat(io, rc, nr, nv, c.CW, c.qout, fo, fsc, sbuf);
        rc += nr * c.CW;
      } else {
      rows:
        for (ap_uint<9> r = 0; r < nr; r++) {
#pragma HLS LOOP_TRIPCOUNT min = 50 max = 256 avg = 100
          const bool real = r < nv;
          wr_row(io, rc, c.CW, real, fo);
          if (c.qout) sbuf[r] = real ? fsc.read() : qz::ONE_BITS;
          rc += c.c_pw;
        }
      }
      if (c.qout) {
        wr_scales(io, rf, nr, sbuf);
        rf += SCH / 4;
      }
      n0 += nr;
    }
    ic += c.c_sw;
    iF += c.f_sw;
  }
}

// ------------------------------------------------------------------------------------------ the region
// The row unit integrates the NORM front end as p_stats -> {p_scal -> p_rsqrt} -> p_affine between its reader and
// its absmax stage (streams: input words in row order, K/4 per row, N rows per item, B items; z words out in the
// same order). Its sizes: f2 (RFD words, BRAM), fs / fv / fm (4 entries), the gamma|beta table (2 KWMAX words).
static void norm_df(const word_t* hp, const word_t* xmem, word_t* ymem, Cfg c) {
#pragma HLS DATAFLOW
#pragma HLS STABLE variable = c
  NR_DECL(word_t, fa, 32);
#pragma HLS BIND_STORAGE variable = fa type = fifo impl = srl
  NR_DECL(word_t, f2, RFD);
  NR_DECL(word_t, fq, RFD);
#ifdef NR_URAM
#pragma HLS BIND_STORAGE variable = f2 type = fifo impl = uram
#else
#pragma HLS BIND_STORAGE variable = f2 type = fifo impl = bram
#endif
#pragma HLS BIND_STORAGE variable = fq type = fifo impl = bram
  NR_DECL(word_t, fz, 32);
#pragma HLS BIND_STORAGE variable = fz type = fifo impl = srl
  NR_DECL(word_t, fo, 32);
#pragma HLS BIND_STORAGE variable = fo type = fifo impl = srl
  NR_DECL(Stat, fs, 4);
  NR_DECL(MV, fv, 4);
  NR_DECL(MR, fm, 4);
  NR_DECL(qz::abits_t, fam, 4);
  NR_DECL(QS, frs, 4);
  NR_DECL(uint32_t, fsc, 8);
  NR_RUN(p_rd(xmem, c, fa));
  NR_RUN(p_stats(c, fa, f2, fs));
  NR_RUN(p_scal(c, fs, fv));
  NR_RUN(p_rsqrt(c, fv, fm));
  NR_RUN(p_affine(hp, c, f2, fm, fz));
  NR_RUN(p_qamax(c, fz, fq, fam));
  NR_RUN(p_qscal(c, fam, frs));
  NR_RUN(p_quant(c, fq, frs, fo, fsc));
  NR_RUN(p_wr(ymem, c, fo, fsc));
  NR_JOIN();
#ifndef __SYNTHESIS__
  g_leftover += !fa.empty() + !f2.empty() + !fq.empty() + !fz.empty() + !fo.empty() + !fs.empty() + !fv.empty() +
                !fm.empty() + !fam.empty() + !frs.empty() + !fsc.empty();
#endif
}

// The mode function: decode the command's fields once, then the DATAFLOW region. Returns ME_OK, or ME_DIMS for a
// command whose counters could not hold it (validate_hw already rejects it; this guard only keeps the region from
// hanging when called without it). hp: the hp port gamma / beta sit behind; xmem / ymem: the io bundle.
static uint32_t norm_mode(const ctr::Cmd& cmd, const word_t* hp, const word_t* xmem, word_t* ymem) {
#pragma HLS INLINE off
  const uint32_t K = cmd.K, N = cmd.N, B = cmd.B;
  const bool qout = cmd.flags & ctr::F_QOUT;
  const uint32_t NP = cmd.p[ctr::P_NP] ? cmd.p[ctr::P_NP] : N;
  if (cmd.op != ctr::OP_NORM || K < 8 || K > (uint32_t)KMAX || (K & 7) || N < 1 || N > ctr::N_MAX || B < 1 ||
      B > ctr::B_MAX || NP < N || NP > ctr::N_MAX || (!qout && NP != N))
    return ctr::ME_DIMS;
  Cfg c;
  c.K = K;
  c.KW = K / 4;
  c.CW = qout ? K / 8 : K / 4;
  c.N = N;
  c.NP = NP;
  c.B = B;
  c.NKW = ap_uint<25>(N) * ap_uint<9>(K / 4);
  c.ln = cmd.flags & ctr::F_LN;
  c.qout = qout;
  c.eps = cmd.p[ctr::P_EPS];
  c.shift = (int)cmd.p[ctr::P_SHIFT];
  c.DV = c.ln ? ap_uint<21>(ap_uint<11>(K) * ap_uint<11>(K)) : ap_uint<21>(K);
  const ctr::Slot& A = cmd.s[ctr::SLOT_A];
  const ctr::Slot& E = cmd.s[ctr::SLOT_E];
  const ctr::Slot& Cs = cmd.s[ctr::SLOT_C];
  const ctr::Slot& F = cmd.s[ctr::SLOT_F];
  c.a_w = baddr_t(A.addr) >> 4;
  c.a_pw = A.pitch >> 4;
  c.a_sw = A.stride >> 4;
  // one burst per item when the gap after each row is at most half a row (the expert's residual rows: 736 of 720)
  c.a_flat = N == 1 || (c.a_pw >= c.KW && 2 * c.a_pw <= 3 * c.KW);
  c.a_len = c.a_flat ? ap_uint<26>(ap_uint<17>(N - 1) * ap_uint<9>(N == 1 ? 0u : (uint32_t)c.a_pw) + c.KW)
                     : ap_uint<26>(0);
  c.e_w = baddr_t(E.addr) >> 4;
  c.c_w = baddr_t(Cs.addr) >> 4;
  c.c_pw = Cs.pitch >> 4;
  c.c_sw = Cs.stride >> 4;
  c.c_flat = NP == 1 || c.c_pw == c.CW;
  c.f_w = baddr_t(F.addr) >> 4;
  c.f_sw = F.stride >> 4;
  norm_df(hp, xmem, ymem, c);
  return ctr::ME_OK;
}

}  // namespace nr
