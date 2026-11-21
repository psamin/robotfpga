// gelu_silu: element-wise activation + fused per-token int16 quantisation, header-only HLS core.
//   mode 0  y = gelu_tanh(x)          SigLIP fc1 output -> fc2 input                 (OP_GELU)
//   mode 1  y = silu(gate) * up       LM / expert gate|up output -> down_proj input  (OP_SILU_MUL)
//   mode 2  y = silu(x)               action-time MLP (F.silu) -> action_time_mlp_out (OP_SILU_MUL + F_NOMUL)
//   out     q[n][k] int16, s[n] FP32 with q * s ~= y, |q| <= 32767, s ~= absmax_n(|y|) / 32767
// Inputs are FP32 rows (the engine's output, row pitch in 16-byte words) read over four 128-bit m_axi ports
// (float_top's hp0..hp3); q rows and the scale vector are written over one 128-bit m_axi port (float_top's io).
// Everything is integer arithmetic; gs_emu.py is the bit-exact torch model (tb_gelu_silu.cpp checks bit equality).
//
// GS_LANES = 8 (default). Each row is read as two contiguous halves (hp0: x[0, K/2), hp1: x[K/2, K); mode 1 also
// hp2/hp3 for up), lanes 0-3 take the first half, 4-7 the second. Output word c of a row is therefore
//   [q[4c], q[4c+1], q[4c+2], q[4c+3], q[K/2+4c], ..., q[K/2+4c+3]]      (16 bits each, element 0 in bits 15:0)
// a fixed permutation of k (gs_emu.out_perm). The next linear's sums are exact integers, so permuting the
// columns of its weights the same way offline (or the rows of the previous linear by the inverse) gives
// bit-identical results with no reorder hardware. README.md "Element order".
// GS_LANES = 4: x on hp0 (up on hp1), 4 consecutive elements per cycle, q rows in natural order, half the
// arithmetic, twice the time.
//
// Two passes per row through an on-chip row buffer (the 'ys' FIFO, one row + slack): pass 1 computes y and the
// row's absmax, a sequential unit turns absmax into the scale, pass 2 quantises the buffered row. DATAFLOW
// overlaps pass 1 of row n + 1 with pass 2 of row n, so every element costs one read of x (and up) and one 16-bit
// write, at 8 per cycle.
//
// Arithmetic (per element; x = (sx, ex, mx) with the hidden bit, ex = 0 -> 0):
//   g(t) = 1 - h(t), t = |x|, h = Phi_tanh (mode 0, range [0, 8)) or sigmoid (modes 1, 2, range [0, 16)), g = 0
//   beyond the range
//   T = floor(t * 2^(TB - RB)) (TB = NSB + 12, RB = 3 | 4); idx = T >> 12, f = T & 4095
//   g21 = max(0, V[idx] - ((D[idx] * f + 2048) >> 12))            Q0.21, ROM word V << 15 | D (gs_tables.py)
//   x >= 0: g18 = min((g21 + 4) >> 3, 2^17 - 1) (Q0.18), Y = (mx << 18) - mx * g18 = |x| (1 - g) 2^(168 - ex)
//   x < 0:  gn = g21 << pz rounded to 17 bits (pz = leading zeros of g21, leading one at bit 16),
//           Y = mx * gn = |x| g 2^(167 + pz - ex)               one DSP (24 x 17) either way; the leading one
//           of Y is at bit 41, 40 or 39, so normalising is a 3-way mux
//   modes 0, 2: y = Y normalised to 24 bits, round half up (FP32, flush to 0)
//   mode 1: Y normalised to 17 bits (round half up), times the 24-bit significand of up on one DSP (24 x 17),
//           normalised to 24 bits, round half up, saturates at FP32 max
//   amax = max |y| over the row (FP32 bit compare)
//   rq = floor(32767 * 2^25 / ma) (17 bits), r = rq * 2^(125 - ea) <= 32767 / amax   (ma, ea of amax)
//   s = 1 / r correctly rounded to FP32 (so q * s is consistent with the r actually used); amax < 2^-111: q = s = 0
//   q = round_half_up(|y| * r) with sign: my * rq (24 x 17, one DSP) >> (ea - ey + 25)
#pragma once
#include <ap_int.h>
#include <hls_stream.h>
#include <stdint.h>

#ifndef GS_NSB
#define GS_NSB 9  // log2 segments per mode: 9 -> GELU 1/64, SiLU 1/32 steps; one BRAM36 per lane pair
#endif
#ifndef GS_KMAX
#define GS_KMAX 3072  // longest row (elements): SigLIP fc1 3072; LM 2560; expert 2048; action-time 720
#endif
#ifndef GS_LANES
#define GS_LANES 8  // 8: rows read as two halves (hp0/hp1, up hp2/hp3), permuted output; 4: natural order, x on hp0, up on hp1
#endif
#if GS_LANES != 8 && GS_LANES != 4
#error "GS_LANES must be 4 or 8"
#endif

#include "gelu_silu_tables.h"

#define GS_CAT2(a, b) a##b
#define GS_CAT(a, b) GS_CAT2(a, b)
#define GS_TAB_INIT GS_CAT(GS_TAB_INIT_, GS_NSB)

namespace gs {

typedef ap_uint<128> Word;
typedef ap_uint<36> woff_t;  // 16-byte word offset (40-bit byte addresses)
constexpr int NSB = GS_NSB;
constexpr int NSEG = 1 << NSB;
constexpr int FB = 12;
constexpr int TB = NSB + FB;
constexpr int VB = 21, DB = 15;
constexpr int LANES = GS_LANES;
constexpr int KMAX = GS_KMAX;
// The row buffer holds a whole row of y plus what pass 1 writes of the next row while the scale is computed
// (~70 cycles). It must exceed KMAX / LANES entries or a row could never complete (deadlock).
// 8 lanes: 512 x 256 bits, 4 lanes: 896 x 128 bits; 4 BRAM36 either way.
constexpr int YDEPTH = KMAX / LANES + 128;
constexpr int QMAX = 32767;

enum { MODE_GELU = 0, MODE_SILU_MUL = 1, MODE_SILU = 2 };

struct YW {  // 8 lanes of y (FP32 bits)
  ap_uint<32> v[LANES];
};
struct RowScale {
  ap_uint<17> rq;
  ap_uint<8> ea;
  bool zero;
  ap_uint<32> s_bits;
};

// ------------------------------------------------------------------ port word access
// Under HLS float_top's WPort and IoWord are ap_uint<128> (= Word). A host build with other 16-byte word types
// (float_linear.h's structs) adds overloads of these two.
static inline Word to_word(const Word& w) {
#pragma HLS INLINE
  return w;
}
static inline void from_word(Word& d, const Word& w) {
#pragma HLS INLINE
  d = w;
}

// ------------------------------------------------------------------ table (one ROM copy per lane pair)
template <int PAIR>
static inline ap_uint<36> tab(ap_uint<NSB + 1> a) {
#pragma HLS INLINE
  static const ap_uint<36> T[2 * NSEG] = GS_TAB_INIT;
#pragma HLS BIND_STORAGE variable = T type = rom_2p impl = bram
  return T[a];
}

// ------------------------------------------------------------------ small fixed-width helpers
// Right shift of a W-bit value by a 5-bit amount as a 5-level mux tree (HLS builds a 32-bit shifter otherwise).
template <int W>
static inline ap_uint<W> rsh5(ap_uint<W> v, ap_uint<5> s) {
#pragma HLS INLINE
  ap_uint<W> r = v;
  if (s[0]) r = r >> 1;
  if (s[1]) r = r >> 2;
  if (s[2]) r = r >> 4;
  if (s[3]) r = r >> 8;
  if (s[4]) r = r >> 16;
  return r;
}
// Leading one of a 21-bit value to bit 20 in 5 stages; pz = the shift (31 for 0, then the value stays 0).
static inline ap_uint<21> norm21(ap_uint<21> v, ap_uint<5>& pz) {
#pragma HLS INLINE
  ap_uint<5> c = 0;
  if (v(20, 5) == 0) {
    v = v << 16;
    c[4] = 1;
  }
  if (v(20, 13) == 0) {
    v = v << 8;
    c[3] = 1;
  }
  if (v(20, 17) == 0) {
    v = v << 4;
    c[2] = 1;
  }
  if (v(20, 19) == 0) {
    v = v << 2;
    c[1] = 1;
  }
  if (v[20] == 0) {
    v = v << 1;
    c[0] = 1;
  }
  pz = c;
  return v;
}

// ------------------------------------------------------------------ pass 1: y bits of one element
// tsilu: SiLU table (modes 1, 2); gmul: times up (mode 1). Round half up of v at bit k is written
// ((v >> k) + 1) >> 1 (= (v + 2^k) >> (k + 1)) so the adders are only as wide as the kept bits.
template <int PAIR>
static inline ap_uint<32> act1(ap_uint<32> xb, ap_uint<32> ub, bool tsilu, bool gmul) {
#pragma HLS INLINE
  const bool sx = xb[31];
  const ap_uint<8> ex = xb(30, 23);
  const bool xz = (ex == 0);
  const ap_uint<24> mx = xz ? ap_uint<24>(0) : (ap_uint<24>)((ap_uint<24>(1) << 23) | (ap_uint<24>)xb(22, 0));
  // t -> fixed point. sh = 150 + RB - TB - ex; in range for sh in [24 - TB, 23], saturated below.
  const ap_int<10> sh = (ap_int<10>)(150 + (tsilu ? 4 : 3) - TB) - (ap_int<10>)ex;
  const bool sat = !xz && sh < 24 - TB;
  const bool inr = !xz && sh >= 24 - TB && sh < 24;
  const ap_uint<5> shu = (ap_uint<5>)sh;
  const ap_uint<TB> T = inr ? (ap_uint<TB>)rsh5<24>(mx, shu) : (ap_uint<TB>)0;
  const ap_uint<NSB> idx = T >> FB;
  const ap_uint<FB> f = T(FB - 1, 0);
  const ap_uint<NSB + 1> addr = ((ap_uint<NSB + 1>)(tsilu ? 1 : 0) << NSB) | idx;
  const ap_uint<36> w = tab<PAIR>(addr);
  const ap_uint<VB> V = w(35, DB);
  const ap_uint<DB> D = w(DB - 1, 0);
  ap_uint<DB + FB> df = D * f;
#ifdef GS_DF_DSP
#pragma HLS BIND_OP variable = df op = mul impl = dsp
#else
#pragma HLS BIND_OP variable = df op = mul impl = fabric
#endif
  const ap_uint<DB> corr = (ap_uint<DB>)(((ap_uint<DB + 2>)(df >> (FB - 1)) + 1) >> 1);  // (df + 2048) >> 12
  const ap_int<VB + 1> gd = (ap_int<VB + 1>)V - (ap_int<VB + 1>)corr;
  const ap_uint<VB> g21 = (sat || gd < 0) ? (ap_uint<VB>)0 : (ap_uint<VB>)gd;
  // x >= 0: |x| (1 - g), g in Q0.18 (17 bits; g < 0.5, so g18 = 2^17 only next to x = 0, clamped)
  const ap_uint<18> g18r = (ap_uint<18>)(((ap_uint<VB>)(g21 >> 2) + 1) >> 1);  // (g21 + 4) >> 3
  const ap_uint<17> g18 = g18r[17] ? (ap_uint<17>)0x1FFFF : (ap_uint<17>)g18r;
  // x < 0: |x| g, g21 normalised to a 17-bit significand (keeps the relative precision of small g)
  ap_uint<5> pzc;
  const ap_uint<VB> g21n = norm21(g21, pzc);
  const ap_uint<18> gnr = (ap_uint<18>)(((ap_uint<19>)(g21n >> 3) + 1) >> 1);  // (g21n + 8) >> 4, [2^16, 2^17]
  ap_uint<17> gn = gnr(16, 0);
  ap_int<7> pz = (ap_int<7>)pzc;
  if (gnr[17]) {
    gn = ap_uint<17>(1) << 16;
    pz = pz - 1;
  }
  const ap_uint<17> G = sx ? gn : g18;
  ap_uint<41> P = mx * G;  // one DSP (24 x 17)
#pragma HLS BIND_OP variable = P op = mul impl = dsp
  // x >= 0: Y in [2^40, 2^42); x < 0: Y in [2^39, 2^41) or 0. So the leading one is at bit 41, 40 or 39, and
  // Yt = (Y << lz) >> 17 (the 25 bits rounding needs) is a 3-way select.
  const ap_uint<42> Y = sx ? (ap_uint<42>)P : (ap_uint<42>)(((ap_uint<42>)mx << 18) - P);
  const bool zY = !(Y[41] || Y[40] || Y[39]);  // Y == 0
  const ap_uint<2> lz = Y[41] ? 0 : (Y[40] ? 1 : 2);
  const ap_uint<25> Yt = Y[41] ? (ap_uint<25>)Y(41, 17) : (Y[40] ? (ap_uint<25>)Y(40, 16) : (ap_uint<25>)Y(39, 15));
  const ap_int<11> e0 = (ap_int<11>)ex - (ap_int<11>)lz + (sx ? (ap_int<11>)(1 - pz) : (ap_int<11>)0);
  if (!gmul) {
    const ap_uint<25> m25 = (ap_uint<25>)(((ap_uint<26>)Yt + 1) >> 1);  // (Yn + 2^17) >> 18
    ap_int<11> ey = e0;
    ap_uint<24> m = m25(23, 0);
    if (m25[24]) {
      m = ap_uint<24>(1) << 23;
      ey = ey + 1;
    }
    ap_uint<32> r = 0;
    if (!(zY || ey <= 0)) r = ((ap_uint<32>)sx << 31) | ((ap_uint<32>)(ap_uint<8>)ey << 23) | (ap_uint<32>)m(22, 0);
    return r;
  }
  // mode 1: x sigmoid(x) rounded to 17 significant bits, times the 24-bit significand of up (one DSP, 24 x 17)
  const ap_uint<18> m18 = (ap_uint<18>)(((ap_uint<19>)(Yt >> 7) + 1) >> 1);  // (Yn + 2^24) >> 25, [2^16, 2^17]
  ap_uint<17> m17 = m18(16, 0);
  ap_int<11> e17 = e0;
  if (m18[17]) {
    m17 = ap_uint<17>(1) << 16;
    e17 = e17 + 1;
  }
  const bool z1 = zY || e17 <= 0;
  const bool su = ub[31];
  const ap_uint<8> eu = ub(30, 23);
  const ap_uint<24> mu = (ap_uint<24>)((ap_uint<24>(1) << 23) | (ap_uint<24>)ub(22, 0));
  ap_uint<41> P3 = mu * m17;  // [2^39, 2^41)
#pragma HLS BIND_OP variable = P3 op = mul impl = dsp
  const bool top = P3[40];
  const ap_uint<25> Ps = top ? (ap_uint<25>)P3(40, 16) : (ap_uint<25>)P3(39, 15);
  const ap_uint<25> m3r = (ap_uint<25>)(((ap_uint<26>)Ps + 1) >> 1);  // top: (P3 + 2^16) >> 17, else >> 16
  ap_int<11> e3 = e17 + (ap_int<11>)eu - 127 + (top ? 1 : 0);
  ap_uint<24> m3 = m3r(23, 0);
  if (m3r[24]) {
    m3 = ap_uint<24>(1) << 23;
    e3 = e3 + 1;
  }
  if (e3 >= 255) {
    e3 = 254;
    m3 = ~ap_uint<24>(0);
  }
  const bool z3 = z1 || eu == 0 || e3 <= 0;
  ap_uint<32> r = 0;
  if (!z3) r = ((ap_uint<32>)(sx ^ su) << 31) | ((ap_uint<32>)(ap_uint<8>)e3 << 23) | (ap_uint<32>)m3(22, 0);
  return r;
}

// ------------------------------------------------------------------ per-row scale (sequential, ~65 cycles/row)
static void row_scale(ap_uint<31> amax, RowScale& rs) {
#pragma HLS INLINE
  const ap_uint<8> ea = amax(30, 23);
  const ap_uint<24> ma = (ap_uint<24>)((ap_uint<24>(1) << 23) | (ap_uint<24>)amax(22, 0));
  const bool zero = ea < 16;
  // rq = floor(32767 * 2^25 / ma): the top 23 bits of the dividend are < ma, so 17 quotient steps
  const ap_uint<40> num = (ap_uint<40>)QMAX << 25;
  ap_uint<25> rem = num(39, 17);
  ap_uint<17> rq = 0;
div1:
  for (int i = 16; i >= 0; i--) {
#pragma HLS PIPELINE II = 1
    rem = (rem << 1) | (ap_uint<25>)num[i];
    const bool ge = rem >= ma;
    if (ge) rem -= ma;
    rq[i] = ge;
  }
  // q25 = floor(2^41 / rq), 26 bits
  ap_uint<18> rem2 = 0;
  ap_uint<26> q25 = 0;
div2:
  for (int i = 41; i >= 0; i--) {
#pragma HLS PIPELINE II = 1
    rem2 = (rem2 << 1) | (ap_uint<18>)(i == 41 ? 1 : 0);
    const bool ge = rem2 >= rq;
    if (ge) rem2 -= rq;
    if (i < 26) q25[i] = ge;
  }
  const bool big = q25[25];
  ap_uint<25> sm = big ? (ap_uint<25>)((q25 + 2) >> 2) : (ap_uint<25>)((q25 + 1) >> 1);
  ap_uint<9> es = (ap_uint<9>)ea - 16 + (big ? 2 : 1);
  if (sm[24]) {
    sm = ap_uint<25>(1) << 23;
    es = es + 1;
  }
  rs.rq = rq;
  rs.ea = ea;
  rs.zero = zero;
  rs.s_bits = zero ? (ap_uint<32>)0 : (ap_uint<32>)(((ap_uint<32>)(ap_uint<8>)es << 23) | (ap_uint<32>)sm(22, 0));
}

// ------------------------------------------------------------------ pass 2: q of one element
static inline ap_int<16> quant1(ap_uint<32> yb, const RowScale& rs) {
#pragma HLS INLINE
  const bool sy = yb[31];
  const ap_uint<8> ey = yb(30, 23);
  const ap_uint<24> my = (ap_uint<24>)((ap_uint<24>(1) << 23) | (ap_uint<24>)yb(22, 0));
  ap_uint<41> P2 = my * rs.rq;
#pragma HLS BIND_OP variable = P2 op = mul impl = dsp
  const ap_uint<9> d = rs.ea - ey;  // >= 0: amax >= |y|
  // q = ((P2 >> (d + 24)) + 1) >> 1, zero when d > 16; P2 <= 32767 * 2^25, so q <= 32767
  const ap_uint<17> t = (ap_uint<17>)(P2 >> 24);
  const ap_uint<17> ts = (d > 16) ? (ap_uint<17>)0 : (ap_uint<17>)(t >> (ap_uint<5>)d);
  ap_uint<16> qm = (ap_uint<16>)((ts + 1) >> 1);
  if (qm > QMAX) qm = QMAX;
  if (ey == 0 || rs.zero) qm = 0;
  return sy ? (ap_int<16>)(-(ap_int<17>)qm) : (ap_int<16>)qm;
}

static inline ap_uint<31> umax31(ap_uint<31> a, ap_uint<31> b) {
#pragma HLS INLINE
  return a > b ? a : b;
}

// ------------------------------------------------------------------ dataflow processes
// pass 1 reads the halves of each row straight from the ports (one burst per port per row).
// Word offsets: row n of x starts at a + n * pa (hp0 first half, hp1 second half), of up at b + n * pb.
template <class RP>
static void gs_pass1(const RP* p0, const RP* p1, const RP* p2, const RP* p3, hls::stream<YW>& ys,
                     hls::stream<ap_uint<31> >& am, int mode, int N, int K, woff_t a, woff_t pa, woff_t b,
                     woff_t pb) {
  const bool tsilu = (mode != MODE_GELU);
  const bool gmul = (mode == MODE_SILU_MUL);
  const int wh = K / 8;  // words per half row
  woff_t ra = a, rb = b;
rows:
  for (int n = 0; n < N; n++) {
#pragma HLS LOOP_TRIPCOUNT min = 50 max = 512
    ap_uint<31> mxl[LANES];
#pragma HLS ARRAY_PARTITION variable = mxl complete
    for (int j = 0; j < LANES; j++) {
#pragma HLS UNROLL
      mxl[j] = 0;
    }
#if GS_LANES == 8
  words:
    for (int w = 0; w < wh; w++) {
#pragma HLS LOOP_TRIPCOUNT min = 90 max = 384
#pragma HLS PIPELINE II = 1
      const Word xa = to_word(p0[ra + w]);
      const Word xb = to_word(p1[ra + wh + w]);
      Word ua = 0, ub = 0;
      if (gmul) {
        ua = to_word(p2[rb + w]);
        ub = to_word(p3[rb + wh + w]);
      }
      YW o;
      o.v[0] = act1<0>(xa(31, 0), ua(31, 0), tsilu, gmul);
      o.v[1] = act1<0>(xa(63, 32), ua(63, 32), tsilu, gmul);
      o.v[2] = act1<1>(xa(95, 64), ua(95, 64), tsilu, gmul);
      o.v[3] = act1<1>(xa(127, 96), ua(127, 96), tsilu, gmul);
      o.v[4] = act1<2>(xb(31, 0), ub(31, 0), tsilu, gmul);
      o.v[5] = act1<2>(xb(63, 32), ub(63, 32), tsilu, gmul);
      o.v[6] = act1<3>(xb(95, 64), ub(95, 64), tsilu, gmul);
      o.v[7] = act1<3>(xb(127, 96), ub(127, 96), tsilu, gmul);
      for (int j = 0; j < LANES; j++) {
#pragma HLS UNROLL
        mxl[j] = umax31(mxl[j], o.v[j](30, 0));
      }
      ys.write(o);
    }
    const ap_uint<31> m01 = umax31(mxl[0], mxl[1]), m23 = umax31(mxl[2], mxl[3]);
    const ap_uint<31> m45 = umax31(mxl[4], mxl[5]), m67 = umax31(mxl[6], mxl[7]);
    am.write(umax31(umax31(m01, m23), umax31(m45, m67)));
#else
  words:
    for (int w = 0; w < 2 * wh; w++) {
#pragma HLS LOOP_TRIPCOUNT min = 180 max = 768
#pragma HLS PIPELINE II = 1
      const Word xa = to_word(p0[ra + w]);
      Word ua = 0;
      if (gmul) ua = to_word(p1[rb + w]);
      YW o;
      o.v[0] = act1<0>(xa(31, 0), ua(31, 0), tsilu, gmul);
      o.v[1] = act1<0>(xa(63, 32), ua(63, 32), tsilu, gmul);
      o.v[2] = act1<1>(xa(95, 64), ua(95, 64), tsilu, gmul);
      o.v[3] = act1<1>(xa(127, 96), ua(127, 96), tsilu, gmul);
      for (int j = 0; j < LANES; j++) {
#pragma HLS UNROLL
        mxl[j] = umax31(mxl[j], o.v[j](30, 0));
      }
      ys.write(o);
    }
    am.write(umax31(umax31(mxl[0], mxl[1]), umax31(mxl[2], mxl[3])));
#endif
    ra += pa;
    rb += pb;
  }
}

// One shared, sequential divider pair: ~65 cycles per row, overlapped with pass 1 of the next row.
static void gs_scale(hls::stream<ap_uint<31> >& am, hls::stream<RowScale>& rss, int N) {
rows:
  for (int n = 0; n < N; n++) {
#pragma HLS LOOP_TRIPCOUNT min = 50 max = 512
#pragma HLS PIPELINE off
    RowScale rs;
    row_scale(am.read(), rs);
    rss.write(rs);
  }
}

// pass 2 writes q row n at c + n * pc (K/8 words) and the N scales, 4 per word (last word zero-padded), at f.
template <class WP>
static void gs_pass2(hls::stream<YW>& ys, hls::stream<RowScale>& rss, WP* ymem, int N, int K, woff_t c, woff_t pc,
                     woff_t f) {
#if GS_LANES == 4
// Meant to share the 4 quantiser DSPs over the II = 2 loop; Vitis HLS 2022.2 still binds 8 (csynth, README).
#pragma HLS ALLOCATION operation instances = mul limit = 4
#endif
  const int wh = K / 8;
  woff_t rc = c, rf = f;
  ap_uint<32> s0 = 0, s1 = 0, s2 = 0;  // scales of rows 4i, 4i + 1, 4i + 2 (lane j of a scale word = row 4i + j)
rows:
  for (int n = 0; n < N; n++) {
#pragma HLS LOOP_TRIPCOUNT min = 50 max = 512
    const RowScale rs = rss.read();
  words:
    for (int w = 0; w < wh; w++) {
#pragma HLS LOOP_TRIPCOUNT min = 90 max = 384
#if GS_LANES == 8
#pragma HLS PIPELINE II = 1
      const YW y = ys.read();
      Word o;
      for (int j = 0; j < LANES; j++) {
#pragma HLS UNROLL
        o(16 * j + 15, 16 * j) = (ap_uint<16>)quant1(y.v[j], rs);
      }
#else
      // two 4-lane entries per output word: elements 8w .. 8w + 7 in order; 4 quantisers shared over the 2 cycles
#pragma HLS PIPELINE II = 2
      const YW y0 = ys.read();
      const YW y1 = ys.read();
      Word o;
      for (int j = 0; j < 4; j++) {
#pragma HLS UNROLL
        o(16 * j + 15, 16 * j) = (ap_uint<16>)quant1(y0.v[j], rs);
        o(16 * j + 79, 16 * j + 64) = (ap_uint<16>)quant1(y1.v[j], rs);
      }
#endif
      from_word(ymem[rc + w], o);
    }
    rc += pc;
    const ap_uint<2> j = n & 3;
    const ap_uint<32> sb = rs.s_bits;
    if (j == 3 || n == N - 1) {  // the word of rows 4i .. 4i + j (lanes past the last row 0)
      Word sw;
      sw(31, 0) = (j == 0) ? sb : s0;
      sw(63, 32) = (j == 1) ? sb : ((j > 1) ? s1 : ap_uint<32>(0));
      sw(95, 64) = (j == 2) ? sb : ((j > 2) ? s2 : ap_uint<32>(0));
      sw(127, 96) = (j == 3) ? sb : ap_uint<32>(0);
      from_word(ymem[rf], sw);
      rf++;
    }
    if (j == 0) s0 = sb;
    if (j == 1) s1 = sb;
    if (j == 2) s2 = sb;
  }
}

template <class RP, class WP>
static void gelu_silu_df(const RP* p0, const RP* p1, const RP* p2, const RP* p3, WP* ymem, int mode, int N, int K,
                         woff_t a, woff_t pa, woff_t b, woff_t pb, woff_t c, woff_t pc, woff_t f) {
#pragma HLS DATAFLOW
  hls::stream<YW> ys("gs_ys");  // the buffered row: pass 2 starts a row once pass 1 has finished it (absmax known)
#pragma HLS STREAM variable = ys depth = YDEPTH
#ifdef GS_YS_URAM
#pragma HLS BIND_STORAGE variable = ys type = fifo impl = uram
#else
#pragma HLS BIND_STORAGE variable = ys type = fifo impl = bram
#endif
  hls::stream<ap_uint<31> > am("gs_am");
#pragma HLS STREAM variable = am depth = 4
  hls::stream<RowScale> rss("gs_rss");
#pragma HLS STREAM variable = rss depth = 4
  gs_pass1(p0, p1, p2, p3, ys, am, mode, N, K, a, pa, b, pb);
  gs_scale(am, rss, N);
  gs_pass2(ys, rss, ymem, N, K, c, pc, f);
}

// The kernel. p0..p3: read ports (float_top hp0..hp3), ymem: write port (float_top io). All offsets and pitches
// in 16-byte words relative to the port pointers (in float_top's sequencer mode the pointers are 0 and these are
// absolute address / 16):
//   a, pa  x rows (mode 1: gate rows): row n at a + n * pa, K FP32 contiguous     (sequencer slot A)
//   b, pb  mode 1 up rows: row n at b + n * pb                                     (slot B; unused otherwise)
//   c, pc  q rows: row n at c + n * pc, K int16 in the order of out_perm (natural for 4 lanes)   (slot C)
//   f      scales: N FP32, 4 per word, ceil(N / 4) words (the unused tail of the last word is written 0) (slot F)
// Returns 0, or 1 (nothing read or written) for a bad shape: K % 8, K > KMAX, N < 1, mode > 2.
template <class RP, class WP>
static int gelu_silu_core(const RP* p0, const RP* p1, const RP* p2, const RP* p3, WP* ymem, int mode, int N, int K,
                          woff_t a, woff_t pa, woff_t b, woff_t pb, woff_t c, woff_t pc, woff_t f) {
  if (K <= 0 || K > KMAX || (K % 8) != 0 || N <= 0 || mode < 0 || mode > 2) return 1;
  gelu_silu_df(p0, p1, p2, p3, ymem, mode, N, K, a, pa, b, pb, c, pc, f);
  return 0;
}

}  // namespace gs
