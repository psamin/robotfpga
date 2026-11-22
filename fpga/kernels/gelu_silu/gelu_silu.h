// gelu_silu: the GELU / SILU_MUL modes of float_top (fpga/kernels/CONTRACT.md 1.0: 1.1, 2.1, 2.2, 3.3, 4.1).
//
//   OP_GELU              y = tanh-GELU(x)            SigLIP fc1 -> fc2
//   OP_SILU_MUL          y = silu(gate) * up         LM / expert gate|up -> down_proj
//   OP_SILU_MUL F_NOMUL  y = silu(x)                 action-time MLP (F.silu)
//   F_QOUT: int16 rows to C + one FP32 scale per row to F, the contract quantizer (quantize/quant_int16.h) applied
//   to y; else FP32 rows of y to C.
//
// Mode function (one call site in the sequencer's dispatch switch; INLINE off, so one hardware instance):
//   case ctr::OP_GELU: case ctr::OP_SILU_MUL: return gs::act_mode(c, io_rd, io_wr);
// or, with the sequencer's unit signature (seq_core.h "Units"), gs::act_unit(c, hp0, hp1, hp2, hp3, io_rd, io_wr).
// c is a decoded record that passed ctr::validate_hw (the sequencer checks before every dispatch); io_rd / io_wr
// are float_top's io bundle (xmem read, ymem write). In list mode the pointers are 0, so word index = byte address
// >> 4. Activations are read only on io (CONTRACT.md 3.3); the hp ports are not used. The mode returns after its
// last write (HLS waits for every write response of a burst before the loop that issued it ends: R-ORD).
//
// Slots (CONTRACT.md 3.3), all addresses 64-bit, < 2^40, 16-byte aligned (validate_hw):
//   A  x (SILU_MUL: gate) FP32 rows: item b row n at A + b A.stride + n A.pitch, K values
//   B  SILU_MUL up rows (same form; absent with F_NOMUL and for GELU)
//   C  output rows: K int16 (F_QOUT) or K FP32, row n of item b at C + b C.stride + n C.pitch
//   F  F_QOUT: the scale vector of item b at F + b F.stride, 4 n bytes for row n; pad4(NP) floats, rows [N, NP) and
//      the pad lanes hold 1.0
//   p4 stored scale = s 2^p4 (an all-zero row stores 1.0); p5 = NP rows written (0 = N), rows [N, NP) all zero
//
// Processes (one DATAFLOW region per command; 4 lanes = 4 FP32 a clock, the io port's width):
//   gs_rd   io reader. Dense (the -12392 list: every GELU / SILU_MUL command): one burst stream per item covering
//           rows [0, N), the pitch gaps read and dropped (A.pitch - row <= GAPW words; SILU_MUL needs the fused
//           gate|up rows: B = A + 4K, B.pitch = A.pitch). Else one burst per row (A row, then B row).
//   gs_fe   THE FRONT END (this kernel's own logic, what the integrated row unit takes): 4 lanes of h(x) from a
//           table and one FP32 multiply per lane. SILU_MUL buffers silu(gate) of the row (gbuf) and multiplies it
//           by up with the same multiplier. Emits y in natural column order.
//   gs_am   per-row absmax of y; y into the row buffer (ys)
//   gs_sc   absmax -> s, R (quant_int16.h, 1 recip bit a clock), stored scale (quant_int16.h stored_scale, p4)
//   gs_qt   4 lanes of quant_int16.h quant_elem (or y passed through for FP32 output)
//   gs_wr   io writer. Dense (C.pitch = the row): one burst per chunk of up to WCH rows, then that chunk's scale
//           words; else one burst per row. Zero rows [N, NP), pad lanes 1.0
//   gs_am .. gs_wr and gs_rd are the row unit's shared backbone (CONTRACT.md 4.1: reader, row buffer + absmax,
//   quantizer, writer); here a private copy, so that the mode runs and synthesises standalone.
//
// Front-end interface (what a shared row unit plugs in, gs_fe): GsCmd fields mode, k4 = K / 4, up0, rowits, n, nb;
// input stream: per row k4 words of x (GELU, SILU), or k4 words of gate then k4 words of up (SILU_MUL), rows in
// order; output stream: per row k4 words of y. Throughput 1 word a clock (SILU_MUL: 1 output word per 2 input
// words, the reader's pace). gbuf holds one row of silu(gate): K <= ACT_K_MAX.
//
// Front-end arithmetic (gs_emu.py is the bit-exact torch model; tb_gelu_silu.cpp checks it bit for bit):
//   h = Phi_tanh (GELU, range R = 8) or sigmoid (SiLU, R = 16); the table holds g(t) = 1 - h(t), t = |x|
//   T = floor(t 2^(TB - rb)) (TB = NSB + 12, rb = 3 | 4): idx = T >> 12, f = T & 4095, T = 0 for |x| < 2^-126
//   g21 = max(0, (V[idx] << 12) + 2048 - D[idx] f) >> 12      Q0.21, 0 for t >= R; ROM word V << 15 | D
//   H = x < 0 ? g21 : 2^21 - g21, h = H 2^-21 (exact FP32)
//   y = fmul(x, h); SILU_MUL: y = fmul(up, y_gate). fmul = the FP32 core: RNE, denormals in and out flushed to 0
//   with the sign kept (contract.ftz); the C model applies the flushes explicitly, the core does them in RTL.
// Limits: g has an absolute resolution of 2^-21, so the negative tails lose relative precision: SiLU gives -0 for
// x <= -15.25 (true value > -4e-6; relative error 4.9e-4 at x = -8), GELU -0 for x <= -8 (true value > -1e-14).
// A SILU_MUL row dominated by a gate below -15.25 times a large up therefore quantizes to the wrong absmax (on the
// -12392 activations: at most 0.037 step, no policy effect; CONTRACT.md 9 #23). NaN / Inf inputs: q is undefined
// (finite garbage) but the row's stored scale is non-finite (stored_scale), so the divergence reaches the next
// GEMM; never a hang or a write outside the declared extents.
#pragma once
#include <ap_int.h>
#include <hls_stream.h>
#include <stdint.h>
#include <string.h>
#ifdef GS_DFSIM
#include <thread>
#endif

#include "../contract.h"
#include "../quantize/quant_int16.h"

#ifndef GS_GUARD
#define GS_GUARD 1  // act_mode's own range check (a duplicate of ctr::validate_hw, see act_mode)
#endif
#ifndef GS_NSB
#define GS_NSB 9  // log2 segments per function: GELU step 1/64, SiLU 1/32; one BRAM36 per lane pair
#endif
#include "gelu_silu_tables.h"

#define GS_CAT2(a, b) a##b
#define GS_CAT(a, b) GS_CAT2(a, b)
#define GS_TAB_INIT GS_CAT(GS_TAB_INIT_, GS_NSB)

#if QZ_RECIP_BITS != 1
#error "CONTRACT.md 2.1: QZ_RECIP_BITS = 1 everywhere"
#endif

// stream depths (named: the threaded deadlock model in tb_gelu_silu.cpp uses the same numbers)
#ifndef GS_D_XIN
#define GS_D_XIN 4  // rd -> fe (shallow: the io adapter buffers the bursts)
#endif
#ifndef GS_D_YF
#define GS_D_YF 4  // fe -> am
#endif
#ifndef GS_D_YS
#define GS_D_YS (ctr::ACT_K_MAX / 4 + 128)  // am -> qt, the row buffer: one row + the scale latency
#endif
#ifndef GS_D_AM
#define GS_D_AM 4  // am -> sc
#endif
#ifndef GS_D_RSF
#define GS_D_RSF 4  // sc -> qt
#endif
#ifndef GS_D_SST
#define GS_D_SST 8  // sc -> wr stored scales
#endif
#ifndef GS_D_OW
#define GS_D_OW 8  // qt -> wr (covers the writer's per-run gap)
#endif

namespace gs {

typedef ap_uint<128> Word;
typedef ap_uint<36> waddr_t;   // 16-byte word index of a byte address < 2^40 (ctr::ADDR_LIMIT)
typedef ap_uint<28> wpitch_t;  // a u32 byte pitch / stride in words
constexpr int NSB = GS_NSB;
constexpr int NSEG = 1 << NSB;
constexpr int FB = 12;
constexpr int TB = NSB + FB;
constexpr int VB = 21, DB = 15;
constexpr int LANES = 4;
constexpr int KMAX = ctr::ACT_K_MAX;  // 3072 values per row
constexpr int K4MAX = KMAX / LANES;   // 768 words per row
// The row buffer must hold one whole row or a row could never complete (deadlock): K > KMAX is rejected.
#ifndef GS_DFSIM_NEGATIVE  // the deadlock model's negative test builds a too-small row buffer on purpose
static_assert(GS_D_YS > K4MAX, "the row buffer must exceed one row");
#endif
// SILU_MUL: iterations between the write of gbuf[w] (gate word w, at the end of the gs_fe pipeline) and its read (up
// word w, near the start). Must exceed the pipeline depth (csynth: 21). Rows with K / 4 < GAP get GAP - K / 4 idle
// iterations (K < 128 only; none in SmolVLA).
constexpr int GAP = 32;
// Dense read: pitch gaps up to GAPW words (1 KB) per row are read and dropped (-12392's action-time SiLU input has
// a 4-word gap: pitch 4 pad32(720)).
constexpr int GAPW = 64;
// Writer chunk: rows per data burst; the chunk's scale words follow it (the scale buffer holds WCH scales).
constexpr int WCH = 64;
constexpr uint32_t ONE_BITS = 0x3F800000u;

enum { MODE_GELU = 0, MODE_SILU_MUL = 1, MODE_SILU = 2 };

// ------------------------------------------------------------------ io word conversion
// float_top's IoWord is ap_uint<128> under HLS; a host build may use a 16-byte struct (float_linear.h).
static inline Word to_word(const Word& w) {
#pragma HLS INLINE
  return w;
}
static inline void from_word(Word& d, const Word& w) {
#pragma HLS INLINE
  d = w;
}
#ifndef __SYNTHESIS__
template <class W>
static inline Word to_word(const W& w) {
  static_assert(sizeof(W) == 16, "io words are 128 bits");
  uint32_t u[4];
  memcpy(u, &w, 16);
  Word r = 0;
  for (int j = 0; j < 4; j++) r(32 * j + 31, 32 * j) = u[j];
  return r;
}
template <class W>
static inline void from_word(W& d, const Word& w) {
  static_assert(sizeof(W) == 16, "io words are 128 bits");
  uint32_t u[4];
  for (int j = 0; j < 4; j++) u[j] = (uint32_t)w(32 * j + 31, 32 * j);
  memcpy(&d, u, 16);
}
#endif

// ------------------------------------------------------------------ the command, as the processes use it
struct GsCmd {
  ap_uint<2> mode;
  bool qout;
  bool rdense;          // gs_rd: one run per item (rlen words, keep rw of every rper)
  bool wdense;          // gs_wr: one data burst per chunk (C.pitch = orw)
  ap_uint<10> k4;       // words of x per row (K / 4), 2..768
  ap_uint<11> rw;       // input words per row: k4, or 2 k4 (SILU_MUL: gate | up)
  ap_uint<11> rper;     // rdense: words between row starts (A.pitch; rw for N = 1)
  ap_uint<28> rlen;     // rdense: words per item, (N - 1) rper + rw
  ap_uint<11> rowits;   // gs_fe iterations per row: k4, or 2 k4 + gap (SILU_MUL)
  ap_uint<11> up0;      // SILU_MUL: first up iteration of a row (k4 + gap)
  ap_uint<10> orw;      // output words per row: k4 / 2 (int16) or k4 (FP32)
  ap_uint<16> n, np, nb;  // rows per item, rows written, items
  ap_int<5> shift;      // p4
  waddr_t a, b, c, f;   // item 0 row 0
  wpitch_t pa, sa, pb, sb, pc, sc, sf;
};

// ------------------------------------------------------------------ table (one ROM copy per lane pair)
template <int PAIR>
static inline ap_uint<36> tab(ap_uint<NSB + 1> a) {
#pragma HLS INLINE
  static const ap_uint<36> T[2 * NSEG] = GS_TAB_INIT;
#pragma HLS BIND_STORAGE variable = T type = rom_2p impl = bram
  return T[a];
}

// Right shift of a W-bit value by a 5-bit amount as a 5-level mux tree.
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

// FP32 bits of h(x) = H 2^-21 (see the header comment).
template <int PAIR>
static inline ap_uint<32> h_bits(ap_uint<32> xb, bool tsilu) {
#pragma HLS INLINE
  const bool sx = xb[31];
  const ap_uint<8> ex = xb(30, 23);
  const bool xz = ex == 0;
  const ap_uint<24> mx = (ap_uint<1>(1), ap_uint<23>(xb(22, 0)));
  // sh = 150 + rb - TB - ex: T in range for sh in [24 - TB, 23], saturated (t >= R) below, 0 above
  const ap_int<10> sh = (ap_int<10>)(150 + (tsilu ? 4 : 3) - TB) - (ap_int<10>)ex;
  const bool sat = !xz && sh < 24 - TB;
  const bool inr = !xz && sh >= 24 - TB && sh < 24;
#ifdef GS_SHIFT_LUT
  const ap_uint<TB> T = inr ? (ap_uint<TB>)rsh5<24>(mx, (ap_uint<5>)sh) : (ap_uint<TB>)0;
#else
  // T = mx >> sh = (mx << k) >> 23 with k = 23 - sh in [0, 23 - (24 - TB)] (<= 21): the shift on a DSP,
  // P = mx 2^(k mod 16), then T = k >= 16 ? P >> 7 : P >> 23 (exact; saves the 5-level mux tree)
  const ap_uint<5> k = (ap_uint<5>)(23 - sh);
  const ap_uint<16> oh = ap_uint<16>(1) << k(3, 0);
  const ap_uint<40> P = mx * oh;
#pragma HLS BIND_OP variable = P op = mul impl = dsp
  const ap_uint<TB> Ts = k[4] ? (ap_uint<TB>)(P >> 7) : (ap_uint<TB>)(P >> 23);
  const ap_uint<TB> T = inr ? Ts : (ap_uint<TB>)0;
#endif
  const ap_uint<NSB> idx = T >> FB;
  const ap_uint<FB> f = T(FB - 1, 0);
  const ap_uint<36> w = tab<PAIR>(((ap_uint<NSB + 1>)(tsilu ? 1 : 0) << NSB) | idx);
  const ap_uint<VB> V = w(35, DB);
  const ap_uint<DB> D = w(DB - 1, 0);
  // one DSP: (V << 12 | 2048) - D f (the low 12 bits of V << 12 are 0, so + 2048 is an OR)
  const ap_uint<DB + FB> df = D * f;
#pragma HLS BIND_OP variable = df op = mul impl = dsp
  const ap_int<VB + FB + 1> t = (ap_int<VB + FB + 1>)(((ap_uint<VB + FB>)V << FB) | (ap_uint<VB + FB>)(1 << (FB - 1))) -
                                (ap_int<VB + FB + 1>)df;
  const ap_uint<VB> g21 = (sat || t < 0) ? (ap_uint<VB>)0 : (ap_uint<VB>)(t >> FB);
  const ap_uint<VB + 1> H = sx ? (ap_uint<VB + 1>)g21 : (ap_uint<VB + 1>)((ap_uint<VB + 1>(1) << VB) - g21);
  // normalise H (22 bits, <= 2^21): leading one to bit 21 in 5 select stages, pz = the shift (31 for H = 0)
  const bool z4 = H(21, 6) == 0;
  const ap_uint<22> v4 = z4 ? (ap_uint<22>)(H << 16) : H;
  const bool z3 = v4(21, 14) == 0;
  const ap_uint<22> v3 = z3 ? (ap_uint<22>)(v4 << 8) : v4;
  const bool z2 = v3(21, 18) == 0;
  const ap_uint<22> v2 = z2 ? (ap_uint<22>)(v3 << 4) : v3;
  const bool z1 = v2(21, 20) == 0;
  const ap_uint<22> v1 = z1 ? (ap_uint<22>)(v2 << 2) : v2;
  const bool z0 = !v1[21];
  const ap_uint<22> v = z0 ? (ap_uint<22>)(v1 << 1) : v1;
  // h = 1.v[20:0] 2^-pz: biased exponent 127 - pz = 0b011 ~pz (pz <= 31), no adder; v[21] == 0 iff H == 0
  const ap_uint<5> npz = (ap_uint<1>(!z4), ap_uint<1>(!z3), ap_uint<1>(!z2), ap_uint<1>(!z1), ap_uint<1>(!z0));
  const ap_uint<32> hb = (ap_uint<1>(0), ap_uint<3>(3), npz, ap_uint<21>(v(20, 0)), ap_uint<2>(0));
  return v[21] ? hb : (ap_uint<32>)0;
}

static inline float bits_f(ap_uint<32> b) {
#pragma HLS INLINE
  const uint32_t u = (uint32_t)b;
  float f;
  memcpy(&f, &u, 4);
  return f;
}
static inline ap_uint<32> f_bits(float f) {
#pragma HLS INLINE
  uint32_t u;
  memcpy(&u, &f, 4);
  return u;
}
#ifndef __SYNTHESIS__
static inline float ftzf(float f) {  // contract.ftz: what the FP32 core does to its inputs and its result
  uint32_t u;
  memcpy(&u, &f, 4);
  if (((u >> 23) & 0xFF) == 0) u &= 0x80000000u;
  memcpy(&f, &u, 4);
  return f;
}
#endif
static inline ap_uint<32> fmul_bits(ap_uint<32> a, ap_uint<32> b) {
#pragma HLS INLINE
#ifdef __SYNTHESIS__
  const float y = bits_f(a) * bits_f(b);
#else
  const float y = ftzf(ftzf(bits_f(a)) * ftzf(bits_f(b)));
#endif
  return f_bits(y);
}

// ------------------------------------------------------------------ gs_rd: io reader
// Dense: one run of rlen words per item, keeping the first rw words of every rper. Sparse: per row a run of k4
// words of A (+ one of B for SILU_MUL).
template <class RM>
static void gs_rd(const GsCmd& c, RM xmem, hls::stream<Word>& xin) {
  const bool mul = c.mode == MODE_SILU_MUL;
  waddr_t ia = c.a, ib = c.b, ra = c.a, rb = c.b;
  ap_uint<16> n = 0, bi = 0;
  bool second = false;  // sparse SILU_MUL: the B run of the current row is next
runs:
  for (;;) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 1024 avg = 1
    const waddr_t base = second ? rb : ra;
    const ap_uint<28> len = c.rdense ? c.rlen : (ap_uint<28>)c.k4;
    const ap_uint<11> per = c.rdense ? c.rper : (ap_uint<11>)c.k4;
    const ap_uint<11> keep = c.rdense ? c.rw : (ap_uint<11>)c.k4;
    ap_uint<11> col = 0;
  data:
    for (ap_uint<28> i = 0; i < len; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 2 max = 786432 avg = 196608
      const Word w = to_word(xmem[base + i]);
      if (col < keep) xin.write(w);
      col = col == per - 1 ? (ap_uint<11>)0 : (ap_uint<11>)(col + 1);
    }
    bool item_end;
    if (c.rdense) {
      item_end = true;
    } else if (mul && !second) {
      second = true;
      item_end = false;
    } else {
      second = false;
      ra += c.pa;
      rb += c.pb;
      item_end = n == c.n - 1;
      n = item_end ? ap_uint<16>(0) : ap_uint<16>(n + 1);
    }
    if (item_end) {
      if (bi == c.nb - 1) break;
      bi++;
      ia += c.sa;
      ib += c.sb;
      ra = ia;
      rb = ib;
    }
  }
}

// ------------------------------------------------------------------ gs_fe: the GELU / SiLU front end
// One multiplier per lane: y = fmul(x, B) with B = h(x) (GELU, SILU, the gate words of SILU_MUL) or B = the
// buffered silu(gate) (the up words of SILU_MUL: y = up * silu(gate), the same product as silu(gate) * up).
static void gs_fe(const GsCmd& c, hls::stream<Word>& xin, hls::stream<Word>& yf) {
  Word gbuf[K4MAX];  // SILU_MUL: silu(gate) of the current row
#ifdef GS_GBUF_URAM
#pragma HLS BIND_STORAGE variable = gbuf type = ram_s2p impl = uram
#else
#pragma HLS BIND_STORAGE variable = gbuf type = ram_s2p impl = bram
#endif
#pragma HLS DEPENDENCE variable = gbuf type = inter dependent = false
  const bool tsilu = c.mode != MODE_GELU;
  const bool mul = c.mode == MODE_SILU_MUL;
  ap_uint<11> it = 0;
  ap_uint<16> n = 0, bi = 0;
words:
  for (;;) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 2 max = 786432 avg = 196608
    const bool gate = mul && it < c.k4;
    const bool up = mul && it >= c.up0;
    const bool out = !mul || up;
    const ap_uint<10> wi = up ? (ap_uint<10>)(it - c.up0) : (ap_uint<10>)it;
    Word xw;  // gap iterations: undefined and unused (no mux)
    if (gate || out) xw = xin.read();
    const Word gw = gbuf[wi];
    Word yw;
    for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
      const ap_uint<32> xb = xw(32 * l + 31, 32 * l);
      const ap_uint<32> hb = l < 2 ? h_bits<0>(xb, tsilu) : h_bits<1>(xb, tsilu);
      const ap_uint<32> ob = up ? (ap_uint<32>)gw(32 * l + 31, 32 * l) : hb;
      yw(32 * l + 31, 32 * l) = fmul_bits(xb, ob);
    }
    if (gate) gbuf[wi] = yw;
    if (out) yf.write(yw);
    if (it == c.rowits - 1) {
      it = 0;
      if (n == c.n - 1) {
        n = 0;
        if (bi == c.nb - 1) break;
        bi++;
      } else {
        n++;
      }
    } else {
      it++;
    }
  }
}

// ------------------------------------------------------------------ gs_am: row absmax + the row buffer
static void gs_am(const GsCmd& c, hls::stream<Word>& yf, hls::stream<Word>& ys, hls::stream<qz::abits_t>& am) {
  ap_uint<10> w = 0;
  ap_uint<16> n = 0, bi = 0;
  qz::abits_t m = 0;
words:
  for (;;) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 2 max = 786432 avg = 196608
    const Word y = yf.read();
    ys.write(y);
    const qz::abits_t mr = qz::amax2(m, qz::word_amax(y));
    if (w == c.k4 - 1) {
      am.write(mr);
      m = 0;
      w = 0;
      if (n == c.n - 1) {
        n = 0;
        if (bi == c.nb - 1) break;
        bi++;
      } else {
        n++;
      }
    } else {
      m = mr;
      w++;
    }
  }
}

// ------------------------------------------------------------------ gs_sc: row scales (quant_int16.h)
static void gs_sc(const GsCmd& c, hls::stream<qz::abits_t>& am, hls::stream<qz::RowScale>& rsf,
                  hls::stream<ap_uint<32> >& sst) {
  ap_uint<16> n = 0, bi = 0;
rows:
  for (;;) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 1024 avg = 256
    const qz::abits_t a = am.read();
    const ap_uint<32> s = qz::scale_bits(a);
    qz::RowScale rs = qz::row_scale_from_s(s);
    rs.R = qz::recip_sig(rs.S);
    rsf.write(rs);
    sst.write(qz::stored_scale(a, s, c.shift));
    if (n == c.n - 1) {
      n = 0;
      if (bi == c.nb - 1) break;
      bi++;
    } else {
      n++;
    }
  }
}

// ------------------------------------------------------------------ gs_qt: 4 quantizer lanes (or FP32 pass-through)
static void gs_qt(const GsCmd& c, hls::stream<Word>& ys, hls::stream<qz::RowScale>& rsf, hls::stream<Word>& ow) {
  qz::RowScale rs;
  ap_uint<64> lo = 0;
  ap_uint<10> w = 0;
  ap_uint<16> n = 0, bi = 0;
words:
  for (;;) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 2 max = 786432 avg = 196608
    if (w == 0) rs = rsf.read();
    const Word y = ys.read();
    ap_uint<64> q4;
    for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
      q4(16 * l + 15, 16 * l) = (ap_uint<16>)qz::quant_elem(y(32 * l + 31, 32 * l), rs);
    }
    if (!c.qout) {
      ow.write(y);
    } else if (w[0]) {
      ow.write(((Word)q4 << 64) | (Word)lo);  // elements 8j .. 8j + 7 of the row: natural order
    }
    lo = q4;
    if (w == c.k4 - 1) {
      w = 0;
      if (n == c.n - 1) {
        n = 0;
        if (bi == c.nb - 1) break;
        bi++;
      } else {
        n++;
      }
    } else {
      w++;
    }
  }
}

// ------------------------------------------------------------------ gs_wr: io writer
// Per item, chunks of up to WCH rows: the chunk's rows (one burst if dense, else one burst per row; zero rows
// [N, NP) written as 0), then its ceil(rows / 4) scale words (F_QOUT; lanes past NP 1.0). The scale of every real
// row is taken from sst at the row's last word, also without F_QOUT (gs_sc writes one per row either way).
template <class WM>
static void gs_wr(const GsCmd& c, WM ymem, hls::stream<Word>& ow, hls::stream<ap_uint<32> >& sst) {
  ap_uint<32> sb[WCH];
#pragma HLS ARRAY_PARTITION variable = sb cyclic factor = 4
  waddr_t ic = c.c, jf = c.f;
  ap_uint<16> bi = 0;
items:
  for (;;) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 1 avg = 1
    waddr_t rc = ic, fc = jf;
    ap_uint<17> r0 = 0;
  chunks:
    for (;;) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 16 avg = 4
      const ap_uint<17> left = (ap_uint<17>)c.np - r0;
      const ap_uint<7> nr = left > WCH ? (ap_uint<7>)WCH : (ap_uint<7>)left;
      const ap_uint<17> rleft = c.n > r0 ? (ap_uint<17>)((ap_uint<17>)c.n - r0) : (ap_uint<17>)0;
      const ap_uint<7> nreal = rleft > nr ? nr : (ap_uint<7>)rleft;
      ap_uint<7> rr = 0;
      ap_uint<10> col = 0;
    runs:
      for (;;) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 64 avg = 1
        const ap_uint<17> len = c.wdense ? (ap_uint<17>)(nr * c.orw) : (ap_uint<17>)c.orw;
      data:
        for (ap_uint<17> i = 0; i < len; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 49152 avg = 12288
          const bool real = rr < nreal;
          Word o = 0;
          if (real) o = ow.read();
          from_word(ymem[rc + i], o);
          if (col == c.orw - 1) {
            const ap_uint<32> s = real ? sst.read() : ap_uint<32>(ONE_BITS);
            sb[rr(5, 0)] = s;
            rr++;
            col = 0;
          } else {
            col++;
          }
        }
        rc += c.wdense ? (waddr_t)len : (waddr_t)c.pc;
        if (rr == nr) break;
      }
      if (c.qout) {
        const ap_uint<5> nw = (ap_uint<5>)((nr + 3) >> 2);
      scales:
        for (ap_uint<5> j = 0; j < nw; j++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 16 avg = 16
          Word sw;
          for (int l = 0; l < 4; l++) {
#pragma HLS UNROLL
            const ap_uint<7> r = (ap_uint<7>)(j * 4 + l);
            sw(32 * l + 31, 32 * l) = r < nr ? sb[r(5, 0)] : ap_uint<32>(ONE_BITS);
          }
          from_word(ymem[fc + j], sw);
        }
        fc += nw;
      }
      r0 += nr;
      if (r0 == c.np) break;
    }
    if (bi == c.nb - 1) break;
    bi++;
    ic += c.sc;
    jf += c.sf;
  }
}

template <class RM, class WM>
static void gs_df(const GsCmd& c, RM xmem, WM ymem) {
#pragma HLS DATAFLOW
#pragma HLS STABLE variable = c
  hls::stream<Word> xin("gs_xin");
#pragma HLS STREAM variable = xin depth = GS_D_XIN
  hls::stream<Word> yf("gs_yf");
#pragma HLS STREAM variable = yf depth = GS_D_YF
  hls::stream<Word> ys("gs_ys");  // the row buffer: the quantizer starts a row once its absmax is known
#pragma HLS STREAM variable = ys depth = GS_D_YS
#ifdef GS_YS_URAM
#pragma HLS BIND_STORAGE variable = ys type = fifo impl = uram
#else
#pragma HLS BIND_STORAGE variable = ys type = fifo impl = bram
#endif
  hls::stream<qz::abits_t> am("gs_am");
#pragma HLS STREAM variable = am depth = GS_D_AM
  hls::stream<qz::RowScale> rsf("gs_rsf");
#pragma HLS STREAM variable = rsf depth = GS_D_RSF
  hls::stream<ap_uint<32> > sst("gs_sst");  // stored scales: the writer takes them after each row's data
#pragma HLS STREAM variable = sst depth = GS_D_SST
  hls::stream<Word> ow("gs_ow");
#pragma HLS STREAM variable = ow depth = GS_D_OW
#ifdef GS_DFSIM  // host deadlock model: every process its own thread, every stream bounded (dfsim/hls_stream.h)
  std::thread t0([&] { gs_rd(c, xmem, xin); });
  std::thread t1([&] { gs_fe(c, xin, yf); });
  std::thread t2([&] { gs_am(c, yf, ys, am); });
  std::thread t3([&] { gs_sc(c, am, rsf, sst); });
  std::thread t4([&] { gs_qt(c, ys, rsf, ow); });
  std::thread t5([&] { gs_wr(c, ymem, ow, sst); });
  t0.join();
  t1.join();
  t2.join();
  t3.join();
  t4.join();
  t5.join();
  const hls::stream<Word>* ws[] = {&xin, &yf, &ys, &ow};
  for (auto* s : ws)
    if (!s->empty()) {
      fprintf(stdout, "DFSIM: stream %s holds %zu words after the command\n", s->name().c_str(), s->size());
      _Exit(8);
    }
  if (!am.empty() || !rsf.empty() || !sst.empty()) {
    fprintf(stdout, "DFSIM: scale streams not empty after the command\n");
    _Exit(8);
  }
#else
  gs_rd(c, xmem, xin);
  gs_fe(c, xin, yf);
  gs_am(c, yf, ys, am);
  gs_sc(c, am, rsf, sst);
  gs_qt(c, ys, rsf, ow);
  gs_wr(c, ymem, ow, sst);
#endif
}

// ------------------------------------------------------------------ command -> GsCmd
// (word units from here on: validate_hw guarantees addresses < 2^40 and 16-byte alignment)
static inline GsCmd gs_plan(const ctr::Cmd& cmd) {
#pragma HLS INLINE
  const bool gelu = cmd.op == ctr::OP_GELU;
  const bool nomul = (cmd.flags & ctr::F_NOMUL) != 0;
  const bool qout = (cmd.flags & ctr::F_QOUT) != 0;
  const uint32_t np = cmd.p[ctr::P_NP] ? cmd.p[ctr::P_NP] : cmd.N;
  GsCmd c;
  c.mode = gelu ? MODE_GELU : (nomul ? MODE_SILU : MODE_SILU_MUL);
  const bool mul = c.mode == MODE_SILU_MUL;
  c.qout = qout;
  c.k4 = cmd.K >> 2;
  const ap_uint<11> gap = mul && c.k4 < GAP ? (ap_uint<11>)(GAP - c.k4) : (ap_uint<11>)0;
  c.up0 = (ap_uint<11>)c.k4 + gap;
  c.rowits = mul ? (ap_uint<11>)(c.up0 + c.k4) : (ap_uint<11>)c.k4;
  c.n = cmd.N;
  c.np = np;
  c.nb = cmd.B;
  c.shift = (ap_int<5>)(int32_t)cmd.p[ctr::P_SHIFT];
  const ctr::Slot &A = cmd.s[ctr::SLOT_A], &B = cmd.s[ctr::SLOT_B], &C = cmd.s[ctr::SLOT_C], &F = cmd.s[ctr::SLOT_F];
  c.a = (waddr_t)(A.addr >> 4);
  c.pa = A.pitch >> 4;
  c.sa = A.stride >> 4;
  c.b = mul ? (waddr_t)(B.addr >> 4) : (waddr_t)0;
  c.pb = mul ? (wpitch_t)(B.pitch >> 4) : (wpitch_t)0;
  c.sb = mul ? (wpitch_t)(B.stride >> 4) : (wpitch_t)0;
  c.rw = mul ? (ap_uint<11>)((ap_uint<11>)c.k4 << 1) : (ap_uint<11>)c.k4;
  // dense read: rows (SILU_MUL: fused gate | up rows, B = A + 4K) at a pitch with at most GAPW words of gap
  const bool one = cmd.N == 1;
  const bool gap_ok = c.pa >= c.rw && c.pa - c.rw <= GAPW;
  const bool fused = !mul || ((ap_uint<37>)c.a + c.k4 == c.b && (one || c.pb == c.pa) && (cmd.B == 1 || c.sb == c.sa));
  c.rdense = (one || gap_ok) && fused;
  c.rper = one ? c.rw : (ap_uint<11>)c.pa;
  const ap_uint<27> body = (ap_uint<16>)(cmd.N - 1) * c.rper;
#pragma HLS BIND_OP variable = body op = mul impl = dsp
  c.rlen = (ap_uint<28>)body + c.rw;
  c.c = (waddr_t)(C.addr >> 4);
  c.pc = C.pitch >> 4;
  c.sc = C.stride >> 4;
  c.orw = qout ? (ap_uint<10>)(c.k4 >> 1) : c.k4;
  c.wdense = np == 1 || c.pc == c.orw;
  c.f = qout ? (waddr_t)(F.addr >> 4) : (waddr_t)0;
  c.sf = qout ? (wpitch_t)(F.stride >> 4) : (wpitch_t)0;
  return c;
}

// ------------------------------------------------------------------ the mode function
// Returns ctr::ME_OK, or (nothing read or written) ME_FLAGS / ME_DIMS / ME_PARAM for a command the dataflow region
// cannot run safely (K > ACT_K_MAX would deadlock the row buffer). That guard (GS_GUARD, default 1) duplicates
// validate_hw, which the sequencer runs before every dispatch and which also checks the slots, alignment and pitches
// that this function trusts: a float_top build may set GS_GUARD=0.
template <class RM, class WM>
static uint32_t act_mode(const ctr::Cmd& cmd, RM xmem, WM ymem) {
#pragma HLS INLINE off
#if GS_GUARD
  {
    const bool gelu = cmd.op == ctr::OP_GELU;
    const bool nomul = (cmd.flags & ctr::F_NOMUL) != 0;
    const bool qout = (cmd.flags & ctr::F_QOUT) != 0;
    const uint32_t np = cmd.p[ctr::P_NP] ? cmd.p[ctr::P_NP] : cmd.N;
    const uint32_t sh = cmd.p[ctr::P_SHIFT];
    // one unsigned compare per range: x in [lo, hi] <=> x - lo <= hi - lo
    if ((!gelu && cmd.op != ctr::OP_SILU_MUL) || (gelu && nomul)) return ctr::ME_FLAGS;
    if (cmd.K - 8u > (uint32_t)KMAX - 8u || (cmd.K & 7) != 0 || cmd.N - 1u >= ctr::N_MAX || cmd.B - 1u >= ctr::B_MAX)
      return ctr::ME_DIMS;
    if (np < cmd.N || np > ctr::N_MAX || sh + ctr::SHIFT_MAX > 2 * ctr::SHIFT_MAX || (!qout && (np != cmd.N || sh)))
      return ctr::ME_PARAM;
  }
#endif
  const GsCmd c = gs_plan(cmd);
  gs_df(c, xmem, ymem);
  return ctr::ME_OK;
}

// The sequencer's unit signature (fpga/kernels/sequencer/seq_core.h "Units"): the hp ports are not used.
template <class WP, class W>
static uint32_t act_unit(const ctr::Cmd& c, const WP* hp0, const WP* hp1, const WP* hp2, const WP* hp3, const W* io_rd,
                         W* io_wr) {
#pragma HLS INLINE
  (void)hp0, (void)hp1, (void)hp2, (void)hp3;
  return act_mode(c, io_rd, io_wr);
}

}  // namespace gs
