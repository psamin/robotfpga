// SOFTMAX unit of the SmolVLA float_top (integration contract 1.0, fpga/kernels/CONTRACT.md; README.md here).
//
// Mode function:  uint32_t smx::softmax_mode(const ctr::Cmd& c, const Word* xmem, Word* ymem)
//   called from the sequencer's dispatch switch (one call site, INLINE off) after ctr::decode and
//   ctr::validate_hw. Reads only through xmem and writes only through ymem (the io bundle); no own m_axi
//   adapter, no own decode, no register. Returns ME_OK (0), or ME_DIMS if K / N / B / NP are outside the
//   contract limits (validate_hw has already rejected those: the check only keeps a bad call from hanging).
//
// Record (CONTRACT.md 3.3, opcode SOFTMAX 0x40): K = L keys per row (8..256, L % 8 == 0), N rows per item,
// B items, flags F_QOUT (required) | F_MASK | F_YIELD (the sequencer's), p0 = score multiplier (FP32 bits),
// p5 = NP rows written (0 = N). Slots, every address absolute (row n of item b at addr + b stride + n pitch):
//   A  scores FP32 rows, L values          C  P int16 rows, L values (rows [N, NP) are zero)
//   F  scales FP32 [pad4(NP)]: 1/sum q per row; rows [N, NP), fully masked rows and pad lanes 1.0
//   D  rank_q int32 [pad4(N)] per item     E  rank_k int32 [L] per item (F_MASK; stride 0 = shared)
// Mask: key k visible to row n iff 0 <= E[k] <= D[n] (int32, exact).
//
// Arithmetic (exact definition: smx_emu.py; integer from the score bits to the scale bits):
//   c = fl32(max(p0, 2^-126) * fl32(log2 e)) = Mc 2^(ec - 16)    (integer RNE product, in the prologue)
//   m = row max over the visible keys (FP32 total-order keys)
//   u = min(fix(m) - fix(x), 2^26 - 1), fix(x) = trunc(x 2^(ec + 21)) on 47 bits, saturating
//   t = u Mc (Q.37); e23 = T23[t.ix] - (S13[t.ix] flo >> 13); h = 2 e23 >> I; q = (h + 257) >> 9 (0 if I >= 17)
//   q = 0 for masked keys; scale = RN(1 / sum q) (exact divider), 1.0 when the sum is 0.
//   These are the bits of the pre-contract kernel (q = (round(e23 / 2^I) + 128) >> 8).
//
// Structure: one DATAFLOW region, every loop II = 1 (scale words II = 4), one io read stream:
//   reader   xmem: rank tables, then each block of <= 256 rows as one burst (one burst per row if the A
//            pitch is not 4 L); masks, the row max; the words + visibility bits into a BRAM row FIFO
//   compute  per row: 1 setup cycle (the max through lane 0's fix shifter), then L/4 words: 4 lanes of
//            2^x, q, the row sum; rows [N, NP) and the pad lanes of the scale vector give q = 0, sum 0
//   divider  RN(1 / sum), one radix-4 restoring unit, one row per 14 cycles (each scale out before the next
//            sum is read: no lookahead, no deadlock)
//   writer   ymem: P rows per block of 256 rows (one burst, or one per row if the C pitch is not 2 L),
//            then the block's scale words
//
// Host builds (g++ with the Vitis HLS include directory) run the HLS source: by default the processes
// one after the other on unbounded streams; with -DSMX_DFSIM and tb_dfsim/ first on the include path
// (tb_dfsim/hls_stream.h) as threads on FIFOs bounded at the depths below (deadlock check).
#pragma once
#include <ap_int.h>
#include <hls_stream.h>
#include <stdint.h>

#include "../contract.h"
#include "softmax_tables.h"

#if defined(SMX_DFSIM) && !defined(__SYNTHESIS__)
#include <thread>
#endif

namespace smx {

typedef ap_uint<128> Word;
typedef ap_uint<36> WAddr;  // 128-bit word index = byte address >> 4 (addresses < 2^40: validate_hw)
typedef ap_uint<132> XW;    // reader -> compute: 4 FP32 scores, visibility of lane l at bit 128 + l

constexpr int L_MAX = (int)ctr::SMX_L_MAX;  // 256 keys
constexpr int W_MAX = L_MAX / 4;            // 64 input words per row
constexpr int BLK = 256;                    // rows per block: one D chunk, one scale burst
// FIFO depths (pragmas and the bounded host model use these numbers)
// (SMX_XS_DEPTH / SMX_RQ_DEPTH override them only for the deadlock model's negative controls)
#ifndef SMX_XS_DEPTH
#define SMX_XS_DEPTH W_MAX  // one row of 256 keys (LUTRAM): >= W_MAX is the deadlock bound
#endif
#ifndef SMX_RQ_DEPTH
#define SMX_RQ_DEPTH (BLK + 32)  // >= BLK: the writer reads a block's scales after its rows (deadlock bound)
#endif
constexpr int XS_DEPTH = SMX_XS_DEPTH;
constexpr int MX_DEPTH = 4;
constexpr int EQ_DEPTH = 16;
constexpr int SQ_DEPTH = 4;
constexpr int RQ_DEPTH = SMX_RQ_DEPTH;
constexpr uint32_t LOG2E_MANT = 0xB8AA3B;  // significand of fl32(log2 e) = 0x3FB8AA3B
constexpr uint32_t ONE = ctr::ONE_BITS;

typedef hls::stream<XW, XS_DEPTH> XStream;
typedef hls::stream<ap_uint<32>, MX_DEPTH> MStream;
typedef hls::stream<Word, EQ_DEPTH> QStream;
typedef hls::stream<ap_uint<23>, SQ_DEPTH> SStream;
typedef hls::stream<ap_uint<32>, RQ_DEPTH> RStream;

struct Params {
  ap_uint<17> B, N, NP, NP4;  // items, rows, rows written, pad4(NP) scales (each <= 65536)
  ap_uint<7> W;               // input words per row, L / 4 (2..64)
  ap_uint<6> W2;              // output words per row, L / 8 (1..32)
  bool mask, a_contig, c_contig;
  ap_uint<17> mc;             // c = mc 2^(ec - 16)
  ap_int<11> kc;              // 153 - ec: fix() shifts M << 24 right by kc - exponent
  WAddr a, c, d, e, f;        // item 0
  ap_uint<28> ap, cp;         // row pitches (words)
  ap_uint<28> as, cs, ds, es, fs;  // item strides (words)
};

// ------------------------------------------------------------------ prologue: Cmd -> Params
// c = fl32(p0' * fl32(log2 e)), p0' = max(p0, 2^-126) (p0 > 0 by validate_hw), as an exact integer RNE product,
// then Mc = c's significand rounded half up to 17 bits.
static inline void c_params(uint32_t p0, ap_uint<17>& mc, ap_int<10>& ec) {
#pragma HLS INLINE
  const ap_uint<32> pb = p0;
  ap_uint<8> e0 = pb.range(30, 23);
  ap_uint<24> m0 = (ap_uint<24>)pb.range(22, 0) | (ap_uint<24>)0x800000;
  if (e0 == 0) {  // below the normal range: 2^-126 (softmax of x c is uniform at that c anyway)
    e0 = 1;
    m0 = 0x800000;
  }
  const ap_uint<24> ml = LOG2E_MANT;
  ap_uint<48> pr = m0 * ml;  // [2^46, 2^48)
#pragma HLS BIND_OP variable = pr op = mul impl = dsp
  const bool hi = pr[47];
  const ap_uint<24> s24 = hi ? (ap_uint<24>)pr.range(47, 24) : (ap_uint<24>)pr.range(46, 23);
  const bool rb = hi ? pr[23] : pr[22];
  const bool st = hi ? pr.range(22, 0) != 0 : pr.range(21, 0) != 0;
  const ap_uint<25> sr = (ap_uint<25>)s24 + (ap_uint<25>)(rb && (st || s24[0]));
  const bool cy = sr[24];                                        // rounded up to 2^24
  const ap_uint<24> sig = cy ? (ap_uint<24>)0x800000 : (ap_uint<24>)sr.range(23, 0);
  const int be = (int)e0 + (hi ? 1 : 0) + (cy ? 1 : 0);           // biased exponent of c (< 255: p0 < 1e30)
  const ap_uint<18> mcr = ((ap_uint<18>)(sig >> 6) + 1) >> 1;    // round half up to 17 bits
  const bool cup = mcr[17];
  mc = cup ? (ap_uint<17>)(mcr >> 1) : (ap_uint<17>)mcr;
  ec = (ap_int<10>)(be - 127 + (cup ? 1 : 0));
}

static inline void make_params(const ctr::Cmd& c, Params& p) {
#pragma HLS INLINE
  const ctr::Slot &A = c.s[ctr::SLOT_A], &C = c.s[ctr::SLOT_C], &D = c.s[ctr::SLOT_D], &E = c.s[ctr::SLOT_E],
                  &F = c.s[ctr::SLOT_F];
  const uint32_t np = c.p[ctr::P_NP] ? c.p[ctr::P_NP] : c.N;
  p.B = c.B;
  p.N = c.N;
  p.NP = np;
  p.NP4 = (np + 3) & ~3u;
  p.W = c.K >> 2;
  p.W2 = c.K >> 3;
  p.mask = (c.flags & ctr::F_MASK) != 0;
  ap_int<10> ec;
  c_params(c.p[ctr::P_EPS], p.mc, ec);
  p.kc = (ap_int<11>)153 - (ap_int<11>)ec;
  p.a = (WAddr)(A.addr >> 4);
  p.c = (WAddr)(C.addr >> 4);
  p.d = (WAddr)(D.addr >> 4);
  p.e = (WAddr)(E.addr >> 4);
  p.f = (WAddr)(F.addr >> 4);
  p.ap = A.pitch >> 4;
  p.cp = C.pitch >> 4;
  p.as = A.stride >> 4;
  p.cs = C.stride >> 4;
  p.ds = D.stride >> 4;
  p.es = E.stride >> 4;
  p.fs = F.stride >> 4;
  p.a_contig = p.ap == p.W;
  p.c_contig = p.cp == p.W2;
}

// ------------------------------------------------------------------ arithmetic helpers
static inline ap_uint<32> lane32(const ap_uint<128>& w, int l) {
#pragma HLS INLINE
  return w.range(32 * l + 31, 32 * l);
}

// FP32 total-order integer key (monotonic in the value; its own inverse on the bit pattern); KMIN if not valid.
static inline ap_int<32> okey(ap_uint<32> b, bool valid) {
#pragma HLS INLINE
  const ap_uint<32> k = b[31] ? (ap_uint<32>)(b ^ 0x7FFFFFFFu) : b;
  return valid ? (ap_int<32>)k : (ap_int<32>)(ap_uint<32>)0x80000000u;
}

// ~G(x) on 48 bits for the FP32 bits b, where G(x) = fix(x) + 2^47 for x >= 0 and fix(x) + 2^47 - 1 for x < 0,
// fix(x) = sign(x) trunc(|x| 2^(e - 150 + ec + 21)) on 47 bits (|.| saturating at 2^47 - 1, e == 0 -> 0):
// the magnitude is M << 24 >> (kc - e), M = 1.mantissa. Then fix(m) - fix(x) = G(m) + ~G(x) + (s_m | !s_x)
// whenever x <= m (only one 48-bit adder per lane).
static inline ap_uint<48> neg_g(ap_uint<32> b, ap_int<11> kc) {
#pragma HLS INLINE
  const ap_uint<8> e = b.range(30, 23);
  const ap_uint<48> m = ((ap_uint<48>)((ap_uint<24>)b.range(22, 0) | (ap_uint<24>)0x800000)) << 24;
  const ap_int<11> k = kc - (ap_int<11>)(ap_uint<11>)e;  // right shift of M << 24
  ap_uint<47> mag;
  if (e == 0 || k >= 48)
    mag = 0;
  else if (k <= 0 || e == 255)
    mag = (ap_uint<47>)0x7FFFFFFFFFFFULL;
  else
    mag = (ap_uint<47>)(m >> (unsigned)k);
  // G = s ? (0, ~mag) : (1, mag)  ->  ~G = s ? (1, mag) : (0, ~mag)
  const bool s = b[31];
  const ap_uint<47> lo = s ? mag : (ap_uint<47>)~mag;
  return ((ap_uint<48>)(s ? 1 : 0) << 47) | (ap_uint<48>)lo;
}

// q of two lanes from their distances u (Q.21) to the row max: one 1024 x 36-bit table (one BRAM36, two read
// ports) per pair of lanes. q = (h + 257) >> 9 with h = 2 e23 >> I, I = integer part of u Mc (0 when I >= 17).
template <int PAIR>
static inline void q_pair(ap_uint<26> u0, ap_uint<26> u1, ap_uint<17> mc, ap_uint<15>& q0, ap_uint<15>& q1) {
#pragma HLS INLINE
  static const ap_uint<36> TAB[1024] = SMX_TAB_INIT;
#pragma HLS BIND_STORAGE variable = TAB type = rom_2p impl = bram
  ap_uint<26> u[2] = {u0, u1};
  ap_uint<15> q[2];
  for (int l = 0; l < 2; l++) {
#pragma HLS UNROLL
    ap_uint<43> t = (ap_uint<43>)u[l] * mc;
#pragma HLS BIND_OP variable = t op = mul impl = dsp
    const ap_uint<6> I = t.range(42, 37);
    const ap_uint<10> ix = t.range(36, 27);
    const ap_uint<13> flo = t.range(26, 14);
    const ap_uint<36> te = TAB[ix];
    const ap_uint<23> T = te.range(35, 13);
    const ap_uint<13> S = te.range(12, 0);
    ap_uint<26> corr = S * flo;
#pragma HLS BIND_OP variable = corr op = mul impl = dsp
    const ap_uint<23> e23 = T - (ap_uint<23>)corr.range(25, 13);
    const ap_uint<24> h = ((ap_uint<24>)e23 << 1) >> (unsigned)I.range(4, 0);
    const ap_uint<24> r = h + (ap_uint<24>)257;
    q[l] = I >= 17 ? (ap_uint<15>)0 : (ap_uint<15>)r.range(23, 9);
  }
  q0 = q[0];
  q1 = q[1];
}

// ------------------------------------------------------------------ reader (xmem)
// One burst of nw words at base, one call site for every read of the unit (so one m_axi read path):
//   KIND_E  the rank_k table (W words)          KIND_D  the block's rank_q table (ceil(rows / 4) words)
//   KIND_X  scores: row max (two partial maxima of alternating words, so the running compare has two
//           clocks) and visibility; data + visibility to xs, each row's max bits to mx after its last word.
enum { KIND_X = 0, KIND_E = 1, KIND_D = 2 };
static void read_burst(const Word* xm, WAddr base, ap_uint<15> nw, ap_uint<2> kind, ap_uint<7> W, bool mask,
                       ap_uint<9>& n, ap_uint<128> etab[W_MAX], ap_uint<128> dtab[BLK / 4], XStream& xs,
                       MStream& mx) {
#pragma HLS INLINE
  ap_uint<7> w = 0;
  ap_int<32> pm1 = 0, pm2 = 0;  // partial max of the previous word's parity / of this word's parity
words_r:
  for (ap_uint<15> i = 0; i < nw; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT max = 16384
#pragma HLS DEPENDENCE variable = etab inter false
#pragma HLS DEPENDENCE variable = dtab inter false
    const Word v = xm[base + i];
    if (kind == KIND_E) etab[i.range(5, 0)] = v;
    if (kind == KIND_D) dtab[i.range(5, 0)] = v;
    const ap_uint<128> ev = etab[w.range(5, 0)];
    const ap_uint<128> dw = dtab[n.range(7, 2)];
    const ap_uint<2> nl = n.range(1, 0);
    const ap_uint<32> dv = nl == 0 ? lane32(dw, 0) : nl == 1 ? lane32(dw, 1) : nl == 2 ? lane32(dw, 2) : lane32(dw, 3);
    ap_uint<4> vis;
    ap_int<32> k[4];
    for (int l = 0; l < 4; l++) {
#pragma HLS UNROLL
      // 0 <= E <= D in int32: D >= 0 and E <= D as unsigned (a negative E is >= 2^31 unsigned)
      vis[l] = !mask || (!dv[31] && lane32(ev, l) <= dv);
      k[l] = okey(lane32(v, l), vis[l]);
    }
    const ap_int<32> k01 = k[1] > k[0] ? k[1] : k[0], k23 = k[3] > k[2] ? k[3] : k[2];
    const ap_int<32> k4 = k23 > k01 ? k23 : k01;
    const ap_int<32> prev = w < 2 ? (ap_int<32>)(ap_uint<32>)0x80000000u : pm2;
    const ap_int<32> nm = k4 > prev ? k4 : prev;
    if (kind == KIND_X) {
      xs.write(((XW)vis << 128) | (XW)v);
      if (w == W - 1) {
        const ap_int<32> fm = nm > pm1 ? nm : pm1;  // W is even: pm1 holds the other parity's max
        const ap_uint<32> fb = fm;
        mx.write(fb[31] ? (ap_uint<32>)(fb ^ 0x7FFFFFFFu) : fb);  // the key's own bit pattern
      }
    }
    if (w == W - 1) {
      w = 0;
      n++;
    } else {
      w++;
    }
    pm2 = pm1;
    pm1 = nm;
  }
}

static void reader(const Word* xm, XStream& xs, MStream& mx, const Params p) {
  ap_uint<128> etab[W_MAX];   // rank_k of keys 4i..4i+3
  ap_uint<128> dtab[BLK / 4];  // rank_q of the block's rows 4i..4i+3
#pragma HLS BIND_STORAGE variable = etab type = ram_s2p impl = lutram
#pragma HLS BIND_STORAGE variable = dtab type = ram_s2p impl = lutram
  WAddr ia = p.a, ie = p.e, id = p.d;
items_r:
  for (ap_uint<17> b = 0; b < p.B; b++) {
#pragma HLS LOOP_TRIPCOUNT max = 24
    const bool load_e = p.mask && (b == 0 || p.es != 0);
    WAddr ra = ia, rd = id;
  blocks_r:
    for (ap_uint<17> r0 = 0; r0 < p.N; r0 += BLK) {
#pragma HLS LOOP_TRIPCOUNT max = 1
      const ap_uint<17> left = p.N - r0;
      const ap_uint<9> rows = left > BLK ? (ap_uint<9>)BLK : (ap_uint<9>)left;
      // bursts of the block: [rank_k (first block of an item that needs it)] [rank_q (F_MASK)] then the scores:
      // one burst if the rows are contiguous (A pitch = 4 L), else one per row
      const bool e_here = load_e && r0 == 0;
      const ap_uint<2> pre = (ap_uint<2>)(e_here ? 1 : 0) + (ap_uint<2>)(p.mask ? 1 : 0);
      const ap_uint<9> nb = p.a_contig ? (ap_uint<9>)1 : rows;
      const ap_uint<15> nw = p.a_contig ? (ap_uint<15>)(rows * p.W) : (ap_uint<15>)p.W;
      const ap_uint<28> step = p.a_contig ? (ap_uint<28>)((ap_uint<28>)p.W << 8) : p.ap;
      ap_uint<9> n = 0;
    bursts_r:
      for (ap_uint<10> j = 0; j < nb + pre; j++) {
#pragma HLS LOOP_TRIPCOUNT max = 3
        ap_uint<2> kind = KIND_X;
        WAddr base = ra;
        ap_uint<15> len = nw;
        if (j < pre) {
          const bool is_e = e_here && j == 0;
          kind = is_e ? KIND_E : KIND_D;
          base = is_e ? ie : rd;
          len = is_e ? (ap_uint<15>)p.W : (ap_uint<15>)((rows + 3) >> 2);
        }
        ap_uint<9> nn = kind == KIND_X ? n : (ap_uint<9>)0;
        read_burst(xm, base, len, kind, p.W, p.mask, nn, etab, dtab, xs, mx);
        if (kind == KIND_X) {
          n = nn;
          ra += step;
        }
      }
      rd += BLK / 4;
    }
    ia += p.as;
    ie += p.es;
    id += p.ds;
  }
}

// ------------------------------------------------------------------ compute
static void compute(XStream& xs, MStream& mx, QStream& eq, SStream& sq, const Params p) {
items_c:
  for (ap_uint<17> b = 0; b < p.B; b++) {
#pragma HLS LOOP_TRIPCOUNT max = 24
    ap_uint<17> n = 0;
    ap_uint<7> w = 0;  // 0: setup (the row max), 1..W: data words
    ap_uint<48> gm = 0;
    bool sm = false;
    ap_uint<23> sum = 0;
    ap_uint<64> pend = 0;
    bool last = false;
  rows_c:
    while (!last) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT max = 16640
      const bool setup = w == 0, real = n < p.N;
      // x / mb: this iteration's FIFO data; their value when not read is never used (no hold register)
      XW x;
      ap_uint<32> mb;
      if (real) {
        if (setup)
          mb = mx.read();
        else
          x = xs.read();
      }
      const ap_uint<32> b0 = setup ? mb : lane32(x, 0);
      ap_uint<48> ng[4];
      ng[0] = neg_g(b0, p.kc);
      for (int l = 1; l < 4; l++) {
#pragma HLS UNROLL
        ng[l] = neg_g(lane32(x, l), p.kc);
      }
      ap_uint<26> u[4];
      for (int l = 0; l < 4; l++) {
#pragma HLS UNROLL
        const bool sx = x[32 * l + 31];
        // G(m) + ~G(x) + carry as one 49-bit adder: (2 G(m) + 1) + (2 ~G(x) + carry), then >> 1
        const ap_uint<49> d2 = (((ap_uint<49>)gm << 1) | 1) + (((ap_uint<49>)ng[l] << 1) | (ap_uint<49>)(sm || !sx));
        const ap_uint<48> d = d2.range(48, 1);
        u[l] = d.range(47, 26) != 0 ? (ap_uint<26>)0x3FFFFFF : (ap_uint<26>)d.range(25, 0);
      }
      ap_uint<15> q[4];
      q_pair<0>(u[0], u[1], p.mc, q[0], q[1]);
      q_pair<1>(u[2], u[3], p.mc, q[2], q[3]);
      ap_uint<64> q4 = 0;
      ap_uint<17> s4 = 0;
      for (int l = 0; l < 4; l++) {
#pragma HLS UNROLL
        const ap_uint<15> ql = (real && x[128 + l]) ? q[l] : (ap_uint<15>)0;
        q4.range(16 * l + 15, 16 * l) = ql;
        s4 += ql;
      }
      const ap_uint<23> ns = sum + s4;
      if (setup) {
        gm = ~ng[0];  // G(m)
        sm = mb[31];
        sum = 0;
      } else {
        sum = ns;
        if (w[0])
          pend = q4;
        else
          eq.write(((Word)q4 << 64) | (Word)pend);
        if (w == p.W) sq.write(ns);
      }
      last = n == p.NP - 1 && w == p.W;
      if (w == p.W) {
        w = 0;
        n++;
      } else {
        w++;
      }
    }
  pad_c:
    for (ap_uint<17> r = p.NP; r < p.NP4; r++) {  // the scale vector's pad lanes: sum 0 -> 1.0
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT max = 3
      sq.write(0);
    }
  }
}

// ------------------------------------------------------------------ divider: RN(1 / S) as FP32 bits (1.0 for S == 0)
// S < 2^23 normalised to d = S << (22 - p) (p = msb S); q = floor(2^47 / d) in (2^24, 2^25] by restoring division,
// two quotient bits per clock (radix 4: 2r against d, 2d, 3d, i.e. two radix-2 steps); round to nearest even
// with the remainder as sticky bit. One unit, 14 clocks per row (load + 13 steps); each row's scale is written
// before the next sum is read, so the unit never waits on a later row (no lookahead, no deadlock).
static inline ap_uint<32> div_finish(ap_uint<26> q, bool sticky, ap_uint<5> pe, bool zero) {
#pragma HLS INLINE
  ap_uint<32> bits;
  if (zero) {
    bits = ONE;
  } else if (q[25]) {  // S = 2^p
    bits = (ap_uint<32>)(127 - pe) << 23;
  } else {
    const bool up = q[0] && (sticky || q[1]);
    const ap_uint<25> sig = (ap_uint<25>)(q >> 1) + (ap_uint<25>)up;
    const ap_uint<8> ex = (ap_uint<8>)(126 - pe) + (ap_uint<8>)sig[24];
    bits = ((ap_uint<9>)ex, (ap_uint<23>)sig.range(22, 0));
  }
  return bits;
}
constexpr int DSTEPS = 13;  // radix-4 steps per row
static void divider(SStream& sq, RStream& rq, const Params p) {
items_d:
  for (ap_uint<17> b = 0; b < p.B; b++) {
#pragma HLS LOOP_TRIPCOUNT max = 24
    const ap_uint<21> tot = ((ap_uint<21>)p.NP4 << 4) - ((ap_uint<21>)p.NP4 << 1);  // NP4 * (DSTEPS + 1)
    ap_uint<4> k = 0;
    ap_uint<24> r = 0;  // partial remainder, < 2 d
    ap_uint<23> d = 0;
    ap_uint<25> d3 = 0;
    ap_uint<26> q = 0;
    ap_uint<5> pe = 0;
    bool zero = false;
  divide:
    for (ap_uint<21> i = 0; i < tot; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT max = 3584
      if (k == 0) {
        const ap_uint<23> v = sq.read();
        ap_uint<5> msb = 0;
        for (int j = 0; j < 23; j++) {
#pragma HLS UNROLL
          if (v[j]) msb = j;
        }
        d = v << (22 - msb);
        d3 = (ap_uint<25>)d + ((ap_uint<25>)d << 1);
        r = (ap_uint<24>)1 << 22;
        q = 0;
        pe = msb;
        zero = v == 0;
      } else {
        const ap_uint<25> R = (ap_uint<25>)r << 1;  // < 4 d
        const ap_int<26> t1 = (ap_int<26>)R - (ap_int<26>)d, t2 = (ap_int<26>)R - ((ap_int<26>)d << 1),
                         t3 = (ap_int<26>)R - (ap_int<26>)d3;
        ap_uint<2> dig;
        ap_uint<24> rem;
        if (t3 >= 0) {
          dig = 3;
          rem = t3;
        } else if (t2 >= 0) {
          dig = 2;
          rem = t2;
        } else if (t1 >= 0) {
          dig = 1;
          rem = t1;
        } else {
          dig = 0;
          rem = R;
        }
        r = rem << 1;
        q = (q << 2) | (ap_uint<26>)dig;
        if (k == DSTEPS) rq.write(div_finish(q, r != 0, pe, zero));
      }
      k = k == DSTEPS ? (ap_uint<4>)0 : (ap_uint<4>)(k + 1);
    }
  }
}

// ------------------------------------------------------------------ writer (ymem)
static void writer(QStream& eq, RStream& rq, Word* ym, const Params p) {
  WAddr ic = p.c, iff = p.f;
items_w:
  for (ap_uint<17> b = 0; b < p.B; b++) {
#pragma HLS LOOP_TRIPCOUNT max = 24
    WAddr rc = ic, rf = iff;
  blocks_w:
    for (ap_uint<17> r0 = 0; r0 < p.NP; r0 += BLK) {
#pragma HLS LOOP_TRIPCOUNT max = 1
      const ap_uint<17> left = p.NP - r0, left4 = p.NP4 - r0;
      const ap_uint<9> rows = left > BLK ? (ap_uint<9>)BLK : (ap_uint<9>)left;
      const ap_uint<7> nsw = left4 > BLK ? (ap_uint<7>)(BLK / 4) : (ap_uint<7>)(left4 >> 2);
      // contiguous rows (C pitch = 2 L): the block is one burst; otherwise one burst per row (pad words of
      // the pitch are never written)
      const ap_uint<9> nb = p.c_contig ? (ap_uint<9>)1 : rows;
      const ap_uint<14> nw = p.c_contig ? (ap_uint<14>)(rows * p.W2) : (ap_uint<14>)p.W2;
      const ap_uint<28> step = p.c_contig ? (ap_uint<28>)((ap_uint<28>)p.W2 << 8) : p.cp;
    bursts_w:
      for (ap_uint<9> j = 0; j < nb; j++) {
#pragma HLS LOOP_TRIPCOUNT max = 1
      data_w:
        for (ap_uint<14> i = 0; i < nw; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT max = 8192
          ym[rc + i] = eq.read();
        }
        rc += step;
      }
    scales_w:
      for (ap_uint<7> k = 0; k < nsw; k++) {
#pragma HLS PIPELINE II = 4
#pragma HLS LOOP_TRIPCOUNT max = 64
        Word o;
        for (int l = 0; l < 4; l++) o.range(32 * l + 31, 32 * l) = rq.read();
        ym[rf + k] = o;
      }
      rf += BLK / 4;
    }
    ic += p.cs;
    iff += p.fs;
  }
}

// ------------------------------------------------------------------ the unit
static void softmax_df(const Word* xm, Word* ym, const Params p) {
#pragma HLS DATAFLOW
#pragma HLS STABLE variable = p
  XStream xs("smx_xs");
  MStream mx("smx_mx");
  QStream eq("smx_eq");
  SStream sq("smx_sq");
  RStream rq("smx_rq");
#pragma HLS STREAM variable = xs depth = XS_DEPTH
#pragma HLS BIND_STORAGE variable = xs type = fifo impl = lutram
#pragma HLS STREAM variable = mx depth = MX_DEPTH
#pragma HLS STREAM variable = eq depth = EQ_DEPTH
#pragma HLS BIND_STORAGE variable = eq type = fifo impl = srl
#pragma HLS STREAM variable = sq depth = SQ_DEPTH
#pragma HLS STREAM variable = rq depth = RQ_DEPTH
#pragma HLS BIND_STORAGE variable = rq type = fifo impl = bram
#if defined(SMX_DFSIM) && !defined(__SYNTHESIS__)
  std::thread t0([&] { reader(xm, xs, mx, p); });
  std::thread t1([&] { compute(xs, mx, eq, sq, p); });
  std::thread t2([&] { divider(sq, rq, p); });
  std::thread t3([&] { writer(eq, rq, ym, p); });
  t0.join();
  t1.join();
  t2.join();
  t3.join();
#else
  reader(xm, xs, mx, p);
  compute(xs, mx, eq, sq, p);
  divider(sq, rq, p);
  writer(eq, rq, ym, p);
#endif
}

// The mode function (one call site in the dispatch switch). Returns ME_OK, or ME_DIMS without touching memory.
static uint32_t softmax_mode(const ctr::Cmd& c, const Word* xmem, Word* ymem) {
#pragma HLS INLINE off
  // validate_hw has checked all of this; the guard only keeps a call that bypassed it from hanging the unit:
  // K in {8, 16, .., 256}, 1 <= N <= 65535, B <= 65535 (decode makes B >= 1), N <= NP <= 65535
  const ap_uint<32> K = c.K, N = c.N, B = c.B, P5 = c.p[ctr::P_NP];
  const bool k_bad = K.range(2, 0) != 0 || K.range(31, 9) != 0 || K.range(8, 3) == 0 || (K[8] && K.range(7, 3) != 0);
  const bool n_bad = N.range(31, 16) != 0 || N.range(15, 0) == 0 || B.range(31, 16) != 0 || B.range(15, 0) == 0;
  const bool np_bad = P5 != 0 && (P5.range(31, 16) != 0 || P5.range(15, 0) < N.range(15, 0));
  if (k_bad || n_bad || np_bad) return ctr::ME_DIMS;
  Params p;
  make_params(c, p);
  softmax_df(xmem, ymem, p);
  return ctr::ME_OK;
}

}  // namespace smx
