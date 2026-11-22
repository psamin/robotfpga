// SOFTMAX unit of the SmolVLA float_top (integration contract 1.0, fpga/kernels/CONTRACT.md; README.md here).
//
// Mode function:  uint32_t smx::softmax_mode(const ctr::Cmd& c, const Word* xmem, Word* ymem)
//   called from the sequencer's dispatch switch (one call site, INLINE off) after ctr::decode and
//   ctr::validate_hw. Reads only through xmem and writes only through ymem (the io bundle); no own m_axi
//   adapter, no own decode, no register. Precondition: validate_hw(c) == ME_OK (K in {8, 16, .., 256},
//   1 <= N, B <= 65535, N <= NP <= 65535, F_QOUT, slot addresses < 2^40); every counter is sized from those
//   limits (CONTRACT.md 3.4), so no command that passed it can hang the unit. Returns ME_OK.
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
// Structure: one DATAFLOW region, every loop II = 1, one io read stream and one io write stream:
//   reader   xmem: rank tables, then each block of <= 256 rows as one burst (one burst per row if the A
//            pitch is not 4 L); masks, the row max; the words + visibility bits into a one-row FIFO
//   compute  per row: 1 setup cycle (the max through lane 0's fix shifter), then L/4 words: 4 lanes of
//            2^x, q, the row sum; rows [N, NP) and the pad lanes of the scale vector give q = 0, sum 0
//   divider  RN(1 / sum), one radix-4 restoring unit, one row per 14 cycles (each scale out before the next
//            sum is read: no lookahead, no deadlock); the last sum of the command carries an end flag
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
// the writer reads a block's scale words after its rows: rq must hold one block's BLK / 4 words. Below about
// BLK / 4 - 1 the divider stalls, then compute (sum FIFO full), while the writer waits for P rows: deadlock.
// Exactly BLK / 4 keeps the LUTRAM at 64 entries (72 made it 128).
#define SMX_RQ_DEPTH (BLK / 4)
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
typedef hls::stream<ap_uint<24>, SQ_DEPTH> SStream;  // [23] last row of the command, [22:0] row sum
typedef hls::stream<Word, RQ_DEPTH> RStream;  // scale words: 4 rows each

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
  // RNE to 24 bits (s24 + rnd) then half up to 17 bits ((sig + 64) >> 7) in one add: the carry of the first
  // rounding (s24 + rnd = 2^24) and of the second both land in mcr = 2^17, i.e. Mc = 2^16 and ec + 1
  const bool rnd = rb && (st || s24[0]);
  const ap_uint<26> T = (ap_uint<26>)s24 + (ap_uint<26>)(rnd ? 65 : 64);
  const ap_uint<19> mcr = T >> 7;  // <= 2^17
  const bool cup = mcr[17];
  mc = (ap_uint<17>)mcr.range(16, 0) | (ap_uint<17>)(cup ? 0x10000 : 0);
  ec = (ap_int<10>)((int)e0 - 127 + (hi ? 1 : 0) + (cup ? 1 : 0));  // biased exponent < 255: p0 < 1e30
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

// 33-bit unsigned max key: [32] = visible, [31:0] = the FP32 total-order key b ^ (b[31] ? 0xFFFFFFFF : 0x80000000)
// (monotonic in the value). A visible key beats every masked one without a select; when no key of a row is
// visible every q is 0 and the max is not used.
typedef ap_uint<33> Key;
// key = (valid, !b[31], b[30:0] ^ b[31] replicated): the sign bit inverts the other 31 bits (one 31-bit xor)
static inline Key okey(ap_uint<32> b, bool valid) {
#pragma HLS INLINE
  const ap_uint<31> sx = (ap_uint<31>)(ap_int<31>)(ap_int<1>)b[31];
  const ap_uint<31> lo = b.range(30, 0) ^ sx;
  const ap_uint<1> v = valid ? 1 : 0, hi = b[31] ? 0 : 1;
  return (v, hi, lo);
}
// the FP32 bits of a key: (!u[31], u[30:0] ^ !u[31] replicated)
static inline ap_uint<32> key_bits(Key k) {
#pragma HLS INLINE
  const ap_uint<1> s = k[31] ? 0 : 1;
  const ap_uint<31> sx = (ap_uint<31>)(ap_int<31>)(ap_int<1>)s;
  const ap_uint<31> lo = k.range(30, 0) ^ sx;
  return (s, lo);
}

// ~G(x) on 48 bits for the FP32 bits b, where G(x) = fix(x) + 2^47 for x >= 0 and fix(x) + 2^47 - 1 for x < 0,
// fix(x) = sign(x) trunc(|x| 2^(e - 150 + ec + 21)) on 47 bits (|.| saturating at 2^47 - 1, e == 0 -> 0):
// the magnitude is M << 24 >> (kc - e), M = 1.mantissa. Then fix(m) - fix(x) = G(m) + ~G(x) + (s_m | !s_x)
// whenever x <= m (only one 48-bit adder per lane).
// The two special cases ride on the shifter instead of a 47-bit select after it: zero (e == 0 or k >= 48) as the
// shift amount 63 (a 48-bit value shifted by >= 48 is 0), saturation (k <= 0 or e == 255; never together with
// zero, as kc <= 279) as an all-ones shifter input with amount 0.
static inline ap_uint<48> neg_g(ap_uint<32> b, ap_int<11> kc) {
#pragma HLS INLINE
  const ap_uint<8> e = b.range(30, 23);
  const ap_int<11> k = kc - (ap_int<11>)(ap_uint<11>)e;  // right shift of M << 24
  const bool zero = e == 0 || k >= 48;
  const bool sat = !zero && (k <= 0 || e == 255);
  const ap_uint<6> amt = zero ? (ap_uint<6>)63 : sat ? (ap_uint<6>)0 : (ap_uint<6>)k.range(5, 0);
  const ap_uint<24> fill = (ap_uint<24>)(ap_int<24>)(ap_int<1>)(sat ? 1 : 0);
  const ap_uint<24> hi = ((ap_uint<24>)b.range(22, 0) | (ap_uint<24>)0x800000) | fill;
  const ap_uint<48> m = (hi, fill);
  const ap_uint<47> mag = (ap_uint<47>)(m >> amt);
  // G = s ? (0, ~mag) : (1, mag)  ->  ~G = s ? (1, mag) : (0, ~mag)
  const bool s = b[31];
  const ap_uint<47> sx = (ap_uint<47>)(ap_int<47>)(ap_int<1>)(s ? 0 : 1);
  const ap_uint<1> top = s ? 1 : 0;
  return (top, (ap_uint<47>)(mag ^ sx));
}

// q of two lanes from their distances u (Q.21) to the row max: one 1024 x 36-bit table (one BRAM36, two read
// ports) per pair of lanes. q = (h + 257) >> 9 with h = 2 e23 >> I, I = integer part of u Mc, with no select:
//   - I in [17, 63] gives h <= (2^24 - 1) >> 17 = 127, so r < 512 and q = 0 (the definition's "0 if I >= 17");
//     the 6-bit shift amount makes h = 0 for I >= 24;
//   - a key that is not kept (masked, a pad row, or d >= 2^26) arrives with u = 2^26 - 1 (the definition's
//     saturation), so t >= (2^26 - 1) 2^16 and I >= 31: q = 0 again.
// So the pipeline carries no per-lane flags past the multiplier (HLS books every delayed value as a shift
// register).
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
    const ap_uint<13> S = te.range(12, 0);
    // e23 = T - floor(S flo / 2^13) = floor((T 2^13 + 8191 - S flo) / 2^13): one DSP (multiply + post-subtract);
    // T 2^13 + 8191 is te with its low 13 bits (S) replaced by ones. The difference is > 2^35 - 2^26 > 0.
    const ap_uint<36> tc = (te.range(35, 13), (ap_uint<13>)0x1FFF);
    ap_uint<36> pc = tc - S * flo;
#pragma HLS BIND_OP variable = pc op = mul impl = dsp
    const ap_uint<23> e23 = pc.range(35, 13);
    const ap_uint<24> h = ((ap_uint<24>)e23 << 1) >> I;
    const ap_uint<24> r = h + (ap_uint<24>)257;
    q[l] = r.range(23, 9);
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
  Key pm1 = 0, pm2 = 0;  // partial max of the previous word's parity / of this word's parity
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
    Key k[4];
    for (int l = 0; l < 4; l++) {
#pragma HLS UNROLL
      // 0 <= E <= D in int32: D >= 0 and E <= D as unsigned (a negative E is >= 2^31 unsigned)
      vis[l] = !mask || (!dv[31] && lane32(ev, l) <= dv);
      k[l] = okey(lane32(v, l), vis[l]);
    }
    const Key k01 = k[1] > k[0] ? k[1] : k[0], k23 = k[3] > k[2] ? k[3] : k[2];
    const Key k4 = k23 > k01 ? k23 : k01;
    const Key prev = w < 2 ? (Key)0 : pm2;
    const Key nm = k4 > prev ? k4 : prev;
    if (kind == KIND_X) {
      xs.write(((XW)vis << 128) | (XW)v);
      if (w == W - 1) {
        const Key fm = nm > pm1 ? nm : pm1;  // W is even: pm1 holds the other parity's max
        mx.write(key_bits(fm));
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
#pragma HLS BIND_STORAGE variable = dtab type = ram_s2p impl = bram
  WAddr ia = p.a, ie = p.e, id = p.d;
items_r:
  for (ap_uint<17> b = 0; b < p.B; b++) {
#pragma HLS LOOP_TRIPCOUNT max = 24
    const bool load_e = p.mask && (b == 0 || p.es != 0);
    WAddr ra = ia;
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
          base = is_e ? ie : (WAddr)(id + (r0 >> 2));
          len = is_e ? (ap_uint<15>)p.W : (ap_uint<15>)((rows + 3) >> 2);
        }
        ap_uint<9> nn = kind == KIND_X ? n : (ap_uint<9>)0;
        read_burst(xm, base, len, kind, p.W, p.mask, nn, etab, dtab, xs, mx);
        if (kind == KIND_X) {
          n = nn;
          ra += step;
        }
      }
    }
    ia += p.as;
    ie += p.es;
    id += p.ds;
  }
}

// ------------------------------------------------------------------ compute
// The sum stream carries, in bit 23, "last row of the command": the divider needs no row or item counters.
static void compute(XStream& xs, MStream& mx, QStream& eq, SStream& sq, const Params p) {
items_c:
  for (ap_uint<17> b = 0; b < p.B; b++) {
#pragma HLS LOOP_TRIPCOUNT max = 24
    const bool blast = b == p.B - 1;
    const bool fin_rows = blast && p.NP == p.NP4;  // the last real/zero row is the command's last scale
    ap_uint<17> n = 0;
    ap_uint<7> w = 0;     // 0: setup (the row max), 1..W: data words
    ap_uint<49> gm2 = 1;  // 2 ~G(m) + 1
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
        // fix(m) - fix(x) = G(m) + ~G(x) + c = ~G(x) - ~G(m) - 1 + c, c = s_m | !s_x, as one 49-bit subtract:
        // ((2 ~G(x) + c) - (2 ~G(m) + 1)) >> 1
        const ap_uint<1> cin = (sm || !sx) ? 1 : 0;
        const ap_uint<49> d2 = (ng[l], cin) - gm2;
        const ap_uint<48> d = d2.range(48, 1);
        // a key that is masked, in a pad row, or at d >= 2^26 gets the saturated u (q = 0, see q_pair)
        const bool keep = real && x[128 + l] && d.range(47, 26) == 0;
        u[l] = keep ? (ap_uint<26>)d.range(25, 0) : (ap_uint<26>)0x3FFFFFF;
      }
      ap_uint<15> q[4];
      q_pair<0>(u[0], u[1], p.mc, q[0], q[1]);
      q_pair<1>(u[2], u[3], p.mc, q[2], q[3]);
      ap_uint<64> q4 = 0;
      ap_uint<17> s4 = 0;
      for (int l = 0; l < 4; l++) {
#pragma HLS UNROLL
        q4.range(16 * l + 15, 16 * l) = q[l];
        s4 += q[l];
      }
      const ap_uint<23> ns = sum + s4;
      const bool row_end = w == p.W, lrow = n == p.NP - 1;
      if (setup) {
        gm2 = (ng[0], (ap_uint<1>)1);  // ng[0] = ~G(m)
        sm = mb[31];
        sum = 0;
      } else {
        sum = ns;
        if (w[0])
          pend = q4;
        else
          eq.write((q4, pend));
        if (row_end) sq.write(((ap_uint<1>)(fin_rows && lrow), ns));
      }
      last = lrow && row_end;
      if (row_end) {
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
      sq.write(((ap_uint<1>)(blast && r == p.NP4 - 1), (ap_uint<23>)0));
    }
  }
}

// ------------------------------------------------------------------ divider: RN(1 / S) as FP32 bits (1.0 for S == 0)
// S < 2^23 normalised to d = S << (22 - p) (p = msb S); q = floor(2^47 / d) in (2^24, 2^25] by restoring division,
// two quotient bits per clock (radix 4: 4 rem against d, 2d, 3d); round to nearest even with the remainder as
// sticky bit. One unit, 14 clocks per row (load + 13 double steps) over the B * NP4 sums of the command; each row's
// scale is written before the next sum is read, so the unit never waits on a later row (no lookahead, no
// deadlock). Rows arrive in command order, pad lanes included, so every item holds NP4 = 0 mod 4 rows and a 2-bit
// counter places each scale in its word; bit 23 of the sum ends the command.
// (Tried and dropped: the divider inside compute's loop saved the process and the sum FIFO but cost +77 HLS
// LUT and 0.10 ns of slack, csynth v10.)
constexpr int DSTEPS = 13;  // double steps per row
struct Div {
  ap_uint<23> rem, d;  // rem < d
  ap_uint<25> d3;
  ap_uint<26> q;
  ap_uint<5> pe;
  bool zero;
};
static inline void div_load(Div& s, ap_uint<23> v) {
#pragma HLS INLINE
  ap_uint<5> msb = 0;
  for (int j = 0; j < 23; j++) {
#pragma HLS UNROLL
    if (v[j]) msb = j;
  }
  s.d = v << (22 - msb);
  s.d3 = (ap_uint<25>)s.d + ((ap_uint<25>)s.d << 1);
  s.rem = (ap_uint<23>)1 << 21;
  s.q = 0;
  s.pe = msb;
  s.zero = v == 0;
}
static inline void div_step(Div& s) {  // radix 4: R = 4 rem against d, 2d, 3d in parallel
#pragma HLS INLINE
  const ap_uint<25> R = (ap_uint<25>)s.rem << 2;
  const ap_int<26> t1 = (ap_int<26>)R - (ap_int<26>)s.d, t2 = (ap_int<26>)R - ((ap_int<26>)s.d << 1),
                   t3 = (ap_int<26>)R - (ap_int<26>)s.d3;
  ap_uint<2> dig;
  if (t3 >= 0) {
    dig = 3;
    s.rem = t3;
  } else if (t2 >= 0) {
    dig = 2;
    s.rem = t2;
  } else if (t1 >= 0) {
    dig = 1;
    s.rem = t1;
  } else {
    dig = 0;
    s.rem = R;
  }
  s.q = (s.q << 2) | (ap_uint<26>)dig;
}
static inline ap_uint<32> div_result(const Div& s) {  // after DSTEPS steps
#pragma HLS INLINE
  ap_uint<32> bits;
  if (s.zero) {
    bits = ONE;
  } else if (s.q[25]) {  // S = 2^p
    bits = (ap_uint<32>)(127 - s.pe) << 23;
  } else {
    const bool up = s.q[0] && (s.rem != 0 || s.q[1]);
    const ap_uint<25> sig = (ap_uint<25>)(s.q >> 1) + (ap_uint<25>)up;
    const ap_uint<8> ex = (ap_uint<8>)(126 - s.pe) + (ap_uint<8>)sig[24];
    bits = ((ap_uint<9>)ex, (ap_uint<23>)sig.range(22, 0));
  }
  return bits;
}
static void divider(SStream& sq, RStream& rq) {
  Div dv = {0, 0, 0, 0, 0, false};
  ap_uint<4> k = 0;
  ap_uint<2> rl = 0;    // lane of the row in its scale word
  ap_uint<128> pk = 0;  // the scale word being filled (lane rl)
  bool fin = false, done = false;
divide:
  while (!done) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT max = 86016
    if (k == 0) {
      const ap_uint<24> sv = sq.read();
      fin = sv[23];
      div_load(dv, sv.range(22, 0));
    } else {
      div_step(dv);
      if (k == DSTEPS) {
        pk = (div_result(dv), (ap_uint<96>)pk.range(127, 32));
        if (rl == 3) rq.write(pk);
      }
    }
    const bool row_end = k == DSTEPS;
    done = row_end && fin;
    k = row_end ? (ap_uint<4>)0 : (ap_uint<4>)(k + 1);
    if (row_end) rl++;
  }
}

// ------------------------------------------------------------------ writer (ymem)
// Per block of <= 256 rows: the P rows (one burst if the C pitch is 2 L, else one per row; the pitch's pad words
// are never written; rows [N, NP) are compute's zero rows), then the block's scale words (pad lanes 1.0) as one
// burst. One write site for both (one m_axi write path).
static void writer(QStream& eq, RStream& rq, Word* ym, const Params p) {
  WAddr ic = p.c, iff = p.f;
items_w:
  for (ap_uint<17> b = 0; b < p.B; b++) {
#pragma HLS LOOP_TRIPCOUNT max = 24
    WAddr rc = ic, rf = iff;
  blocks_w:
    for (ap_uint<17> r0 = 0; r0 < p.NP; r0 += BLK) {
#pragma HLS LOOP_TRIPCOUNT max = 1
      const ap_uint<17> left = p.NP - r0;
      const ap_uint<9> rows = left > BLK ? (ap_uint<9>)BLK : (ap_uint<9>)left;
      const ap_uint<7> nsw = (rows + 3) >> 2;  // the block's scale words (the last one padded with 1.0)
      const ap_uint<9> nb = p.c_contig ? (ap_uint<9>)1 : rows;  // data bursts; burst nb = the scales
      const ap_uint<14> nw = p.c_contig ? (ap_uint<14>)(rows * p.W2) : (ap_uint<14>)p.W2;
      const ap_uint<28> step = p.c_contig ? (ap_uint<28>)((ap_uint<28>)p.W2 << 8) : p.cp;
    bursts_w:
      for (ap_uint<9> j = 0; j <= nb; j++) {
#pragma HLS LOOP_TRIPCOUNT max = 2
        const bool sc = j == nb;
        const WAddr base = sc ? rf : rc;
        const ap_uint<14> len = sc ? (ap_uint<14>)nsw : nw;
      words_w:
        for (ap_uint<14> i = 0; i < len; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_FLATTEN off
#pragma HLS LOOP_TRIPCOUNT max = 8192
          ym[base + i] = sc ? rq.read() : eq.read();
        }
        if (!sc) rc += step;
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
#pragma HLS BIND_STORAGE variable = rq type = fifo impl = lutram
#if defined(SMX_DFSIM) && !defined(__SYNTHESIS__)
  std::thread t0([&] { reader(xm, xs, mx, p); });
  std::thread t1([&] { compute(xs, mx, eq, sq, p); });
  std::thread t2([&] { divider(sq, rq); });
  std::thread t3([&] { writer(eq, rq, ym, p); });
  t0.join();
  t1.join();
  t2.join();
  t3.join();
#else
  reader(xm, xs, mx, p);
  compute(xs, mx, eq, sq, p);
  divider(sq, rq);
  writer(eq, rq, ym, p);
#endif
}

// The mode function: one call site in the sequencer's dispatch switch, after ctr::decode and ctr::validate_hw
// returned ME_OK (the precondition: K in {8, 16, .., 256}, 1 <= N, B <= 65535, N <= NP <= 65535, F_QOUT; every
// counter of the unit is sized from these limits, CONTRACT.md 3.4). Returns ME_OK.
static uint32_t softmax_mode(const ctr::Cmd& c, const Word* xmem, Word* ymem) {
#pragma HLS INLINE off
  Params p;
  make_params(c, p);
  softmax_df(xmem, ymem, p);
  return ctr::ME_OK;
}

}  // namespace smx
