// ROPE mode of float_top under the integration contract (fpga/kernels/CONTRACT.md 1.1, 2.1, 2.2, 3.3; opcode 0x60).
//
//   uint32_t rope::rope_mode(const ctr::Cmd& c, const Word* hp, const Word* xmem, Word* ymem)
//
// One call site in the sequencer's dispatch switch (`case ctr::OP_ROPE: return rope::rope_mode(c, w0, xmem, ymem);`),
// INLINE off, no m_axi adapter, no decode and no register of its own: hp is the top's hp0 port (w0, read only: the
// RoPE table, a host-written constant), xmem / ymem are the top's io port (activations are read and written only on
// io). Every address is the record's absolute byte address (the pointer registers are 0), handled as a 128-bit word
// index of 36 bits: validate_hw guarantees addr < 2^40 and the host's validate every extent end <= 2^40 (CONTRACT.md
// 3.2), so no sum wraps for a list that passed validate; there is no int anywhere in the address path.
//
// Record (validate_hw has checked it; rope_mode re-checks the fields that size its loops and returns ME_* with nothing
// read or written when they are off, so no record can hang the dataflow region):
//   K = H * 64 values per row (H = 1..64 heads, p1 = SEG = 64), N rows (1..65535), B items (1..65535)
//   A   FP32 rows: head j of row n of item b = 64 FP32 at A + b A.stride + n A.pitch + 256 j, natural order
//   E   table rows: row n of item b at E + b E.stride + n E.pitch (pitch >= 256): words 0-7 NC[0..31] =
//       round(-cos 2^17), words 8-15 NS[0..31] = round(-sin 2^17), int32 slots of which bits [17:0] are used
//       (rope_emu.table_rows)
//   p5  NP rows written (0 = N). Rows [N, NP) are zero rows (q = 0, scale 1.0); they re-read row N - 1 (inside
//       the declared extents) and zero it
//   output, one of:
//     (no flag)  FP32 rows: segment j of row n at C + b C.stride + n C.pitch + j p2 (C == A in place allowed:
//                every segment is written after it was read); NP == N
//     F_QOUT     int16: segment j of row n (64 int16) at C + b C.stride + n C.pitch + j p2; its scale at
//                F + b F.stride + j p3 + 4 n; each head's vector is pad4(NP) floats, pad lanes 1.0.
//                The contract quantizer (quantize/quant_int16.h) over each (row, head) of the FP32 rotation; stored
//                scale s * 2^p4 (p4 = -3 for Q), an all-zero segment stores 1.0
//     F_KW       the KW weight tiling (as QUANT_W): value d of key n of head j in lane n % 8 of the word at
//                C + b C.stride + j p2 + ((n % 32) / 8) C.pitch + ((n / 32) 64 + d) 16; NP % 32 == 0; the per-key
//                scale (the blob's s_row) at F + b F.stride + j p3 + 4 n
//
// Processes (one DATAFLOW region per command; segments in the order (b, j, n): head-major, as quant_mode.h's QUANT_W):
//   rd      xmem + hp: per segment one 16-word x unit and one 16-word table unit (II = 16, unconditional bursts
//           pipelined across segments)
//   rot_stage, emit_stage (rope_core.h): the rotation -> FP32 words in natural order + each segment's absmax
//   sc      absmax -> s = qz::scale_bits, the stored scale (qz::stored_scale), the divider seed
//   rc      R = floor(2^49 / S): 2 iterative restoring dividers (qz::recip_step, 1 bit a clock), round robin
//   qt      4 lanes of qz::quant_elem: 16 FP32 words -> 8 int16 words per segment; FP32 words pass through
//   kw      F_KW: 8 keys x 8 int16 words -> 64 words of 8 keys (lock-step ping-pong transposer); else pass-through
//   pk      8-word write units with their addresses: data, then scale units (32 scales; a vector that does not end
//           on a unit ends with a shift-back unit over its last 8 words, re-writing the overlap with the same data);
//           Q vectors shorter than 8 words (NP <= 28) make the whole command single-word units
//   wr      ymem: one 8-beat burst per unit (II = 8), or single words
// rd / sc / rc / qt / kw / pk / wr are what the shared row unit (quantize/quant_mode.h) replaces; rope_core.h's two stages
// are the rope-specific front end (CONTRACT.md 4.1). ROPE_DFSIM builds every process as a thread on bounded FIFOs
// (dfsim/hls_stream.h): the deadlock model of CONTRACT.md 7.
#pragma once
#include <stdint.h>

#include "../contract.h"
#include "../quantize/quant_int16.h"
#include "rope_core.h"

#ifdef ROPE_DFSIM
#include <thread>
#endif

// FIFO depths (the bounded-FIFO model, dfsim/hls_stream.h, uses the same numbers; -D overrides them for tests)
#ifndef ROPE_D_XS
#define ROPE_D_XS 32  // x words: two 16-word read units
#endif
#ifndef ROPE_D_CS
#define ROPE_D_CS 32  // table words
#endif
#ifndef ROPE_D_RS
#define ROPE_D_RS 4   // rotated groups
#endif
#ifndef ROPE_D_FW
#define ROPE_D_FW 64  // FP32 words waiting for their segment's (s, R): 16 + the sc + rc latency (~35 clocks)
#endif
#ifndef ROPE_D_AM
#define ROPE_D_AM 4   // absmax per segment
#endif
#ifndef ROPE_D_SD
#define ROPE_D_SD 4   // divider seeds
#endif
#ifndef ROPE_D_SP
#define ROPE_D_SP 8   // (s, stored) per segment: waits beside the divider
#endif
#ifndef ROPE_D_RF
#define ROPE_D_RF 4   // R per segment
#endif
#ifndef ROPE_D_OW
#define ROPE_D_OW 32  // qt -> kw words
#endif
#ifndef ROPE_D_OT
#define ROPE_D_OT 32  // kw -> pk words
#endif
#ifndef ROPE_D_SS
#define ROPE_D_SS 32  // stored scales: KW holds two blocks (16) before pk reads them
#endif
#ifndef ROPE_D_HD
#define ROPE_D_HD 8   // unit headers
#endif
#ifndef ROPE_D_DD
#define ROPE_D_DD 32  // unit words
#endif

namespace rope {

typedef ap_uint<36> waddr_t;  // 128-bit word index: byte address bits [39:4]
typedef ap_uint<28> wstep_t;  // a u32 byte pitch / stride / segment step, in words
typedef ap_uint<17> cnt_t;    // B, N, NP <= 65535 (17 bits: a loop counter reaches its bound)
typedef ap_uint<7> seg_t;     // H = K / 64 in 1..64

constexpr int OUT_F32 = 0, OUT_Q = 1, OUT_KW = 2;

struct Args {
  ap_uint<2> out;                                         // OUT_F32 / OUT_Q / OUT_KW
  cnt_t B, N, NP;                                         // items, rows read, rows written
  seg_t H;                                                // heads per row
  waddr_t a, e, c, f;                                     // slot addresses (words)
  wstep_t a_p, a_s, e_p, e_s, c_p, c_s, f_s, segc, segf;  // pitches, strides, p2, p3 (words)
  ap_uint<15> npw;                                        // Q: words of one scale vector, ceil(NP / 4)
  bool wsmall;                                            // Q with npw < 8: every write a single word
};

// ------------------------------------------------------------------------------------- reader
// Segment (b, j, n): x row min(n, N - 1) of head j (zeroed for n >= N) and table row min(n, N - 1).
static void rd(const Word* xmem, const Word* hp, const Args g, hls::stream<Word>& xs, hls::stream<Word>& cs) {
  cnt_t b = 0, n = 0;
  seg_t j = 0;
  waddr_t ib = g.a, jb = g.a, xa = g.a, eb = g.e, ea = g.e;
rd:
  for (;;) {
#pragma HLS PIPELINE II = 16
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 1000 avg = 300
    Word v[WORDS], t[WORDS];
#pragma HLS ARRAY_PARTITION variable = v complete
#pragma HLS ARRAY_PARTITION variable = t complete
    for (int k = 0; k < WORDS; k++) {
#pragma HLS UNROLL
      v[k] = xmem[xa + k];  // unconditional: one 16-beat burst on each port, pipelined across segments
      t[k] = hp[ea + k];
    }
    const bool live = n < g.N;
    for (int k = 0; k < WORDS; k++) {
#pragma HLS UNROLL
      xs.write(live ? v[k] : Word(0));
      cs.write(t[k]);
    }
    const bool lastn = n == g.NP - 1, lastj = j == g.H - 1, lastb = b == g.B - 1;
    if (lastn && lastj && lastb) break;
    if (!lastn) {
      if (n + 1 < g.N) {  // rows >= N re-read row N - 1
        xa += g.a_p;
        ea += g.e_p;
      }
      n++;
    } else {
      n = 0;
      if (!lastj) {
        j++;
        jb += WORDS;
        xa = jb;
        ea = eb;
      } else {
        j = 0;
        b++;
        ib += g.a_s;
        jb = ib;
        xa = ib;
        eb += g.e_s;
        ea = eb;
      }
    }
  }
}

// ------------------------------------------------------------------------------------- contract quantizer
struct SP {          // sc -> qt, one per segment
  ap_uint<32> s;     // s = fl32(max(a, 1e-30) / 32767), 1.0 for a == 0
  ap_uint<32> st;    // the stored scale: s 2^p4, 1.0 for a == 0, the absmax for a non-finite segment
};
struct SD {          // sc -> rc: the divider's start
  ap_uint<24> S;     // significand of s
  qz::RecipSeed sd;
};

// absmax -> s (qz::scale_bits), the stored scale (qz::stored_scale) and the divider seed, one segment a clock
static void sc_stage(hls::stream<abits_t>& am, hls::stream<SD>& sdf, hls::stream<SP>& spf, rcount_t n,
                     ap_int<5> shift) {
sc:
  for (rcount_t i = 0; i < n; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 0 max = 1200 avg = 300
    const abits_t a = am.read();
    const ap_uint<32> s = qz::scale_bits(a);
    SD d;
    d.S = (ap_uint<1>(1), ap_uint<23>(s(22, 0)));
    d.sd = qz::recip_seed(d.S);
    sdf.write(d);
    SP t;
    t.s = s;
    t.st = qz::stored_scale(a, s, shift);
    spf.write(t);
  }
}

// R = min(floor(2^49 / S), 2^26 - 1) with qz::recip_step (QZ_RECIP_BITS = 1, CONTRACT.md 2.1): 2 restoring dividers,
// segments dealt round robin. One divide takes 27 clocks (load + 25 steps + hand-over), so the pair delivers one R
// every 13.5 clocks; a segment arrives every 16. (qz::recip_unrolled, one R a clock, synthesised at 3.5k LUT here: the
// shared row unit needs that rate for short rows, ROPE does not.)
static void rc_stage(hls::stream<SD>& sdf, hls::stream<ap_uint<26> >& rf, rcount_t n) {
  ap_uint<24> S0 = 0, S1 = 0;
  qz::RecipState st0, st1;
  st0.r = 0;
  st0.q = 0;
  st0.n = 0;
  st0.pow2 = false;
  st1 = st0;
  bool busy0 = false, busy1 = false;
  rcount_t ni = 0, no = 0;  // segments loaded / handed over
  bool more = n != 0;       // segments left to load
  ap_uint<1> ei = 0, eo = 0;
  if (n == 0) return;
rc:
  for (;;) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 27 max = 20000 avg = 5000
    // every decision uses the state at the start of the clock (a freed engine is reloaded one clock later)
    const bool fin = eo == 0 ? (busy0 && st0.n == 0) : (busy1 && st1.n == 0);
    const bool free_ei = ei == 0 ? !busy0 : !busy1;
    bool done = false;
    if (fin) {
      rf.write(qz::recip_result(eo == 0 ? st0 : st1));
      no++;
      done = no == n;
    }
    SD in;
    in.S = 0;
    in.sd.r0 = 0;
    in.sd.pow2 = false;
    const bool ld = more && free_ei && sdf.read_nb(in);
    const qz::RecipState fresh = qz::recip_load(in.sd);
    qz::RecipState s0 = st0, s1 = st1;
    qz::recip_step(s0, S0);  // the quotient bits of both engines
    qz::recip_step(s1, S1);
    const bool ld0 = ld && ei == 0, ld1 = ld && ei == 1;
    st0 = ld0 ? fresh : s0;  // one 2:1 mux behind the step
    st1 = ld1 ? fresh : s1;
    if (ld0) S0 = in.S;
    if (ld1) S1 = in.S;
    busy0 = ld0 || (busy0 && !(fin && eo == 0));
    busy1 = ld1 || (busy1 && !(fin && eo == 1));
    if (fin) eo = ~eo;
    if (ld) {
      ei = ~ei;
      ni++;
      more = ni != n;
    }
    if (done) break;
  }
}

// 16 FP32 words per segment -> 8 int16 words (word i = values 8i..8i+7) + the stored scale after them; without
// quant the FP32 words pass through.
static void qt_stage(hls::stream<Word>& fw, hls::stream<SP>& spf, hls::stream<ap_uint<26> >& rf, hls::stream<Word>& ow,
                     hls::stream<ap_uint<32> >& ss, rcount_t segs, bool quant) {
  qz::RowScale rs = qz::row_scale_from_s(qz::ONE_BITS);
  ap_uint<32> st = qz::ONE_BITS;
  ap_uint<64> lo = 0;
  ap_uint<4> w = 0;
  rcount_t r = 0;
  if (segs == 0) return;
qt:
  for (;;) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 16 max = 20000 avg = 5000
    if (quant && w == 0) {
      const SP t = spf.read();
      rs = qz::row_scale_from_s(t.s);
      rs.R = rf.read();
      st = t.st;
    }
    const Word x = fw.read();
    if (!quant) {
      ow.write(x);
    } else {
      ap_uint<64> h4;
      for (int l = 0; l < 4; l++) {
#pragma HLS UNROLL
        h4.range(16 * l + 15, 16 * l) = (ap_uint<16>)qz::quant_elem(lane(x, l), rs);
      }
      if (w[0]) {
        Word o;
        o.range(63, 0) = lo;
        o.range(127, 64) = h4;
        ow.write(o);
      } else {
        lo = h4;
      }
      if (w == WORDS - 1) ss.write(st);
    }
    if (w == WORDS - 1) {
      r++;
      if (r == segs) break;
    }
    w++;
  }
}

// ------------------------------------------------------------------------------------- KW transposer
static inline ap_uint<16> sel8(const Word& q, ap_uint<3> i) {  // int16 lane i: an 8:1 mux, not a shifter
#pragma HLS INLINE
  ap_uint<16> r;
  switch ((int)i) {
    case 0: r = q.range(15, 0); break;
    case 1: r = q.range(31, 16); break;
    case 2: r = q.range(47, 32); break;
    case 3: r = q.range(63, 48); break;
    case 4: r = q.range(79, 64); break;
    case 5: r = q.range(95, 80); break;
    case 6: r = q.range(111, 96); break;
    default: r = q.range(127, 112); break;
  }
  return r;
}

// KW: block k = keys 8k'..8k'+7 of one head (8 int16 words per key, key t's word i = values 8i..8i+7) in, output
// word d = lane t: value d of key t. Lock-step: iteration (k, w) stores word w of block k and emits word w of block
// k - 1 from the other slot, so 64 (nblk + 1) iterations. Else words pass through.
static void kw_stage(hls::stream<Word>& ow, hls::stream<Word>& ot, rcount_t nwords, rcount_t nblk, bool kw) {
  Word kb[8][2 * 8];  // bank t = key t of the block; entry slot * 8 + i
#pragma HLS ARRAY_PARTITION variable = kb dim = 1 complete
#pragma HLS BIND_STORAGE variable = kb type = ram_s2p impl = lutram
#pragma HLS DEPENDENCE variable = kb inter false
  if (!kw) {
  pass:
    for (rcount_t i = 0; i < nwords; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 16 max = 20000 avg = 5000
      ot.write(ow.read());
    }
    return;
  }
  rcount_t blk = 0;
  ap_uint<6> w = 0;
kw:
  for (;;) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 128 max = 20000 avg = 3000
    const ap_uint<1> sl = blk[0];
    if (blk != nblk) {
      const Word x = ow.read();
      const ap_uint<4> a = ((ap_uint<4>)sl << 3) | (ap_uint<4>)w.range(2, 0);
      for (int t = 0; t < 8; t++) {
#pragma HLS UNROLL
        if (w.range(5, 3) == t) kb[t][a] = x;
      }
    }
    if (blk != 0) {
      const ap_uint<4> a = ((ap_uint<4>)(sl ^ 1) << 3) | (ap_uint<4>)w.range(5, 3);
      Word o;
      for (int t = 0; t < 8; t++) {
#pragma HLS UNROLL
        o.range(16 * t + 15, 16 * t) = sel8(kb[t][a], w.range(2, 0));
      }
      ot.write(o);
    }
    w++;
    if (w == 0) {
      if (blk == nblk) break;
      blk++;
    }
  }
}

// ------------------------------------------------------------------------------------- packer and writer
struct UH {  // unit header: the unit's first word; last = end of the command (no words follow)
  waddr_t addr;
  bool last;
};

static inline Word four(ap_uint<32> a, ap_uint<32> b, ap_uint<32> c, ap_uint<32> d) {
#pragma HLS INLINE
  Word o;
  o.range(31, 0) = a;
  o.range(63, 32) = b;
  o.range(95, 64) = c;
  o.range(127, 96) = d;
  return o;
}

// Data words arrive in the final order of their units: F32 16 per segment (2 units), Q 8 per segment (1 unit), KW
// 64 per block (8 units). Scales go through a 16-word ring (4 lanes) and leave as 8-word units: Q per head every 32
// rows plus a shift-back tail; KW every 32 keys (NP % 32 == 0: no tail). With wsmall every word is its own unit.
static void pk_stage(const Args g, hls::stream<Word>& ot, hls::stream<ap_uint<32> >& ss, hls::stream<UH>& hd,
                     hls::stream<Word>& dd) {
  ap_uint<32> rg0[16], rg1[16], rg2[16], rg3[16];  // scale ring: lane l of word wi % 16
#pragma HLS BIND_STORAGE variable = rg0 type = ram_s2p impl = lutram
#pragma HLS BIND_STORAGE variable = rg1 type = ram_s2p impl = lutram
#pragma HLS BIND_STORAGE variable = rg2 type = ram_s2p impl = lutram
#pragma HLS BIND_STORAGE variable = rg3 type = ram_s2p impl = lutram
#pragma HLS DEPENDENCE variable = rg0 inter false
#pragma HLS DEPENDENCE variable = rg1 inter false
#pragma HLS DEPENDENCE variable = rg2 inter false
#pragma HLS DEPENDENCE variable = rg3 inter false
  const bool kw = g.out == OUT_KW, q = g.out == OUT_Q, f32 = g.out == OUT_F32;
  const ap_uint<7> last_w = f32 ? 15 : kw ? 63 : 7;  // last data word of a segment / block
  const ap_uint<5> step = kw ? 8 : 1;               // rows per data step
  cnt_t b = 0, n = 0;  // segment (F32, Q: row n) or block (KW: first key n)
  seg_t j = 0;
  ap_uint<7> w = 0;
  waddr_t ib = g.c, jb = g.c, sa = g.c, gb = g.c;  // item / head / segment or KW port array / KW group base
  ap_uint<2> p = 0;                                // KW port (n % 32) / 8
  waddr_t fi = g.f, fj = g.f;                      // scale vectors: item / head base
  bool sem = false, done = false;                  // emitting a scale unit; command finished
  waddr_t sua = 0;                                 // the scale unit's first word
  ap_uint<4> s0 = 0;                               // its first ring word
  ap_uint<5> sk = 0;                               // iterations into the unit: SDLY idle, then sul words
  const ap_uint<5> SDLY = 8;  // the ring words a unit reads were written >= 8 iterations earlier (DEPENDENCE false)
  const ap_uint<5> send = g.wsmall ? (ap_uint<5>)(SDLY + 1) : (ap_uint<5>)(SDLY + 8);
  // The step-boundary decisions (last row of the head / head / item, scale unit due and its first word) depend only on
  // n, j, b, which change only at a boundary and stay for >= 8 iterations (a step has >= 8 data words; a scale unit
  // emission leaves them alone). They are computed every iteration into registers (r_*) and used at the next boundary,
  // so the boundary iteration only muxes registered values (timing: the adder -> compare -> priority chain is cut).
  bool r_lastn = step == g.NP, r_lastj = g.H == 1, r_lastb = g.B == 1, r_due = false;
  ap_uint<15> r_u0 = 0;
pk:
  for (;;) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 16 max = 20000 avg = 3000
    // this step's boundary decisions from the registered n, j, b (taken over at the end of the iteration)
    const cnt_t nend = n + step;  // rows done in this head after this step
    const bool c_lastn = nend == g.NP, c_lastj = j == g.H - 1, c_lastb = b == g.B - 1;
    const ap_uint<15> wq = (ap_uint<15>)(n >> 2);  // Q: the scale word of row n
    const bool qword = q && (n.range(1, 0) == 3 || c_lastn);
    bool c_due = false;
    ap_uint<15> c_u0 = 0;
    if (kw && nend.range(4, 0) == 0) {  // 32 keys complete: words nend / 4 - 8 ..
      c_due = true;
      c_u0 = (ap_uint<15>)((nend >> 2) - 8);
    } else if (qword && g.wsmall) {
      c_due = true;
      c_u0 = wq;
    } else if (qword && wq.range(2, 0) == 7) {
      c_due = true;
      c_u0 = wq - 7;
    } else if (qword && c_lastn) {  // npw = wq + 1 >= 8, not a multiple of 8: shift-back tail
      c_due = true;
      c_u0 = g.npw - 8;
    }
    if (sem) {
      const ap_uint<4> ri = s0 + (ap_uint<4>)(ap_uint<5>)(sk - SDLY);  // ring word s0 + (sk - SDLY), mod 16
      if (sk == SDLY) {
        UH h;
        h.addr = sua;
        h.last = false;
        hd.write(h);
      }
      if (sk >= SDLY) dd.write(four(rg0[ri], rg1[ri], rg2[ri], rg3[ri]));
      sk++;
      if (sk == send) {
        sem = false;
        if (done) {
          UH h;
          h.addr = 0;
          h.last = true;
          hd.write(h);
          break;
        }
      }
    } else {
      const Word x = ot.read();
      if (g.wsmall || w.range(2, 0) == 0) {
        UH h;
        h.addr = sa + w;
        h.last = false;
        hd.write(h);
      }
      dd.write(x);
      // scales into the ring: KW key n + w during words 0..7 of its block, Q row n at the segment's last word
      const bool take = (kw && w < 8) || (q && w == 7);
      if (take) {
        const ap_uint<32> s = ss.read();
        const cnt_t key = kw ? (cnt_t)(n + w) : n;
        const ap_uint<4> ri = key.range(5, 2);
        const ap_uint<2> ln = key.range(1, 0);
        const bool padl = q && n == g.NP - 1;  // Q: pad lanes of the vector's last word are 1.0
        if (ln == 0) rg0[ri] = s;
        if (ln == 1 || (padl && ln < 1)) rg1[ri] = ln == 1 ? s : (ap_uint<32>)qz::ONE_BITS;
        if (ln == 2 || (padl && ln < 2)) rg2[ri] = ln == 2 ? s : (ap_uint<32>)qz::ONE_BITS;
        if (ln == 3 || padl) rg3[ri] = ln == 3 ? s : (ap_uint<32>)qz::ONE_BITS;
      }
      if (w == last_w) {
        w = 0;
        const bool lastn = r_lastn, lastj = r_lastj, lastb = r_lastb, due = r_due;
        if (due) {  // a scale unit follows this step
          sem = true;
          sk = 0;
          s0 = r_u0.range(3, 0);
          sua = fj + r_u0;
        }
        if (lastn && lastj && lastb) {
          done = true;
          if (!due) {
            UH h;
            h.addr = 0;
            h.last = true;
            hd.write(h);
            break;
          }
        } else if (!lastn) {  // next row / block of this head
          n = nend;
          if (kw && p == 3) {  // next 32-key group: port 0 of group + 1
            gb += 64;
            sa = gb;
          } else {
            sa += g.c_p;
          }
          p++;
        } else {
          n = 0;
          p = 0;
          if (!lastj) {
            j++;
            jb += g.segc;
            fj += g.segf;
          } else {
            j = 0;
            b++;
            ib += g.c_s;
            jb = ib;
            fi += g.f_s;
            fj = fi;
          }
          sa = jb;
          gb = jb;
        }
      } else {
        w++;
      }
    }
    r_lastn = c_lastn;
    r_lastj = c_lastj;
    r_lastb = c_lastb;
    r_due = c_due;
    r_u0 = c_u0;
  }
}

static void wr_stage(Word* ymem, bool wsmall, hls::stream<UH>& hd, hls::stream<Word>& dd) {
  if (wsmall) {
  wr1:
    for (;;) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 2 max = 4096 avg = 300
      const UH h = hd.read();
      if (h.last) break;
      ymem[h.addr] = dd.read();
    }
  } else {
  wr8:
    for (;;) {
#pragma HLS PIPELINE II = 8
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 4096 avg = 600
      const UH h = hd.read();
      if (h.last) break;
      for (int k = 0; k < 8; k++) {
#pragma HLS UNROLL
        ymem[h.addr + k] = dd.read();  // unconditional: one 8-beat burst, pipelined across units
      }
    }
  }
}

// ------------------------------------------------------------------------------------- the region
static void rope_df(const Word* hp, const Word* xmem, Word* ymem, const Args g, rcount_t segs, rcount_t nq,
                    rcount_t nwords, rcount_t nblk, bool quant, bool kw, ap_int<5> shift, bool wsmall) {
#pragma HLS DATAFLOW
  // one region run per command: the scalars are constant for the whole run (no propagation FIFOs)
#pragma HLS STABLE variable = g
#pragma HLS STABLE variable = segs
#pragma HLS STABLE variable = nq
#pragma HLS STABLE variable = nwords
#pragma HLS STABLE variable = nblk
#pragma HLS STABLE variable = quant
#pragma HLS STABLE variable = kw
#pragma HLS STABLE variable = shift
#pragma HLS STABLE variable = wsmall
  hls::stream<Word> xs("rope_xs"), cs("rope_cs"), fw("rope_fw"), ow("rope_ow"), ot("rope_ot"), dd("rope_dd");
  hls::stream<Rot> rs("rope_rs");
  hls::stream<abits_t> am("rope_am");
  hls::stream<SD> sdf("rope_sd");
  hls::stream<SP> spf("rope_sp");
  hls::stream<ap_uint<26> > rf("rope_rf");
  hls::stream<ap_uint<32> > ss("rope_ss");
  hls::stream<UH> hd("rope_hd");
#pragma HLS STREAM variable = xs depth = ROPE_D_XS
#pragma HLS STREAM variable = cs depth = ROPE_D_CS
#pragma HLS STREAM variable = rs depth = ROPE_D_RS
#pragma HLS STREAM variable = fw depth = ROPE_D_FW
#pragma HLS STREAM variable = am depth = ROPE_D_AM
#pragma HLS STREAM variable = sdf depth = ROPE_D_SD
#pragma HLS STREAM variable = spf depth = ROPE_D_SP
#pragma HLS STREAM variable = rf depth = ROPE_D_RF
#pragma HLS STREAM variable = ow depth = ROPE_D_OW
#pragma HLS STREAM variable = ot depth = ROPE_D_OT
#pragma HLS STREAM variable = ss depth = ROPE_D_SS
#pragma HLS STREAM variable = hd depth = ROPE_D_HD
#pragma HLS STREAM variable = dd depth = ROPE_D_DD
#pragma HLS BIND_STORAGE variable = xs type = fifo impl = lutram
#pragma HLS BIND_STORAGE variable = cs type = fifo impl = lutram
#pragma HLS BIND_STORAGE variable = fw type = fifo impl = lutram
#pragma HLS BIND_STORAGE variable = ow type = fifo impl = lutram
#pragma HLS BIND_STORAGE variable = ot type = fifo impl = lutram
#pragma HLS BIND_STORAGE variable = ss type = fifo impl = lutram
#pragma HLS BIND_STORAGE variable = dd type = fifo impl = lutram
#ifndef ROPE_DFSIM
  rd(xmem, hp, g, xs, cs);
  rot_stage(xs, cs, rs, segs);
  emit_stage(rs, fw, am, segs, quant);
  sc_stage(am, sdf, spf, nq, shift);
  rc_stage(sdf, rf, nq);
  qt_stage(fw, spf, rf, ow, ss, segs, quant);
  kw_stage(ow, ot, nwords, nblk, kw);
  pk_stage(g, ot, ss, hd, dd);
  wr_stage(ymem, wsmall, hd, dd);
#else  // threads on bounded FIFOs: a deadlock blocks a stream and dfsim/hls_stream.h reports it
  std::thread t0([&] { rd(xmem, hp, g, xs, cs); });
  std::thread t1([&] { rot_stage(xs, cs, rs, segs); });
  std::thread t2([&] { emit_stage(rs, fw, am, segs, quant); });
  std::thread t3([&] { sc_stage(am, sdf, spf, nq, shift); });
  std::thread t3b([&] { rc_stage(sdf, rf, nq); });
  std::thread t4([&] { qt_stage(fw, spf, rf, ow, ss, segs, quant); });
  std::thread t5([&] { kw_stage(ow, ot, nwords, nblk, kw); });
  std::thread t6([&] { pk_stage(g, ot, ss, hd, dd); });
  wr_stage(ymem, wsmall, hd, dd);
  t0.join();
  t1.join();
  t2.join();
  t3.join();
  t3b.join();
  t4.join();
  t5.join();
  t6.join();
#endif
}

// ------------------------------------------------------------------------------------- the mode function
// Returns ctr::ME_OK, or the ME_* of a record whose loop sizes are off (nothing read or written then). The sequencer
// has already run ctr::validate_hw, which rejects all of these (and much more) first.
static uint32_t rope_mode(const ctr::Cmd& c, const Word* hp, const Word* xmem, Word* ymem) {
#pragma HLS INLINE off
  const uint32_t K = c.K, N = c.N, B = c.B, NP = c.p[ctr::P_NP] ? c.p[ctr::P_NP] : c.N;
  const bool kw = c.flags & ctr::F_KW, q = c.flags & ctr::F_QOUT;
  const int32_t sh = (int32_t)c.p[ctr::P_SHIFT];
  if (c.op != ctr::OP_ROPE || c.p[ctr::P_SEG] != ctr::ROPE_SEG || K < ctr::ROPE_SEG || K % ctr::ROPE_SEG ||
      K / ctr::ROPE_SEG > ctr::SEG_MAX || N < 1 || N > ctr::N_MAX || B < 1 || B > ctr::B_MAX)
    return ctr::ME_DIMS;
  if (kw && q) return ctr::ME_FLAGS;
  if (NP < N || NP > ctr::N_MAX || (!kw && !q && NP != N) || sh < -(int32_t)ctr::SHIFT_MAX ||
      sh > (int32_t)ctr::SHIFT_MAX)
    return ctr::ME_PARAM;
  if (kw && NP % ctr::ENGINE_LANES) return ctr::ME_DIMS;
  Args g;
  g.out = kw ? OUT_KW : q ? OUT_Q : OUT_F32;
  g.B = B;
  g.N = N;
  g.NP = NP;
  g.H = K / ctr::ROPE_SEG;
  const ctr::Slot &A = c.s[ctr::SLOT_A], &C = c.s[ctr::SLOT_C], &E = c.s[ctr::SLOT_E], &F = c.s[ctr::SLOT_F];
  g.a = (waddr_t)(A.addr >> 4);
  g.e = (waddr_t)(E.addr >> 4);
  g.c = (waddr_t)(C.addr >> 4);
  g.f = (waddr_t)(F.addr >> 4);
  g.a_p = A.pitch >> 4;
  g.a_s = A.stride >> 4;
  g.e_p = E.pitch >> 4;
  g.e_s = E.stride >> 4;
  g.c_p = C.pitch >> 4;
  g.c_s = C.stride >> 4;
  g.f_s = F.stride >> 4;
  g.segc = c.p[ctr::P_SEGC] >> 4;
  g.segf = c.p[ctr::P_SEGF] >> 4;
  g.npw = (ap_uint<15>)((NP + 3) >> 2);
  g.wsmall = q && g.npw < 8;
  const rcount_t segs = (rcount_t)((ap_uint<34>)((ap_uint<17>)B * (ap_uint<17>)NP) * (ap_uint<7>)g.H);
  const bool quant = q || kw;
  const rcount_t nwords = kw ? (rcount_t)0 : q ? (rcount_t)(segs << 3) : (rcount_t)(segs << 4);
  rope_df(hp, xmem, ymem, g, segs, quant ? segs : (rcount_t)0, nwords, kw ? (rcount_t)(segs >> 3) : (rcount_t)0, quant,
          kw, (ap_int<5>)sh, g.wsmall);
  return ctr::ME_OK;
}

}  // namespace rope
