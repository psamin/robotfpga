// ROPE mode of float_top under the integration contract (fpga/kernels/CONTRACT.md 1.1, 2.1, 2.2, 3.3; opcode 0x60).
//
//   uint32_t rope::rope_mode(const ctr::Cmd& c, const Word* hp, const Word* xmem, Word* ymem)
//
// One call site in the sequencer's dispatch switch (`case ctr::OP_ROPE: return rope::rope_mode(c, w0, xmem, ymem);`),
// INLINE off, no m_axi adapter of its own: hp is the top's hp0 port (w0, read only: the RoPE table, a host-written
// constant), xmem / ymem are the top's io port (activations are read and written only on io). Every address is the
// record's absolute byte address (the pointer registers are 0), computed as a 128-bit word index of 36 bits
// (validate_hw guarantees addr < 2^40; all sums are formed in that width, no int).
//
// Record fields (validate_hw has checked them; rope_mode re-checks the few that size its loops and returns ME_* with
// nothing touched when they are off, so no record can hang the dataflow region):
//   K = H * 64 values per row (H = 1..64 heads, p1 = SEG = 64), N rows (1..65535), B items (1..65535)
//   A   FP32 rows: row (b, n) at A + b A.stride + n A.pitch, head j = values [64 j, 64 j + 64), natural order
//   E   table rows: row (b, n) at E + b E.stride + n E.pitch (pitch >= 256): words 0-7 NC[0..31] = round(-cos 2^17),
//       words 8-15 NS[0..31] = round(-sin 2^17), int32 slots of which bits [17:0] are used (rope_emu.table_rows)
//   p5 NP rows written (0 = N): rows [N, NP) are zero rows (q = 0, scale 1.0); they read nothing
//   output, one of:
//     (no flag)  FP32 rows: segment j of row n at C + b C.stride + n C.pitch + j p2, 64 FP32 (C == A in place is
//                allowed); NP == N
//     F_QOUT     int16: segment j of row n at C + b C.stride + n C.pitch + j p2 (64 int16), its scale at
//                F + b F.stride + j p3 + 4 n; each head's scale vector is written in whole words, pad4(NP) floats,
//                pad lanes 1.0. The contract quantizer (quantize/quant_int16.h) over each (row, head) of the FP32
//                rotation; stored scale s * 2^p4 (p4 = -3 for Q), an all-zero segment stores 1.0
//     F_KW       the KW weight tiling (as QUANT_W): key n, value d of head j in lane n % 8 of the word at
//                C + b C.stride + j p2 + ((n % 32) / 8) C.pitch + ((n / 32) 64 + d) 16; NP % 32 == 0; the per-key
//                scale (blob s_row) at F + b F.stride + j p3 + 4 n
//
// Processes (one DATAFLOW region, II = 1 each, 4 values per clock):
//   rd_x    xmem: token order, one burst of H * 16 words per row (KW: 8-key blocks, one 16-word burst per (key, head))
//   rd_t    hp:   one table row per row (KW: per (key, head))
//   rot_stage, emit_stage (rope_core.h): the rotation -> FP32 words in natural order + each segment's absmax
//   sc_stage  absmax -> s = qz::scale_bits (the contract's s), the divider seed
//   rc_stage  2 restoring dividers (1 quotient bit a clock each, QZ_RECIP_BITS = 1): R = floor(2^49 / S)
//   qt_stage  4 lanes of qz::quant_elem; 8 int16 per output word; FP32 words pass through without F_QOUT / F_KW
//   wr_out    ymem: FP32 rows | int16 segments + scale words | KW blocks (8 keys x 64 d, transposed on chip)
// rd_x / rd_t / sc / rc / qt / wr_out are the parts the shared row unit replaces (CONTRACT.md 4.1); rope_core.h's two
// stages are the rope-specific front end. ROPE_DFSIM builds every process as a thread on bounded FIFOs (dfsim/).
#pragma once
#include <stdint.h>

#include "../contract.h"
#include "../quantize/quant_int16.h"
#include "rope_core.h"

#ifdef ROPE_DFSIM
#include <thread>
#endif

// FIFO depths (also used by the bounded-FIFO model, dfsim/hls_stream.h)
#define ROPE_D_XS 32  // x words
#define ROPE_D_CS 32  // table words
#define ROPE_D_RS 4   // rotated groups
#define ROPE_D_FW 64  // FP32 words waiting for their segment's R (>= 16 needed; 64 covers sc + rc latency)
#define ROPE_D_AM 4
#define ROPE_D_SD 4
#define ROPE_D_SP 8
#define ROPE_D_RF 4
#define ROPE_D_OW 32  // output words (KW: covers the 64-clock transposed write of a block)
#define ROPE_D_SS 8   // stored scales

namespace rope {

typedef ap_uint<36> waddr_t;  // 128-bit word index: byte address bits [39:4]
typedef ap_uint<28> wstep_t;  // a u32 byte pitch / stride / segment step in words
typedef ap_uint<17> cnt_t;    // B, N, NP <= 65535 (17 bits: loop counters reach the bound)
typedef ap_uint<7> seg_t;     // H = K / 64 in 1..64

constexpr int OUT_F32 = 0, OUT_Q = 1, OUT_KW = 2;

struct Args {
  ap_uint<2> out;    // OUT_F32 / OUT_Q / OUT_KW
  cnt_t B, N, NP;    // items, rows read, rows written (KW / Q: NP >= N; F32: NP == N)
  seg_t H;           // heads per row
  rcount_t rows;     // B * NP * H segments
  waddr_t a, e, c, f;                                   // item 0, row 0
  wstep_t a_p, a_s, e_p, e_s, c_p, c_s, f_s, segc, segf;  // in words
  ap_uint<32> sh_add;  // p4 << 23: added to the bits of a non-zero segment's s
  bool contig;         // the segments of a row are adjacent in C (one burst per row)
};

// ------------------------------------------------------------------------------------- readers
// x rows in segment order. Token order (F32, Q): row (b, n) as one burst of H * 16 words; rows [N, NP) zeros.
// KW: per 8-key block n0 and head j, keys n0..n0+7 one 16-word burst each (keys >= N zeros).
static void rd_x(const Word* xmem, const Args g, hls::stream<Word>& xs) {
  const ap_uint<11> rw = (ap_uint<11>)g.H * WORDS;  // words per row, <= 1024
  waddr_t ib = g.a;
items:
  for (cnt_t b = 0; b < g.B; b++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 1
    if (g.out != OUT_KW) {
      waddr_t rb = ib;
    rows:
      for (cnt_t n = 0; n < g.N; n++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 50
      row:
        for (ap_uint<11> w = 0; w < rw; w++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 16 max = 240
          xs.write(xmem[rb + w]);
        }
        rb += g.a_p;
      }
    zrows:
      for (cnt_t n = g.N; n < g.NP; n++) {
#pragma HLS LOOP_TRIPCOUNT min = 0 max = 2
      zrow:
        for (ap_uint<11> w = 0; w < rw; w++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 16 max = 240
          xs.write(0);
        }
      }
    } else {
      waddr_t blk = ib;
    blocks:
      for (cnt_t n0 = 0; n0 < g.NP; n0 += 8) {
#pragma HLS LOOP_TRIPCOUNT min = 4 max = 8
        const ap_uint<4> nv = n0 >= g.N ? (ap_uint<4>)0 : (g.N - n0 >= 8) ? (ap_uint<4>)8 : (ap_uint<4>)(g.N - n0);
        waddr_t sb = blk;
      heads:
        for (seg_t j = 0; j < g.H; j++) {
#pragma HLS LOOP_TRIPCOUNT min = 5 max = 5
        keys:
          for (ap_uint<4> t = 0; t < nv; t++) {
#pragma HLS LOOP_TRIPCOUNT min = 0 max = 8
          kw:
            for (ap_uint<5> w = 0; w < WORDS; w++) {
#pragma HLS PIPELINE II = 1
              xs.write(xmem[sb + t * g.a_p + w]);
            }
          }
        zkeys:
          for (ap_uint<8> i = (ap_uint<8>)nv * WORDS; i < 8 * WORDS; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 0 max = 128
            xs.write(0);
          }
          sb += WORDS;
        }
        blk += g.a_p * 8;
      }
    }
    ib += g.a_s;
  }
}

// table rows, one per token (token order) or one per (key, head) (KW); rows >= N zeros.
static void rd_t(const Word* hp, const Args g, hls::stream<Word>& cs) {
  waddr_t ib = g.e;
titems:
  for (cnt_t b = 0; b < g.B; b++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 1
    if (g.out != OUT_KW) {
      waddr_t rb = ib;
    trows:
      for (cnt_t n = 0; n < g.N; n++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 50
      trow:
        for (ap_uint<5> w = 0; w < WORDS; w++) {
#pragma HLS PIPELINE II = 1
          cs.write(hp[rb + w]);
        }
        rb += g.e_p;
      }
    tzrows:
      for (cnt_t n = g.N; n < g.NP; n++) {
#pragma HLS LOOP_TRIPCOUNT min = 0 max = 2
      tzrow:
        for (ap_uint<5> w = 0; w < WORDS; w++) {
#pragma HLS PIPELINE II = 1
          cs.write(0);
        }
      }
    } else {
      waddr_t blk = ib;
    tblocks:
      for (cnt_t n0 = 0; n0 < g.NP; n0 += 8) {
#pragma HLS LOOP_TRIPCOUNT min = 4 max = 8
        const ap_uint<4> nv = n0 >= g.N ? (ap_uint<4>)0 : (g.N - n0 >= 8) ? (ap_uint<4>)8 : (ap_uint<4>)(g.N - n0);
      theads:
        for (seg_t j = 0; j < g.H; j++) {
#pragma HLS LOOP_TRIPCOUNT min = 5 max = 5
        tkeys:
          for (ap_uint<4> t = 0; t < nv; t++) {
#pragma HLS LOOP_TRIPCOUNT min = 0 max = 8
          tkw:
            for (ap_uint<5> w = 0; w < WORDS; w++) {
#pragma HLS PIPELINE II = 1
              cs.write(hp[blk + t * g.e_p + w]);
            }
          }
        tzkeys:
          for (ap_uint<8> i = (ap_uint<8>)nv * WORDS; i < 8 * WORDS; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 0 max = 128
            cs.write(0);
          }
        }
        blk += g.e_p * 8;
      }
    }
    ib += g.e_s;
  }
}

// ------------------------------------------------------------------------------------- contract quantizer, 4 lanes
struct SP {  // sc -> qt: the segment's s and whether it is all zero
  ap_uint<32> s;
  bool zero;
};
struct SD {  // sc -> rc: divider start
  ap_uint<24> S;
  qz::RecipSeed sd;
};

// absmax -> s = fl32(max(a, 1e-30) / 32767) (1.0 for a == 0), one segment a clock
static void sc_stage(hls::stream<abits_t>& am, hls::stream<SD>& sdf, hls::stream<SP>& spf, rcount_t n) {
sc:
  for (rcount_t r = 0; r < n; r++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 0 max = 1200
    const abits_t a = am.read();
    const ap_uint<32> s = qz::scale_bits(a);
    const qz::RowScale rs = qz::row_scale_from_s(s);
    SD d;
    d.S = rs.S;
    d.sd = qz::recip_seed(rs.S);
    sdf.write(d);
    SP p;
    p.s = s;
    p.zero = a == 0;
    spf.write(p);
  }
}

// R = min(floor(2^49 / S), 2^26 - 1): NE restoring dividers, segments dealt round robin (2 + RECIP_STEPS clocks
// each: 27 at 1 bit a clock), so one leaves every 13.5 clocks; a segment arrives every 16.
constexpr int NE = 2;
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
#pragma HLS LOOP_TRIPCOUNT min = 27 max = 20000
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

// 16 FP32 words per segment -> 8 int16 words (natural order: word i = values 8i..8i+7) + the stored scale
// (zero segment: 1.0; else s * 2^p4 by an exponent add). Without quant the FP32 words pass through.
static void qt_stage(hls::stream<Word>& fw, hls::stream<SP>& spf, hls::stream<ap_uint<26> >& rf, hls::stream<Word>& ow,
                     hls::stream<ap_uint<32> >& ss, rcount_t rows, bool quant, ap_uint<32> sh_add) {
  qz::RowScale rs;
  rs.s_bits = qz::ONE_BITS;
  rs.S = 1 << 23;
  rs.R = 0;
  rs.T = 1 << 24;
  rs.B = 150;
  ap_uint<32> stored = qz::ONE_BITS;
  ap_uint<64> lo = 0;
  ap_uint<4> w = 0;
  rcount_t r = 0;
  if (rows == 0) return;
qt:
  for (;;) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 16 max = 20000
    if (quant && w == 0) {
      const SP p = spf.read();
      rs = qz::row_scale_from_s(p.s);
      rs.R = rf.read();
      stored = p.zero ? (ap_uint<32>)qz::ONE_BITS : (ap_uint<32>)(p.s + sh_add);
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
      if (w == WORDS - 1) ss.write(stored);
    }
    if (w == WORDS - 1) {
      r++;
      if (r == rows) break;
    }
    w++;
  }
}

// ------------------------------------------------------------------------------------- writer
static inline Word four(ap_uint<32> a, ap_uint<32> b, ap_uint<32> c, ap_uint<32> d) {
#pragma HLS INLINE
  Word o;
  o.range(31, 0) = a;
  o.range(63, 32) = b;
  o.range(95, 64) = c;
  o.range(127, 96) = d;
  return o;
}

static void wr_out(Word* ymem, const Args g, hls::stream<Word>& ow, hls::stream<ap_uint<32> >& ss) {
  // Q: the scales of the current 4 rows, one bank per lane (row n % 4), indexed by head
  ap_uint<32> sb0[64], sb1[64], sb2[64], sb3[64];
#pragma HLS BIND_STORAGE variable = sb0 type = ram_s2p impl = lutram
#pragma HLS BIND_STORAGE variable = sb1 type = ram_s2p impl = lutram
#pragma HLS BIND_STORAGE variable = sb2 type = ram_s2p impl = lutram
#pragma HLS BIND_STORAGE variable = sb3 type = ram_s2p impl = lutram
  // KW: one 8-key x 64-value block, one bank per key (t), 8 int16 words each
  Word kb[8][8];
#pragma HLS ARRAY_PARTITION variable = kb dim = 1 complete
#pragma HLS BIND_STORAGE variable = kb type = ram_s2p impl = lutram
  ap_uint<32> ks[8];
#pragma HLS ARRAY_PARTITION variable = ks complete
  const ap_uint<11> rw16 = (ap_uint<11>)g.H * WORDS, rw8 = (ap_uint<11>)g.H * 8;
  waddr_t ib = g.c, fb = g.f;
witems:
  for (cnt_t b = 0; b < g.B; b++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 1
    if (g.out == OUT_F32) {
      waddr_t rb = ib;
    frows:
      for (cnt_t n = 0; n < g.NP; n++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 50
        if (g.contig) {
        frow:
          for (ap_uint<11> w = 0; w < rw16; w++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 16 max = 80
            ymem[rb + w] = ow.read();
          }
        } else {
          waddr_t sb = rb;
        fsegs:
          for (seg_t j = 0; j < g.H; j++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 5
          fseg:
            for (ap_uint<5> w = 0; w < WORDS; w++) {
#pragma HLS PIPELINE II = 1
              ymem[sb + w] = ow.read();
            }
            sb += g.segc;
          }
        }
        rb += g.c_p;
      }
    } else if (g.out == OUT_Q) {
      waddr_t rb = ib, fq = fb;
      ap_uint<2> t4 = 0;
    qrows:
      for (cnt_t n = 0; n < g.NP; n++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 52
        if (g.contig) {
        qrow:
          for (ap_uint<11> w = 0; w < rw8; w++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 8 max = 120
            ymem[rb + w] = ow.read();
            if (w.range(2, 0) == 7) {
              const ap_uint<32> s = ss.read();
              const ap_uint<6> j = w.range(8, 3);
              if (t4 == 0) sb0[j] = s;
              if (t4 == 1) sb1[j] = s;
              if (t4 == 2) sb2[j] = s;
              if (t4 == 3) sb3[j] = s;
            }
          }
        } else {
          waddr_t sb = rb;
        qsegs:
          for (seg_t j = 0; j < g.H; j++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 15
          qseg:
            for (ap_uint<4> w = 0; w < 8; w++) {
#pragma HLS PIPELINE II = 1
              ymem[sb + w] = ow.read();
            }
            const ap_uint<32> s = ss.read();
            const ap_uint<6> jj = j;
            if (t4 == 0) sb0[jj] = s;
            if (t4 == 1) sb1[jj] = s;
            if (t4 == 2) sb2[jj] = s;
            if (t4 == 3) sb3[jj] = s;
            sb += g.segc;
          }
        }
        if (t4 == 3 || n == g.NP - 1) {  // one scale word per head: rows 4 (n / 4) .. 4 (n / 4) + 3
          const ap_uint<32> one = qz::ONE_BITS;
          waddr_t fj = fq;
        qflush:
          for (seg_t j = 0; j < g.H; j++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 15
            const ap_uint<6> jj = j;
            ymem[fj] = four(sb0[jj], t4 >= 1 ? sb1[jj] : one, t4 >= 2 ? sb2[jj] : one, t4 == 3 ? sb3[jj] : one);
            fj += g.segf;
          }
          fq++;
        }
        t4++;
        rb += g.c_p;
      }
    } else {  // OUT_KW
      waddr_t gb = ib, fq = fb;  // group base (n0 / 32) 64 words; scale word n0 / 4
      ap_uint<2> p = 0;          // port (n0 % 32) / 8
    kblocks:
      for (cnt_t n0 = 0; n0 < g.NP; n0 += 8) {
#pragma HLS LOOP_TRIPCOUNT min = 4 max = 8
        waddr_t jb = gb + p * g.c_p, fj = fq;
      kheads:
        for (seg_t j = 0; j < g.H; j++) {
#pragma HLS LOOP_TRIPCOUNT min = 5 max = 5
        kin:
          for (ap_uint<7> i = 0; i < 64; i++) {  // key t = i / 8, its int16 word i % 8 (values 8 (i % 8) ..)
#pragma HLS PIPELINE II = 1
            const Word q = ow.read();
            const ap_uint<3> t = i.range(5, 3);
            for (int k = 0; k < 8; k++) {
#pragma HLS UNROLL
              if (t == k) kb[k][i.range(2, 0)] = q;
            }
            if (i.range(2, 0) == 7) {
              const ap_uint<32> s = ss.read();
              for (int k = 0; k < 8; k++) {
#pragma HLS UNROLL
                if (t == k) ks[k] = s;
              }
            }
          }
        kout:
          for (ap_uint<7> d = 0; d < 64; d++) {  // word d: lane t = key n0 + t, value d
#pragma HLS PIPELINE II = 1
            Word o;
            const ap_uint<3> ln = d.range(2, 0);
            for (int k = 0; k < 8; k++) {
#pragma HLS UNROLL
              const Word q = kb[k][d.range(5, 3)];
              ap_uint<16> v[8];  // an explicit 8:1 mux (a variable bit range becomes a 128-bit shifter)
#pragma HLS ARRAY_PARTITION variable = v complete
              for (int m = 0; m < 8; m++) {
#pragma HLS UNROLL
                v[m] = q.range(16 * m + 15, 16 * m);
              }
              o.range(16 * k + 15, 16 * k) = v[ln];
            }
            ymem[jb + d] = o;
          }
        ksc:
          for (ap_uint<2> h = 0; h < 2; h++) {
#pragma HLS PIPELINE II = 1
            ymem[fj + h] = h == 0 ? four(ks[0], ks[1], ks[2], ks[3]) : four(ks[4], ks[5], ks[6], ks[7]);
          }
          jb += g.segc;
          fj += g.segf;
        }
        fq += 2;
        if (p == 3) gb += 64;
        p++;
      }
    }
    ib += g.c_s;
    fb += g.f_s;
  }
}

// ------------------------------------------------------------------------------------- the region
static void rope_df(const Word* hp, const Word* xmem, Word* ymem, const Args g, rcount_t rows, ap_uint<7> trep,
                    bool quant, rcount_t nq, ap_uint<32> sh_add) {
#pragma HLS DATAFLOW
  hls::stream<Word> xs("rope_xs"), cs("rope_cs"), fw("rope_fw"), ow("rope_ow");
  hls::stream<Rot> rs("rope_rs");
  hls::stream<abits_t> am("rope_am");
  hls::stream<SD> sdf("rope_sd");
  hls::stream<SP> spf("rope_sp");
  hls::stream<ap_uint<26> > rf("rope_rf");
  hls::stream<ap_uint<32> > ss("rope_ss");
#pragma HLS STREAM variable = xs depth = ROPE_D_XS
#pragma HLS STREAM variable = cs depth = ROPE_D_CS
#pragma HLS STREAM variable = rs depth = ROPE_D_RS
#pragma HLS STREAM variable = fw depth = ROPE_D_FW
#pragma HLS STREAM variable = am depth = ROPE_D_AM
#pragma HLS STREAM variable = sdf depth = ROPE_D_SD
#pragma HLS STREAM variable = spf depth = ROPE_D_SP
#pragma HLS STREAM variable = rf depth = ROPE_D_RF
#pragma HLS STREAM variable = ow depth = ROPE_D_OW
#pragma HLS STREAM variable = ss depth = ROPE_D_SS
#pragma HLS BIND_STORAGE variable = fw type = fifo impl = lutram
#pragma HLS BIND_STORAGE variable = xs type = fifo impl = srl
#pragma HLS BIND_STORAGE variable = cs type = fifo impl = srl
#pragma HLS BIND_STORAGE variable = ow type = fifo impl = srl
#ifndef ROPE_DFSIM
  rd_x(xmem, g, xs);
  rd_t(hp, g, cs);
  rot_stage(xs, cs, rs, rows, trep);
  emit_stage(rs, fw, am, rows, quant);
  sc_stage(am, sdf, spf, nq);
  rc_stage(sdf, rf, nq);
  qt_stage(fw, spf, rf, ow, ss, rows, quant, sh_add);
  wr_out(ymem, g, ow, ss);
#else  // threads on bounded FIFOs: a deadlock blocks a stream and dfsim/hls_stream.h reports it
  std::thread t0([&] { rd_x(xmem, g, xs); });
  std::thread t1([&] { rd_t(hp, g, cs); });
  std::thread t2([&] { rot_stage(xs, cs, rs, rows, trep); });
  std::thread t3([&] { emit_stage(rs, fw, am, rows, quant); });
  std::thread t4([&] { sc_stage(am, sdf, spf, nq); });
  std::thread t5([&] { rc_stage(sdf, rf, nq); });
  std::thread t6([&] { qt_stage(fw, spf, rf, ow, ss, rows, quant, sh_add); });
  wr_out(ymem, g, ow, ss);
  t0.join();
  t1.join();
  t2.join();
  t3.join();
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
  if (c.op != ctr::OP_ROPE || c.p[ctr::P_SEG] != ctr::ROPE_SEG || K < ctr::ROPE_SEG || K % ctr::ROPE_SEG ||
      K / ctr::ROPE_SEG > ctr::SEG_MAX || N < 1 || N > ctr::N_MAX || B < 1 || B > ctr::B_MAX)
    return ctr::ME_DIMS;
  if (kw && q) return ctr::ME_FLAGS;
  if (NP < N || NP > ctr::N_MAX || (!kw && !q && NP != N)) return ctr::ME_PARAM;
  if (kw && NP % ctr::ENGINE_LANES) return ctr::ME_DIMS;
  Args g;
  g.out = kw ? OUT_KW : q ? OUT_Q : OUT_F32;
  g.B = B;
  g.N = N;
  g.NP = NP;
  g.H = K / ctr::ROPE_SEG;
  g.rows = (rcount_t)((ap_uint<34>)((ap_uint<17>)B * (ap_uint<17>)NP) * (ap_uint<7>)g.H);
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
  g.sh_add = (ap_uint<32>)(c.p[ctr::P_SHIFT] << 23);
  g.contig = g.H == 1 || g.segc == (g.out == OUT_F32 ? WORDS : 8);
  const bool quant = g.out != OUT_F32;
  const ap_uint<7> trep = g.out == OUT_KW ? (ap_uint<7>)1 : (ap_uint<7>)g.H;
  rope_df(hp, xmem, ymem, g, g.rows, trep, quant, quant ? g.rows : (rcount_t)0, g.sh_add);
  return ctr::ME_OK;
}

}  // namespace rope
