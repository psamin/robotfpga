// The float_top row unit's backbone (CONTRACT.md 4.1) with its identity front end: opcodes QUANT_X, QUANT_W,
// QUANT_WT and COPY of fpga/kernels/contract.h. It takes a decoded ctr::Cmd and the top's io pointers:
//
//   uint32_t qz::row_mode(const ctr::Cmd& c, const word_t* xmem, word_t* ymem)    (INLINE off: one instance)
//   qz::RowUnit::row(c, hp0, hp1, hp2, hp3, io_rd, io_wr)   the sequencer's unit signature (seq_core.h), same call
//
// It is called from the sequencer's dispatch switch after ctr::validate_hw(c) == ME_OK, from one call site for all
// of the unit's opcodes. It returns ME_OK, or ME_DIMS without touching memory for an opcode of another unit. Every
// slot address is an absolute byte address (the pointer registers are 0 in list mode). Activations are read only
// through xmem and everything is written only through ymem, both on the io bundle (CONTRACT.md 3.3). The unit has no
// m_axi adapter, no decode and no register of its own; hp0..hp3 are not used by these four opcodes.
//
// Layouts (CONTRACT.md 1.1, all little-endian, every address / pitch / stride a multiple of 16 bytes):
//   A  FP32 rows: value (b, n, k) at A + b A.stride + n A.pitch + 4k
//   C  QUANT_X: int16 rows; segment j (p1 = 64) of row n at C + b C.stride + n C.pitch + j p2 (p1 = 0: the row)
//      QUANT_W: KW tiling, word (n, d) of segment j at C + b C.stride + j p2 + ((n%32)/8) C.pitch
//               + ((n/32) SEG + d) 16, lane n%8, SEG = p1 ? 64 : K                    (rows n < NP, NP % 32 == 0)
//      QUANT_WT: VW tiling, word (c, n) of segment j at C + b C.stride + j p2 + ((c%32)/8) C.pitch
//               + ((c/32) NP + n) 16, lane c%8                                         (keys n < NP, NP % 8 == 0)
//      COPY: FP32 rows as A
//   F  stored scales (stored_scale: s 2^p4, zero rows 1.0): QUANT_X / QUANT_W per (row n, segment j) at
//      F + b F.stride + j p3 + 4n, pad4(NP) floats per vector, pad lanes 1.0; QUANT_WT per channel c of segment
//      j at F + b F.stride + j p3 + 4c. Rows / keys [N, NP) are written as zeros with scale 1.0.
//
// Every DDR access is a fixed-size unit inside a pipelined loop, which lets HLS keep requests in flight across rows
// (a variable-length burst per row instead pays the whole DDR latency on every row). Reads: 16 words (64 FP32), at
// any row pitch and in any order. Writes: 8 words (64 int16 / 32 FP32 / 32 scales). A row or vector whose length is
// not a multiple of the unit ends with a "shift-back" unit that covers its last 16 (8) words, re-reading (re-writing
// with the same data) the overlap, so nothing outside the declared extents is ever read or written. Rows shorter
// than one unit (K < 64 read, int16 rows of K < 64, FP32 rows of K < 32, scale vectors of NP <= 28) use single-word
// units (slower, never in the -12392 hot path).
//
// Processes (one DATAFLOW region per command; every loop II = 1 except the m_axi unit loops):
//   rd   xmem -> tokens: 16-word units in the op's order. QUANT_X: rows (b, n); rows of more than ROW_BUF = 3,072
//        values twice (an absmax pass, then a data pass: CONTRACT.md ROW_BUF). Segmented QUANT_X and QUANT_W:
//        (b, j, n) segment-major. QUANT_WT: per item a column-absmax pass and a data pass per 64-column chunk,
//        interleaved P1(0) P1(1) P2(0) P1(2) P2(1) ... so a chunk's scales are ready before its data. COPY: (b, n).
//   [front end]  identity here: the slot where NORM / GELU / SILU / ROPE put their FP32 producers (README)
//   amx  per-segment absmax (QUANT_WT: per column, 64 at a time) -> am; data words -> d, the row buffer
//        (FIFO 1,024 x 128 bit in BRAM: one QUANT_X row of up to 3,072 values)
//   ss   absmax -> s = scale_bits, the stored scale (stored_scale with p4)
//   sr   two 1-bit-a-clock restoring dividers (QZ_RECIP_BITS = 1): R = floor(2^49 / S), one scale per 13.5 clocks
//   qt   4 lanes of quant_elem a clock (QUANT_WT: each lane its own column scale from a 64-entry table)
//   wb   int16 / FP32 words and stored scales -> 8-word units with their addresses, in qt's order: rows
//        (shift-back tail through an 8-word ring, zero rows), KW (8 rows x 64 int16 transposed into 64 words of 8
//        rows) and VW (8 keys x 64 channels reordered into 8-word key runs) through one 8 x 8-word block, and the
//        scale vectors (32 scales a unit; pad lanes and rows [N, NP) 1.0)
//   wr   units -> ymem (header + 8 words a unit)
// Throughput: 1 input word (4 values) a clock in, 4 int16 (half a word) a clock out.
#pragma once
#include <hls_stream.h>
#ifdef QZ_DFSIM
#include <thread>
#endif

#include "../contract.h"
#include "quant_int16.h"

namespace qz {

typedef ap_uint<36> wa_t;  // 128-bit word index = byte address / 16, i.e. byte addresses < 2^40 = ctr::ADDR_LIMIT.
                           // CONTRACT.md 3.2: validate_hw bounds every slot address below 2^40 and the host's
                           // validate() every extent end, so after that check the datapath may be 40 bits wide. A
                           // list that skipped validate() can wrap inside the 2^40 space; it cannot hang the unit.
typedef ap_uint<28> wp_t;  // a pitch / stride / segment step in words (u32 bytes / 16)
constexpr unsigned ROWBUF_W = ctr::ROW_BUF / 4;  // 768 words: QUANT_X rows up to 3,072 values are read once

#ifndef QZ_SC_ENGINES
#define QZ_SC_ENGINES 2  // reciprocal engines in sr: 2 give one scale per 13.5 clocks (one per 16 words is needed)
#endif
constexpr int NE = QZ_SC_ENGINES;
static_assert(NE == 2 || NE == 4, "QZ_SC_ENGINES: 2 or 4");

// stream depths (named so the threaded deadlock model in tb_quant.cpp uses the same numbers)
#ifndef QZ_D_XS
#define QZ_D_XS 16     // rd -> amx tokens
#endif
#ifndef QZ_D_AM
#define QZ_D_AM 64     // amx -> ss: holds one chunk's 64 column absmax (QUANT_WT, see amx)
#endif
#ifndef QZ_D_D
#define QZ_D_D 1024    // amx -> qt row buffer (BRAM): must hold the longest row read once (ROWBUF_W = 768 words);
#endif                 // 1,024 x 128 bit is 4 BRAM36 either way
#ifndef QZ_D_SS
#define QZ_D_SS 4      // ss -> sr
#endif
#ifndef QZ_D_SF
#define QZ_D_SF 4      // sr -> qt scale tokens
#endif
#ifndef QZ_D_OW
#define QZ_D_OW 64     // qt -> wb int16 / FP32 words: one 8 x 8 block, so qt runs on while wb drains a block
#endif
#ifndef QZ_D_SQ
#define QZ_D_SQ 64     // qt -> wb stored scales (QUANT_WT writes 64 in a row)
#endif
#ifndef QZ_D_HD
#define QZ_D_HD 4      // wb -> wr unit headers
#endif
#ifndef QZ_D_DD
#define QZ_D_DD 16     // wb -> wr unit words
#endif
static_assert(QZ_D_D >= ROWBUF_W, "the row buffer must hold one row read once (deadlock otherwise)");
static_assert(QZ_D_AM >= 64, "amx emits a QUANT_WT chunk's 64 column absmax in one go");

// what the processes do (row_plan derives it from ctr::Cmd)
constexpr uint8_t KX = 0;   // QUANT_X, whole rows
constexpr uint8_t KXS = 1;  // QUANT_X, 64-value segments
constexpr uint8_t KW = 2;   // QUANT_W
constexpr uint8_t KT = 3;   // QUANT_WT
constexpr uint8_t KC = 4;   // COPY

// The command as the processes need it. Passed by value into the DATAFLOW region and marked STABLE there (it does
// not change while the region runs), so HLS adds no per-field channels.
struct RP {
  uint8_t kind;
  ap_uint<17> B, N, NP;     // items, rows (keys) per item, rows (keys) written (>= N); all <= 65535
  ap_uint<13> W;            // FP32 words per input row (K / 4 <= 4096)
  ap_uint<13> WO;           // output words per row: KX K / 8, KC K / 4, KXS 8 (one segment)
  ap_uint<9> nseg;          // KXS / KW: segments of 64 values (<= 64); KT: column chunks of 64 (<= 256)
  bool half;                // KT: the last chunk has 32 columns
  bool kseg;                // KT: p1 = 64 (chunk cc is segment cc) - else p1 = 0 (one segment of K columns)
  bool rsmall, wsmall;      // single-word read units (W < 16) / write units
  bool twopass;             // KX: rows of more than ROWBUF_W words are read twice
  ap_int<5> shift;          // p4
  wa_t a, c, f;             // slot A, C, F addresses (words)
  wp_t ap, as, cp, cs, fs;  // pitches / strides (words)
  wp_t segc, segf;          // p2 / p3 (words)
  ap_uint<30> cp2, cp3;     // 2 C.pitch, 3 C.pitch (KW / VW port offsets)
};

// ---------------------------------------------------------------- stream tokens
struct XT {             // rd -> amx (the front-end slot)
  word_t w;
  ap_uint<1> fwd;       // data word: to the row buffer
  ap_uint<1> amx;       // counts for the absmax (per segment, or per column in a QUANT_WT pass 1)
  ap_uint<1> last;      // last word of a segment (absmax out) / of a QUANT_WT chunk's pass 1
  ap_uint<1> lam;       // with last: the command's last absmax
  ap_uint<1> end;       // last token of the command
  ap_uint<4> col;       // QUANT_WT pass 1: word position inside the 64-column chunk
};
struct AM {             // amx -> ss
  abits_t a;
  ap_uint<1> end;
};
struct SS {             // ss -> sr: s = fl32(max(a, 1e-30) / 32767) (1.0 for a == 0) as RowScale fields
  ap_uint<24> S;
  ap_uint<25> T;
  ap_uint<9> B;
#ifdef QZ_IMPL_FDIV
  ap_uint<32> s;
#endif
  RecipSeed seed;       // recip_seed(S): the divider's start state
  ap_uint<32> st;       // the stored scale
  ap_uint<1> end;
};
struct ST {             // sr -> qt: the quantizer's row parameters (RowScale) + the stored scale
  ap_uint<24> S;
  ap_uint<26> R;
  ap_uint<25> T;
  ap_uint<9> B;
#ifdef QZ_IMPL_FDIV
  ap_uint<32> s;
#endif
  ap_uint<32> st;
  ap_uint<1> end;
};
struct UH {             // unit header (wb -> wr)
  wa_t addr;            // first word
  ap_uint<1> last;      // end of the command (no words follow)
};

// ---------------------------------------------------------------- rd: xmem -> tokens
static void qz_rd(const word_t* mem, const RP p, hls::stream<XT>& xs) {
  const bool kt = p.kind == KT, kc = p.kind == KC, kx = p.kind == KX;
  const bool segmaj = p.kind == KXS || p.kind == KW;
  if (p.rsmall) {  // rows of < 64 values (KX, KC, KT with K = 32): single words, (b, [pass,] n, w)
    ap_uint<17> b = 0, n = 0;
    ap_uint<4> w = 0;
    ap_uint<1> pass = 0;  // KT: 0 = column absmax, 1 = data
    wa_t ib = p.a, row = p.a;
  rd1:
    for (;;) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 2 max = 4096 avg = 400
      XT t;
      t.w = mem[row + w];
      const bool rowend = w == p.W - 1, lastrow = n == p.N - 1, lastitem = b == p.B - 1;
      t.fwd = !kt || pass == 1;
      t.amx = kx || (kt && pass == 0);
      t.last = kx ? rowend : (kt && pass == 0 && rowend && lastrow);
      t.lam = t.last && lastitem && (kt || lastrow);
      t.col = w;
      const bool itemend = rowend && lastrow && (!kt || pass == 1);
      t.end = itemend && lastitem;
      xs.write(t);
      if (t.end) break;
      if (!rowend) {
        w++;
      } else {
        w = 0;
        if (!lastrow) {
          n++;
          row += p.ap;
        } else {
          n = 0;
          if (kt && pass == 0) {
            pass = 1;
            row = ib;
          } else {
            pass = 0;
            b++;
            ib += p.as;
            row = ib;
          }
        }
      }
    }
    return;
  }
  // 16-word units. Loop order (b, j, n, pass, u): j is the KXS / KW segment or the QUANT_WT step (c1 / c2 count the
  // chunks whose pass 1 / pass 2 is done), pass the QUANT_X re-read, u the unit in a QUANT_X / COPY row.
  const ap_uint<9> nu = (p.W + 15) >> 4;  // KX / KC units per row
  const ap_uint<4> tail = p.W(3, 0);     // KX / KC: words of the last unit that are new (0 = all 16)
  const ap_uint<9> nl = p.nseg - 1;
  ap_uint<17> b = 0, n = 0;
  ap_uint<9> j = 0, u = 0, c1 = 0, c2 = 0;
  bool pass = false;
  wa_t ib = p.a, rb = p.a;  // item base; row base
rd16:
  for (;;) {
#pragma HLS PIPELINE II = 16
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 65536 avg = 1500
    // QUANT_WT step: P1(c1) while c1 < nseg and c1 - c2 < 2, else P2(c2): P1(0) P1(1) P2(0) P1(2) P2(1) ... P2(nl)
    const bool p1 = kt && c1 != p.nseg && ap_uint<9>(c1 - c2) != 2;
    const ap_uint<9> cc = p1 ? c1 : c2;
    const bool lastunit = u == nu - 1;
    // the unit's first word and the number of leading words that are not new (shift-back)
    ap_uint<13> uo;
    ap_uint<5> skip = 0;
    if (segmaj) {
      uo = ap_uint<13>(j) << 4;
    } else if (kt) {
      if (p.half && cc == nl) {
        uo = p.W - 16;
        skip = 8;
      } else {
        uo = ap_uint<13>(cc) << 4;
      }
    } else if (lastunit && tail != 0) {
      uo = p.W - 16;
      skip = ap_uint<5>(16) - tail;
    } else {
      uo = ap_uint<13>(u) << 4;
    }
    const wa_t addr = rb + uo;
    const bool lastrow = n == p.N - 1, lastitem = b == p.B - 1;
    const bool rowdone = segmaj || kt || lastunit;            // this pass over the row is done
    const bool rowfin = rowdone && (!p.twopass || pass);      // the row is done (KX two-pass: both passes)
    const bool lastj = segmaj ? j == nl : kt ? (!p1 && c2 == nl) : true;
    const bool fin = rowfin && lastrow && lastj && lastitem;
    const bool amx = kx ? (!p.twopass || !pass) : kt ? p1 : segmaj;
    const bool last = kx ? (lastunit && amx) : kt ? (p1 && lastrow) : segmaj;
    const bool lam = last && lastitem && (kt ? cc == nl : lastrow && (!segmaj || j == nl));
    for (int k = 0; k < 16; k++) {
#pragma HLS UNROLL
      const word_t v = mem[addr + k];  // unconditional: one 16-beat burst, pipelined across iterations
      if (k >= skip) {
        XT t;
        t.w = v;
        t.fwd = kx ? (!p.twopass || pass) : kt ? !p1 : true;
        t.amx = amx;
        t.last = k == 15 && last;
        t.lam = k == 15 && lam;
        t.col = ap_uint<4>(k - skip);
        t.end = k == 15 && fin;
        xs.write(t);
      }
    }
    if (fin) break;
    // advance: u (KX / KC), then the KX re-read, then n, then j (KXS / KW / KT), then b
    if (!rowdone) {
      u++;
    } else {
      u = 0;
      if (p.twopass && !pass) {
        pass = true;
      } else {
        pass = false;
        if (!lastrow) {
          n++;
          rb += p.ap;
        } else {
          n = 0;
          if (!lastj) {
            if (segmaj) j++;
            else if (p1) c1++;
            else c2++;
          } else {
            j = 0;
            c1 = 0;
            c2 = 0;
            b++;
            ib += p.as;
          }
          rb = ib;
        }
      }
    }
  }
}

// ---------------------------------------------------------------- amx: absmax + row buffer
static void qz_amx(const RP p, hls::stream<XT>& xs, hls::stream<AM>& am, hls::stream<word_t>& d) {
  const bool kt = p.kind == KT;
  abits_t cm[4][16];  // QUANT_WT: absmax of column 4 w + l of the current 64-column chunk
#pragma HLS ARRAY_PARTITION variable = cm dim = 1 complete
#pragma HLS BIND_STORAGE variable = cm type = ram_2p impl = lutram
#pragma HLS DEPENDENCE variable = cm inter false  // a column recurs after >= 8 words (one chunk row)
  if (kt) {
  clr:
    for (int w = 0; w < 16; w++) {
#pragma HLS PIPELINE II = 1
      for (int l = 0; l < 4; l++) {
#pragma HLS UNROLL
        cm[l][w] = 0;
      }
    }
  }
  abits_t m = 0;
  ap_uint<9> cc = 0;  // KT: chunk whose pass 1 is running
outer:
  for (;;) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 512 avg = 4
    bool pass_end = false, end = false, lam = false;
  words:
    for (;;) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 2 max = 1048576 avg = 20000
      const XT t = xs.read();
      if (t.fwd) d.write(t.w);
      if (t.amx) {
        if (kt) {
          for (int l = 0; l < 4; l++) {
#pragma HLS UNROLL
            const abits_t x = abs_bits(word_f32(t.w, l)), o = cm[l][t.col];
            cm[l][t.col] = amax2(o, x);
          }
        } else {
          const abits_t mm = amax2(m, word_amax(t.w));
          if (t.last) {
            AM o;
            o.a = mm;
            o.end = t.lam;
            am.write(o);
            m = 0;
          } else {
            m = mm;
          }
        }
      }
      pass_end = kt && t.amx && t.last;
      lam = t.lam;
      end = t.end;
      if (pass_end || end) break;
    }
    if (pass_end) {  // the chunk's column absmax, in column order, and clear for the next chunk
      const ap_uint<7> cw = (p.half && cc == p.nseg - 1) ? 32 : 64;
    emit:
      for (ap_uint<7> c = 0; c < cw; c++) {
#pragma HLS PIPELINE II = 1
        AM o;
        o.a = cm[c(1, 0)][c(5, 2)];
        o.end = lam && c == cw - 1;
        am.write(o);
        cm[c(1, 0)][c(5, 2)] = 0;
      }
      cc = cc == p.nseg - 1 ? ap_uint<9>(0) : ap_uint<9>(cc + 1);
    }
    if (end) break;
  }
}

// ---------------------------------------------------------------- ss: absmax -> s and the stored scale, one a clock
static void qz_ss(const RP p, hls::stream<AM>& am, hls::stream<SS>& ss) {
  if (p.kind == KC) return;
scales:
  for (;;) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 65536 avg = 2000
    const AM t = am.read();
    const ap_uint<32> s = scale_bits(t.a);
    const RowScale rs = row_scale_from_s(s);
    SS o;
    o.S = rs.S;
    o.T = rs.T;
    o.B = rs.B;
#ifdef QZ_IMPL_FDIV
    o.s = s;
#endif
    o.seed = recip_seed(rs.S);
    o.st = stored_scale(t.a, s, p.shift);
    o.end = t.end;
    ss.write(o);
    if (t.end) break;
  }
}

// ---------------------------------------------------------------- sr: R = floor(2^49 / S), NE restoring dividers
// Each engine runs recip_step (quant_int16.h, 1 quotient bit a clock) 25 times on its own S. Inputs go to the
// engines round robin and leave in the same order. ss is read without blocking, so a result waiting for qt never
// stops the engines from taking new work (and qt waiting for a result never waits on ss).
static void qz_sr(const RP p, hls::stream<SS>& ss, hls::stream<ST>& sf) {
  if (p.kind == KC) return;
  RecipState es[NE];
  SS ev[NE];  // the engine's job (S, T, B, stored scale, end)
  bool busy[NE];
#pragma HLS ARRAY_PARTITION variable = es complete
#pragma HLS ARRAY_PARTITION variable = ev complete
#pragma HLS ARRAY_PARTITION variable = busy complete
  for (int e = 0; e < NE; e++) {
#pragma HLS UNROLL
    busy[e] = false;
  }
  ap_uint<2> ein = 0, eout = 0;
  bool got_end = false;
recip:
  for (;;) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 27 max = 1000000 avg = 30000
    // every decision below uses the state at the start of the clock: an engine finished last clock leaves now, a
    // free engine takes a job now, the others step
    const ap_uint<2> eo = eout & (NE - 1), ei = ein & (NE - 1);
    const bool out = busy[eo] && es[eo].n == 0;
    const bool can_in = !got_end && !busy[ei];
    bool fin = false;
    if (out) {
      ST o;
      o.S = ev[eo].S;
      o.R = recip_result(es[eo]);
      o.T = ev[eo].T;
      o.B = ev[eo].B;
#ifdef QZ_IMPL_FDIV
      o.s = ev[eo].s;
#endif
      o.st = ev[eo].st;
      o.end = ev[eo].end;
      sf.write(o);
      fin = ev[eo].end;
    }
    for (int e = 0; e < NE; e++) {
#pragma HLS UNROLL
      if (busy[e]) recip_step(es[e], ev[e].S);  // a no-op once its 25 bits are out (n == 0)
    }
    if (out) {
      busy[eo] = false;
      eout++;
    }
    if (fin) break;
    if (can_in) {
      SS t;
      if (ss.read_nb(t)) {
        ev[ei] = t;
        es[ei] = recip_load(t.seed);
        busy[ei] = true;
        got_end = t.end;
        ein++;
      }
    }
  }
}

// ---------------------------------------------------------------- qt: 4 lanes
static void qz_qt(const RP p, hls::stream<word_t>& d, hls::stream<ST>& sf, hls::stream<word_t>& ow,
                  hls::stream<ap_uint<32>>& sq) {
  if (p.kind == KC) {  // FP32 rows straight through
    ap_uint<13> w = 0;
    ap_uint<17> n = 0, b = 0;
  copy:
    for (;;) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 2 max = 100000 avg = 20000
      ow.write(d.read());
      if (w != p.W - 1) {
        w++;
      } else {
        w = 0;
        if (n != p.N - 1) {
          n++;
        } else {
          n = 0;
          if (b == p.B - 1) break;
          b++;
        }
      }
    }
    return;
  }
  const bool kt = p.kind == KT;
  const ap_uint<13> SW = p.kind == KX ? p.W : ap_uint<13>(16);  // words per segment (KX: the row)
  ap_uint<24> tS[4][16];  // QUANT_WT: the row parameters of column 4 w + l of the chunk
  ap_uint<26> tR[4][16];
  ap_uint<25> tT[4][16];
  ap_uint<9> tB[4][16];
#pragma HLS ARRAY_PARTITION variable = tS dim = 1 complete
#pragma HLS ARRAY_PARTITION variable = tR dim = 1 complete
#pragma HLS ARRAY_PARTITION variable = tT dim = 1 complete
#pragma HLS ARRAY_PARTITION variable = tB dim = 1 complete
#pragma HLS BIND_STORAGE variable = tS type = ram_2p impl = lutram
#pragma HLS BIND_STORAGE variable = tR type = ram_2p impl = lutram
#pragma HLS BIND_STORAGE variable = tT type = ram_2p impl = lutram
#pragma HLS BIND_STORAGE variable = tB type = ram_2p impl = lutram
#ifdef QZ_IMPL_FDIV
  ap_uint<32> tF[4][16];
#pragma HLS ARRAY_PARTITION variable = tF dim = 1 complete
  ap_uint<32> cs = ONE_BITS;
#endif
  ap_uint<64> lo = 0;      // the 4 int16 of the even word of a pair
  ap_uint<13> wi = 0;      // word in the segment (KT: in the chunk row)
  ap_uint<17> n = 0, b = 0;
  ap_uint<9> cc = 0;       // KT chunk
  ap_uint<7> lc = 0;       // KT: table entries loaded
  bool endseg = false;     // KX / KXS / KW: the current segment is the command's last
  bool load = kt;          // KT: loading the chunk's column scales
  ap_uint<7> cw = (p.half && p.nseg == 1) ? 32 : 64;
quant:
  for (;;) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 2 max = 1048576 avg = 20000
#pragma HLS DEPENDENCE variable = tS inter false  // an entry is rewritten >= 2 clocks after its last read (a row /
#pragma HLS DEPENDENCE variable = tR inter false  // segment is >= 2 words); within a clock the write comes first
#pragma HLS DEPENDENCE variable = tT inter false
#pragma HLS DEPENDENCE variable = tB inter false
    if (load) {
      const ST t = sf.read();
      tS[lc(1, 0)][lc(5, 2)] = t.S;
      tR[lc(1, 0)][lc(5, 2)] = t.R;
      tT[lc(1, 0)][lc(5, 2)] = t.T;
      tB[lc(1, 0)][lc(5, 2)] = t.B;
#ifdef QZ_IMPL_FDIV
      tF[lc(1, 0)][lc(5, 2)] = t.s;
#endif
      sq.write(t.st);
      if (lc == cw - 1) {
        lc = 0;
        load = false;
      } else {
        lc++;
      }
    } else {
      if (!kt && wi == 0) {  // a new row / segment: its parameters into entry 0 of every lane's table
        const ST t = sf.read();
        for (int l = 0; l < 4; l++) {
#pragma HLS UNROLL
          tS[l][0] = t.S;
          tR[l][0] = t.R;
          tT[l][0] = t.T;
          tB[l][0] = t.B;
        }
#ifdef QZ_IMPL_FDIV
        cs = t.s;
#endif
        endseg = t.end;
        sq.write(t.st);
      }
      const word_t w = d.read();
      const ap_uint<4> ti = kt ? ap_uint<4>(wi(3, 0)) : ap_uint<4>(0);
      ap_uint<64> h = 0;
      for (int l = 0; l < 4; l++) {
#pragma HLS UNROLL
        RowScale rs;
        rs.S = tS[l][ti];
        rs.R = tR[l][ti];
        rs.T = tT[l][ti];
        rs.B = tB[l][ti];
#ifdef QZ_IMPL_FDIV
        rs.s_bits = kt ? ap_uint<32>(tF[l][ti]) : cs;
#else
        rs.s_bits = 0;
#endif
        h(16 * l + 15, 16 * l) = ap_uint<16>(quant_elem(word_f32(w, l), rs));
      }
      if (wi[0]) ow.write((word_t(h) << 64) | word_t(lo));
      else lo = h;
      const ap_uint<13> wlast = kt ? ap_uint<13>((cw >> 2) - 1) : ap_uint<13>(SW - 1);
      if (wi != wlast) {
        wi++;
      } else {
        wi = 0;
        if (kt) {
          if (n != p.N - 1) {
            n++;
          } else {
            n = 0;
            if (cc != p.nseg - 1) {
              cc++;
            } else {
              cc = 0;
              if (b == p.B - 1) break;
              b++;
            }
            cw = (p.half && cc == p.nseg - 1) ? 32 : 64;
            load = true;
          }
        } else if (endseg) {
          break;
        }
      }
    }
  }
}

// ---------------------------------------------------------------- wb: words + stored scales -> units for wr
// One writer for data and scales. qt writes a segment's stored scale to sq before the segment's words to ow, and wb
// reads them in that same order, so no merge and no lookahead is needed (and no circular wait through qt).
// Every unit is a header (its first word's address) then US words (8; 1 in wsmall mode).
//
// Rows mode (KX, KXS, KC): rows in qt's order. Row n of WO words is units of US words, the last one shifted back to end
// at the row's end (its first words are re-sent from an 8-word ring); rows [N, NP) are zeros with scale 1.0. A row's
// scale goes into a 32-entry window as the row starts; after every 32nd row, and after the vector's last row, the
// window goes out as a scale unit. The vector's last unit is a shift-back unit over its last 8 words, its lanes past
// NP are 1.0. wsmall: a one-word scale unit after every 4th row and after the last.
// Block mode (KW, KT): 8 rows (keys) x RW words are filled into one 8 x 8-word block (rows / keys >= N as zeros),
// then sent as units: KW unit u = words d = 8u..8u+7, word d = lane d % 8 of each row's word u (8 rows of one int16
// column); KT unit u = word u of keys 0..7. One block, not two: qt makes half a word a clock and wb fills and drains at
// a word a clock each, so a single buffer keeps pace (ow holds the block qt makes meanwhile). KW: each row's scale
// goes into the window as the row is filled, and every 4th block (32 rows; NP % 32 == 0) is followed by a scale unit.
// KT: a chunk starts with its 64 (32) channel scales, sent as 2 (1) scale units before the chunk's blocks.
static inline ap_uint<16> lane16(const word_t& x, ap_uint<3> k) {
#pragma HLS INLINE
  ap_uint<16> l[8];
#pragma HLS ARRAY_PARTITION variable = l complete
  for (int i = 0; i < 8; i++) {
#pragma HLS UNROLL
    l[i] = x(16 * i + 15, 16 * i);
  }
  return l[k];
}

static void qz_wb(const RP p, hls::stream<word_t>& ow, hls::stream<ap_uint<32>>& sq, hls::stream<UH>& hd,
                  hls::stream<word_t>& dd) {
  const bool blk = p.kind == KW || p.kind == KT, kt = p.kind == KT, scl = p.kind != KC;
  const ap_uint<4> USm = p.wsmall ? 0 : 7;  // unit words - 1
  ap_uint<32> win[4][16];                   // stored scales: entry e at [e % 4][(e / 4) % 16] (rows modes, KW: % 8)
#pragma HLS ARRAY_PARTITION variable = win dim = 1 complete
#pragma HLS BIND_STORAGE variable = win type = ram_2p impl = lutram
  // ---- rows mode state
  word_t ring[8];
#pragma HLS BIND_STORAGE variable = ring type = ram_2p impl = lutram
  const ap_uint<9> nj = p.kind == KX || p.kind == KC ? ap_uint<9>(1) : ap_uint<9>(p.nseg);  // vectors / item
  const ap_uint<17> NR = p.kind == KC ? p.N : p.NP;                          // rows per (item, segment)
  const ap_uint<13> nu = p.wsmall ? p.WO : ap_uint<13>((p.WO + 7) >> 3);    // units per row
  const ap_uint<3> tl = p.wsmall ? ap_uint<3>(0) : ap_uint<3>(p.WO(2, 0));  // new words of the row's last unit
  const ap_uint<15> Lw = (p.NP + 3) >> 2;                                     // words of a scale vector (not KT)
  // ---- block mode state
  word_t buf[8][8];  // [row / key][word]
#pragma HLS ARRAY_PARTITION variable = buf dim = 1 complete
#pragma HLS BIND_STORAGE variable = buf type = ram_2p impl = lutram
#pragma HLS DEPENDENCE variable = buf inter false  // filled, then drained, then refilled: never both in one clock
  const ap_uint<14> ngr = p.NP >> 3;  // blocks per (item, segment / chunk)
  // ---- common
  ap_uint<17> b = 0, n = 0;  // item; row (rows mode) / block row base 8 g (block mode)
  ap_uint<9> j = 0;          // segment (KXS, KW) / chunk (KT)
  ap_uint<14> g = 0;         // block
  ap_uint<13> u = 0;         // unit of a row (rows) / of a block (blocks)
  ap_uint<4> k = 0;          // word of a unit
  ap_uint<3> r = 0, w = 0;   // block fill: row, word
  ap_uint<7> e = 0;          // KT: chunk scale entries loaded
  ap_uint<15> sw0 = 0;       // first word of the scale unit being sent
  ap_uint<2> su = 0;         // KT: scale units of the chunk sent
  // phase: 0 data (rows) / block fill, 1 block drain, 2 scale unit, 3 KT chunk scale load
  ap_uint<2> ph = kt ? 3 : 0;
  bool after_blk = false;    // a scale unit sent after a KW block / a row: then advance
  wa_t cb = p.c, jb = p.c, rowa = p.c;  // item base, segment base, row base (rows mode)
  wa_t gb = p.c, kb = p.c;              // block base; KT chunk base (jb + 2 cc NP for p1 = 0)
  wa_t fb = p.f, vj = p.f;              // scale vectors: item base, vector base
  const wp_t vstep = (kt && !p.kseg) ? wp_t(16) : p.segf;  // QUANT_WT p1 = 0: chunk cc's 64 scales at + 16 cc words
wb:
  for (;;) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 8 max = 1048576 avg = 20000
    const bool lastb = b == p.B - 1, lastj = j == nj - 1;
    const bool lastchunk = j == p.nseg - 1;
    bool adv = false;  // rows mode: the row and its scale unit are done; block mode: the block (and scales) done
    if (ph == 3) {     // ---- KT: the chunk's channel scales into the window
      const ap_uint<7> cw = (p.half && lastchunk) ? 32 : 64;
      win[e(1, 0)][e(5, 2)] = sq.read();
      if (e == cw - 1) {
        e = 0;
        ph = 2;
        sw0 = 0;
        su = 0;
      } else {
        e++;
      }
    } else if (ph == 2) {  // ---- one word of a scale unit
      const ap_uint<4> wi = kt ? ap_uint<4>(sw0(3, 0) + k) : ap_uint<4>(ap_uint<3>(sw0(2, 0) + k(2, 0)));
      word_t o;
      for (int l = 0; l < 4; l++) {
#pragma HLS UNROLL
        // a lane past the vector's NP entries (rows modes: the pad lanes of the last word) is 1.0
        const bool pad = !kt && !blk && ap_uint<17>(((sw0 + k) << 2) + l) >= p.NP;
        o(32 * l + 31, 32 * l) = pad ? ap_uint<32>(ONE_BITS) : win[l][wi];
      }
      if (k == 0) {
        UH h;
        h.addr = vj + sw0;
        h.last = 0;
        hd.write(h);
      }
      dd.write(o);
      if (k == USm) {
        k = 0;
        if (kt) {
          const ap_uint<2> nsu = (p.half && lastchunk) ? 1 : 2;
          if (su == nsu - 1) {
            ph = 0;  // the chunk's blocks
          } else {
            su++;
            sw0 = 8;
          }
        } else {
          ph = 0;
          adv = true;
        }
      } else {
        k++;
      }
    } else if (!blk) {  // ---- rows mode: one word of the current row's unit
      const bool lastu = u == nu - 1;
      const bool shb = lastu && tl != 0;
      const ap_uint<13> ps = shb ? ap_uint<13>(p.WO - 8) : p.wsmall ? u : ap_uint<13>(u << 3);
      const ap_uint<4> nf = shb ? ap_uint<4>(ap_uint<4>(8) - tl) : ap_uint<4>(0);
      const bool zrow = n >= p.N;
      if (scl && u == 0 && k == 0) {  // the row's stored scale (rows [N, NP): 1.0)
        win[n(1, 0)][n(4, 2)] = zrow ? ap_uint<32>(ONE_BITS) : sq.read();
      }
      const ap_uint<3> ri = ap_uint<3>(ps(2, 0) + k(2, 0));
      word_t o = 0;
      if (zrow) {
        o = 0;
      } else if (k < nf) {
        o = ring[ri];
      } else {
        o = ow.read();
        ring[ri] = o;
      }
      if (k == 0) {
        UH h;
        h.addr = rowa + ps;
        h.last = 0;
        hd.write(h);
      }
      dd.write(o);
      if (k == USm) {
        k = 0;
        if (!lastu) {
          u++;
        } else {
          u = 0;
          // the row is done: a scale unit after every 32nd row (wsmall: every 4th) and after the vector's last row
          const bool vend = n == NR - 1;
          const bool full = p.wsmall ? n(1, 0) == 3 : n(4, 0) == 31;
          if (scl && (full || vend)) {
            ph = 2;
            sw0 = p.wsmall ? ap_uint<15>(n >> 2) : full ? ap_uint<15>((n >> 2) - 7) : ap_uint<15>(Lw - 8);
          } else {
            adv = true;
          }
        }
      } else {
        k++;
      }
    } else if (ph == 0) {  // ---- block fill: word w of row r of block g
      const ap_uint<3> RWm = (kt && p.half && lastchunk) ? 3 : 7;  // words per row in the block, - 1
      const ap_uint<17> row = n | r;
      const bool real = row < p.N;
      if (!kt && w == 0) {  // KW: the row's stored scale (rows [N, NP): 1.0)
        win[row(1, 0)][ap_uint<4>(row(4, 2))] = real ? sq.read() : ap_uint<32>(ONE_BITS);
      }
      buf[r][w] = real ? ow.read() : word_t(0);
      if (w != RWm) {
        w++;
      } else {
        w = 0;
        if (r != 7) {
          r++;
        } else {
          r = 0;
          ph = 1;
        }
      }
    } else {  // ---- block drain: word k of unit u
      const ap_uint<3> RWm = (kt && p.half && lastchunk) ? 3 : 7;
      word_t o;
      if (kt) {
        o = buf[k(2, 0)][u(2, 0)];
      } else {
        for (int q = 0; q < 8; q++) {
#pragma HLS UNROLL
          o(16 * q + 15, 16 * q) = lane16(buf[q][u(2, 0)], k(2, 0));
        }
      }
      if (k == 0) {
        // KW: (g % 4) pitch + (g / 4) 64 + 8 u;  KT: (u % 4) pitch + (u / 4) NP + 8 g (gb holds the 8 g part)
        const ap_uint<2> pi = kt ? ap_uint<2>(u(1, 0)) : ap_uint<2>(g(1, 0));
        const wa_t po = pi == 0 ? wa_t(0) : pi == 1 ? wa_t(p.cp) : pi == 2 ? wa_t(p.cp2) : wa_t(p.cp3);
        const wa_t in = kt ? (u[2] ? wa_t(p.NP) : wa_t(0)) : wa_t(ap_uint<6>(u(2, 0)) << 3);
        UH h;
        h.addr = gb + po + in;
        h.last = 0;
        hd.write(h);
      }
      dd.write(o);
      if (k == 7) {
        k = 0;
        if (u != (kt ? RWm : ap_uint<3>(7))) {
          u++;
        } else {
          u = 0;
          if (!kt && g(1, 0) == 3) {  // KW: 32 rows done: their scale unit
            ph = 2;
            sw0 = ap_uint<15>(g >> 2) << 3;
          } else {
            ph = 0;
            adv = true;
          }
        }
      } else {
        k++;
      }
    }
    if (adv) {  // ---- advance to the next row (rows mode) / block (block mode); the end
      if (!blk) {
        const bool lastn = n == NR - 1;
        if (lastn && lastj && lastb) break;
        if (!lastn) {
          n++;
          rowa += p.cp;
        } else {
          n = 0;
          if (!lastj) {
            j++;
            jb += p.segc;
            vj += p.segf;
          } else {
            j = 0;
            b++;
            cb += p.cs;
            jb = cb;
            fb += p.fs;
            vj = fb;
          }
          rowa = jb;
        }
      } else {
        const bool lastg = g == ngr - 1;
        if (lastg && lastchunk && lastb) break;
        if (!lastg) {
          g++;
          n += 8;
          if (kt) gb += 8;
          else if (g(1, 0) == 0) gb += 64;  // KW: the next 32-row group
        } else {
          g = 0;
          n = 0;
          if (!lastchunk) {
            j++;
            vj += vstep;
            if (kt && !p.kseg) {
              kb += wa_t(p.NP) << 1;
            } else {
              jb += p.segc;
              kb = jb;
            }
          } else {
            j = 0;
            b++;
            cb += p.cs;
            jb = cb;
            kb = cb;
            fb += p.fs;
            vj = fb;
          }
          gb = kt ? kb : jb;
          if (kt) ph = 3;  // the next chunk's scales first
        }
      }
    }
  }
  UH h;
  h.addr = 0;
  h.last = 1;
  hd.write(h);
}

// ---------------------------------------------------------------- wr: units -> ymem
static void qz_wr(word_t* mem, const RP p, hls::stream<UH>& hd, hls::stream<word_t>& dd) {
  if (p.wsmall) {
  wr1:
    for (;;) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 2 max = 4096 avg = 300
      const UH h = hd.read();
      if (h.last) break;
      mem[h.addr] = dd.read();
    }
  } else {
  wr8:
    for (;;) {
#pragma HLS PIPELINE II = 8
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 131072 avg = 3000
      const UH h = hd.read();
      if (h.last) break;
      for (int k = 0; k < 8; k++) {
#pragma HLS UNROLL
        mem[h.addr + k] = dd.read();  // unconditional: one 8-beat burst, pipelined across iterations
      }
    }
  }
}

// ---------------------------------------------------------------- the command -> RP
// ME_OK and the plan, or ME_DIMS for an opcode of another unit. Assumes ctr::validate_hw(c) == ME_OK (so N, B, NP
// <= 65535, K <= 16384, K % 8 == 0, p1 in {0, 64}, |p4| <= 8, every slot below 2^40).
static inline uint32_t row_plan(const ctr::Cmd& c, RP& p) {
#pragma HLS INLINE
  const uint32_t K = c.K, seg = c.p[ctr::P_SEG];
  if (c.op == ctr::OP_QUANT_X) p.kind = seg ? KXS : KX;
  else if (c.op == ctr::OP_QUANT_W) p.kind = KW;
  else if (c.op == ctr::OP_QUANT_WT) p.kind = KT;
  else if (c.op == ctr::OP_COPY) p.kind = KC;
  else return ctr::ME_DIMS;  // not an opcode of this unit
  // QUANT_W with p1 = 0 and K != 64: one scale over a whole row of K > 64 values in the KW tiling. validate()
  // accepts it but the -12392 list never builds it (every QUANT_W is per head, p1 = 64); see README "Contract notes".
  if (c.op == ctr::OP_QUANT_W && seg == 0 && K != ctr::SEG_HEAD) return ctr::ME_DIMS;
  if (K < 8 || K > ctr::ROW_K_MAX || (K & 7) || c.N == 0 || c.B == 0) return ctr::ME_DIMS;
  p.B = c.B;
  p.N = c.N;
  p.NP = (c.op == ctr::OP_COPY || c.p[ctr::P_NP] == 0) ? c.N : c.p[ctr::P_NP];
  p.W = K >> 2;
  p.WO = (c.op == ctr::OP_QUANT_X && !seg) ? ap_uint<13>(K >> 3) : c.op == ctr::OP_COPY ? ap_uint<13>(K >> 2)
                                                                                        : ap_uint<13>(8);
  p.nseg = c.op == ctr::OP_QUANT_WT ? (K + 63) >> 6 : seg ? K >> 6 : 1u;
  p.half = c.op == ctr::OP_QUANT_WT && (K & 63) != 0;
  p.kseg = seg != 0;
  p.twopass = p.kind == KX && (K >> 2) > ROWBUF_W;
  p.shift = (int32_t)c.p[ctr::P_SHIFT];
  const uint32_t npw = (c.op == ctr::OP_QUANT_W || c.op == ctr::OP_QUANT_WT) ? 8u : (uint32_t)(p.NP + 3) >> 2;
  p.rsmall = p.W < 16;  // only KX, KC, KT can have K < 64 (KXS / KW have 64-value segments)
  p.wsmall = ((p.kind == KX || p.kind == KC) && p.WO < 8) || ((p.kind == KX || p.kind == KXS) && npw < 8);
  p.a = c.s[ctr::SLOT_A].addr >> 4;
  p.c = c.s[ctr::SLOT_C].addr >> 4;
  p.f = c.s[ctr::SLOT_F].addr >> 4;
  p.ap = c.s[ctr::SLOT_A].pitch >> 4;
  p.as = c.s[ctr::SLOT_A].stride >> 4;
  p.cp = c.s[ctr::SLOT_C].pitch >> 4;
  p.cs = c.s[ctr::SLOT_C].stride >> 4;
  p.fs = c.s[ctr::SLOT_F].stride >> 4;
  p.segc = c.p[ctr::P_SEGC] >> 4;
  p.segf = c.p[ctr::P_SEGF] >> 4;
  p.cp2 = ap_uint<30>(p.cp) << 1;
  p.cp3 = p.cp2 + p.cp;
  return ctr::ME_OK;
}

// ---------------------------------------------------------------- the dataflow region and the mode
static void row_df(const word_t* xmem, word_t* ymem, const RP p) {
#pragma HLS DATAFLOW
#pragma HLS STABLE variable = p
  hls::stream<XT> xs("qz_xs");
  hls::stream<AM> am("qz_am");
  hls::stream<word_t> d("qz_d");
  hls::stream<SS> ss("qz_ss");
  hls::stream<ST> sf("qz_sf");
  hls::stream<word_t> ow("qz_ow");
  hls::stream<ap_uint<32>> sq("qz_sq");
  hls::stream<UH> hd("qz_hd");
  hls::stream<word_t> dd("qz_dd");
#pragma HLS STREAM variable = xs depth = QZ_D_XS
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
#pragma HLS BIND_STORAGE variable = dd type = fifo impl = lutram
#ifdef QZ_DFSIM  // host deadlock model: every process its own thread, every stream bounded (dfsim/hls_stream.h)
  std::thread t0([&] { qz_rd(xmem, p, xs); });
  std::thread t1([&] { qz_amx(p, xs, am, d); });
  std::thread t2([&] { qz_ss(p, am, ss); });
  std::thread t3([&] { qz_sr(p, ss, sf); });
  std::thread t4([&] { qz_qt(p, d, sf, ow, sq); });
  std::thread t5([&] { qz_wb(p, ow, sq, hd, dd); });
  std::thread t6([&] { qz_wr(ymem, p, hd, dd); });
  t0.join();
  t1.join();
  t2.join();
  t3.join();
  t4.join();
  t5.join();
  t6.join();
#else
  qz_rd(xmem, p, xs);
  qz_amx(p, xs, am, d);
  qz_ss(p, am, ss);
  qz_sr(p, ss, sf);
  qz_qt(p, d, sf, ow, sq);
  qz_wb(p, ow, sq, hd, dd);
  qz_wr(ymem, p, hd, dd);
#endif
}

// The row unit's mode function for QUANT_X / QUANT_W / QUANT_WT / COPY: one call site in the sequencer's dispatch
// switch. Returns ME_OK, or ME_DIMS (nothing read or written). Returns after the last write response (R-ORD): the
// DATAFLOW region ends when wr's last write has been acknowledged.
static uint32_t row_mode(const ctr::Cmd& c, const word_t* xmem, word_t* ymem) {
#pragma HLS INLINE off
  RP p;
  const uint32_t r = row_plan(c, p);
  if (r != ctr::ME_OK) return r;
  row_df(xmem, ymem, p);
  return ctr::ME_OK;
}

// The sequencer's unit policy entry (fpga/kernels/sequencer/seq_core.h "Units": row(c, hp0..hp3, io_rd, io_wr)).
// In synthesis the io word is ap_uint<128> (= word_t); on the host any 16-byte word type with the same bytes.
struct RowUnit {
  template <class WP, class W>
  static uint32_t row(const ctr::Cmd& c, const WP*, const WP*, const WP*, const WP*, const W* io_rd, W* io_wr) {
#pragma HLS INLINE
    static_assert(sizeof(W) == sizeof(word_t), "io words are 128 bits");
    return row_mode(c, reinterpret_cast<const word_t*>(io_rd), reinterpret_cast<word_t*>(io_wr));
  }
};

}  // namespace qz
