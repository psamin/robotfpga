// The float_top row unit's backbone (CONTRACT.md 4.1) with its identity front end: opcodes QUANT_X,
// QUANT_W, QUANT_WT and COPY of fpga/kernels/contract.h, taking a decoded ctr::Cmd and the top's io pointers.
//
//   uint32_t qz::row_mode(const ctr::Cmd& c, const word_t* xmem, word_t* ymem)    (INLINE off: one instance)
//
// called from the sequencer's dispatch switch after ctr::validate_hw(c) == ME_OK (one call site for all of the
// unit's opcodes). It returns ME_OK, or ME_DIMS without touching memory for a command this build does not run
// (an opcode of another unit; QUANT_W with p1 = 0 and K != 64, see README "Contract notes"). Every slot
// address is an absolute byte address (the pointer registers are 0 in list mode); activations are read only
// through xmem and everything is written only through ymem (both on the io bundle). No own m_axi adapter, no
// own decode, no register.
//
// Layouts (CONTRACT.md 1.1, all little-endian, every address / pitch / stride a multiple of 16 bytes):
//   A  FP32 rows: value (b, n, k) at A + b A.stride + n A.pitch + 4k
//   C  QUANT_X: int16 rows; segment j (p1 = 64) of row n at C + b C.stride + n C.pitch + j p2 (p1 = 0: the row)
//      QUANT_W: KW tiling, word (n, d) of segment j at C + b C.stride + j p2 + ((n%32)/8) C.pitch
//               + ((n/32) 64 + d) 16, lane n%8                                       (rows n < NP, NP % 32 == 0)
//      QUANT_WT: VW tiling, word (c, n) of segment j at C + b C.stride + j p2 + ((c%32)/8) C.pitch
//               + ((c/32) NP + n) 16, lane c%8                                       (keys n < NP, NP % 8 == 0)
//      COPY: FP32 rows as A
//   F  stored scales (stored_scale: s 2^p4, zero rows 1.0): QUANT_X / QUANT_W per (row n, segment j) at
//      F + b F.stride + j p3 + 4n, pad4(NP) floats per vector, pad lanes 1.0; QUANT_WT per channel c of segment
//      j at F + b F.stride + j p3 + 4c. Rows / keys [N, NP) are written as zeros with scale 1.0.
//
// Every DDR access is a fixed-size unit inside a pipelined loop, which is what lets HLS keep requests in
// flight across rows (measured on probes: a 16-word read unit per II = 16 iteration, an 8-word write unit per
// II = 8 iteration, pipeline depths 89 / 79 with the m_axi latency 64; a variable-length burst per row instead
// pays the whole DDR latency on every row). Reads: 16 words (64 FP32), at any row pitch and in any order (the
// segment-major orders below cost no latency). Writes: 8 words (64 int16 / 32 FP32 / 32 scales). A row or
// vector whose length is not a multiple of the unit ends with a "shift-back" unit that covers its last 16 (8)
// words, re-reading (re-writing with the same data) the overlap, so nothing outside the declared extents is
// ever read or written. Rows shorter than one unit (K < 64 read, int16 rows of K < 64, FP32 rows of K < 32,
// scale vectors of NP <= 28) use single-word units (slower, never in the hot path).
//
// Processes (one DATAFLOW region per command, every loop II = 1 except the m_axi unit loops):
//   rd   xmem -> tokens: 16-word units, in the op's order: QUANT_X whole rows (b, n); QUANT_X segmented and
//        QUANT_W (b, j, n) segment-major; QUANT_WT per item a column-absmax pass and a data pass per 64-column
//        chunk, interleaved P1(0) P1(1) P2(0) P1(2) P2(1) ... so the chunk's scales are ready before its data;
//        COPY (b, n)
//   [front end]  (identity here: the slot where NORM / GELU / SILU / ROPE put their FP32 producers)
//   amx  per-segment absmax (QUANT_WT: per-column, 64 columns at a time) -> am; data words -> D, the row
//        buffer (FIFO 4096 x 128 bit in 2 URAM: every row up to ROW_K_MAX = 16384 values is read once)
//   sc   absmax -> s = scale_bits, R = recip_unrolled (25 stages, one per clock), stored_scale
//   qt   4 lanes of quant_elem a clock (QUANT_WT: each lane its own column scale from a 64-entry table)
//   sw   stored scales -> 8-word scale units (32 scales; pad lanes and rows [N, NP) 1.0)
//   pk   int16 / FP32 words -> 8-word data units: rows (with the shift-back tail), zero rows, KW (8 rows x 64
//        int16 transposed into 64 words of 8 rows) and VW (8 keys x 64 channels reordered into 8-word
//        key runs) through a double-buffered 8 x 8-word block; merges sw's units
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

typedef ap_uint<46> wa_t;  // 128-bit word index: byte address / 16. 46 bits = byte addresses < 2^50, so no
                           // address of a validate_hw-legal command (addr < 2^40, + B stride + N pitch) wraps
typedef ap_uint<28> wp_t;  // a pitch / stride / segment step in words (u32 bytes / 16)

// stream depths (named so the threaded deadlock model in tb_quant.cpp uses the same numbers)
#ifndef QZ_D_XS
#define QZ_D_XS 32     // rd -> amx tokens
#endif
#ifndef QZ_D_AM
#define QZ_D_AM 64     // amx -> sc: must hold one chunk's 64 column absmax (QUANT_WT order, see amx)
#endif
#ifndef QZ_D_D
#define QZ_D_D 4096    // amx -> qt row buffer: must hold the longest row read once (16384 / 4 words)
#endif
#ifndef QZ_D_SF
#define QZ_D_SF 16     // sc -> qt scale tokens
#endif
#ifndef QZ_D_OW
#define QZ_D_OW 32     // qt -> pk int16 / FP32 words
#endif
#ifndef QZ_D_SQ
#define QZ_D_SQ 64     // qt -> sw stored scales (QUANT_WT writes 64 in a row)
#endif
#ifndef QZ_D_SH
#define QZ_D_SH 4      // sw -> pk unit headers (written after the unit's words)
#endif
#ifndef QZ_D_SD
#define QZ_D_SD 32     // sw -> pk unit words
#endif
#ifndef QZ_D_HD
#define QZ_D_HD 8      // pk -> wr unit headers
#endif
#ifndef QZ_D_DD
#define QZ_D_DD 32     // pk -> wr unit words
#endif

enum : uint8_t { KX = 0, KXS = 1, KW = 2, KT = 3, KC = 4 };  // QUANT_X rows / QUANT_X segmented / QUANT_W /
                                                              // QUANT_WT / COPY

// The command as the processes need it (row_plan derives it from ctr::Cmd).
struct RP {
  ap_uint<3> kind;
  ap_uint<17> B, N, NP;     // items, rows (keys) per item, rows (keys) written (>= N)
  ap_uint<13> W;            // FP32 words per input row (K / 4, <= 4096)
  ap_uint<13> WO;           // output words per row: KX K / 8, KC K / 4, KXS 8 (one segment)
  ap_uint<8> nseg;          // KXS / KW: segments of 64 values; KT: column chunks of 64 (the last may be 32)
  bool half;                // KT: the last chunk has 32 columns
  bool kseg;                // KT: p1 = 64 (chunk cc is segment cc) - else p1 = 0 (one segment of K columns)
  bool rsmall, wsmall;      // single-word read units (W < 16) / write units
  ap_int<5> shift;          // p4
  ap_uint<14> npw;          // KX / KXS / KW: words of one scale vector, ceil(NP / 4)
  ap_uint<40> nsc;          // scale tokens: KX B N, KXS / KW B nseg N, KT B K
  wa_t a, c, f;             // slot A, C, F addresses (words)
  wp_t ap, as, cp, cs, fs;  // pitches / strides (words)
  wp_t segc, segf;          // p2 / p3 (words)
};

// ---------------------------------------------------------------- stream tokens
struct XT {             // rd -> amx
  word_t w;
  ap_uint<1> fwd;       // data word: to the row buffer
  ap_uint<1> amx;       // counts for the absmax (per segment, or per column in a QUANT_WT pass 1)
  ap_uint<1> last;      // last word of a segment (absmax out) / of a QUANT_WT chunk's pass 1
  ap_uint<1> end;       // last token of the command
  ap_uint<4> col;       // QUANT_WT pass 1: word position inside the 64-column chunk
};
struct ST {             // sc -> qt
  ap_uint<32> s;        // s = fl32(max(a, 1e-30) / 32767), 1.0 for a == 0
  ap_uint<26> R;        // min(floor(2^49 / S), 2^26 - 1)
  ap_uint<32> st;       // the stored scale
};
struct UH {             // unit header (sw -> pk, pk -> wr)
  wa_t addr;            // first word
  ap_uint<1> last;      // sw: the command's last scale unit;  pk -> wr: end of the command (no words follow)
};

// ---------------------------------------------------------------- rd: xmem -> tokens
static void qz_rd(const word_t* mem, const RP p, hls::stream<XT>& xs) {
  const bool kt = p.kind == KT, kc = p.kind == KC, kx = p.kind == KX;
  const bool segmaj = p.kind == KXS || p.kind == KW;
  if (p.rsmall) {  // rows of < 64 values (KX, KC, KT with K = 32): single words, (b, [pass,] n, w)
    ap_uint<17> b = 0, n = 0;
    ap_uint<13> w = 0;
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
      t.col = w(3, 0);
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
  // 16-word units. Unit state: item b, outer index j (KXS / KW segment; KT step), row n, unit u in the row.
  const ap_uint<9> nu = (p.W + 15) >> 4;  // KX / KC units per row
  const ap_uint<5> tail = p.W(3, 0);     // KX / KC: words of the last unit that are new (0 = all 16)
  ap_uint<17> b = 0, n = 0;
  ap_uint<9> j = 0, u = 0;               // j: KXS / KW segment, KT step t (0 .. 2 nch - 1)
  wa_t ib = p.a, rowo = 0;               // item base; offset of row n inside the item
rd16:
  for (;;) {
#pragma HLS PIPELINE II = 16
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 65536 avg = 1500
    // KT step t -> (chunk cc, pass): P1(0) | P1(1) P2(0) | P1(2) P2(1) | ... | P2(nch - 1)
    const ap_uint<9> nst = ap_uint<9>(p.nseg) << 1;
    ap_uint<9> cc;
    bool p1;
    if (j == 0) {
      cc = 0;
      p1 = true;
    } else if (j == nst - 1) {
      cc = p.nseg - 1;
      p1 = false;
    } else if (j[0]) {
      cc = (j + 1) >> 1;
      p1 = true;
    } else {
      cc = (j >> 1) - 1;
      p1 = false;
    }
    const bool lastchunk = cc == p.nseg - 1;
    // the unit's first word and the number of leading words that are not new (shift-back)
    wa_t uo;
    ap_uint<5> skip = 0;
    if (segmaj) {
      uo = ap_uint<13>(j(7, 0)) << 4;
    } else if (kt) {
      if (p.half && lastchunk) {
        uo = p.W - 16;
        skip = 8;
      } else {
        uo = ap_uint<13>(cc(7, 0)) << 4;
      }
    } else if (u == nu - 1 && tail != 0) {
      uo = p.W - 16;
      skip = 16 - tail;
    } else {
      uo = ap_uint<13>(u) << 4;
    }
    const wa_t addr = ib + rowo + uo;
    word_t v[16];
#pragma HLS ARRAY_PARTITION variable = v complete
    for (int k = 0; k < 16; k++) {
#pragma HLS UNROLL
      v[k] = mem[addr + k];  // unconditional: one 16-beat burst, pipelined across iterations
    }
    const bool lastrow = n == p.N - 1, lastitem = b == p.B - 1;
    const bool rowdone = segmaj || kt || u == nu - 1;
    const bool lastj = segmaj ? j == p.nseg - 1 : kt ? j == nst - 1 : true;
    const bool fin = rowdone && lastrow && lastj && lastitem;
    for (int k = 0; k < 16; k++) {
#pragma HLS UNROLL
      if (k >= skip) {
        XT t;
        t.w = v[k];
        t.fwd = !kt || !p1;
        t.amx = (kt && p1) || kx || segmaj;
        t.last = k == 15 && (kt ? (p1 && lastrow) : (!kc && rowdone));
        t.col = ap_uint<4>(k - skip);
        t.end = k == 15 && fin;
        xs.write(t);
      }
    }
    if (fin) break;
    // advance: u (KX / KC), then n, then j (KXS / KW / KT), then b
    if (!rowdone) {
      u++;
    } else {
      u = 0;
      if (!lastrow) {
        n++;
        rowo += p.ap;
      } else {
        n = 0;
        rowo = 0;
        if (!lastj) {
          j++;
        } else {
          j = 0;
          b++;
          ib += p.as;
        }
      }
    }
  }
}

// ---------------------------------------------------------------- amx: absmax + row buffer
static void qz_amx(const RP p, hls::stream<XT>& xs, hls::stream<abits_t>& am, hls::stream<word_t>& d) {
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
    bool pass_end = false, end = false;
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
            am.write(mm);
            m = 0;
          } else {
            m = mm;
          }
        }
      }
      pass_end = kt && t.amx && t.last;
      end = t.end;
      if (pass_end || end) break;
    }
    if (pass_end) {  // the chunk's column absmax, in column order, and clear for the next chunk
      const ap_uint<7> cw = (p.half && cc == p.nseg - 1) ? 32 : 64;
    emit:
      for (ap_uint<7> c = 0; c < cw; c++) {
#pragma HLS PIPELINE II = 1
        am.write(cm[c(1, 0)][c(5, 2)]);
        cm[c(1, 0)][c(5, 2)] = 0;
      }
      cc = cc == p.nseg - 1 ? ap_uint<9>(0) : ap_uint<9>(cc + 1);
    }
    if (end) break;
  }
}

// ---------------------------------------------------------------- sc: absmax -> (s, R, stored), one a clock
static void qz_sc(const RP p, hls::stream<abits_t>& am, hls::stream<ST>& sf) {
scales:
  for (ap_uint<40> i = 0; i < p.nsc; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 65536 avg = 2000
    const abits_t a = am.read();
    ST t;
    t.s = scale_bits(a);
    t.R = recip_unrolled((ap_uint<1>(1), ap_uint<23>(t.s(22, 0))));
    t.st = stored_scale(a, t.s, p.shift);
    sf.write(t);
  }
}

// ---------------------------------------------------------------- qt: 4 lanes
static void qz_qt(const RP p, hls::stream<word_t>& d, hls::stream<ST>& sf, hls::stream<word_t>& ow,
                  hls::stream<ap_uint<32>>& sq) {
  if (p.kind == KC) {  // FP32 rows straight through
    const ap_uint<44> words = ap_uint<44>(p.B * p.N) * p.W;
  copy:
    for (ap_uint<44> i = 0; i < words; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 2 max = 100000 avg = 20000
      ow.write(d.read());
    }
    return;
  }
  const bool kt = p.kind == KT;
  const ap_uint<13> SW = p.kind == KX ? p.W : ap_uint<13>(16);  // words per segment (KX: the row)
  ap_uint<32> ts[4][16];  // QUANT_WT: s and R of column 4 w + l of the chunk
  ap_uint<26> tr[4][16];
#pragma HLS ARRAY_PARTITION variable = ts dim = 1 complete
#pragma HLS ARRAY_PARTITION variable = tr dim = 1 complete
#pragma HLS BIND_STORAGE variable = ts type = ram_2p impl = lutram
#pragma HLS BIND_STORAGE variable = tr type = ram_2p impl = lutram
  ap_uint<32> s_cur = ONE_BITS;
  ap_uint<26> r_cur = 0;
  ap_uint<64> lo = 0;      // the 4 int16 of the even word of a pair
  ap_uint<13> wi = 0;      // word in the segment (KT: in the chunk row)
  ap_uint<17> n = 0, b = 0;
  ap_uint<9> cc = 0;       // KT chunk
  ap_uint<7> lc = 0;       // KT: table entries loaded
  ap_uint<40> seg = 0;     // KX / KXS / KW: segments done
  bool load = kt;          // KT: loading the chunk's column scales
  ap_uint<7> cw = (p.half && p.nseg == 1) ? 32 : 64;
quant:
  for (;;) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 2 max = 1048576 avg = 20000
#pragma HLS DEPENDENCE variable = ts inter false
#pragma HLS DEPENDENCE variable = tr inter false
    if (load) {
      const ST t = sf.read();
      ts[lc(1, 0)][lc(5, 2)] = t.s;
      tr[lc(1, 0)][lc(5, 2)] = t.R;
      sq.write(t.st);
      if (lc == cw - 1) {
        lc = 0;
        load = false;
      } else {
        lc++;
      }
    } else {
      if (!kt && wi == 0) {
        const ST t = sf.read();
        s_cur = t.s;
        r_cur = t.R;
        sq.write(t.st);
      }
      const word_t w = d.read();
      ap_uint<64> h = 0;
      for (int l = 0; l < 4; l++) {
#pragma HLS UNROLL
        RowScale rs = row_scale_from_s(kt ? ts[l][wi(3, 0)] : s_cur);
        rs.R = kt ? tr[l][wi(3, 0)] : r_cur;
        h(16 * l + 15, 16 * l) = ap_uint<16>(quant_elem(word_f32(w, l), rs));
      }
      if (wi[0]) ow.write((word_t(h) << 64) | word_t(lo));
      else lo = h;
      const ap_uint<13> wlast = kt ? ap_uint<13>(cw >> 2) - 1 : SW - 1;
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
        } else {
          if (seg == p.nsc - 1) break;
          seg++;
        }
      }
    }
  }
}

// ---------------------------------------------------------------- sw: stored scales -> scale units
// Vectors: KX per item (NP rows), KXS / KW per (item, segment) (NP rows), KT per (item, chunk) (cw channels).
// Entry e < V comes from sq, entries [V, pad4(L)) are 1.0. A unit is 32 entries (8 words); a vector whose
// word count is not a multiple of 8 ends with a shift-back unit over its last 8 words (the window holds the
// last 32 entries). wsmall: one-word units. Each unit's words go to sd before its header goes to sh, so pk
// only picks a unit whose words are all there.
static void qz_sw(const RP p, hls::stream<ap_uint<32>>& sq, hls::stream<UH>& sh, hls::stream<word_t>& sd) {
  if (p.kind == KC) return;
  const bool kt = p.kind == KT;
  ap_uint<32> win[4][8];
#pragma HLS ARRAY_PARTITION variable = win dim = 1 complete
#pragma HLS BIND_STORAGE variable = win type = ram_2p impl = lutram
  const ap_uint<9> nvj = (p.kind == KX) ? ap_uint<9>(1) : ap_uint<9>(p.nseg);  // vectors per item
  wa_t fb = p.f;
items:
  for (ap_uint<17> b = 0; b < p.B; b++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 16 avg = 2
    wa_t vj = fb;  // vector j's base: + j p3, or (QUANT_WT with p1 = 0) + 16 j
  vecs:
    for (ap_uint<9> j = 0; j < nvj; j++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 64 avg = 8
      ap_uint<17> L, V;
      if (kt) {
        L = (p.half && j == p.nseg - 1) ? 32 : 64;
        V = L;
      } else {
        L = (p.NP + 3) & ~ap_uint<17>(3);
        V = p.N;
      }
      const ap_uint<15> Lw = L >> 2;  // words of the vector
      const bool lastvec = b == p.B - 1 && j == nvj - 1;
      if (p.wsmall) {
      ents1:
        for (ap_uint<17> e = 0; e < L; e++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 4 max = 28 avg = 16
          const ap_uint<32> v = e < V ? sq.read() : ap_uint<32>(ONE_BITS);
          win[e(1, 0)][0] = v;
          if (e(1, 0) == 3) {
            sd.write((word_t(v) << 96) | (word_t(win[2][0]) << 64) | (word_t(win[1][0]) << 32) | word_t(win[0][0]));
            UH h;
            h.addr = vj + (e >> 2);
            h.last = lastvec && e == L - 1;
            sh.write(h);
          }
        }
      } else {
      groups:
        for (ap_uint<17> e0 = 0; e0 < L; e0 += 32) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 2048 avg = 4
          const ap_uint<17> left = L - e0;
          const ap_uint<6> ne = left > 32 ? ap_uint<17>(32) : left;
        ents:
          for (ap_uint<6> k = 0; k < ne; k++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 4 max = 32 avg = 32
            const ap_uint<17> e = e0 + k;
            win[e(1, 0)][e(4, 2)] = e < V ? sq.read() : ap_uint<32>(ONE_BITS);
          }
          const bool fin = e0 + ne == L;
          // a full group is a unit; a partial last group (Lw % 8 != 0) is the shift-back unit over the last
          // 8 words. (A full group that ends the vector has Lw % 8 == 0: no tail unit after it.)
          const ap_uint<15> w0 = ne == 32 ? ap_uint<15>(e0 >> 2) : ap_uint<15>(Lw - 8);
        unit:
          for (int k = 0; k < 8; k++) {
#pragma HLS PIPELINE II = 1
            const ap_uint<3> ws = ap_uint<3>(w0 + k);
            sd.write((word_t(win[3][ws]) << 96) | (word_t(win[2][ws]) << 64) | (word_t(win[1][ws]) << 32) |
                     word_t(win[0][ws]));
          }
          UH h;
          h.addr = vj + w0;
          h.last = lastvec && fin;
          sh.write(h);
        }
      }
      vj += (kt && !p.kseg) ? wa_t(16) : wa_t(p.segf);
    }
    fb += p.fs;
  }
}

// ---------------------------------------------------------------- pk: data units (+ sw's units) -> wr
// Rows mode (KX, KXS, KC): units in input order; row n of WO words is units of US words, the last one shifted
// back to end at the row's end (its first words are re-sent from an 8-word ring); rows [N, NP) are zeros.
// Block mode (KW, KT): a filler puts 8 rows (keys) x RW words into one half of a double buffer (rows / keys
// >= N as zeros) while the drainer sends the other half as units: KW unit u = 8 words d = 8u..8u+7, word d =
// lane d % 8 of each row's word u (8 rows of one int16 column); KT unit u = word u of keys 0..7.
// At every unit boundary a complete scale unit from sw goes first. A data unit whose first word must come
// from qt is only started once that word is there: then the rest of the unit belongs to the same segment,
// which qt can finish without waiting on sw (no circular wait through sq / sh).
static void qz_pk(const RP p, hls::stream<word_t>& ow, hls::stream<UH>& sh, hls::stream<word_t>& sd,
                  hls::stream<UH>& hd, hls::stream<word_t>& dd) {
  const bool blk = p.kind == KW || p.kind == KT, kt = p.kind == KT;
  const ap_uint<4> US = p.wsmall ? 1 : 8;
  word_t buf[8][16];  // block mode: [row / key][half x 8 + word]
#pragma HLS ARRAY_PARTITION variable = buf dim = 1 complete
#pragma HLS BIND_STORAGE variable = buf type = ram_2p impl = lutram
#pragma HLS DEPENDENCE variable = buf inter false
  word_t ring[8];  // rows mode: the row's last 8 words
#pragma HLS BIND_STORAGE variable = ring type = ram_2p impl = lutram
#pragma HLS DEPENDENCE variable = ring inter false
  const ap_uint<14> ngr = p.NP >> 3;  // block mode: blocks per (item, segment / chunk)
  const ap_uint<9> nj = (p.kind == KX || p.kind == KC) ? ap_uint<9>(1) : ap_uint<9>(p.nseg);
  const ap_uint<17> NR = p.kind == KC ? p.N : p.NP;                          // rows mode: rows per (item, segment)
  const ap_uint<13> nu = p.wsmall ? p.WO : ap_uint<13>((p.WO + 7) >> 3);   // rows mode: units per row
  const ap_uint<4> tl = p.wsmall ? ap_uint<4>(0) : ap_uint<4>(p.WO(2, 0));  // rows mode: new words, last unit
  bool sdone = p.kind == KC, ddone = false;
  // filler (block mode): item, segment / chunk, block, row, word; buffer half
  ap_uint<17> fi = 0;
  ap_uint<9> fj = 0;
  ap_uint<14> fg = 0;
  ap_uint<3> fr = 0, fw = 0;
  ap_uint<1> fh = 0;
  bool fdone = !blk, full0 = false, full1 = false;
  // output: item, segment / chunk, row (rows mode) or block (block mode), unit, word of the unit
  ap_uint<17> ob = 0, on = 0;
  ap_uint<9> oj = 0;
  ap_uint<13> ou = 0;
  ap_uint<4> ok = 0, oleft = 0;
  ap_uint<1> oh = 0;
  bool oscl = false;
  wa_t cb = p.c, jb = p.c, rowa = p.c;  // item base, segment base, row base (rows mode)
  wa_t kb = 0;                          // KT with p1 = 0: 2 ccs NP (words) for chunk ccs
pk:
  for (;;) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 8 max = 1048576 avg = 20000
    // ---------------- filler (block mode)
    const ap_uint<4> FRW = (kt && p.half && fj == p.nseg - 1) ? ap_uint<4>(4) : ap_uint<4>(8);
    if (!fdone && (fh ? !full1 : !full0)) {
      const ap_uint<17> row = (ap_uint<17>(fg) << 3) | fr;
      word_t x = 0;
      const bool got = row < p.N ? ow.read_nb(x) : true;
      if (got) {
        buf[fr][(ap_uint<4>(fh) << 3) | fw] = x;
        if (fw != FRW - 1) {
          fw++;
        } else {
          fw = 0;
          if (fr != 7) {
            fr++;
          } else {
            fr = 0;
            if (fh) full1 = true;
            else full0 = true;
            fh = ~fh;
            if (fg != ngr - 1) {
              fg++;
            } else {
              fg = 0;
              if (fj != p.nseg - 1) {
                fj++;
              } else {
                fj = 0;
                if (fi == p.B - 1) fdone = true;
                fi++;
              }
            }
          }
        }
      }
    }
    // ---------------- output
    const ap_uint<4> nun = (kt && p.half && oj == p.nseg - 1) ? ap_uint<4>(4) : ap_uint<4>(8);  // block units
    const bool lastu = ou == nu - 1;  // rows mode
    const ap_uint<13> ps = (lastu && tl != 0) ? ap_uint<13>(p.WO - 8) : ap_uint<13>(p.wsmall ? ou : ap_uint<13>(ou << 3));
    const ap_uint<4> nf = (lastu && tl != 0) ? ap_uint<4>(8 - tl) : ap_uint<4>(0);
    const bool zrow = on >= p.N;
    if (oleft == 0) {  // a unit boundary: a scale unit first, then the next data unit, then the end
      UH h;
      if (!sdone && sh.read_nb(h)) {
        UH o;
        o.addr = h.addr;
        o.last = 0;
        hd.write(o);
        sdone = h.last;
        oscl = true;
        oleft = US;
        ok = 0;
      } else if (!ddone) {
        bool ready;
        wa_t a;
        if (blk) {
          ready = oh ? full1 : full0;
          if (kt) {  // (u % 4) pitch + ((2 ccs + u / 4) NP + 8 g)
            const wa_t pp = (ou[1] ? wa_t(p.cp << 1) : wa_t(0)) + (ou[0] ? wa_t(p.cp) : wa_t(0));
            a = jb + pp + kb + (ou[2] ? wa_t(p.NP) : wa_t(0)) + (wa_t(on) << 3);
          } else {   // (g % 4) pitch + (g / 4) 64 + 8 u
            const wa_t pp = (on[1] ? wa_t(p.cp << 1) : wa_t(0)) + (on[0] ? wa_t(p.cp) : wa_t(0));
            a = jb + pp + (wa_t(on >> 2) << 6) + (wa_t(ou(2, 0)) << 3);
          }
        } else {
          ready = zrow || nf != 0 || !ow.empty();
          a = rowa + ps;
        }
        if (ready) {
          UH o;
          o.addr = a;
          o.last = 0;
          hd.write(o);
          oscl = false;
          oleft = US;
          ok = 0;
        }
      } else if (sdone) {
        UH o;
        o.addr = 0;
        o.last = 1;
        hd.write(o);
        break;
      }
    }
    if (oleft != 0) {  // one word of the current unit
      word_t o = 0;
      if (oscl) {
        o = sd.read();
      } else if (blk) {
        const ap_uint<4> e = (ap_uint<4>(oh) << 3) | ap_uint<4>(ou(2, 0));
        word_t rw[8];
#pragma HLS ARRAY_PARTITION variable = rw complete
        for (int r = 0; r < 8; r++) {
#pragma HLS UNROLL
          rw[r] = buf[r][e];
        }
        const ap_uint<3> k3 = ok(2, 0);
        if (kt) {
          o = rw[k3];
        } else {
          for (int r = 0; r < 8; r++) {
#pragma HLS UNROLL
            o(16 * r + 15, 16 * r) = rw[r](16 * k3 + 15, 16 * k3);
          }
        }
      } else {
        const ap_uint<3> ri = ap_uint<3>(ps + ok);
        if (zrow) {
          o = 0;
        } else if (ok < nf) {
          o = ring[ri];
        } else {
          o = ow.read();
          ring[ri] = o;
        }
      }
      dd.write(o);
      ok++;
      oleft--;
      if (oleft == 0 && !oscl) {  // a data unit is done: advance
        if (blk) {
          if (ou != nun - 1) {
            ou++;
          } else {
            ou = 0;
            if (oh) full1 = false;
            else full0 = false;
            oh = ~oh;
            if (on != ngr - 1) {
              on++;
            } else {
              on = 0;
              if (oj != p.nseg - 1) {
                oj++;
                if (kt && !p.kseg) kb += wa_t(p.NP) << 1;
                else jb += p.segc;
              } else {
                oj = 0;
                kb = 0;
                if (ob == p.B - 1) ddone = true;
                ob++;
                cb += p.cs;
                jb = cb;
              }
            }
          }
        } else {
          if (!lastu) {
            ou++;
          } else {
            ou = 0;
            if (on != NR - 1) {
              on++;
              rowa += p.cp;
            } else {
              on = 0;
              if (oj != nj - 1) {
                oj++;
                jb += p.segc;
              } else {
                oj = 0;
                if (ob == p.B - 1) ddone = true;
                ob++;
                cb += p.cs;
                jb = cb;
              }
              rowa = jb;
            }
          }
        }
      }
    }
  }
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
// ME_OK and the plan, or ME_DIMS for what this build does not run. Assumes ctr::validate_hw(c) == ME_OK.
static inline uint32_t row_plan(const ctr::Cmd& c, RP& p) {
#pragma HLS INLINE
  const uint32_t K = c.K, seg = c.p[ctr::P_SEG];
  if (c.op == ctr::OP_QUANT_X) p.kind = seg ? KXS : KX;
  else if (c.op == ctr::OP_QUANT_W) p.kind = KW;
  else if (c.op == ctr::OP_QUANT_WT) p.kind = KT;
  else if (c.op == ctr::OP_COPY) p.kind = KC;
  else return ctr::ME_DIMS;  // not an opcode of this unit
  if (c.op == ctr::OP_QUANT_W && seg == 0 && K != ctr::SEG_HEAD) return ctr::ME_DIMS;  // KW: 64-value segments
  if (K < 8 || K > ctr::ROW_K_MAX || (K & 7) || c.N == 0 || c.B == 0) return ctr::ME_DIMS;
  p.B = c.B;
  p.N = c.N;
  p.NP = c.p[ctr::P_NP] ? c.p[ctr::P_NP] : c.N;
  if (c.op == ctr::OP_COPY) p.NP = c.N;
  p.W = K >> 2;
  p.WO = c.op == ctr::OP_QUANT_X && !seg ? ap_uint<13>(K >> 3) : c.op == ctr::OP_COPY ? ap_uint<13>(K >> 2)
                                                                                      : ap_uint<13>(8);
  p.nseg = c.op == ctr::OP_QUANT_WT ? (K + 63) >> 6 : seg ? K >> 6 : (c.op == ctr::OP_QUANT_W ? 1u : 1u);
  p.half = c.op == ctr::OP_QUANT_WT && (K & 63) != 0;
  p.kseg = seg != 0;
  p.shift = (int32_t)c.p[ctr::P_SHIFT];
  p.npw = (uint32_t)(p.NP + 3) >> 2;
  p.rsmall = p.W < 16;  // only KX, KC, KT can have K < 64 (KXS / KW have 64-value segments)
  p.wsmall = (p.kind == KX && (p.WO < 8 || p.npw < 8)) || (p.kind == KC && p.WO < 8) || (p.kind == KXS && p.npw < 8);
  const ap_uint<40> rows = ap_uint<40>(c.B) * c.N;
  p.nsc = p.kind == KX ? rows : (p.kind == KXS || p.kind == KW) ? ap_uint<40>(rows * p.nseg)
        : p.kind == KT ? ap_uint<40>(ap_uint<40>(c.B) * K) : ap_uint<40>(0);
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
  return ctr::ME_OK;
}

// ---------------------------------------------------------------- the dataflow region and the mode
static void row_df(const word_t* xmem, word_t* ymem, const RP p) {
#pragma HLS DATAFLOW
  hls::stream<XT> xs("qz_xs");
  hls::stream<abits_t> am("qz_am");
  hls::stream<word_t> d("qz_d");
  hls::stream<ST> sf("qz_sf");
  hls::stream<word_t> ow("qz_ow");
  hls::stream<ap_uint<32>> sq("qz_sq");
  hls::stream<UH> sh("qz_sh"), hd("qz_hd");
  hls::stream<word_t> sd("qz_sd"), dd("qz_dd");
#pragma HLS STREAM variable = xs depth = QZ_D_XS
#pragma HLS STREAM variable = am depth = QZ_D_AM
#pragma HLS STREAM variable = d depth = QZ_D_D
#pragma HLS STREAM variable = sf depth = QZ_D_SF
#pragma HLS STREAM variable = ow depth = QZ_D_OW
#pragma HLS STREAM variable = sq depth = QZ_D_SQ
#pragma HLS STREAM variable = sh depth = QZ_D_SH
#pragma HLS STREAM variable = sd depth = QZ_D_SD
#pragma HLS STREAM variable = hd depth = QZ_D_HD
#pragma HLS STREAM variable = dd depth = QZ_D_DD
#ifdef QZ_D_BRAM
#pragma HLS BIND_STORAGE variable = d type = fifo impl = bram
#else
#pragma HLS BIND_STORAGE variable = d type = fifo impl = uram
#endif
#pragma HLS BIND_STORAGE variable = xs type = fifo impl = lutram
#pragma HLS BIND_STORAGE variable = am type = fifo impl = lutram
#pragma HLS BIND_STORAGE variable = sq type = fifo impl = lutram
#pragma HLS BIND_STORAGE variable = ow type = fifo impl = lutram
#pragma HLS BIND_STORAGE variable = sd type = fifo impl = lutram
#pragma HLS BIND_STORAGE variable = dd type = fifo impl = lutram
#ifdef QZ_DFSIM  // host deadlock model: every process its own thread, every stream bounded (dfsim/hls_stream.h)
  std::thread t0([&] { qz_rd(xmem, p, xs); });
  std::thread t1([&] { qz_amx(p, xs, am, d); });
  std::thread t2([&] { qz_sc(p, am, sf); });
  std::thread t3([&] { qz_qt(p, d, sf, ow, sq); });
  std::thread t4([&] { qz_sw(p, sq, sh, sd); });
  std::thread t5([&] { qz_pk(p, ow, sh, sd, hd, dd); });
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
  qz_sc(p, am, sf);
  qz_qt(p, d, sf, ow, sq);
  qz_sw(p, sq, sh, sd);
  qz_pk(p, ow, sh, sd, hd, dd);
  qz_wr(ymem, p, hd, dd);
#endif
}

// The row unit's mode function for QUANT_X / QUANT_W / QUANT_WT / COPY: one call site in the sequencer's
// dispatch switch. Returns ME_OK, or ME_DIMS (nothing read or written).
static uint32_t row_mode(const ctr::Cmd& c, const word_t* xmem, word_t* ymem) {
#pragma HLS INLINE off
  RP p;
  const uint32_t r = row_plan(c, p);
  if (r != ctr::ME_OK) return r;
  row_df(xmem, ymem, p);
  return ctr::ME_OK;
}

}  // namespace qz
