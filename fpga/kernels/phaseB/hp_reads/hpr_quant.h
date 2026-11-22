// hp_reads, row unit part: QUANT_X / QUANT_W / QUANT_WT / COPY reading slot A over io AND hp1, 8 lanes.
// Built against fpga/kernels/quantize/quant_mode.h (the integrator's current row-unit backbone, qz::row_mode):
// the plan (qz::RP, qz::row_plan), the token / scale types and the ss / sr processes are used as they are; the
// reader, absmax, quantizer and writer are qz's with the changes below. Everything else (layouts, the quantizer
// arithmetic, the unit-grid shift-back rule, the write units) is qz's, so every output byte is qz::row_mode's.
//
//   uint32_t hpq::row_mode_hpr(const ctr::Cmd& c, uint32_t hpr, const word_t* hp1, const word_t* xmem,
//                              word_t* ymem)        (INLINE off: one instance; hpr = record field 56)
//
// What changes against qz::row_mode (INTEGRATION.md has the per-process diff):
//   * The unit grid is 8 words (128 B, 32 FP32) instead of 16: unit v of a row pass covers words [8v, 8v + 8),
//     the last one shifted back to end at the row's end (its leading words are not new). With HPR on (hpr ==
//     hpr::HPR_A_HP1 and K >= 64) unit 2v + 0 is read over io and 2v + 1 over hp1, in the same "slot"; with HPR
//     off every unit goes over io (one per slot) and hp1 is not touched: the 4-lane behaviour of qz.
//   * rd0 (io) and rd1 (hp1) walk the same sequence of slots (rd1 only when HPR is on). A slot whose odd unit
//     does not exist (a row pass with an odd unit count, e.g. K = 720) re-reads the even unit's words over hp1
//     (inside the declared extent) as tokens marked not new.
//   * amx reads both token streams in lockstep: the absmax of a segment is the integer max over both halves
//     (exact, order-free), QUANT_WT's column absmax lives in two 8-word tables (columns 0..31 / 32..63). Every
//     token of a data pass goes to its row buffer with its "new" bit (d0 from io, d1 from hp1), so both row
//     buffers hold the same number of entries per row.
//   * qt quantizes 8 lanes a clock: lanes 0..3 the d0 entry (DSP datapath, as qz), lanes 4..7 the d1 entry
//     (fabric multipliers by default: HPR_G1_DSP=0 adds no DSP). Both use the same row / segment scale (both halves
//     are one row); QUANT_WT lanes 4..7 use the upper column table. Each group packs its own int16 words.
//   * mg (new) merges the two groups' output words back into the row's word order (rows modes: owm for wb), or
//     fills them into 8 per-row FIFOs (block modes, 2 blocks deep), so wb drains one block while the next one
//     fills: qt makes a word a clock at 8 lanes and wb's single 8 x 8 buffer (fill, then drain) could not keep up.
//   * wb: rows modes as qz (reading owm); block modes drain from the row FIFOs (no fill phase) and read QUANT_W's
//     row scales from sq while draining the block's first unit.
//   * sr: QZ_SC_ENGINES = 4 (one scale per 6.75 clocks): a 64-value segment takes 8 clocks at 8 lanes.
// Throughput with HPR on: 2 input words (8 FP32) a clock in, 1 output word a clock out (io's write rate).
#pragma once
#ifndef QZ_SC_ENGINES
#define QZ_SC_ENGINES 4
#endif
#if QZ_SC_ENGINES != 4
#error "hpr_quant.h needs QZ_SC_ENGINES = 4 (one segment scale per 8 clocks at 8 lanes)"
#endif
#include "../../quantize/quant_mode.h"
#include "hpr_contract.h"
#ifdef HQ_DFSIM
#include <thread>
#endif
#ifdef QZ_IMPL_FDIV
#error "hpr_quant.h implements the integer quantizer datapath only"
#endif

#ifndef HPR_G1_DSP
#define HPR_G1_DSP 0  // 1: lanes 4..7 use DSPs as lanes 0..3 do (+12 DSP); 0: fabric multipliers (+0 DSP)
#endif

namespace hpq {

using qz::abits_t;
using qz::fbits_t;
using qz::q_t;
using qz::RowScale;
using qz::RP;
using qz::word_t;
using qz::wa_t;
using qz::wp_t;
using qz::KX;
using qz::KXS;
using qz::KW;
using qz::KT;
using qz::KC;

// stream depths (named for the bounded-FIFO model)
#ifndef HQ_D_XS
#define HQ_D_XS 16    // rd -> amx tokens, each group
#endif
#ifndef HQ_D_D0
#define HQ_D_D0 1024  // row buffer of lanes 0..3: one row of up to 768 words + its unit padding (HPR off)
#endif
#ifndef HQ_D_D1
#define HQ_D_D1 512   // row buffer of lanes 4..7: half a row (HPR on), <= 48 slots x 8 entries
#endif
#ifndef HQ_D_OW
#define HQ_D_OW 64    // qt -> mg, each group
#endif
#ifndef HQ_D_OM
#define HQ_D_OM 16    // mg -> wb (rows modes)
#endif
#ifndef HQ_D_RQ
#define HQ_D_RQ 16    // mg -> wb, one FIFO per block row: two blocks
#endif
static_assert(HQ_D_D0 >= qz::ROWBUF_W + 8, "row buffer 0 must hold one row read once, with its unit padding");
static_assert(HQ_D_D1 >= (qz::ROWBUF_W / 8 + 1) / 2 * 8, "row buffer 1 must hold half a row");

// The plan: qz's RP plus the unit-grid invariants of this unit (computed once per command).
struct HP {
  RP q;
  bool on;                 // HPR active: odd units over hp1, 8 lanes
  ap_uint<10> num1;        // rows modes (KX / KC): units per row pass - 1 (W <= 4096 words: <= 511)
  ap_uint<10> nslm1;       // rows modes: slots per row pass - 1
  ap_uint<13> uoL;         // rows modes: first word of the last unit (W - 8 if shifted back, else 8 num1)
  ap_uint<4> skL;          // rows modes: leading words of the last unit that are not new
  ap_uint<13> sem1;        // qt: entries per segment - 1 (rows modes; KT below)
  ap_uint<3> lopu;         // mg rows: log2 of output words per input unit (int16: 2 -> 4 words, COPY: 3 -> 8)
};

static inline uint32_t plan(const ctr::Cmd& c, uint32_t hpr, HP& h) {
#pragma HLS INLINE
  const uint32_t r = qz::row_plan(c, h.q);
  if (r != ctr::ME_OK) return r;
  const RP& p = h.q;
  h.on = hpr::hpr_active_row(c, hpr) && !p.rsmall;
  const uint32_t W = p.W, nu = (W + 7) >> 3, rem = W & 7;
  h.num1 = nu - 1;
  const uint32_t nsl = h.on ? (nu + 1) >> 1 : nu;
  h.nslm1 = nsl - 1;
  h.uoL = rem ? W - 8 : 8 * (nu - 1);
  h.skL = rem ? 8 - rem : 0;
  const bool rowsk = p.kind == KX || p.kind == KC;
  h.sem1 = p.rsmall ? (W - 1) : rowsk ? (8 * nsl - 1) : (h.on ? 7u : 15u);
  h.lopu = p.kind == KC ? 3 : 2;
  return ctr::ME_OK;
}

// ---------------------------------------------------------------- tokens
struct XT {             // rd -> amx
  word_t w;
  ap_uint<1> nw;        // a new word (else: a shifted-back word already read, or a filler slot)
  ap_uint<1> fwd;       // a data pass: the token goes to the row buffer (with nw)
  ap_uint<1> amx;       // an absmax pass (counts when nw)
  ap_uint<1> last;      // last token of a segment's absmax pass / of a QUANT_WT chunk's pass 1
  ap_uint<1> lam;       // with last: the command's last absmax
  ap_uint<1> end;       // last token of the command
  ap_uint<4> col;       // QUANT_WT pass 1: word position inside the 64-column chunk (0..15)
};
typedef ap_uint<129> DE;  // row buffer entry: [127:0] word, [128] new
static inline DE de(const word_t& w, bool v) {
#pragma HLS INLINE
  return (DE(v ? 1 : 0) << 128) | DE(w);
}

// ---------------------------------------------------------------- rd: one port's units
// G = 0: io (xmem), G = 1: hp1 (only when h.on). One iteration = one slot = one 8-word unit = one 8-beat burst
// (II = 8: requests stay in flight across rows).
template <int G>
static void hq_rd(const word_t* mem, const HP h, hls::stream<XT>& xs) {
  const RP& p = h.q;
  const bool on = h.on;
  if (G == 1 && !on) return;
  const bool kt = p.kind == KT, kx = p.kind == KX;
  const bool segmaj = p.kind == KXS || p.kind == KW;
  if (G == 0 && QZ_SMALL && p.rsmall) {  // rows of < 64 values: qz's single-word reads (b, [pass,] n, w), io only
    ap_uint<17> b = 0, n = 0;
    ap_uint<4> w = 0;
    ap_uint<1> pass = 0;
    wa_t ib = p.a, row = p.a;
  rd1:
    for (;;) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 2 max = 4096 avg = 400
      XT t;
      t.w = mem[row + w];
      const bool rowend = w == p.Wm1, lastrow = n == p.Nm1, lastitem = b == p.Bm1;
      t.nw = 1;
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
  // 8-word units. Loop order (b, j, n, pass, u) as qz_rd: j the KXS / KW segment or the QUANT_WT step, pass the
  // QUANT_X re-read, u the slot in a row pass.
  const ap_uint<9> nl = p.nsegm1;
  ap_uint<17> b = 0, n = 0;
  ap_uint<9> j = 0, c1 = 0, c2 = 0;
  ap_uint<10> u = 0;
  bool pass = false;
  wa_t ib = p.a, rb = p.a;
rd8:
  for (;;) {
#pragma HLS PIPELINE II = 8
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 131072 avg = 3000
    const bool p1 = kt && c1 != p.nseg && ap_uint<9>(c1 - c2) != 2;
    const ap_uint<9> cc = p1 ? c1 : c2;
    const bool khalf = kt && p.half && cc == nl;  // QUANT_WT's 32-column last chunk: one unit
    // this group's unit in the slot: v = 2u + G (HPR on) or u (off); a missing odd unit re-reads the even one
    ap_uint<13> uo;
    ap_uint<4> skip = 0;
    bool fill = false;
    bool lastslot;
    if (segmaj || kt) {
      const ap_uint<1> v = on ? ap_uint<1>(G) : ap_uint<1>(u[0]);
      fill = khalf && v == 1;
      const ap_uint<13> base = segmaj ? ap_uint<13>(ap_uint<13>(j) << 4) : ap_uint<13>(ap_uint<13>(cc) << 4);
      uo = base + ((v == 1 && !fill) ? 8 : 0);
      lastslot = on || khalf || u[0] == 1;
    } else {
      const ap_uint<10> v = on ? ap_uint<10>((u << 1) | G) : u;
      fill = v > h.num1;
      const bool lu = v >= h.num1;  // the last unit (or a filler slot: re-read the last unit, not new)
      uo = lu ? h.uoL : ap_uint<13>(ap_uint<13>(v) << 3);
      skip = (lu && !fill) ? h.skL : ap_uint<4>(0);
      lastslot = u == h.nslm1;
    }
    const wa_t addr = rb + uo;
    const bool lastrow = n == p.Nm1, lastitem = b == p.Bm1;
    const bool rowfin = lastslot && (!p.twopass || pass);
    const bool lastj = segmaj ? j == nl : kt ? (!p1 && c2 == nl) : true;
    const bool fin = rowfin && lastrow && lastj && lastitem;
    const bool amx = kx ? (!p.twopass || !pass) : kt ? p1 : segmaj;
    const bool last = lastslot && (kx ? amx : kt ? (p1 && lastrow) : segmaj);
    const bool lam = last && lastitem && (kt ? cc == nl : lastrow && (!segmaj || j == nl));
    const bool fwd = kx ? (!p.twopass || pass) : kt ? !p1 : true;
    const ap_uint<4> col0 = (kt && !on && u[0]) || (kt && on && G == 1) ? 8 : 0;
    for (int k = 0; k < 8; k++) {
#pragma HLS UNROLL
      XT t;
      t.w = mem[addr + k];  // unconditional: one 8-beat burst, pipelined across iterations
      t.nw = !fill && ap_uint<4>(k) >= skip;
      t.fwd = fwd;
      t.amx = amx;
      t.last = k == 7 && last;
      t.lam = k == 7 && lam;
      t.col = col0 + k;
      t.end = k == 7 && fin;
      xs.write(t);
    }
    if (fin) break;
    if (!lastslot) {
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

// ---------------------------------------------------------------- amx: absmax over both groups + row buffers
static void hq_amx(const HP h, hls::stream<XT>& xs0, hls::stream<XT>& xs1, hls::stream<qz::AM>& am,
                   hls::stream<DE>& d0, hls::stream<DE>& d1) {
  const RP& p = h.q;
  const bool kt = p.kind == KT, on = h.on;
  abits_t cl[4][8], chi[4][8];  // QUANT_WT: column absmax of chunk words 0..7 / 8..15, lane l
#pragma HLS ARRAY_PARTITION variable = cl dim = 1 complete
#pragma HLS ARRAY_PARTITION variable = chi dim = 1 complete
#pragma HLS BIND_STORAGE variable = cl type = ram_2p impl = lutram
#pragma HLS BIND_STORAGE variable = chi type = ram_2p impl = lutram
#pragma HLS DEPENDENCE variable = cl inter false  // a column word recurs after >= 8 tokens (one slot)
#pragma HLS DEPENDENCE variable = chi inter false
  if (kt) {
  clr:
    for (int w = 0; w < 8; w++) {
#pragma HLS PIPELINE II = 1
      for (int l = 0; l < 4; l++) {
#pragma HLS UNROLL
        cl[l][w] = 0;
        chi[l][w] = 0;
      }
    }
  }
  abits_t m = 0;
  ap_uint<9> cc = 0;
outer:
  for (;;) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 512 avg = 4
    bool pass_end = false, end = false, lam = false;
  words:
    for (;;) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 2 max = 1048576 avg = 20000
      const XT t = xs0.read();
      XT t1;
      t1.w = 0;
      t1.nw = 0;
      if (on) t1 = xs1.read();
      if (t.fwd) {
        d0.write(de(t.w, t.nw));
        if (on) d1.write(de(t1.w, t1.nw));
      }
      if (t.amx) {
        if (kt) {
          // lo table: group 0's words 0..7; hi table: group 1's words 8..15 (on) or group 0's (off)
          const ap_uint<3> a = t.col(2, 0);  // the same entry for both tables (on: col0 = k, col1 = 8 + k)
          const bool wlo = t.nw && !t.col[3];
          const bool whi = on ? bool(t1.nw) : bool(t.nw && t.col[3]);
          for (int l = 0; l < 4; l++) {
#pragma HLS UNROLL
            const abits_t x0 = qz::abs_bits(qz::word_f32(t.w, l));
            const abits_t x1 = qz::abs_bits(qz::word_f32(on ? t1.w : t.w, l));
            const abits_t ol = cl[l][a], oh = chi[l][a];
            if (wlo) cl[l][a] = qz::amax2(ol, x0);
            if (whi) chi[l][a] = qz::amax2(oh, x1);
          }
        } else {
          const abits_t a0 = t.nw ? qz::word_amax(t.w) : abits_t(0);
          const abits_t a1 = t1.nw ? qz::word_amax(t1.w) : abits_t(0);
          const abits_t mm = qz::amax2(m, qz::amax2(a0, a1));
          if (t.last) {
            qz::AM o;
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
    if (pass_end) {  // the chunk's column absmax in column order, cleared for the next chunk
      const ap_uint<7> cw = (p.half && cc == p.nsegm1) ? 32 : 64;
    emit:
      for (ap_uint<7> c = 0; c < cw; c++) {
#pragma HLS PIPELINE II = 1
        qz::AM o;
        const ap_uint<3> e = c(4, 2);
        o.a = c[5] ? chi[c(1, 0)][e] : cl[c(1, 0)][e];
        o.end = lam && c == cw - 1;
        am.write(o);
        if (c[5]) chi[c(1, 0)][e] = 0;
        else cl[c(1, 0)][e] = 0;
      }
      cc = cc == p.nsegm1 ? ap_uint<9>(0) : ap_uint<9>(cc + 1);
    }
    if (end) break;
  }
}

// ---------------------------------------------------------------- the quantizer of lanes 4..7
// quant_int16.h's quant_elem_recip with the products bound to fabric (HPR_G1_DSP = 0): the same arithmetic, so
// the same bits (tb_hpr checks both against quant_elem_recip on 0.4 M rows x ~600 values).
static inline q_t quant_elem_g1(fbits_t xb, const RowScale& rs) {
#pragma HLS INLINE
#if HPR_G1_DSP
  return qz::quant_elem_recip(xb, rs);
#else
  ap_uint<8> E = xb(30, 23);
  bool neg = xb[31];
  ap_uint<24> X = (ap_uint<1>(1), ap_uint<23>(xb(22, 0)));
  ap_uint<50> P = X * rs.R;
#pragma HLS BIND_OP variable = P op = mul impl = fabric
  ap_uint<26> Qc = P >> 24;
  ap_uint<25> Qlo = Qc(24, 0);
  ap_uint<17> Slo = rs.S(16, 0);
  ap_uint<7> Shi = rs.S(23, 17);
  ap_uint<42> p1 = Qlo * Slo;
#pragma HLS BIND_OP variable = p1 op = mul impl = fabric
  ap_uint<8> qlo = Qc(7, 0);
  ap_uint<15> p2 = qlo * Shi;
#pragma HLS BIND_OP variable = p2 op = mul impl = fabric
  ap_uint<25> lo = p1(24, 0) + (ap_uint<25>(p2(7, 0)) << 17);
  bool lz = lo == 0;
  bool ge = !lz && lo <= rs.T;
  bool st0 = ge ? lo != rs.T : !lz;
  ap_uint<26> Q = Qc + ap_uint<26>(ge);
  bool big = Q[25];
  ap_uint<26> Z = big ? Q : ap_uint<26>(Q << 1);
  ap_uint<25> M = Z >> 2;
  bool rb = Z[1], st = Z[0] | st0;
  if (rb & (st | M[0])) M += 1;
  ap_int<10> sh = ap_int<10>(rs.B) - ap_int<10>(E) + (big ? 0 : 1);
  q_t q = qz::rne_sat(M, sh, neg);
  return E == 0 ? q_t(0) : q;
#endif
}

// ---------------------------------------------------------------- qt: 8 lanes
static void hq_qt(const HP h, hls::stream<DE>& d0, hls::stream<DE>& d1, hls::stream<qz::ST>& sf,
                  hls::stream<word_t>& ow0, hls::stream<word_t>& ow1, hls::stream<ap_uint<32>>& sq) {
  const RP& p = h.q;
  const bool on = h.on;
  if (p.kind == KC) {  // FP32 rows straight through: each group's new words to its stream
    ap_uint<13> w = 0;
    ap_uint<17> n = 0, b = 0;
  copy:
    for (;;) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 2 max = 100000 avg = 20000
      const DE e0 = d0.read();
      DE e1 = 0;
      if (on) e1 = d1.read();
      if (e0[128]) ow0.write(word_t(e0(127, 0)));
      if (on && e1[128]) ow1.write(word_t(e1(127, 0)));
      if (w != h.sem1) {
        w++;
      } else {
        w = 0;
        if (n != p.Nm1) {
          n++;
        } else {
          n = 0;
          if (b == p.Bm1) break;
          b++;
        }
      }
    }
    return;
  }
  const bool kt = p.kind == KT;
  // the row parameters: rows modes entry 0 of both tables; QUANT_WT column 4 w + l (w < 8: lo, else hi)
  ap_uint<24> lS[4][8], hS[4][8];
  ap_uint<26> lR[4][8], hR[4][8];
  ap_uint<25> lT[4][8], hT[4][8];
  ap_uint<9> lB[4][8], hB[4][8];
#pragma HLS ARRAY_PARTITION variable = lS dim = 1 complete
#pragma HLS ARRAY_PARTITION variable = lR dim = 1 complete
#pragma HLS ARRAY_PARTITION variable = lT dim = 1 complete
#pragma HLS ARRAY_PARTITION variable = lB dim = 1 complete
#pragma HLS ARRAY_PARTITION variable = hS dim = 1 complete
#pragma HLS ARRAY_PARTITION variable = hR dim = 1 complete
#pragma HLS ARRAY_PARTITION variable = hT dim = 1 complete
#pragma HLS ARRAY_PARTITION variable = hB dim = 1 complete
#pragma HLS BIND_STORAGE variable = lS type = ram_2p impl = lutram
#pragma HLS BIND_STORAGE variable = lR type = ram_2p impl = lutram
#pragma HLS BIND_STORAGE variable = lT type = ram_2p impl = lutram
#pragma HLS BIND_STORAGE variable = lB type = ram_2p impl = lutram
#pragma HLS BIND_STORAGE variable = hS type = ram_2p impl = lutram
#pragma HLS BIND_STORAGE variable = hR type = ram_2p impl = lutram
#pragma HLS BIND_STORAGE variable = hT type = ram_2p impl = lutram
#pragma HLS BIND_STORAGE variable = hB type = ram_2p impl = lutram
  ap_uint<64> lo0 = 0, lo1 = 0;  // the 4 int16 of the even entry of a pair, per group
  bool ph0 = false, ph1 = false;  // per group: an even entry is held in lo
  word_t pend0 = 0, pend1 = 0;   // the last finished word of each group, written on the next clock
  bool pv0 = false, pv1 = false;
  ap_uint<13> wi = 0;
  ap_uint<17> n = 0, b = 0;
  ap_uint<9> cc = 0;
  ap_uint<7> lc = 0;
  bool endseg = false;
  bool load = kt;
  ap_uint<7> cw = (p.half && p.nseg == 1) ? 32 : 64;
quant:
  for (;;) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 2 max = 1048576 avg = 20000
#pragma HLS DEPENDENCE variable = lS inter false
#pragma HLS DEPENDENCE variable = lR inter false
#pragma HLS DEPENDENCE variable = lT inter false
#pragma HLS DEPENDENCE variable = lB inter false
#pragma HLS DEPENDENCE variable = hS inter false
#pragma HLS DEPENDENCE variable = hR inter false
#pragma HLS DEPENDENCE variable = hT inter false
#pragma HLS DEPENDENCE variable = hB inter false
    if (pv0) {
      ow0.write(pend0);
      pv0 = false;
    }
    if (pv1) {
      ow1.write(pend1);
      pv1 = false;
    }
    const bool rd_sf = load || (!kt && wi == 0);
    qz::ST t;
    t.S = 0;
    t.R = 0;
    t.T = 0;
    t.B = 0;
    t.st = 0;
    t.end = 0;
    if (rd_sf) {
      t = sf.read();
      sq.write(t.st);
    }
    // one store site per table: KT load entry lc -> lane lc % 4, word lc / 4 (lo: < 8, hi: >= 8); a new row /
    // segment (not KT): entry 0 of both tables of every lane
    const ap_uint<3> wad = load ? ap_uint<3>(lc(4, 2)) : ap_uint<3>(0);
    for (int l = 0; l < 4; l++) {
#pragma HLS UNROLL
      const bool mine = !load || lc(1, 0) == l;
      if (rd_sf && mine && !(load && lc[5])) {
        lS[l][wad] = t.S;
        lR[l][wad] = t.R;
        lT[l][wad] = t.T;
        lB[l][wad] = t.B;
      }
      if (rd_sf && mine && (!load || lc[5])) {
        hS[l][wad] = t.S;
        hR[l][wad] = t.R;
        hT[l][wad] = t.T;
        hB[l][wad] = t.B;
      }
    }
    if (load) {
      if (lc == cw - 1) {
        lc = 0;
        load = false;
      } else {
        lc++;
      }
    } else {
      if (rd_sf) endseg = t.end;
      const DE e0 = d0.read();
      DE e1 = 0;
      if (on) e1 = d1.read();
      const ap_uint<3> ti = kt ? ap_uint<3>(wi(2, 0)) : ap_uint<3>(0);
      const bool g0hi = kt && !on && wi[3];  // HPR off, QUANT_WT: lanes 0..3 on chunk words 8..15
      ap_uint<64> h0 = 0, h1 = 0;
      for (int l = 0; l < 4; l++) {
#pragma HLS UNROLL
        RowScale rl, rh;
        rl.S = lS[l][ti];
        rl.R = lR[l][ti];
        rl.T = lT[l][ti];
        rl.B = lB[l][ti];
        rl.s_bits = 0;
        rh.S = hS[l][ti];
        rh.R = hR[l][ti];
        rh.T = hT[l][ti];
        rh.B = hB[l][ti];
        rh.s_bits = 0;
        const RowScale r0 = g0hi ? rh : rl;
        h0(16 * l + 15, 16 * l) = ap_uint<16>(qz::quant_elem_recip(qz::word_f32(word_t(e0(127, 0)), l), r0));
        h1(16 * l + 15, 16 * l) = ap_uint<16>(quant_elem_g1(qz::word_f32(word_t(e1(127, 0)), l), rh));
      }
      if (e0[128]) {
        if (ph0) {
          pend0 = (word_t(h0) << 64) | word_t(lo0);
          pv0 = true;
        } else {
          lo0 = h0;
        }
        ph0 = !ph0;
      }
      if (on && e1[128]) {
        if (ph1) {
          pend1 = (word_t(h1) << 64) | word_t(lo1);
          pv1 = true;
        } else {
          lo1 = h1;
        }
        ph1 = !ph1;
      }
      const ap_uint<13> wlast = kt ? ap_uint<13>((cw == 64 && !on) ? 15 : 7) : h.sem1;
      if (wi != wlast) {
        wi++;
      } else {
        wi = 0;
        if (kt) {
          if (n != p.Nm1) {
            n++;
          } else {
            n = 0;
            if (cc != p.nsegm1) {
              cc++;
            } else {
              cc = 0;
              if (b == p.Bm1) break;
              b++;
            }
            cw = (p.half && cc == p.nsegm1) ? 32 : 64;
            load = true;
          }
        } else if (endseg) {
          break;
        }
      }
    }
  }
  if (pv0) ow0.write(pend0);
  if (pv1) ow1.write(pend1);
}

// ---------------------------------------------------------------- mg: the two groups' words in row order
// Rows modes (KX, KXS, KC): word o of every real row (rows < N) comes from group (o >> lopu) & 1 (int16: 4 words
// per 8-word input unit, FP32: 8), in order -> owm. Block modes (KW, KT): block rows r = 0..7 of every
// (item, segment / chunk, block), RW = 8 (4 in a 32-column QUANT_WT chunk) words each, word w from group w >> 2;
// rows >= N are zeros -> row FIFO r.
static void hq_mg(const HP h, hls::stream<word_t>& ow0, hls::stream<word_t>& ow1, hls::stream<word_t>& owm,
                  hls::stream<word_t> rq[8]) {
  const RP& p = h.q;
  const bool on = h.on;
  if (!p.blk) {
    ap_uint<13> o = 0;
    ap_uint<17> n = 0, b = 0;
    ap_uint<9> j = 0;
  rows:
    for (;;) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 2 max = 1048576 avg = 10000
      const bool g = on && ((o >> h.lopu) & 1);
      const word_t x = g ? ow1.read() : ow0.read();
      owm.write(x);
      if (o != p.WO - 1) {
        o++;
      } else {
        o = 0;
        if (n != p.Nm1) {
          n++;
        } else {
          n = 0;
          if (j != p.njm1) {
            j++;
          } else {
            j = 0;
            if (b == p.Bm1) break;
            b++;
          }
        }
      }
    }
    return;
  }
  const bool kt = p.kind == KT;
  ap_uint<3> r = 0, w = 0;
  ap_uint<14> g = 0;
  ap_uint<9> j = 0;
  ap_uint<17> b = 0;
blocks:
  for (;;) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 64 max = 1048576 avg = 10000
    const bool khalf = kt && p.half && j == p.nsegm1;
    const ap_uint<17> frow = (ap_uint<17>(g) << 3) | r;
    const bool real = frow < p.N;
    word_t x = 0;
    if (real) x = (on && w[2]) ? ow1.read() : ow0.read();
    for (int q = 0; q < 8; q++) {
#pragma HLS UNROLL
      if (r == q) rq[q].write(x);
    }
    if (w != (khalf ? 3 : 7)) {
      w++;
    } else {
      w = 0;
      if (r != 7) {
        r++;
      } else {
        r = 0;
        if (g != p.ngrm1) {
          g++;
        } else {
          g = 0;
          if (j != p.nsegm1) {
            j++;
          } else {
            j = 0;
            if (b == p.Bm1) break;
            b++;
          }
        }
      }
    }
  }
}

// ---------------------------------------------------------------- wb: words + stored scales -> units for wr
// qz_wb with two changes: rows modes read owm (mg's merge, row order as qz's ow); block modes have no fill phase:
// a drain unit pops word u of the block's 8 rows from the row FIFOs at its first word, and QUANT_W reads the
// block's 8 row scales from sq during the drain of unit 0 (word k = row k of the block; rows >= N store 1.0).
static void hq_wb(const RP p, hls::stream<word_t>& ow, hls::stream<word_t> rq[8], hls::stream<ap_uint<32>>& sq,
                  hls::stream<qz::UH>& hd, hls::stream<word_t>& dd) {
  const bool blk = p.blk, kt = p.kind == KT, scl = p.kind != KC;
  const ap_uint<4> USm = (QZ_SMALL && p.wsmall) ? 0 : 7;
  ap_uint<32> win[4][16];
#pragma HLS ARRAY_PARTITION variable = win dim = 1 complete
#pragma HLS BIND_STORAGE variable = win type = ram_2p impl = lutram
  word_t ring[8];
#pragma HLS BIND_STORAGE variable = ring type = ram_2p impl = lutram
  word_t cur[8];  // block mode: word u of the block's rows 0..7 (popped at the unit's first word)
#pragma HLS ARRAY_PARTITION variable = cur complete
  ap_uint<17> b = 0, n = 0;
  ap_uint<9> j = 0;
  ap_uint<14> g = 0;
  ap_uint<13> u = 0;
  ap_uint<4> k = 0;
  ap_uint<7> e = 0;
  ap_uint<15> sw0 = 0;
  bool su = false;
  // phase: 0 data (rows), 1 block drain, 2 scale unit, 3 KT chunk scale load
  ap_uint<2> ph = kt ? 3 : blk ? 1 : 0;
  wa_t cb = p.c, jb = p.c, rowa = p.c;
  wa_t gb = p.c, kb = p.c;
  wa_t fb = p.f, vj = p.f;
wb:
  for (;;) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 8 max = 1048576 avg = 20000
    const bool lastb = b == p.Bm1, lastj = j == p.njm1, lastchunk = j == p.nsegm1;
    const bool khalf = kt && p.half && lastchunk;
    const bool p3 = ph == 3, p2 = ph == 2, p0 = ph == 0, p1 = ph == 1;
    const bool rows = !blk && p0;
    const bool lastu = u == p.num1;
    const bool shb = rows && lastu && p.nf != 0;
    const bool zrow = n >= p.N;
    const bool kwsc = p1 && !kt && u == 0;          // QUANT_W: row k's scale of this block
    const ap_uint<17> frow = n | ap_uint<17>(k(2, 0));
    const bool freal = frow < p.N;
    const bool resend = shb && k < p.nf;
    const bool rd_ow = rows && !zrow && !resend;
    const bool rd_sq = p3 || (rows && scl && u == 0 && k == 0 && !zrow) || (kwsc && freal);
    const bool emit = !p3;
    word_t x = 0;
    if (rd_ow) x = ow.read();
    ap_uint<32> sv = qz::ONE_BITS;
    if (rd_sq) sv = sq.read();
    if (p1 && k == 0) {
      for (int q = 0; q < 8; q++) {
#pragma HLS UNROLL
        cur[q] = rq[q].read();
      }
    }
    const bool wwin = p3 || (rows && scl && u == 0 && k == 0) || kwsc;
    const ap_uint<6> we = p3 ? e(5, 0) : rows ? ap_uint<6>(n(4, 0)) : ap_uint<6>(frow(4, 0));
    if (wwin) win[we(1, 0)][we(5, 2)] = sv;
    const ap_uint<13> ps = shb ? p.WO8 : (QZ_SMALL && p.wsmall) ? u : ap_uint<13>(u << 3);
    const ap_uint<3> ri = ap_uint<3>(ps(2, 0) + k(2, 0));
    if (rows && rd_ow) ring[ri] = x;
    const ap_uint<4> wi = kt ? ap_uint<4>(sw0(3, 0) + k) : ap_uint<4>(ap_uint<3>(sw0(2, 0) + k(2, 0)));
    const bool lastw = !blk && ap_uint<15>(sw0 + k) == p.Lwm1 && p.npl != 0;
    word_t ws, wk;
    for (int l = 0; l < 4; l++) {
#pragma HLS UNROLL
      ws(32 * l + 31, 32 * l) = (lastw && ap_uint<2>(l) >= p.npl) ? ap_uint<32>(qz::ONE_BITS) : win[l][wi];
    }
    for (int q = 0; q < 8; q++) {
#pragma HLS UNROLL
      wk(16 * q + 15, 16 * q) = qz::lane16(cur[q], k(2, 0));
    }
    const word_t wt = cur[k(2, 0)];
    const word_t o = p2 ? ws : p1 ? (kt ? wt : wk) : zrow ? word_t(0) : resend ? ring[ri] : x;
    const ap_uint<2> pi = kt ? ap_uint<2>(u(1, 0)) : ap_uint<2>(g(1, 0));
    const wa_t po = pi == 0 ? wa_t(0) : pi == 1 ? wa_t(p.cp) : pi == 2 ? wa_t(p.cp2) : wa_t(p.cp3);
    const wa_t in = kt ? (u[2] ? wa_t(p.NP) : wa_t(0)) : wa_t(ap_uint<6>(u(2, 0)) << 3);
    const wa_t ua = p2 ? wa_t(vj + sw0) : p1 ? wa_t(gb + po + in) : wa_t(rowa + ps);
    if (emit) {
      if (k == 0) {
        qz::UH hh;
        hh.addr = ua;
        hh.last = 0;
        hd.write(hh);
      }
      dd.write(o);
    }
    bool adv = false;
    if (p3) {
      if (e == (khalf ? 31 : 63)) {
        e = 0;
        ph = 2;
        sw0 = 0;
        su = false;
      } else {
        e++;
      }
    } else if (k != USm) {
      k++;
    } else {
      k = 0;
      if (p2) {
        if (kt) {
          if (su || khalf) {
            ph = 1;  // the chunk's blocks
          } else {
            su = true;
            sw0 = 8;
          }
        } else {
          ph = blk ? 1 : 0;
          adv = true;
        }
      } else if (rows) {
        if (!lastu) {
          u++;
        } else {
          u = 0;
          const bool vend = n == p.NRm1;
          const bool full = (QZ_SMALL && p.wsmall) ? n(1, 0) == 3 : n(4, 0) == 31;
          if (scl && (full || vend)) {
            ph = 2;
            sw0 = (QZ_SMALL && p.wsmall) ? ap_uint<15>(n >> 2) : full ? ap_uint<15>((n >> 2) - 7) : p.Lw8;
          } else {
            adv = true;
          }
        }
      } else {  // block drain
        if (u != (khalf ? 3 : 7)) {
          u++;
        } else {
          u = 0;
          if (!kt && g(1, 0) == 3) {
            ph = 2;
            sw0 = ap_uint<15>(g >> 2) << 3;
          } else {
            adv = true;
          }
        }
      }
    }
    if (adv) {
      if (!blk) {
        const bool lastn = n == p.NRm1;
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
        const bool lastg = g == p.ngrm1;
        if (lastg && lastchunk && lastb) break;
        if (!lastg) {
          if (kt) gb += 8;
          else if (g(1, 0) == 3) gb += 64;
          g++;
          n += 8;
        } else {
          g = 0;
          n = 0;
          if (!lastchunk) {
            j++;
            vj += p.vstep;
            if (kt && !p.kseg) {
              kb += p.np2;
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
          if (kt) ph = 3;
        }
      }
    }
  }
  qz::UH hh;
  hh.addr = 0;
  hh.last = 1;
  hd.write(hh);
}

// ---------------------------------------------------------------- the dataflow region and the mode
static void row_df_hpr(const word_t* hp1, const word_t* xmem, word_t* ymem, const HP h) {
#pragma HLS DATAFLOW
#pragma HLS STABLE variable = h
  hls::stream<XT> xs0("hq_xs0"), xs1("hq_xs1");
  hls::stream<qz::AM> am("hq_am");
  hls::stream<DE> d0("hq_d0"), d1("hq_d1");
  hls::stream<qz::SS> ss("hq_ss");
  hls::stream<qz::ST> sf("hq_sf");
  hls::stream<word_t> ow0("hq_ow0"), ow1("hq_ow1"), owm("hq_owm");
  hls::stream<word_t> rq[8];
  hls::stream<ap_uint<32>> sq("hq_sq");
  hls::stream<qz::UH> hd("hq_hd");
  hls::stream<word_t> dd("hq_dd");
#pragma HLS STREAM variable = xs0 depth = HQ_D_XS
#pragma HLS STREAM variable = xs1 depth = HQ_D_XS
#pragma HLS STREAM variable = am depth = QZ_D_AM
#pragma HLS STREAM variable = d0 depth = HQ_D_D0
#pragma HLS STREAM variable = d1 depth = HQ_D_D1
#pragma HLS STREAM variable = ss depth = QZ_D_SS
#pragma HLS STREAM variable = sf depth = QZ_D_SF
#pragma HLS STREAM variable = ow0 depth = HQ_D_OW
#pragma HLS STREAM variable = ow1 depth = HQ_D_OW
#pragma HLS STREAM variable = owm depth = HQ_D_OM
#pragma HLS STREAM variable = rq depth = HQ_D_RQ
#pragma HLS STREAM variable = sq depth = QZ_D_SQ
#pragma HLS STREAM variable = hd depth = QZ_D_HD
#pragma HLS STREAM variable = dd depth = QZ_D_DD
#pragma HLS BIND_STORAGE variable = d0 type = fifo impl = bram
#pragma HLS BIND_STORAGE variable = d1 type = fifo impl = bram
#pragma HLS BIND_STORAGE variable = am type = fifo impl = lutram
#pragma HLS BIND_STORAGE variable = sq type = fifo impl = lutram
#pragma HLS BIND_STORAGE variable = ow0 type = fifo impl = lutram
#pragma HLS BIND_STORAGE variable = ow1 type = fifo impl = lutram
#pragma HLS BIND_STORAGE variable = xs0 type = fifo impl = lutram
#pragma HLS BIND_STORAGE variable = xs1 type = fifo impl = lutram
#pragma HLS BIND_STORAGE variable = dd type = fifo impl = lutram
#ifdef HQ_DFSIM
  std::thread t0([&] { hq_rd<0>(xmem, h, xs0); });
  std::thread t1([&] { hq_rd<1>(hp1, h, xs1); });
  std::thread t2([&] { hq_amx(h, xs0, xs1, am, d0, d1); });
  std::thread t3([&] { qz::qz_ss(h.q, am, ss); });
  std::thread t4([&] { qz::qz_sr(h.q, ss, sf); });
  std::thread t5([&] { hq_qt(h, d0, d1, sf, ow0, ow1, sq); });
  std::thread t6([&] { hq_mg(h, ow0, ow1, owm, rq); });
  std::thread t7([&] { hq_wb(h.q, owm, rq, sq, hd, dd); });
  std::thread t8([&] { qz::qz_wr(ymem, h.q, hd, dd); });
  t0.join();
  t1.join();
  t2.join();
  t3.join();
  t4.join();
  t5.join();
  t6.join();
  t7.join();
  t8.join();
#else
  hq_rd<0>(xmem, h, xs0);
  hq_rd<1>(hp1, h, xs1);
  hq_amx(h, xs0, xs1, am, d0, d1);
  qz::qz_ss(h.q, am, ss);
  qz::qz_sr(h.q, ss, sf);
  hq_qt(h, d0, d1, sf, ow0, ow1, sq);
  hq_mg(h, ow0, ow1, owm, rq);
  hq_wb(h.q, owm, rq, sq, hd, dd);
  qz::qz_wr(ymem, h.q, hd, dd);
#endif
}

// The row unit's mode function for QUANT_X / QUANT_W / QUANT_WT / COPY with hp_reads: one call site in the
// sequencer's dispatch switch (replaces qz::row_mode). hpr = record field 56 (hpr::FLD_HPR), already checked by
// hpr::validate_hpr. Returns ME_OK, or ME_DIMS (nothing read or written) as qz::row_mode. Returns after the last
// write response (R-ORD).
static uint32_t row_mode_hpr(const ctr::Cmd& c, uint32_t hpr, const word_t* hp1, const word_t* xmem,
                             word_t* ymem) {
#pragma HLS INLINE off
  HP h;
  const uint32_t r = plan(c, hpr, h);
  if (r != ctr::ME_OK) return r;
  row_df_hpr(hp1, xmem, ymem, h);
  return ctr::ME_OK;
}

}  // namespace hpq
