"""Apply the emit_amax patch to the integrator's fpga/hls/gemm_contract.h and fpga/kernels/contract.h.

Every edit is anchored on text that must occur exactly once in the source; if the integrator has changed a region
the patch touches, the script stops and names the anchor (re-base by hand then). Writes patched/hls/gemm_contract.h
and patched/kernels/contract.h, and the unified diffs patch_gemm_contract.diff / patch_contract.diff.

    python make_patch.py [--hls ../../../hls/gemm_contract.h] [--contract ../../contract.h]
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sub1(s: str, old: str, new: str, what: str) -> str:
    n = s.count(old)
    if n != 1:
        raise SystemExit(f"anchor '{what}' found {n} times (want 1): re-base the patch")
    return s.replace(old, new)


def between(s: str, start: str, end: str, new: str, what: str) -> str:
    if s.count(start) != 1:
        raise SystemExit(f"start anchor of '{what}' found {s.count(start)} times")
    a = s.index(start)
    b = s.index(end, a)
    return s[:a] + new + s[b:]


# ---------------------------------------------------------------- gemm_contract.h
HDR_OLD = """#pragma once
#include "int16_linear.h"
#include "../kernels/contract.h"
"""
HDR_NEW = """//
// Phase B emit_amax (fpga/kernels/phaseB/emit_amax/emit_amax.h, INTEGRATION.md):
//   p7 = P_AMX_SG (producer): the store also writes the absmax of every (row, segment of P_AMX_SG
//     groups) of the bits it writes to slot H, one word per 4 rows after a segment's last group (the store loop is
//     word-level at II = 1: one io word a clock as before, + ceil(rows / 4) clocks per segment and tile).
//   p8 = P_XQ (consumer): slot A holds FP32 rows and F the absmax parts; load_io quantizes X on the way in
//     (AMX_QL values a clock) with the contract quantizer and sends s_tok = s 2^p4 itself. No QUANT_X command.
#pragma once
#include "int16_linear.h"
#include "../kernels/contract.h"
#include "../kernels/phaseB/emit_amax/emit_amax.h"
"""

ITEM_OLD = """  int N, K, groups, nblk, ntiles, nch;
  bool bias, addd;
};
"""
ITEM_NEW = """  int N, K, groups, nblk, ntiles, nch;
  bool bias, addd;
  // emit_amax
  int asg;           // producer: groups per absmax segment (0 = no absmax output)
  uint64_t ha, hp;   // slot H of this item, bytes between absmax parts
  int xq;            // consumer: absmax parts at F (0 = int16 X rows, the 1.0 GEMM)
  uint64_t fpp;      // F pitch: bytes between parts
  int shift;         // consumer: s_tok = s 2^p4
};
"""

XGEN_OLD = """  int a, c, n0;
  uint64_t row, blk_row0;
};
"""
XGEN_NEW = """  int a, c, n0;
  uint64_t row, blk_row0;
  int bt;  // block of the tile (emit_amax: the loader's scale bank row is bt * TB + a)
};
static inline void x_next(const Item& it, XGen& x) {
#pragma HLS INLINE
  if (x.a == TB - 1) {
    x.a = 0;
    if (x.c == it.nch - 1) {
      x.c = 0;
      x.blk_row0 += (uint64_t)TB * it.xp;
      x.n0 += TB;
      x.bt = (x.bt + 1) & (NBX - 1);
    } else {
      x.c++;
    }
    x.row = x.blk_row0;
  } else {
    x.a++;
    x.row += it.xp;
  }
}
"""

XWALK_OLD = """    xr.write(live ? v : zero_word());
  }
  if (x.a == TB - 1) {
    x.a = 0;
    if (x.c == it.nch - 1) {
      x.c = 0;
      x.blk_row0 += (uint64_t)TB * it.xp;
      x.n0 += TB;
    } else {
      x.c++;
    }
    x.row = x.blk_row0;
  } else {
    x.a++;
    x.row += it.xp;
  }
}
"""
XWALK_NEW = """    xr.write(live ? v : zero_word());
  }
  x_next(it, x);
}

// ---- emit_amax consumer: FP32 X rows quantized on load (P_XQ)
// The quantizer state of the 64 rows of a tile in two banks (tile parity): load_io fills tile t + 1's bank before
// it reads X(t + 1), and tile 0's before X(0). stk holds the s_tok words of the bank's tile.
typedef famx::XScale XBank[2][NTT];
typedef IoWord SBank[2][NTT / F4];
// Tile t's bank from the absmax parts at F: a = max over the parts (exact), then one row a clock through s, the
// divider and the stored scale (emit_amax.h xscale: the row unit's path).
template <class W>
static inline void xq_scales(const W* io, const Item& it, int t, int bank, XBank& rsb, SBank& stk) {
#pragma HLS INLINE
  const uint64_t f_words = (uint64_t)ctr::pad4((uint32_t)it.N) / F4;
  famx::abits am[NTT / F4][F4];
#pragma HLS ARRAY_PARTITION variable = am complete dim = 2
#pragma HLS BIND_STORAGE variable = am type = ram_2p impl = lutram
  uint64_t pw = it.fa >> 4;  // word index of part j
  const uint64_t t0 = (uint64_t)t * (NTT / F4);
  int j = 0, w = 0;
xq_am:
  for (int i = 0; i < it.xq * (NTT / F4); i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS DEPENDENCE variable = am inter false
#pragma HLS LOOP_TRIPCOUNT min = 16 max = 1024
    const uint64_t wi = t0 + (uint64_t)w;
    const IoWord v = wi < f_words ? to_io(io[pw + wi]) : zero_word();
    for (int l = 0; l < F4; l++) {
#pragma HLS UNROLL
      const famx::abits a = famx::abits(famx::lane_bits(v, l) & 0x7FFFFFFFu);
      am[w][l] = j == 0 ? a : famx::amax2(am[w][l], a);
    }
    if (w == NTT / F4 - 1) {
      w = 0;
      j++;
      pw += it.fpp >> 4;
    } else {
      w++;
    }
  }
  ap_uint<32> sv[F4];  // the stored scales of rows 4 w .. 4 w + 3 (constant lanes: no 128-bit shifter)
#pragma HLS ARRAY_PARTITION variable = sv complete
xq_sc:
  for (int r = 0; r < NTT; r++) {
#pragma HLS PIPELINE II = 1
    ap_uint<32> st;
    rsb[bank][r] = famx::xscale(am[r >> 2][r & 3], st, ap_int<5>(it.shift));
    sv[r & 3] = st;
    if ((r & 3) == 3) {
      IoWord sw = zero_word();
      for (int l = 0; l < F4; l++) {
#pragma HLS UNROLL
        famx::set_lane_bits(sw, l, l == 3 ? (uint32_t)st : (uint32_t)sv[l]);
      }
      stk[bank][r >> 2] = sw;
    }
  }
}
// Units of X (x_unit's walk) from FP32 rows until the credit xg reaches thr, AMX_QL values a clock: a live unit reads
// its chunk's kc / 4 words (16 at most, none past the row's K values) and sends 8 int16 words; a unit of a token
// n >= N reads nothing and sends 8 zero words (8 clocks).
template <class W>
static inline void xq_run(const W* io, const Item& it, XGen& x, Fifo16<IoWord>& xr, const XBank& rsb, int bank,
                          uint64_t& xg, uint64_t thr, uint64_t step) {
#pragma HLS INLINE
  const int klast = it.K - CK * (it.nch - 1);
  int w = 0;  // clock of the unit
  bool live = false;
  uint64_t base = 0;
  int kw = 0;  // FP32 words of the chunk
  IoWord v = zero_word();
  ap_uint<16> qb[2 * F4];  // the output word's 8 int16 (constant lanes when it is sent)
#pragma HLS ARRAY_PARTITION variable = qb complete
xq_x:
  while (xg < thr) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 0 max = 65536
    if (w == 0) {
      live = x.n0 + x.a < it.N;
      base = (x.row + (uint64_t)x.c * (CK * 4)) >> 4;
      kw = (x.c == it.nch - 1 ? klast : CK) / F4;
    }
    bool last;
    if (live) {
      const int wd = w / famx::QIT, ph = w % famx::QIT;  // FP32 word of the chunk, lane group
      if (ph == 0) v = wd < kw ? to_io(io[base + (uint64_t)wd]) : zero_word();
      const famx::XScale rs = rsb[bank][x.bt * TB + x.a];
      ap_uint<16> qn[famx::QL];
      for (int l = 0; l < famx::QL; l++) {
#pragma HLS UNROLL
        const int lane = ph * famx::QL + l;
        qn[l] = ap_uint<16>(famx::quant_elem_fab(qz::fbits_t(famx::lane_bits(v, lane)), rs));
        qb[(wd & 1) * F4 + lane] = qn[l];
      }
      if ((wd & 1) && ph == famx::QIT - 1) {
        IoWord o = zero_word();
        for (int j = 0; j < 2 * F4; j++) {
#pragma HLS UNROLL
          // this clock's lanes straight from the quantizer, the others from qb
          const bool now = j >= F4 + ph * famx::QL;
          set16(o, j, (uint16_t)(now ? qn[(j - F4 - ph * famx::QL) & (famx::QL - 1)] : qb[j]));
        }
        xr.write(o);
      }
      last = w == 2 * UW * famx::QIT - 1;
    } else {
      xr.write(zero_word());
      last = w == UW - 1;
    }
    if (last) {
      w = 0;
      x_next(it, x);
      xg += step;
    } else {
      w++;
    }
  }
}
"""

LOADIO_START = """template <class W>
static void load_io(const W* io, const Item it, Fifo16<IoWord>& xr, Fifo16<IoWord>& stok_s, Fifo16<IoWord>& srow_s,
                    Fifo16<IoWord>& bias_s, Fifo16<IoWord>& ds) {
"""
LOADIO_END = "// ---------------------------------------------------------------- process 3:"
LOADIO_ORIG_SHA = None  # filled by --record; checked when set
LOADIO_NEW = LOADIO_START + """  const uint64_t f_words = (uint64_t)ctr::pad4((uint32_t)it.N) / F4;  // s_tok words the slot holds
  XGen x;
  x.a = 0;
  x.c = 0;
  x.n0 = 0;
  x.row = it.xa;
  x.blk_row0 = it.xa;
  x.bt = 0;
  const int G = it.groups;
  XBank rsb;  // emit_amax consumer (it.xq != 0): quantizer state and s_tok words of tiles t, t + 1
#pragma HLS BIND_STORAGE variable = rsb type = ram_2p impl = bram
  SBank stk;
#pragma HLS ARRAY_PARTITION variable = stk complete dim = 1
#pragma HLS BIND_STORAGE variable = stk type = ram_2p impl = lutram
  // One pass t = -1 .. ntiles - 1 (t = -1: the prologue, X(0) as one credit run of step 1), so that every loop, and
  // the emit_amax quantizer and divider, have one call site each. The io read order is the 1.0 order: s_tok(0),
  // X(0), s_row, bias, then per tile t: s_tok(t) (t >= 1), per group g: D(t, g), slice g of X(t + 1). With P_XQ the
  // absmax parts of tile t + 1 come first in iteration t (the s_tok words then come from the bank, on chip).
  const int units0 = fstream16::blocks_of_tile(it.nblk, 0) * it.nch * TB;
tiles_io:
  for (int t = -1; t < it.ntiles; t++) {
#pragma HLS LOOP_TRIPCOUNT min = 2 max = 1025
    if (it.xq && t + 1 < it.ntiles) xq_scales(io, it, t + 1, (t + 1) & 1, rsb, stk);
    if (t != 0) {  // s_tok of tile ts (t = -1: tile 0)
      const int ts = t < 0 ? 0 : t;
    stok_t:
      for (int w = 0; w < NTT / F4; w++) {
#pragma HLS PIPELINE II = 1
        const uint64_t wi = (uint64_t)ts * (NTT / F4) + w;
        stok_s.write(it.xq ? stk[ts & 1][w] : wi < f_words ? to_io(io[(it.fa >> 4) + wi]) : ones_word());
      }
    }
    const bool pro = t < 0;
    const int nv = pro ? 0 : valid_rows(it.N, t);
    const int unext = pro ? units0 : t + 1 < it.ntiles ? fstream16::blocks_of_tile(it.nblk, t + 1) * it.nch * TB : 0;
    const uint64_t step = pro ? 1 : (uint64_t)G;
    uint64_t thr = 0, xg = 0;  // credit: (g + 1) * unext against units emitted * step
    RowWalk rw0;
    rw0.start(pro ? 0 : (uint64_t)t * NTT, it.dR, it.dP1, it.dp);
  groups_io:
    for (int g = 0; g < (pro ? 1 : G); g++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 256
      if (it.addd && !pro) {
        RowWalk rw = rw0;
      load_dr:
        for (int i = 0; i < nv; i++) {
#pragma HLS PIPELINE II = UW
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 64
          const uint64_t base = (it.da + rw.off() + (uint64_t)g * (LANES * 4)) >> 4;
          for (int w = 0; w < YU; w++) {
#pragma HLS UNROLL
            ds.write(to_io(io[base + w]));
          }
          rw.next(it.dR, it.dP1, it.dp);
        }
      }
      thr += (uint64_t)unext;
      if (it.xq) {
        xq_run(io, it, x, xr, rsb, (t + 1) & 1, xg, thr, step);
      } else {
      load_xs:
        while (xg < thr) {
#pragma HLS PIPELINE II = UW
#pragma HLS LOOP_TRIPCOUNT min = 0 max = 1024
          x_unit(io, it, x, xr);
          xg += step;
        }
      }
    }
    if (pro) {
    load_srow:
      for (int i = 0; i < G * (LANES / F4); i++) {
#pragma HLS PIPELINE II = 1
        srow_s.write(to_io(io[(it.ea >> 4) + i]));
      }
      if (it.bias) {
      load_bias:
        for (int i = 0; i < G * (LANES / F4); i++) {
#pragma HLS PIPELINE II = 1
          bias_s.write(to_io(io[(it.ga >> 4) + i]));
        }
      }
    }
  }
}

"""

STORE_START = "// ---------------------------------------------------------------- process 7: row-major Y\n"
STORE_END = "// ---------------------------------------------------------------- one item\ntemplate"
STORE_NEW = """// ---------------------------------------------------------------- process 7: row-major Y
// One io word a clock (the 1.0 loop wrote 8-word units at II = 8). emit_amax (it.asg != 0): the absmax of every
// (row, segment) over the bits written, as FP32 bits. After a row's unit in a segment's last group its value goes
// into lane i % 4 of pk, and every 4 rows (and after row nv - 1) pk is written to slot H in an extra clock.
template <class W>
static void store(Fifo16<IoWord>& ys, W* io, const Item it) {
  famx::abits amem[NTT];  // running absmax of the tile's rows over the groups of the current segment
#pragma HLS BIND_STORAGE variable = amem type = ram_2p impl = lutram
  const bool amx = it.asg != 0;
  const uint64_t hR4 = it.yR >> 2;  // absmax words per C row group (validate: R % 4 == 0)
  RowWalk aw;                        // absmax word tw (rows 4 tw .. 4 tw + 3) of the item: continues across tiles
  aw.start(0, hR4, it.hp, 16);
tiles_st:
  for (int t = 0; t < it.ntiles; t++) {
    const int nv = valid_rows(it.N, t);
    RowWalk rw0;
    rw0.start((uint64_t)t * NTT, it.yR, it.yP1, it.yp);
    RowWalk rw = rw0;
    const RowWalk aw0 = aw;  // the tile's first absmax word
    RowWalk awx = aw;        // the current segment's next absmax word
    uint64_t sb = 0;         // segment s: s * hp (R = 0)
    int g = 0, i = 0, w = 0, gs = 0;
    uint64_t base = 0;
    famx::abits run = 0;
    famx::abits pk[F4];  // absmax of rows 4 w .. 4 w + 3 (constant lanes: no 128-bit shifter)
#pragma HLS ARRAY_PARTITION variable = pk complete
    for (int l = 0; l < F4; l++) {
#pragma HLS UNROLL
      pk[l] = 0;
    }
    bool apend = false, afin = false;  // an absmax word is due in this clock; it is the segment's last
  store_y:
    while (true) {
#pragma HLS PIPELINE II = 1
#pragma HLS DEPENDENCE variable = amem inter false
#pragma HLS LOOP_TRIPCOUNT min = 8 max = 131072
      bool done = false;
      uint64_t wa;
      IoWord wv;
      if (apend) {
        wa = (it.ha + sb + awx.off()) >> 4;
        wv = zero_word();
        for (int l = 0; l < F4; l++) {
#pragma HLS UNROLL
          famx::set_lane_bits(wv, l, (uint32_t)pk[l]);
          pk[l] = 0;
        }
        apend = false;
        awx.next(hR4, it.hp, 16);
        if (afin) {
          if (g == it.groups) {
            aw = awx;  // the next tile starts 16 words on (only the last tile holds fewer than 64 rows)
            done = true;
          } else {
            awx = aw0;
            sb += it.hp;
          }
        }
      } else {
        if (w == 0) base = (it.ya + rw.off() + (uint64_t)g * (LANES * 4)) >> 4;
        wv = ys.read();
        wa = base + (uint64_t)w;
        const famx::abits m4 = famx::word_amax(wv);
        const famx::abits r = w == 0 ? m4 : famx::amax2(run, m4);
        run = r;
        if (w == YU - 1) {
          w = 0;
          if (amx) {
            const famx::abits a = gs == 0 ? r : famx::amax2(r, amem[i]);
            if (gs == it.asg - 1) {
              pk[i & 3] = a;
              if ((i & 3) == 3 || i == nv - 1) {
                apend = true;
                afin = i == nv - 1;
              }
            } else {
              amem[i] = a;
            }
          }
          if (i == nv - 1) {
            i = 0;
            g++;
            rw = rw0;
            gs = gs == it.asg - 1 ? 0 : gs + 1;
          } else {
            i++;
            rw.next(it.yR, it.yP1, it.yp);
          }
          done = g == it.groups && !apend;
        } else {
          w++;
        }
      }
      io[wa] = from_io<W>(wv);
      if (done) break;
    }
  }
}

"""

PLAN_OLD1 = "static inline Item plan_item(const ctr::Cmd& c, uint64_t b) {"
PLAN_NEW1 = """// hoff = b * H.stride, kept as a running sum by gemm_unit (no new multiplier)
static inline Item plan_item(const ctr::Cmd& c, uint64_t b, uint64_t hoff) {"""
PLAN_OLD2 = """  it.nch = (int)((c.K + CK - 1) / CK);
  return it;
}
"""
PLAN_NEW2 = """  it.nch = (int)((c.K + CK - 1) / CK);
  const ctr::Slot& H = c.s[ctr::SLOT_H];
  it.asg = (int)c.p[ctr::P_AMX_SG];
  it.ha = H.addr + hoff;
  it.hp = H.pitch;
  it.xq = (int)c.p[ctr::P_XQ];
  it.fpp = F.pitch;
  it.shift = (int)(int32_t)c.p[ctr::P_SHIFT];
  return it;
}
"""
UNIT_OLD = """items:
  for (uint32_t b = 0; b < c.B; b++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 24
    const Item it = plan_item(c, b);
    gemm_item(hp0, hp1, hp2, hp3, io_rd, io_wr, it);
  }
"""
UNIT_NEW = """  uint64_t hoff = 0;
items:
  for (uint32_t b = 0; b < c.B; b++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 24
    const Item it = plan_item(c, b, hoff);
    gemm_item(hp0, hp1, hp2, hp3, io_rd, io_wr, it);
    hoff += c.s[ctr::SLOT_H].stride;
  }
"""


def patch_gemm(s: str) -> str:
    s = sub1(s, HDR_OLD, HDR_NEW, "header includes")
    s = sub1(s, ITEM_OLD, ITEM_NEW, "struct Item tail")
    s = sub1(s, XGEN_OLD, XGEN_NEW, "struct XGen")
    s = sub1(s, XWALK_OLD, XWALK_NEW, "x_unit walk")
    s = between(s, LOADIO_START, LOADIO_END, LOADIO_NEW, "load_io")
    s = between(s, STORE_START, STORE_END, STORE_NEW, "store")
    s = sub1(s, PLAN_OLD1, PLAN_NEW1, "plan_item signature")
    s = sub1(s, PLAN_OLD2, PLAN_NEW2, "plan_item tail")
    s = sub1(s, UNIT_OLD, UNIT_NEW, "gemm_unit items loop")
    return s


# ---------------------------------------------------------------- contract.h
C_PARAMS_OLD = """constexpr uint32_t P_NP = 5;      // row ops / SOFTMAX: rows written incl. zero rows [N, NP) (0 = N);
                                  // QUANT_WT: keys of the V^T blob (its K)
"""
C_PARAMS_NEW = """constexpr uint32_t P_NP = 5;      // row ops / SOFTMAX: rows written incl. zero rows [N, NP) (0 = N);
                                  // QUANT_WT: keys of the V^T blob (its K)
                                  // p6: GEMM P_ACT (phase B emit_act)
constexpr uint32_t P_AMX_SG = 7;  // GEMM (phase B emit_amax): absmax output, groups per segment (0 = none)
constexpr uint32_t P_XQ = 8;      // GEMM (phase B emit_amax): X is FP32 rows quantized on load, P_XQ absmax parts
                                  // at F (0 = int16 X rows). GEMM p4 (P_SHIFT) with P_XQ: s_tok = s 2^p4
"""
C_LIMITS_OLD = """constexpr uint32_t SHIFT_MAX = 8;            // |p4|
"""
C_LIMITS_NEW = """constexpr uint32_t SHIFT_MAX = 8;            // |p4|
constexpr uint32_t AMX_NS_MAX = 64;          // GEMM absmax segments (M / 32 / P_AMX_SG)
constexpr uint32_t AMX_PARTS_MAX = 64;       // GEMM P_XQ: absmax parts per row
"""
C_GEMM_OLD = """    const uint64_t R = p[P_SEGC], P1 = p[P_SEGF];
    if (R > N || (R == 0 && P1 != 0) || P1 % ALIGN) return ME_PARAM;
    for (uint32_t i = 0; i < NPARAMS; i++)
      if (i != P_SEGC && i != P_SEGF && p[i]) return ME_PARAM;
    used = (1u << SLOT_A) | (1u << SLOT_B) | (1u << SLOT_C) | (1u << SLOT_E) | (1u << SLOT_F);
    if ((r = check_slot<FULL>(c.s[SLOT_A], B, N, 2 * K, 2 * K, false))) return r;
"""
C_GEMM_NEW = """    const uint64_t R = p[P_SEGC], P1 = p[P_SEGF];
    if (R > N || (R == 0 && P1 != 0) || P1 % ALIGN) return ME_PARAM;
    for (uint32_t i = 0; i < NPARAMS; i++)
      if (i != P_SEGC && i != P_SEGF && i != P_SHIFT && i != P_AMX_SG && i != P_XQ && p[i]) return ME_PARAM;
    // emit_amax: absmax output (producer, P_AMX_SG), X quantized on load (consumer, P_XQ)
    const uint64_t sg = p[P_AMX_SG], xq = p[P_XQ], G = M / ENGINE_LANES;
    const int32_t sh = (int32_t)p[P_SHIFT];
    if (sh < -(int32_t)SHIFT_MAX || sh > (int32_t)SHIFT_MAX || (xq == 0 && sh != 0)) return ME_PARAM;
    if (xq > AMX_PARTS_MAX || sg > G || (sg && R && (sg != G || R % 4))) return ME_PARAM;
    if (FULL && sg && (G % sg || G / sg > AMX_NS_MAX)) return ME_PARAM;  // host only (a divider)
    used = (1u << SLOT_A) | (1u << SLOT_B) | (1u << SLOT_C) | (1u << SLOT_E) | (1u << SLOT_F);
    if ((r = check_slot<FULL>(c.s[SLOT_A], B, N, (xq ? 4 : 2) * K, (xq ? 4 : 2) * K, false))) return r;
    if (sg) {  // part p of row t at H + b stride + p pitch + 4 row (R = 0: p = segment, row = t; else t / R, t % R)
      used |= 1u << SLOT_H;
      const Slot& h = c.s[SLOT_H];
      const uint64_t rows = R ? (N < R ? N : R) : N;
      const bool multi = R ? N > R : G > sg;
      if (!slot_aligned(h) || (multi && h.pitch < 4 * (uint64_t)pad4((uint32_t)rows))) return ME_ALIGN;
      if (h.addr >= ADDR_LIMIT) return ME_ADDR;
      if (FULL) {
        const uint64_t parts = R ? (N - 1) / R + 1 : G / sg;
        if (h.addr + (B - 1) * (uint64_t)h.stride + (parts - 1) * (uint64_t)h.pitch + 4 * pad4((uint32_t)rows) >
            ADDR_LIMIT)
          return ME_ADDR;
      }
    }
"""
C_F_OLD = """    if ((r = check_slot<FULL>(c.s[SLOT_F], B, 1, 4 * pad4(c.N), 0, true))) return r;
    if (bias) {
      used |= 1u << SLOT_G;"""
C_F_NEW = """    if (xq) {  // xq parts of pad4(N) absmax values: part j at F + b stride + j pitch
      const Slot& f = c.s[SLOT_F];
      if (!slot_aligned(f) || (xq > 1 && f.pitch < 4 * (uint64_t)pad4(c.N))) return ME_ALIGN;
      if (f.addr >= ADDR_LIMIT ||
          (FULL && f.addr + (B - 1) * (uint64_t)f.stride + (xq - 1) * (uint64_t)f.pitch + 4 * pad4(c.N) > ADDR_LIMIT))
        return ME_ADDR;
    } else if ((r = check_slot<FULL>(c.s[SLOT_F], B, 1, 4 * pad4(c.N), 0, true))) {
      return r;
    }
    if (bias) {
      used |= 1u << SLOT_G;"""
C_FLAGS_OLD = "constexpr uint32_t SLOT_H = 7;  // unused in 1.0 (must be 0)"
C_FLAGS_NEW = "constexpr uint32_t SLOT_H = 7;  // GEMM with P_AMX_SG (phase B emit_amax): absmax parts out; else 0"


def patch_contract(s: str) -> str:
    s = sub1(s, C_FLAGS_OLD, C_FLAGS_NEW, "SLOT_H")
    s = sub1(s, C_PARAMS_OLD, C_PARAMS_NEW, "params")
    s = sub1(s, C_LIMITS_OLD, C_LIMITS_NEW, "limits")
    s = sub1(s, C_GEMM_OLD, C_GEMM_NEW, "validate GEMM params / slot A")
    s = sub1(s, C_F_OLD, C_F_NEW, "validate GEMM slot F")
    return s


def write_diff(a: Path, b_text: str, label_a: str, label_b: str, out: Path) -> int:
    d = difflib.unified_diff(a.read_text(encoding="utf-8").splitlines(True), b_text.splitlines(True), label_a, label_b)
    t = "".join(d)
    out.write_text(t, encoding="utf-8", newline="\n")
    return t.count("\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hls", type=Path, default=HERE.parents[2] / "hls" / "gemm_contract.h")
    ap.add_argument("--contract", type=Path, default=HERE.parents[1] / "contract.h")
    a = ap.parse_args()
    g = patch_gemm(a.hls.read_text(encoding="utf-8"))
    c = patch_contract(a.contract.read_text(encoding="utf-8"))
    (HERE / "patched" / "hls").mkdir(parents=True, exist_ok=True)
    (HERE / "patched" / "kernels").mkdir(parents=True, exist_ok=True)
    (HERE / "patched" / "hls" / "gemm_contract.h").write_text(g, encoding="utf-8", newline="\n")
    (HERE / "patched" / "kernels" / "contract.h").write_text(c, encoding="utf-8", newline="\n")
    n1 = write_diff(a.hls, g, "a/fpga/hls/gemm_contract.h", "b/fpga/hls/gemm_contract.h", HERE / "patch_gemm_contract.diff")
    n2 = write_diff(a.contract, c, "a/fpga/kernels/contract.h", "b/fpga/kernels/contract.h", HERE / "patch_contract.diff")
    for p in (a.hls, a.contract):
        print(f"base {p}: sha256 {hashlib.sha256(p.read_bytes()).hexdigest()}")
    print(f"wrote patched/hls/gemm_contract.h, patched/kernels/contract.h; diffs {n1} + {n2} lines")


if __name__ == "__main__":
    main()
