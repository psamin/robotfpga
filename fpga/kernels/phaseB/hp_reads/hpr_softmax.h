// hp_reads, softmax part: SOFTMAX reading its score rows over io AND hp1, 8 lanes, two 1/sum dividers.
// Built against fpga/kernels/softmax/softmax_core.h (smx::softmax_mode, the integrator's current softmax unit):
// its parameters (smx::Params, make_params), its arithmetic helpers (okey, key_bits, neg_g, q_pair's table and
// arithmetic, the divider's div_load / div_step / div_result) are used as they are, so every output byte is
// smx::softmax_mode's (tb_hpr_smx checks the whole memory image).
//
//   uint32_t hps::softmax_mode_hpr(const ctr::Cmd& c, uint32_t hpr, const Word* hp1, const Word* xmem, Word* ymem)
//
// HPR on (hpr == hpr::HPR_A_HP1 and L % 64 == 0, so a row is an even number of 8-word units): word 16v + 8G + k
// (k < 8) of every score row is read by reader G, G = 0 over io and G = 1 over hp1, as one 8-beat burst per unit.
// Each reader loads its own copy of the rank tables (rank_k, the block's rank_q) over its own port. Each sends its
// half row's maximum KEY (33 bits: the visibility bit kept, so a half with no visible key cannot win); compute
// takes the max of the two (the row max of the definition: the max is exact and order-free) and runs 8 lanes a
// clock, lanes 0..3 on reader 0's word, lanes 4..7 on reader 1's (fabric multipliers by default, HPR_G1_DSP=0:
// no DSP added; the second exp table is a second BRAM36 pair). The row sum is the same exact integer. Two
// restoring dividers take alternate rows (a 64-key row takes 9 clocks, one divider 14). The writer interleaves
// the two groups' P words back into row order (word o of a row from group (o >> 2) & 1).
// HPR off: reader 0 reads every row as smx's reader does (one burst per block when contiguous), 4 lanes.
#pragma once
#include "../../softmax/softmax_core.h"
#include "hpr_contract.h"
#if defined(HS_DFSIM) && !defined(__SYNTHESIS__)
#include <thread>
#endif

#ifndef HPR_G1_DSP
#define HPR_G1_DSP 0
#endif

namespace hps {

using smx::Key;
using smx::Params;
using smx::WAddr;
using smx::Word;
using smx::XW;

constexpr int W_MAX = smx::W_MAX;  // 64 words per row
constexpr int BLK = smx::BLK;      // 256 rows per block
#ifndef HS_XS_DEPTH
#define HS_XS_DEPTH (W_MAX / 2)  // one half row per group (the compute reads both halves in lockstep)
#endif
#ifndef HS_XS0_DEPTH
#define HS_XS0_DEPTH W_MAX       // group 0 also carries the whole row when HPR is off
#endif
#ifndef HS_EQ_DEPTH
#define HS_EQ_DEPTH 16
#endif
typedef hls::stream<XW, HS_XS0_DEPTH> XS0;
typedef hls::stream<XW, HS_XS_DEPTH> XS1;
typedef hls::stream<Key, smx::MX_DEPTH> KS;
typedef hls::stream<Word, HS_EQ_DEPTH> QS;
typedef smx::SStream SS;
typedef smx::RStream RS;

struct HPS {
  Params p;
  bool on;
  ap_uint<3> nu2m1;  // HPR on: unit pairs per row - 1 (W / 16 - 1: 0..3)
};

// ------------------------------------------------------------------ reader G
// HPR off (G = 0 only): smx's reader, sending the row max as a KEY. HPR on: per row, the units of this group as
// 8-beat bursts (II = 8); the rank tables over this group's port.
template <int G, class XSG>
static void hs_reader(const Word* xm, XSG& xs, KS& mx, const HPS h) {
  const Params& p = h.p;
  if (G == 1 && !h.on) return;
  ap_uint<128> etab[W_MAX];
  ap_uint<128> dtab[BLK / 4];
#pragma HLS BIND_STORAGE variable = etab type = ram_s2p impl = lutram
#pragma HLS BIND_STORAGE variable = dtab type = ram_s2p impl = bram
  WAddr ia = p.a, ie = p.e, id = p.d;
items_r:
  for (ap_uint<17> b = 0; b < p.B; b++) {
#pragma HLS LOOP_TRIPCOUNT max = 24
    const bool load_e = p.mask && (b == 0 || p.es != 0);
    if (load_e) {
    ld_e:
      for (ap_uint<7> i = 0; i < p.W; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT max = 64
        etab[i.range(5, 0)] = xm[ie + i];
      }
    }
    WAddr ra = ia;
  blocks_r:
    for (ap_uint<17> r0 = 0; r0 < p.N; r0 += BLK) {
#pragma HLS LOOP_TRIPCOUNT max = 1
      const ap_uint<17> left = p.N - r0;
      const ap_uint<9> rows = left > BLK ? (ap_uint<9>)BLK : (ap_uint<9>)left;
      if (p.mask) {
      ld_d:
        for (ap_uint<7> i = 0; i < (ap_uint<7>)((rows + 3) >> 2); i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT max = 64
          dtab[i.range(5, 0)] = xm[id + (r0 >> 2) + i];
        }
      }
      if (!h.on) {
        // smx's KIND_X burst(s): the whole block if contiguous (A pitch = 4 L), else one per row
        const ap_uint<9> nb = p.a_contig ? (ap_uint<9>)1 : rows;
        const ap_uint<15> nw = p.a_contig ? (ap_uint<15>)(rows * p.W) : (ap_uint<15>)p.W;
        const ap_uint<28> step = p.a_contig ? (ap_uint<28>)((ap_uint<28>)p.W << 8) : p.ap;
        ap_uint<9> n = 0;
      bursts_r:
        for (ap_uint<10> j = 0; j < nb; j++) {
#pragma HLS LOOP_TRIPCOUNT max = 3
          ap_uint<7> w = 0;
          Key pm1 = 0, pm2 = 0;
        words_r:
          for (ap_uint<15> i = 0; i < nw; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT max = 16384
            const Word v = xm[ra + i];
            const ap_uint<128> ev = etab[w.range(5, 0)];
            const ap_uint<128> dw = dtab[n.range(7, 2)];
            const ap_uint<2> nl = n.range(1, 0);
            const ap_uint<32> dv = nl == 0   ? smx::lane32(dw, 0)
                                   : nl == 1 ? smx::lane32(dw, 1)
                                   : nl == 2 ? smx::lane32(dw, 2)
                                             : smx::lane32(dw, 3);
            ap_uint<4> vis;
            Key k[4];
            for (int l = 0; l < 4; l++) {
#pragma HLS UNROLL
              vis[l] = !p.mask || (!dv[31] && smx::lane32(ev, l) <= dv);
              k[l] = smx::okey(smx::lane32(v, l), vis[l]);
            }
            const Key k01 = k[1] > k[0] ? k[1] : k[0], k23 = k[3] > k[2] ? k[3] : k[2];
            const Key k4 = k23 > k01 ? k23 : k01;
            const Key prev = w < 2 ? (Key)0 : pm2;
            const Key nm = k4 > prev ? k4 : prev;
            xs.write(((XW)vis << 128) | (XW)v);
            if (w == p.W - 1) {
              const Key fm = nm > pm1 ? nm : pm1;
              mx.write(fm);
              w = 0;
              n++;
            } else {
              w++;
            }
            pm2 = pm1;
            pm1 = nm;
          }
          ra += step;
        }
      } else {
        // this group's units: words 16 v + 8 G .. + 7 of each row, one 8-beat burst each
        ap_uint<9> n = 0;
        ap_uint<3> v = 0;
        WAddr rb = ra;
        Key hm = 0;  // running max of this half row
      units_r:
        for (;;) {
#pragma HLS PIPELINE II = 8
#pragma HLS LOOP_TRIPCOUNT max = 1024
          const WAddr ua = rb + ((ap_uint<10>)v << 4) + (G ? 8 : 0);
          const ap_uint<128> dw = dtab[n.range(7, 2)];
          const ap_uint<2> nl = n.range(1, 0);
          const ap_uint<32> dv = nl == 0   ? smx::lane32(dw, 0)
                                 : nl == 1 ? smx::lane32(dw, 1)
                                 : nl == 2 ? smx::lane32(dw, 2)
                                           : smx::lane32(dw, 3);
          Key um = 0;
          for (int k = 0; k < 8; k++) {
#pragma HLS UNROLL
            const Word x = xm[ua + k];  // unconditional: one 8-beat burst, pipelined across iterations
            const ap_uint<128> ev = etab[((ap_uint<6>)v << 4) + (G ? 8 : 0) + k];
            ap_uint<4> vis;
            for (int l = 0; l < 4; l++) {
#pragma HLS UNROLL
              vis[l] = !p.mask || (!dv[31] && smx::lane32(ev, l) <= dv);
              const Key kk = smx::okey(smx::lane32(x, l), vis[l]);
              um = kk > um ? kk : um;
            }
            xs.write(((XW)vis << 128) | (XW)x);
          }
          const Key nm = um > hm ? um : hm;
          const bool row_end = v == h.nu2m1;
          if (row_end) {
            mx.write(nm);
            hm = 0;
            v = 0;
            rb += p.ap;
            if (n == rows - 1) break;
            n++;
          } else {
            hm = nm;
            v++;
          }
        }
        ra += (ap_uint<28>)rows * p.ap;  // (unused after the last block: N <= 256 rows per block only matters
                                         //  when the pitch steps blocks; rows are rb-addressed above)
      }
    }
    ia += p.as;
    ie += p.es;
    id += p.ds;
  }
}

// ------------------------------------------------------------------ the exp lanes of group 1 (fabric products)
template <int PAIR>
static inline void q_pair_g1(ap_uint<26> u0, ap_uint<26> u1, ap_uint<17> mc, ap_uint<15>& q0, ap_uint<15>& q1) {
#pragma HLS INLINE
#if HPR_G1_DSP
  smx::q_pair<PAIR + 2>(u0, u1, mc, q0, q1);
#else
  static const ap_uint<36> TAB[1024] = SMX_TAB_INIT;
#pragma HLS BIND_STORAGE variable = TAB type = rom_2p impl = bram
  ap_uint<26> u[2] = {u0, u1};
  ap_uint<15> q[2];
  for (int l = 0; l < 2; l++) {
#pragma HLS UNROLL
    ap_uint<43> t = (ap_uint<43>)u[l] * mc;
#pragma HLS BIND_OP variable = t op = mul impl = fabric
    const ap_uint<6> I = t.range(42, 37);
    const ap_uint<10> ix = t.range(36, 27);
    const ap_uint<13> flo = t.range(26, 14);
    const ap_uint<36> te = TAB[ix];
    const ap_uint<13> S = te.range(12, 0);
    const ap_uint<36> tc = (te.range(35, 13), (ap_uint<13>)0x1FFF);
    const ap_uint<26> sf = S * flo;
#pragma HLS BIND_OP variable = sf op = mul impl = fabric
    const ap_uint<36> pc = tc - sf;
    const ap_uint<23> e23 = pc.range(35, 13);
    const ap_uint<24> h = ((ap_uint<24>)e23 << 1) >> I;
    const ap_uint<24> r = h + (ap_uint<24>)257;
    q[l] = r.range(23, 9);
  }
  q0 = q[0];
  q1 = q[1];
#endif
}

// ------------------------------------------------------------------ compute: 8 lanes (4 when HPR is off)
static void hs_compute(XS0& xs0, XS1& xs1, KS& mx0, KS& mx1, QS& eq0, QS& eq1, SS& sq, const HPS h) {
  const Params& p = h.p;
  const bool on = h.on;
  const ap_uint<7> WE = on ? (ap_uint<7>)(p.W >> 1) : p.W;  // words per row and group
items_c:
  for (ap_uint<17> b = 0; b < p.B; b++) {
#pragma HLS LOOP_TRIPCOUNT max = 24
    const bool blast = b == p.B - 1;
    const bool fin_rows = blast && p.NP == p.NP4;
    ap_uint<17> n = 0;
    ap_uint<7> w = 0;
    ap_uint<49> gm2 = 1;
    bool sm = false;
    ap_uint<23> sum = 0;
    ap_uint<64> pend0 = 0, pend1 = 0;
    bool last = false;
  rows_c:
    while (!last) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT max = 8448
      const bool setup = w == 0, real = n < p.N;
      XW x0, x1 = 0;
      Key k0 = 0, k1 = 0;
      if (real) {
        if (setup) {
          k0 = mx0.read();
          if (on) k1 = mx1.read();
        } else {
          x0 = xs0.read();
          if (on) x1 = xs1.read();
        }
      }
      const ap_uint<32> mb = smx::key_bits(k1 > k0 ? k1 : k0);
      const ap_uint<32> b0 = setup ? mb : smx::lane32(x0, 0);
      ap_uint<48> ng[8];
      ng[0] = smx::neg_g(b0, p.kc);
      for (int l = 1; l < 4; l++) {
#pragma HLS UNROLL
        ng[l] = smx::neg_g(smx::lane32(x0, l), p.kc);
      }
      for (int l = 0; l < 4; l++) {
#pragma HLS UNROLL
        ng[4 + l] = smx::neg_g(smx::lane32(x1, l), p.kc);
      }
      ap_uint<26> u[8];
      for (int l = 0; l < 8; l++) {
#pragma HLS UNROLL
        const XW& xx = l < 4 ? x0 : x1;
        const int ll = l & 3;
        const bool sx = xx[32 * ll + 31];
        const ap_uint<1> cin = (sm || !sx) ? 1 : 0;
        const ap_uint<49> d2 = (ng[l], cin) - gm2;
        const ap_uint<48> d = d2.range(48, 1);
        const bool keep = real && (l < 4 || on) && xx[128 + ll] && d.range(47, 26) == 0;
        u[l] = keep ? (ap_uint<26>)d.range(25, 0) : (ap_uint<26>)0x3FFFFFF;
      }
      ap_uint<15> q[8];
      smx::q_pair<0>(u[0], u[1], p.mc, q[0], q[1]);
      smx::q_pair<1>(u[2], u[3], p.mc, q[2], q[3]);
      q_pair_g1<0>(u[4], u[5], p.mc, q[4], q[5]);
      q_pair_g1<1>(u[6], u[7], p.mc, q[6], q[7]);
      ap_uint<64> qa = 0, qb = 0;
      ap_uint<18> s8 = 0;
      for (int l = 0; l < 4; l++) {
#pragma HLS UNROLL
        qa.range(16 * l + 15, 16 * l) = q[l];
        qb.range(16 * l + 15, 16 * l) = q[4 + l];
        s8 += q[l];
        s8 += q[4 + l];
      }
      const ap_uint<23> ns = sum + s8;
      const bool row_end = w == WE, lrow = n == p.NP - 1;
      if (setup) {
        gm2 = (ng[0], (ap_uint<1>)1);
        sm = mb[31];
        sum = 0;
      } else {
        sum = ns;
        if (w[0]) {
          pend0 = qa;
          pend1 = qb;
        } else {
          eq0.write((qa, pend0));
          if (on) eq1.write((qb, pend1));
        }
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
    for (ap_uint<17> r = p.NP; r < p.NP4; r++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT max = 3
      sq.write(((ap_uint<1>)(blast && r == p.NP4 - 1), (ap_uint<23>)0));
    }
  }
}

// ------------------------------------------------------------------ divider: two of smx's, alternate rows
// Engine e takes sums 2i + e in order; both step every clock (an idle engine's state is never used); results leave
// in row order, 4 per scale word, as smx's divider writes them. sq is read without blocking, so a missing sum never
// stops the other engine; rq is written blocking (as smx's: the writer drains a block's scales after its rows).
static void hs_divider(SS& sq, RS& rq) {
  smx::Div dv[2];
#pragma HLS ARRAY_PARTITION variable = dv complete
  ap_uint<4> k[2] = {0, 0};
  bool busy[2] = {false, false}, fe[2] = {false, false};
#pragma HLS ARRAY_PARTITION variable = k complete
#pragma HLS ARRAY_PARTITION variable = busy complete
#pragma HLS ARRAY_PARTITION variable = fe complete
  for (int e = 0; e < 2; e++) {
#pragma HLS UNROLL
    dv[e].rem = 0;
    dv[e].d = 0;
    dv[e].d3 = 0;
    dv[e].q = 0;
    dv[e].pe = 0;
    dv[e].zero = false;
  }
  ap_uint<1> ein = 0, eout = 0;
  ap_uint<2> rl = 0;
  ap_uint<128> pk = 0;
  bool got_end = false;
divide:
  for (;;) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT max = 86016
    bool done = false, sent = false;
    if (busy[eout] && k[eout] == smx::DSTEPS) {
      const ap_uint<128> npk = (smx::div_result(dv[eout]), (ap_uint<96>)pk.range(127, 32));
      pk = npk;
      if (rl == 3) rq.write(npk);
      rl++;
      sent = true;
      done = fe[eout];
    }
    ap_uint<24> sv = 0;
    bool ld = false;
    if (!got_end && !busy[ein]) ld = sq.read_nb(sv);
    for (int e = 0; e < 2; e++) {
#pragma HLS UNROLL
      if (ld && ein == e) {
        smx::div_load(dv[e], sv.range(22, 0));
        fe[e] = sv[23];
        k[e] = 0;  // steps done: DSTEPS steps on the next DSTEPS clocks, then the result
        busy[e] = true;
      } else {
        if (busy[e] && k[e] != smx::DSTEPS) {
          smx::div_step(dv[e]);
          k[e] = k[e] + 1;
        }
        if (sent && eout == e) busy[e] = false;
      }
    }
    if (sent) eout = ~eout;
    if (ld) {
      ein = ~ein;
      got_end = sv[23];
    }
    if (done) break;
  }
}

// ------------------------------------------------------------------ writer
static void hs_writer(QS& eq0, QS& eq1, RS& rq, Word* ym, const HPS h) {
  const Params& p = h.p;
  const bool on = h.on;
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
      const ap_uint<7> nsw = (rows + 3) >> 2;
      const ap_uint<9> nb = p.c_contig ? (ap_uint<9>)1 : rows;
      const ap_uint<14> nw = p.c_contig ? (ap_uint<14>)(rows * p.W2) : (ap_uint<14>)p.W2;
      const ap_uint<28> step = p.c_contig ? (ap_uint<28>)((ap_uint<28>)p.W2 << 8) : p.cp;
    bursts_w:
      for (ap_uint<9> j = 0; j <= nb; j++) {
#pragma HLS LOOP_TRIPCOUNT max = 2
        const bool sc = j == nb;
        const WAddr base = sc ? rf : rc;
        const ap_uint<14> len = sc ? (ap_uint<14>)nsw : nw;
        ap_uint<6> o = 0;  // word of the row (P rows: W2 <= 32 words)
      words_w:
        for (ap_uint<14> i = 0; i < len; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_FLATTEN off
#pragma HLS LOOP_TRIPCOUNT max = 8192
          const bool g = on && o[2];
          ym[base + i] = sc ? rq.read() : g ? eq1.read() : eq0.read();
          o = (o == p.W2 - 1) ? (ap_uint<6>)0 : (ap_uint<6>)(o + 1);
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
static void softmax_df_hpr(const Word* hp1, const Word* xm, Word* ym, const HPS h) {
#pragma HLS DATAFLOW
#pragma HLS STABLE variable = h
  XS0 xs0("hs_xs0");
  XS1 xs1("hs_xs1");
  KS mx0("hs_mx0"), mx1("hs_mx1");
  QS eq0("hs_eq0"), eq1("hs_eq1");
  SS sq("hs_sq");
  RS rq("hs_rq");
#pragma HLS STREAM variable = xs0 depth = HS_XS0_DEPTH
#pragma HLS STREAM variable = xs1 depth = HS_XS_DEPTH
#pragma HLS BIND_STORAGE variable = xs0 type = fifo impl = lutram
#pragma HLS BIND_STORAGE variable = xs1 type = fifo impl = lutram
#pragma HLS STREAM variable = mx0 depth = 4
#pragma HLS STREAM variable = mx1 depth = 4
#pragma HLS STREAM variable = eq0 depth = HS_EQ_DEPTH
#pragma HLS STREAM variable = eq1 depth = HS_EQ_DEPTH
#pragma HLS BIND_STORAGE variable = eq0 type = fifo impl = srl
#pragma HLS BIND_STORAGE variable = eq1 type = fifo impl = srl
#pragma HLS STREAM variable = sq depth = 4
#pragma HLS STREAM variable = rq depth = 64
#pragma HLS BIND_STORAGE variable = rq type = fifo impl = lutram
#if defined(HS_DFSIM) && !defined(__SYNTHESIS__)
  std::thread t0([&] { hs_reader<0>(xm, xs0, mx0, h); });
  std::thread t1([&] { hs_reader<1>(hp1, xs1, mx1, h); });
  std::thread t2([&] { hs_compute(xs0, xs1, mx0, mx1, eq0, eq1, sq, h); });
  std::thread t3([&] { hs_divider(sq, rq); });
  std::thread t4([&] { hs_writer(eq0, eq1, rq, ym, h); });
  t0.join();
  t1.join();
  t2.join();
  t3.join();
  t4.join();
#else
  hs_reader<0>(xm, xs0, mx0, h);
  hs_reader<1>(hp1, xs1, mx1, h);
  hs_compute(xs0, xs1, mx0, mx1, eq0, eq1, sq, h);
  hs_divider(sq, rq);
  hs_writer(eq0, eq1, rq, ym, h);
#endif
}

// The softmax unit's mode function with hp_reads (replaces smx::softmax_mode at its one call site). hpr = record
// field 56, already checked by hpr::validate_hpr. Returns ME_OK.
static uint32_t softmax_mode_hpr(const ctr::Cmd& c, uint32_t hpr, const Word* hp1, const Word* xmem, Word* ymem) {
#pragma HLS INLINE off
  HPS h;
  smx::make_params(c, h.p);
  h.on = hpr::hpr_active_smx(c, hpr);
  h.nu2m1 = (c.K >> 6) - 1;
  softmax_df_hpr(hp1, xmem, ymem, h);
  return ctr::ME_OK;
}

}  // namespace hps
