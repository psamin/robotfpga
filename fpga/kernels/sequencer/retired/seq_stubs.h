// Stub mode functions for the standalone sequencer top (seq_top.cpp). The real modes are other
// kernels' work; these exist so the FSM can be synthesised with realistic port traffic and so the
// csynth report can show whether a mode called from the dispatch switch is instantiated once.
//
//  - gemm_stub: a real (small) int16 engine: 32 lanes x GTP tokens of int16 x int16 -> int48 MACs
//    (GTP * 32 DSPs), weights over the four hp ports in the engine's layout (port p, group g, k: one
//    128-bit word = 8 int16 lanes), X int16 rows over io, Y = acc * s_row * s_tok * p0 (+ bias)
//    (+ D) (+= C), row-major FP32. Its arithmetic is checked in tb_seq.cpp.
//  - stream_stub<KIND>: placeholder for every other mode: B items x N rows x row-words of A (and B
//    for binary kinds) through a trivial integer transform into C. Not the real arithmetic.
// All address generation uses running pointers (adds, no runtime multiplies).
#pragma once
#include "seq_core.h"

namespace seq {

#ifndef __SYNTHESIS__
extern bool g_dry;  // host only: modes return at once (decode / trace tests on real lists)
#endif

constexpr int GLN = 32;          // lanes (output rows) per group: 4 ports x 8 int16
constexpr int GTP = 2;           // tokens per cycle -> GTP * GLN MACs
constexpr int GNT = 8;           // tokens per X tile
constexpr int GNB = GNT / GTP;   // token blocks per tile (accumulator revisit distance)
constexpr int GKMAX = 1024;      // longest reduction of the stub

#ifdef __VITIS_HLS__
typedef ap_int<48> acc_t;
typedef ap_uint<128> WPortS;
static inline int16_t lane16(const Word& w, int l) {
#pragma HLS INLINE
  return (int16_t)(uint16_t)w.range(16 * l + 15, 16 * l);
}
#else
typedef int64_t acc_t;
static inline int16_t lane16(const Word& w, int l) {
  int16_t v;
  memcpy(&v, (const char*)&w + 2 * l, 2);
  return v;
}
#endif

static inline float f32_of(uint32_t u) {
#pragma HLS INLINE
  float f;
  memcpy(&f, &u, 4);
  return f;
}
static inline uint32_t u32_of(float f) {
#pragma HLS INLINE
  uint32_t u;
  memcpy(&u, &f, 4);
  return u;
}

template <class WP, class W>
static uint32_t gemm_stub(const Cmd& c, const WP* w0, const WP* w1, const WP* w2, const WP* w3, const W* xmem,
                          W* ymem) {
#pragma HLS INLINE off
#ifndef __SYNTHESIS__
  if (g_dry) return ME_OK;
#endif
  if (c.M == 0 || c.M % GLN || c.K == 0 || c.K % 8 || c.K > (uint32_t)GKMAX || c.N == 0) return ME_DIMS;
  if ((c.a.addr | c.b.addr | c.c.addr | c.e.addr | c.f.addr | c.a.pitch | c.c.pitch | c.b.pitch) & 15) return ME_ALIGN;
  static int16_t xb[GTP][GNB][GKMAX];
#pragma HLS ARRAY_PARTITION variable = xb complete dim = 1
#pragma HLS ARRAY_PARTITION variable = xb cyclic factor = 8 dim = 3
  float xs[GNT];
#pragma HLS ARRAY_PARTITION variable = xs complete
  acc_t acc[GTP][GLN][GNB];
#pragma HLS ARRAY_PARTITION variable = acc complete dim = 1
#pragma HLS ARRAY_PARTITION variable = acc complete dim = 2
  float sr[GLN], bs[GLN];
#pragma HLS ARRAY_PARTITION variable = sr complete
#pragma HLS ARRAY_PARTITION variable = bs complete
  const uint32_t groups = c.M / GLN, k8 = c.K / 8;
  const float p0 = f32_of(c.p0);
  const bool bias = c.flags & F_BIAS, accf = c.flags & F_ACC, addd = c.flags & F_ADD_D;
  addr_t pa = c.a.addr, pf = c.f.addr, pc = c.c.addr, pd = c.d.addr, pw = c.b.addr, pe = c.e.addr;
  uint32_t gi = 0;  // items since the weight pointer last moved (GQA: B operand advances every G items)
items:
  for (uint32_t b = 0; b < c.B; b++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 16
    addr_t ra = pa, rf = pf, rc = pc, rd = pd;
  tiles:
    for (uint32_t t0 = 0; t0 < c.N; t0 += GNT) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 64
      const uint32_t nt = (c.N - t0) < (uint32_t)GNT ? (c.N - t0) : (uint32_t)GNT;
    load_x:
      for (int r = 0; r < GNT; r++) {
        for (uint32_t i = 0; i < k8; i++) {
#pragma HLS LOOP_TRIPCOUNT min = 8 max = 128
#pragma HLS PIPELINE II = 1
          Word wx = zword();
          if ((uint32_t)r < nt) wx = to_word(xmem[(ra >> 4) + i]);
          for (int j = 0; j < 8; j++) {
#pragma HLS UNROLL
            xb[r % GTP][r / GTP][i * 8 + j] = lane16(wx, j);
          }
        }
        if ((uint32_t)r < nt) {
          const Word ws = to_word(xmem[(rf >> 4)]);
          xs[r] = f32_of(wget(ws, (int)((rf >> 2) & 3)));
          ra += c.a.pitch;
          rf += 4;
        } else {
          xs[r] = 0.0f;
        }
      }
      addr_t gw = pw, ge = pe;
    groups_loop:
      for (uint32_t g = 0; g < groups; g++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 96
      sc_load:
        for (int q = 0; q < GLN / 4; q++) {
#pragma HLS PIPELINE II = 1
          const Word s4 = to_word(w0[(ge >> 4) + q]);
          Word b4 = zword();
          if (bias) b4 = to_word(w0[((ge + c.M * 4) >> 4) + q]);
          for (int j = 0; j < 4; j++) {
#pragma HLS UNROLL
            sr[q * 4 + j] = f32_of(wget(s4, j));
            bs[q * 4 + j] = bias ? f32_of(wget(b4, j)) : 0.0f;
          }
        }
        Word wd[4];
#pragma HLS ARRAY_PARTITION variable = wd complete
        const addr_t p0a = gw >> 4, p1a = (gw + c.b.pitch) >> 4, p2a = (gw + 2 * c.b.pitch) >> 4,
                     p3a = (gw + 3 * c.b.pitch) >> 4;
        uint32_t k = 0, bb = 0;
      mac:
        for (uint32_t it = 0; it < c.K * GNB; it++) {
#pragma HLS LOOP_TRIPCOUNT min = 32 max = 4096
#pragma HLS PIPELINE II = 1
#pragma HLS DEPENDENCE variable = acc inter distance = GNB true
          if (bb == 0) {
            wd[0] = to_word(w0[p0a + k]);
            wd[1] = to_word(w1[p1a + k]);
            wd[2] = to_word(w2[p2a + k]);
            wd[3] = to_word(w3[p3a + k]);
          }
          for (int a = 0; a < GTP; a++) {
#pragma HLS UNROLL
            const int16_t x = xb[a][bb][k];
            for (int l = 0; l < GLN; l++) {
#pragma HLS UNROLL
              const int16_t w = lane16(wd[l / 8], l % 8);
              const acc_t prev = (k == 0) ? (acc_t)0 : acc[a][l][bb];
              acc[a][l][bb] = prev + (acc_t)((int32_t)x * (int32_t)w);
            }
          }
          if (bb == GNB - 1) {
            bb = 0;
            k++;
          } else {
            bb++;
          }
        }
        addr_t yc = rc + g * (GLN * 4), yd = rd + g * (GLN * 4);
      emit:
        for (uint32_t r = 0; r < nt; r++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 8
          for (int q = 0; q < GLN / 4; q++) {
#pragma HLS PIPELINE II = 1
            Word old = zword(), dd = zword(), o = zword();
            if (accf) old = to_word(xmem[(yc >> 4) + q]);
            if (addd) dd = to_word(xmem[(yd >> 4) + q]);
            for (int j = 0; j < 4; j++) {
#pragma HLS UNROLL
              const int l = q * 4 + j;
              float y = (float)(int64_t)acc[r % GTP][l][r / GTP] * sr[l] * xs[r] * p0 + bs[l];
              if (addd) y += f32_of(wget(dd, j));  // order: bias, then D, then C (exec_ref.op_gemm)
              if (accf) y += f32_of(wget(old, j));
              wset(o, j, u32_of(y));
            }
#ifdef __VITIS_HLS__
            ymem[(yc >> 4) + q] = o;
#else
            ymem[(yc >> 4) + q] = from_word<W>(o);
#endif
          }
          yc += c.c.pitch;
          yd += c.d.pitch;
        }
        gw += (addr_t)c.K * 16;
        ge += GLN * 4;
      }
      for (int r = 0; r < GNT; r++) {  // advance the output rows of this tile
#pragma HLS UNROLL
        if ((uint32_t)r < nt) {
          rc += c.c.pitch;
          rd += c.d.pitch;
        }
      }
    }
    pa += c.a.stride;
    pf += c.f.stride;
    pc += c.c.stride;
    pd += c.d.stride;
    if (++gi == c.G) {
      gi = 0;
      pw += c.b.stride;
      pe += c.e.stride;
    }
  }
  return ME_OK;
}

// Placeholder for the non-GEMM modes (NOT their arithmetic): B items x N rows of ceil(4 K / 16) words.
template <int KIND, class W>
static uint32_t stream_stub(const Cmd& c, const W* xmem, W* ymem) {
#pragma HLS INLINE off
#ifndef __SYNTHESIS__
  if (g_dry) return ME_OK;
#endif
  if (c.N == 0 || c.K == 0) return ME_DIMS;
  if ((c.a.addr | c.c.addr | c.b.addr) & 15) return ME_ALIGN;
  const uint32_t words = (c.K * 4 + 15) / 16;
  const bool binary = (KIND == 1 && (c.flags & F_RESID)) || (KIND == 4 && !(c.flags & F_NOMUL)) ||
                      (KIND == 6 && c.op == OP_AXPY);
  addr_t pa = c.a.addr, pb = c.b.addr, pc = c.c.addr;
s_items:
  for (uint32_t b = 0; b < c.B; b++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 16
    addr_t ra = pa, rb = pb, rc = pc;
  s_rows:
    for (uint32_t n = 0; n < c.N; n++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 512
    s_words:
      for (uint32_t i = 0; i < words; i++) {
#pragma HLS LOOP_TRIPCOUNT min = 4 max = 768
#pragma HLS PIPELINE II = 1
        const Word x = to_word(xmem[(ra >> 4) + i]);
        Word y = zword(), o = zword();
        if (binary) y = to_word(xmem[(rb >> 4) + i]);
        for (int j = 0; j < 4; j++) {
#pragma HLS UNROLL
          wset(o, j, (wget(x, j) + wget(y, j)) ^ (0x01010101u * (KIND + 1)));
        }
#ifdef __VITIS_HLS__
        ymem[(rc >> 4) + i] = o;
#else
        ymem[(rc >> 4) + i] = from_word<W>(o);
#endif
      }
      ra += c.a.pitch;
      rb += c.b.pitch;
      rc += c.c.pitch;
    }
    pa += c.a.stride;
    pb += c.b.stride;
    pc += c.c.stride;
  }
  return ME_OK;
}

}  // namespace seq
