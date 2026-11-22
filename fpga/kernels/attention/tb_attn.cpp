// Host testbench of the attention path under the float_top integration contract (fpga/kernels/CONTRACT.md 5): the
// clause test of C5. Attention is no unit of its own: every record goes to the unit the contract names for it, through
// one dispatch switch shaped like the sequencer's, and the units are their HLS mode code (the same source csynth
// builds), not models:
//
//   QUANT_X / QUANT_W / QUANT_WT   qz::row_mode      (fpga/kernels/quantize/quant_mode.h, the row unit)
//   SOFTMAX                        smx::softmax_mode (fpga/kernels/softmax/softmax_core.h, the softmax unit)
//   GEMM                           contract model (C slot rows, items, row groups R / P1, bias / D / F_ACC on
//                                  ctr::ref::gemm_out), or with -DATTN_REAL_ENGINE the engine's own
//                                  fstream16::linear_int16_core per item (the blob read as is; X / Y converted from /
//                                  to the contract rows, because the engine does not take the contract layouts yet:
//                                  CONTRACT.md 8, E-2 / E-3 / E-5)
//
//   tb_attn --unit           QUANT_X / QUANT_W / QUANT_WT on adversarial commands (zero, -0, denormal, 1e-32,
//                            3e38, ties, outliers, shifts -8..8, items, pitch and segment gaps, bases up to 0xFF_xxxx_xxxx)
//                            against an independent reference written from the contract formulas (ref_* below); the
//                            whole memory image is compared, guard zones included, so a write outside the declared
//                            extents fails; non-finite inputs must finish and stay inside the extents
//   tb_attn <case dir>...    a contract command list from make_vectors.py (real -12392 attention calls of all four
//                            sites + an adversarial site): every record is decoded, validated (validate and
//                            validate_hw) and dispatched; every intermediate is compared bit for bit with
//                            make_vectors.py's expectation (contract.py arithmetic and smx_emu, asserted equal there to
//                            the torch emulation the full-policy check runs); a row op's whole image is also compared
//                            with the reference, and a GEMM / SOFTMAX may change no word outside its declared outputs
//
// Build variants:
//   -DATTN_MODELS                row ops by the reference models, SOFTMAX bits loaded from smx_emu (no unit code)
//   -DATTN_REAL_ENGINE -DFSTREAM_ARITH=16   GEMM by the engine core (fpga/hls/int16_linear.h)
//   -DQZ_DFSIM -DSMX_DFSIM -Idfsim       both units' DATAFLOW processes as threads on FIFOs bounded at the pragma depths
//                                (deadlock model, CONTRACT.md 7; dfsim/hls_stream.h); a stuck FIFO exits with code 7
//
//   export PATH=/c/Xilinx/Vitis_HLS/2022.2/tps/win64/msys64/mingw64/bin:/c/Xilinx/Vitis_HLS/2022.2/tps/win64/msys64/usr/bin:/c/Windows/System32:/c/Windows
//   sh run_tests.sh              (all four builds, logs in results/)
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "../quantize/quant_mode.h"
#include "../softmax/softmax_core.h"
#ifdef ATTN_REAL_ENGINE
#include "../../hls/int16_linear.h"
#endif

#if defined(QZ_DFSIM)
int qz_dfsim_depth(const std::string& n) {
  // stream "qz_<x>" has depth QZ_D_<X> (quant_mode.h names them so); every macro the row unit has defined so far
  static const std::map<std::string, int> d = {
#define QZ_ENTRY(n, m) {n, m},
#ifdef QZ_D_XS
      QZ_ENTRY("qz_xs", QZ_D_XS)
#endif
#ifdef QZ_D_AM
      QZ_ENTRY("qz_am", QZ_D_AM)
#endif
#ifdef QZ_D_D
      QZ_ENTRY("qz_d", QZ_D_D)
#endif
#ifdef QZ_D_SS
      QZ_ENTRY("qz_ss", QZ_D_SS)
#endif
#ifdef QZ_D_SF
      QZ_ENTRY("qz_sf", QZ_D_SF)
#endif
#ifdef QZ_D_OW
      QZ_ENTRY("qz_ow", QZ_D_OW)
#endif
#ifdef QZ_D_SQ
      QZ_ENTRY("qz_sq", QZ_D_SQ)
#endif
#ifdef QZ_D_SH
      QZ_ENTRY("qz_sh", QZ_D_SH)
#endif
#ifdef QZ_D_SD
      QZ_ENTRY("qz_sd", QZ_D_SD)
#endif
#ifdef QZ_D_TH
      QZ_ENTRY("qz_th", QZ_D_TH)
#endif
#ifdef QZ_D_TD
      QZ_ENTRY("qz_td", QZ_D_TD)
#endif
#ifdef QZ_D_HD
      QZ_ENTRY("qz_hd", QZ_D_HD)
#endif
#ifdef QZ_D_DD
      QZ_ENTRY("qz_dd", QZ_D_DD)
#endif
#undef QZ_ENTRY
  };
  auto it = d.find(n);
  if (it != d.end()) return it->second;
  if (n.compare(0, 3, "qz_") == 0) {
    printf("dfsim: row-unit stream %s has no depth here (tb_attn's table is stale)\n", n.c_str());
    exit(8);
  }
  return 0;  // not a row-unit stream: the template depth (softmax) or the HLS default
}
#endif

typedef ap_uint<128> Word;
using ctr::Cmd;
using ctr::Slot;

// ---------------------------------------------------------------- DDR model: [base, base + bytes) as 128-bit words
struct Mem {
  static constexpr size_t GUARD = 1 << 16;  // words (1 MB) of canary below and above the image, compared too
  uint64_t base = 0;
  std::vector<Word> w;                      // [guard][image][guard]
  std::vector<uint8_t>* touched = nullptr;  // if set: every word a set_* writes
  void init(uint64_t b, size_t bytes, uint32_t fill) {
    base = b;
    w.assign((bytes + 15) / 16 + 2 * GUARD, Word(0));
    for (auto& x : w)
      for (int j = 0; j < 4; j++) x.range(32 * j + 31, 32 * j) = fill;
  }
  size_t idx(uint64_t a) const {
    if (a < base || a - base >= 16 * (uint64_t)(w.size() - 2 * GUARD)) {
      fprintf(stderr, "address 0x%llx outside the image [0x%llx, +0x%llx)\n", (unsigned long long)a,
              (unsigned long long)base, (unsigned long long)(16 * (w.size() - 2 * GUARD)));
      exit(4);
    }
    return GUARD + (size_t)((a - base) >> 4);
  }
  uint32_t u32(uint64_t a) const { return (uint32_t)w[idx(a)].range(32 * ((a >> 2) & 3) + 31, 32 * ((a >> 2) & 3)); }
  void mark(uint64_t a) {
    if (touched) (*touched)[idx(a)] = 1;
  }
  void set_u32(uint64_t a, uint32_t v) {
    mark(a);
    w[idx(a)].range(32 * ((a >> 2) & 3) + 31, 32 * ((a >> 2) & 3)) = v;
  }
  uint16_t u16(uint64_t a) const { return (uint16_t)w[idx(a)].range(16 * ((a >> 1) & 7) + 15, 16 * ((a >> 1) & 7)); }
  void set_u16(uint64_t a, uint16_t v) {
    mark(a);
    w[idx(a)].range(16 * ((a >> 1) & 7) + 15, 16 * ((a >> 1) & 7)) = v;
  }
  float f32(uint64_t a) const { return ctr::u2f(u32(a)); }
  void set_f32(uint64_t a, float f) { set_u32(a, ctr::f2u(f)); }
  // the io pointers of the mode functions: register value 0, so word index = byte address >> 4
  Word* ptr() { return w.data() + GUARD - (base >> 4); }
};

// ---------------------------------------------------------------- reference row ops (contract text, CONTRACT.md 1.1)
// QUANT_W (KW) / QUANT_WT (VW)
static void ref_quant_kv(const Cmd& c, Mem& m) {
  const Slot &A = c.s[ctr::SLOT_A], &C = c.s[ctr::SLOT_C], &F = c.s[ctr::SLOT_F];
  const uint32_t W = c.p[ctr::P_SEG] ? ctr::SEG_HEAD : c.K, nseg = c.K / W;
  const uint32_t NP = c.p[ctr::P_NP] ? c.p[ctr::P_NP] : c.N;
  const uint64_t segc = c.p[ctr::P_SEGC], segf = c.p[ctr::P_SEGF];
  const int shift = (int32_t)c.p[ctr::P_SHIFT];
  std::vector<float> x;
  std::vector<int16_t> q;
  for (uint64_t b = 0; b < c.B; b++)
    for (uint64_t j = 0; j < nseg; j++) {
      if (c.op == ctr::OP_QUANT_W) {  // KW: one scale per (key, segment); rows [N, NP) zero, scale 1.0
        x.resize(W);
        q.resize(W);
        for (uint64_t n = 0; n < NP; n++) {
          for (uint64_t d = 0; d < W; d++) x[d] = n < c.N ? m.f32(ctr::row_addr(A, b, n) + 4 * (j * W + d)) : 0.0f;
          const float s = ctr::ref::quant_row(x.data(), W, q.data(), shift);
          for (uint64_t d = 0; d < W; d++) {
            const uint64_t wa = C.addr + b * C.stride + j * segc + ((n % 32) / 8) * C.pitch + ((n / 32) * W + d) * 16;
            m.set_u16(wa + 2 * (n % 8), (uint16_t)q[d]);
          }
          m.set_f32(ctr::scale_addr(F, b, j, segf, n), s);
        }
      } else {  // VW: one scale per channel over the item's N keys; keys [N, NP) zero
        x.resize(c.N);
        q.resize(c.N);
        for (uint64_t ch = 0; ch < W; ch++) {
          for (uint64_t n = 0; n < c.N; n++) x[n] = m.f32(ctr::row_addr(A, b, n) + 4 * (j * W + ch));
          const float s = ctr::ref::quant_row(x.data(), c.N, q.data(), shift);
          for (uint64_t n = 0; n < NP; n++) {
            const uint64_t wa = C.addr + b * C.stride + j * segc + ((ch % 32) / 8) * C.pitch + ((ch / 32) * NP + n) * 16;
            m.set_u16(wa + 2 * (ch % 8), n < c.N ? (uint16_t)q[n] : (uint16_t)0);
          }
          m.set_f32(ctr::scale_addr(F, b, j, segf, ch), s);
        }
      }
    }
}

// QUANT_X (F_QOUT int16 rows, per (row, segment) scales; rows [N, NP) zero, scale 1.0; pad lanes 1.0)
static void ref_quant_x(const Cmd& c, Mem& m) {
  const Slot &A = c.s[ctr::SLOT_A], &C = c.s[ctr::SLOT_C], &F = c.s[ctr::SLOT_F];
  const uint32_t W = c.p[ctr::P_SEG] ? ctr::SEG_HEAD : c.K, nseg = c.K / W;
  const uint32_t NP = c.p[ctr::P_NP] ? c.p[ctr::P_NP] : c.N;
  const uint64_t segc = c.p[ctr::P_SEGC], segf = c.p[ctr::P_SEGF];
  std::vector<float> x(W);
  std::vector<int16_t> q(W);
  for (uint64_t b = 0; b < c.B; b++)
    for (uint64_t j = 0; j < nseg; j++)
      for (uint64_t n = 0; n < ctr::pad4(NP); n++) {
        float s = 1.0f;
        if (n < NP) {
          for (uint64_t d = 0; d < W; d++) x[d] = n < c.N ? m.f32(ctr::row_addr(A, b, n) + 4 * (j * W + d)) : 0.0f;
          s = ctr::ref::quant_row(x.data(), W, q.data(), (int32_t)c.p[ctr::P_SHIFT]);
          for (uint64_t d = 0; d < W; d++) m.set_u16(ctr::seg_addr(C, b, n, j, segc) + 2 * d, (uint16_t)q[d]);
        }
        m.set_f32(ctr::scale_addr(F, b, j, segf, n), s);
      }
}

static void ref_row(const Cmd& c, Mem& m) {
  if (c.op == ctr::OP_QUANT_X)
    ref_quant_x(c, m);
  else
    ref_quant_kv(c, m);
}

// ---------------------------------------------------------------- GEMM (CONTRACT.md 2.3, 3.3)
static void gemm_epilogue_write(const Cmd& c, Mem& m, uint64_t b, uint64_t n, uint64_t mm, float y) {
  const Slot& Y = c.s[ctr::SLOT_C];
  const uint64_t R = c.p[ctr::P_SEGC], P1 = c.p[ctr::P_SEGF];
  const uint64_t ya = ctr::row_addr2(Y, b, n, R, P1) + 4 * mm;
  float bias, add;
  const bool hb = c.flags & ctr::F_BIAS, hd = c.flags & (ctr::F_ADD_D | ctr::F_ACC);
  if (hb) bias = m.f32(c.s[ctr::SLOT_G].addr + b * c.s[ctr::SLOT_G].stride + 4 * mm);
  if (c.flags & ctr::F_ADD_D) add = m.f32(ctr::row_addr(c.s[ctr::SLOT_D], b, n) + 4 * mm);
  if (c.flags & ctr::F_ACC) add = m.f32(ya);
  if (hb) y = ctr::ref::fadd_hw(y, bias);
  if (hd) y = ctr::ref::fadd_hw(y, add);
  m.set_f32(ya, y);
}

#ifndef ATTN_REAL_ENGINE
static void gemm(const Cmd& c, Mem& m) {
  const Slot &X = c.s[ctr::SLOT_A], &Wt = c.s[ctr::SLOT_B], &E = c.s[ctr::SLOT_E], &Fs = c.s[ctr::SLOT_F];
  std::vector<int16_t> xr(c.K), wr((size_t)c.M * c.K);
  for (uint64_t b = 0; b < c.B; b++) {
    for (uint64_t mm = 0; mm < c.M; mm++)
      for (uint64_t k = 0; k < c.K; k++)
        wr[mm * c.K + k] = (int16_t)m.u16(ctr::w_word_addr(Wt, b, mm, k, c.K) + 2 * ctr::w_lane(mm));
    for (uint64_t n = 0; n < c.N; n++) {
      for (uint64_t k = 0; k < c.K; k++) xr[k] = (int16_t)m.u16(ctr::row_addr(X, b, n) + 2 * k);
      const float st = m.f32(Fs.addr + b * Fs.stride + 4 * n);
      for (uint64_t mm = 0; mm < c.M; mm++) {
        int64_t acc = 0;
        for (uint64_t k = 0; k < c.K; k++) acc += (int32_t)xr[k] * (int32_t)wr[mm * c.K + k];
        const float y = ctr::ref::gemm_out(acc, m.f32(E.addr + b * E.stride + 4 * mm), st, nullptr, nullptr);
        gemm_epilogue_write(c, m, b, n, mm, y);
      }
    }
  }
}
static const char* gemm_name = "contract model";
#else
// The engine's own core per item, on its per-call layout: xmem = [s_row M][s_tok Np][X: Np/32][K][32 tokens],
// port p = [M/32][K] words read from the contract blob as is, Y[M/32][Np][32] back to the contract rows.
static void gemm(const Cmd& c, Mem& m) {
  using fstream::IoWord;
  using fstream::WPort;
  const Slot &X = c.s[ctr::SLOT_A], &Wt = c.s[ctr::SLOT_B], &E = c.s[ctr::SLOT_E], &Fs = c.s[ctr::SLOT_F];
  const uint32_t M = c.M, K = c.K, N = c.N, np = ctr::pad32(N), ws = M / 32 * K;
  for (uint64_t b = 0; b < c.B; b++) {
    std::vector<WPort> w[4];
    for (int p = 0; p < 4; p++) {
      w[p].resize(ws);
      for (uint32_t i = 0; i < ws; i++)
        for (int l = 0; l < 8; l++) w[p][i].v[l] = m.u16(Wt.addr + b * Wt.stride + p * (uint64_t)Wt.pitch + 16 * i + 2 * l);
    }
    const size_t nx = M / 4 + np / 4 + (size_t)np * K / 8, xo = M / 4 + np / 4;
    std::vector<uint32_t> xu(4 * nx, 0u);
    for (uint32_t mm = 0; mm < M; mm++) xu[mm] = m.u32(E.addr + b * E.stride + 4 * mm);
    for (uint32_t n = 0; n < np; n++) xu[M + n] = n < N ? m.u32(Fs.addr + b * Fs.stride + 4 * n) : ctr::ONE_BITS;
    std::vector<uint16_t> xh(8 * (nx - xo), 0);
    for (uint32_t n = 0; n < N; n++)
      for (uint32_t k = 0; k < K; k++) {
        const uint32_t blk = n / 32, a = n % 32;
        xh[(((size_t)blk * K + k) * 4 + a / 8) * 8 + a % 8] = m.u16(ctr::row_addr(X, b, n) + 2 * k);
      }
    std::vector<IoWord> xm(nx), ym((size_t)M * np / 4);
    memcpy(xm.data(), xu.data(), 16 * xo);
    memcpy(xm.data() + xo, xh.data(), 16 * (nx - xo));
    const int st = fstream16::linear_int16_core(w[0].data(), w[1].data(), w[2].data(), w[3].data(), xm.data(),
                                                ym.data(), (int)M, (int)K, (int)N, (int)(M / 4), (int)((M + np) / 4));
    if (st) {
      printf("  engine core returned %d\n", st);
      exit(5);
    }
    for (uint32_t n = 0; n < N; n++)
      for (uint32_t mm = 0; mm < M; mm++) {
        const size_t e = ((size_t)(mm / 32) * np + n) * 32 + mm % 32;
        gemm_epilogue_write(c, m, b, n, mm, ym[e / 4].v[e % 4]);
      }
  }
}
static const char* gemm_name = "int16_linear.h core";
#endif

// ---------------------------------------------------------------- the dispatch switch (as the sequencer's)
#ifdef ATTN_MODELS
static const char* units = "reference models (row ops), smx_emu bits (SOFTMAX)";
#else
static const char* units = "qz::row_mode (row ops), smx::softmax_mode (SOFTMAX)";
#endif
// returns the mode's ME_* code; -1: an opcode this testbench does not run
static int dispatch(const Cmd& c, Mem& m, const char*& what) {
  switch (c.op) {
    case ctr::OP_GEMM:
      gemm(c, m);
      what = "GEMM";
      return ctr::ME_OK;
    case ctr::OP_QUANT_X:
    case ctr::OP_QUANT_W:
    case ctr::OP_QUANT_WT:
      what = c.op == ctr::OP_QUANT_X ? "QUANT_X  (row unit)" : c.op == ctr::OP_QUANT_W ? "QUANT_W  (row unit)"
                                                                                      : "QUANT_WT (row unit)";
#ifdef ATTN_MODELS
      ref_row(c, m);
      return ctr::ME_OK;
#else
      return (int)qz::row_mode(c, m.ptr(), m.ptr());
#endif
    case ctr::OP_SOFTMAX:
      what = "SOFTMAX  (softmax unit)";
#ifdef ATTN_MODELS
      return -2;  // the caller loads smx_emu's bits
#else
      return (int)smx::softmax_mode(c, m.ptr(), m.ptr());
#endif
    case ctr::OP_END:
      what = "END";
      return ctr::ME_OK;
    default:
      return -1;
  }
}

// declared output words of a GEMM / SOFTMAX record (row ops are compared against the reference image instead)
static void mark_outputs(const Cmd& c, const Mem& m, std::vector<uint8_t>& t) {
  auto span = [&](uint64_t a, uint64_t bytes) {
    for (uint64_t x = a & ~15ull; x < a + bytes; x += 16) t[m.idx(x)] = 1;
  };
  const Slot& C = c.s[ctr::SLOT_C];
  if (c.op == ctr::OP_GEMM) {
    for (uint64_t b = 0; b < c.B; b++)
      for (uint64_t n = 0; n < c.N; n++) span(ctr::row_addr2(C, b, n, c.p[ctr::P_SEGC], c.p[ctr::P_SEGF]), 4ull * c.M);
  } else if (c.op == ctr::OP_SOFTMAX) {
    const uint64_t NP = c.p[ctr::P_NP] ? c.p[ctr::P_NP] : c.N;
    for (uint64_t b = 0; b < c.B; b++) {
      for (uint64_t n = 0; n < NP; n++) span(ctr::row_addr(C, b, n), 2ull * c.K);
      span(c.s[ctr::SLOT_F].addr + b * c.s[ctr::SLOT_F].stride, 4ull * ctr::pad4((uint32_t)NP));
    }
  }
}

// ---------------------------------------------------------------- unit tests of the row ops the attention uses
static float edge_value(std::mt19937& g, int pattern, uint32_t i) {
  std::uniform_real_distribution<float> u(-1.0f, 1.0f);
  switch (pattern) {
    case 0: return u(g) * 3.0f;
    case 1: return 0.0f;
    case 2: return (i % 2) ? -0.0f : 0.0f;
    case 3: return ctr::u2f(g() & 0x807FFFFFu);                // denormals (and +-0)
    case 4: return u(g) * 1e-32f;                              // below the 1e-30 floor
    case 5: return u(g) * 3.0e38f;                             // near FLT_MAX
    case 6: return (float)((int)(g() % 64) - 32) + 0.5f;       // ties
    case 7: return i == 3 ? 1000.0f : u(g) * 1e-3f;            // one outlier
    case 8: return 1.0f;                                       // all equal
    default: return ldexpf(u(g), (int)(g() % 200) - 100);      // wide exponents
  }
}

struct UCase {
  const char* name;
  uint32_t op, K, N, NP, seg, B;
  int shift;
  uint32_t apad, cpad, sgap;  // extra bytes: A pitch, C pitch (port pitch / row pitch), segment gaps
  uint64_t base;
};

// layout of a unit case: A rows, C (blob or int16 rows), F scales; returns the image end
static uint64_t unit_cmd(const UCase& u, Cmd& c) {
  const uint32_t W = u.seg ? 64 : u.K, nseg = u.K / W, NP = u.NP ? u.NP : u.N;
  const bool vw = u.op == ctr::OP_QUANT_WT, qx = u.op == ctr::OP_QUANT_X;
  const uint64_t apitch = 4ull * u.K + u.apad;
  uint64_t cpitch, segc, blob;
  if (qx) {  // int16 rows: head-major (one head block per segment) or whole rows
    cpitch = (nseg > 1 ? 2ull * W : 2ull * u.K) + u.cpad;
    segc = nseg > 1 ? (uint64_t)NP * cpitch + u.sgap : 0;
    blob = nseg > 1 ? nseg * segc : (uint64_t)NP * cpitch;
  } else {
    const uint64_t port = vw ? (uint64_t)(W / 32) * NP * 16 : (uint64_t)(NP / 32) * W * 16;
    cpitch = port + u.cpad;
    segc = nseg > 1 ? 4 * cpitch + u.sgap : 0;
    blob = nseg > 1 ? nseg * segc : 4 * cpitch;
  }
  const uint64_t nsc = vw ? W : NP;
  const uint64_t segf = nseg > 1 ? 4 * ctr::pad4((uint32_t)nsc) + u.sgap : 0;
  const uint64_t sv = nseg > 1 ? nseg * segf : 4 * ctr::pad4((uint32_t)nsc);
  const uint64_t a0 = u.base + 256, c0 = (a0 + (uint64_t)u.B * u.N * apitch + 4096) / 4096 * 4096;
  const uint64_t f0 = c0 + (uint64_t)u.B * blob + 256 + 16;
  memset(&c, 0, sizeof c);
  c.op = u.op;
  c.flags = qx ? ctr::F_QOUT : 0;
  c.K = u.K;
  c.N = u.N;
  c.B = u.B;
  c.s[ctr::SLOT_A] = {a0, (uint32_t)apitch, (uint32_t)(u.N * apitch)};
  c.s[ctr::SLOT_C] = {c0, (uint32_t)cpitch, (uint32_t)blob};
  c.s[ctr::SLOT_F] = {f0, 0, (uint32_t)sv};
  c.p[ctr::P_SEG] = u.seg;
  c.p[ctr::P_SEGC] = (uint32_t)segc;
  c.p[ctr::P_SEGF] = (uint32_t)segf;
  c.p[ctr::P_SHIFT] = (uint32_t)u.shift;
  c.p[ctr::P_NP] = u.NP;
  return f0 + (uint64_t)u.B * sv + 256;
}

static int g_gaps = 0;  // contract-legal commands the row unit declines (ME_DIMS, nothing touched)

static int run_unit_case(const UCase& u, uint32_t seed) {
  std::mt19937 g(seed);
  const bool vw = u.op == ctr::OP_QUANT_WT;
  Cmd c;
  const uint64_t end = unit_cmd(u, c);
  // through the record format: what the sequencer hands the mode
  uint32_t f[ctr::REC_FIELDS];
  ctr::encode(c, f);
  Cmd d;
  if (ctr::decode(f, d) != ctr::ST_END) {
    printf("  %-30s decode failed\n", u.name);
    return 1;
  }
  const uint32_t v = ctr::validate(d), vh = ctr::validate_hw(d);
  if (v || vh) {
    printf("  %-30s validate %u validate_hw %u (test bug)\n", u.name, v, vh);
    return 1;
  }
  Mem m1, m2;
  m1.init(u.base, end - u.base, 0xA5A5A5A5u);  // canary everywhere
  for (uint64_t b = 0; b < u.B; b++)
    for (uint64_t n = 0; n < u.N; n++) {
      const int pat = (int)((n * 7 + b * 3 + seed) % 10);
      for (uint64_t k = 0; k < u.K; k++)
        m1.set_f32(ctr::row_addr(d.s[ctr::SLOT_A], b, n) + 4 * k, edge_value(g, pat, (uint32_t)k));
    }
  if (vw) {  // VW: also columns with their own patterns (a channel of zeros, of denormals, of 1e-32 ...)
    for (uint64_t b = 0; b < u.B; b++)
      for (uint64_t k = 0; k < u.K; k += 13) {
        const int pat = (int)((k / 13 + seed) % 10);
        for (uint64_t n = 0; n < u.N; n++)
          m1.set_f32(ctr::row_addr(d.s[ctr::SLOT_A], b, n) + 4 * k, edge_value(g, pat, (uint32_t)n));
      }
  }
  m2 = m1;
  const Mem m0 = m1;
  const auto t0 = std::chrono::steady_clock::now();
  const uint32_t r = qz::row_mode(d, m1.ptr(), m1.ptr());
  const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  const char* opn = u.op == ctr::OP_QUANT_X ? "QUANT_X " : vw ? "QUANT_WT" : "QUANT_W ";
  if (r == ctr::ME_DIMS) {  // declined: must not have touched memory
    const bool same = m1.w == m0.w;
    printf("  %-30s %s K %5u N %5u: row unit returns ME_DIMS (%s); validate accepts it: CONTRACT GAP\n", u.name, opn,
           u.K, u.N, same ? "nothing touched" : "BUT MEMORY CHANGED");
    g_gaps++;
    return same ? 0 : 1;
  }
  ref_row(d, m2);
  size_t bad = 0, first = 0;
  for (size_t i = 0; i < m1.w.size(); i++)
    if (m1.w[i] != m2.w[i]) {
      if (!bad) first = i;
      bad++;
    }
  const bool outside = bad && (first < Mem::GUARD || first >= m1.w.size() - Mem::GUARD);
  const uint32_t NP = u.NP ? u.NP : u.N;
  printf("  %-30s %s K %5u N %5u NP %5u W %4u B %u shift %+d base 0x%010llx: %s (%zu words differ%s) %.0f ms\n",
         u.name, opn, u.K, u.N, NP, u.seg ? 64 : u.K, u.B, u.shift, (unsigned long long)u.base,
         bad || r ? "FAIL" : "bit-exact", bad, bad ? "" : ", none outside the extents", ms);
  if (bad)
    printf("    first difference at 0x%llx%s\n", (unsigned long long)(m1.base + 16 * first - 16 * Mem::GUARD),
           outside ? " (in a guard zone: a write past the declared extents)" : "");
  return (bad || r) ? 1 : 0;
}

// Non-finite inputs (CONTRACT.md 4.1, review defects 21 / 27): q is undefined, but the mode must finish and must not
// write outside the declared extents. The reference marks every word it writes; the mode may change no other word.
static int nonfinite_case(uint32_t op) {
  UCase u = {"non-finite rows / columns", op, 320, 45, op == ctr::OP_QUANT_X ? 48u : 64u, 64, 1, 0, 0, 0, 0,
             0x800000000ull};
  Cmd c;
  const uint64_t end = unit_cmd(u, c);
  if (ctr::validate(c)) {
    printf("  nonfinite: validate %u (test bug)\n", ctr::validate(c));
    return 1;
  }
  Mem m1, m2;
  m1.init(u.base, end - u.base, 0x5A5A5A5Au);
  std::mt19937 g(77);
  const float inf = ctr::u2f(0x7F800000u), nan = ctr::u2f(0x7FC00001u);
  for (uint64_t n = 0; n < u.N; n++)
    for (uint64_t k = 0; k < u.K; k++) {
      float v = edge_value(g, 0, (uint32_t)k);
      if (n % 7 == 1 && k % 64 == 5) v = inf;
      if (n % 7 == 3 && k % 64 == 9) v = -inf;
      if (n % 11 == 4 && k % 64 == 33) v = nan;
      if (k == 70) v = n % 2 ? nan : inf;  // a whole column (VW channel 6 of head 1)
      m1.set_f32(ctr::row_addr(c.s[ctr::SLOT_A], 0, n) + 4 * k, v);
    }
  m2 = m1;
  const Mem init = m1;
  std::vector<uint8_t> t(m2.w.size(), 0);
  m2.touched = &t;
  const uint32_t r = qz::row_mode(c, m1.ptr(), m1.ptr());
  ref_row(c, m2);
  size_t outside = 0, changed = 0;
  for (size_t i = 0; i < m1.w.size(); i++)
    if (m1.w[i] != init.w[i]) {
      changed++;
      if (!t[i]) outside++;
    }
  const char* opn = op == ctr::OP_QUANT_X ? "QUANT_X " : op == ctr::OP_QUANT_WT ? "QUANT_WT" : "QUANT_W ";
  printf("  %-30s %s: returned %u, finished, %zu words written, %zu outside the declared extents: %s\n", u.name, opn,
         r, changed, outside, outside || r ? "FAIL" : "ok");
  return outside || r ? 1 : 0;
}

static int unit_tests() {
  const uint64_t HI = 0xFF00000000ull, LO = 0x40000000ull, MID = 0x800000000ull;
  const UCase cs[] = {
      // QUANT_X: Q head-major with p4 = -3 (the 1/8), o_proj input rows
      {"qx Q head-major (15 heads)", ctr::OP_QUANT_X, 960, 50, 52, 64, 1, -3, 0, 0, 0, MID},
      {"qx Q vision (12 heads, 2 items)", ctr::OP_QUANT_X, 768, 256, 256, 64, 2, -3, 64, 0, 0, HI},
      {"qx Q gaps, shift +8", ctr::OP_QUANT_X, 320, 45, 48, 64, 1, 8, 16, 32, 16, LO},
      {"qx Q shift -8, 1 row", ctr::OP_QUANT_X, 192, 1, 4, 64, 3, -8, 0, 0, 0, MID},
      {"qx o.in rows (960)", ctr::OP_QUANT_X, 960, 50, 0, 0, 1, 0, 0, 0, 0, HI},
      {"qx o.in rows pitch gap", ctr::OP_QUANT_X, 768, 33, 0, 0, 2, 0, 16, 16, 0, LO},
      // QUANT_W (KW)
      {"kw lm (5 heads, 45 keys)", ctr::OP_QUANT_W, 320, 45, 64, 64, 1, 0, 0, 0, 0, MID},
      {"kw vision (12 heads)", ctr::OP_QUANT_W, 768, 512, 512, 64, 1, 0, 0, 0, 0, HI},
      {"kw items 3, pitch gaps", ctr::OP_QUANT_W, 320, 50, 64, 64, 3, 0, 48, 32, 16, LO},
      {"kw shift -8", ctr::OP_QUANT_W, 128, 33, 64, 64, 2, -8, 16, 0, 0, MID},
      {"kw shift +8", ctr::OP_QUANT_W, 64, 1, 32, 64, 1, 8, 0, 0, 0, MID},
      {"kw shift -3", ctr::OP_QUANT_W, 192, 31, 32, 64, 1, -3, 0, 16, 0, HI},
      {"kw 300 keys, 17 heads", ctr::OP_QUANT_W, 1088, 300, 320, 64, 1, 0, 0, 0, 32, HI},
      {"kw W=K=64 (p1 0)", ctr::OP_QUANT_W, 64, 40, 64, 0, 2, 0, 0, 0, 0, LO},
      {"kw W=K=200 (p1 0)", ctr::OP_QUANT_W, 200, 70, 96, 0, 1, 1, 32, 0, 0, MID},
      // QUANT_WT (VW)
      {"vw lm (5 heads, 45 keys)", ctr::OP_QUANT_WT, 320, 45, 64, 64, 1, 0, 0, 0, 0, MID},
      {"vw vision (2 items)", ctr::OP_QUANT_WT, 768, 256, 256, 64, 2, 0, 0, 0, 0, HI},
      {"vw suffix NP 56", ctr::OP_QUANT_WT, 320, 50, 56, 64, 1, 0, 16, 48, 16, LO},
      {"vw 1 key", ctr::OP_QUANT_WT, 64, 1, 8, 64, 3, 2, 0, 0, 0, MID},
      {"vw shift -8, 3 heads", ctr::OP_QUANT_WT, 192, 77, 80, 64, 1, -8, 0, 0, 0, MID},
      {"vw W=K=96 (p1 0)", ctr::OP_QUANT_WT, 96, 40, 40, 0, 2, 0, 16, 0, 0, LO},
      {"vw W=K=320 (p1 0)", ctr::OP_QUANT_WT, 320, 45, 64, 0, 1, 0, 0, 0, 0, HI},
      {"vw W=K=32 (p1 0)", ctr::OP_QUANT_WT, 32, 9, 16, 0, 1, -2, 0, 0, 0, MID},
  };
  int fails = 0;
  printf("== unit: QUANT_X / QUANT_W / QUANT_WT (qz::row_mode) vs the contract reference, whole image compared\n");
  uint32_t seed = 1;
  for (const auto& u : cs) fails += run_unit_case(u, seed++);
  fails += nonfinite_case(ctr::OP_QUANT_X);
  fails += nonfinite_case(ctr::OP_QUANT_W);
  fails += nonfinite_case(ctr::OP_QUANT_WT);
  printf("  %d cases, %d failures, %d contract gaps\n", (int)(sizeof cs / sizeof cs[0]) + 3, fails, g_gaps);
  return fails;
}

// ---------------------------------------------------------------- contract list cases (make_vectors.py)
static std::vector<uint8_t> load_bytes(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) {
    fprintf(stderr, "cannot open %s\n", path.c_str());
    exit(2);
  }
  const size_t n = (size_t)f.tellg();
  std::vector<uint8_t> v(n);
  f.seekg(0);
  f.read(reinterpret_cast<char*>(v.data()), (std::streamsize)n);
  return v;
}
static void put_bytes(Mem& m, uint64_t a, const std::vector<uint8_t>& v) {
  for (size_t i = 0; i < v.size(); i += 2) m.set_u16(a + i, (uint16_t)(v[i] | (v[i + 1] << 8)));
}

static int run_case(const std::string& dir) {
  // meta.txt: base, list address, image bytes, records; script.txt: check / emu lines
  std::ifstream mf(dir + "/meta.txt");
  unsigned long long base, list;
  size_t bytes, nrec;
  mf >> std::hex >> base >> list >> std::dec >> bytes >> nrec;
  const std::vector<uint8_t> img = load_bytes(dir + "/mem.bin");
  Mem m;
  m.init(base, bytes, 0);
  put_bytes(m, base, img);
  struct Line {
    std::string kind, name, file;
    uint64_t addr, n;
  };
  std::multimap<size_t, Line> emus, checks;
  std::ifstream sc(dir + "/script.txt");
  std::string line;
  while (std::getline(sc, line)) {
    std::istringstream is(line);
    Line L;
    size_t at;
    is >> L.kind >> at;
    if (L.kind == "emu") {
      is >> std::hex >> L.addr >> std::dec >> L.file;
      emus.insert({at, L});
    } else if (L.kind == "check") {
      is >> L.name >> std::hex >> L.addr >> std::dec >> L.n >> L.file;
      checks.insert({at, L});
    }
  }
  printf("== %s (image 0x%llx + %zu bytes, %zu records at 0x%llx)\n   units: %s; GEMM: %s\n", dir.c_str(), base, bytes,
         nrec, list, units, gemm_name);
  int fails = 0, nchecks = 0;
  for (size_t i = 0; i < nrec; i++) {
    uint32_t f[ctr::REC_FIELDS];
    for (uint32_t j = 0; j < ctr::REC_FIELDS; j++) f[j] = m.u32(list + 256 * i + 4 * j);
    Cmd c;
    const uint32_t st = ctr::decode(f, c);
    if (st != ctr::ST_END) {
      printf("  record %zu: decode status 0x%x\n", i, st);
      return fails + 1;
    }
    const uint32_t v = ctr::validate(c), vh = ctr::validate_hw(c);
    if (v || vh) {
      printf("  record %zu (op 0x%02x): validate %u validate_hw %u\n", i, c.op, v, vh);
      return fails + 1;
    }
    const char* what = "";
    const bool rowop = c.op == ctr::OP_QUANT_X || c.op == ctr::OP_QUANT_W || c.op == ctr::OP_QUANT_WT;
    const Mem before = m;
    const auto t0 = std::chrono::steady_clock::now();
    const int r = dispatch(c, m, what);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (r == -1) {
      printf("  record %zu: op 0x%02x has no unit here\n", i, c.op);
      return fails + 1;
    }
    if (r == -2) {  // ATTN_MODELS: the softmax unit's bits from smx_emu
      auto lr = emus.equal_range(i);
      for (auto it = lr.first; it != lr.second; ++it)
        put_bytes(m, it->second.addr, load_bytes(dir + "/" + it->second.file));
      what = "SOFTMAX  (smx_emu bits)";
    } else if (r != (int)ctr::ME_OK) {
      printf("  [%2zu] %s returned ME %d\n", i, what, r);
      fails++;
    }
    printf("  [%2zu] %-24s M %4u K %4u N %4u B %2u %7.1f ms\n", i, what, c.M, c.K, c.N, c.B, ms);
    // extents: a row op's whole image equals the reference's; a GEMM / SOFTMAX changes only its declared outputs
    if (rowop && r == (int)ctr::ME_OK) {
      Mem refm = before;
      ref_row(c, refm);
      size_t bad = 0;
      for (size_t k = 0; k < m.w.size(); k++) bad += m.w[k] != refm.w[k];
      printf("       whole image vs the reference row op: %s", bad ? "FAIL" : "equal");
      if (bad) printf(" (%zu words differ)", bad);
      printf("\n");
      if (bad) fails++;
    } else if ((c.op == ctr::OP_GEMM || c.op == ctr::OP_SOFTMAX) && r >= 0) {
      std::vector<uint8_t> t(m.w.size(), 0);
      mark_outputs(c, m, t);
      size_t out = 0;
      for (size_t k = 0; k < m.w.size(); k++) out += (m.w[k] != before.w[k]) && !t[k];
      if (out) {
        printf("       %zu words changed outside the declared outputs: FAIL\n", out);
        fails++;
      }
    }
    auto cr = checks.equal_range(i);
    for (auto it = cr.first; it != cr.second; ++it) {
      const Line& L = it->second;
      const std::vector<uint8_t> ref = load_bytes(dir + "/" + L.file);
      nchecks++;
      size_t bad = 0, negz = 0;
      double maxd = 0.0;
      for (size_t k = 0; k + 1 < ref.size(); k += 2) {
        const uint16_t a = m.u16(L.addr + k), e = (uint16_t)(ref[k] | (ref[k + 1] << 8));
        if (a != e) bad++;
      }
      if (ref.size() != L.n) bad++;
      const bool f32 = L.name.find("f32") != std::string::npos;
      if (f32) {
        for (size_t k = 0; k + 3 < ref.size(); k += 4) {
          uint32_t e;
          memcpy(&e, &ref[k], 4);
          if (e == 0x80000000u) negz++;
          if (bad) maxd = std::fmax(maxd, std::fabs((double)m.f32(L.addr + k) - (double)ctr::u2f(e)));
        }
      }
      printf("       check %-16s %8zu bytes: %s", L.name.c_str(), ref.size(), bad ? "FAIL" : "bit-exact");
      if (negz) printf(" (%zu expected -0.0 values)", negz);
      if (bad) printf(" (%zu 16-bit halves differ, max |diff| %.3g)", bad, maxd);
      printf("\n");
      if (bad) fails++;
    }
    if (c.op == ctr::OP_END) break;
  }
  printf("  %d checks, %d failures\n", nchecks, fails);
  return fails;
}

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IONBF, 0);
  if (argc < 2) {
    fprintf(stderr, "usage: tb_attn [--unit] [<case dir>...]\n");
    return 2;
  }
#if defined(QZ_DFSIM) || defined(SMX_DFSIM)
  printf("threaded bounded-FIFO model: row unit %s, softmax unit %s\n",
#ifdef QZ_DFSIM
         "threaded",
#else
         "sequential",
#endif
#ifdef SMX_DFSIM
         "threaded"
#else
         "sequential"
#endif
  );
#endif
  int fails = 0;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--unit"))
      fails += unit_tests();
    else
      fails += run_case(argv[i]);
  }
  printf(fails ? "FAILED (%d)\n" : "ALL PASS\n", fails);
  return fails ? 1 : 0;
}
