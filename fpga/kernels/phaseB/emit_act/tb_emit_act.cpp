// Host testbench of emit_act (GELU / SILU / SILU_MUL in the GEMM emit) against the contract's unfused sequence.
//
// 1. h unit: ea::h_bits (0 DSP: mux-tree shift, fabric D f) == gs::h_bits (the gelu_silu kernel, DSP shift and
//    product) on every table position T < 2^TB of both functions and both signs, and on random bits of every
//    exponent (zeros, denormals, saturation, Inf / NaN patterns).
// 2. GEMM cases, each run twice on the same DDR image (same addresses, same junk in every pad and gap):
//      unfused (contract 1.0): GEMM (natural gate|up blob, FP32 Y) -> GELU / SILU_MUL / SILU+NOMUL with FP32 out
//                              (O), and again with F_QOUT (int16 Q + scales S)
//      fused   (this patch):   GEMM with p6 = P_ACT (interleaved blob for SILU_MUL) -> O, then QUANT_X(O) -> Q, S
//    Checks: every command passes ctr::validate and validate_hw; the two final images are equal byte for byte
//    outside the unfused Y buffer (so O, Q and S are bit-identical, and the fused path writes nothing else), and
//    the fused image's Y buffer is untouched; O also equals a third, direct reference (ctr::ref::gemm_out +
//    gs::h_bits + ctr::ref::fmul_hw) computed from the operands.
//    Cases: the four -12392 act GEMMs at full size (SigLIP fc1, LM and expert gate|up, action-time in with its D
//    row), and random shapes / scales / pitches / items / bias / D (rows and broadcast), N % 32 != 0, K % 64 != 0,
//    extreme operands (zero, denormal-product, saturating and tail inputs).
//    g++ -std=c++14 -O2 -ffp-contract=off -DFSTREAM_ARITH=16 -DGS_NSB=10 [-DFSTREAM_CHAOS=30] [-DFSTREAM_NBX=1]
//        -I<tree>/hls -I<Vitis>/include tb_emit_act.cpp
//    tb_emit_act [random cases] [seed] [base_hex] [--no-real] [--no-h]
#define __USE_MINGW_ANSI_STDIO 1
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <random>
#include <string>
#include <vector>

#include "float_linear.h"
#include "gemm_contract.h"
#include "../kernels/gelu_silu/gelu_silu.h"
#include "../kernels/quantize/quant_mode.h"

#if GS_NSB != EA_NSB
#error "build with -DGS_NSB=10 (= EA_NSB): the fused and the unfused activation must use the same table"
#endif

using fstream::IoWord;
using fstream::WPort;

static std::mt19937_64 rng;
static uint64_t rnd(uint64_t n) { return n ? rng() % n : 0; }
static double urand() { return (double)(rng() >> 11) * (1.0 / 9007199254740992.0); }

// ---------------------------------------------------------------- 1. the h unit
static int check_h() {
  long long n = 0, bad = 0;
  auto one = [&](uint32_t xb, bool ts) {
    const uint32_t a = (uint32_t)ea::h_bits((ap_uint<32>)xb, ts);
    const uint32_t b = (uint32_t)gs::h_bits<0>((ap_uint<32>)xb, ts);
    n++;
    if (a != b && bad++ < 10) printf("h mismatch x=0x%08x silu=%d: ea 0x%08x gs 0x%08x\n", xb, (int)ts, a, b);
  };
  for (int ts = 0; ts < 2; ts++) {
    const int rb = ts ? 4 : 3;
    for (uint32_t T = 0; T < (1u << ea::TB); T++) {  // x = T 2^(rb - TB) exactly: every (segment, fraction)
      const float x = ldexpf((float)T, rb - ea::TB);
      uint32_t u;
      memcpy(&u, &x, 4);
      one(u, ts);
      one(u | 0x80000000u, ts);
    }
    for (uint32_t e = 0; e < 256; e++)
      for (int i = 0; i < 2048; i++) {
        uint32_t m = (uint32_t)rng() & 0x7FFFFF;
        if (i == 0) m = 0;
        if (i == 1) m = 0x7FFFFF;
        one((e << 23) | m, ts);
        one(0x80000000u | (e << 23) | m, ts);
      }
  }
  printf("%s: h unit ea::h_bits == gs::h_bits on %lld inputs (NSB %d, every table position, every exponent)\n",
         bad ? "FAIL" : "PASS", n, ea::NSB);
  return bad ? 1 : 0;
}

// ---------------------------------------------------------------- DDR image
struct Img {
  uint64_t base = 0;
  std::vector<uint8_t> b;
  uint64_t top = 0;
  uint64_t alloc(uint64_t bytes, uint64_t align = 64) {
    top = (top + align - 1) / align * align;
    const uint64_t a = base + top;
    top += bytes + 256;  // slack: the X reader may read up to 112 bytes past an item's last row
    if (top > b.size()) b.resize(top + 4096, 0);
    return a;
  }
  uint8_t* p(uint64_t a) { return b.data() + (a - base); }
  float f32(uint64_t a) {
    float f;
    memcpy(&f, p(a), 4);
    return f;
  }
  uint32_t u32(uint64_t a) {
    uint32_t u;
    memcpy(&u, p(a), 4);
    return u;
  }
  void set_f32(uint64_t a, float f) { memcpy(p(a), &f, 4); }
  void set_i16(uint64_t a, int16_t v) { memcpy(p(a), &v, 2); }
  IoWord* io() { return (IoWord*)b.data() - (base >> 4); }
  const WPort* hp() { return (const WPort*)b.data() - (base >> 4); }
};

static float bits_f(uint32_t u) {
  float f;
  memcpy(&f, &u, 4);
  return f;
}
static uint32_t f_bits(float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  return u;
}

// ---------------------------------------------------------------- one case: operands in natural layout
struct Case {
  std::string name;
  int act = 0;         // ea::ACT_*
  uint32_t M = 0;      // blob rows (SILU_MUL: 2F, natural = [gate F rows; up F rows])
  uint32_t K = 0, N = 0, B = 1;
  uint32_t F = 0;      // output columns
  bool bias = false;   // F_BIAS
  int dmode = 0;       // 0 none, 1 D rows, 2 D broadcast (pitch 0)
  std::vector<int16_t> X, W;           // [b][n][k], [b][m][k] natural rows
  std::vector<float> srow, bia, stok;  // [b][m], [b][m], [b][n]
  std::vector<float> D;                // [b][n or 1][m]
};

static int16_t q_of(int kind) {
  if (kind == 0) return (int16_t)((int)rnd(65535) - 32767);
  if (kind == 1) return (int16_t)((int)rnd(201) - 100);
  if (kind == 2) return rnd(2) ? 32767 : -32767;
  // kind 3: mostly zero, some full range (sparse rows)
  return rnd(8) == 0 ? (int16_t)((int)rnd(65535) - 32767) : 0;
}

// natural blob row of fused blob row r (EPI_SILU_MUL: per 32 rows, 16 gate then 16 up)
static uint32_t nat_row(const Case& cs, uint32_t r) {
  if (cs.act != ea::ACT_SILU_MUL) return r;
  const uint32_t i = r / 32, c = r % 32;
  return c < 16 ? 16 * i + c : cs.F + 16 * i + (c - 16);
}

static void fill(Case& cs, double target, int xkind, int wkind) {
  const uint32_t M = cs.M, K = cs.K, N = cs.N, B = cs.B;
  cs.X.resize((size_t)B * N * K);
  cs.W.resize((size_t)B * M * K);
  for (auto& v : cs.X) v = q_of(xkind);
  for (auto& v : cs.W) v = q_of(wkind);
  // zero rows: some tokens and some weight rows
  for (uint32_t b = 0; b < B; b++) {
    if (N > 2 && rnd(3) == 0) memset(&cs.X[((size_t)b * N + rnd(N)) * K], 0, 2 * K);
    if (rnd(3) == 0) memset(&cs.W[((size_t)b * M + rnd(M)) * K], 0, 2 * K);
  }
  double xs = 0, ws = 0;
  for (auto v : cs.X) xs += (double)v * v;
  for (auto v : cs.W) ws += (double)v * v;
  const double accrms = sqrt((double)K) * sqrt(xs / cs.X.size() + 1) * sqrt(ws / cs.W.size() + 1);
  cs.srow.resize((size_t)B * M);
  cs.bia.resize((size_t)B * M);
  cs.stok.resize((size_t)B * N);
  for (uint32_t i = 0; i < B * M; i++) {
    double s = target / accrms * ldexp(1.0 + urand(), (int)rnd(9) - 5);
    if (rnd(64) == 0) s = ldexp(1.0, -120);  // denormal products: flushed
    if (rnd(64) == 0) s = target / accrms * 4096.0;  // saturating rows (|x| >> 16)
    cs.srow[i] = (float)s;
    cs.bia[i] = cs.bias ? (float)((urand() * 2 - 1) * 3.0) : 0.0f;
    if (cs.bias && rnd(40) == 0) cs.bia[i] = -0.0f;
  }
  for (uint32_t i = 0; i < B * N; i++) {
    double s = ldexp(1.0 + urand(), (int)rnd(5) - 2);
    if (rnd(50) == 0) s = 1e-30;
    cs.stok[i] = (float)s;
  }
  if (cs.dmode) {
    const uint32_t dn = cs.dmode == 2 ? 1 : N;
    cs.D.resize((size_t)B * dn * M);
    for (auto& v : cs.D) {
      v = (float)((urand() * 2 - 1) * 4.0);
      if (rnd(64) == 0) v = ldexpf(1.0f, -127);  // a denormal addend (flushed)
    }
  }
}

// ---------------------------------------------------------------- buffers (identical in both images)
struct Lay {
  ctr::Slot X, Wn, Wf, En, Ef, Gn, Gf, Fs, D, Y, O, Q, S;
};

static void put_blob(Img& m, const Case& cs, const ctr::Slot& w, bool fused) {
  for (uint32_t b = 0; b < cs.B; b++)
    for (uint32_t r = 0; r < cs.M; r++) {
      const uint32_t nr = fused ? nat_row(cs, r) : r;
      for (uint32_t k = 0; k < cs.K; k++)
        m.set_i16(ctr::w_word_addr(w, b, r, k, cs.K) + 2 * ctr::w_lane(r), cs.W[((size_t)b * cs.M + nr) * cs.K + k]);
    }
}
static void put_vec(Img& m, const Case& cs, const ctr::Slot& s, const std::vector<float>& v, bool fused) {
  for (uint32_t b = 0; b < cs.B; b++)
    for (uint32_t r = 0; r < cs.M; r++) m.set_f32(s.addr + b * s.stride + 4 * r, v[(size_t)b * cs.M + (fused ? nat_row(cs, r) : r)]);
}

static Lay layout(Img& m, const Case& cs) {
  Lay L;
  const uint32_t M = cs.M, K = cs.K, N = cs.N, B = cs.B, F = cs.F;
  auto sl = [&](uint64_t row_bytes, uint64_t rows, uint64_t slack_rows) {
    const uint64_t pitch = (row_bytes + 15) / 16 * 16 + 16 * rnd(slack_rows);
    const uint64_t stride = rows * pitch + 16 * rnd(4);
    const uint64_t a = m.alloc(B * stride);
    return ctr::Slot{a, (uint32_t)pitch, B > 1 ? (uint32_t)stride : 0};
  };
  L.X = sl(2 * K, N, 3);
  const uint64_t port = (uint64_t)(M / 32) * K * 16, wp = port + 16 * rnd(3), ws = 4 * wp + 16 * rnd(3);
  L.Wn = {m.alloc(B * ws), (uint32_t)wp, B > 1 ? (uint32_t)ws : 0};
  L.Wf = {m.alloc(B * ws), (uint32_t)wp, B > 1 ? (uint32_t)ws : 0};
  auto vec = [&](uint64_t bytes) {
    const uint64_t stride = (bytes + 15) / 16 * 16 + 16 * rnd(2);
    return ctr::Slot{m.alloc(B * stride), 0, B > 1 ? (uint32_t)stride : 0};
  };
  L.En = vec(4 * M);
  L.Ef = vec(4 * M);
  L.Gn = cs.bias ? vec(4 * M) : ctr::Slot{0, 0, 0};
  L.Gf = cs.bias ? vec(4 * M) : ctr::Slot{0, 0, 0};
  L.Fs = vec(4 * ctr::pad4(N));
  if (cs.dmode == 1) L.D = sl(4 * M, N, 3);
  else if (cs.dmode == 2) {
    const uint64_t stride = 4 * M + 16 * rnd(3);
    L.D = {m.alloc(B * stride), 0, B > 1 ? (uint32_t)stride : 0};
  } else L.D = {0, 0, 0};
  // unfused Y: gate | up rows of M FP32 columns; SILU_MUL reads the up half at Y + 4F with the same pitch
  L.Y = sl(4 * M, N, 3);
  L.O = sl(4 * F, N, 3);
  L.Q = sl(2 * F, N, 3);
  L.S = vec(4 * ctr::pad4(N));
  return L;
}

static void put_operands(Img& m, const Case& cs, const Lay& L) {
  for (size_t i = 0; i < m.b.size(); i++) m.b[i] = (uint8_t)rng();  // junk everywhere first
  const uint32_t K = cs.K, N = cs.N;
  for (uint32_t b = 0; b < cs.B; b++) {
    for (uint32_t n = 0; n < N; n++)
      for (uint32_t k = 0; k < K; k++) m.set_i16(ctr::row_addr(L.X, b, n) + 2 * k, cs.X[((size_t)b * N + n) * K + k]);
    for (uint32_t n = 0; n < ctr::pad4(N); n++)
      m.set_f32(L.Fs.addr + b * L.Fs.stride + 4 * n, n < N ? cs.stok[(size_t)b * N + n] : 1.0f);
    if (cs.dmode) {
      const uint32_t dn = cs.dmode == 2 ? 1 : N;
      for (uint32_t n = 0; n < dn; n++)
        for (uint32_t r = 0; r < cs.M; r++)
          m.set_f32(ctr::row_addr(L.D, b, n) + 4 * r, cs.D[((size_t)b * dn + n) * cs.M + r]);
    }
  }
  put_blob(m, cs, L.Wn, false);
  put_blob(m, cs, L.Wf, true);
  put_vec(m, cs, L.En, cs.srow, false);
  put_vec(m, cs, L.Ef, cs.srow, true);
  if (cs.bias) {
    put_vec(m, cs, L.Gn, cs.bia, false);
    put_vec(m, cs, L.Gf, cs.bia, true);
  }
}

static ctr::Cmd gemm_cmd(const Case& cs, const Lay& L, bool fused) {
  ctr::Cmd c;
  memset(&c, 0, sizeof c);
  c.op = ctr::OP_GEMM;
  c.M = cs.M;
  c.K = cs.K;
  c.N = cs.N;
  c.B = cs.B;
  c.s[ctr::SLOT_A] = L.X;
  c.s[ctr::SLOT_B] = fused ? L.Wf : L.Wn;
  c.s[ctr::SLOT_C] = fused ? L.O : L.Y;
  c.s[ctr::SLOT_E] = fused ? L.Ef : L.En;
  c.s[ctr::SLOT_F] = L.Fs;
  if (cs.bias) {
    c.flags |= ctr::F_BIAS;
    c.s[ctr::SLOT_G] = fused ? L.Gf : L.Gn;
  }
  if (cs.dmode) {
    c.flags |= ctr::F_ADD_D;
    c.s[ctr::SLOT_D] = L.D;
  }
  if (fused) c.p[ctr::P_ACT] = (uint32_t)cs.act;
  return c;
}
static ctr::Cmd act_cmd(const Case& cs, const Lay& L, bool qout) {
  ctr::Cmd c;
  memset(&c, 0, sizeof c);
  c.op = cs.act == ea::ACT_GELU ? ctr::OP_GELU : ctr::OP_SILU_MUL;
  c.flags = (qout ? ctr::F_QOUT : 0) | (cs.act == ea::ACT_SILU ? ctr::F_NOMUL : 0);
  c.K = cs.F;
  c.N = cs.N;
  c.B = cs.B;
  c.s[ctr::SLOT_A] = L.Y;
  if (cs.act == ea::ACT_SILU_MUL) c.s[ctr::SLOT_B] = {L.Y.addr + 4 * cs.F, L.Y.pitch, L.Y.stride};
  c.s[ctr::SLOT_C] = qout ? L.Q : L.O;
  if (qout) c.s[ctr::SLOT_F] = L.S;
  return c;
}
static ctr::Cmd quant_cmd(const Case& cs, const Lay& L) {
  ctr::Cmd c;
  memset(&c, 0, sizeof c);
  c.op = ctr::OP_QUANT_X;
  c.flags = ctr::F_QOUT;
  c.K = cs.F;
  c.N = cs.N;
  c.B = cs.B;
  c.s[ctr::SLOT_A] = L.O;
  c.s[ctr::SLOT_C] = L.Q;
  c.s[ctr::SLOT_F] = L.S;
  return c;
}

// direct reference of O[b][n][c] from the operands (ctr::ref epilogue, gs::h_bits, ctr::ref::fmul_hw)
static float ref_x(const Case& cs, uint32_t b, uint32_t n, uint32_t r) {
  int64_t acc = 0;
  const int16_t* x = &cs.X[((size_t)b * cs.N + n) * cs.K];
  const int16_t* w = &cs.W[((size_t)b * cs.M + r) * cs.K];
  for (uint32_t k = 0; k < cs.K; k++) acc += (int32_t)x[k] * (int32_t)w[k];
  const float bias = cs.bia[(size_t)b * cs.M + r];
  float add = 0.0f;
  if (cs.dmode) add = cs.D[((size_t)b * (cs.dmode == 2 ? 1 : cs.N) + (cs.dmode == 2 ? 0 : n)) * cs.M + r];
  return ctr::ref::gemm_out(acc, cs.srow[(size_t)b * cs.M + r], cs.stok[(size_t)b * cs.N + n], cs.bias ? &bias : nullptr,
                            cs.dmode ? &add : nullptr);
}
static float ref_h(float x, bool ts) { return bits_f((uint32_t)gs::h_bits<0>((ap_uint<32>)f_bits(x), ts)); }
static float ref_out(const Case& cs, uint32_t b, uint32_t n, uint32_t c) {
  if (cs.act == ea::ACT_SILU_MUL) {
    const float g = ref_x(cs, b, n, c), u = ref_x(cs, b, n, cs.F + c);
    return ctr::ref::fmul_hw(u, ctr::ref::fmul_hw(g, ref_h(g, true)));
  }
  const float x = ref_x(cs, b, n, c);
  return ctr::ref::fmul_hw(x, ref_h(x, cs.act != ea::ACT_GELU));
}

struct Stat {
  long long vals = 0, zero = 0, negzero = 0, sat = 0;
};

static bool run_case(Case& cs, uint64_t base, Stat& st, bool verbose) {
  Img m0;
  m0.base = base;
  const Lay L = layout(m0, cs);
  put_operands(m0, cs, L);
  const ctr::Cmd gu = gemm_cmd(cs, L, false), a1 = act_cmd(cs, L, false), a2 = act_cmd(cs, L, true);
  const ctr::Cmd gf = gemm_cmd(cs, L, true), qx = quant_cmd(cs, L);
  const ctr::Cmd* all[] = {&gu, &a1, &a2, &gf, &qx};
  const char* nm[] = {"GEMM", "ACT", "ACT+QOUT", "GEMM+P_ACT", "QUANT_X"};
  for (int i = 0; i < 5; i++) {
    const uint32_t v = ctr::validate(*all[i]), vh = ctr::validate_hw(*all[i]);
    if (v != ctr::ME_OK || vh != ctr::ME_OK) {
      printf("FAIL %s: %s invalid (validate %u, validate_hw %u)\n", cs.name.c_str(), nm[i], v, vh);
      return false;
    }
  }
  Img mu = m0, mf = m0;
  // unfused
  {
    IoWord* io = mu.io();
    uint32_t r = fgemm::gemm_unit(gu, mu.hp(), mu.hp(), mu.hp(), mu.hp(), (const IoWord*)io, io);
    r |= gs::act_mode(a1, (const IoWord*)io, io);
    r |= gs::act_mode(a2, (const IoWord*)io, io);
    if (r != ctr::ME_OK) {
      printf("FAIL %s: unfused path returned %u\n", cs.name.c_str(), r);
      return false;
    }
  }
  // fused
  {
    IoWord* io = mf.io();
    uint32_t r = fgemm::gemm_unit(gf, mf.hp(), mf.hp(), mf.hp(), mf.hp(), (const IoWord*)io, io);
    if (r != ctr::ME_OK) {
      printf("FAIL %s: fused GEMM returned %u\n", cs.name.c_str(), r);
      return false;
    }
    // the fused GEMM alone: O must equal the unfused O (Q, S are still the pre-image)
    r = qz::row_mode(qx, reinterpret_cast<const qz::word_t*>(io), reinterpret_cast<qz::word_t*>(io));
    if (r != ctr::ME_OK) {
      printf("FAIL %s: QUANT_X returned %u\n", cs.name.c_str(), r);
      return false;
    }
  }
  // 1. whole images: equal outside the unfused Y buffer; the fused image's Y buffer untouched
  const uint64_t y0 = L.Y.addr - base, y1 = y0 + (uint64_t)cs.B * (L.Y.stride ? L.Y.stride : (uint64_t)cs.N * L.Y.pitch);
  size_t diff = 0, ydiff = 0, first = 0;
  for (size_t j = 0; j < m0.b.size(); j++) {
    const bool iny = j >= y0 && j < y1;
    if (iny) {
      ydiff += mf.b[j] != m0.b[j];
    } else if (mu.b[j] != mf.b[j]) {
      if (!diff) first = j;
      diff++;
    }
  }
  // 2. O against the direct reference; activation statistics
  size_t rdiff = 0;
  for (uint32_t b = 0; b < cs.B; b++)
    for (uint32_t n = 0; n < cs.N; n++)
      for (uint32_t c = 0; c < cs.F; c++) {
        const uint32_t got = mf.u32(ctr::row_addr(L.O, b, n) + 4 * c);
        const uint32_t want = f_bits(ref_out(cs, b, n, c));
        if (got != want && rdiff++ < 3)
          printf("  ref mismatch %s b=%u n=%u c=%u got 0x%08x want 0x%08x\n", cs.name.c_str(), b, n, c, got, want);
        st.vals++;
        st.zero += (got & 0x7FFFFFFFu) == 0;
        st.negzero += got == 0x80000000u;
        const float x = ref_x(cs, b, n, c);
        st.sat += fabsf(x) >= (cs.act == ea::ACT_GELU ? 8.0f : 16.0f);
      }
  const bool ok = diff == 0 && ydiff == 0 && rdiff == 0;
  if (!ok || verbose)
    printf("%s %s: act %d M=%u K=%u N=%u B=%u F=%u bias=%d D=%d: %zu bytes differ outside Y (first +0x%zx), %zu in Y, "
           "%zu vs direct ref\n",
           ok ? "ok  " : "FAIL", cs.name.c_str(), cs.act, cs.M, cs.K, cs.N, cs.B, cs.F, (int)cs.bias, cs.dmode, diff,
           first, ydiff, rdiff);
  return ok;
}

static Case real_case(const char* name, int act, uint32_t M, uint32_t K, uint32_t N, bool bias, int dmode) {
  Case cs;
  cs.name = name;
  cs.act = act;
  cs.M = M;
  cs.K = K;
  cs.N = N;
  cs.B = 1;
  cs.F = ea::out_cols((uint32_t)act, M);
  cs.bias = bias;
  cs.dmode = dmode;
  fill(cs, act == ea::ACT_GELU ? 2.0 : 3.0, 0, 0);
  return cs;
}

static bool random_case(Case& cs, int idx) {
  static const uint32_t Ks[] = {8, 16, 56, 64, 72, 120, 128, 136, 256, 720, 768, 960, 1000};
  static const uint32_t Ns[] = {1, 2, 3, 31, 32, 33, 45, 50, 63, 64, 65, 96, 127, 129};
  cs = Case();
  cs.act = 1 + (int)rnd(3);
  cs.name = "rand" + std::to_string(idx);
  cs.M = 32 * (uint32_t)(1 + rnd(12));
  cs.K = Ks[rnd(sizeof Ks / 4)];
  cs.N = Ns[rnd(sizeof Ns / 4)];
  cs.B = rnd(4) == 0 ? 2 + (uint32_t)rnd(2) : 1;
  cs.F = ea::out_cols((uint32_t)cs.act, cs.M);
  cs.bias = rnd(2) != 0;
  cs.dmode = cs.act == ea::ACT_SILU_MUL ? 0 : (int)rnd(3);
  static const double targets[] = {0.05, 0.5, 2.0, 6.0, 30.0};
  fill(cs, targets[rnd(5)], (int)rnd(4), (int)rnd(4));
  return true;
}

#ifndef EA_TB_NO_MAIN
int main(int argc, char** argv) {
  int cases = 300;
  uint64_t seed = 1, base = 0x800000000ull;
  bool real = true, hchk = true;
  int pos = 0;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--no-real")) real = false;
    else if (!strcmp(argv[i], "--no-h")) hchk = false;
    else if (pos == 0) cases = atoi(argv[i]), pos++;
    else if (pos == 1) seed = strtoull(argv[i], nullptr, 0), pos++;
    else base = strtoull(argv[i], nullptr, 16);
  }
  rng.seed(seed);
  int fail = 0;
  if (hchk) fail += check_h();
  // validation of the new parameter (patched contract.h): rejected combinations
  {
    Case cs = real_case("v", ea::ACT_SILU_MUL, 64, 8, 4, false, 0);
    Img m;
    m.base = base;
    const Lay L = layout(m, cs);
    ctr::Cmd c = gemm_cmd(cs, L, true);
    int vf = 0;
    c.p[ctr::P_ACT] = 4;
    vf += ctr::validate(c) != ctr::ME_PARAM;
    c.p[ctr::P_ACT] = ea::ACT_SILU_MUL;
    c.flags |= ctr::F_ACC;
    vf += ctr::validate(c) != ctr::ME_FLAGS;
    c.flags &= ~ctr::F_ACC;
    c.flags |= ctr::F_ADD_D;
    c.s[ctr::SLOT_D] = L.Y;
    vf += ctr::validate(c) != ctr::ME_FLAGS;
    c.flags &= ~ctr::F_ADD_D;
    c.s[ctr::SLOT_D] = {0, 0, 0};
    c.p[ctr::P_SEGC] = 2;
    vf += ctr::validate_hw(c) != ctr::ME_PARAM;
    c.p[ctr::P_SEGC] = 0;
    c.s[ctr::SLOT_C].pitch = 4 * 32 - 16;  // below 4 F (F = 32)
    vf += ctr::validate(c) != ctr::ME_ALIGN;
    c.s[ctr::SLOT_C].pitch = 4 * 32;  // = 4 F: accepted for SILU_MUL (an unfused GEMM needs 4 M)
    vf += ctr::validate(c) != ctr::ME_OK;
    c.p[ctr::P_ACT] = 0;
    vf += ctr::validate(c) != ctr::ME_ALIGN;
    printf("%s: P_ACT validation (bad value, ACC, ADD_D with SILU_MUL, row groups, C pitch 4F)\n", vf ? "FAIL" : "PASS");
    fail += vf != 0;
  }
  Stat st;
  int pass = 0, nfail = 0;
  if (real) {
    // the four act GEMMs of the -12392 list (contract_plan.py): fc1 + gelu, LM and expert gate|up + silu * up,
    // action_time_mlp_in (D = the step's time row, broadcast) + silu
    Case rc[] = {real_case("siglip_fc1_gelu", ea::ACT_GELU, 3072, 768, 512, true, 0),
                 real_case("lm_gate_up_silu_mul", ea::ACT_SILU_MUL, 5120, 960, 45, false, 0),
                 real_case("expert_gate_up_silu_mul", ea::ACT_SILU_MUL, 4096, 720, 50, false, 0),
                 real_case("action_time_in_silu", ea::ACT_SILU, 736, 720, 50, false, 2)};
    for (auto& cs : rc) {
      if (run_case(cs, base, st, true)) pass++;
      else nfail++;
    }
  }
  for (int i = 0; i < cases; i++) {
    Case cs;
    random_case(cs, i);
    if (run_case(cs, base, st, false)) pass++;
    else if (++nfail > 5) break;
  }
  fail += nfail;
  printf("%s: emit_act %d of %d GEMM cases: fused (GEMM + P_ACT, QUANT_X) == unfused (GEMM, GELU/SILU_MUL FP32 and "
         "F_QOUT) byte for byte, and O == direct reference; %lld output values (%lld zero, %lld -0, %lld with |x| at "
         "or past the table range); base 0x%llx, seed %llu\n",
         fail ? "FAIL" : "PASS", pass, pass + nfail, st.vals, st.zero, st.negzero, st.sat, (unsigned long long)base,
         (unsigned long long)seed);
  return fail ? 1 : 0;
}
#endif  // EA_TB_NO_MAIN
