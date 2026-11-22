// Host testbench of emit_amax (fpga/kernels/phaseB/emit_amax): the patched GEMM unit (patched/hls/gemm_contract.h)
// against the contract's unfused sequence. Sections:
//   Q  the loader's fabric quantizer (emit_amax.h) == quant_int16.h quant_elem == ctr::ref::quant_elem, and its scale
//      path == ctr::ref::quant_row, on random rows and edge values (ties, double rounding, denormals, saturation)
//   R  regression: tb_gemm's random commands with every 1.0 feature (P_AMX_SG = P_XQ = 0), whole DDR image bit-exact
//      against ctr::ref (the patch changes nothing when the new parameters are 0)
//   P  producer: random GEMMs with an absmax output (segments, row groups, items, bias, D, ACC, N % 64 != 0): the
//      whole image == ctr::ref's Y + the absmax parts computed from it (nothing else written)
//   C  chains: producer(s) -> [unfused: QUANT_X (qz::row_mode, the row unit's mode function) -> int16 GEMM] against
//      [fused: GEMM with P_XQ]; the consumer images must be equal bit for bit outside QUANT_X's own outputs. The
//      -12392 shapes (contract_plan.py) and random ones. --real <dir>: producer outputs = real policy rows.
//   g++ -std=c++14 -O2 -DFSTREAM_ARITH=16 -ffp-contract=off -I<this>/patched/hls -I<repo>/fpga/hls
//       -I<repo>/fpga/kernels/quantize -I<Vitis_HLS>/include tb_amax.cpp tb_rowmode.cpp
#define __USE_MINGW_ANSI_STDIO 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <random>
#include <string>
#include <vector>

#include "float_linear.h"
#include "gemm_contract.h"  // patched/hls (first on the include path)

using fstream::IoWord;
using fstream::WPort;

uint32_t tb_row_mode(const ctr::Cmd& c, void* io_shifted);  // tb_rowmode.cpp: qz::row_mode (unpatched contract)
uint32_t tb_row_validate(const ctr::Cmd& c);

static int g_fail = 0;
static int cov[8];  // random chains run: segments > 1, row groups, items, ACC, ADD_D, shift, consumer N > 64, K % 64
#define CHECK(cond, ...)          \
  do {                            \
    if (!(cond)) {                \
      printf("FAIL: " __VA_ARGS__); \
      printf("\n");               \
      g_fail++;                   \
    }                             \
  } while (0)

// ---------------------------------------------------------------- DDR image (tb_gemm.cpp's)
struct Img {
  uint64_t base = 0;
  std::vector<uint8_t> b;
  uint64_t top = 0;
  uint64_t alloc(uint64_t bytes, uint64_t align = 64) {
    top = (top + align - 1) / align * align;
    const uint64_t a = base + top;
    top += bytes + 256;  // slack: the int16 X reader may read up to 112 bytes past an item's last row
    if (top > b.size()) {
      const size_t old = b.size();
      b.resize(top + 4096, 0);
      for (size_t i = old; i < b.size(); i++) b[i] = (uint8_t)((junk_state = junk_state * 6364136223846793005ull + 1) >> 56);
    }
    return a;
  }
  uint64_t junk_state = 0x243F6A8885A308D3ull;
  uint8_t* p(uint64_t a) { return b.data() + (a - base); }
  float f32(uint64_t a) {
    float f;
    memcpy(&f, p(a), 4);
    return f;
  }
  void set_f32(uint64_t a, float f) { memcpy(p(a), &f, 4); }
  uint32_t u32(uint64_t a) {
    uint32_t u;
    memcpy(&u, p(a), 4);
    return u;
  }
  void set_u32(uint64_t a, uint32_t u) { memcpy(p(a), &u, 4); }
  int16_t i16(uint64_t a) {
    int16_t v;
    memcpy(&v, p(a), 2);
    return v;
  }
  void set_i16(uint64_t a, int16_t v) { memcpy(p(a), &v, 2); }
  IoWord* io() { return (IoWord*)b.data() - (base >> 4); }
};

static std::mt19937_64 rng;
static uint64_t rnd(uint64_t n) { return n ? rng() % n : 0; }
static float rnd_scale() {
  const int e = (int)rnd(40) - 30;
  float s = ldexpf(1.0f + (float)rnd(1 << 20) / (1 << 20), e);
  if (rnd(50) == 0) s = ldexpf(1.0f, -120);
  return s;
}
static int16_t rnd_q(int kind) {
  if (kind == 0) return (int16_t)((int)rnd(65535) - 32767);
  if (kind == 1) return (int16_t)((int)rnd(201) - 100);
  return rnd(2) ? 32767 : -32767;
}
static uint32_t f2u(float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  return u;
}
static float u2f(uint32_t u) {
  float f;
  memcpy(&f, &u, 4);
  return f;
}

// ---------------------------------------------------------------- reference GEMM (tb_gemm.cpp's ref_gemm, + XQ)
// With P_XQ the reference quantizes the FP32 X rows with ctr::ref::quant_row (the absmax from the rows themselves,
// not from the parts: the fused path must agree) and uses its stored scale as s_tok.
static void ref_gemm(const ctr::Cmd& c, Img& m) {
  const ctr::Slot &X = c.s[ctr::SLOT_A], &Wt = c.s[ctr::SLOT_B], &Y = c.s[ctr::SLOT_C], &D = c.s[ctr::SLOT_D];
  const ctr::Slot &E = c.s[ctr::SLOT_E], &Fs = c.s[ctr::SLOT_F], &G = c.s[ctr::SLOT_G];
  const uint64_t R = c.p[ctr::P_SEGC], P1 = c.p[ctr::P_SEGF];
  const bool xq = c.p[ctr::P_XQ] != 0;
  std::vector<int16_t> xr(c.K), wr((size_t)c.M * c.K);
  std::vector<float> xf(c.K);
  // read every input before any output is written (C may alias D)
  for (uint64_t b = 0; b < c.B; b++) {
    for (uint64_t mm = 0; mm < c.M; mm++)
      for (uint64_t k = 0; k < c.K; k++)
        wr[mm * c.K + k] = m.i16(ctr::w_word_addr(Wt, b, mm, k, c.K) + 2 * ctr::w_lane(mm));
    std::vector<float> out((size_t)c.N * c.M);
    for (uint64_t n = 0; n < c.N; n++) {
      float st;
      if (xq) {
        for (uint64_t k = 0; k < c.K; k++) xf[k] = m.f32(ctr::row_addr(X, b, n) + 4 * k);
        st = ctr::ref::quant_row(xf.data(), c.K, xr.data(), (int)(int32_t)c.p[ctr::P_SHIFT]);
      } else {
        for (uint64_t k = 0; k < c.K; k++) xr[k] = m.i16(ctr::row_addr(X, b, n) + 2 * k);
        st = m.f32(Fs.addr + b * Fs.stride + 4 * n);
      }
      for (uint64_t mm = 0; mm < c.M; mm++) {
        int64_t acc = 0;
        for (uint64_t k = 0; k < c.K; k++) acc += (int32_t)xr[k] * (int32_t)wr[mm * c.K + k];
        const uint64_t ya = ctr::row_addr2(Y, b, n, R, P1) + 4 * mm;
        float bias = 0.0f, add = 0.0f;
        const bool hb = c.flags & ctr::F_BIAS;
        if (hb) bias = m.f32(G.addr + b * G.stride + 4 * mm);
        if (c.flags & ctr::F_ADD_D) add = m.f32(ctr::row_addr(D, b, n) + 4 * mm);
        if (c.flags & ctr::F_ACC) add = m.f32(ya);
        const bool hd = c.flags & (ctr::F_ADD_D | ctr::F_ACC);
        out[n * c.M + mm] = ctr::ref::gemm_out(acc, m.f32(E.addr + b * E.stride + 4 * mm), st, hb ? &bias : nullptr,
                                               hd ? &add : nullptr);
      }
    }
    for (uint64_t n = 0; n < c.N; n++)
      for (uint64_t mm = 0; mm < c.M; mm++) m.set_f32(ctr::row_addr2(Y, b, n, R, P1) + 4 * mm, out[n * c.M + mm]);
  }
}
// The absmax parts the contract text defines (emit_amax.h header), from the reference image's Y.
static void ref_amax(const ctr::Cmd& c, Img& m) {
  const uint64_t sg = c.p[ctr::P_AMX_SG];
  if (!sg) return;
  const ctr::Slot &Y = c.s[ctr::SLOT_C], &H = c.s[ctr::SLOT_H];
  const uint64_t R = c.p[ctr::P_SEGC], P1 = c.p[ctr::P_SEGF], G = c.M / 32, ns = G / sg;
  for (uint64_t b = 0; b < c.B; b++)
    for (uint64_t s = 0; s < ns; s++)
      for (uint64_t t = 0; t < ctr::pad4(c.N); t++) {
        uint32_t a = 0;
        const uint64_t part = s + (R ? t / R : 0), row = R ? t % R : t;
        if (R && t >= c.N && (t / R) != (c.N - 1) / R) continue;  // a part beyond the last: nothing written
        if (t < c.N)
          for (uint64_t mm = s * sg * 32; mm < (s + 1) * sg * 32; mm++) {
            const uint32_t v = m.u32(ctr::row_addr2(Y, b, t, R, P1) + 4 * mm) & 0x7FFFFFFFu;
            a = v > a ? v : a;
          }
        m.set_u32(H.addr + b * H.stride + part * H.pitch + 4 * row, a);
      }
}

static uint32_t run_gemm(const ctr::Cmd& c, Img& m) {
  const WPort* hp = (const WPort*)m.b.data() - (m.base >> 4);
  return fgemm::gemm_unit(c, hp, hp, hp, hp, (const IoWord*)m.io(), m.io());
}

// first differing byte, or -1
static long long first_diff(const Img& a, const Img& b, size_t* count = nullptr) {
  long long f = -1;
  size_t n = 0;
  for (size_t j = 0; j < a.b.size() && j < b.b.size(); j++)
    if (a.b[j] != b.b[j]) {
      if (f < 0) f = (long long)j;
      n++;
    }
  if (count) *count = n;
  return f;
}

// ================================================================ Q: quantizer
static void test_quantizer(long long rows) {
  long long vals = 0, bad_q = 0, bad_q2 = 0, bad_s = 0, sat = 0, ties = 0;
  std::vector<float> x(64);
  std::vector<int16_t> qref(64);
  for (long long r = 0; r < rows; r++) {
    // a row: random exponent range, sometimes tiny / denormal / zero / all-equal / huge
    const int kind = (int)rnd(10);
    const int e0 = (int)rnd(250) - 140;
    for (int j = 0; j < 64; j++) {
      float v;
      if (kind == 0) v = u2f((uint32_t)rnd(0x00800000u) | (rnd(2) ? 0x80000000u : 0));  // denormals
      else if (kind == 1) v = 0.0f;
      else if (kind == 2) v = ldexpf(1.0f + (float)rnd(1 << 23) / (1 << 23), e0) * (rnd(2) ? 1 : -1);
      else v = ldexpf((float)((int64_t)rnd(1ull << 24) - (1 << 23)) / (1 << 23), e0 + (int)rnd(12) - 6);
      x[j] = v;
    }
    if (kind == 3) x[rnd(64)] = -0.0f;
    if (kind == 4) {  // values that land on k + 0.5 of the grid of s (ties, and their neighbours)
      for (int j = 1; j < 64; j++) {
        const float s = ctr::ref::scale_of_amax(f2u(x[0]) & 0x7FFFFFFF);
        float t = ((float)rnd(32767) + 0.5f) * s;
        const int d = (int)rnd(5) - 2;
        x[j] = u2f(f2u(t) + d) * (rnd(2) ? 1 : -1);
      }
      if (fabsf(x[0]) < 1e-30f) x[0] = 1.0f;
    }
    uint32_t a = 0;
    for (int j = 0; j < 64; j++) a = std::max(a, f2u(x[j]) & 0x7FFFFFFFu);
    const int shift = (int)rnd(17) - 8;
    const float sref = ctr::ref::quant_row(x.data(), 64, qref.data(), shift);
    ap_uint<32> st;
    const famx::XScale xs = famx::xscale(famx::abits(a), st, ap_int<5>(shift));
    const qz::RowScale rs = qz::row_scale(qz::abits_t(a));
    bad_s += (uint32_t)st != f2u(sref);
    for (int j = 0; j < 64; j++) {
      const qz::q_t q = famx::quant_elem_fab(qz::fbits_t(f2u(x[j])), xs);
      const qz::q_t q2 = qz::quant_elem(qz::fbits_t(f2u(x[j])), rs);
      bad_q += (int)q != (int)qref[j];
      bad_q2 += (int)q != (int)q2;
      sat += (int)q == 32767 || (int)q == -32767;
      const float v = x[j] / ctr::ref::scale_of_amax(a);
      ties += fabsf(v - nearbyintf(v)) == 0.5f;
      vals++;
    }
  }
  printf("Q  quantizer: %lld rows, %lld values (%lld at +-32767, %lld exact ties): %lld q != ctr::ref, %lld q != "
         "quant_int16.h, %lld stored scales != ctr::ref\n",
         rows, vals, sat, ties, bad_q, bad_q2, bad_s);
  CHECK(bad_q == 0 && bad_q2 == 0 && bad_s == 0, "Q quantizer mismatches");
}

// ================================================================ R: regression (tb_gemm.cpp's make_case)
static bool make_case(ctr::Cmd& c, Img& m, uint64_t base, int idx) {
  m = Img();
  m.base = base;
  memset(&c, 0, sizeof c);
  c.op = ctr::OP_GEMM;
  c.tag = idx;
  static const uint32_t Ks[] = {8, 16, 56, 64, 72, 120, 128, 136, 256, 720, 768, 960, 1000, 2048, 4096};
  static const uint32_t Ns[] = {1, 2, 31, 32, 33, 45, 50, 63, 64, 65, 96, 127, 128, 129, 144, 156, 200, 256, 257};
  const int shape = (int)rnd(10);
  uint32_t M, K, N, B;
  if (shape == 0) {
    M = 32 * (uint32_t)(1 + rnd(256));
    K = 8 * (uint32_t)(1 + rnd(16));
    N = Ns[rnd(sizeof Ns / 4)];
  } else if (shape == 1) {
    M = 32 * (uint32_t)(1 + rnd(2));
    K = 8 * (uint32_t)(256 + rnd(257));
    N = 1 + (uint32_t)rnd(70);
  } else {
    M = 32 * (uint32_t)(1 + rnd(12));
    K = Ks[rnd(sizeof Ks / 4)];
    if (K > 1024 && M > 96) M = 96;
    N = Ns[rnd(sizeof Ns / 4)];
  }
  B = rnd(4) == 0 ? 1 + (uint32_t)rnd(3) : 1;
  c.M = M;
  c.K = K;
  c.N = N;
  c.B = B;
  const int xkind = (int)rnd(3), wkind = (int)rnd(3);
  const uint64_t xp = (2 * (uint64_t)K + 15) / 16 * 16 + 16 * rnd(3);
  const uint64_t xstride = N * xp + 16 * rnd(4);
  c.s[ctr::SLOT_A] = {m.alloc(B * xstride), (uint32_t)xp, B > 1 ? (uint32_t)xstride : 0};
  const uint64_t port = (uint64_t)(M / 32) * K * 16, wp = port + 16 * rnd(3), ws = 4 * wp + 16 * rnd(3);
  c.s[ctr::SLOT_B] = {m.alloc(B * ws), (uint32_t)wp, B > 1 ? (uint32_t)ws : 0};
  uint64_t R = 0, P1 = 0, yp, ystride;
  const int ymode = (int)rnd(3);
  if (ymode == 1 && N > 1) {
    R = 1 + rnd(N);
    yp = 4 * (uint64_t)M + 16 * rnd(3);
    P1 = R * yp + 16 * rnd(4);
    ystride = ((N - 1) / R + 1) * P1;
  } else if (ymode == 2 && N > 1) {
    R = 1 + rnd(N);
    const uint64_t ng = (N - 1) / R + 1;
    P1 = 4 * (uint64_t)M;
    yp = ng * P1 + 16 * rnd(2);
    ystride = R * yp;
  } else {
    yp = 4 * (uint64_t)M + 16 * rnd(3);
    ystride = N * yp + 16 * rnd(3);
  }
  c.s[ctr::SLOT_C] = {m.alloc(B * ystride), (uint32_t)yp, B > 1 ? (uint32_t)ystride : 0};
  c.p[ctr::P_SEGC] = (uint32_t)R;
  c.p[ctr::P_SEGF] = (uint32_t)P1;
  const int ep = (int)rnd(5);
  if (rnd(2)) c.flags |= ctr::F_BIAS;
  if (ep == 1 || ep == 2) {
    c.flags |= ctr::F_ADD_D;
    const uint64_t dp = ep == 2 ? 0 : 4 * (uint64_t)M + 16 * rnd(3);
    const uint64_t dstride = (ep == 2 ? 1 : N) * (dp ? dp : 4 * (uint64_t)M) + 16 * rnd(3);
    c.s[ctr::SLOT_D] = {m.alloc(B * dstride), (uint32_t)dp, B > 1 ? (uint32_t)dstride : 0};
  } else if (ep == 3 && R == 0) {
    c.flags |= ctr::F_ADD_D;
    c.s[ctr::SLOT_D] = c.s[ctr::SLOT_C];
  } else if (ep == 4) {
    c.flags |= ctr::F_ACC;
  }
  const uint64_t estride = 4 * (uint64_t)M + 16 * rnd(2);
  c.s[ctr::SLOT_E] = {m.alloc(B * estride), 0, B > 1 ? (uint32_t)estride : 0};
  const uint64_t fstride = 4 * (uint64_t)ctr::pad4(N) + 16 * rnd(2);
  c.s[ctr::SLOT_F] = {m.alloc(B * fstride), 0, B > 1 ? (uint32_t)fstride : 0};
  if (c.flags & ctr::F_BIAS) {
    const uint64_t gstride = 4 * (uint64_t)M + 16 * rnd(2);
    c.s[ctr::SLOT_G] = {m.alloc(B * gstride), 0, B > 1 ? (uint32_t)gstride : 0};
  }
  if (ctr::validate(c) != ctr::ME_OK) return false;
  for (size_t i = 0; i < m.b.size(); i++) m.b[i] = (uint8_t)rng();
  for (uint64_t b = 0; b < B; b++) {
    for (uint64_t n = 0; n < N; n++)
      for (uint64_t k = 0; k < K; k++) m.set_i16(ctr::row_addr(c.s[ctr::SLOT_A], b, n) + 2 * k, rnd_q(xkind));
    for (uint64_t mm = 0; mm < M; mm++)
      for (uint64_t k = 0; k < K; k++)
        m.set_i16(ctr::w_word_addr(c.s[ctr::SLOT_B], b, mm, k, K) + 2 * ctr::w_lane(mm), rnd_q(wkind));
    for (uint64_t mm = 0; mm < M; mm++)
      m.set_f32(c.s[ctr::SLOT_E].addr + b * c.s[ctr::SLOT_E].stride + 4 * mm, rnd_scale());
    for (uint64_t n = 0; n < ctr::pad4(N); n++)
      m.set_f32(c.s[ctr::SLOT_F].addr + b * c.s[ctr::SLOT_F].stride + 4 * n, n < N ? rnd_scale() : 1.0f);
    if (c.flags & ctr::F_BIAS)
      for (uint64_t mm = 0; mm < M; mm++)
        m.set_f32(c.s[ctr::SLOT_G].addr + b * c.s[ctr::SLOT_G].stride + 4 * mm,
                  (float)((int)rnd(2001) - 1000) * 0.37f);
    if (c.flags & (ctr::F_ADD_D | ctr::F_ACC)) {
      for (uint64_t n = 0; n < N; n++)
        for (uint64_t mm = 0; mm < M; mm++) {
          const uint64_t a = (c.flags & ctr::F_ACC)
                                 ? ctr::row_addr2(c.s[ctr::SLOT_C], b, n, R, P1) + 4 * mm
                                 : ctr::row_addr(c.s[ctr::SLOT_D], b, n) + 4 * mm;
          float v = (float)((int)rnd(20001) - 10000) * 0.013f;
          if (rnd(64) == 0) v = ldexpf(1.0f, -127);
          m.set_f32(a, v);
        }
    }
  }
  return true;
}

static void test_regression(int cases, uint64_t base) {
  int pass = 0, tried = 0;
  for (int i = 0; pass < cases && g_fail < 10; i++) {
    ctr::Cmd c;
    Img m;
    tried++;
    if (!make_case(c, m, base, i)) continue;
    CHECK(ctr::validate_hw(c) == ctr::ME_OK, "R validate_hw rejects a validated 1.0 command");
    Img ref = m;
    ref_gemm(c, ref);
    const uint32_t r = run_gemm(c, m);
    size_t nd;
    const long long f = first_diff(ref, m, &nd);
    if (r != ctr::ME_OK || f >= 0) {
      CHECK(false, "R case %d M=%u K=%u N=%u B=%u flags=0x%x R=%u: %zu bytes differ (first +0x%llx)", i, c.M, c.K, c.N,
            c.B, c.flags, c.p[ctr::P_SEGC], nd, f);
    } else {
      pass++;
    }
  }
  printf("R  regression: %d 1.0 commands bit-exact vs ctr::ref (%d drawn, base 0x%llx)\n", pass, tried,
         (unsigned long long)base);
}

// ================================================================ chains
// A GEMM's operands, filled with random data; X int16 rows (or nothing: the consumer's X is the producer's output).
struct GemmSpec {
  uint32_t M, K, N, B = 1;
  uint32_t flags = 0;
  uint64_t R = 0, P1 = 0;
};
static void fill_w(Img& m, const ctr::Cmd& c, int wkind, uint32_t mvalid) {
  for (uint64_t b = 0; b < c.B; b++) {
    for (uint64_t mm = 0; mm < c.M; mm++)
      for (uint64_t k = 0; k < c.K; k++)
        m.set_i16(ctr::w_word_addr(c.s[ctr::SLOT_B], b, mm, k, c.K) + 2 * ctr::w_lane(mm),
                  mm < mvalid ? rnd_q(wkind) : 0);
    for (uint64_t mm = 0; mm < c.M; mm++)
      m.set_f32(c.s[ctr::SLOT_E].addr + b * c.s[ctr::SLOT_E].stride + 4 * mm, mm < mvalid ? rnd_scale() : 1.0f);
    if (c.flags & ctr::F_BIAS)
      for (uint64_t mm = 0; mm < c.M; mm++)
        m.set_f32(c.s[ctr::SLOT_G].addr + b * c.s[ctr::SLOT_G].stride + 4 * mm,
                  mm < mvalid ? (float)((int)rnd(2001) - 1000) * 0.37f : 0.0f);
  }
}
// Weights / s_row / bias of a command in fresh memory.
static void alloc_w(Img& m, ctr::Cmd& c) {
  const uint64_t port = (uint64_t)(c.M / 32) * c.K * 16, wp = port + 16 * rnd(2), ws = 4 * wp + 16 * rnd(2);
  c.s[ctr::SLOT_B] = {m.alloc(c.B * ws + 64, 4096), (uint32_t)wp, c.B > 1 ? (uint32_t)ws : 0};
  const uint64_t es = 4 * (uint64_t)c.M + 16 * rnd(2);
  c.s[ctr::SLOT_E] = {m.alloc(c.B * es), 0, c.B > 1 ? (uint32_t)es : 0};
  if (c.flags & ctr::F_BIAS) c.s[ctr::SLOT_G] = {m.alloc(c.B * es), 0, c.B > 1 ? (uint32_t)es : 0};
}

struct Chain {
  std::string name;
  Img m;
  std::vector<ctr::Cmd> prod;  // producers, in order (the last ones carry the absmax)
  ctr::Cmd qx, cu, cf;         // unfused QUANT_X + int16 consumer; fused consumer
};

// Run one chain: producers, then both consumer paths; compare.
static bool run_chain(Chain& ch, bool verbose) {
  for (auto& c : ch.prod) {
    CHECK(ctr::validate(c) == ctr::ME_OK, "%s: producer invalid (ME %u)", ch.name.c_str(), ctr::validate(c));
    CHECK(ctr::validate_hw(c) == ctr::ME_OK, "%s: producer validate_hw", ch.name.c_str());
  }
  CHECK(tb_row_validate(ch.qx) == ctr::ME_OK, "%s: QUANT_X invalid (ME %u)", ch.name.c_str(), tb_row_validate(ch.qx));
  CHECK(ctr::validate(ch.cu) == ctr::ME_OK, "%s: unfused consumer invalid (ME %u)", ch.name.c_str(),
        ctr::validate(ch.cu));
  CHECK(ctr::validate(ch.cf) == ctr::ME_OK, "%s: fused consumer invalid (ME %u)", ch.name.c_str(),
        ctr::validate(ch.cf));
  CHECK(ctr::validate_hw(ch.cf) == ctr::ME_OK, "%s: fused consumer validate_hw", ch.name.c_str());
  if (g_fail) return false;
  // producers: hardware vs ctr::ref + ref_amax
  Img ref = ch.m;
  for (auto& c : ch.prod) {
    ref_gemm(c, ref);
    ref_amax(c, ref);
    run_gemm(c, ch.m);
  }
  size_t nd;
  long long f = first_diff(ref, ch.m, &nd);
  CHECK(f < 0, "%s: producer image != ref (+absmax): %zu bytes differ, first +0x%llx", ch.name.c_str(), nd, f);
  if (f >= 0) return false;
  // unfused: QUANT_X (row unit) then the int16 GEMM; also the contract reference of the same sequence
  Img un = ch.m, fu = ch.m, rf = ch.m;
  const uint32_t rq = tb_row_mode(ch.qx, un.io());
  CHECK(rq == ctr::ME_OK, "%s: row_mode returned %u", ch.name.c_str(), rq);
  run_gemm(ch.cu, un);
  run_gemm(ch.cf, fu);
  ref_gemm(ch.cf, rf);  // the contract text of the fused command: quant_row of the FP32 rows, then the epilogue
  // compare everything but QUANT_X's outputs (int16 rows C, scales F): copy them from un into fu and rf
  auto copy_rows = [&](Img& dst, const Img& src, const ctr::Slot& s, uint64_t B, uint64_t rows, uint64_t bytes) {
    for (uint64_t b = 0; b < B; b++)
      for (uint64_t n = 0; n < rows; n++) {
        const uint64_t a = ctr::row_addr(s, b, n) - dst.base;
        memcpy(dst.b.data() + a, src.b.data() + a, bytes);
      }
  };
  for (Img* d : {&fu, &rf}) {
    copy_rows(*d, un, ch.qx.s[ctr::SLOT_C], ch.qx.B, ch.qx.N, 2 * ch.qx.K);
    const ctr::Slot& F = ch.qx.s[ctr::SLOT_F];
    for (uint64_t b = 0; b < ch.qx.B; b++) {
      const uint64_t a = F.addr + b * F.stride - d->base;
      memcpy(d->b.data() + a, un.b.data() + a, 4 * ctr::pad4(ch.qx.N));
    }
  }
  size_t n1, n2;
  const long long f1 = first_diff(un, fu, &n1), f2 = first_diff(rf, fu, &n2);
  CHECK(f1 < 0, "%s: fused consumer != unfused (QUANT_X + GEMM): %zu bytes differ, first +0x%llx", ch.name.c_str(), n1,
        f1);
  CHECK(f2 < 0, "%s: fused consumer != contract reference: %zu bytes differ, first +0x%llx", ch.name.c_str(), n2, f2);
  if (verbose)
    printf("C  %-34s producers %zu, consumer M=%u K=%u N=%u B=%u parts=%u shift=%d: %s\n", ch.name.c_str(),
           ch.prod.size(), ch.cf.M, ch.cf.K, ch.cf.N, ch.cf.B, ch.cf.p[ctr::P_XQ], (int)(int32_t)ch.cf.p[ctr::P_SHIFT],
           f1 < 0 && f2 < 0 ? "bit-exact" : "MISMATCH");
  return f1 < 0 && f2 < 0;
}

// The consumer pair over FP32 rows Xf (slot view: rows N, K columns, B items) with absmax parts at Hv (parts, pitch).
static void make_consumers(Chain& ch, const ctr::Slot& Xf, uint32_t K, uint32_t N, uint32_t B, const ctr::Slot& Hv,
                           uint32_t parts, int shift, uint32_t Mc, uint32_t cflags) {
  Img& m = ch.m;
  // QUANT_X: FP32 rows -> int16 rows + scales
  ctr::Cmd& q = ch.qx;
  memset(&q, 0, sizeof q);
  q.op = ctr::OP_QUANT_X;
  q.flags = ctr::F_QOUT;
  q.K = K;
  q.N = N;
  q.B = B;
  const uint64_t xp = 2 * (uint64_t)K + 16 * rnd(2), xs = N * xp + 16 * rnd(2);
  const uint64_t fs = 4 * (uint64_t)ctr::pad4(N) + 16 * rnd(2);
  q.s[ctr::SLOT_A] = Xf;
  q.s[ctr::SLOT_C] = {m.alloc(B * xs), (uint32_t)xp, B > 1 ? (uint32_t)xs : 0};
  q.s[ctr::SLOT_F] = {m.alloc(B * fs), 0, B > 1 ? (uint32_t)fs : 0};
  q.p[ctr::P_SHIFT] = (uint32_t)shift;
  // the consumer GEMM (shared weights, output)
  ctr::Cmd c;
  memset(&c, 0, sizeof c);
  c.op = ctr::OP_GEMM;
  c.M = Mc;
  c.K = K;
  c.N = N;
  c.B = B;
  c.flags = cflags;
  alloc_w(m, c);
  const uint64_t yp = 4 * (uint64_t)Mc + 16 * rnd(2), ys = N * yp + 16 * rnd(2);
  c.s[ctr::SLOT_C] = {m.alloc(B * ys), (uint32_t)yp, B > 1 ? (uint32_t)ys : 0};
  if (cflags & ctr::F_ADD_D) {
    c.s[ctr::SLOT_D] = {m.alloc(B * ys), (uint32_t)yp, B > 1 ? (uint32_t)ys : 0};
    for (uint64_t b = 0; b < B; b++)
      for (uint64_t n = 0; n < N; n++)
        for (uint64_t mm = 0; mm < Mc; mm++)
          m.set_f32(ctr::row_addr(c.s[ctr::SLOT_D], b, n) + 4 * mm, (float)((int)rnd(20001) - 10000) * 0.013f);
  }
  if (cflags & ctr::F_ACC)
    for (uint64_t b = 0; b < B; b++)
      for (uint64_t n = 0; n < N; n++)
        for (uint64_t mm = 0; mm < Mc; mm++)
          m.set_f32(ctr::row_addr(c.s[ctr::SLOT_C], b, n) + 4 * mm, (float)((int)rnd(20001) - 10000) * 0.013f);
  fill_w(m, c, (int)rnd(3), Mc);
  ch.cu = c;
  ch.cu.s[ctr::SLOT_A] = q.s[ctr::SLOT_C];
  ch.cu.s[ctr::SLOT_F] = q.s[ctr::SLOT_F];
  ch.cf = c;
  ch.cf.s[ctr::SLOT_A] = Xf;
  ch.cf.s[ctr::SLOT_F] = Hv;
  ch.cf.p[ctr::P_XQ] = parts;
  ch.cf.p[ctr::P_SHIFT] = (uint32_t)shift;
}

// A producer GEMM writing FP32 rows at Y (pitch, stride, R/P1) with an absmax output at H. X int16 rows random;
// data: 0 random, 1 random with zero rows / exact copies of the row max, 2 = real rows via D (X = 0).
struct Prod {
  ctr::Cmd c;
};
static ctr::Cmd make_producer(Img& m, uint32_t M, uint32_t K, uint32_t N, uint32_t B, ctr::Slot Y, uint64_t R,
                              uint64_t P1, uint32_t flags, uint32_t sg, ctr::Slot H, uint32_t mvalid) {
  ctr::Cmd c;
  memset(&c, 0, sizeof c);
  c.op = ctr::OP_GEMM;
  c.M = M;
  c.K = K;
  c.N = N;
  c.B = B;
  c.flags = flags;
  c.p[ctr::P_SEGC] = (uint32_t)R;
  c.p[ctr::P_SEGF] = (uint32_t)P1;
  c.p[ctr::P_AMX_SG] = sg;
  c.s[ctr::SLOT_C] = Y;
  if (sg) c.s[ctr::SLOT_H] = H;
  const uint64_t xp = 2 * (uint64_t)K + 16 * rnd(2), xs = N * xp;
  c.s[ctr::SLOT_A] = {m.alloc(B * xs), (uint32_t)xp, B > 1 ? (uint32_t)xs : 0};
  const uint64_t fs = 4 * (uint64_t)ctr::pad4(N);
  c.s[ctr::SLOT_F] = {m.alloc(B * fs), 0, B > 1 ? (uint32_t)fs : 0};
  alloc_w(m, c);
  if (flags & ctr::F_ADD_D) {
    const uint64_t dp = 4 * (uint64_t)M + 16 * rnd(2), dsz = N * dp;
    c.s[ctr::SLOT_D] = {m.alloc(B * dsz), (uint32_t)dp, B > 1 ? (uint32_t)dsz : 0};
  }
  const int xkind = (int)rnd(3);
  for (uint64_t b = 0; b < B; b++) {
    for (uint64_t n = 0; n < N; n++) {
      const bool zero_row = rnd(23) == 0;
      for (uint64_t k = 0; k < K; k++)
        m.set_i16(ctr::row_addr(c.s[ctr::SLOT_A], b, n) + 2 * k, zero_row ? 0 : rnd_q(xkind));
    }
    for (uint64_t n = 0; n < ctr::pad4(N); n++)
      m.set_f32(c.s[ctr::SLOT_F].addr + b * c.s[ctr::SLOT_F].stride + 4 * n, n < N ? rnd_scale() : 1.0f);
    if (flags & ctr::F_ADD_D)
      for (uint64_t n = 0; n < N; n++)
        for (uint64_t mm = 0; mm < M; mm++)
          m.set_f32(ctr::row_addr(c.s[ctr::SLOT_D], b, n) + 4 * mm,
                    mm < mvalid ? (float)((int)rnd(20001) - 10000) * 0.013f : 0.0f);
    if (flags & ctr::F_ACC)
      for (uint64_t n = 0; n < N; n++)
        for (uint64_t mm = 0; mm < M; mm++)
          m.set_f32(ctr::row_addr2(Y, b, n, R, P1) + 4 * mm,
                    mm < mvalid ? (float)((int)rnd(20001) - 10000) * 0.013f : 0.0f);
  }
  fill_w(m, c, (int)rnd(3), mvalid);
  return c;
}

// ---- the -12392 producer/consumer pairs (contract_plan.py), plus random variations
static Img fresh(uint64_t base) {
  Img m;
  m.base = base;
  return m;
}
static void junk(Img& m) {
  for (size_t i = 0; i < m.b.size(); i++) m.b[i] = (uint8_t)rng();
}

// P.V with row groups (LM prefix Tq = 45 / expert Tq = 50): 5 items x 3 query heads x 64 columns, then o_proj
// over [Tq][960] with 15 parts. two = the expert self-attention's P.V pair (pv0, then pv1 with F_ACC + absmax).
static void chain_pv(uint64_t base, uint32_t Tq, bool two, uint32_t Mo, bool verbose) {
  Chain ch;
  ch.name = std::string("pv->o_proj Tq=") + std::to_string(Tq) + (two ? " (pv0, pv1+ACC)" : "");
  ch.m = fresh(base);
  Img& m = ch.m;
  const uint32_t TqP = ctr::pad4(Tq), NQ = 3 * TqP, items = 5, D = 960;
  const uint64_t Op = 4 * D;
  const ctr::Slot O = {m.alloc((uint64_t)TqP * Op + 4096), (uint32_t)Op, 3 * 256};
  const uint64_t Hp = 4 * TqP + 16 * rnd(2);
  const ctr::Slot H = {m.alloc(15 * Hp + 64), (uint32_t)Hp, (uint32_t)(3 * Hp)};
  const uint32_t Kp = two ? 64 : 64;
  // allocate everything first, then junk, then the data (make_producer writes the data)
  m.alloc(1 << 20);
  junk(m);
  if (two) ch.prod.push_back(make_producer(m, 64, Kp, NQ, items, O, TqP, 256, 0, 0, ctr::Slot{0, 0, 0}, 64));
  ch.prod.push_back(make_producer(m, 64, two ? 64 : Kp, NQ, items, O, TqP, 256, two ? ctr::F_ACC : 0, 2, H, 64));
  if (two) {  // pv1 reads C as written by pv0: no fresh ACC data
    ctr::Cmd& p1 = ch.prod.back();
    (void)p1;
  }
  make_consumers(ch, ctr::Slot{O.addr, (uint32_t)Op, 0}, D, Tq, 1, ctr::Slot{H.addr, (uint32_t)Hp, 0}, 15, 0, Mo,
                 ctr::F_ADD_D);
  run_chain(ch, verbose);
}
// Vision: P.V per image (12 items = heads, R = 0, item stride 256 B), O [512][768]; o_proj over both images, 12 parts.
static void chain_vision_o(uint64_t base, uint32_t T, uint32_t Mo, bool verbose) {
  Chain ch;
  ch.name = "vision pv x2 -> o_proj T=" + std::to_string(T);
  ch.m = fresh(base);
  Img& m = ch.m;
  const uint32_t D = 768, heads = 12;
  const uint64_t Op = 4 * D, Hp = 4 * 2 * (uint64_t)T;  // part j of row n (both images): H + j Hp + 4 n
  const uint64_t Oa = m.alloc(2ull * T * Op + 4096), Ha = m.alloc(heads * Hp + 64);
  m.alloc(1 << 20);
  junk(m);
  for (int img = 0; img < 2; img++)
    ch.prod.push_back(make_producer(m, 64, 256, T, heads, ctr::Slot{Oa + (uint64_t)img * T * Op, (uint32_t)Op, 256}, 0,
                                    0, 0, 2, ctr::Slot{Ha + 4ull * img * T, 0, (uint32_t)Hp}, 64));
  make_consumers(ch, ctr::Slot{Oa, (uint32_t)Op, 0}, D, 2 * T, 1, ctr::Slot{Ha, (uint32_t)Hp, 0}, heads, 0, Mo,
                 ctr::F_BIAS | ctr::F_ADD_D);
  run_chain(ch, verbose);
}
// Vision Q: the qkv GEMM (M = 2304, absmax per 64-column head, 36 segments), consumer Q.K^T per image with
// B = 12 heads (X row n of head j at QKV + n pitch + 256 j, F part = head j), shift -3.
static void chain_vision_q(uint64_t base, uint32_t T, bool verbose) {
  for (int img = 0; img < 2; img++) {
    Chain ch;
    ch.name = "vision qkv -> Q.K^T img " + std::to_string(img) + " T=" + std::to_string(T);
    ch.m = fresh(base);
    Img& m = ch.m;
    const uint32_t M = 2304, N = 2 * T;
    const uint64_t Yp = 4 * (uint64_t)M, Hp = 4 * (uint64_t)ctr::pad4(N);
    const uint64_t Ya = m.alloc((uint64_t)N * Yp + 4096), Ha = m.alloc(36 * Hp + 64);
    m.alloc(1 << 20);
    junk(m);
    ch.prod.push_back(make_producer(m, M, 64, N, 1, ctr::Slot{Ya, (uint32_t)Yp, 0}, 0, 0, ctr::F_BIAS, 2,
                                    ctr::Slot{Ha, (uint32_t)Hp, 0}, M));
    make_consumers(ch, ctr::Slot{Ya + (uint64_t)img * T * Yp, (uint32_t)Yp, 256}, 64, T, 12,
                   ctr::Slot{Ha + 4ull * img * T, 0, (uint32_t)Hp}, 1, -3, 256, 0);
    run_chain(ch, verbose);
  }
}
// Whole-row producers: a_in -> t_in (Mp 736, K 720: pad columns zero), action_out + Euler (ACC, M = 32) -> a_in
// (K = 32), LM odd qkv (M = 1600, 5 segments of 10 groups) -> v_proj over part 4 (K = 320).
static void chain_rows(uint64_t base, const char* name, uint32_t Mp, uint32_t mvalid, uint32_t Kc, uint32_t N,
                       uint32_t sg, uint32_t part, uint32_t pflags, uint32_t Mo, bool verbose) {
  Chain ch;
  ch.name = name;
  ch.m = fresh(base);
  Img& m = ch.m;
  const uint64_t Yp = 4 * (uint64_t)Mp + 16 * rnd(2), Hp = 4 * (uint64_t)ctr::pad4(N) + 16 * rnd(2);
  const uint32_t G = Mp / 32, ns = G / sg;
  const uint64_t Ya = m.alloc((uint64_t)N * Yp + 4096), Ha = m.alloc(ns * Hp + 64);
  m.alloc(1 << 20);
  junk(m);
  ch.prod.push_back(make_producer(m, Mp, 64 + 8 * (uint32_t)rnd(40), N, 1, ctr::Slot{Ya, (uint32_t)Yp, 0}, 0, 0,
                                  pflags, sg, ctr::Slot{Ha, (uint32_t)Hp, 0}, mvalid));
  make_consumers(ch, ctr::Slot{Ya + 4ull * part * sg * 32, (uint32_t)Yp, 0}, Kc, N, 1,
                 ctr::Slot{Ha + part * Hp, 0, 0}, 1, 0, Mo, ctr::F_BIAS);
  run_chain(ch, verbose);
}

// Random chains: random producer shapes / segments / row groups / items / epilogues, consumer over all parts.
static void chain_random(uint64_t base, int idx, bool verbose) {
  Chain ch;
  ch.m = fresh(base);
  Img& m = ch.m;
  static const uint32_t Ns[] = {1, 2, 3, 4, 5, 31, 32, 33, 45, 50, 63, 64, 65, 96, 127, 128, 129, 156, 200, 257};
  const uint32_t N = Ns[rnd(sizeof Ns / 4)];
  const uint32_t B = rnd(3) == 0 ? 1 + (uint32_t)rnd(3) : 1;
  uint32_t G = 1 + (uint32_t)rnd(12);
  const int mode = (int)rnd(3);  // 0 whole row, 1 segments, 2 row groups (single segment)
  uint32_t sg = G;
  if (mode == 1) {
    static const uint32_t S[] = {1, 2, 3, 4};
    sg = S[rnd(4)];
    G = sg * (1 + (uint32_t)rnd(8));
  }
  const uint32_t M = 32 * G, ns = G / sg;
  const uint32_t Kp = 8 * (1 + (uint32_t)rnd(40));
  uint64_t R = 0, P1 = 0, Yp, Ystr;
  if (mode == 2 && N > 4) {
    R = 4 * (1 + rnd((N + 3) / 4));
    if (R > N) R = N - N % 4;
    if (R == 0) R = 4;
    const uint64_t ng = (N - 1) / R + 1;
    P1 = 4 * (uint64_t)M;  // groups side by side: consumer row i = parts (b, grp)
    Yp = ng * P1 + 16 * rnd(2);
    Ystr = 0;              // set below (items side by side too)
    (void)Ystr;
  } else {
    Yp = 4 * (uint64_t)M + 16 * rnd(2);
  }
  const uint32_t flags = (rnd(2) ? ctr::F_BIAS : 0) | (rnd(3) == 0 ? ctr::F_ADD_D : rnd(3) == 0 ? ctr::F_ACC : 0);
  ctr::Cmd pc;
  uint32_t Kc, Nc, parts;
  ctr::Slot Xf, Hv;
  if (R) {  // consumer rows i < R: items and row groups side by side: K = B ng M
    const uint64_t ng = (N - 1) / R + 1;
    const uint64_t rowb = B * ng * 4 * (uint64_t)M;
    const uint64_t Hp = 4 * (uint64_t)ctr::pad4((uint32_t)R) + 16 * rnd(2);
    const uint64_t Ya = m.alloc(R * (rowb + 16) + 4096), Ha = m.alloc(B * ng * Hp + 64);
    m.alloc(1 << 20);
    junk(m);
    const ctr::Slot Y = {Ya, (uint32_t)(rowb + 16 * rnd(2)), (uint32_t)(ng * 4 * M)};
    pc = make_producer(m, M, Kp, N, B, Y, R, ng > 1 ? P1 : 0, flags, sg, ctr::Slot{Ha, (uint32_t)Hp, (uint32_t)(ng * Hp)}, M);
    if (ng == 1) pc.p[ctr::P_SEGF] = 0, pc.p[ctr::P_SEGC] = 0;  // one group: plain rows
    if (ng == 1) pc.s[ctr::SLOT_H].pitch = 0;
    Kc = (uint32_t)(B * ng * M);
    Nc = (uint32_t)(ng == 1 ? N : R);
    if (ng > 1 && N % R && (rnd(2) == 0)) Nc = (uint32_t)(N % R);  // the rows every part holds
    if (ng > 1 && N % R) Nc = std::min<uint32_t>(Nc, (uint32_t)(N % R));
    parts = (uint32_t)(B * ng);
    Xf = {Ya, Y.pitch, 0};
    Hv = {Ha, (uint32_t)Hp, 0};
    if (ng == 1) {  // plain rows: items side by side in a row as well (stride = 4 M)
      Hv = {Ha, (uint32_t)Hp, 0};
    }
  } else {
    const uint64_t Hp = 4 * (uint64_t)ctr::pad4(N) + 16 * rnd(2);
    const uint64_t Ystride = N * Yp + 16 * rnd(2);
    const uint64_t Ya = m.alloc(B * Ystride + 4096), Ha = m.alloc(B * ns * Hp + 64);
    m.alloc(1 << 20);
    junk(m);
    const ctr::Slot Y = {Ya, (uint32_t)Yp, B > 1 ? (uint32_t)Ystride : 0};
    pc = make_producer(m, M, Kp, N, B, Y, 0, 0, flags, sg, ctr::Slot{Ha, ns > 1 ? (uint32_t)Hp : 0, (uint32_t)(ns * Hp)}, M);
    // consumer: either all segments (K = M, parts = ns) or one segment j (K = 32 sg, parts = 1); items as producer
    if (rnd(2) || ns == 1) {
      Kc = M;
      parts = ns;
      Xf = {Ya, (uint32_t)Yp, B > 1 ? (uint32_t)Ystride : 0};
      Hv = {Ha, (uint32_t)Hp, (uint32_t)(ns * Hp)};
    } else {
      const uint32_t j = (uint32_t)rnd(ns);
      Kc = 32 * sg;
      parts = 1;
      Xf = {Ya + 4ull * j * Kc, (uint32_t)Yp, B > 1 ? (uint32_t)Ystride : 0};
      Hv = {Ha + j * Hp, 0, (uint32_t)(ns * Hp)};
    }
    Nc = rnd(3) == 0 && N > 1 ? 1 + (uint32_t)rnd(N) : N;
  }
  ch.prod.push_back(pc);
  const uint32_t Bc = R ? 1 : B;
  const int shift = rnd(3) == 0 ? (int)rnd(17) - 8 : 0;
  const uint32_t Mo = 32 * (1 + (uint32_t)rnd(4));
  if (Kc > 4096) {  // beyond GEMM_K_MAX
    cov[7]++;
    return;
  }
  make_consumers(ch, Xf, Kc, Nc, Bc, Hv, parts, shift, Mo, rnd(2) ? ctr::F_BIAS : 0);
  ch.name = "random " + std::to_string(idx) + " (P: M=" + std::to_string(M) + " N=" + std::to_string(N) +
            " B=" + std::to_string(B) + " sg=" + std::to_string(sg) + " R=" + std::to_string(R) + ")";
  if (ctr::validate(ch.cf) != ctr::ME_OK || ctr::validate(pc) != ctr::ME_OK) {
    cov[7]++;
    return;
  }
  if (run_chain(ch, verbose)) {
    cov[0] += ns > 1;
    cov[1] += R != 0;
    cov[2] += B > 1;
    cov[3] += (flags & ctr::F_ACC) != 0;
    cov[4] += (flags & ctr::F_ADD_D) != 0;
    cov[5] += shift != 0;
    cov[6] += Nc > 64;
  }
}

// ---- validation of the new fields
static void test_validate() {
  int bad = 0;
  Img m = fresh(0x800000000ull);
  ctr::Cmd c;
  memset(&c, 0, sizeof c);
  c.op = ctr::OP_GEMM;
  c.M = 128;
  c.K = 64;
  c.N = 50;
  c.B = 1;
  c.s[ctr::SLOT_A] = {m.alloc(50 * 256), 256, 0};
  alloc_w(m, c);
  c.s[ctr::SLOT_C] = {m.alloc(50 * 512), 512, 0};
  c.s[ctr::SLOT_F] = {m.alloc(256), 0, 0};
  const ctr::Cmd ok = c;
  auto expect = [&](const ctr::Cmd& x, uint32_t me, const char* what) {
    const uint32_t r = ctr::validate(x);
    if (r != me) {
      printf("FAIL: validate %s: ME %u, expected %u\n", what, r, me);
      bad++;
    }
  };
  expect(ok, ctr::ME_OK, "1.0 GEMM");
  c = ok;
  c.p[ctr::P_AMX_SG] = 2;
  expect(c, ctr::ME_ALIGN, "AMX 2 segments, H pitch 0");
  c.s[ctr::SLOT_H] = {m.alloc(1024), 208, 0};
  expect(c, ctr::ME_OK, "AMX sg 2 (2 segments)");
  c.s[ctr::SLOT_H].pitch = 192;
  expect(c, ctr::ME_ALIGN, "AMX part pitch < 4 pad4(N)");
  c = ok;
  c.p[ctr::P_AMX_SG] = 3;
  c.s[ctr::SLOT_H] = {m.alloc(1024), 208, 0};
  expect(c, ctr::ME_PARAM, "AMX sg 3 does not divide G 4");
  c.p[ctr::P_AMX_SG] = 5;
  expect(c, ctr::ME_PARAM, "AMX sg > G");
  c = ok;
  c.p[ctr::P_SEGC] = 26;
  c.p[ctr::P_SEGF] = 512;
  c.s[ctr::SLOT_C].pitch = 1024;
  c.p[ctr::P_AMX_SG] = 4;
  c.s[ctr::SLOT_H] = {m.alloc(1024), 112, 0};
  expect(c, ctr::ME_PARAM, "AMX row groups R % 4 != 0");
  c.p[ctr::P_SEGC] = 28;
  c.s[ctr::SLOT_H].pitch = 112;
  expect(c, ctr::ME_OK, "AMX row groups R = 28");
  c.p[ctr::P_AMX_SG] = 2;
  expect(c, ctr::ME_PARAM, "AMX row groups with 2 segments");
  c = ok;
  c.s[ctr::SLOT_H] = {m.alloc(1024), 208, 0};
  expect(c, ctr::ME_SLOT, "H without P_AMX_SG");
  c = ok;
  c.p[ctr::P_XQ] = 3;
  c.s[ctr::SLOT_F] = {m.alloc(3 * 208), 208, 0};
  expect(c, ctr::ME_OK, "XQ 3 parts, FP32 A pitch 256 = 4 K");
  c.s[ctr::SLOT_A].pitch = 192;
  expect(c, ctr::ME_ALIGN, "XQ FP32 A pitch < 4 K");
  c.s[ctr::SLOT_A].pitch = 256;
  c.s[ctr::SLOT_F].pitch = 192;
  expect(c, ctr::ME_ALIGN, "XQ part pitch < 4 pad4(N)");
  c.s[ctr::SLOT_F].pitch = 208;
  c.p[ctr::P_SHIFT] = (uint32_t)-3;
  expect(c, ctr::ME_OK, "XQ shift -3");
  c.p[ctr::P_SHIFT] = 9;
  expect(c, ctr::ME_PARAM, "XQ shift 9");
  c = ok;
  c.p[ctr::P_SHIFT] = 1;
  expect(c, ctr::ME_PARAM, "shift without XQ");
  c = ok;
  c.p[ctr::P_XQ] = 65;
  expect(c, ctr::ME_PARAM, "XQ 65 parts");
  c = ok;
  c.p[10] = 1;
  expect(c, ctr::ME_PARAM, "GEMM p10");
  printf("V  validate: new-field rules %s\n", bad ? "FAIL" : "pass (20 cases)");
  g_fail += bad;
}

// ---- real rows (policy dumps, check_policy.py --dump-dir): producer X = 0, D = the real rows -> Y = the rows (FTZ)
struct RealRows {
  std::string name;
  uint32_t n, k;
  std::vector<float> v;
};
static bool load_real(const std::string& path, std::vector<RealRows>& out) {
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) return false;
  char mg[4];
  while (fread(mg, 1, 4, f) == 4) {
    uint32_t h[3];
    if (fread(h, 4, 3, f) != 3) break;
    RealRows r;
    r.n = h[0];
    r.k = h[1];
    r.name.resize(h[2]);
    if (fread(&r.name[0], 1, h[2], f) != h[2]) break;
    r.v.resize((size_t)r.n * r.k);
    if (fread(r.v.data(), 4, r.v.size(), f) != r.v.size()) break;
    out.push_back(r);
  }
  fclose(f);
  return true;
}
// one real block of rows [n][k] (k % 32 == 0): a producer with X = 0 and D = rows writes them (FTZ) with an absmax
// per 64 columns (k % 64 == 0) or per row; the consumer quantizes them on load.
static void chain_real(uint64_t base, const RealRows& rr, bool verbose) {
  Chain ch;
  ch.name = "real " + rr.name + " [" + std::to_string(rr.n) + "][" + std::to_string(rr.k) + "]";
  ch.m = fresh(base);
  Img& m = ch.m;
  const uint32_t N = rr.n, K = rr.k, M = (K + 31) / 32 * 32, G = M / 32;
  const uint32_t sg = (K % 64 == 0 && G % 2 == 0) ? 2 : G, ns = G / sg;
  const uint64_t Yp = 4 * (uint64_t)M, Hp = 4 * (uint64_t)ctr::pad4(N);
  const uint64_t Ya = m.alloc((uint64_t)N * Yp + 4096), Ha = m.alloc(ns * Hp + 64);
  m.alloc(1 << 16);
  junk(m);
  ctr::Cmd p = make_producer(m, M, 8, N, 1, ctr::Slot{Ya, (uint32_t)Yp, 0}, 0, 0, ctr::F_ADD_D, sg,
                             ctr::Slot{Ha, ns > 1 ? (uint32_t)Hp : 0, 0}, M);
  for (uint64_t n = 0; n < N; n++) {
    for (uint64_t k = 0; k < 8; k++) m.set_i16(ctr::row_addr(p.s[ctr::SLOT_A], 0, n) + 2 * k, 0);
    for (uint64_t mm = 0; mm < M; mm++)
      m.set_f32(ctr::row_addr(p.s[ctr::SLOT_D], 0, n) + 4 * mm, mm < K ? rr.v[n * K + mm] : 0.0f);
  }
  ch.prod.push_back(p);
  make_consumers(ch, ctr::Slot{Ya, (uint32_t)Yp, 0}, K, N, 1, ctr::Slot{Ha, (uint32_t)Hp, 0}, ns, 0, 64, 0);
  run_chain(ch, verbose);
}

int main(int argc, char** argv) {
  int qrows = 200000, rcases = 150, ncr = 120;
  uint64_t seed = 1;
  std::string real;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--quick")) qrows = 20000, rcases = 30, ncr = 30;
    else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = strtoull(argv[++i], nullptr, 0);
    else if (!strcmp(argv[i], "--real") && i + 1 < argc) real = argv[++i];
  }
  rng.seed(seed);
  printf("emit_amax tb: AMX_QL=%d NBX=%d%s, seed %llu\n", famx::QL, fstream16::NBX,
#ifdef FSTREAM_CHAOS
         " CHAOS",
#else
         "",
#endif
         (unsigned long long)seed);
  test_quantizer(qrows);
  test_validate();
  test_regression(rcases, 0x800000000ull);
  test_regression(rcases / 3, 0x40000000ull);
  // -12392 shapes
  for (uint64_t base : {0x800000000ull, 0x40000000ull}) {
    chain_pv(base, 45, false, 960, true);   // LM prefix: o_proj 960 x 960, N = 45
    chain_pv(base, 50, false, 736, true);   // expert cross: o_proj 736 x 960, N = 50
    chain_pv(base, 50, true, 736, true);    // expert self: pv0 + pv1 (ACC)
    chain_vision_o(base, 256, 768, true);   // vision o_proj over 512 rows
    chain_vision_q(base, 256, true);        // vision Q per head, p4 = -3
    chain_rows(base, "a_in -> t_in (736 -> 720)", 736, 720, 720, 50, 23, 0, ctr::F_BIAS, 736, true);
    chain_rows(base, "a_out+Euler (ACC) -> a_in (K 32)", 32, 32, 32, 50, 1, 0, ctr::F_BIAS | ctr::F_ACC, 736, true);
    chain_rows(base, "LM odd qkv seg 4 -> v_proj (K 320)", 1600, 1600, 320, 45, 10, 4, 0, 320, true);
  }
  int nr = 0;
  for (int i = 0; i < ncr && g_fail < 10; i++, nr++) chain_random(i % 2 ? 0x800000000ull : 0x40000000ull, i, false);
  printf("C  random chains: %d drawn, %d skipped (K > 4096, > 64 parts or invalid); bit-exact ones with segments > 1: "
         "%d, row groups %d, items %d, ACC %d, ADD_D %d, shift %d, consumer N > 64 %d\n",
         nr, cov[7], cov[0], cov[1], cov[2], cov[3], cov[4], cov[5], cov[6]);
  if (!real.empty()) {
    std::vector<RealRows> rows;
    if (!load_real(real, rows)) {
      printf("FAIL: cannot read %s\n", real.c_str());
      g_fail++;
    }
    for (auto& r : rows) chain_real(0x800000000ull, r, true);
    printf("C  real rows: %zu blocks from %s\n", rows.size(), real.c_str());
  }
  printf("%s: tb_amax (%d failures)\n", g_fail ? "FAIL" : "PASS", g_fail);
  return g_fail ? 1 : 0;
}
