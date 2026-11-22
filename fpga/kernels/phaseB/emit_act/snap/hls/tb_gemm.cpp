// Host testbench of the contract GEMM unit (gemm_contract.h) against the contract's reference arithmetic
// (fpga/kernels/contract.h ctr::ref::gemm_out, CONTRACT.md 2.3 / 3.3): random commands with every slot feature
// (items, row groups R / P1, bias, D with a pitch or broadcast, D aliasing C, F_ACC, N % 32 != 0, K % 64 != 0,
// pitches above the row, addresses above 4 GB), each validated by ctr::validate, run on a DDR image; the whole image
// after the unit must equal the reference's bit for bit (so no word outside the declared outputs changes).
//   g++ -std=c++14 -O2 -DFSTREAM_ARITH=16 -ffp-contract=off [-DFSTREAM_CHAOS=30] [-DFSTREAM_NBX=1] tb_gemm.cpp
//   tb_gemm [cases] [seed] [base_hex]
#define __USE_MINGW_ANSI_STDIO 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <random>
#include <vector>

#include "float_linear.h"
#include "gemm_contract.h"

using fstream::IoWord;
using fstream::WPort;

struct Img {
  uint64_t base = 0;
  std::vector<uint8_t> b;
  uint64_t top = 0;  // bump allocator (byte offset)
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
  void set_f32(uint64_t a, float f) { memcpy(p(a), &f, 4); }
  uint32_t u32(uint64_t a) {
    uint32_t u;
    memcpy(&u, p(a), 4);
    return u;
  }
  int16_t i16(uint64_t a) {
    int16_t v;
    memcpy(&v, p(a), 2);
    return v;
  }
  void set_i16(uint64_t a, int16_t v) { memcpy(p(a), &v, 2); }
};

static std::mt19937_64 rng;
static uint64_t rnd(uint64_t n) { return n ? rng() % n : 0; }
static float rnd_scale() {
  const int e = (int)rnd(40) - 30;  // 2^-30 .. 2^9, sometimes tiny (denormal products)
  float s = ldexpf(1.0f + (float)rnd(1 << 20) / (1 << 20), e);
  if (rnd(50) == 0) s = ldexpf(1.0f, -120);
  return s;
}
static int16_t rnd_q(int kind) {
  if (kind == 0) return (int16_t)((int)rnd(65535) - 32767);   // full range, never -32768
  if (kind == 1) return (int16_t)((int)rnd(201) - 100);
  return rnd(2) ? 32767 : -32767;                              // extremes
}

// ---------------------------------------------------------------- reference (contract text)
static void ref_gemm(const ctr::Cmd& c, Img& m) {
  const ctr::Slot &X = c.s[ctr::SLOT_A], &Wt = c.s[ctr::SLOT_B], &Y = c.s[ctr::SLOT_C], &D = c.s[ctr::SLOT_D];
  const ctr::Slot &E = c.s[ctr::SLOT_E], &Fs = c.s[ctr::SLOT_F], &G = c.s[ctr::SLOT_G];
  const uint64_t R = c.p[ctr::P_SEGC], P1 = c.p[ctr::P_SEGF];
  std::vector<int16_t> xr(c.K), wr((size_t)c.M * c.K);
  for (uint64_t b = 0; b < c.B; b++) {
    for (uint64_t mm = 0; mm < c.M; mm++)
      for (uint64_t k = 0; k < c.K; k++) wr[mm * c.K + k] = m.i16(ctr::w_word_addr(Wt, b, mm, k, c.K) + 2 * ctr::w_lane(mm));
    for (uint64_t n = 0; n < c.N; n++) {
      for (uint64_t k = 0; k < c.K; k++) xr[k] = m.i16(ctr::row_addr(X, b, n) + 2 * k);
      const float st = m.f32(Fs.addr + b * Fs.stride + 4 * n);
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
        const float y = ctr::ref::gemm_out(acc, m.f32(E.addr + b * E.stride + 4 * mm), st, hb ? &bias : nullptr,
                                           hd ? &add : nullptr);
        m.set_f32(ya, y);
      }
    }
  }
}

// ---------------------------------------------------------------- one random command
struct Case {
  ctr::Cmd c;
  Img img;
};

static bool make_case(Case& cs, uint64_t base, int idx) {
  Img& m = cs.img;
  m = Img();
  m.base = base;
  ctr::Cmd& c = cs.c;
  memset(&c, 0, sizeof c);
  c.op = ctr::OP_GEMM;
  c.tag = idx;
  static const uint32_t Ks[] = {8, 16, 56, 64, 72, 120, 128, 136, 256, 720, 768, 960, 1000, 2048, 4096};
  static const uint32_t Ns[] = {1, 2, 31, 32, 33, 45, 50, 63, 64, 65, 96, 127, 128, 129, 144, 156, 200, 256, 257};
  const int shape = (int)rnd(10);
  uint32_t M, K, N, B;
  if (shape == 0) {  // big M
    M = 32 * (uint32_t)(1 + rnd(256));
    K = 8 * (uint32_t)(1 + rnd(16));
    N = Ns[rnd(sizeof Ns / 4)];
  } else if (shape == 1) {  // long K
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
  // X rows
  const uint64_t xp = (2 * (uint64_t)K + 15) / 16 * 16 + 16 * rnd(3);
  const uint64_t xstride = N * xp + 16 * rnd(4);
  c.s[ctr::SLOT_A] = {m.alloc(B * xstride), (uint32_t)xp, B > 1 ? (uint32_t)xstride : 0};
  // weights: 4 port arrays
  const uint64_t port = (uint64_t)(M / 32) * K * 16, wp = port + 16 * rnd(3), ws = 4 * wp + 16 * rnd(3);
  c.s[ctr::SLOT_B] = {m.alloc(B * ws), (uint32_t)wp, B > 1 ? (uint32_t)ws : 0};
  // output rows, optional row groups
  uint64_t R = 0, P1 = 0, yp, ystride;
  const int ymode = (int)rnd(3);
  if (ymode == 1 && N > 1) {  // stacked groups: P1 >= R * pitch
    R = 1 + rnd(N);
    yp = 4 * (uint64_t)M + 16 * rnd(3);
    P1 = R * yp + 16 * rnd(4);
    ystride = ((N - 1) / R + 1) * P1;
  } else if (ymode == 2 && N > 1) {  // the attention P.V layout: groups side by side in a row (P1 = 4 M)
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
  // epilogue
  const int ep = (int)rnd(5);  // 0 none, 1 D rows, 2 D broadcast, 3 D == C (in place), 4 F_ACC
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
  // contents: everything random first (junk in pads and gaps), then the operands
  for (size_t i = 0; i < m.b.size(); i++) m.b[i] = (uint8_t)rng();
  for (uint64_t b = 0; b < B; b++) {
    for (uint64_t n = 0; n < N; n++)
      for (uint64_t k = 0; k < K; k++) m.set_i16(ctr::row_addr(c.s[ctr::SLOT_A], b, n) + 2 * k, rnd_q(xkind));
    for (uint64_t mm = 0; mm < M; mm++)
      for (uint64_t k = 0; k < K; k++)
        m.set_i16(ctr::w_word_addr(c.s[ctr::SLOT_B], b, mm, k, K) + 2 * ctr::w_lane(mm), rnd_q(wkind));
    for (uint64_t mm = 0; mm < M; mm++) m.set_f32(c.s[ctr::SLOT_E].addr + b * c.s[ctr::SLOT_E].stride + 4 * mm, rnd_scale());
    for (uint64_t n = 0; n < ctr::pad4(N); n++)
      m.set_f32(c.s[ctr::SLOT_F].addr + b * c.s[ctr::SLOT_F].stride + 4 * n, n < N ? rnd_scale() : 1.0f);
    if (c.flags & ctr::F_BIAS)
      for (uint64_t mm = 0; mm < M; mm++)
        m.set_f32(c.s[ctr::SLOT_G].addr + b * c.s[ctr::SLOT_G].stride + 4 * mm, (float)((int)rnd(2001) - 1000) * 0.37f);
    const bool hd = c.flags & (ctr::F_ADD_D | ctr::F_ACC);
    if (hd) {
      for (uint64_t n = 0; n < N; n++)
        for (uint64_t mm = 0; mm < M; mm++) {
          const uint64_t a = (c.flags & ctr::F_ACC)
                                 ? ctr::row_addr2(c.s[ctr::SLOT_C], b, n, R, P1) + 4 * mm
                                 : ctr::row_addr(c.s[ctr::SLOT_D], b, n) + 4 * mm;
          float v = (float)((int)rnd(20001) - 10000) * 0.013f;
          if (rnd(64) == 0) v = ldexpf(1.0f, -127);  // a denormal addend (flushed)
          m.set_f32(a, v);
        }
    }
  }
  return true;
}

// review R7: the 43-bit capture is exact only if no operand is -32768: every producer clamps to +-32767
static bool clamp_ok(const ctr::Cmd& c, Img& m) {
  for (uint64_t b = 0; b < c.B; b++) {
    for (uint64_t n = 0; n < c.N; n++)
      for (uint64_t k = 0; k < c.K; k++)
        if (m.i16(ctr::row_addr(c.s[ctr::SLOT_A], b, n) + 2 * k) == -32768) return false;
    for (uint64_t mm = 0; mm < c.M; mm++)
      for (uint64_t k = 0; k < c.K; k++)
        if (m.i16(ctr::w_word_addr(c.s[ctr::SLOT_B], b, mm, k, c.K) + 2 * ctr::w_lane(mm)) == -32768) return false;
  }
  return true;
}

int main(int argc, char** argv) {
  const int cases = argc > 1 ? atoi(argv[1]) : 200;
  const uint64_t seed = argc > 2 ? strtoull(argv[2], nullptr, 0) : 1;
  const uint64_t base = argc > 3 ? strtoull(argv[3], nullptr, 16) : 0x800000000ull;
  rng.seed(seed);
  int pass = 0, fail = 0, tried = 0;
  long long words = 0;
  int feat[8] = {0};
  for (int i = 0; pass + fail < cases; i++) {
    Case cs;
    tried++;
    if (!make_case(cs, base, i)) continue;
    const ctr::Cmd& c = cs.c;
    if (!clamp_ok(c, cs.img)) {
      printf("FAIL: case %d holds -32768 (R7)\n", i);
      return 2;
    }
    if (ctr::validate_hw(c) != ctr::ME_OK) {
      printf("FAIL: validate_hw rejects a validated command\n");
      return 2;
    }
    Img ref = cs.img;
    ref_gemm(c, ref);
    IoWord* io = (IoWord*)cs.img.b.data() - (base >> 4);
    const WPort* hp = (const WPort*)cs.img.b.data() - (base >> 4);
    const uint32_t r = fgemm::gemm_unit(c, hp, hp, hp, hp, (const IoWord*)io, io);
    size_t diff = 0, first = 0;
    for (size_t j = 0; j < ref.b.size(); j++)
      if (ref.b[j] != cs.img.b[j]) {
        if (!diff) first = j;
        diff++;
      }
    words += (long long)ref.b.size() / 16;
    feat[0] += c.B > 1;
    feat[1] += c.p[ctr::P_SEGC] != 0;
    feat[2] += (c.flags & ctr::F_BIAS) != 0;
    feat[3] += (c.flags & ctr::F_ADD_D) && c.s[ctr::SLOT_D].pitch == 0;
    feat[4] += (c.flags & ctr::F_ADD_D) && c.s[ctr::SLOT_D].addr == c.s[ctr::SLOT_C].addr;
    feat[5] += (c.flags & ctr::F_ACC) != 0;
    feat[6] += c.N % 32 != 0;
    feat[7] += c.K % 64 != 0;
    if (r != ctr::ME_OK || diff) {
      fail++;
      printf("FAIL case %d: M=%u K=%u N=%u B=%u flags=0x%x R=%u P1=%u ret=%u, %zu bytes differ (first at +0x%zx)\n", i,
             c.M, c.K, c.N, c.B, c.flags, c.p[ctr::P_SEGC], c.p[ctr::P_SEGF], r, (unsigned long long)diff, (unsigned long long)first);
      if (fail > 5) break;
    } else {
      pass++;
    }
  }
  printf("%s: tb_gemm %d of %d commands bit-exact vs ctr::ref (%d drawn, %lld image words compared, base 0x%llx); "
         "items %d, row groups %d, bias %d, D broadcast %d, D==C %d, ACC %d, N%%32 %d, K%%64 %d\n",
         fail ? "FAIL" : "PASS", pass, pass + fail, tried, words, (unsigned long long)base, feat[0], feat[1], feat[2],
         feat[3], feat[4], feat[5], feat[6], feat[7]);
  return fail ? 1 : 0;
}
