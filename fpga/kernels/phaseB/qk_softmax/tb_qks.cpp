// Host testbench of the phase-B fusion qk_softmax (OP_QK_SOFTMAX, contract_qks.h).
//
// Each case is one attention command (B items of N query rows x L keys, head dim K) in a DDR image at a 40-bit
// base. The reference is the contract 1.0 sequence the fusion replaces, run with the units' own HLS code:
//   GEMM   Q.K^T -> FP32 scores in DDR        fgemm::gemm_unit  (snap/hls/gemm_contract.h, the integrator's unit)
//   SOFTMAX scores -> P int16 + scales         smx::softmax_mode (snap/kernels/softmax/softmax_core.h)
// The fused command runs fgemm_qks::gemm_unit (gemm_qks.h: the same unit with the fused path) on a copy of the
// same image. Pass = after restoring the score scratch of the reference to its initial bytes, the two images are
// equal byte for byte (so P, the scales, the zero rows [N, NP), the scale pad lanes and every byte outside the
// declared outputs agree), and the fused run never wrote the score scratch.
//
//   tb_qks [random cases] [seed] [base_hex]      random commands (+ the plan's 4 shapes, + adversarial ones)
//   tb_qks --vec <file> ...                      real -12392 calls dumped by policy_check.py --dump-vec: the
//                                                fused unit's P / scales must also equal the torch emulation's
// Build: see run_tests.sh (Vitis MinGW g++, -ffp-contract=off, Vitis HLS include dir for ap_int / hls_stream).
#define __USE_MINGW_ANSI_STDIO 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <random>
#include <vector>

#include "float_linear.h"
#include "gemm_contract.h"  // the reference GEMM unit (namespace fgemm)
#include "../kernels/softmax/softmax_core.h"
#define fgemm fgemm_qks  // the fused unit: the same file with the fused path, in its own namespace
#include "gemm_qks.h"
#undef fgemm

using fstream::IoWord;
using fstream::WPort;

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
  void set_f32(uint64_t a, float f) { memcpy(p(a), &f, 4); }
  void set_i16(uint64_t a, int16_t v) { memcpy(p(a), &v, 2); }
  void set_i32(uint64_t a, int32_t v) { memcpy(p(a), &v, 4); }
  int16_t i16(uint64_t a) {
    int16_t v;
    memcpy(&v, p(a), 2);
    return v;
  }
  float f32(uint64_t a) {
    float f;
    memcpy(&f, p(a), 4);
    return f;
  }
};

static std::mt19937_64 rng;
static uint64_t rnd(uint64_t n) { return n ? rng() % n : 0; }

// One attention command in both forms, with the inputs in the image.
struct Case {
  Img img;
  ctr::Cmd fused, gemm, smx;
  uint64_t s_addr = 0, s_bytes = 0;  // the reference's score scratch [s_addr, s_addr + s_bytes)
  const char* name = "";
};

struct Shape {
  uint32_t L, K, N, B, NP;
  bool mask;
  float p0;
  int qkind, kkind, skind, rkind;  // operand / scale / rank generators
  uint32_t qgap, pgap, sgap, xs_gap;  // pitch gaps (words) of Q, P, S rows; item stride gaps
  bool rk_shared;                  // rank_k stride 0 (one vector for every item)
  uint32_t tqp;                    // > 0: stacked-head Q layout (rows g * tqp + i, rows i >= tq zero, scale 1.0)
  uint32_t tq;
  int site;                        // rank pattern: 0 random, 1 prefix, 2 expert self, 3 expert cross
  uint32_t npre;                   // expert self: valid prefix keys (rows [0, pad32(npre)))
};

static int16_t rnd_q(int kind) {
  if (kind == 0) return (int16_t)((int)rnd(65535) - 32767);
  if (kind == 1) return (int16_t)((int)rnd(201) - 100);
  if (kind == 2) return rnd(2) ? 32767 : -32767;
  // gaussian-ish, as a quantized activation row: most values small, the row max near 32767
  const double g = std::normal_distribution<double>(0.0, 6000.0)(rng);
  return (int16_t)(g > 32767 ? 32767 : g < -32767 ? -32767 : (int)g);
}
static float rnd_scale(int kind) {
  if (kind == 0) {  // wide: denormal products, huge scores
    const int e = (int)rnd(40) - 30;
    float s = ldexpf(1.0f + (float)rnd(1 << 20) / (1 << 20), e);
    if (rnd(50) == 0) s = ldexpf(1.0f, -120);
    return s;
  }
  // realistic: |score| up to ~10-30 (Q scale ~ amax / 32767 / 8, K scale ~ amax / 32767)
  return ldexpf(1.0f + (float)rnd(1 << 20) / (1 << 20), -17 + (int)rnd(3));
}

static bool build(Case& cs, const Shape& sh, uint64_t base) {
  Img& m = cs.img;
  m = Img();
  m.base = base;
  const uint32_t L = sh.L, K = sh.K, N = sh.N, B = sh.B, NP = sh.NP;
  // Q rows (A), s_tok (G)
  const uint64_t qp = (2 * (uint64_t)K + 15) / 16 * 16 + 16 * sh.qgap;
  const uint64_t qs = N * qp + 16 * sh.xs_gap;
  const uint64_t qa = m.alloc(B * qs);
  const uint64_t gs = 4 * (uint64_t)ctr::pad4(N) + 16 * sh.xs_gap;
  const uint64_t ga = m.alloc(B * gs);
  // KW blob (B), s_row (H)
  const uint64_t port = (uint64_t)(L / 32) * K * 16, wp = port + 16 * sh.xs_gap, ws = 4 * wp + 16 * sh.xs_gap;
  const uint64_t wa = m.alloc(B * ws, 4096);
  const uint64_t hs = 4 * (uint64_t)L + 16 * sh.xs_gap;
  const uint64_t ha = m.alloc(B * hs);
  // ranks (D, E)
  const uint64_t ds = 4 * (uint64_t)ctr::pad4(N) + 16 * sh.xs_gap, es = sh.rk_shared ? 0 : 4 * (uint64_t)L + 16 * sh.xs_gap;
  const uint64_t da = sh.mask ? m.alloc(B * ds) : 0;
  const uint64_t ea = sh.mask ? m.alloc(sh.rk_shared ? 4 * (uint64_t)L : B * es) : 0;
  // P rows (C), scales (F)
  const uint64_t pp = 2 * (uint64_t)L + 16 * sh.pgap, ps = NP * pp + 16 * sh.xs_gap;
  const uint64_t pa = m.alloc(B * ps);
  const uint64_t fs = 4 * (uint64_t)ctr::pad4(NP) + 16 * sh.xs_gap;
  const uint64_t fa = m.alloc(B * fs);
  // the reference's scores (FP32 rows)
  const uint64_t sp = 4 * (uint64_t)L + 16 * sh.sgap, sst = N * sp;
  const uint64_t sa = m.alloc(B * sst);
  cs.s_addr = sa;
  cs.s_bytes = B * sst;
  auto stride = [&](uint64_t s) { return B > 1 ? (uint32_t)s : 0u; };
  // ---- the fused command
  ctr::Cmd& c = cs.fused;
  memset(&c, 0, sizeof c);
  c.op = ctr::OP_QK_SOFTMAX;
  c.flags = ctr::F_QOUT | (sh.mask ? ctr::F_MASK : 0);
  c.M = L;
  c.K = K;
  c.N = N;
  c.B = B;
  c.s[ctr::SLOT_A] = {qa, (uint32_t)qp, stride(qs)};
  c.s[ctr::SLOT_B] = {wa, (uint32_t)wp, stride(ws)};
  c.s[ctr::SLOT_C] = {pa, (uint32_t)pp, stride(ps)};
  if (sh.mask) {
    c.s[ctr::SLOT_D] = {da, 0, stride(ds)};
    c.s[ctr::SLOT_E] = {ea, 0, sh.rk_shared ? 0u : stride(es)};
  }
  c.s[ctr::SLOT_F] = {fa, 0, stride(fs)};
  c.s[ctr::SLOT_G] = {ga, 0, stride(gs)};
  c.s[ctr::SLOT_H] = {ha, 0, stride(hs)};
  c.p[ctr::P_EPS] = ctr::f2u(sh.p0);
  if (NP != N) c.p[ctr::P_NP] = NP;
  // ---- the unfused pair
  ctr::Cmd& g = cs.gemm;
  memset(&g, 0, sizeof g);
  g.op = ctr::OP_GEMM;
  g.M = L;
  g.K = K;
  g.N = N;
  g.B = B;
  g.s[ctr::SLOT_A] = c.s[ctr::SLOT_A];
  g.s[ctr::SLOT_B] = c.s[ctr::SLOT_B];
  g.s[ctr::SLOT_C] = {sa, (uint32_t)sp, stride(sst)};
  g.s[ctr::SLOT_E] = c.s[ctr::SLOT_H];
  g.s[ctr::SLOT_F] = c.s[ctr::SLOT_G];
  ctr::Cmd& s = cs.smx;
  memset(&s, 0, sizeof s);
  s.op = ctr::OP_SOFTMAX;
  s.flags = c.flags;
  s.K = L;
  s.N = N;
  s.B = B;
  s.s[ctr::SLOT_A] = g.s[ctr::SLOT_C];
  s.s[ctr::SLOT_C] = c.s[ctr::SLOT_C];
  s.s[ctr::SLOT_F] = c.s[ctr::SLOT_F];
  s.s[ctr::SLOT_D] = c.s[ctr::SLOT_D];
  s.s[ctr::SLOT_E] = c.s[ctr::SLOT_E];
  s.p[ctr::P_EPS] = c.p[ctr::P_EPS];
  s.p[ctr::P_NP] = c.p[ctr::P_NP];
  if (ctr::validate_qks(c) || ctr::validate_qks_hw(c) || ctr::validate(g) || ctr::validate(s)) return false;
  // ---- contents: junk everywhere (pads, gaps, outputs), then the operands
  for (size_t i = 0; i < m.b.size(); i++) m.b[i] = (uint8_t)rng();
  for (uint64_t b = 0; b < B; b++) {
    for (uint64_t n = 0; n < N; n++) {
      const bool zero_row = sh.tqp && (n % sh.tqp) >= sh.tq;  // a padded query row of a stacked head
      for (uint64_t k = 0; k < K; k++) m.set_i16(qa + b * qs + n * qp + 2 * k, zero_row ? 0 : rnd_q(sh.qkind));
    }
    for (uint64_t n = 0; n < ctr::pad4(N); n++) {
      const bool zero_row = n >= N || (sh.tqp && (n % sh.tqp) >= sh.tq);
      m.set_f32(ga + b * gs + 4 * n, zero_row ? 1.0f : rnd_scale(sh.skind) * 0.125f);
    }
    for (uint64_t mm = 0; mm < L; mm++) {
      const bool pad_key = sh.site == 2 ? !((mm < sh.npre) || (mm >= ctr::pad32(sh.npre) && mm < ctr::pad32(sh.npre) + sh.tq))
                         : sh.site ? mm >= sh.npre : false;
      for (uint64_t k = 0; k < K; k++)
        m.set_i16(ctr::w_word_addr(c.s[ctr::SLOT_B], b, mm, k, K) + 2 * ctr::w_lane(mm), pad_key ? 0 : rnd_q(sh.kkind));
      m.set_f32(ha + b * hs + 4 * mm, pad_key ? 1.0f : rnd_scale(sh.skind));
    }
    if (sh.mask) {
      for (uint64_t n = 0; n < ctr::pad4(N); n++) {
        int32_t r;
        const uint32_t i = sh.tqp ? (uint32_t)(n % sh.tqp) : (uint32_t)n;
        if (sh.site == 1) r = (n >= N || i >= sh.tq) ? -1 : (i == sh.tq - 1 ? 1 : 0);
        else if (sh.site == 2) r = (n >= N || i >= sh.tq) ? -1 : 2 + (int32_t)i;
        else if (sh.site == 3) r = (n >= N || i >= sh.tq) ? -1 : 0;
        else r = rnd(6) == 0 ? -1 - (int32_t)rnd(3) : (int32_t)rnd(1 + rnd(8));
        if (n >= N && sh.site == 0) r = (int32_t)rng();  // the pad lanes of rank_q are junk
        m.set_i32(da + b * ds + 4 * n, r);
      }
      if (b == 0 || !sh.rk_shared) {
        for (uint64_t k = 0; k < L; k++) {
          int32_t r;
          if (sh.site == 1) r = k < sh.npre ? (k == sh.npre - 1 ? 1 : 0) : -1;
          else if (sh.site == 2) {
            const uint32_t lp = ctr::pad32(sh.npre);
            r = k < sh.npre ? (k == sh.npre - 1 ? 1 : 0) : (k >= lp && k < lp + sh.tq) ? 2 + (int32_t)(k - lp) : -1;
          } else if (sh.site == 3) r = k < sh.npre ? 0 : -1;
          else r = rnd(5) == 0 ? (int32_t)(0x80000000u | (uint32_t)rng()) : (int32_t)rnd(1 + rnd(9));
          m.set_i32(ea + b * es + 4 * k, r);
        }
      }
    }
  }
  return true;
}

struct Result {
  bool ok;
  size_t diff, first, swrites;
  uint32_t ru, rf;
  // what the fused P holds (so a pass is not a pass on trivial rows): rows, rows with every q = 0 (fully masked),
  // rows whose max q is 32767, nonzero q, rows with >= 2 nonzero q
  long rows, zero_rows, max_rows, nonzero, spread_rows;
};

static void p_stats(const Case& cs, Img& m, Result& r) {
  const ctr::Cmd& c = cs.fused;
  for (uint64_t b = 0; b < c.B; b++)
    for (uint64_t n = 0; n < c.N; n++) {
      int nz = 0, mx = 0;
      for (uint64_t k = 0; k < c.M; k++) {
        const int q = m.i16(ctr::row_addr(c.s[ctr::SLOT_C], b, n) + 2 * k);
        nz += q != 0;
        mx = q > mx ? q : mx;
      }
      r.rows++;
      r.zero_rows += nz == 0;
      r.max_rows += mx == 32767;
      r.nonzero += nz;
      r.spread_rows += nz >= 2;
    }
}

static Result run_case(Case& cs) {
  Result res = {true, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  const uint64_t base = cs.img.base;
  Img u = cs.img, f = cs.img;
  {  // reference: GEMM then SOFTMAX
    IoWord* io = (IoWord*)u.b.data() - (base >> 4);
    const WPort* hp = (const WPort*)u.b.data() - (base >> 4);
    res.ru = fgemm::gemm_unit(cs.gemm, hp, hp, hp, hp, (const IoWord*)io, io);
    smx::Word* sio = (smx::Word*)nullptr;
    (void)sio;
    // the SOFTMAX unit takes ap_uint<128> words: run it on an ap_uint copy of the image, then copy back
    std::vector<smx::Word> aw(u.b.size() / 16);
    for (size_t i = 0; i < aw.size(); i++) {
      IoWord w;
      memcpy(&w, u.b.data() + 16 * i, 16);
      aw[i] = qks::io2a(w);
    }
    smx::Word* a = aw.data() - (base >> 4);
    res.ru |= smx::softmax_mode(cs.smx, a, a);
    for (size_t i = 0; i < aw.size(); i++) {
      const IoWord w = qks::a2io(aw[i]);
      memcpy(u.b.data() + 16 * i, &w, 16);
    }
    // restore the score scratch: the fused command does not have it
    memcpy(u.p(cs.s_addr), cs.img.p(cs.s_addr), cs.s_bytes);
  }
  {  // fused
    IoWord* io = (IoWord*)f.b.data() - (base >> 4);
    const WPort* hp = (const WPort*)f.b.data() - (base >> 4);
    res.rf = fgemm_qks::gemm_unit(cs.fused, hp, hp, hp, hp, (const IoWord*)io, io);
  }
  for (size_t j = 0; j < u.b.size(); j++)
    if (u.b[j] != f.b[j]) {
      if (!res.diff) res.first = j;
      res.diff++;
    }
  for (size_t j = 0; j < cs.s_bytes; j++)
    if (f.b[cs.s_addr - base + j] != cs.img.b[cs.s_addr - base + j]) res.swrites++;
  res.ok = res.ru == 0 && res.rf == 0 && res.diff == 0 && res.swrites == 0;
  p_stats(cs, f, res);
  cs.img = f;  // the fused result (for --vec)
  return res;
}

static Shape plain(uint32_t L, uint32_t K, uint32_t N, uint32_t B) {
  Shape s;
  memset(&s, 0, sizeof s);
  s.L = L;
  s.K = K;
  s.N = N;
  s.B = B;
  s.NP = N;
  s.p0 = 1.0f;
  s.qkind = 3;
  s.kkind = 3;
  s.skind = 1;
  return s;
}

static Shape random_shape() {
  static const uint32_t Ns[] = {1, 2, 3, 4, 31, 32, 33, 45, 48, 50, 63, 64, 65, 96, 100, 127, 128, 129, 135, 144,
                                150, 156, 192, 200, 255, 256, 257, 300};
  static const uint32_t Ks[] = {64, 64, 64, 64, 8, 16, 40, 72, 128, 136};
  static const float P0[] = {1.0f, 1.0f, 1.0f, 0.125f, 0.7f, 3.0f, 1e-3f};
  Shape s = plain(32 * (uint32_t)(1 + rnd(8)), Ks[rnd(10)], Ns[rnd(sizeof Ns / 4)], 1 + (uint32_t)rnd(3));
  s.NP = s.N + (rnd(3) == 0 ? (uint32_t)rnd(9) : 0);
  s.mask = rnd(2);
  s.p0 = P0[rnd(7)];
  s.qkind = (int)rnd(4);
  s.kkind = (int)rnd(4);
  s.skind = rnd(4) == 0 ? 0 : 1;
  s.qgap = (uint32_t)rnd(3);
  s.pgap = rnd(3) == 0 ? (uint32_t)rnd(3) : 0;
  s.sgap = rnd(4) == 0 ? (uint32_t)rnd(2) : 0;
  s.xs_gap = (uint32_t)rnd(3);
  s.rk_shared = rnd(3) == 0;
  return s;
}

// the -12392 list's four shapes (contract_plan.py), with their stacked-head Q and rank patterns
static Shape plan_shape(int i) {
  Shape s;
  if (i == 0) {  // vision: 24 items (12 heads x 2 images) of 256 x 256, no mask
    s = plain(256, 64, 256, 24);
  } else if (i == 1) {  // LM prefix: 3 heads x 48 rows (45 valid), 64 keys (45 valid), 5 items
    s = plain(64, 64, 144, 5);
    s.mask = true;
    s.site = 1;
    s.tqp = 48;
    s.tq = 45;
    s.npre = 45;
    s.rk_shared = true;
  } else if (i == 2) {  // expert self: 3 x 52 rows (50 valid), 128 keys: prefix 45 in [0, 64), suffix 50 in [64, 114)
    s = plain(128, 64, 156, 5);
    s.mask = true;
    s.site = 2;
    s.tqp = 52;
    s.tq = 50;
    s.npre = 45;
    s.rk_shared = true;
  } else {  // expert cross: 3 x 52 rows, 64 keys (45 valid)
    s = plain(64, 64, 156, 5);
    s.mask = true;
    s.site = 3;
    s.tqp = 52;
    s.tq = 50;
    s.npre = 45;
    s.rk_shared = true;
  }
  return s;
}

// --vec: one real attention item per file, written by policy_check.py (all little-endian):
//   u32 magic 0x514B5331, L, K, N, mask, NP; f32 p0;
//   i16 Q[N][K]; f32 s_tok[N]; i16 Kb[L][K]; f32 s_row[L]; i32 rank_q[N]; i32 rank_k[L] (if mask);
//   u16 P[N][L] and f32 scale[N] of the torch emulation (attn_int16: gemm_out + smx_emu.kernel)
static int run_vec(const char* path, uint64_t base) {
  FILE* fp = fopen(path, "rb");
  if (!fp) {
    printf("FAIL: cannot open %s\n", path);
    return 1;
  }
  uint32_t h[6];
  float p0;
  if (fread(h, 4, 6, fp) != 6 || fread(&p0, 4, 1, fp) != 1 || h[0] != 0x514B5331) {
    printf("FAIL: %s: bad header\n", path);
    return 1;
  }
  const uint32_t L = h[1], K = h[2], N = h[3], NP = h[5];
  const bool mask = h[4] != 0;
  std::vector<int16_t> Q((size_t)N * K), Kb((size_t)L * K);
  std::vector<float> st(N), sr(L), esc(N);
  std::vector<int32_t> rq(N), rk(L);
  std::vector<uint16_t> eP((size_t)N * L);
  bool ok = fread(Q.data(), 2, Q.size(), fp) == Q.size() && fread(st.data(), 4, N, fp) == N &&
            fread(Kb.data(), 2, Kb.size(), fp) == Kb.size() && fread(sr.data(), 4, L, fp) == L;
  if (mask) ok = ok && fread(rq.data(), 4, N, fp) == N && fread(rk.data(), 4, L, fp) == L;
  ok = ok && fread(eP.data(), 2, eP.size(), fp) == eP.size() && fread(esc.data(), 4, N, fp) == N;
  fclose(fp);
  if (!ok) {
    printf("FAIL: %s: short file\n", path);
    return 1;
  }
  Shape s = plain(L, K, N, 1);
  s.NP = NP;
  s.mask = mask;
  s.p0 = p0;
  Case cs;
  cs.name = path;
  if (!build(cs, s, base)) {
    printf("FAIL: %s: the command does not validate\n", path);
    return 1;
  }
  // overwrite the random operands with the real ones
  Img& m = cs.img;
  const ctr::Cmd& c = cs.fused;
  for (uint32_t n = 0; n < N; n++) {
    for (uint32_t k = 0; k < K; k++) m.set_i16(ctr::row_addr(c.s[ctr::SLOT_A], 0, n) + 2 * k, Q[(size_t)n * K + k]);
    m.set_f32(c.s[ctr::SLOT_G].addr + 4 * n, st[n]);
    if (mask) m.set_i32(c.s[ctr::SLOT_D].addr + 4 * n, rq[n]);
  }
  for (uint32_t n = N; n < ctr::pad4(N); n++) m.set_f32(c.s[ctr::SLOT_G].addr + 4 * n, 1.0f);
  for (uint32_t mm = 0; mm < L; mm++) {
    for (uint32_t k = 0; k < K; k++)
      m.set_i16(ctr::w_word_addr(c.s[ctr::SLOT_B], 0, mm, k, K) + 2 * ctr::w_lane(mm), Kb[(size_t)mm * K + k]);
    m.set_f32(c.s[ctr::SLOT_H].addr + 4 * mm, sr[mm]);
    if (mask) m.set_i32(c.s[ctr::SLOT_E].addr + 4 * mm, rk[mm]);
  }
  const Result r = run_case(cs);
  size_t pd = 0, sd = 0;
  for (uint32_t n = 0; n < N; n++) {
    for (uint32_t k = 0; k < L; k++)
      if ((uint16_t)cs.img.i16(ctr::row_addr(c.s[ctr::SLOT_C], 0, n) + 2 * k) != eP[(size_t)n * L + k]) pd++;
    const float v = cs.img.f32(c.s[ctr::SLOT_F].addr + 4 * n);
    if (memcmp(&v, &esc[n], 4)) sd++;
  }
  const bool pass = r.ok && pd == 0 && sd == 0;
  printf("%s: vec %s L=%u K=%u N=%u mask=%d: fused == unfused HLS (%zu bytes differ, %zu score-scratch writes), "
         "fused == torch emulation (%zu P, %zu scale mismatches)\n",
         pass ? "PASS" : "FAIL", path, L, K, N, (int)mask, r.diff, r.swrites, pd, sd);
  return pass ? 0 : 1;
}

static void report(const Case& cs, const Shape& s, const Result& r, const char* tag) {
  printf("%s %s: L=%u K=%u N=%u NP=%u B=%u mask=%d p0=%g ret %u/%u, %zu bytes differ (first at +0x%zx), %zu "
         "score-scratch writes; P: %ld rows, %ld all-zero, %ld with q=32767, %ld with >= 2 nonzero, %.1f nonzero/row\n",
         r.ok ? "ok  " : "FAIL", tag, s.L, s.K, s.N, s.NP, s.B, (int)s.mask, s.p0, r.ru, r.rf, r.diff, r.first,
         r.swrites, r.rows, r.zero_rows, r.max_rows, r.spread_rows, r.rows ? (double)r.nonzero / r.rows : 0.0);
  (void)cs;
}

int main(int argc, char** argv) {
  if (argc > 1 && !strcmp(argv[1], "--vec")) {
    int bad = 0;
    for (int i = 2; i < argc; i++) bad += run_vec(argv[i], 0x800000000ull);
    printf("%s: %d of %d real calls\n", bad ? "FAIL" : "PASS", argc - 2 - bad, argc - 2);
    return bad ? 1 : 0;
  }
  const int cases = argc > 1 ? atoi(argv[1]) : 200;
  const uint64_t seed = argc > 2 ? strtoull(argv[2], nullptr, 0) : 1;
  const uint64_t base = argc > 3 ? strtoull(argv[3], nullptr, 16) : 0x800000000ull;
  const bool plan = argc > 4 ? atoi(argv[4]) != 0 : true;
  rng.seed(seed);
  int pass = 0, fail = 0, tried = 0;
  long long rows = 0;
  int feat[8] = {0};
  long st_rows = 0, st_zero = 0, st_spread = 0;
  // the plan's shapes first
  for (int i = 0; plan && i < 4; i++) {
    Shape s = plan_shape(i);
    Case cs;
    if (!build(cs, s, base)) {
      printf("FAIL: plan shape %d does not validate\n", i);
      return 2;
    }
    const Result r = run_case(cs);
    static const char* nm[] = {"plan vision", "plan prefix", "plan expert_self", "plan expert_cross"};
    report(cs, s, r, nm[i]);
    (r.ok ? pass : fail)++;
    rows += (long long)s.N * s.B;
  }
  // adversarial: every tile-boundary row count at L = 32 and 256, NP > N, one-row items, fully masked items
  {
    static const uint32_t Nb[] = {1, 63, 64, 65, 127, 128, 129, 192, 193};
    for (uint32_t i = 0; i < sizeof Nb / 4; i++)
      for (uint32_t L : {32u, 256u}) {
        Shape s = plain(L, 64, Nb[i], 2);
        s.NP = Nb[i] + (i & 1 ? 3 : 0);
        s.mask = i % 3 != 0;
        s.rk_shared = i & 1;
        s.skind = i == 4 ? 0 : 1;
        Case cs;
        if (!build(cs, s, base)) continue;
        if (i == 5) {  // every row fully masked in item 1
          for (uint32_t n = 0; n < ctr::pad4(s.N); n++) cs.img.set_i32(cs.fused.s[ctr::SLOT_D].addr + cs.fused.s[ctr::SLOT_D].stride + 4 * n, -1);
        }
        const Result r = run_case(cs);
        report(cs, s, r, "edge");
        (r.ok ? pass : fail)++;
        rows += (long long)s.N * s.B;
      }
  }
  for (int i = 0; pass + fail < cases + 4 + 18; i++) {
    tried++;
    Shape s = random_shape();
    Case cs;
    if (!build(cs, s, base)) continue;
    const Result r = run_case(cs);
    if (!r.ok) report(cs, s, r, "random");
    (r.ok ? pass : fail)++;
    rows += (long long)s.N * s.B;
    st_rows += r.rows;
    st_zero += r.zero_rows;
    st_spread += r.spread_rows;
    feat[0] += s.B > 1;
    feat[1] += s.mask;
    feat[2] += s.NP != s.N;
    feat[3] += s.N % 64 != 0;
    feat[4] += s.pgap != 0;
    feat[5] += s.rk_shared && s.mask;
    feat[6] += s.p0 != 1.0f;
    feat[7] += s.skind == 0;
    if (fail > 5) break;
  }
  printf("%s: tb_qks %d of %d commands: fused == GEMM + SOFTMAX (whole image, base 0x%llx, %lld rows); random: "
         "items %d, mask %d, NP>N %d, N%%64 %d, P pitch gap %d, shared rank_k %d, p0!=1 %d, wide scales %d (%d drawn)\n",
         fail ? "FAIL" : "PASS", pass, pass + fail, (unsigned long long)base, rows, feat[0], feat[1], feat[2], feat[3],
         feat[4], feat[5], feat[6], feat[7], tried);
  printf("random P rows: %ld, all-zero %ld, >= 2 nonzero %ld\n", st_rows, st_zero, st_spread);
  return fail ? 1 : 0;
}
