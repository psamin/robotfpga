// Host testbench of hp_reads' row unit part (hpr_quant.h): hpq::row_mode_hpr against the integrator's unfused
// qz::row_mode (fpga/kernels/quantize/quant_mode.h), memory image against memory image, word for word.
//   sh run_tests.sh quant [--plan list.bin] [--random N]       (DFSIM=1: the threaded bounded-FIFO model)
// Every case runs three ways on copies of one memory image (NaN-magic outside the inputs):
//   ref    qz::row_mode(c, io, io)                                  the contract's unfused 4-lane mode
//   off    hpq::row_mode_hpr(c, 0, nullptr, io, io)                 HPR off: io only (hp1 is a null pointer)
//   hpr    hpq::row_mode_hpr(c, HPR_A_HP1, hp1view, io, io)         HPR on: slot A split over io and hp1
// hp1view is a separate copy of memory that holds only slot A's extent (every other word NaN-magic): a value read
// over hp1 from anywhere else would reach an output. Both must equal ref in every word (outputs, gaps, guards).
// ref itself is checked against the independent ctr::ref image (CONTRACT.md 1.1 formulas) as tb_quant does.
// Sections: quant_elem_g1 == quant_elem_recip; edge commands; random contract-valid commands; the -12392 list.
// Helpers (Mem, gen_val, extents, random_cmd, mk, reference) are copies of fpga/kernels/quantize/tb_quant.cpp's.
#include <cfenv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "hpr_quant.h"

using namespace qz;

#ifdef HQ_DFSIM
int qz_dfsim_depth(const std::string& n) {
  static const std::map<std::string, int> d = {
      {"qz_xs", QZ_D_XS}, {"qz_am", QZ_D_AM}, {"qz_d", QZ_D_D},   {"qz_ss", QZ_D_SS}, {"qz_sf", QZ_D_SF},
      {"qz_ow", QZ_D_OW}, {"qz_sq", QZ_D_SQ}, {"qz_hd", QZ_D_HD}, {"qz_dd", QZ_D_DD},
      {"hq_xs0", HQ_D_XS}, {"hq_xs1", HQ_D_XS}, {"hq_am", QZ_D_AM}, {"hq_d0", HQ_D_D0}, {"hq_d1", HQ_D_D1},
      {"hq_ss", QZ_D_SS}, {"hq_sf", QZ_D_SF}, {"hq_ow0", HQ_D_OW}, {"hq_ow1", HQ_D_OW}, {"hq_owm", HQ_D_OM},
      {"anon", HQ_D_RQ},  // the 8 row FIFOs (an array of streams has no names)
      {"hq_sq", QZ_D_SQ}, {"hq_hd", QZ_D_HD}, {"hq_dd", QZ_D_DD}};
  auto it = d.find(n);
  if (it == d.end()) {
    printf("dfsim: stream %s has no depth\n", n.c_str());
    exit(8);
  }
  return it->second;
}
#endif

static long g_fail = 0;
#define CHECK(cond, ...)                            \
  do {                                              \
    if (!(cond)) {                                  \
      if (g_fail < 30) {                            \
        printf("FAIL %s:%d: ", __FILE__, __LINE__); \
        printf(__VA_ARGS__);                        \
        printf("\n");                               \
      }                                             \
      g_fail++;                                     \
    }                                               \
  } while (0)

// ---------------------------------------------------------------- copies of tb_quant.cpp's helpers
static const uint32_t MAGIC = 0xFFC0A5A5u;
struct Mem {
  uint64_t base;
  std::vector<word_t> v;
  Mem(uint64_t b, size_t bytes) : base(b), v((bytes + 15) / 16) {
    for (auto& w : v)
      for (int j = 0; j < 4; j++) w(32 * j + 31, 32 * j) = MAGIC;
  }
  word_t* ptr() { return reinterpret_cast<word_t*>(reinterpret_cast<uintptr_t>(v.data()) - (uintptr_t)(base >> 4) * 16); }
  word_t& w(uint64_t addr) { return v[(addr - base) >> 4]; }
  void set_f32(uint64_t a, uint32_t bits) { w(a)(32 * ((a >> 2) & 3) + 31, 32 * ((a >> 2) & 3)) = bits; }
  void set_i16(uint64_t a, int16_t q) { w(a)(16 * ((a >> 1) & 7) + 15, 16 * ((a >> 1) & 7)) = (uint16_t)q; }
};

static float gen_val(std::mt19937_64& rng, int cls, float mul) {
  std::normal_distribution<float> nd(0.f, 1.f);
  std::student_t_distribution<float> td(2.f);
  switch (cls) {
    case 0: return nd(rng) * mul;
    case 1: return td(rng) * mul;
    case 2: return (rng() & 1) ? -0.f : 0.f;
    case 3: return nd(rng) * 1e-36f;
    case 4: return u2f((uint32_t)(rng() & 0x807FFFFFu));
    case 5: return (rng() % 64 == 0) ? 1e36f * nd(rng) : nd(rng);
    default: {
      const int k = (int)(rng() % 32767);
      return (float)(k + 0.5) * ((rng() & 1) ? 1.f : -1.f);
    }
  }
}

static void reference(const ctr::Cmd& c, const std::vector<float>& x, Mem& e) {
  const uint32_t B = c.B, N = c.N, K = c.K, sg = c.p[ctr::P_SEG];
  const uint32_t NP = c.p[ctr::P_NP] ? c.p[ctr::P_NP] : N;
  const int sh = (int32_t)c.p[ctr::P_SHIFT];
  const ctr::Slot &C = c.s[ctr::SLOT_C], &F = c.s[ctr::SLOT_F];
  const uint64_t segc = c.p[ctr::P_SEGC], segf = c.p[ctr::P_SEGF];
  auto X = [&](uint32_t b, uint32_t n, uint32_t k) { return x[((size_t)b * N + n) * K + k]; };
  if (c.op == ctr::OP_COPY) {
    for (uint32_t b = 0; b < B; b++)
      for (uint32_t n = 0; n < N; n++)
        for (uint32_t k = 0; k < K; k++) e.set_f32(ctr::row_addr(C, b, n) + 4 * k, f2u(X(b, n, k)));
    return;
  }
  if (c.op == ctr::OP_QUANT_WT) {
    const uint32_t SEG = sg ? sg : K;
    std::vector<float> col(N);
    std::vector<int16_t> q(N);
    for (uint32_t b = 0; b < B; b++)
      for (uint32_t ch = 0; ch < K; ch++) {
        const uint32_t j = ch / SEG, cc = ch % SEG;
        for (uint32_t n = 0; n < N; n++) col[n] = X(b, n, ch);
        const float s = ctr::ref::quant_row(col.data(), N, q.data(), sh);
        ctr::Slot w = C;
        w.addr += j * segc;
        for (uint32_t n = 0; n < NP; n++) e.set_i16(ctr::w_word_addr(w, b, cc, n, NP) + 2 * ctr::w_lane(cc), n < N ? q[n] : 0);
        e.set_f32(F.addr + (uint64_t)b * F.stride + j * segf + 4 * cc, f2u(s));
      }
    return;
  }
  const bool kw = c.op == ctr::OP_QUANT_W;
  const uint32_t SEG = sg ? sg : K, nseg = K / SEG;
  std::vector<int16_t> q(SEG);
  for (uint32_t b = 0; b < B; b++)
    for (uint32_t j = 0; j < nseg; j++)
      for (uint32_t n = 0; n < ((NP + 3) & ~3u); n++) {
        float s = 1.0f;
        if (n < N) {
          s = ctr::ref::quant_row(&x[((size_t)b * N + n) * K + j * SEG], SEG, q.data(), sh);
        } else {
          for (uint32_t k = 0; k < SEG; k++) q[k] = 0;
        }
        e.set_f32(F.addr + (uint64_t)b * F.stride + j * segf + 4 * n, f2u(s));
        if (n >= NP) continue;
        for (uint32_t k = 0; k < SEG; k++) {
          uint64_t a;
          if (kw) {
            ctr::Slot w = C;
            w.addr += j * segc;
            a = ctr::w_word_addr(w, b, n, k, SEG) + 2 * ctr::w_lane(n);
          } else {
            a = ctr::seg_addr(C, b, n, j, segc) + 2 * k;
          }
          e.set_i16(a, q[k]);
        }
      }
}

struct Ext {
  uint64_t a, c, f;
};
static Ext extents(const ctr::Cmd& c) {
  const uint64_t B = c.B, N = c.N, K = c.K, sg = c.p[ctr::P_SEG];
  const uint64_t NP = c.p[ctr::P_NP] ? c.p[ctr::P_NP] : N, SEG = sg ? sg : K, nseg = K / SEG;
  const uint64_t segc = c.p[ctr::P_SEGC], segf = c.p[ctr::P_SEGF];
  const ctr::Slot &A = c.s[ctr::SLOT_A], &C = c.s[ctr::SLOT_C], &F = c.s[ctr::SLOT_F];
  Ext x;
  x.a = (B - 1) * A.stride + (N - 1) * A.pitch + 4 * K;
  if (c.op == ctr::OP_COPY) {
    x.c = (B - 1) * C.stride + (N - 1) * C.pitch + 4 * K;
    x.f = 0;
  } else if (c.op == ctr::OP_QUANT_W) {
    x.c = (B - 1) * C.stride + (nseg - 1) * segc + 3 * C.pitch + (NP / 32) * SEG * 16;
    x.f = (B - 1) * F.stride + (nseg - 1) * segf + 4 * NP;
  } else if (c.op == ctr::OP_QUANT_WT) {
    const uint64_t ns = sg ? K / 64 : 1;
    x.c = (B - 1) * C.stride + (ns - 1) * segc + 3 * C.pitch + (SEG / 32) * NP * 16;
    x.f = (B - 1) * F.stride + (ns - 1) * segf + 4 * SEG;
  } else {
    const uint64_t row = nseg > 1 ? 2 * SEG : 2 * K;
    x.c = (B - 1) * C.stride + (nseg - 1) * segc + (NP - 1) * C.pitch + row;
    x.f = (B - 1) * F.stride + (nseg - 1) * segf + 4 * ((NP + 3) & ~3ull);
  }
  return x;
}

static bool random_cmd(std::mt19937_64& rng, uint32_t op, ctr::Cmd& c) {
  memset(&c, 0, sizeof c);
  c.op = op;
  auto r = [&](uint64_t n) { return (uint32_t)(rng() % n); };
  c.B = 1 + (r(3) == 0 ? r(4) : 0);
  c.N = r(4) == 0 ? 1 + r(8) : 1 + r(r(2) ? 80 : 300);
  const uint32_t Ks[] = {8, 16, 24, 32, 40, 56, 64, 96, 128, 192, 320, 520, 720, 736, 768, 960, 1040, 2560, 3072, 4104};
  if (op == ctr::OP_QUANT_X || op == ctr::OP_COPY) {
    c.K = Ks[r(20)];
    if (r(40) == 0) c.K = 12288 + 8 * r(3);
  } else {
    c.K = 64 * (1 + r(r(2) ? 5 : 13));
  }
  if (op == ctr::OP_QUANT_WT && r(3) == 0) c.K = 32 * (1 + r(20));
  const uint32_t K = c.K;
  bool seg = false;
  if (op == ctr::OP_QUANT_X) seg = K % 64 == 0 && K / 64 <= 64 && r(2);
  if (op == ctr::OP_QUANT_W) seg = true;
  if (op == ctr::OP_QUANT_WT) seg = K % 64 == 0 && r(3) != 0;
  if (op == ctr::OP_QUANT_W && K == 64 && r(2)) seg = false;
  c.p[ctr::P_SEG] = seg ? 64 : 0;
  const uint32_t SEG = seg ? 64 : K, nseg = K / SEG;
  uint32_t NP = c.N;
  if (op == ctr::OP_QUANT_W) NP = ctr::pad32(c.N) + 32 * r(2);
  else if (op == ctr::OP_QUANT_WT) NP = ctr::pad8(c.N) + 8 * r(3);
  else if (op == ctr::OP_QUANT_X && r(3) == 0) NP = c.N + r(9);
  if (NP > ctr::N_MAX) NP = ctr::N_MAX;
  if (NP != c.N || r(2)) c.p[ctr::P_NP] = (op == ctr::OP_COPY) ? 0 : NP;
  if (op != ctr::OP_COPY) c.p[ctr::P_SHIFT] = r(3) == 0 ? (uint32_t)((int)r(17) - 8) : 0;
  c.flags = op == ctr::OP_QUANT_X ? ctr::F_QOUT : 0;
  ctr::Slot& A = c.s[ctr::SLOT_A];
  A.pitch = 4 * K + 16 * (r(3) ? 0 : 1 + r(8));
  if (c.N == 1 && r(2)) A.pitch = 0;
  A.stride = c.B > 1 ? (uint32_t)(c.N * (uint64_t)(A.pitch ? A.pitch : 4 * K) + 16 * r(5)) : 0;
  if (c.B > 1 && r(5) == 0) A.stride = 0;
  ctr::Slot& C = c.s[ctr::SLOT_C];
  ctr::Slot& F = c.s[ctr::SLOT_F];
  uint64_t citem = 0, fitem = 0;
  if (op == ctr::OP_COPY) {
    C.pitch = 4 * K + 16 * r(3);
    citem = (uint64_t)c.N * C.pitch;
  } else if (op == ctr::OP_QUANT_X) {
    if (nseg > 1 && r(2)) {
      C.pitch = 128;
      c.p[ctr::P_SEGC] = NP * 128 + 16 * r(3);
      citem = (uint64_t)nseg * c.p[ctr::P_SEGC];
    } else if (nseg > 1) {
      C.pitch = 2 * K + 16 * r(3);
      c.p[ctr::P_SEGC] = 128;
      citem = (uint64_t)NP * C.pitch;
    } else {
      C.pitch = 2 * K + 16 * r(3);
      if (NP == 1 && r(2)) C.pitch = 0;
      citem = (uint64_t)NP * (C.pitch ? C.pitch : 2 * K) + 16;
    }
  } else if (op == ctr::OP_QUANT_W) {
    const uint32_t port = (NP / 32) * SEG * 16;
    C.pitch = port + 16 * r(3);
    if (nseg > 1) c.p[ctr::P_SEGC] = 4 * C.pitch + 16 * r(2);
    citem = (uint64_t)(nseg > 1 ? nseg * c.p[ctr::P_SEGC] : 4 * C.pitch);
  } else {
    const uint32_t port = (SEG / 32) * NP * 16;
    C.pitch = port + 16 * r(3);
    if (nseg > 1) c.p[ctr::P_SEGC] = 4 * C.pitch + 16 * r(2);
    citem = (uint64_t)(nseg > 1 ? nseg * c.p[ctr::P_SEGC] : 4 * C.pitch);
  }
  C.stride = c.B > 1 ? (uint32_t)(citem + 16 * r(4)) : 0;
  if (op != ctr::OP_COPY) {
    const uint32_t nsc = op == ctr::OP_QUANT_WT ? SEG : NP;
    if (nseg > 1) c.p[ctr::P_SEGF] = 4 * ctr::pad4(nsc) + 16 * r(3);
    fitem = nseg > 1 ? (uint64_t)nseg * c.p[ctr::P_SEGF] : 4 * ctr::pad4(nsc);
    F.stride = c.B > 1 ? (uint32_t)(fitem + 16 * r(3)) : 0;
  }
  A.addr = 0x10000;
  C.addr = 0x10000000;
  if (op != ctr::OP_COPY) F.addr = 0x20000000;
  return ctr::validate(c) == ctr::ME_OK;
}

static ctr::Cmd mk(uint32_t op, uint32_t B, uint32_t N, uint32_t K, uint32_t seg, uint32_t NP, int shift) {
  ctr::Cmd c;
  memset(&c, 0, sizeof c);
  c.op = op;
  c.flags = op == ctr::OP_QUANT_X ? ctr::F_QOUT : 0;
  c.B = B;
  c.N = N;
  c.K = K;
  c.p[ctr::P_SEG] = seg;
  c.p[ctr::P_NP] = (op == ctr::OP_COPY) ? 0 : NP;
  c.p[ctr::P_SHIFT] = (uint32_t)shift;
  const uint32_t SEG = seg ? seg : K, nseg = K / SEG, np = NP ? NP : N;
  ctr::Slot &A = c.s[ctr::SLOT_A], &C = c.s[ctr::SLOT_C], &F = c.s[ctr::SLOT_F];
  A.pitch = 4 * K;
  A.stride = B > 1 ? N * A.pitch : 0;
  uint64_t citem;
  if (op == ctr::OP_COPY) {
    C.pitch = 4 * K;
    citem = (uint64_t)N * C.pitch;
  } else if (op == ctr::OP_QUANT_X) {
    if (nseg > 1) {
      C.pitch = 128;
      c.p[ctr::P_SEGC] = np * 128;
      citem = (uint64_t)nseg * np * 128;
    } else {
      C.pitch = 2 * K;
      citem = (uint64_t)np * 2 * K;
    }
  } else {
    const uint32_t port = op == ctr::OP_QUANT_W ? (np / 32) * SEG * 16 : (SEG / 32) * np * 16;
    C.pitch = port;
    if (nseg > 1) c.p[ctr::P_SEGC] = 4 * port;
    citem = 4ull * port * nseg;
  }
  C.stride = B > 1 ? (uint32_t)citem : 0;
  if (op != ctr::OP_COPY) {
    const uint32_t nsc = op == ctr::OP_QUANT_WT ? SEG : np;
    if (nseg > 1) c.p[ctr::P_SEGF] = 4 * ctr::pad4(nsc);
    F.stride = B > 1 ? nseg * 4 * ctr::pad4(nsc) : 0;
  }
  A.addr = 0x10000;
  C.addr = 0x10000000;
  if (op != ctr::OP_COPY) F.addr = 0x20000000;
  return c;
}

// ---------------------------------------------------------------- the three-way comparison
struct Stats {
  long cases = 0, hpr_active = 0, values = 0, words = 0, bad_ref = 0, bad_off = 0, bad_hpr = 0;
  std::map<std::string, long> by;
};
static Stats g_st;

static long diff(const Mem& a, const Mem& b, uint64_t* first) {
  long bad = 0;
  for (size_t i = 0; i < a.v.size(); i++)
    if (a.v[i] != b.v[i]) {
      if (!bad && first) *first = a.base + 16 * i;
      bad++;
    }
  return bad;
}

static bool run3(const char* name, ctr::Cmd c, uint64_t base, std::mt19937_64& rng, int cls_mix, bool verbose = false) {
  const Ext x = extents(c);
  const uint64_t g = 256;
  const uint64_t a0 = base + g, c0 = (a0 + x.a + g + 4095) & ~4095ull, f0 = (c0 + x.c + g + 255) & ~255ull;
  const uint64_t end = f0 + x.f + g;
  c.s[ctr::SLOT_A].addr = a0;
  c.s[ctr::SLOT_C].addr = c0;
  if (c.op != ctr::OP_COPY) c.s[ctr::SLOT_F].addr = f0;
  // through the 256-byte record with field 56 and back (what the sequencer hands the unit)
  uint32_t f[ctr::REC_FIELDS];
  hpr::encode_hpr(c, hpr::HPR_A_HP1, f);
  ctr::Cmd d;
  CHECK(ctr::decode(f, d) == ctr::ST_END, "%s: decode", name);
  const uint32_t hprf = f[hpr::FLD_HPR];
  CHECK(ctr::validate(d) == ctr::ME_OK && hpr::validate_hpr(d, hprf) == ctr::ME_OK, "%s: validate %u / hpr %u", name,
        ctr::validate(d), hpr::validate_hpr(d, hprf));
  Mem m(base, end - base);
  std::vector<float> v((size_t)d.B * d.N * d.K);
  for (uint32_t b = 0; b < d.B; b++)
    for (uint32_t n = 0; n < d.N; n++) {
      const int cls = cls_mix ? (int)(rng() % 7) : 0;
      const float mul = std::exp2f((float)((int)(rng() % 40) - 20));
      for (uint32_t k = 0; k < d.K; k++)
        v[((size_t)b * d.N + n) * d.K + k] =
            (b && d.s[ctr::SLOT_A].stride == 0) ? v[(size_t)n * d.K + k] : gen_val(rng, cls, mul);
    }
  const ctr::Slot& A = d.s[ctr::SLOT_A];
  Mem hv(base, end - base);  // the hp1 view: slot A's extent only
  for (uint32_t b = 0; b < d.B; b++)
    for (uint32_t n = 0; n < d.N; n++)
      for (uint32_t k = 0; k < d.K; k++) {
        const uint64_t a = ctr::row_addr(A, b, n) + 4 * k;
        m.set_f32(a, f2u(v[((size_t)b * d.N + n) * d.K + k]));
        hv.set_f32(a, f2u(v[((size_t)b * d.N + n) * d.K + k]));
      }
  Mem e = m, r = m, mo = m, mh = m;
  reference(d, v, r);
  CHECK(row_mode(d, e.ptr(), e.ptr()) == ctr::ME_OK, "%s: qz::row_mode", name);
  CHECK(hpq::row_mode_hpr(d, 0, nullptr, mo.ptr(), mo.ptr()) == ctr::ME_OK, "%s: row_mode_hpr off", name);
  CHECK(hpq::row_mode_hpr(d, hprf, hv.ptr(), mh.ptr(), mh.ptr()) == ctr::ME_OK, "%s: row_mode_hpr", name);
  hpq::HP hp;
  hpq::plan(d, hprf, hp);
  uint64_t fr = 0, fo = 0, fh = 0;
  const long br = diff(e, r, &fr), bo = diff(mo, e, &fo), bh = diff(mh, e, &fh);
  g_st.cases++;
  g_st.hpr_active += hp.on;
  g_st.values += (long)d.B * d.N * d.K;
  g_st.words += (long)m.v.size();
  g_st.bad_ref += br;
  g_st.bad_off += bo;
  g_st.bad_hpr += bh;
  g_st.by[name]++;
  CHECK(br == 0, "%s: qz::row_mode vs ctr::ref: %ld words differ (first 0x%llx)", name, br, (unsigned long long)fr);
  CHECK(bo == 0, "%s: HPR off vs qz::row_mode: %ld words differ (first 0x%llx)", name, bo, (unsigned long long)fo);
  CHECK(bh == 0, "%s: HPR on vs qz::row_mode: %ld words differ (first 0x%llx)", name, bh, (unsigned long long)fh);
  if (verbose || br || bo || bh)
    printf("  %-10s op %02x B %5u N %5u K %5u p1 %2u NP %5u sh %2d hpr %d: words %zu, differ ref %ld off %ld hpr %ld\n",
           name, d.op, d.B, d.N, d.K, d.p[ctr::P_SEG], d.p[ctr::P_NP], (int32_t)d.p[ctr::P_SHIFT], (int)hp.on,
           m.v.size(), br, bo, bh);
  return br == 0 && bo == 0 && bh == 0;
}

// ---------------------------------------------------------------- sections
static void test_g1_lanes(long rows) {
  std::mt19937_64 rng(12345);
  std::uniform_real_distribution<float> ud(-1.f, 1.f);
  long n = 0, f0 = g_fail;
  for (long r = 0; r < rows; r++) {
    uint32_t ab;
    const int kind = r % 3;
    if (kind == 0) ab = (uint32_t)(rng() % 0x7F000000u);
    else if (kind == 1) ab = (uint32_t)((100u + rng() % 60) << 23) | (uint32_t)(rng() & 0x7FFFFF);
    else ab = TINY_BITS - 1000 + (uint32_t)(rng() % 3000);
    const RowScale rs = row_scale(abits_t(ab));
    const float a = u2f(ab), s = u2f((uint32_t)rs.s_bits);
    auto chk = [&](float xv) {
      CHECK((int)hpq::quant_elem_g1(fbits_t(f2u(xv)), rs) == (int)quant_elem_recip(fbits_t(f2u(xv)), rs), "g1 x=%a", xv);
      n++;
    };
    chk(a);
    chk(-a);
    chk(0.f);
    chk(u2f(1));
    for (int i = 0; i < 300; i++) chk(ud(rng) * a);
    for (int i = 0; i < 300; i++) {
      const int k = (int)(rng() % 32768);
      chk((float)((k + ((i & 1) ? 0.5 : 0.0)) * (double)s));
    }
  }
  printf("quant_elem_g1 (lanes 4..7, %s) == quant_elem_recip: %ld elements, %ld failures\n",
         HPR_G1_DSP ? "DSP" : "fabric", n, g_fail - f0);
}

static void test_edges(std::mt19937_64& rng) {
  long f0 = g_fail;
  struct E {
    const char* nm;
    uint32_t op, B, N, K, seg, NP;
    int sh;
  } es[] = {
      {"X.k8", ctr::OP_QUANT_X, 1, 3, 8, 0, 0, 0},           {"X.k32.n50", ctr::OP_QUANT_X, 1, 50, 32, 0, 0, 0},
      {"X.k64", ctr::OP_QUANT_X, 1, 7, 64, 0, 0, 0},         {"X.k96", ctr::OP_QUANT_X, 2, 5, 96, 0, 0, 1},
      {"X.k720", ctr::OP_QUANT_X, 1, 50, 720, 0, 0, 0},      {"X.k736", ctr::OP_QUANT_X, 1, 50, 736, 0, 0, 0},
      {"X.k960.np48", ctr::OP_QUANT_X, 1, 45, 960, 0, 48, 0}, {"X.k3072", ctr::OP_QUANT_X, 1, 9, 3072, 0, 0, 0},
      {"X.k3080", ctr::OP_QUANT_X, 1, 5, 3080, 0, 0, 0},     {"X.k12288", ctr::OP_QUANT_X, 1, 32, 12288, 0, 0, 0},
      {"X.k16384", ctr::OP_QUANT_X, 1, 3, 16384, 0, 0, 0},   {"X.q.vis", ctr::OP_QUANT_X, 1, 64, 768, 64, 0, -3},
      {"X.q.np", ctr::OP_QUANT_X, 2, 45, 960, 64, 48, -3},   {"X.n1", ctr::OP_QUANT_X, 3, 1, 64, 0, 0, 8},
      {"X.np29", ctr::OP_QUANT_X, 1, 29, 64, 64, 0, 0},      {"W.k64", ctr::OP_QUANT_W, 1, 50, 64, 0, 64, 0},
      {"W.k320", ctr::OP_QUANT_W, 1, 45, 320, 64, 64, 0},    {"W.n512", ctr::OP_QUANT_W, 1, 512, 768, 64, 512, 0},
      {"W.b3", ctr::OP_QUANT_W, 3, 7, 128, 64, 32, 5},       {"WT.k320", ctr::OP_QUANT_WT, 1, 50, 320, 64, 64, 0},
      {"WT.vis", ctr::OP_QUANT_WT, 2, 256, 768, 64, 256, 0}, {"WT.k32", ctr::OP_QUANT_WT, 1, 9, 32, 0, 16, 0},
      {"WT.k96", ctr::OP_QUANT_WT, 2, 13, 96, 0, 24, -2},    {"WT.k160", ctr::OP_QUANT_WT, 1, 21, 160, 0, 24, 0},
      {"WT.n1", ctr::OP_QUANT_WT, 1, 1, 64, 64, 8, 0},       {"C.k8", ctr::OP_COPY, 1, 4, 8, 0, 0, 0},
      {"C.k3072", ctr::OP_COPY, 8, 4, 3072, 0, 0, 0},        {"C.k720", ctr::OP_COPY, 1, 12, 720, 0, 0, 0},
      {"C.k72", ctr::OP_COPY, 1, 5, 72, 0, 0, 0},
  };
  for (const auto& e : es) {
    const ctr::Cmd c = mk(e.op, e.B, e.N, e.K, e.seg, e.NP, e.sh);
    CHECK(ctr::validate(c) == ctr::ME_OK, "edge %s invalid (%u)", e.nm, ctr::validate(c));
    run3(e.nm, c, 0x8ull << 32, rng, 1, true);
  }
  printf("edge commands: %zu, %ld failures\n", sizeof es / sizeof es[0], g_fail - f0);
}

static void test_random(int n, std::mt19937_64& rng) {
  const uint32_t ops[] = {ctr::OP_QUANT_X, ctr::OP_QUANT_W, ctr::OP_QUANT_WT, ctr::OP_COPY};
  const char* names[] = {"rnd.X", "rnd.W", "rnd.WT", "rnd.COPY"};
  long f0 = g_fail;
  int done = 0, tried = 0;
  const uint64_t bases[] = {0x0ull, 0x8ull << 32, 0x7FFF0000ull, 0xFF00000000ull, 0x123450000ull};
  while (done < n) {
    tried++;
    const int oi = (int)(rng() % 4);
    ctr::Cmd c;
    if (!random_cmd(rng, ops[oi], c)) continue;
    run3(names[oi], c, bases[done % 5], rng, 1);
    done++;
  }
  printf("random commands: %d run (%d generated), %ld failures\n", done, tried, g_fail - f0);
}

static void test_plan(const char* path, bool all) {
  FILE* fp = fopen(path, "rb");
  if (!fp) {
    printf("cannot open %s\n", path);
    g_fail++;
    return;
  }
  std::vector<uint8_t> buf;
  uint8_t tmp[65536];
  size_t got;
  while ((got = fread(tmp, 1, sizeof tmp, fp)) > 0) buf.insert(buf.end(), tmp, tmp + got);
  fclose(fp);
  const size_t nrec = buf.size() / ctr::REC_BYTES;
  std::mt19937_64 rng(2024);
  std::map<std::string, int> seen;
  std::map<uint32_t, int> nrun, ntot;
  long f0 = g_fail;
  for (size_t i = 0; i < nrec; i++) {
    uint32_t f[ctr::REC_FIELDS];
    memcpy(f, &buf[i * ctr::REC_BYTES], ctr::REC_BYTES);
    ctr::Cmd c;
    if (ctr::decode(f, c) != ctr::ST_END) {
      CHECK(false, "plan record %zu does not decode", i);
      continue;
    }
    if (c.op != ctr::OP_QUANT_X && c.op != ctr::OP_QUANT_W && c.op != ctr::OP_QUANT_WT && c.op != ctr::OP_COPY) continue;
    ntot[c.op]++;
    char key[256];
    snprintf(key, sizeof key, "%u %u %u %u %u %u %u %u %u %u %u %u %u %u %u", c.op, c.flags, c.K, c.N, c.B,
             c.s[0].pitch, c.s[0].stride, c.s[2].pitch, c.s[2].stride, c.s[5].stride, c.p[1], c.p[2], c.p[3], c.p[4],
             c.p[5]);
    if (!all && seen.count(key)) continue;
    seen[key]++;
    nrun[c.op]++;
    const char* nm = c.op == ctr::OP_QUANT_X ? "plan.X" : c.op == ctr::OP_QUANT_W ? "plan.W" : c.op == ctr::OP_QUANT_WT ? "plan.WT" : "plan.COPY";
    run3(nm, c, (i & 1) ? (0x8ull << 32) : 0x40000000ull, rng, 1, false);
  }
  printf("plan %s: %zu records; QUANT_X %d, QUANT_W %d, QUANT_WT %d, COPY %d; run %s: %d / %d / %d / %d, %ld failures\n",
         path, nrec, ntot[ctr::OP_QUANT_X], ntot[ctr::OP_QUANT_W], ntot[ctr::OP_QUANT_WT], ntot[ctr::OP_COPY],
         all ? "all" : "each distinct shape once", nrun[ctr::OP_QUANT_X], nrun[ctr::OP_QUANT_W],
         nrun[ctr::OP_QUANT_WT], nrun[ctr::OP_COPY], g_fail - f0);
}

static void test_validate() {
  long f0 = g_fail;
  ctr::Cmd c = mk(ctr::OP_QUANT_X, 1, 4, 64, 0, 0, 0);
  CHECK(hpr::validate_hpr(c, 0) == ctr::ME_OK, "hpr 0");
  CHECK(hpr::validate_hpr(c, hpr::HPR_A_HP1) == ctr::ME_OK, "hpr hp1");
  CHECK(hpr::validate_hpr(c, 1) == ctr::ME_SLOT, "hpr hp0 must be rejected");
  CHECK(hpr::validate_hpr(c, 0x6) == ctr::ME_SLOT, "hpr hp1|hp2 not built");
  CHECK(hpr::validate_hpr(c, 0x20) == ctr::ME_SLOT, "B ports reserved");
  c.op = ctr::OP_NORM;
  CHECK(hpr::validate_hpr(c, hpr::HPR_A_HP1) == ctr::ME_SLOT, "NORM not built");
  // a 1.0 decoder ignores field 56: the record decodes to the same Cmd as without it
  ctr::Cmd a = mk(ctr::OP_QUANT_X, 1, 4, 64, 0, 0, 0), x, y;
  uint32_t f0r[ctr::REC_FIELDS], f1r[ctr::REC_FIELDS];
  ctr::encode(a, f0r);
  hpr::encode_hpr(a, hpr::HPR_A_HP1, f1r);
  CHECK(ctr::decode(f0r, x) == ctr::ST_END && ctr::decode(f1r, y) == ctr::ST_END, "decode");
  CHECK(memcmp(&x, &y, sizeof x) == 0 && ctr::validate(y) == ctr::ME_OK, "contract 1.0 ignores field 56");
  printf("FLD_HPR rules: %ld failures\n", g_fail - f0);
}

int main(int argc, char** argv) {
  fesetround(FE_TONEAREST);
  bool quick = false, plan_all = false;
  int nrand = 600;
  std::vector<const char*> plans;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--quick")) quick = true;
    else if (!strcmp(argv[i], "--random") && i + 1 < argc) nrand = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--plan") && i + 1 < argc) plans.push_back(argv[++i]);
    else if (!strcmp(argv[i], "--plan-all")) plan_all = true;
  }
#ifdef HQ_DFSIM
  printf("threaded bounded-FIFO model (HQ_DFSIM): every process a thread, every stream at its pragma depth\n");
#else
  test_g1_lanes(quick ? 2000 : 100000);
#endif
  test_validate();
  std::mt19937_64 rng(77);
  test_edges(rng);
  test_random(nrand, rng);
  for (const char* p : plans) test_plan(p, plan_all);
  printf("cases %ld (HPR active in %ld), %ld input values, %ld memory words x 3 compared; words differing: qz vs ref %ld, "
         "off vs qz %ld, hpr vs qz %ld\n",
         g_st.cases, g_st.hpr_active, g_st.values, g_st.words, g_st.bad_ref, g_st.bad_off, g_st.bad_hpr);
  printf(g_fail ? "TB FAILED: %ld failures\n" : "TB PASSED\n", g_fail);
  return g_fail ? 1 : 0;
}
