// Host testbench of hp_reads' softmax part (hpr_softmax.h): hps::softmax_mode_hpr against the integrator's
// unfused smx::softmax_mode (fpga/kernels/softmax/softmax_core.h), memory image against memory image.
//   sh run_tests.sh smx [--plan list.bin] [--random N]        (DFSIM=1: the threaded bounded-FIFO model)
// Every case runs three ways on copies of one memory image (0xA5 everywhere the inputs are not):
//   ref  smx::softmax_mode(c, io, io); off  softmax_mode_hpr(c, 0, nullptr, io, io);
//   hpr  softmax_mode_hpr(c, HPR_A_HP1, hp1view, io, io), hp1view = a copy holding only the score rows and the
//        rank tables (every other byte 0xA5), so a value read over hp1 from anywhere else would reach an output.
// Both must equal ref in every word. smx::softmax_mode itself is checked against smx_emu by the softmax agent's
// tb_softmax (vectors from make_vectors.py); this tb proves the fused unit equal to it.
#include <cfenv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "hpr_softmax.h"

using smx::Word;

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

struct Mem {
  uint64_t base;
  std::vector<Word> v;
  Mem(uint64_t b, size_t bytes) : base(b), v((bytes + 15) / 16) {
    for (auto& w : v)
      for (int j = 0; j < 4; j++) w.range(32 * j + 31, 32 * j) = 0xA5A5A5A5u;
  }
  Word* ptr() { return reinterpret_cast<Word*>(reinterpret_cast<uintptr_t>(v.data()) - (uintptr_t)(base >> 4) * 16); }
  Word& w(uint64_t a) { return v[(a - base) >> 4]; }
  void set32(uint64_t a, uint32_t bits) { w(a).range(32 * ((a >> 2) & 3) + 31, 32 * ((a >> 2) & 3)) = bits; }
};

struct Stats {
  long cases = 0, active = 0, rows = 0, words = 0, bad_off = 0, bad_hpr = 0;
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

// value classes of a score row
static float score(std::mt19937_64& rng, int cls, float sig) {
  std::normal_distribution<float> nd(0.f, 1.f);
  switch (cls) {
    case 0: return nd(rng) * sig;
    case 1: return (rng() % 16 == 0) ? 40.f * sig : nd(rng) * sig;  // a dominant key
    case 2: return 3.0f;                                            // all equal
    case 3: return nd(rng) * 1e-30f;                                // tiny
    case 4: return (float)((int)(rng() % 9) - 4) * 0.125f;          // ties on a grid
    default: return nd(rng) * 1e4f;                                 // far-off keys (q = 0)
  }
}

// c (slots relative) is placed at base; A, C, F, D, E get their own regions with gaps.
static bool run3(const char* name, ctr::Cmd c, uint64_t base, std::mt19937_64& rng, bool verbose = false) {
  const uint64_t B = c.B, N = c.N, L = c.K, NP = c.p[ctr::P_NP] ? c.p[ctr::P_NP] : c.N;
  ctr::Slot &A = c.s[ctr::SLOT_A], &C = c.s[ctr::SLOT_C], &F = c.s[ctr::SLOT_F], &D = c.s[ctr::SLOT_D],
            &E = c.s[ctr::SLOT_E];
  const bool mask = c.flags & ctr::F_MASK;
  const uint64_t xa = (B - 1) * A.stride + (N - 1) * A.pitch + 4 * L;
  const uint64_t xc = (B - 1) * C.stride + (NP - 1) * C.pitch + 2 * L;
  const uint64_t xf = (B - 1) * F.stride + 4 * ctr::pad4((uint32_t)NP);
  const uint64_t xd = mask ? (B - 1) * D.stride + 4 * ctr::pad4((uint32_t)N) : 0;
  const uint64_t xe = mask ? (B - 1) * E.stride + 4 * L : 0;
  const uint64_t g = 512;
  const uint64_t a0 = base + g, c0 = (a0 + xa + g + 4095) & ~4095ull, f0 = (c0 + xc + g + 255) & ~255ull;
  const uint64_t d0 = (f0 + xf + g + 255) & ~255ull, e0 = (d0 + xd + g + 255) & ~255ull, end = e0 + xe + g;
  A.addr = a0;
  C.addr = c0;
  F.addr = f0;
  if (mask) {
    D.addr = d0;
    E.addr = e0;
  }
  uint32_t f[ctr::REC_FIELDS];
  hpr::encode_hpr(c, hpr::HPR_A_HP1, f);
  ctr::Cmd d;
  CHECK(ctr::decode(f, d) == ctr::ST_END, "%s: decode", name);
  const uint32_t hprf = f[hpr::FLD_HPR];
  const uint32_t v = ctr::validate(d), vh = hpr::validate_hpr(d, hprf);
  CHECK(v == ctr::ME_OK && vh == ctr::ME_OK, "%s: validate %u / hpr %u", name, v, vh);
  if (v != ctr::ME_OK) return false;
  Mem m(base, end - base), hv(base, end - base);
  for (uint64_t b = 0; b < B; b++) {
    if (b && A.stride == 0) break;
    for (uint64_t n = 0; n < N; n++) {
      const int cls = (int)(rng() % 6);
      const float sig = std::exp2f((float)((int)(rng() % 12) - 4));
      for (uint64_t k = 0; k < L; k++) {
        const uint32_t x = ctr::f2u(score(rng, cls, sig));
        m.set32(ctr::row_addr(A, b, n) + 4 * k, x);
        hv.set32(ctr::row_addr(A, b, n) + 4 * k, x);
      }
    }
  }
  if (mask) {
    for (uint64_t b = 0; b < B; b++) {
      const int kind = (int)(rng() % 3);
      if (!(b && D.stride == 0))
        for (uint64_t n = 0; n < ctr::pad4((uint32_t)N); n++) {
          int32_t r = kind == 0 ? (int32_t)(rng() % 4) : kind == 1 ? (int32_t)(2 + n) : (int32_t)(rng() % 7) - 1;
          if (rng() % 17 == 0) r = -1;  // a fully masked row
          m.set32(D.addr + b * D.stride + 4 * n, (uint32_t)r);
          hv.set32(D.addr + b * D.stride + 4 * n, (uint32_t)r);
        }
      if (!(b && E.stride == 0))
        for (uint64_t k = 0; k < L; k++) {
          int32_t r = kind == 1 ? (int32_t)(k < L / 2 ? (rng() % 2) : 2 + (k - L / 2)) : (int32_t)(rng() % 5) - 1;
          m.set32(E.addr + b * E.stride + 4 * k, (uint32_t)r);
          hv.set32(E.addr + b * E.stride + 4 * k, (uint32_t)r);
        }
    }
  }
  Mem e = m, mo = m, mh = m;
  smx::softmax_mode(d, e.ptr(), e.ptr());
  hps::softmax_mode_hpr(d, 0, nullptr, mo.ptr(), mo.ptr());
  hps::softmax_mode_hpr(d, hprf, hv.ptr(), mh.ptr(), mh.ptr());
  const bool on = hpr::hpr_active_smx(d, hprf);
  uint64_t fo = 0, fh = 0;
  const long bo = diff(mo, e, &fo), bh = diff(mh, e, &fh);
  const long changed = diff(e, m, nullptr);
  g_st.cases++;
  g_st.active += on;
  g_st.rows += (long)B * N;
  g_st.words += (long)m.v.size();
  g_st.bad_off += bo;
  g_st.bad_hpr += bh;
  CHECK(changed > 0, "%s: the unfused mode wrote nothing", name);
  CHECK(bo == 0, "%s: HPR off vs smx: %ld words differ (first 0x%llx)", name, bo, (unsigned long long)fo);
  CHECK(bh == 0, "%s: HPR on vs smx: %ld words differ (first 0x%llx)", name, bh, (unsigned long long)fh);
  if (verbose || bo || bh)
    printf("  %-12s B %3llu N %4llu L %3llu NP %4llu mask %d hpr %d: words %zu (written %ld), differ off %ld hpr %ld\n",
           name, (unsigned long long)B, (unsigned long long)N, (unsigned long long)L, (unsigned long long)NP,
           (int)mask, (int)on, m.v.size(), changed, bo, bh);
  return bo == 0 && bh == 0;
}

static ctr::Cmd mk(uint32_t B, uint32_t N, uint32_t L, uint32_t NP, bool mask, bool contig_a, bool contig_c,
                   bool shared_e, float mul) {
  ctr::Cmd c;
  memset(&c, 0, sizeof c);
  c.op = ctr::OP_SOFTMAX;
  c.flags = ctr::F_QOUT | (mask ? ctr::F_MASK : 0);
  c.B = B;
  c.N = N;
  c.K = L;
  c.p[ctr::P_EPS] = ctr::f2u(mul);
  c.p[ctr::P_NP] = NP == N ? 0 : NP;
  ctr::Slot &A = c.s[ctr::SLOT_A], &C = c.s[ctr::SLOT_C], &F = c.s[ctr::SLOT_F], &D = c.s[ctr::SLOT_D],
            &E = c.s[ctr::SLOT_E];
  A.pitch = 4 * L + (contig_a ? 0 : 48);
  A.stride = B > 1 ? N * A.pitch + 32 : 0;
  C.pitch = 2 * L + (contig_c ? 0 : 16);
  C.stride = B > 1 ? NP * C.pitch + 64 : 0;
  F.stride = B > 1 ? 4 * ctr::pad4(NP) + 16 : 0;
  if (mask) {
    D.stride = B > 1 ? 4 * ctr::pad4(N) : 0;
    E.stride = (B > 1 && !shared_e) ? 4 * L : 0;
  }
  A.addr = 0x1000;
  C.addr = 0x100000;
  F.addr = 0x200000;
  if (mask) {
    D.addr = 0x300000;
    E.addr = 0x400000;
  }
  return c;
}

static void test_edges(std::mt19937_64& rng) {
  long f0 = g_fail;
  struct E {
    const char* nm;
    uint32_t B, N, L, NP;
    bool mask, ca, cc, se;
    float mul;
  } es[] = {
      {"vision", 24, 256, 256, 256, false, true, true, false, 1.0f},  // Q.K^T of 2 images x 12 heads
      {"lm.prefix", 5, 135, 64, 136, true, true, true, true, 1.0f},  // 3 x 45 query rows per KV head
      {"exp.self", 5, 150, 128, 152, true, true, true, true, 1.0f},
      {"exp.cross", 5, 150, 64, 152, true, false, false, true, 1.0f},
      {"l64.n1", 1, 1, 64, 1, false, true, true, false, 1.0f},
      {"l192", 2, 37, 192, 40, true, false, true, false, 0.37f},
      {"l256.big", 1, 300, 256, 301, true, true, false, false, 1.0f},
      {"l8", 3, 5, 8, 8, true, true, true, false, 1.0f},    // HPR not active (L % 64)
      {"l40", 1, 9, 40, 9, false, false, true, false, 2.f}, // HPR not active
      {"l96", 2, 21, 96, 21, true, true, true, false, 1.0f},
      {"l128.tiny", 1, 64, 128, 64, false, true, true, false, 1e-20f},
  };
  for (const auto& e : es) {
    const ctr::Cmd c = mk(e.B, e.N, e.L, e.NP, e.mask, e.ca, e.cc, e.se, e.mul);
    run3(e.nm, c, 0x8ull << 32, rng, true);
  }
  printf("edge commands: %zu, %ld failures\n", sizeof es / sizeof es[0], g_fail - f0);
}

static void test_random(int n, std::mt19937_64& rng) {
  long f0 = g_fail;
  const uint64_t bases[] = {0x0ull, 0x8ull << 32, 0x7FFF0000ull, 0xFF00000000ull};
  for (int i = 0; i < n; i++) {
    auto r = [&](uint64_t k) { return (uint32_t)(rng() % k); };
    const uint32_t L = r(2) ? 64 * (1 + r(4)) : 8 * (1 + r(32));
    const uint32_t B = 1 + (r(3) == 0 ? r(6) : 0);
    const uint32_t N = r(5) == 0 ? 1 + r(4) : 1 + r(r(3) ? 160 : 520);
    const uint32_t NP = N + (r(3) == 0 ? r(9) : 0);
    const float mul = r(4) == 0 ? std::exp2f((float)((int)r(20) - 10)) : 1.0f;
    const ctr::Cmd c = mk(B, N, L, NP, r(3) != 0, r(3) != 0, r(3) != 0, r(2), mul);
    run3("rnd", c, bases[i % 4], rng);
  }
  printf("random commands: %d, %ld failures\n", n, g_fail - f0);
}

static void test_plan(const char* path) {
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
  std::mt19937_64 rng(99);
  std::map<std::string, int> seen;
  int total = 0, ran = 0;
  long f0 = g_fail;
  for (size_t i = 0; i < nrec; i++) {
    uint32_t f[ctr::REC_FIELDS];
    memcpy(f, &buf[i * ctr::REC_BYTES], ctr::REC_BYTES);
    ctr::Cmd c;
    if (ctr::decode(f, c) != ctr::ST_END || c.op != ctr::OP_SOFTMAX) continue;
    total++;
    char key[200];
    snprintf(key, sizeof key, "%u %u %u %u %u %u %u %u %u %u %u", c.flags, c.K, c.N, c.B, c.s[0].pitch, c.s[0].stride,
             c.s[2].pitch, c.s[2].stride, c.s[3].stride, c.s[4].stride, c.p[ctr::P_NP]);
    if (seen.count(key)) continue;
    seen[key]++;
    ran++;
    // re-based: the strides / pitches of the record, local addresses
    run3("plan", c, (i & 1) ? (0x8ull << 32) : 0x40000000ull, rng, true);
  }
  printf("plan %s: %d SOFTMAX records, %d distinct shapes run, %ld failures\n", path, total, ran, g_fail - f0);
}

int main(int argc, char** argv) {
  fesetround(FE_TONEAREST);
  int nrand = 300;
  std::vector<const char*> plans;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--random") && i + 1 < argc) nrand = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--plan") && i + 1 < argc) plans.push_back(argv[++i]);
  }
#ifdef HS_DFSIM
  printf("threaded bounded-FIFO model (HS_DFSIM / SMX_DFSIM): every process a thread, every stream at its depth\n");
#endif
  std::mt19937_64 rng(5);
  test_edges(rng);
  test_random(nrand, rng);
  for (const char* p : plans) test_plan(p);
  printf("cases %ld (HPR active in %ld), %ld rows, %ld memory words x 3 compared; words differing: off vs smx %ld, "
         "hpr vs smx %ld\n",
         g_st.cases, g_st.active, g_st.rows, g_st.words, g_st.bad_off, g_st.bad_hpr);
  printf(g_fail ? "TB FAILED: %ld failures\n" : "TB PASSED\n", g_fail);
  return g_fail ? 1 : 0;
}
