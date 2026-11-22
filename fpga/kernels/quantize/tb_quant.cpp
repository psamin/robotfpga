// Host testbench of the row unit's QUANT_X / QUANT_W / QUANT_WT / COPY modes (quant_mode.h, quant_int16.h)
// against the contract (fpga/kernels/contract.h: ctr::ref::quant_row, the layouts of CONTRACT.md 1.1).
//   build / run: sh run_tests.sh [--quick] [--plan list.bin] [--exhaustive NS] [captured.qz ...]
//   Built with -DQZ_DFSIM -Idfsim the same tests run as a threaded bounded-FIFO model (deadlock check).
// Sections:
//   1-3  arithmetic: scale_bits (8 exponents x 2^23 + small), recip_sig and recip_unrolled (all 2^23 S),
//        quant_elem on 0.4 M rows x ~600 adversarial values (+ exhaustive slices), stored_scale
//   4    the mode on random contract-valid commands: every op, segments, NP zero rows, shift, items, pitches,
//        strides, gaps, 32-column chunks, single-word units, bases up to 2^40; memory is NaN-magic filled and
//        compared word for word against an independent reference built from ctr::ref::quant_row
//   5    the -12392 chunk list (contract_plan.py --out): every QUANT_X / QUANT_W / QUANT_WT / COPY record,
//        decoded from its 256 bytes and re-based into local memory with its own pitches, strides, p2, p3
//   6    rejects (ME_DIMS, nothing touched), non-finite rows (scale +Inf / NaN, no write outside extents)
//   7    captured activations (check_policy.py --dump): QUANT_X of real rows == torch's q and s
#include <cfenv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "quant_mode.h"

using namespace qz;

#ifdef QZ_DFSIM
int qz_dfsim_depth(const std::string& n) {
  static const std::map<std::string, int> d = {
      {"qz_xs", QZ_D_XS}, {"qz_am", QZ_D_AM}, {"qz_d", QZ_D_D},   {"qz_ss", QZ_D_SS}, {"qz_sf", QZ_D_SF},
      {"qz_ow", QZ_D_OW}, {"qz_sq", QZ_D_SQ}, {"qz_hd", QZ_D_HD}, {"qz_dd", QZ_D_DD}};
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

static float ref_scale(float absmax) { return absmax == 0.f ? 1.f : fmaxf(absmax, 1e-30f) / 32767.0f; }
static int ref_q(float x, float s) {
  float r = nearbyintf(x / s);  // FE_TONEAREST: half to even, as torch.round
  if (r > 32767.f) r = 32767.f;
  if (r < -32767.f) r = -32767.f;
  return (int)r;
}

// ================================================================ 1-3. arithmetic
static void test_scale_exhaustive() {
  long n = 0, f0 = g_fail;
  const int exps[] = {27, 28, 64, 127, 128, 200, 253, 254};  // 27: exponent of 1e-30f (the clamp)
  for (int e : exps)
    for (uint32_t m = 0; m < (1u << 23); m++) {
      const uint32_t ab = ((uint32_t)e << 23) | m;
      CHECK((uint32_t)scale_bits(abits_t(ab)) == f2u(ref_scale(u2f(ab))), "scale a=%08x", ab);
      n++;
    }
  std::mt19937 rng(1);
  for (int i = 0; i < 2000000; i++) {
    const uint32_t ab = i < 1000 ? (uint32_t)i : rng() % (TINY_BITS + 1000);
    CHECK((uint32_t)scale_bits(abits_t(ab)) == f2u(ref_scale(u2f(ab))), "scale(small) a=%08x", ab);
    n++;
  }
  printf("scale_bits: %ld absmax values (8 exponents x all 2^23 significands + 2M small, zero -> 1.0), %ld failures\n",
         n, g_fail - f0);
}

static void test_recip_exhaustive() {
  long f0 = g_fail;
  for (uint32_t S = 1u << 23; S < (1u << 24); S++) {
    uint64_t want = (1ULL << 49) / S;
    if (want > (1ULL << 26) - 1) want = (1ULL << 26) - 1;
    CHECK((uint32_t)recip_sig(ap_uint<24>(S)) == want, "recip_sig S=%06x", S);
    CHECK((uint32_t)recip_unrolled(ap_uint<24>(S)) == want, "recip_unrolled S=%06x", S);
  }
  printf("recip_sig and recip_unrolled (%d bit a step, %d steps): all 2^23 significands, %ld failures\n", RECIP_BITS,
         RECIP_STEPS, g_fail - f0);
}

static long n_elem = 0, n_tie = 0, n_dbl = 0, n_fdiv_fail = 0;
static void check_elem(float x, const RowScale& rs) {
  const float s = u2f((uint32_t)rs.s_bits);
  const int want = ref_q(x, s);
  const float t = fabsf(x / s);
  if (t < 32767.f && t - floorf(t) == 0.5f) {
    n_tie++;
    const double te = fabs((double)x / (double)s);
    if (te != (double)t && (long)nearbyint(te) != (long)nearbyintf(t)) n_dbl++;
  }
  CHECK((int)quant_elem_recip(fbits_t(f2u(x)), rs) == want, "elem x=%a s=%a", x, s);
  if ((int)quant_elem_fdiv(fbits_t(f2u(x)), rs) != want) n_fdiv_fail++;
  n_elem++;
}

static void test_elements(long rows) {
  std::mt19937_64 rng(12345);
  std::normal_distribution<float> nd(0.f, 1.f);
  std::uniform_real_distribution<float> ud(-1.f, 1.f);
  long f0 = g_fail;
  for (long r = 0; r < rows; r++) {
    uint32_t ab;
    const int kind = r % 4;
    if (kind == 0) ab = (uint32_t)(rng() % 0x7F000000u);
    else if (kind == 1) ab = (uint32_t)((100u + rng() % 60) << 23) | (uint32_t)(rng() & 0x7FFFFF);
    else if (kind == 2) ab = f2u(fabsf(nd(rng)) * 8.f + 1e-3f);
    else ab = TINY_BITS - 1000 + (uint32_t)(rng() % 3000);
    const float a = u2f(ab);
    const RowScale rs = row_scale(abits_t(ab));
    const float s = u2f((uint32_t)rs.s_bits);
    CHECK((uint32_t)rs.s_bits == f2u(ref_scale(a)), "row scale");
    check_elem(a, rs);
    check_elem(-a, rs);
    check_elem(0.f, rs);
    check_elem(-0.f, rs);
    check_elem(u2f(1), rs);
    check_elem(u2f(0x807FFFFFu), rs);
    for (int i = 0; i < 64; i++) check_elem(ud(rng) * a, rs);
    for (int i = 0; i < 64; i++) {
      const uint32_t xb = (uint32_t)(rng() % ((uint64_t)ab + 1));
      check_elem(u2f(xb | ((uint32_t)(rng() & 1) << 31)), rs);
    }
    for (int i = 0; i < 64; i++) {  // around half-integers and integers of x / s
      const int k = (int)(rng() % 32768);
      const uint32_t vb = f2u((float)((k + ((i & 1) ? 0.5 : 0.0)) * (double)s));
      for (int dlt = -3; dlt <= 3; dlt++) {
        const float x = u2f(vb + dlt);
        if (fabsf(x) <= a) {
          check_elem(x, rs);
          check_elem(-x, rs);
        }
      }
    }
    for (int i = 0; i < 16; i++) {
      const uint32_t vb = f2u((float)(((i & 1) ? 1.5 : 0.5) * (double)s));
      for (int dlt = -2; dlt <= 2; dlt++) check_elem(u2f(vb + dlt), rs);
    }
  }
  const RowScale rz = row_scale(abits_t(0));
  CHECK((uint32_t)rz.s_bits == ONE_BITS, "zero row scale");
  check_elem(0.f, rz);
  check_elem(-0.f, rz);
  printf("quant_elem_recip: %ld elements over %ld rows, %ld failures; quant_elem_fdiv %ld failures; coverage: %ld "
         "ties of fl32(x/s), %ld double-rounding cases\n",
         n_elem, rows, g_fail - f0, n_fdiv_fail, n_tie, n_dbl);
}

static void test_elements_exhaustive(int ns) {
  std::mt19937_64 rng(99);
  long f0 = g_fail, n = 0;
  for (int k = 0; k < ns; k++) {
    uint32_t sb;
    if (k == 0) sb = 0x3F800000u;
    else if (k == 1) sb = 0x3FFFFFFFu;
    else sb = (uint32_t)((100 + rng() % 40) << 23) | (uint32_t)(rng() & 0x7FFFFF);
    RowScale rs = row_scale_from_s(ap_uint<32>(sb));
    rs.R = recip_sig(rs.S);
    const int Es = (int)(sb >> 23);
    const float s = u2f(sb);
    for (int d = -3; d <= 16; d++) {
      const uint32_t E = (uint32_t)(Es + d);
      for (uint32_t m = 0; m < (1u << 23); m++) {
        const uint32_t xb = (E << 23) | m | ((m & 1) << 31);
        const int want = ref_q(u2f(xb), s);
        if ((int)quant_elem_recip(fbits_t(xb), rs) != want) CHECK(false, "exhaustive x=%08x s=%08x", xb, sb);
        n++;
      }
    }
  }
  printf("quant_elem_recip exhaustive: %d scales x 20 exponents x all 2^23 significands = %ld elements, %ld failures\n",
         ns, n, g_fail - f0);
}

static void test_stored_scale() {
  long f0 = g_fail, n = 0;
  std::mt19937_64 rng(5);
  for (int i = 0; i < 2000000; i++) {
    uint32_t ab = i < 100 ? (uint32_t)i : (uint32_t)(rng() % 0x7F800000u);
    if (i % 7 == 0) ab = TINY_BITS - 50 + (uint32_t)(rng() % 100);
    const int sh = (int)(rng() % 17) - 8;
    const uint32_t s = (uint32_t)scale_bits(abits_t(ab));
    const float want = ab == 0 ? 1.0f : ldexpf(ref_scale(u2f(ab)), sh);  // ctr::ref::quant_row's stored scale
    CHECK((uint32_t)stored_scale(abits_t(ab), ap_uint<32>(s), ap_int<5>(sh)) == f2u(want), "stored a=%08x sh=%d", ab,
          sh);
    n++;
  }
  // non-finite: +Inf and NaN absmax are stored as such (whatever the shift)
  for (int sh = -8; sh <= 8; sh++) {
    const uint32_t inf = 0x7F800000u, nan = 0x7FC00123u;
    CHECK((uint32_t)stored_scale(abits_t(inf), scale_bits(abits_t(inf)), ap_int<5>(sh)) == inf, "inf");
    CHECK((uint32_t)stored_scale(abits_t(nan), scale_bits(abits_t(nan)), ap_int<5>(sh)) == nan, "nan");
    CHECK((uint32_t)stored_scale(abits_t(0), scale_bits(abits_t(0)), ap_int<5>(sh)) == ONE_BITS, "zero");
  }
  printf("stored_scale: %ld (absmax, shift) pairs == ldexp(s, shift), zero rows 1.0, +Inf / NaN kept, %ld failures\n",
         n, g_fail - f0);
}

// ================================================================ 4-7. the mode
static const uint32_t MAGIC = 0xFFC0A5A5u;  // a NaN: if a gap were ever read into a row, its scale would show it
struct Mem {
  uint64_t base;  // byte address of v[0] (any 16-B aligned value below 2^40)
  std::vector<word_t> v;
  Mem(uint64_t b, size_t bytes) : base(b), v((bytes + 15) / 16) {
    for (auto& w : v)
      for (int j = 0; j < 4; j++) w(32 * j + 31, 32 * j) = MAGIC;
  }
  word_t* ptr() {  // ptr()[addr >> 4] is the word at byte address addr
    return reinterpret_cast<word_t*>(reinterpret_cast<uintptr_t>(v.data()) - (uintptr_t)(base >> 4) * 16);
  }
  word_t& w(uint64_t addr) { return v[(addr - base) >> 4]; }
  void set_f32(uint64_t a, uint32_t bits) { w(a)(32 * ((a >> 2) & 3) + 31, 32 * ((a >> 2) & 3)) = bits; }
  uint32_t f32(uint64_t a) { return (uint32_t)w(a)(32 * ((a >> 2) & 3) + 31, 32 * ((a >> 2) & 3)); }
  void set_i16(uint64_t a, int16_t q) { w(a)(16 * ((a >> 1) & 7) + 15, 16 * ((a >> 1) & 7)) = (uint16_t)q; }
};

// value classes of an input row
static float gen_val(std::mt19937_64& rng, int cls, float mul) {
  std::normal_distribution<float> nd(0.f, 1.f);
  std::student_t_distribution<float> td(2.f);
  switch (cls) {
    case 0: return nd(rng) * mul;
    case 1: return td(rng) * mul;                          // heavy tails
    case 2: return (rng() & 1) ? -0.f : 0.f;               // all-zero row
    case 3: return nd(rng) * 1e-36f;                       // below the 1e-30 clamp, denormals
    case 4: return u2f((uint32_t)(rng() & 0x807FFFFFu));   // denormals only
    case 5: return (rng() % 64 == 0) ? 1e36f * nd(rng) : nd(rng);  // huge (finite) outliers
    default: {                                              // values on ties of x / s for s of absmax 32767
      const int k = (int)(rng() % 32767);
      return (float)(k + 0.5) * ((rng() & 1) ? 1.f : -1.f);
    }
  }
}

struct Stats {
  long cases = 0, values = 0, words = 0, bad_words = 0;
  std::map<std::string, long> by_op;
};
static Stats g_st;

// The contract's expected image (independent of the HLS code: ctr::ref + the CONTRACT.md 1.1 formulas).
// x[b][n][k] are the input values; nonfin: rows whose q is undefined (only their scale is checked).
static void reference(const ctr::Cmd& c, const std::vector<float>& x, Mem& e, std::vector<uint64_t>* skip_q) {
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
        bool nf = false;
        for (uint32_t n = 0; n < N; n++) {
          col[n] = X(b, n, ch);
          nf |= ((f2u(col[n]) >> 23) & 0xFF) == 0xFF;
        }
        const float s = ctr::ref::quant_row(col.data(), N, q.data(), sh);
        ctr::Slot w = C;
        w.addr += j * segc;
        for (uint32_t n = 0; n < NP; n++) {
          const uint64_t a = ctr::w_word_addr(w, b, cc, n, NP) + 2 * ctr::w_lane(cc);
          if (nf && skip_q) skip_q->push_back(a);
          e.set_i16(a, n < N ? q[n] : 0);
        }
        e.set_f32(F.addr + (uint64_t)b * F.stride + j * segf + 4 * cc, f2u(s));
      }
    return;
  }
  const bool kw = c.op == ctr::OP_QUANT_W;
  const uint32_t SEG = sg ? sg : K, nseg = K / SEG;
  std::vector<int16_t> q(SEG);
  for (uint32_t b = 0; b < B; b++)
    for (uint32_t j = 0; j < nseg; j++) {
      for (uint32_t n = 0; n < ((NP + 3) & ~3u); n++) {
        float s = 1.0f;
        bool nf = false;
        if (n < N) {
          for (uint32_t k = 0; k < SEG; k++) nf |= ((f2u(X(b, n, j * SEG + k)) >> 23) & 0xFF) == 0xFF;
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
          if (nf && skip_q) skip_q->push_back(a);
          e.set_i16(a, q[k]);
        }
      }
    }
}

// extents of the slots the unit writes or reads (bytes from the slot address), as validate() computes them
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

// Run one command (slot addresses already final) on memory holding x; compare with the reference.
static bool run_case(const char* name, ctr::Cmd c, Mem& m, const std::vector<float>& x, bool nonfin_ok = false,
                     bool verbose = false) {
  // through the 256-byte record and back: what the sequencer hands the mode
  uint32_t f[ctr::REC_FIELDS];
  ctr::encode(c, f);
  ctr::Cmd d;
  const uint32_t st = ctr::decode(f, d);
  CHECK(st == ctr::ST_END, "%s: decode %u", name, st);
  const uint32_t v = ctr::validate(d), vh = ctr::validate_hw(d);
  CHECK(v == ctr::ME_OK && vh == ctr::ME_OK, "%s: validate %u / validate_hw %u", name, v, vh);
  if (v != ctr::ME_OK) return false;
  // inputs
  const ctr::Slot& A = d.s[ctr::SLOT_A];
  for (uint32_t b = 0; b < d.B; b++)
    for (uint32_t n = 0; n < d.N; n++)
      for (uint32_t k = 0; k < d.K; k++) m.set_f32(ctr::row_addr(A, b, n) + 4 * k, f2u(x[((size_t)b * d.N + n) * d.K + k]));
  Mem e = m;
  std::vector<uint64_t> skipq;
  reference(d, x, e, nonfin_ok ? &skipq : nullptr);
  for (uint64_t a : skipq) {  // q of a non-finite row is undefined: take the kernel's
    (void)a;
  }
  const uint32_t r = row_mode(d, m.ptr(), m.ptr());
  CHECK(r == ctr::ME_OK, "%s: row_mode returned %u", name, r);
  if (nonfin_ok)
    for (uint64_t a : skipq) e.w(a) = m.w(a);  // the whole word: other lanes of it are checked by the other rows
  long bad = 0;
  uint64_t first = 0;
  static int dumped = 0;
  const uint64_t ca = d.s[ctr::SLOT_C].addr, fa = d.s[ctr::SLOT_F].addr;
  for (size_t i = 0; i < m.v.size(); i++)
    if (m.v[i] != e.v[i]) {
      if (!bad) first = m.base + 16 * i;
      bad++;
      if (getenv("QZ_DUMP") && dumped < atoi(getenv("QZ_DUMP"))) {
        dumped++;
        const uint64_t a = m.base + 16 * i;
        printf("    diff at 0x%llx (%s +0x%llx): got  %s\n                                    want %s\n",
               (unsigned long long)a, a >= fa && d.op != ctr::OP_COPY ? "F" : a >= ca ? "C" : "A/guard",
               (unsigned long long)(a >= fa && d.op != ctr::OP_COPY ? a - fa : a >= ca ? a - ca : 0),
               m.v[i].to_string(16).c_str(), e.v[i].to_string(16).c_str());
      }
    }
  g_st.cases++;
  g_st.words += (long)m.v.size();
  g_st.bad_words += bad;
  g_st.values += (long)d.B * d.N * d.K;
  g_st.by_op[name]++;
  CHECK(bad == 0, "%s: op %02x B %u N %u K %u p1 %u NP %u shift %d: %ld words differ, first at 0x%llx", name, d.op, d.B,
        d.N, d.K, d.p[ctr::P_SEG], d.p[ctr::P_NP], (int32_t)d.p[ctr::P_SHIFT], bad, (unsigned long long)first);
  if (verbose || bad)
    printf("  %-10s op %02x B %5u N %5u K %5u p1 %2u NP %5u sh %2d: %ld of %zu words differ\n", name, d.op, d.B, d.N,
           d.K, d.p[ctr::P_SEG], d.p[ctr::P_NP], (int32_t)d.p[ctr::P_SHIFT], bad, m.v.size());
  return bad == 0;
}

// Place slots A, C, F of c (extents from the command) at base, base + ..., with guard gaps; fill x; run.
static bool place_and_run(const char* name, ctr::Cmd c, uint64_t base, std::mt19937_64& rng, int cls_mix,
                          bool verbose = false) {
  const Ext x = extents(c);
  const uint64_t g = 256;
  const uint64_t a0 = base + g, c0 = (a0 + x.a + g + 4095) & ~4095ull, f0 = (c0 + x.c + g + 255) & ~255ull;
  const uint64_t end = f0 + x.f + g;
  c.s[ctr::SLOT_A].addr = a0;
  c.s[ctr::SLOT_C].addr = c0;
  if (c.op != ctr::OP_COPY) c.s[ctr::SLOT_F].addr = f0;
  Mem m(base, end - base);
  std::vector<float> v((size_t)c.B * c.N * c.K);
  for (uint32_t b = 0; b < c.B; b++)
    for (uint32_t n = 0; n < c.N; n++) {
      const int cls = cls_mix ? (int)(rng() % 7) : 0;
      const float mul = std::exp2f((float)((int)(rng() % 40) - 20));
      for (uint32_t k = 0; k < c.K; k++)  // A.stride == 0: every item reads item 0's rows
        v[((size_t)b * c.N + n) * c.K + k] =
            (b && c.s[ctr::SLOT_A].stride == 0) ? v[(size_t)n * c.K + k] : gen_val(rng, cls, mul);
    }
  return run_case(name, c, m, v, false, verbose);
}

// a random contract-valid command of op (slot addresses are placed by place_and_run)
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
  if (op == ctr::OP_QUANT_WT && r(3) == 0) c.K = 32 * (1 + r(20));  // p1 = 0: K % 32 (half chunks)
  const uint32_t K = c.K;
  bool seg = false;
  if (op == ctr::OP_QUANT_X) seg = K % 64 == 0 && K / 64 <= 64 && r(2);
  if (op == ctr::OP_QUANT_W) seg = true;
  if (op == ctr::OP_QUANT_WT) seg = K % 64 == 0 && r(3) != 0;
  if (op == ctr::OP_QUANT_W && K == 64 && r(2)) seg = false;  // p1 = 0 with K = 64: one segment
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
  // A: rows of K FP32 with a pitch, items with a stride
  ctr::Slot& A = c.s[ctr::SLOT_A];
  A.pitch = 4 * K + 16 * (r(3) ? 0 : 1 + r(8));
  if (c.N == 1 && r(2)) A.pitch = 0;
  A.stride = c.B > 1 ? (uint32_t)(c.N * (uint64_t)(A.pitch ? A.pitch : 4 * K) + 16 * r(5)) : 0;
  if (c.B > 1 && r(5) == 0) A.stride = 0;  // items read the same rows
  ctr::Slot& C = c.s[ctr::SLOT_C];
  ctr::Slot& F = c.s[ctr::SLOT_F];
  uint64_t citem = 0, fitem = 0;
  if (op == ctr::OP_COPY) {
    C.pitch = 4 * K + 16 * r(3);
    citem = (uint64_t)c.N * C.pitch;
  } else if (op == ctr::OP_QUANT_X) {
    if (nseg > 1 && r(2)) {  // head-major: segment j = a block of NP rows of 128 B
      C.pitch = 128;
      c.p[ctr::P_SEGC] = NP * 128 + 16 * r(3);
      citem = (uint64_t)nseg * c.p[ctr::P_SEGC];
    } else if (nseg > 1) {   // in-row segments
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
  // provisional addresses for validate (place_and_run moves them)
  A.addr = 0x10000;
  C.addr = 0x10000000;
  if (op != ctr::OP_COPY) F.addr = 0x20000000;
  return ctr::validate(c) == ctr::ME_OK;
}

static void test_random(int n, std::mt19937_64& rng, int& tried) {
  const uint32_t ops[] = {ctr::OP_QUANT_X, ctr::OP_QUANT_W, ctr::OP_QUANT_WT, ctr::OP_COPY};
  const char* names[] = {"rnd.X", "rnd.W", "rnd.WT", "rnd.COPY"};
  long f0 = g_fail;
  int done = 0, small = 0;
  const uint64_t bases[] = {0x0ull, 0x8ull << 32, 0x7FFF0000ull, 0xFF00000000ull, 0x123450000ull};
  while (done < n) {
    tried++;
    const int oi = (int)(rng() % 4);
    ctr::Cmd c;
    if (!random_cmd(rng, ops[oi], c)) continue;
    RP p;
    row_plan(c, p);
    small += p.rsmall || p.wsmall;
    place_and_run(names[oi], c, bases[done % 5], rng, 1);
    done++;
  }
  printf("random commands: %d run (%d generated, %d with single-word units), %ld failures\n", done, tried, small,
         g_fail - f0);
}

// ---- 5. the -12392 chunk list: every record of this unit, re-based into local memory
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
    CHECK(ctr::validate_hw(c) == ctr::ME_OK, "plan record %zu: validate_hw", i);
    char key[256];  // the shape: everything but the addresses
    snprintf(key, sizeof key, "%u %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u", c.op, c.flags, c.K, c.N, c.B,
             c.s[0].pitch, c.s[0].stride, c.s[2].pitch, c.s[2].stride, c.s[5].stride, c.p[1], c.p[2], c.p[3], c.p[4],
             c.p[5], (unsigned)(c.s[5].pitch), 0u);
    if (!all && seen.count(key)) continue;
    seen[key]++;
    nrun[c.op]++;
    const char* nm = c.op == ctr::OP_QUANT_X ? "plan.X" : c.op == ctr::OP_QUANT_W ? "plan.W" : c.op == ctr::OP_QUANT_WT ? "plan.WT" : "plan.COPY";
    place_and_run(nm, c, (i & 1) ? (0x8ull << 32) : 0x40000000ull, rng, 1, false);
  }
  printf("plan %s: %zu records; this unit's: QUANT_X %d, QUANT_W %d, QUANT_WT %d, COPY %d; run %s: %d / %d / %d / %d, "
         "%ld failures\n",
         path, nrec, ntot[ctr::OP_QUANT_X], ntot[ctr::OP_QUANT_W], ntot[ctr::OP_QUANT_WT], ntot[ctr::OP_COPY],
         all ? "all" : "each distinct shape once", nrun[ctr::OP_QUANT_X], nrun[ctr::OP_QUANT_W],
         nrun[ctr::OP_QUANT_WT], nrun[ctr::OP_COPY], g_fail - f0);
}

// ---- edge cases by hand
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

static void test_edges(std::mt19937_64& rng) {
  long f0 = g_fail;
  struct E {
    const char* nm;
    uint32_t op, B, N, K, seg, NP;
    int sh;
  } es[] = {
      {"X.k8", ctr::OP_QUANT_X, 1, 3, 8, 0, 0, 0},           {"X.k32.n50", ctr::OP_QUANT_X, 1, 50, 32, 0, 0, 0},
      {"X.k720", ctr::OP_QUANT_X, 1, 50, 720, 0, 0, 0},      {"X.k960.np48", ctr::OP_QUANT_X, 1, 45, 960, 0, 48, 0},
      {"X.k12288", ctr::OP_QUANT_X, 1, 32, 12288, 0, 0, 0}, {"X.k16384", ctr::OP_QUANT_X, 1, 3, 16384, 0, 0, 0},
      {"X.q.vis", ctr::OP_QUANT_X, 1, 64, 768, 64, 0, -3},  {"X.q.np", ctr::OP_QUANT_X, 2, 45, 960, 64, 48, -3},
      {"X.n1", ctr::OP_QUANT_X, 3, 1, 64, 0, 0, 8},          {"X.np29", ctr::OP_QUANT_X, 1, 29, 64, 64, 0, 0},
      {"W.k64", ctr::OP_QUANT_W, 1, 50, 64, 0, 64, 0},       {"W.k320", ctr::OP_QUANT_W, 1, 45, 320, 64, 64, 0},
      {"W.n512", ctr::OP_QUANT_W, 1, 512, 768, 64, 512, 0},  {"W.b3", ctr::OP_QUANT_W, 3, 7, 128, 64, 32, 5},
      {"WT.k320", ctr::OP_QUANT_WT, 1, 50, 320, 64, 64, 0},  {"WT.vis", ctr::OP_QUANT_WT, 2, 256, 768, 64, 256, 0},
      {"WT.k32", ctr::OP_QUANT_WT, 1, 9, 32, 0, 16, 0},      {"WT.k96", ctr::OP_QUANT_WT, 2, 13, 96, 0, 24, -2},
      {"WT.n1", ctr::OP_QUANT_WT, 1, 1, 64, 64, 8, 0},       {"C.k8", ctr::OP_COPY, 1, 4, 8, 0, 0, 0},
      {"C.k3072", ctr::OP_COPY, 8, 4, 3072, 0, 0, 0},        {"C.k720", ctr::OP_COPY, 1, 12, 720, 0, 0, 0},
  };
  for (const auto& e : es) {
    const ctr::Cmd c = mk(e.op, e.B, e.N, e.K, e.seg, e.NP, e.sh);
    CHECK(ctr::validate(c) == ctr::ME_OK, "edge %s invalid (%u)", e.nm, ctr::validate(c));
    place_and_run(e.nm, c, 0x8ull << 32, rng, 1, true);
  }
  printf("edge commands: %zu, %ld failures\n", sizeof es / sizeof es[0], g_fail - f0);
}

// ---- 6. rejects and non-finite rows
static void test_rejects_nonfinite(std::mt19937_64& rng) {
  long f0 = g_fail;
  // QUANT_W with p1 = 0 and K != 64 (validate_hw accepts it; this unit does not run it), and other opcodes
  {
    ctr::Cmd c = mk(ctr::OP_QUANT_W, 1, 32, 128, 0, 32, 0);
    c.s[ctr::SLOT_C].pitch = (32 / 32) * 128 * 16;
    c.s[ctr::SLOT_A].addr = 0x1000;
    c.s[ctr::SLOT_C].addr = 0x10000;
    c.s[ctr::SLOT_F].addr = 0x20000;
    CHECK(ctr::validate(c) == ctr::ME_OK, "reject case must be contract-valid");
    Mem m(0, 0x30000);
    const Mem m0 = m;
    CHECK(row_mode(c, m.ptr(), m.ptr()) == ctr::ME_DIMS, "QUANT_W p1 = 0 K = 128 must be ME_DIMS");
    CHECK(m.v == m0.v, "a rejected command wrote memory");
    ctr::Cmd n = c;
    n.op = ctr::OP_NORM;
    CHECK(row_mode(n, m.ptr(), m.ptr()) == ctr::ME_DIMS, "NORM is not this build's");
    CHECK(m.v == m0.v, "a rejected command wrote memory");
  }
  // non-finite rows: scale +Inf / NaN stored, the rest of the image exact (q of those rows undefined)
  const uint32_t ops[] = {ctr::OP_QUANT_X, ctr::OP_QUANT_X, ctr::OP_QUANT_W, ctr::OP_QUANT_WT};
  const uint32_t segs[] = {0, 64, 64, 64};
  for (int t = 0; t < 4; t++) {
    ctr::Cmd c = mk(ops[t], 1, 40, 192, segs[t], ops[t] == ctr::OP_QUANT_W ? 64 : ops[t] == ctr::OP_QUANT_WT ? 40 : 0, 0);
    const Ext x = extents(c);
    const uint64_t base = 0x40000000, a0 = base + 256, c0 = (a0 + x.a + 4096 + 4095) & ~4095ull, fa = c0 + x.c + 512;
    c.s[ctr::SLOT_A].addr = a0;
    c.s[ctr::SLOT_C].addr = c0;
    c.s[ctr::SLOT_F].addr = fa;
    Mem m(base, fa + x.f + 512 - base);
    std::vector<float> v((size_t)c.N * c.K);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto& y : v) y = nd(rng);
    v[3 * c.K + 70] = INFINITY;   // row 3 (column 70)
    v[7 * c.K + 5] = -INFINITY;   // row 7
    v[11 * c.K + 100] = u2f(0x7FC00042u);  // row 11: NaN
    run_case(t == 0 ? "nf.X" : t == 1 ? "nf.Xseg" : t == 2 ? "nf.W" : "nf.WT", c, m, v, true, true);
  }
  printf("rejects (ME_DIMS, nothing written) and non-finite rows (+-Inf, NaN): %ld failures\n", g_fail - f0);
}

// ---- 7. captured activations: QUANT_X == torch quant_sym (q; s except all-zero rows, where 1.0)
static void test_captured(const char* path) {
  FILE* fp = fopen(path, "rb");
  if (!fp) {
    printf("cannot open %s\n", path);
    g_fail++;
    return;
  }
  long total = 0, bad = 0;
  int ncase = 0;
  char magic[4];
  while (fread(magic, 1, 4, fp) == 4) {
    int32_t rows, cols, nl;
    if (memcmp(magic, "QZ01", 4) || fread(&rows, 4, 1, fp) != 1 || fread(&cols, 4, 1, fp) != 1 || fread(&nl, 4, 1, fp) != 1) {
      g_fail++;
      break;
    }
    std::vector<char> name(nl + 1, 0);
    std::vector<float> x((size_t)rows * cols), s(rows);
    std::vector<int16_t> q((size_t)rows * cols);
    if (fread(name.data(), 1, nl, fp) != (size_t)nl || fread(x.data(), 4, x.size(), fp) != x.size() ||
        fread(q.data(), 2, q.size(), fp) != q.size() || fread(s.data(), 4, s.size(), fp) != s.size()) {
      g_fail++;
      break;
    }
    if (cols % 8) continue;  // the contract's rows are K % 8 == 0
    ctr::Cmd c = mk(ctr::OP_QUANT_X, 1, rows, cols, 0, 0, 0);
    const Ext ex = extents(c);
    const uint64_t a0 = 0x1000, c0 = (a0 + ex.a + 4095) & ~4095ull, fa = (c0 + ex.c + 4095) & ~4095ull;
    c.s[ctr::SLOT_A].addr = a0;
    c.s[ctr::SLOT_C].addr = c0;
    c.s[ctr::SLOT_F].addr = fa;
    Mem m(0, fa + ex.f + 256);
    for (int r = 0; r < rows; r++)
      for (int k = 0; k < cols; k++) m.set_f32(a0 + (uint64_t)r * 4 * cols + 4 * k, f2u(x[(size_t)r * cols + k]));
    CHECK(row_mode(c, m.ptr(), m.ptr()) == ctr::ME_OK, "captured %s", name.data());
    for (int r = 0; r < rows; r++) {
      bool zero = true;
      for (int k = 0; k < cols; k++) zero &= (f2u(x[(size_t)r * cols + k]) & 0x7FFFFFFF) == 0;
      const uint32_t sg = m.f32(fa + 4 * r);
      if (!(zero ? sg == ONE_BITS : sg == f2u(s[r]))) bad++;
      for (int k = 0; k < cols; k++) {
        const word_t& w = m.w(c0 + (uint64_t)r * 2 * cols + 2 * k);
        const int16_t g = (int16_t)(uint16_t)w(16 * ((k) & 7) + 15, 16 * ((k) & 7));
        if (g != q[(size_t)r * cols + k]) bad++;
      }
    }
    total += (long)rows * cols;
    ncase++;
  }
  fclose(fp);
  CHECK(bad == 0, "captured: %ld mismatches", bad);
  printf("captured %s: %d tensors through QUANT_X (row_mode), %ld values, %ld q / s mismatches vs torch\n", path, ncase,
         total, bad);
}

int main(int argc, char** argv) {
  fesetround(FE_TONEAREST);
  bool quick = false, plan_all = false;
  int exh = 0, nrand = 600;
  std::vector<const char*> files, plans;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--version")) return 0;
    if (!strcmp(argv[i], "--quick")) quick = true;
    else if (!strcmp(argv[i], "--exhaustive") && i + 1 < argc) exh = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--random") && i + 1 < argc) nrand = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--plan") && i + 1 < argc) plans.push_back(argv[++i]);
    else if (!strcmp(argv[i], "--plan-all")) plan_all = true;
    else files.push_back(argv[i]);
  }
#ifdef QZ_DFSIM
  printf("threaded bounded-FIFO model (QZ_DFSIM): every process a thread, every stream at its pragma depth\n");
#else
  if (!quick) {
    test_scale_exhaustive();
    test_recip_exhaustive();
  }
  test_elements(quick ? 20000 : 400000);
  if (exh) test_elements_exhaustive(exh);
  test_stored_scale();
#endif
  std::mt19937_64 rng(77);
  test_edges(rng);
  test_rejects_nonfinite(rng);
  int tried = 0;
  test_random(nrand, rng, tried);
  for (const char* p : plans) test_plan(p, plan_all);
  for (const char* p : files) test_captured(p);
  printf("mode: %ld commands, %ld input values, %ld memory words compared, %ld differ\n", g_st.cases, g_st.values,
         g_st.words, g_st.bad_words);
  printf(g_fail ? "TB FAILED: %ld failures\n" : "TB PASSED\n", g_fail);
  return g_fail ? 1 : 0;
}
