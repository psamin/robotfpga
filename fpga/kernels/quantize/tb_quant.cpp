// Host testbench of the int16 quantizer (quant_int16.h + quant_mode.h via quant_top.cpp) against
//   s = fl32(max(absmax, 1e-30f) / 32767) (1.0f for an all-zero row), q = clamp(nearbyint(fl32(x / s)), +-32767)
// which is precision_emulation.py quant_sym (torch's division and round half to even, both IEEE) with
// pack_float.py's all-zero-row scale.
//   build: see README.md (g++ -O2 -std=c++14 -I<Vitis_HLS>/include tb_quant.cpp quant_top.cpp)
//   run:   tb_quant [--quick] [--exhaustive NS] [captured.qz ...]
//          captured.qz: written by check_policy.py --dump (real activations with torch's q and s)
#include <cfenv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "quant_mode.h"

int quant_top(const qz::word_t* hp0, const qz::word_t* hp1, qz::word_t* io, unsigned B, unsigned N, unsigned K,
              unsigned in_off, unsigned in_pitch, unsigned in_bstride, unsigned out_off, unsigned out_pitch,
              unsigned out_bstride, unsigned scl_off, unsigned scl_bstride);

using namespace qz;

static long g_fail = 0;
#define CHECK(cond, ...)                            \
  do {                                              \
    if (!(cond)) {                                  \
      if (g_fail < 20) {                            \
        printf("FAIL %s:%d: ", __FILE__, __LINE__); \
        printf(__VA_ARGS__);                        \
        printf("\n");                               \
      }                                             \
      g_fail++;                                     \
    }                                               \
  } while (0)

static float ref_scale(float absmax) { return absmax == 0.f ? 1.f : fmaxf(absmax, 1e-30f) / 32767.0f; }
static int ref_q(float x, float s) {
  float t = x / s;
  float r = nearbyintf(t);  // FE_TONEAREST: half to even, as torch.round
  if (r > 32767.f) r = 32767.f;
  if (r < -32767.f) r = -32767.f;
  return (int)r;
}

// ---------------------------------------------------------------- 1. scale, exhaustive per exponent
static void test_scale_exhaustive() {
  long n = 0, f0 = g_fail;
  const int exps[] = {27, 28, 64, 127, 128, 200, 253, 254};  // 27: exponent of 1e-30f (the clamp)
  for (int e : exps) {
    for (uint32_t m = 0; m < (1u << 23); m++) {
      uint32_t ab = ((uint32_t)e << 23) | m;
      uint32_t got = (uint32_t)scale_bits(abits_t(ab));
      uint32_t want = f2u(ref_scale(u2f(ab)));
      CHECK(got == want, "scale a=%08x got %08x want %08x", ab, got, want);
      n++;
    }
  }
  // below the clamp: zero, denormals, small normals
  std::mt19937 rng(1);
  for (int i = 0; i < 2000000; i++) {
    uint32_t ab = i < 1000 ? (uint32_t)i : rng() % (TINY_BITS + 1000);
    uint32_t got = (uint32_t)scale_bits(abits_t(ab));
    uint32_t want = f2u(ref_scale(u2f(ab)));
    CHECK(got == want, "scale(small) a=%08x got %08x want %08x", ab, got, want);
    n++;
  }
  printf("scale_bits: %ld absmax values (8 exponents x all 2^23 significands + 2M small, zero -> 1.0), %ld failures\n",
         n, g_fail - f0);
}

// ---------------------------------------------------------------- 2. reciprocal, exhaustive
static void test_recip_exhaustive() {
  long f0 = g_fail;
  for (uint32_t S = 1u << 23; S < (1u << 24); S++) {
    uint64_t want = (1ULL << 49) / S;
    if (want > (1ULL << 26) - 1) want = (1ULL << 26) - 1;
    uint32_t got = (uint32_t)recip_sig(ap_uint<24>(S));
    CHECK(got == want, "recip S=%06x got %07x want %07llx", S, got, (unsigned long long)want);
  }
  printf("recip_sig (%d bits a step, %d steps): all 2^23 significands, %ld failures\n", RECIP_BITS, RECIP_STEPS,
         g_fail - f0);
}

// ---------------------------------------------------------------- 3. elements
static long n_elem = 0, n_fdiv_fail = 0, n_tie = 0, n_dbl = 0;
static void check_elem(float x, const RowScale& rs) {
  float s = u2f((uint32_t)rs.s_bits);
  int want = ref_q(x, s);
  {  // coverage: fl32(x / s) exactly on a half-integer (a tie), and ties the exact quotient is not on
    float t = fabsf(x / s);
    if (t < 32767.f && t - floorf(t) == 0.5f) {
      n_tie++;
      double te = fabs((double)x / (double)s);  // 53 bits: tells an exact tie from a rounded-onto one
      if (te != (double)t && (long)nearbyint(te) != (long)nearbyintf(t)) n_dbl++;
    }
  }
  int got = (int)quant_elem_recip(fbits_t(f2u(x)), rs);
  CHECK(got == want, "elem x=%a (%08x) s=%a got %d want %d (x/s=%a)", x, f2u(x), s, got, want, x / s);
  int gotf = (int)quant_elem_fdiv(fbits_t(f2u(x)), rs);
  if (gotf != want) n_fdiv_fail++;
  n_elem++;
}

static void test_elements(long rows) {
  std::mt19937_64 rng(12345);
  std::normal_distribution<float> nd(0.f, 1.f);
  long f0 = g_fail;
  for (long r = 0; r < rows; r++) {
    // a random absmax: random bits over the whole useful exponent range, or a "realistic" one
    uint32_t ab;
    int kind = r % 4;
    if (kind == 0) ab = (uint32_t)(rng() % 0x7F000000u);  // any finite magnitude (incl. tiny)
    else if (kind == 1) ab = (uint32_t)((100u + rng() % 60) << 23) | (uint32_t)(rng() & 0x7FFFFF);  // 2^-27..2^32
    else if (kind == 2) ab = f2u(fabsf(nd(rng)) * 8.f + 1e-3f);
    else ab = TINY_BITS - 1000 + (uint32_t)(rng() % 3000);  // around the clamp
    float a = u2f(ab);
    RowScale rs = row_scale(abits_t(ab));
    float s = u2f((uint32_t)rs.s_bits);
    CHECK((uint32_t)rs.s_bits == f2u(ref_scale(a)), "row scale");
    check_elem(a, rs);
    check_elem(-a, rs);
    check_elem(0.f, rs);
    check_elem(-0.f, rs);
    check_elem(u2f(1), rs);            // smallest denormal
    check_elem(u2f(0x807FFFFFu), rs);  // largest negative denormal
    for (int i = 0; i < 64; i++) {      // uniform in [-a, a]
      std::uniform_real_distribution<float> ud(-1.f, 1.f);
      check_elem(ud(rng) * a, rs);
    }
    for (int i = 0; i < 64; i++) {  // random bit patterns with |x| <= a
      uint32_t xb = (uint32_t)(rng() % ((uint64_t)ab + 1));
      check_elem(u2f(xb | ((uint32_t)(rng() & 1) << 31)), rs);
    }
    for (int i = 0; i < 64; i++) {  // around half-integers and integers of x / s
      int k = (int)(rng() % 32768);
      double t = k + ((i & 1) ? 0.5 : 0.0);
      float v = (float)(t * (double)s);
      uint32_t vb = f2u(v);
      for (int dlt = -3; dlt <= 3; dlt++) {
        float x = u2f(vb + dlt);
        if (fabsf(x) <= a) {
          check_elem(x, rs);
          check_elem(-x, rs);
        }
      }
    }
    for (int i = 0; i < 16; i++) {  // small quotients: around 0.5 and 1.5
      double t = (i & 1) ? 1.5 : 0.5;
      uint32_t vb = f2u((float)(t * (double)s));
      for (int dlt = -2; dlt <= 2; dlt++) check_elem(u2f(vb + dlt), rs);
    }
  }
  // the all-zero row: s = 1.0, q(0) = 0
  RowScale rz = row_scale(abits_t(0));
  CHECK((uint32_t)rz.s_bits == ONE_BITS, "zero row scale");
  check_elem(0.f, rz);
  check_elem(-0.f, rz);
  printf("quant_elem_recip: %ld elements over %ld rows, %ld failures; quant_elem_fdiv (host float "
         "division model) %ld failures; coverage: %ld ties of fl32(x/s), %ld double-rounding cases "
         "(round(exact x/s) != rne(fl32(x/s)))\n",
         n_elem, rows, g_fail - f0, n_fdiv_fail, n_tie, n_dbl);
}

// ---- 3b. exhaustive slices: every x significand at every exponent where q can be nonzero, for NS
// row scales (the significand of s first at its extremes 2^23 and 2^24 - 1, then random).
static void test_elements_exhaustive(int ns) {
  std::mt19937_64 rng(99);
  long f0 = g_fail, n = 0;
  for (int k = 0; k < ns; k++) {
    uint32_t sb;  // the scale itself; the absmax only matters through s
    if (k == 0) sb = 0x3F800000u;           // S = 2^23
    else if (k == 1) sb = 0x3FFFFFFFu;      // S = 2^24 - 1
    else sb = (uint32_t)((100 + rng() % 40) << 23) | (uint32_t)(rng() & 0x7FFFFF);
    RowScale rs = row_scale_from_s(ap_uint<32>(sb));
    rs.R = recip_sig(rs.S);
    const int Es = (int)(sb >> 23);
    float s = u2f(sb);
    for (int d = -3; d <= 16; d++) {  // x / s in [2^(d-1), 2^(d+1)): q = 0 below, saturated above
      const uint32_t E = (uint32_t)(Es + d);
      for (uint32_t m = 0; m < (1u << 23); m++) {
        uint32_t xb = (E << 23) | m | ((m & 1) << 31);  // both signs
        float x = u2f(xb);
        int want = ref_q(x, s);
        int got = (int)quant_elem_recip(fbits_t(xb), rs);
        if (got != want) CHECK(false, "exhaustive x=%08x s=%08x got %d want %d", xb, sb, got, want);
        n++;
      }
    }
  }
  printf("quant_elem_recip exhaustive: %d scales x 20 exponents x all 2^23 significands = %ld elements, %ld failures\n",
         ns, n, g_fail - f0);
}

// ---------------------------------------------------------------- 4. the top: items, strides, guards
static const uint32_t MAGIC = 0xA5A5A5A5u;
static void fill_magic(std::vector<word_t>& v) {
  for (auto& w : v)
    for (int j = 0; j < 4; j++) w(32 * j + 31, 32 * j) = MAGIC;
}

struct TopCase {
  int B, N, K;
  int in_off, in_pad, in_gap;    // input: offset, row pitch - K/4, item gap (words)
  int out_off, out_pad, out_gap;  // output: offset, row pitch - K/8 (0 = contiguous rows), item gap
  int scl_gap;                    // scale vectors: item gap beyond ceil(N/4) words
  int dist;
};

static void run_top_case(const TopCase& t, std::mt19937_64& rng, const float* given = nullptr,
                         const int16_t* want_q = nullptr, const float* want_s = nullptr, const char* name = nullptr) {
  const int B = t.B, N = t.N, K = t.K;
  const unsigned in_pitch = K / 4 + t.in_pad, in_bstride = (unsigned)N * in_pitch + t.in_gap;
  const unsigned out_pitch = K / 8 + t.out_pad, out_bstride = (unsigned)N * out_pitch + t.out_gap;
  const unsigned sw = (N + 3) / 4, scl_bstride = sw + t.scl_gap;
  const unsigned out_off = t.out_off, scl_off = out_off + B * out_bstride + 3;
  const size_t in_words = t.in_off + (size_t)B * in_bstride + 2;
  const size_t io_words = scl_off + (size_t)B * scl_bstride + 8;
  std::vector<word_t> in(in_words), io(io_words);
  fill_magic(in);  // gaps hold garbage (NaN-ish bit patterns): must never be read into a row
  fill_magic(io);
  std::vector<float> x((size_t)B * N * K);
  std::normal_distribution<float> nd(0.f, 1.f);
  std::student_t_distribution<float> td(2.f);
  for (int b = 0; b < B; b++)
    for (int n = 0; n < N; n++) {
      const int r = b * N + n;
      float rowmul = std::exp2f((float)((int)(rng() % 40) - 20));
      for (int c = 0; c < K; c++) {
        float v;
        if (given) v = given[(size_t)r * K + c];
        else if (t.dist == 0) v = nd(rng) * rowmul;
        else if (t.dist == 1) v = td(rng) * rowmul;  // heavy tails (activation outliers)
        else v = (r % 5 == 0) ? ((c & 1) ? -0.f : 0.f) : (r % 5 == 1) ? nd(rng) * 1e-36f : nd(rng) * rowmul;
        x[(size_t)r * K + c] = v;
        in[t.in_off + (size_t)b * in_bstride + (size_t)n * in_pitch + c / 4](32 * (c % 4) + 31, 32 * (c % 4)) = f2u(v);
      }
    }
  int ret = quant_top(in.data(), in.data(), io.data(), B, N, K, t.in_off, in_pitch, in_bstride, out_off, out_pitch,
                      out_bstride, scl_off, scl_bstride);
  CHECK(ret == 0, "quant_top returned %d", ret);
  // expected image of io: magic everywhere except the rows and the scale vectors
  std::vector<word_t> want(io_words);
  fill_magic(want);
  long bad_ref = 0;
  for (int b = 0; b < B; b++) {
    for (int n = 0; n < N; n++) {
      const int r = b * N + n;
      float a = 0.f;
      for (int c = 0; c < K; c++) a = fmaxf(a, fabsf(x[(size_t)r * K + c]));
      float s = ref_scale(a);
      if (want_s && f2u(want_s[r]) != f2u(s) && a != 0.f) bad_ref++;
      want[scl_off + (size_t)b * scl_bstride + n / 4](32 * (n % 4) + 31, 32 * (n % 4)) = f2u(s);
      for (int c = 0; c < K; c++) {
        int q = ref_q(x[(size_t)r * K + c], s);
        if (want_q && want_q[(size_t)r * K + c] != q) bad_ref++;
        word_set_q(want[out_off + (size_t)b * out_bstride + (size_t)n * out_pitch + c / 8], c % 8, q_t(q));
      }
    }
    for (int n = N; n < (int)(4 * sw); n++)  // pad lanes of the last scale word
      want[scl_off + (size_t)b * scl_bstride + n / 4](32 * (n % 4) + 31, 32 * (n % 4)) = ONE_BITS;
  }
  long bad = 0;
  for (size_t i = 0; i < io_words; i++)
    if (io[i] != want[i]) bad++;
  CHECK(bad == 0 && bad_ref == 0, "top %s B=%d N=%d K=%d: %ld io words differ from the C reference, %ld vs torch",
        name ? name : "", B, N, K, bad, bad_ref);
  if (name)
    printf("  %-44s %5d x %-5d: %ld io words differ from the C reference, %ld q/s vs torch\n", name, B * N, K, bad,
           bad_ref);
}

static void test_top() {
  std::mt19937_64 rng(7);
  long f0 = g_fail;
  const TopCase cs[] = {
      // B  N    K      in_off pad gap  out_off pad gap  sgap dist
      {1, 1, 16, 0, 0, 0, 0, 0, 0, 0, 0},       {1, 3, 32, 5, 1, 0, 2, 2, 0, 0, 1},
      {1, 50, 720, 0, 0, 0, 0, 0, 0, 0, 0},     {1, 5, 720, 1, 3, 0, 0, 1, 0, 0, 1},
      {1, 81, 960, 0, 0, 0, 0, 0, 0, 0, 1},     {1, 17, 768, 0, 0, 0, 0, 0, 0, 0, 2},
      {12, 37, 64, 3, 128, 0, 1, 0, 0, 0, 0},   // q heads: strided rows, B items
      {2, 21, 320, 0, 0, 4, 0, 0, 7, 2, 1},     // cache k|v: 2 items, gaps
      {3, 2, 32, 0, 0, 0, 0, 0, 0, 1, 2},       {1, 6, 2560, 0, 0, 0, 0, 4, 0, 0, 0},
      {1, 4, 3072, 0, 0, 0, 0, 0, 0, 0, 1},     {1, 2, 4096, 0, 2, 0, 0, 0, 0, 0, 0},
      {1, 3, 1024, 0, 0, 0, 0, 0, 0, 0, 1},     {1, 2, 1040, 0, 0, 0, 0, 0, 0, 0, 0},  // 1024 = the one-pass limit
      {1, 3, 12288, 0, 4, 0, 0, 0, 0, 0, 1},    {1, 130, 48, 0, 0, 0, 0, 0, 0, 0, 2},  // 3 segments
      {5, 1, 64, 0, 0, 0, 0, 0, 0, 0, 0},       {2, 65, 16, 0, 1, 3, 0, 1, 2, 1, 2},
  };
  for (auto& c : cs) run_top_case(c, rng);
  // parameter checks: nothing may be written
  std::vector<word_t> d(64);
  fill_magic(d);
  CHECK(quant_top(d.data(), d.data(), d.data(), 1, 1, 24, 0, 6, 6, 0, 3, 3, 8, 1) == 1, "K %% 16");
  CHECK(quant_top(d.data(), d.data(), d.data(), 1, 2, 32, 0, 7, 14, 0, 4, 8, 8, 1) == 1, "in_pitch");
  CHECK(quant_top(d.data(), d.data(), d.data(), 1, 2, 32, 0, 8, 16, 0, 3, 6, 8, 1) == 1, "out_pitch");
  CHECK(quant_top(d.data(), d.data(), d.data(), 2, 5, 32, 0, 8, 40, 0, 4, 20, 50, 1) == 1, "scl_bstride");
  CHECK(quant_top(d.data(), d.data(), d.data(), 0, 5, 32, 0, 8, 40, 0, 4, 20, 50, 2) == 1, "B = 0");
  CHECK(quant_top(d.data(), d.data(), d.data(), 1, 0, 32, 0, 8, 40, 0, 4, 20, 50, 2) == 1, "N = 0");
  CHECK(quant_top(d.data(), d.data(), d.data(), 1, 1, 0, 0, 8, 40, 0, 4, 20, 50, 2) == 1, "K = 0");
  bool untouched = true;
  for (auto& w : d)
    for (int j = 0; j < 4; j++) untouched &= (uint32_t)w(32 * j + 31, 32 * j) == MAGIC;
  CHECK(untouched, "an invalid command wrote memory");
  printf("quant_top: %zu shapes (items, strides, gaps, offsets, 2-pass rows, zero / denormal rows, segments, "
         "guards) + 7 invalid commands, %ld failures\n",
         sizeof cs / sizeof cs[0], g_fail - f0);
}

// ---------------------------------------------------------------- 5. captured activations
// file: "QZ01", int32 rows, int32 cols, int32 name_len, name, rows*cols float32 x,
//       rows*cols int16 q (torch quant_sym), rows float32 s (torch quant_sym).
static void test_captured(const char* path) {
  FILE* f = fopen(path, "rb");
  if (!f) {
    printf("cannot open %s\n", path);
    g_fail++;
    return;
  }
  long total = 0, total_bad = 0;
  int ncase = 0, ntop = 0;
  char magic[4];
  std::mt19937_64 rng(3);
  while (fread(magic, 1, 4, f) == 4) {
    int32_t rows, cols, nl;
    if (memcmp(magic, "QZ01", 4) || fread(&rows, 4, 1, f) != 1 || fread(&cols, 4, 1, f) != 1 ||
        fread(&nl, 4, 1, f) != 1) {
      printf("bad record in %s\n", path);
      g_fail++;
      break;
    }
    std::vector<char> name(nl + 1, 0);
    std::vector<float> x((size_t)rows * cols), s(rows);
    std::vector<int16_t> q((size_t)rows * cols);
    if (fread(name.data(), 1, nl, f) != (size_t)nl || fread(x.data(), 4, x.size(), f) != x.size() ||
        fread(q.data(), 2, q.size(), f) != q.size() || fread(s.data(), 4, s.size(), f) != s.size()) {
      printf("truncated %s\n", path);
      g_fail++;
      break;
    }
    // element level: the header's functions against torch directly
    long bad = 0;
    for (int r = 0; r < rows; r++) {
      uint32_t a = 0;
      for (int c = 0; c < cols; c++) a = std::max(a, f2u(x[(size_t)r * cols + c]) & 0x7FFFFFFFu);
      RowScale rs = row_scale(abits_t(a));
      if ((uint32_t)rs.s_bits != f2u(s[r]) && a != 0) bad++;
      for (int c = 0; c < cols; c++)
        if ((int)quant_elem(fbits_t(f2u(x[(size_t)r * cols + c])), rs) != q[(size_t)r * cols + c]) bad++;
    }
    total += (long)rows * cols;
    total_bad += bad;
    CHECK(bad == 0, "captured %s: %ld mismatches", name.data(), bad);
    // and through the top (one item; row length a multiple of 16)
    if (cols % 16 == 0) {
      TopCase t = {1, rows, cols, 0, 0, 0, 0, 0, 0, 0, 0};
      run_top_case(t, rng, x.data(), q.data(), s.data(), ntop < 6 ? name.data() : nullptr);
      ntop++;
    }
    ncase++;
  }
  fclose(f);
  printf("captured %s: %d tensors (%d also through quant_top), %ld elements, %ld mismatches vs torch\n", path, ncase,
         ntop, total, total_bad);
}

int main(int argc, char** argv) {
  fesetround(FE_TONEAREST);
  bool quick = false;
  int exh = 0;
  std::vector<const char*> files;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--version")) return 0;  // run_tests.sh: can this binary run at all?
    if (!strcmp(argv[i], "--quick")) quick = true;
    else if (!strcmp(argv[i], "--exhaustive") && i + 1 < argc) exh = atoi(argv[++i]);
    else files.push_back(argv[i]);
  }
  if (!quick) {
    test_scale_exhaustive();
    test_recip_exhaustive();
  }
  test_elements(quick ? 20000 : 400000);
  if (exh) test_elements_exhaustive(exh);
  test_top();
  for (const char* p : files) test_captured(p);
  printf(g_fail ? "TB FAILED: %ld failures\n" : "TB PASSED\n", g_fail);
  return g_fail ? 1 : 0;
}
