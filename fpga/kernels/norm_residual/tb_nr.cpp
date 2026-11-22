// Host testbench of the NORM mode: runs nr_top (the HLS code, compiled by g++) on whole DDR images written by
// gen_vectors.py and compares the image afterwards with the one nr_emul.py expects, byte for byte (outputs
// bit-exact, everything else untouched), and the return code with contract.py's validate.
// Two builds of the same sources:
//   sequential   the processes of the DATAFLOW region run one after another on unbounded streams (C simulation)
//   -DNR_DFSIM   every process in its own thread, every stream bounded at its declared depth (nr_dfsim.h):
//                a FIFO-depth deadlock hangs here as on the IP and the watchdog exits with code 3
//   g++ -std=c++14 -O2 -ffp-contract=off -I<Vitis_HLS>/include -I.. -I../quantize tb_nr.cpp nr_top.cpp -o tb.exe
//   tb.exe [--units | --units-exhaustive] case1.bin ...   exit code 0 iff every unit test and case matches
// The image sits at its DDR address: the ports point at image - base / 16, so the 64-bit / 40-bit address path
// is exercised (cases at 4 GB, 32 GB and 64 GB).
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <random>
#include <vector>

#include "norm_residual.h"

namespace nr {
long g_leftover = 0;
}

int nr_top(const nr::word_t* w2, const nr::word_t* xmem, nr::word_t* ymem, int M, int K, int N);

static int run_case(const char* path) {
  FILE* fh = fopen(path, "rb");
  if (!fh) {
    printf("%s: cannot open\n", path);
    return 1;
  }
  uint32_t magic, W;
  uint64_t base, list;
  int32_t want;
  if (fread(&magic, 4, 1, fh) != 1 || magic != 0x3243524Eu || fread(&base, 8, 1, fh) != 1 ||
      fread(&list, 8, 1, fh) != 1 || fread(&want, 4, 1, fh) != 1 || fread(&W, 4, 1, fh) != 1) {
    printf("%s: bad header\n", path);
    fclose(fh);
    return 1;
  }
  std::vector<uint8_t> in(16 * (size_t)W), ex(16 * (size_t)W);
  if (fread(in.data(), 16, W, fh) != W || fread(ex.data(), 16, W, fh) != W) {
    printf("%s: short image\n", path);
    fclose(fh);
    return 1;
  }
  fclose(fh);
  std::vector<nr::word_t> mem(W);
  for (uint32_t i = 0; i < W; i++) {
    nr::word_t w = 0;
    for (int b = 0; b < 16; b++) w.range(8 * b + 7, 8 * b) = in[16 * (size_t)i + b];
    mem[i] = w;
  }
  nr::word_t* p = mem.data() - (base >> 4);  // word index = byte address / 16, as on the bus
  const long left0 = nr::g_leftover;
  const int rc = nr_top(p, p, p, 0, (int)(uint32_t)(list >> 4), 0);
  const long left = nr::g_leftover - left0;
  long bad = 0, outs = 0;
  for (uint32_t i = 0; i < W; i++) {
    for (int b = 0; b < 16; b++) {
      const uint8_t got = (uint8_t)mem[i].range(8 * b + 7, 8 * b);
      const uint8_t wantb = ex[16 * (size_t)i + b];
      if (wantb != in[16 * (size_t)i + b]) outs++;
      if (got != wantb) {
        if (bad < 6)
          printf("  mismatch at 0x%llx: got %02X want %02X (input %02X)\n",
                 (unsigned long long)(base + 16 * (uint64_t)i + b), got, wantb, in[16 * (size_t)i + b]);
        bad++;
      }
    }
  }
  const char* name = strrchr(path, '/');
  name = name ? name + 1 : path;
  const bool ok = bad == 0 && rc == want && left == 0;
  printf("%-28s base 0x%010llx rc 0x%04x (want 0x%04x) %s: %ld of %zu bytes differ, %ld changed by the call%s\n",
         name, (unsigned long long)base, rc, want, ok ? "BIT-EXACT" : "FAIL", bad, in.size(), outs,
         left ? ", LEFTOVER STREAM DATA" : "");
  return ok ? 0 : 1;
}

// The restoring divider against an exact reference: rn(n / d) 2^k with __int128 long division.
static uint32_t div_ref(unsigned __int128 n, unsigned __int128 d, int k, bool neg) {
  if (n == 0) return 0;
  auto bl = [](unsigned __int128 v) {
    int b = 0;
    while (v) {
      b++;
      v >>= 1;
    }
    return b;
  };
  int s = 24 - (bl(n) - bl(d));
  unsigned __int128 q, r;
  for (;;) {
    const unsigned __int128 nn = s >= 0 ? n << s : n, dd = s >= 0 ? d : d << -s;
    q = nn / dd;
    r = nn % dd;
    if (q >= ((unsigned __int128)1 << 25))
      s--;
    else if (q < ((unsigned __int128)1 << 24))
      s++;
    else
      break;
  }
  uint64_t m = (uint64_t)(q >> 1);
  const bool rb = q & 1, st = r != 0;
  if (rb && (st || (m & 1))) m++;
  int ex = 1 - s;
  if (m == (1ull << 24)) {
    m >>= 1;
    ex++;
  }
  const int field = ex + 23 + 127 + k;
  const uint32_t sign = neg ? 0x80000000u : 0u;
  if (field <= 0) return 0;
  if (field >= 255) return sign | 0x7F800000u;
  return sign | ((uint32_t)field << 23) | ((uint32_t)m & 0x7FFFFF);
}

// rn(1 / sqrt(v)) by plain __int128 integer arithmetic (an independent statement of rsqrt_cr's definition)
static uint32_t rsqrt_ref(uint32_t vb) {
  const int f = (vb >> 23) & 0xFF, e = f - 150;
  uint64_t M = 0x800000u | (vb & 0x7FFFFFu);
  int e2 = e;
  if (e & 1) {
    M <<= 1;
    e2 = e - 1;
  }
  const unsigned __int128 one76 = (unsigned __int128)1 << 76;
  const uint64_t D = (uint64_t)(one76 / M);
  const bool r1 = (one76 % M) != 0;
  uint64_t lo = 0, hi = 1ull << 27;  // isqrt(D) by bisection
  while (hi - lo > 1) {
    const uint64_t mid = (lo + hi) / 2;
    if (mid * mid <= D)
      lo = mid;
    else
      hi = mid;
  }
  const uint64_t R = lo;
  bool st = r1 || R * R != D;
  const bool top = R >= (1ull << 26);
  uint64_t m = top ? R >> 3 : R >> 2;
  const bool rb = top ? (R >> 2) & 1 : (R >> 1) & 1;
  st = st || (top ? (R & 3) != 0 : (R & 1) != 0);
  int field = (top ? 26 : 25) - 38 - e2 / 2 + 127;
  if (rb && (st || (m & 1))) m++;
  if (m == (1ull << 24)) {
    m >>= 1;
    field++;
  }
  return ((uint32_t)field << 23) | ((uint32_t)m & 0x7FFFFFu);
}

static int unit_tests(bool exhaustive) {
  std::mt19937_64 g(7);
  long bad = 0;
  const int NT = 400000;
  for (int it = 0; it < NT; it++) {
    const uint32_t K = 8 * (1 + (uint32_t)(g() % 128));
    const bool sq = it & 1;
    const uint64_t d = sq ? (uint64_t)K * K : K;
    unsigned __int128 n = ((unsigned __int128)(g() & 0xFF) << 64) | g();  // 72 bits
    const int bits = 1 + (int)(g() % 72);
    if (bits < 72) n &= (((unsigned __int128)1) << bits) - 1;
    if (it % 9 == 0) n = (unsigned __int128)d * (g() % 100000);  // exact quotients
    if (it % 13 == 0) n = (unsigned __int128)d * (2 * (g() % 4096) + 1) << (g() % 20);  // ties and near ties
    if (it % 17 == 0) n = (unsigned __int128)1 << (g() % 72);  // powers of two
    const int k = (int)(g() % 520) - 400;
    const bool neg = (it % 3) == 0;
    ap_uint<72> na = 0;
    na(63, 0) = (uint64_t)n;
    na(71, 64) = (uint64_t)(n >> 64);
    const uint32_t got = nr::div_rn<21>(na, ap_uint<21>(d), k, neg);
    const uint32_t want = div_ref(n, d, k, neg);
    if (got != want) {
      if (bad < 5) printf("  div_rn K=%u sq=%d k=%d: got %08X want %08X\n", K, sq, k, got, want);
      bad++;
    }
  }
  printf("unit div_rn (%d random 72-bit / K or K^2 divisions, ties, exact quotients, powers of two): %s\n", NT,
         bad ? "FAIL" : "exact");
  long badf = 0;
  const int NF = 2000000;
  for (int it = 0; it < NF; it++) {
    uint32_t a = (uint32_t)g() & 0x7FFFFFFFu, b = (uint32_t)g() & 0x7FFFFFFFu;
    if (it % 4 == 1) b = (a & 0x7F800000u) | ((uint32_t)g() & 0x7FFFFF);           // equal exponents
    if (it % 4 == 2) b = a - (((uint32_t)(g() % 28)) << 23);                        // close exponents
    if (it % 16 == 3) a = 0x7F7FFFFFu - (uint32_t)(g() % 64);                       // near FLT_MAX
    if (it % 32 == 5) b = (uint32_t)(g() % 0x00800000u);                            // denormal (flushed)
    if (((a >> 23) & 0xFF) == 255 || ((b >> 23) & 0xFF) == 255 || (b & 0x80000000u)) continue;
    const float fa = ctr::ref::ftz(ctr::u2f(a)), fb = ctr::ref::ftz(ctr::u2f(b));
    volatile float s = fa + fb;  // SSE, round to nearest even
    const uint32_t want = ctr::f2u(s), got = nr::fadd_pos(a, b);
    if (got != want) {
      if (badf < 5) printf("  fadd_pos %08X + %08X: got %08X want %08X\n", a, b, got, want);
      badf++;
    }
  }
  printf("unit fadd_pos (%d random non-negative pairs vs the host's IEEE add, inputs flushed): %s\n", NF,
         badf ? "FAIL" : "exact");
  long badr = 0, nr_ = 0, badm = 0;
  // correctly rounded: with v = M' 2^e' (e' even) and the result y = m 2^(f-150), the exact 1 / sqrt(v) lies
  // between the two neighbouring midpoints: (2m-1)^2 M' <= 2^T <= (2m+1)^2 M', T = 302 - 2f - e'
  // (at m = 2^23 the lower neighbour is half an ulp closer: (4m-1)^2 M' <= 2^(T+2)); no ties (see README)
  auto mid_ok = [](uint32_t vb, uint32_t yb) {
    const int f = (vb >> 23) & 0xFF, e = f - 150;
    unsigned __int128 M = 0x800000u | (vb & 0x7FFFFFu);
    int e2 = e;
    if (e & 1) {
      M <<= 1;
      e2 = e - 1;
    }
    const int fy = (yb >> 23) & 0xFF;
    const unsigned __int128 m = 0x800000u | (yb & 0x7FFFFFu);
    const int T = 302 - 2 * fy - e2;
    if (T < 2 || T > 124) return false;
    const unsigned __int128 one = 1;
    const bool lo = m == 0x800000u ? (4 * m - 1) * (4 * m - 1) * M <= (one << (T + 2))
                                   : (2 * m - 1) * (2 * m - 1) * M <= (one << T);
    const bool hi = (one << T) <= (2 * m + 1) * (2 * m + 1) * M;
    return lo && hi;
  };
  auto chk = [&](uint32_t vb) {
    const uint32_t got = nr::rsqrt_cr(vb), want = rsqrt_ref(vb);
    nr_++;
    if (got != want) {
      if (badr < 5) printf("  rsqrt_cr %08X: got %08X want %08X\n", vb, got, want);
      badr++;
    }
    if (!mid_ok(vb, got)) {
      if (badm < 5) printf("  rsqrt_cr %08X -> %08X is not correctly rounded\n", vb, got);
      badm++;
    }
  };
  if (exhaustive) {  // every significand at both exponent parities, plus the range ends
    for (uint32_t m = 0; m < (1u << 23); m++) {
      chk((126u << 23) | m);
      chk((127u << 23) | m);
    }
  } else {
    for (int it = 0; it < 300000; it++) chk(((1 + (uint32_t)(g() % 254)) << 23) | ((uint32_t)g() & 0x7FFFFF));
  }
  for (uint32_t f = 1; f < 255; f++) {
    chk(f << 23);
    chk((f << 23) | 0x7FFFFF);
  }
  printf("unit rsqrt_cr (%ld values%s vs an __int128 reference): %s; exact midpoint test: %s\n", nr_,
         exhaustive ? ", every significand x 2 parities" : " random", badr ? "FAIL" : "exact",
         badm ? "FAIL" : "all correctly rounded");
  return (bad || badf || badr || badm) ? 1 : 0;
}

int main(int argc, char** argv) {
#ifdef NR_DFSIM
  printf("build: threaded DATAFLOW model, streams bounded at their declared depths\n");
#else
  printf("build: sequential C simulation\n");
#endif
  int fails = 0, n = 0;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--units") || !strcmp(argv[i], "--units-exhaustive")) {
      fails += unit_tests(!strcmp(argv[i], "--units-exhaustive"));
      continue;
    }
    fails += run_case(argv[i]);
    n++;
  }
#ifdef NR_DFSIM
  printf("FIFO peaks over all cases (words or entries held / declared depth):\n");
  for (auto& p : dfsim::peaks()) printf("  %-4s %4zu / %4zu\n", p.name.c_str(), p.peak, p.depth);
#endif
  printf("%s: %d case(s), %d failing\n", fails ? "FAIL" : "ALL PASS", n, fails);
  return fails ? 1 : 0;
}
