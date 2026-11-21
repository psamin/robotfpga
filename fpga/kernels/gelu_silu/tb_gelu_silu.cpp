// Host testbench of gelu_silu_top (C model of the HLS code, g++ with the Vitis HLS headers).
//   tb_gelu_silu CASE_DIR [CASE_DIR ...]
// Each case (make_vectors.py / policy_check.py --dump): meta.txt "mode N K nsb", x.bin, [u.bin], q.bin, s.bin.
// Memory layout as the sequencer builds it (build_smolvla.py): one input buffer, mode 1 rows [gate | up] (pitch
// 2K floats, up at +K floats), modes 0 / 2 rows of K floats with a 16-float pad (pitch test); one output buffer
// with q rows at a pitch of K/8 + 3 words (guard words checked untouched) and the scale vector after them.
// Checks (1) bit equality of q and s with the torch emulation gs_emu.py, after undoing the output word permutation,
// (2) the error against a float64 activation in units of the token's quantisation step s:
//     kernel+quant |q s - y_ref| / s   (the rounding to int16 alone is <= 0.5)
// and prints the cycle-free figures the README quotes. Exit code 1 on any mismatch.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "gelu_silu.h"

int gelu_silu_top(const gs::Word* w0, const gs::Word* w1, const gs::Word* w2, const gs::Word* w3, gs::Word* ymem,
                  int mode, int N, int K, unsigned a, unsigned pa, unsigned b, unsigned pb, unsigned c, unsigned pc,
                  unsigned f);

template <class T>
static std::vector<T> slurp(const std::string& p, size_t n) {
  std::vector<T> v(n);
  FILE* f = fopen(p.c_str(), "rb");
  if (!f) {
    fprintf(stderr, "cannot open %s\n", p.c_str());
    exit(2);
  }
  if (fread(v.data(), sizeof(T), n, f) != n) {
    fprintf(stderr, "short read %s\n", p.c_str());
    exit(2);
  }
  fclose(f);
  return v;
}

static float b2f(uint32_t b) {
  float f;
  memcpy(&f, &b, 4);
  return f;
}
static uint32_t f2b(float f) {
  uint32_t b;
  memcpy(&b, &f, 4);
  return b;
}

static double act_ref(int mode, double x, double u) {
  if (mode == 0) return 0.5 * x * (1.0 + std::tanh(std::sqrt(2.0 / 3.14159265358979323846) * (x + 0.044715 * x * x * x)));
  if (mode == 2) return x / (1.0 + std::exp(-x));
  return x / (1.0 + std::exp(-x)) * u;
}

static int run_case(const std::string& d) {
  int mode, N, K, nsb;
  FILE* f = fopen((d + "/meta.txt").c_str(), "r");
  if (!f || fscanf(f, "%d %d %d %d", &mode, &N, &K, &nsb) != 4) {
    fprintf(stderr, "bad meta in %s\n", d.c_str());
    return 1;
  }
  fclose(f);
  if (nsb != GS_NSB) {
    fprintf(stderr, "%s: vectors for NSB %d, kernel built with %d\n", d.c_str(), nsb, GS_NSB);
    return 1;
  }
  const size_t ne = (size_t)N * K;
  auto x = slurp<float>(d + "/x.bin", ne);
  std::vector<float> u(ne, 0.0f);
  if (mode == 1) u = slurp<float>(d + "/u.bin", ne);
  auto qe = slurp<int16_t>(d + "/q.bin", ne);
  auto se = slurp<uint32_t>(d + "/s.bin", N);

  // input image: rows at pitch pa words; mode 1 [gate | up]
  const unsigned pa = mode == 1 ? (unsigned)(2 * K / 4) : (unsigned)(K / 4 + 4);
  const unsigned a = 0, b = mode == 1 ? (unsigned)(K / 4) : 0u;
  std::vector<gs::Word> in((size_t)N * pa + 4);
  for (int n = 0; n < N; n++)
    for (int k = 0; k < K; k++) {
      const size_t e = (size_t)n * K + k;
      in[a + (size_t)n * pa + k / 4](32 * (k % 4) + 31, 32 * (k % 4)) = f2b(x[e]);
      if (mode == 1) in[b + (size_t)n * pa + k / 4](32 * (k % 4) + 31, 32 * (k % 4)) = f2b(u[e]);
    }
  // output image: q rows at pitch pc (3 guard words per row), then the scales
  const int wq = K / 8;
  const unsigned pc = (unsigned)(wq + 3), c = 0, fo = (unsigned)(N * pc);
  const gs::Word guard = (gs::Word(0xDEADBEEFu) << 96) | (gs::Word(0x12345678u) << 32) | gs::Word(0x5A5A5A5Au);
  std::vector<gs::Word> y((size_t)fo + (N + 3) / 4 + 2, guard);
  const int rc = gelu_silu_top(in.data(), in.data(), in.data(), in.data(), y.data(), mode, N, K, a, pa, b, pa, c, pc, fo);
  if (rc != 0) {
    fprintf(stderr, "%s: kernel returned %d\n", d.c_str(), rc);
    return 1;
  }
  long bad_guard = 0;
  for (int n = 0; n < N; n++)
    for (unsigned g = wq; g < pc; g++)
      if (y[(size_t)n * pc + g] != guard) bad_guard++;
  for (size_t i = fo + (N + 3) / 4; i < y.size(); i++)
    if (y[i] != guard) bad_guard++;
  for (int n = N; n < 4 * ((N + 3) / 4); n++)
    if ((uint32_t)y[fo + n / 4](32 * (n % 4) + 31, 32 * (n % 4)) != 0) bad_guard++;  // padded scale lanes: 0
  long bad_q = 0, bad_s = 0;
  double emax = 0, esum2 = 0, rmax_ok = 0;
  long cnt = 0;
  for (int n = 0; n < N; n++) {
    const uint32_t sb = (uint32_t)y[fo + n / 4](32 * (n % 4) + 31, 32 * (n % 4));
    if (sb != se[n]) {
      if (bad_s < 5) fprintf(stderr, "  s mismatch row %d: %08x vs %08x\n", n, sb, se[n]);
      bad_s++;
    }
    const double s = b2f(sb);
    for (int p = 0; p < K; p++) {
      const int c = p / 8, i = p % 8;
      const int k = gs::LANES == 4 ? p : (i < 4 ? 4 * c + i : K / 2 + 4 * c + (i - 4));  // gs_emu.out_perm(K, LANES)
      const int16_t q = (int16_t)(uint16_t)y[(size_t)n * pc + c](16 * i + 15, 16 * i);
      const size_t e = (size_t)n * K + k;
      if (q != qe[e]) {
        if (bad_q < 5) fprintf(stderr, "  q mismatch row %d k %d: %d vs %d (x %.9g u %.9g)\n", n, k, q, qe[e], x[e], u[e]);
        bad_q++;
      }
      if (s > 0 && std::isfinite(x[e]) && std::fabs(x[e]) < 1e30 && std::fabs(u[e]) < 1e20) {
        const double r = act_ref(mode, x[e], u[e]);
        const double err = std::fabs(q * s - r) / s;
        if (err > emax) emax = err;
        esum2 += err * err;
        cnt++;
      }
    }
  }
  printf("%-28s lanes %d mode %d N %4d K %4d: q mismatches %ld, s mismatches %ld, guard %ld; |q s - y_ref| / s max %.4f "
         "rms %.4f (%ld el.)\n",
         d.c_str(), gs::LANES, mode, N, K, bad_q, bad_s, bad_guard, emax, cnt ? std::sqrt(esum2 / cnt) : 0.0, cnt);
  (void)rmax_ok;
  return (bad_q || bad_s || bad_guard) ? 1 : 0;
}

int main(int argc, char** argv) {
  int fails = 0;
  for (int i = 1; i < argc; i++) fails += run_case(argv[i]);
  // bad-shape guard
  gs::Word dummy[4];
  const int bad[][3] = {{0, 1, gs::KMAX + 8}, {0, 1, 12}, {3, 1, 64}, {0, 0, 64}, {-1, 1, 64}};
  for (const auto& t : bad)
    if (gelu_silu_top(dummy, dummy, dummy, dummy, dummy, t[0], t[1], t[2], 0, 0, 0, 0, 0, 0, 0) != 1) {
      fprintf(stderr, "bad shape mode %d N %d K %d not rejected\n", t[0], t[1], t[2]);
      fails++;
    }
  printf(fails ? "FAIL (%d)\n" : "PASS\n", fails);
  return fails ? 1 : 0;
}
