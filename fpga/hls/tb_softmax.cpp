// C simulation of the engine's row softmax (mode 1 of float_top): the core in float_linear.h against a
// float64 softmax of the same FP32 inputs. Plain g++ (+OpenMP), no Xilinx tools:
//   g++ -std=c++14 -O2 -Wno-unknown-pragmas -Wno-unused-label -fopenmp -o build/tb_softmax tb_softmax.cpp float_shim.cpp
// With the bit-accurate model of the Xilinx FP exp core the FPGA runs (hls::exp -> HLS_FPO_EXPF):
//   g++ ... -DFSTREAM_FPO_EXP -I$XILINX_HLS/include -o build/tb_softmax_fpo tb_softmax.cpp float_shim.cpp \
//       -L$XILINX_HLS/win64/tools/fpo_v7_1 -lIp_floating_point_v7_1_bitacc_cmodel -lgmp -lmpfr
//   (lnx64/tools/fpo_v7_1 on Linux; that directory must be on PATH / LD_LIBRARY_PATH to run.)
// Rows: random N(0, 1) and N(0, 8) logits, uniform [-50, 50] (large dynamic range), ties at the max,
// LeRobot's masked attention (-2.3819763e38 fill), -inf masks, a constant row, one spike, and lengths
// 4 .. 4096 including rows shorter than SMX_P words. Reports, for the kernel and for a plain FP32
// softmax (expf, sequential sum, division), over the outputs in the normal range: max |error| and max
// error in FP32 ulps against the exact softmax of the FP32 inputs ("exact"), and max ulps against the
// exact softmax of the FP32-rounded differences fl(x_i - m) ("d32": every FP32 softmax, torch's too,
// subtracts the max in FP32 first, and for differences near 100 that rounding alone moves exp by up to
// 2^-18 relative = 64 ulps; d32 isolates the kernel's own exp/sum/normalise error). Also the largest
// |error| among outputs the reference puts below FLT_MIN (the core flushes those to zero), and that the
// batched call equals row-by-row calls bit for bit.
// PASS if the kernel's max |error| <= 2e-6, its max d32 error <= 16 ulp and its max exact error is at
// most 1.5x the plain FP32 softmax's + 4 ulp (an indexing or FIFO bug is O(1) in all three).
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <random>
#include <string>
#include <vector>

extern "C" int softmax_rows_f32(const float* x, float* y, int N, int L);

#if defined(FSTREAM_FPO_EXP)
static const char* KERNEL_EXP = "Xilinx FP exp core (bit-accurate cmodel)";
#else
static const char* KERNEL_EXP = "libm expf";
#endif

static double ulp32(double ref) {
  const float r = (float)ref;
  return (double)nextafterf(r, HUGE_VALF) - (double)r;
}

struct Stats {
  double max_abs = 0, max_ulp = 0, max_ulp_d32 = 0, max_abs_tiny = 0;
  long n = 0, n_tiny = 0;
  void add(double y, double ref, double ref_d32) {
    const double err = fabs(y - ref);
    if (ref >= FLT_MIN) {
      if (err > max_abs) max_abs = err;
      if (err / ulp32(ref) > max_ulp) max_ulp = err / ulp32(ref);
      const double err_d32 = fabs(y - ref_d32) / ulp32(ref_d32);
      if (err_d32 > max_ulp_d32) max_ulp_d32 = err_d32;
      n++;
    } else {
      if (err > max_abs_tiny) max_abs_tiny = err;
      n_tiny++;
    }
  }
  void fold(const Stats& o) {
    if (o.max_abs > max_abs) max_abs = o.max_abs;
    if (o.max_ulp > max_ulp) max_ulp = o.max_ulp;
    if (o.max_ulp_d32 > max_ulp_d32) max_ulp_d32 = o.max_ulp_d32;
    if (o.max_abs_tiny > max_abs_tiny) max_abs_tiny = o.max_abs_tiny;
    n += o.n;
    n_tiny += o.n_tiny;
  }
};

// References in float64 (exact to ~1e-16, far below an FP32 ulp): y = softmax of the FP32 inputs with
// the differences exact; yd = softmax of the FP32-rounded differences fl(x_i - m).
static void softmax64(const float* x, double* y, double* yd, int L) {
  float m = x[0];
  for (int i = 1; i < L; i++) m = x[i] > m ? x[i] : m;
  double s = 0, sd = 0;
  for (int i = 0; i < L; i++) {
    y[i] = exp((double)x[i] - (double)m);
    yd[i] = exp((double)(float)(x[i] - m));
    s += y[i];
    sd += yd[i];
  }
  for (int i = 0; i < L; i++) {
    y[i] /= s;
    yd[i] /= sd;
  }
}

// Plain FP32 softmax: libm expf, one sequential sum, a division per element.
static void softmax32_plain(const float* x, float* y, int L) {
  float m = x[0];
  for (int i = 1; i < L; i++) m = x[i] > m ? x[i] : m;
  float s = 0.0f;
  for (int i = 0; i < L; i++) {
    y[i] = expf(x[i] - m);
    s += y[i];
  }
  for (int i = 0; i < L; i++) y[i] = y[i] / s;
}

struct Case {
  std::string name;
  int N, L;
  std::vector<float> x;
};

int main(int argc, char** argv) {
  const bool big = !(argc > 1 && strcmp(argv[1], "--no-big") == 0);  // --no-big: skip the 12288-row case
  std::mt19937_64 rng(7);
  std::normal_distribution<float> normal(0.0f, 1.0f);
  std::uniform_real_distribution<float> uni(-50.0f, 50.0f);
  std::vector<Case> cases;
  auto add = [&](const char* name, int N, int L, auto fill) {
    Case c{name, N, L, std::vector<float>((size_t)N * L)};
    for (int r = 0; r < N; r++) fill(&c.x[(size_t)r * L], L, r);
    cases.push_back(std::move(c));
  };
  add("N(0,1) logits 64x1024", 64, 1024, [&](float* v, int L, int) { for (int i = 0; i < L; i++) v[i] = normal(rng); });
  add("N(0,8) logits 64x1024", 64, 1024, [&](float* v, int L, int) { for (int i = 0; i < L; i++) v[i] = 8.0f * normal(rng); });
  add("uniform [-50,50] 64x1024", 64, 1024, [&](float* v, int L, int) { for (int i = 0; i < L; i++) v[i] = uni(rng); });
  add("ties at the max 8x1024", 8, 1024, [&](float* v, int L, int r) {
    for (int i = 0; i < L; i++) v[i] = normal(rng);
    const float top = 7.5f + r;
    for (int k = 0; k < 5 + r; k++) v[(k * 131 + r * 17) % L] = top;  // several copies of the max
  });
  add("lerobot masked (-2.38e38 fill) 8x1024", 8, 1024, [&](float* v, int L, int r) {
    for (int i = 0; i < L; i++) v[i] = (i % (3 + r) == 0) ? normal(rng) : -2.3819763e38f;
  });
  add("-inf mask 8x1024", 8, 1024, [&](float* v, int L, int r) {
    for (int i = 0; i < L; i++) v[i] = (i < 100 + 50 * r) ? normal(rng) : -HUGE_VALF;
  });
  add("constant row 2x1024", 2, 1024, [&](float* v, int L, int r) { for (int i = 0; i < L; i++) v[i] = 0.25f * (r + 1); });
  add("one spike (+200) 2x1024", 2, 1024, [&](float* v, int L, int r) {
    for (int i = 0; i < L; i++) v[i] = normal(rng);
    v[300 + r] = 200.0f;
  });
  add("L=4 16 rows", 16, 4, [&](float* v, int L, int) { for (int i = 0; i < L; i++) v[i] = uni(rng); });
  add("L=36 (9 words < SMX_P) 16 rows", 16, 36, [&](float* v, int L, int) { for (int i = 0; i < L; i++) v[i] = normal(rng); });
  add("L=52 (expert self-attn) 16 rows", 16, 52, [&](float* v, int L, int) { for (int i = 0; i < L; i++) v[i] = normal(rng); });
  add("L=180 (expert cross-attn) 16 rows", 16, 180, [&](float* v, int L, int) { for (int i = 0; i < L; i++) v[i] = uni(rng); });
  add("L=1000 (250 words) 8 rows", 8, 1000, [&](float* v, int L, int) { for (int i = 0; i < L; i++) v[i] = normal(rng); });
  add("L=4096 (max) 4 rows", 4, 4096, [&](float* v, int L, int) { for (int i = 0; i < L; i++) v[i] = uni(rng); });
  if (big) {
    add("12288 x 1024 N(0,1) (one SigLIP call)", 12288, 1024, [&](float* v, int L, int) { for (int i = 0; i < L; i++) v[i] = normal(rng); });
  }

  printf("kernel exp: %s\n", KERNEL_EXP);
  printf("%-40s | %11s %9s %7s %9s | %11s %9s %7s %9s | %6s\n", "case", "kern max|e|", "exact ulp", "d32 ulp", "tiny |e|",
         "plain max|e|", "exact ulp", "d32 ulp", "tiny |e|", "n_tiny");
  Stats all_k, all_p;
  bool ok = true;
  for (const Case& c : cases) {
    std::vector<float> y((size_t)c.N * c.L), yp(c.L), yr(c.L);
    std::vector<double> ref(c.L), ref_d32(c.L);
    const int st = softmax_rows_f32(c.x.data(), y.data(), c.N, c.L);
    if (st) {
      printf("FAIL: softmax_rows_f32 status %d on %s\n", st, c.name.c_str());
      return 1;
    }
    Stats sk, sp;
    for (int r = 0; r < c.N; r++) {
      const float* x = &c.x[(size_t)r * c.L];
      softmax64(x, ref.data(), ref_d32.data(), c.L);
      softmax32_plain(x, yp.data(), c.L);
      for (int i = 0; i < c.L; i++) {
        sk.add(y[(size_t)r * c.L + i], ref[i], ref_d32[i]);
        sp.add(yp[i], ref[i], ref_d32[i]);
      }
      if (c.N <= 64) {  // the batched call must equal a row-by-row call bit for bit
        if (softmax_rows_f32(x, yr.data(), 1, c.L) || memcmp(yr.data(), &y[(size_t)r * c.L], sizeof(float) * c.L)) {
          printf("FAIL: batched and single-row results differ on %s row %d\n", c.name.c_str(), r);
          ok = false;
        }
      }
    }
    printf("%-40s | %11.3g %9.2f %7.2f %9.3g | %11.3g %9.2f %7.2f %9.3g | %6ld\n", c.name.c_str(), sk.max_abs, sk.max_ulp,
           sk.max_ulp_d32, sk.max_abs_tiny, sp.max_abs, sp.max_ulp, sp.max_ulp_d32, sp.max_abs_tiny, sk.n_tiny);
    all_k.fold(sk);
    all_p.fold(sp);
  }
  // argument checks: status 1, nothing else
  float dummy[8] = {0};
  ok = ok && softmax_rows_f32(dummy, dummy, 1, 6) == 1 && softmax_rows_f32(dummy, dummy, 0, 4) == 1 &&
       softmax_rows_f32(dummy, dummy, 1, 4100) == 1;
  ok = ok && all_k.max_abs <= 2e-6 && all_k.max_ulp_d32 <= 16 && all_k.max_ulp <= 1.5 * all_p.max_ulp + 4;
  printf("%s: engine softmax (%s) over %ld outputs: max |err| %.3g, max %.2f ulp vs exact, %.2f ulp vs d32, below-FLT_MIN "
         "max |err| %.3g; plain FP32 expf softmax: max |err| %.3g, %.2f ulp vs exact, %.2f ulp vs d32, below-FLT_MIN max |err| %.3g\n",
         ok ? "PASS" : "FAIL", KERNEL_EXP, all_k.n + all_k.n_tiny, all_k.max_abs, all_k.max_ulp, all_k.max_ulp_d32, all_k.max_abs_tiny,
         all_p.max_abs, all_p.max_ulp, all_p.max_ulp_d32, all_p.max_abs_tiny);
  return ok ? 0 : 1;
}
