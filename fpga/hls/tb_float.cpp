// C simulation of the full-precision linear engine (issue #148). Plain g++ (+OpenMP), no Xilinx tools.
//   ./tb_float DIR    DIR from pack_float.py or policy/smolvla_fpga.py: packed.bin, x.bin,
//                     y_ref.bin (float64), sumabs.bin (float64, sum_k |x||w|), dims.txt "Mp K N M"
// PASS if every output is within 1e-5 * sum_k |x_k w_k| of the float64 reference: FP32 rounding of
// a few hundred chained adds stays far inside that, while any indexing or packing bug is O(1).
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#if defined(FSTREAM_ARITH) && FSTREAM_ARITH == 16
extern "C" int linear_int16(const uint16_t* w_packed, const float* s_row, const int16_t* xq, const float* s_tok,
                            float* y, int M, int K, int N);
extern "C" int linear_int16_raw(const uint16_t* w_packed, const void* xmem, void* ytiled, int M, int K, int N);
extern "C" void linear_int16_sim_stats(long* v, int reset);
#else
extern "C" int linear_bf16(const uint16_t* w_packed, const float* x, float* y, int M, int K, int N);
#endif

template <class T>
static std::vector<T> slurp(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    fprintf(stderr, "cannot open %s\n", path.c_str());
    exit(2);
  }
  std::vector<char> raw((std::istreambuf_iterator<char>(f)), {});
  std::vector<T> out(raw.size() / sizeof(T));
  memcpy(out.data(), raw.data(), out.size() * sizeof(T));
  return out;
}

#if defined(FSTREAM_ARITH) && FSTREAM_ARITH == 16
// int16 mode (-DFSTREAM_ARITH=16), cases from pack_float.py --tests-int16: dims.txt "Mp K N M int16", packed.bin
// (int16 W packed), srow.bin (FP32 [Mp]), xq.bin (int16 [N][K]), stok.bin (FP32 [N]), y_ref.bin (float64: the
// exact integer product dequantized in float64), y_float.bin (float64: the unquantized x w^T), sumabs.bin
// (float64 sum_k |x_k w_k| of the dequantized operands), xmem.bin (pack_xmem_int16 image, cases with K <= 4096).
// PASS if every output is within 1e-6 * sum|xw| of y_ref (only three FP32 roundings of an exact sum remain: the
// integer -> FP32 conversion and the two scale multiplies, <= 2e-7 relative) and, where xmem.bin exists, one
// raw engine call on the Python-packed DDR image gives exactly the same bits.
static bool file_exists(const std::string& path) {
  FILE* f = fopen(path.c_str(), "rb");
  if (f) fclose(f);
  return f != nullptr;
}

int int16_units_main();  // tb_int16_units.cpp

int main(int argc, char** argv) {
  if (argc == 2 && !strcmp(argv[1], "--units")) return int16_units_main();
  if (argc != 2) {
    fprintf(stderr, "usage: %s DIR | --units\n", argv[0]);
    return 2;
  }
  std::string d = argv[1];
  int Mp, K, N, M;
  char kind[16] = {0};
  FILE* f = fopen((d + "/dims.txt").c_str(), "r");
  if (!f || fscanf(f, "%d %d %d %d %15s", &Mp, &K, &N, &M, kind) != 5 || strcmp(kind, "int16")) {
    fprintf(stderr, "%s: not an int16 case\n", d.c_str());
    return 2;
  }
  fclose(f);
  auto w = slurp<uint16_t>(d + "/packed.bin");
  auto srow = slurp<float>(d + "/srow.bin");
  auto xq = slurp<int16_t>(d + "/xq.bin");
  auto stok = slurp<float>(d + "/stok.bin");
  auto ref = slurp<double>(d + "/y_ref.bin");
  auto yfl = slurp<double>(d + "/y_float.bin");
  auto sa = slurp<double>(d + "/sumabs.bin");
  std::vector<float> y((size_t)N * Mp);
  int st = linear_int16(w.data(), srow.data(), xq.data(), stok.data(), y.data(), Mp, K, N);
  double worst = 0, max_abs = 0, worst_f = 0, max_abs_f = 0;
  for (int n = 0; n < N; n++)
    for (int m = 0; m < M; m++) {
      size_t i = (size_t)n * Mp + m;
      double err = fabs((double)y[i] - ref[i]);
      double norm = err / (sa[i] + 1e-30);
      if (norm > worst) worst = norm;
      if (err > max_abs) max_abs = err;
      double ef = fabs((double)y[i] - yfl[i]);
      double nf = ef / (sa[i] + 1e-30);
      if (nf > worst_f) worst_f = nf;
      if (ef > max_abs_f) max_abs_f = ef;
    }
  bool ok = st == 0 && worst <= 1e-6;
  std::string raw = "n/a (K > 4096)";
  if (const char* dump = getenv("TB_DUMP_Y")) {  // the output bits, to compare builds (e.g. FSTREAM_CHAOS)
    FILE* o = fopen(dump, "wb");
    if (o) {
      fwrite(y.data(), sizeof(float), y.size(), o);
      fclose(o);
    }
  }
  long sim[5] = {0, 0, 0, 0, 0};
  if (file_exists(d + "/xmem.bin")) {
    auto xm = slurp<uint32_t>(d + "/xmem.bin");
    const int np = (N + 31) / 32 * 32;
    std::vector<float> yt((size_t)Mp * np);
    linear_int16_sim_stats(sim, 1);
    int st2 = linear_int16_raw(w.data(), xm.data(), yt.data(), Mp, K, N);
    linear_int16_sim_stats(sim, 1);
    size_t diff = 0;
    for (int n = 0; n < N; n++)
      for (int m = 0; m < Mp; m++) {
        const float a = yt[((size_t)(m / 32) * np + n) * 32 + m % 32], b = y[(size_t)n * Mp + m];
        if (memcmp(&a, &b, sizeof a)) diff++;
      }
    raw = (st2 == 0 && diff == 0) ? "bit-identical" : "MISMATCH (" + std::to_string(diff) + " values)";
    ok = ok && st2 == 0 && diff == 0;
  }
  printf("%s: int16 linear M=%d (padded %d) K=%d N=%d, vs dequantized integer product: max |err| %.3g, max err/sum|xw| "
         "%.3g; vs unquantized float64: max |err| %.3g, max err/sum|xw| %.3g; raw DDR-image call %s",
         ok ? "PASS" : "FAIL", M, Mp, K, N, max_abs, worst, max_abs_f, worst_f, raw.c_str());
  if (sim[0])  // C model clocks of the raw call's compute16 (FIFOs are never late or full unless FSTREAM_CHAOS)
    printf("; compute16 %ld clocks, %ld MAC steps, idle: X %ld, weights %ld, shadow %ld", sim[0], sim[1], sim[2], sim[3],
           sim[4]);
  printf("\n");
  return ok ? 0 : 1;
}
#else
int main(int argc, char** argv) {
  if (argc != 2) {
    fprintf(stderr, "usage: %s DIR\n", argv[0]);
    return 2;
  }
  std::string d = argv[1];
  int Mp, K, N, M;
  FILE* f = fopen((d + "/dims.txt").c_str(), "r");
  if (!f || fscanf(f, "%d %d %d %d", &Mp, &K, &N, &M) != 4) return 2;
  fclose(f);
  auto w = slurp<uint16_t>(d + "/packed.bin");
  auto x = slurp<float>(d + "/x.bin");
  auto ref = slurp<double>(d + "/y_ref.bin");
  auto sa = slurp<double>(d + "/sumabs.bin");
  std::vector<float> y((size_t)N * Mp);
  int st = linear_bf16(w.data(), x.data(), y.data(), Mp, K, N);
  double worst = 0, max_abs = 0;
  for (int n = 0; n < N; n++)
    for (int m = 0; m < M; m++) {
      size_t i = (size_t)n * Mp + m;
      double err = fabs((double)y[i] - ref[i]);
      double norm = err / (sa[i] + 1e-30);
      if (norm > worst) worst = norm;
      if (err > max_abs) max_abs = err;
    }
  bool ok = st == 0 && worst <= 1e-5;
  printf("%s: float linear M=%d (padded %d) K=%d N=%d, max |err| %.3g, max err/sum|xw| %.3g, %zu weight bytes\n",
         ok ? "PASS" : "FAIL", M, Mp, K, N, max_abs, worst, w.size() * 2);
  return ok ? 0 : 1;
}
#endif
