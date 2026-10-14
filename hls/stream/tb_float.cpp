// C simulation of the full-precision linear engine (issue #148). Plain g++ (+OpenMP), no Xilinx tools.
//   ./tb_float DIR    DIR from pack_float.py or scripts/smolvla_fpga.py: packed.bin, x.bin,
//                     y_ref.bin (float64), sumabs.bin (float64, sum_k |x||w|), dims.txt "Mp K N M"
// PASS if every output is within 1e-5 * sum_k |x_k w_k| of the float64 reference: FP32 rounding of
// a few hundred chained adds stays far inside that, while any indexing or packing bug is O(1).
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include <fstream>
#include <iterator>
#include <string>
#include <vector>

extern "C" int linear_bf16(const uint16_t* w_packed, const float* x, float* y, int M, int K, int N);

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
