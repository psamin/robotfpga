// C-simulation testbench for the DDR weight-streaming engine (issue #148). Plain g++, no Xilinx tools.
//
//   ./tb_stream policy DIR     DIR from pack.py: table.bin, words.bin, bias.bin, vectors/
//   ./tb_stream linear DIR     DIR with words.bin, x.bin, y_ref.bin, dims.txt
// Prints PASS/FAIL, the weight bytes streamed per inference and the words read per inference.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "stream_engine.h"

using namespace stream;

static std::vector<uint8_t> slurp(const std::string& path, bool required = true) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    if (required) {
      fprintf(stderr, "cannot open %s\n", path.c_str());
      exit(2);
    }
    return {};
  }
  return std::vector<uint8_t>(std::istreambuf_iterator<char>(f), {});
}

static int run_policy(const std::string& dir) {
  auto tb = slurp(dir + "/table.bin"), wb = slurp(dir + "/words.bin"), bb = slurp(dir + "/bias.bin");
  const int n_layers = (int)(tb.size() / sizeof(LayerDesc));
  const LayerDesc* table = (const LayerDesc*)tb.data();
  const Word* wmem = (const Word*)wb.data();
  const int32_t* bmem = (const int32_t*)bb.data();
  int pass = 0, n = 0;
  for (int v = 0; v < 10000; v++) {
    char name[64];
    snprintf(name, sizeof name, "/vectors/%03d_in.bin", v);
    auto in = slurp(dir + name, false);
    if (in.empty()) break;
    snprintf(name, sizeof name, "/vectors/%03d_out.bin", v);
    auto want = slurp(dir + name);
    int8_t out[4096];
    int st = policy_stream(table, n_layers, wmem, bmem, in.data(), out);
    bool ok = st == 0 && memcmp(out, want.data(), want.size()) == 0;
    if (!ok && n - pass < 5) {
      int d = 0;
      while (d < (int)want.size() && out[d] == (int8_t)want[d]) d++;
      printf("FAIL vector %03d: status %d, first mismatch at %d (got %d want %d)\n", v, st, d, out[d], (int8_t)want[d]);
    }
    pass += ok;
    n++;
  }
  printf("%s: policy %d/%d vectors bit-exact, %d layers, %zu weight bytes streamed from DDR per inference\n",
         pass == n && n > 0 ? "PASS" : "FAIL", pass, n, n_layers, wb.size());
  return pass == n && n > 0 ? 0 : 1;
}

static int run_linear(const std::string& dir) {
  int M, K, N;
  FILE* f = fopen((dir + "/dims.txt").c_str(), "r");
  if (!f || fscanf(f, "%d %d %d", &M, &K, &N) != 3) {
    fprintf(stderr, "bad dims.txt in %s\n", dir.c_str());
    return 2;
  }
  fclose(f);
  auto wb = slurp(dir + "/words.bin"), xb = slurp(dir + "/x.bin"), yb = slurp(dir + "/y_ref.bin");
  std::vector<int32_t> y((size_t)N * M);
  int st = linear_stream((const Word*)wb.data(), (const int8_t*)xb.data(), y.data(), M, K, N);
  bool ok = st == 0 && memcmp(y.data(), yb.data(), yb.size()) == 0;
  long passes = (N + N_TILE - 1) / N_TILE;
  printf("%s: linear M=%d K=%d N=%d int32 exact, weight bytes read %ld (%ld pass%s)\n", ok ? "PASS" : "FAIL", M, K,
         N, passes * (long)wb.size(), passes, passes == 1 ? "" : "es");
  return ok ? 0 : 1;
}

int main(int argc, char** argv) {
  if (argc != 3) {
    fprintf(stderr, "usage: %s policy|linear DIR\n", argv[0]);
    return 2;
  }
  std::string mode = argv[1];
  return mode == "policy" ? run_policy(argv[2]) : run_linear(argv[2]);
}
