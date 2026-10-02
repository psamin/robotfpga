// C-model testbench: runs policy_kernel against the golden vectors.
// Builds with plain clang/gcc (make csim) and also works as a Vitis HLS csim testbench
// for the kernel on its own.
//
//   ./tb_kernel [artifacts_dir]     default ../artifacts/standin
#include <stdio.h>
#include <stdlib.h>

#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include <stdint.h>

struct VecSrc {
  std::vector<uint8_t> data;
  size_t pos = 0;
};

struct VecSink {
  std::vector<uint8_t> data;
  std::vector<bool> last;
};

static uint8_t read_byte(VecSrc& s, bool& last) {
  if (s.pos >= s.data.size()) {
    fprintf(stderr, "FAIL: kernel read past the end of a %zu-byte transfer\n", s.data.size());
    exit(1);
  }
  last = s.pos + 1 == s.data.size();
  return s.data[s.pos++];
}

static void write_byte(VecSink& s, uint8_t v, bool last) {
  s.data.push_back(v);
  s.last.push_back(last);
}

#include "policy_kernel.h"

using namespace policy;

static std::vector<uint8_t> slurp(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    fprintf(stderr, "cannot open %s\n", path.c_str());
    exit(2);
  }
  return std::vector<uint8_t>(std::istreambuf_iterator<char>(f), {});
}

// Minimal manifest reader: the 8 "shift" values in layer order.
static std::vector<int> read_shifts(const std::string& path) {
  std::vector<uint8_t> raw = slurp(path);
  std::string text(raw.begin(), raw.end());
  std::vector<int> shifts;
  for (size_t p = text.find("\"shift\""); p != std::string::npos; p = text.find("\"shift\"", p + 1))
    shifts.push_back(atoi(text.c_str() + text.find(':', p) + 1));
  if (shifts.size() != N_LAYERS) {
    fprintf(stderr, "expected %d shifts in %s, found %zu\n", N_LAYERS, path.c_str(), shifts.size());
    exit(2);
  }
  return shifts;
}

static int first_diff(const uint8_t* a, const uint8_t* b, size_t n) {
  for (size_t i = 0; i < n; i++)
    if (a[i] != b[i]) return (int)i;
  return -1;
}

int main(int argc, char** argv) {
  std::string dir = argc > 1 ? argv[1] : "../artifacts/standin";
  std::vector<int> s = read_shifts(dir + "/manifest.json");
  uint32_t lo = s[0] | s[1] << 8 | s[2] << 16 | s[3] << 24;
  uint32_t hi = s[4] | s[5] << 8 | s[6] << 16 | s[7] << 24;
  int failures = 0;

  VecSrc wsrc{slurp(dir + "/weights.bin")};
  VecSink none;
  if (wsrc.data.size() != WEIGHT_FILE_BYTES) {
    fprintf(stderr, "weights.bin is %zu bytes, expected %d\n", wsrc.data.size(), WEIGHT_FILE_BYTES);
    return 1;
  }
  if (policy_kernel(MODE_LOAD_WEIGHTS, wsrc, none, lo, hi) != OK || wsrc.pos != wsrc.data.size()) {
    fprintf(stderr, "FAIL: weight load\n");
    return 1;
  }

  static const int layer_bytes[N_LAYERS] = {48 * 48 * 16, 24 * 24 * 32, 12 * 12 * 64, 6 * 6 * 96,
                                            3 * 3 * 128,  256,          256,          48};
  std::vector<std::vector<int8_t>> layers(N_LAYERS);
  for (int k = 0; k < N_LAYERS; k++) layers[k].resize(layer_bytes[k]);

  int n_vectors = 0;
  for (int v = 0; v < 1000; v++) {
    char name[64];
    snprintf(name, sizeof name, "/vectors/%03d_in.bin", v);
    std::ifstream probe(dir + name);
    if (!probe) break;
    n_vectors++;

    bool dump = v < 5;
    for (int k = 0; k < N_LAYERS; k++) debug_layers[k] = dump ? layers[k].data() : nullptr;

    VecSrc in{slurp(dir + name)};
    VecSink out;
    int status = policy_kernel(MODE_INFER, in, out, lo, hi);
    snprintf(name, sizeof name, "/vectors/%03d_out.bin", v);
    std::vector<uint8_t> want = slurp(dir + name);

    bool tlast_ok = out.last.size() == OUT_LEN && out.last.back();
    for (int i = 0; i + 1 < (int)out.last.size(); i++) tlast_ok &= !out.last[i];
    int d = out.data.size() == want.size() ? first_diff(out.data.data(), want.data(), want.size()) : 0;
    if (status != OK || !tlast_ok || d >= 0 || in.pos != in.data.size()) {
      failures++;
      printf("FAIL vector %03d: status %d, tlast %s, consumed %zu, first mismatch at %d", v, status,
             tlast_ok ? "ok" : "BAD", in.pos, d);
      if (d >= 0 && d < (int)out.data.size()) printf(" (got %d want %d)", (int8_t)out.data[d], (int8_t)want[d]);
      printf("\n");
    }
    for (int k = 0; dump && k < N_LAYERS; k++) {
      snprintf(name, sizeof name, "/vectors/%03d_layer%d.bin", v, k + 1);
      std::vector<uint8_t> ref = slurp(dir + name);
      int ld = ref.size() == layers[k].size()
                   ? first_diff((const uint8_t*)layers[k].data(), ref.data(), ref.size())
                   : 0;
      if (ld >= 0) {
        failures++;
        printf("FAIL vector %03d layer %d: first mismatch at byte %d (got %d want %d)\n", v, k + 1, ld,
               layers[k][ld], (int8_t)ref[ld]);
      }
    }
  }
  for (int k = 0; k < N_LAYERS; k++) debug_layers[k] = nullptr;

  // Error paths: a bad shift still returns 48 bytes, and a short packet's early TLAST is flagged.
  {
    VecSrc in{slurp(dir + "/vectors/000_in.bin")};
    VecSink out;
    int status = policy_kernel(MODE_INFER, in, out, lo & ~0xffu, hi);  // s1 = 0
    if (status != BAD_SHIFT || out.data.size() != OUT_LEN || !out.last.back()) {
      failures++;
      printf("FAIL: shift 0 should give BAD_SHIFT and 48 zero bytes (status %d)\n", status);
    }
  }
  {
    VecSrc in{slurp(dir + "/vectors/000_in.bin")};
    in.data.push_back(0);  // one byte too long: TLAST lands one beat late
    VecSink out;
    if (policy_kernel(MODE_INFER, in, out, lo, hi) != BAD_TLAST) {
      failures++;
      printf("FAIL: misplaced TLAST not reported\n");
    }
  }

  if (n_vectors == 0) {
    printf("FAIL: no vectors found in %s/vectors\n", dir.c_str());
    return 1;
  }
  printf("%s: %d vectors, layer dumps on the first 5, %d failure(s)\n", failures ? "FAIL" : "PASS", n_vectors,
         failures);
  return failures ? 1 : 0;
}
