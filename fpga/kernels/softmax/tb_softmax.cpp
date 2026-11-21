// Host testbench of the softmax kernel: runs softmax_core.h (the HLS source itself, compiled with g++ against the
// Vitis HLS headers) on the DDR images written by make_vectors.py and compares the whole memory image after the
// call, word for word, with the image the bit-exact torch emulation (smx_emu.py) predicts. Unwritten words hold
// 0xA5 canaries, so stray or missing writes fail too. One flat array serves as every port (absolute addresses).
//   build.sh      (compiles both builds and runs every case in build/cases)
//   tb_softmax <case dir> [<case dir> ...]
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <chrono>
#include <string>
#include <vector>

#include "softmax_core.h"

using smx::Word;

static std::vector<Word> load_words(const std::string& path, size_t n) {
  std::vector<Word> v(n, 0);
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) {
    fprintf(stderr, "cannot open %s\n", path.c_str());
    exit(2);
  }
  std::vector<uint32_t> buf(4 * n);
  if (fread(buf.data(), 16, n, f) != n) {
    fprintf(stderr, "short read %s\n", path.c_str());
    exit(2);
  }
  fclose(f);
  for (size_t i = 0; i < n; i++) {
    Word w = 0;
    for (int j = 0; j < 4; j++) w.range(32 * j + 31, 32 * j) = buf[4 * i + j];
    v[i] = w;
  }
  return v;
}

static int run_case(const std::string& dir) {
  FILE* f = fopen((dir + "/params.txt").c_str(), "r");
  if (!f) {
    fprintf(stderr, "cannot open %s/params.txt\n", dir.c_str());
    return 2;
  }
  long long rec_off, total;
  int st_fp1, st_fp0;
  if (fscanf(f, "%lld %lld %d %d", &rec_off, &total, &st_fp1, &st_fp0) != 4) {
    fprintf(stderr, "bad params in %s\n", dir.c_str());
    return 2;
  }
  fclose(f);
  const int want_st = SMX_FP32_OUT ? st_fp1 : st_fp0;
  std::vector<Word> mem = load_words(dir + "/mem.bin", (size_t)total);
  std::vector<Word> expect =
      want_st == smx::ST_OK ? load_words(dir + "/expect.bin", (size_t)total) : std::vector<Word>(mem);

  auto t0 = std::chrono::steady_clock::now();
  const int st = smx::softmax_record(mem.data(), mem.data(), mem.data(), (smx::WAddr)rec_off);
  double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

  long long bad = 0;
  for (long long i = 0; i < total; i++) {
    if (mem[i] != expect[i]) {
      if (bad < 5)
        printf("  mismatch word %lld: got %s want %s\n", i, mem[i].to_string(16).c_str(),
               expect[i].to_string(16).c_str());
      bad++;
    }
  }
  const bool ok = bad == 0 && st == want_st;
  const char* name = strrchr(dir.c_str(), '/');
  printf("%-28s status %d (want %d), %lld words, %lld mismatches (%.1f ms) %s\n", name ? name + 1 : dir.c_str(), st,
         want_st, total, bad, ms, ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s <case dir>...\n", argv[0]);
    return 2;
  }
  int fails = 0;
  for (int i = 1; i < argc; i++) fails += run_case(argv[i]) != 0;
  printf("SMX_FP32_OUT=%d SMX_ENGINE=%d: %d of %d cases passed\n", SMX_FP32_OUT, SMX_ENGINE, argc - 1 - fails,
         argc - 1);
  return fails ? 1 : 0;
}
