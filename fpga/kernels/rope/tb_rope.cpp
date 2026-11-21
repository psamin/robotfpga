// Host testbench: runs rope_top (the C model of the dataflow top) on the vectors of gen_vectors.py and
// compares every output bit with rope_emu.py. Per case it runs
//   F32 contiguous        x_pitch = y_pitch = 16 H, separate output buffer
//   F32 in place, pitched rows of 16 H words inside a buffer of pitch 16 H + 24 (like K inside the k|v
//                         buffer): checks the result and that the 24 gap words per token are untouched
//   INT16 pitched input   q words, scale words (head-major) and the zero padding lanes
// Build and run (Vitis MinGW g++, clean PATH; see README.md):
//   g++ -std=c++14 -O2 -I/c/Xilinx/Vitis_HLS/2022.2/include tb_rope.cpp rope_top.cpp -o build/tb_rope.exe
//   ./build/tb_rope.exe build/vectors
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <string>
#include <vector>

#include "rope_core.h"

using namespace rope;

void rope_top(const Word* x, const Word* cs, Word* y, Word* s, int T, int H, int x_pitch, int y_pitch, int mode);

template <class T>
static std::vector<T> load(const std::string& path, size_t n) {
  std::vector<T> v(n);
  FILE* f = fopen(path.c_str(), "rb");
  if (!f || fread(v.data(), sizeof(T), n, f) != n) {
    fprintf(stderr, "cannot read %s (%d items)\n", path.c_str(), (int)n);
    exit(2);
  }
  fclose(f);
  return v;
}

static uint32_t lane32(const Word& w, int j) { return (uint32_t)w.range(32 * j + 31, 32 * j); }

// natural x [T][H][64] -> words with token pitch `pitch` (gap words filled with a marker)
static std::vector<Word> pack_rows(const std::vector<uint32_t>& x, int T, int H, int pitch, uint32_t fill) {
  std::vector<Word> w((size_t)T * pitch);
  for (size_t i = 0; i < w.size(); i++)
    for (int j = 0; j < 4; j++) w[i].range(32 * j + 31, 32 * j) = fill + (uint32_t)i;
  for (int t = 0; t < T; t++)
    for (int k = 0; k < H * 64; k++) w[(size_t)t * pitch + k / 4].range(32 * (k % 4) + 31, 32 * (k % 4)) = x[(size_t)t * H * 64 + k];
  return w;
}

static long cmp_rows(const std::vector<Word>& y, int T, int H, int pitch, const std::vector<uint32_t>& ef,
                     const char* name, const char* tag) {
  long bad = 0;
  for (int t = 0; t < T; t++)
    for (int k = 0; k < H * 64; k++) {
      const uint32_t got = lane32(y[(size_t)t * pitch + k / 4], k % 4), want = ef[(size_t)t * H * 64 + k];
      if (got != want) {
        if (bad < 5) printf("  %s %s t=%d k=%d: got %08x want %08x\n", name, tag, t, k, got, want);
        bad++;
      }
    }
  return bad;
}

int main(int argc, char** argv) {
  const std::string dir = argc > 1 ? argv[1] : "build/vectors";
  FILE* cf = fopen((dir + "/cases.txt").c_str(), "r");
  if (!cf) {
    fprintf(stderr, "no %s/cases.txt (run gen_vectors.py)\n", dir.c_str());
    return 2;
  }
  char name[256];
  long bad_total = 0, n_total = 0;
  while (fscanf(cf, "%255s", name) == 1) {
    const std::string d = dir + "/" + name;
    int T = 0, H = 0;
    FILE* mf = fopen((d + "/meta.txt").c_str(), "r");
    if (!mf || fscanf(mf, "%d %d", &T, &H) != 2) return 2;
    fclose(mf);
    const int rows = T * H, bw = H * 16, tq = (T + 3) / 4;
    auto xr = load<uint32_t>(d + "/x.bin", (size_t)rows * 64);
    auto tr = load<uint32_t>(d + "/tab.bin", (size_t)T * 64);
    auto ef = load<uint32_t>(d + "/f32.bin", (size_t)rows * 64);
    auto eq = load<int16_t>(d + "/q0.bin", (size_t)rows * 64);
    auto es = load<uint32_t>(d + "/s0.bin", (size_t)H * tq * 4);
    std::vector<Word> cs((size_t)T * 16);
    for (size_t i = 0; i < cs.size(); i++)
      for (int j = 0; j < 4; j++) cs[i].range(32 * j + 31, 32 * j) = tr[4 * i + j];
    // ---- F32, contiguous
    auto x = pack_rows(xr, T, H, bw, 0);
    std::vector<Word> y((size_t)T * bw);
    rope_top(x.data(), cs.data(), y.data(), nullptr, T, H, bw, bw, MODE_F32);
    const long bad_f = cmp_rows(y, T, H, bw, ef, name, "f32");
    // ---- F32, in place with a token pitch of 16 H + 24 words
    const int pitch = bw + 24;
    auto xp = pack_rows(xr, T, H, pitch, 0xA5000000u);
    const auto xp0 = xp;
    rope_top(xp.data(), cs.data(), xp.data(), nullptr, T, H, pitch, pitch, MODE_F32);
    long bad_p = cmp_rows(xp, T, H, pitch, ef, name, "f32 in place");
    for (int t = 0; t < T; t++)
      for (int i = bw; i < pitch; i++)
        if (xp[(size_t)t * pitch + i] != xp0[(size_t)t * pitch + i]) bad_p++;
    // ---- INT16, pitched input
    std::vector<Word> yq((size_t)rows * 8), ys((size_t)H * tq);
    for (auto& v : ys) v = ~(Word)0;  // must be overwritten, padding lanes with 0
    auto xq = pack_rows(xr, T, H, pitch, 0x5A000000u);
    rope_top(xq.data(), cs.data(), yq.data(), ys.data(), T, H, pitch, 0, MODE_INT16);
    long bad_q = 0, bad_s = 0;
    for (size_t i = 0; i < (size_t)rows * 64; i++) {
      const int16_t got = (int16_t)(uint16_t)yq[i / 8].range(16 * (i % 8) + 15, 16 * (i % 8));
      if (got != eq[i]) {
        if (bad_q < 5) printf("  %s q[%d]: got %d want %d\n", name, (int)i, got, eq[i]);
        bad_q++;
      }
    }
    for (size_t i = 0; i < (size_t)H * tq * 4; i++) {
      const uint32_t got = lane32(ys[i / 4], i % 4);
      if (got != es[i]) {
        if (bad_s < 5) printf("  %s s[h=%d, t=%d]: got %08x want %08x\n", name, (int)(i / (tq * 4)), (int)(i % (tq * 4)), got,
               es[i]);
        bad_s++;
      }
    }
    printf("%-16s T=%3d H=%2d rows=%5d  f32 %ld/%d  f32-in-place %ld/%d  q %ld/%d  s %ld/%d\n", name, T, H, rows,
           bad_f, rows * 64, bad_p, rows * 64, bad_q, rows * 64, bad_s, H * tq * 4);
    bad_total += bad_f + bad_p + bad_q + bad_s;
    n_total += (long)rows * 64 * 3 + H * tq * 4;
  }
  fclose(cf);
  printf("TOTAL mismatches %ld of %ld values -> %s\n", bad_total, n_total, bad_total ? "FAIL" : "PASS");
  return bad_total ? 1 : 0;
}
