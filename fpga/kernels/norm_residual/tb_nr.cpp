// Host testbench of the NORM mode: runs nr_top (the HLS code, compiled by g++) on whole DDR images written by
// gen_vectors.py and compares the image afterwards with the one nr_emul.py expects, byte for byte (outputs
// bit-exact, everything else untouched), and the return code with contract.py's validate.
// Two builds of the same sources:
//   sequential   the processes of the DATAFLOW region run one after another on unbounded streams (C simulation)
//   -DNR_DFSIM   every process in its own thread, every stream bounded at its declared depth (nr_dfsim.h):
//                a FIFO-depth deadlock hangs here as on the IP and the watchdog exits with code 3
//   g++ -std=c++14 -O2 -ffp-contract=off -I<Vitis_HLS>/include -I.. -I../quantize tb_nr.cpp nr_top.cpp -o tb.exe
//   tb.exe [--units] case1.bin case2.bin ...       exit code 0 iff every case matches
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

static int unit_tests() {
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
    const int k = (int)(g() % 520) - 400;
    const bool neg = (it % 3) == 0;
    ap_uint<72> na = 0;
    na(63, 0) = (uint64_t)n;
    na(71, 64) = (uint64_t)(n >> 64);
    const ap_uint<5> ld = nr::bitlen((uint32_t)d);
    const ap_uint<72> dn = ap_uint<72>(d) << (72 - (int)ld);
    const uint32_t got = nr::div_rn(na, dn, ld, k, neg);
    const uint32_t want = div_ref(n, d, k, neg);
    if (got != want) {
      if (bad < 5) printf("  div_rn K=%u sq=%d k=%d: got %08X want %08X\n", K, sq, k, got, want);
      bad++;
    }
  }
  printf("unit div_rn (%d random 72-bit / K or K^2 divisions, ties, exact quotients): %s\n", NT,
         bad ? "FAIL" : "exact");
  return bad ? 1 : 0;
}

int main(int argc, char** argv) {
#ifdef NR_DFSIM
  printf("build: threaded DATAFLOW model, streams bounded at their declared depths\n");
#else
  printf("build: sequential C simulation\n");
#endif
  int fails = 0, n = 0;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--units")) {
      fails += unit_tests();
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
