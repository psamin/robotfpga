// Host testbench of the norm_residual kernel: runs nr_top (the HLS code, compiled by g++) on whole DDR images
// written by gen_vectors.py and compares the result with the image nr_emul.py expects, word for word (outputs
// bit-exact, everything else untouched). One shared image serves all four ports (sequencer-mode addressing).
//   g++ -std=c++14 -O2 -ffp-contract=off -I<Vitis_HLS>/include tb_nr.cpp nr_top.cpp -o tb_nr.exe
//   tb_nr.exe case1.bin case2.bin ...       exit code 0 iff every case matches
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <math.h>

#include <random>
#include <vector>

#include "norm_residual.h"

int nr_top(const nr::word_t* hp0, const nr::word_t* hp1, const nr::word_t* hp2, nr::word_t* io, unsigned flags,
           int K, int N, int B, unsigned eps, unsigned a_off, unsigned a_pitch, unsigned a_stride, unsigned b_off,
           unsigned b_pitch, unsigned b_stride, unsigned e_off, unsigned c_off, unsigned c_pitch, unsigned c_stride,
           unsigned d_off, unsigned d_pitch, unsigned d_stride, unsigned f_off, unsigned f_stride);

static nr::word_t to_word(const uint32_t* u) {
  nr::word_t w = 0;
  for (int l = 0; l < 4; l++) w.range(32 * l + 31, 32 * l) = u[l];
  return w;
}

static int run_case(const char* path) {
  FILE* fh = fopen(path, "rb");
  if (!fh) {
    printf("%s: cannot open\n", path);
    return 1;
  }
  uint32_t hdr[2];
  if (fread(hdr, 4, 2, fh) != 2 || hdr[0] != 0x3143524Eu || hdr[1] != 20) {
    printf("%s: bad header\n", path);
    fclose(fh);
    return 1;
  }
  uint32_t c[20], W;
  if (fread(c, 4, 20, fh) != 20 || fread(&W, 4, 1, fh) != 1) {
    printf("%s: short read\n", path);
    fclose(fh);
    return 1;
  }
  std::vector<uint32_t> in(4 * (size_t)W), ex(4 * (size_t)W);
  if (fread(in.data(), 16, W, fh) != W || fread(ex.data(), 16, W, fh) != W) {
    printf("%s: short image\n", path);
    fclose(fh);
    return 1;
  }
  fclose(fh);
  std::vector<nr::word_t> mem(W);
  for (uint32_t i = 0; i < W; i++) mem[i] = to_word(&in[4 * (size_t)i]);
  const int rc = nr_top(mem.data(), mem.data(), mem.data(), mem.data(), c[0], (int)c[1], (int)c[2], (int)c[3], c[4],
                        c[5], c[6], c[7], c[8], c[9], c[10], c[11], c[12], c[13], c[14], c[15], c[16], c[17], c[18],
                        c[19]);
  long bad = 0, outs = 0;
  for (uint32_t i = 0; i < W; i++) {
    for (int l = 0; l < 4; l++) {
      const uint32_t got = (uint32_t)mem[i].range(32 * l + 31, 32 * l);
      const uint32_t want = ex[4 * (size_t)i + l];
      if (want != in[4 * (size_t)i + l]) outs++;
      if (got != want) {
        if (bad < 8)
          printf("  mismatch word %u lane %d: got %08X want %08X (input %08X)\n", i, l, got, want,
                 in[4 * (size_t)i + l]);
        bad++;
      }
    }
  }
  const char* name = strrchr(path, '/');
  name = name ? name + 1 : path;
  printf("%-28s K=%4u N=%3u B=%u flags=0x%02X rc=%d  %s (%ld of %lu u32 differ, %ld changed by the call)\n", name,
         c[1], c[2], c[3], c[0], rc, bad == 0 && rc == 0 ? "BIT-EXACT" : "FAIL", bad, 4ul * W, outs);
  return bad == 0 && rc == 0 ? 0 : 1;
}

// Dimension checks: nothing written, return 1.
static int bad_dims() {
  std::vector<nr::word_t> mem(64, nr::word_t(0x5A));
  int fails = 0;
  const int Ks[] = {0, 4, 12, nr::KMAX + 8};
  for (int K : Ks) {
    const int rc = nr_top(mem.data(), mem.data(), mem.data(), mem.data(), 0x1E, K, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0);
    if (rc != 1) fails++;
  }
  if (nr_top(mem.data(), mem.data(), mem.data(), mem.data(), 0x1E, 64, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0) != 1)
    fails++;
  for (auto& w : mem)
    if (w != nr::word_t(0x5A)) fails++;
  printf("bad-dimension calls: %s\n", fails ? "FAIL" : "rejected, nothing written");
  return fails ? 1 : 0;
}

// Scalar units against independent references: i2f_seq / i2f_rne (all 72-bit widths, ties, sparse values)
// against long double (exact below 2^64) and each other; quant1 against double rne(z * inv) over random z, A.
static int unit_tests() {
  std::mt19937_64 g(5);
  long bad = 0;
  const int NI = 1000000;
  for (int it = 0; it < NI; it++) {
    const int bl = 1 + (int)(g() % 72);
    ap_uint<72> a = 0;
    a(63, 0) = g();
    a(71, 64) = g() & 0xFF;
    if (bl < 72) a = a & ((ap_uint<72>(1) << bl) - 1);
    if (it % 7 == 0) a = (a >> 30) << 30;
    if (it % 11 == 0) a = (ap_uint<72>(1) << (g() % 72)) | (ap_uint<72>(1) << (g() % 72));
    const bool neg = it & 1;
    const uint32_t x = nr::f2u(nr::i2f_rne<72>(a, neg)), y = nr::f2u(nr::i2f72(a, neg));
    if (x != y) bad++;
    if (a < (ap_uint<72>(1) << 64)) {
      long double ref = 0;
      for (int b = 63; b >= 0; b--) ref = ref * 2 + (a[b] ? 1 : 0);
      const uint32_t want = a == 0 ? 0u : nr::f2u((float)(neg ? -ref : ref));  // the kernel's zero is +0
      if (want != x) bad++;
    }
  }
  printf("unit i2f (72-bit, %d values): %s\n", NI, bad ? "FAIL" : "exact");
  long qbad = 0;
  std::uniform_real_distribution<float> u(-1.f, 1.f);
  const int NQ = 2000000;
  for (int it = 0; it < NQ; it++) {
    const float A = expf(u(g) * 40.f);
    const float z = u(g) * A;
    const float r = nr::recip_nr(A);
    const uint32_t ib = nr::f2u(r * 32767.0f) & nr::INV_MASK;
    double p = (double)z * (double)nr::u2f(ib);  // exact: 24 x 17 bits
    double q = nearbyint(p);
    q = q > 32767 ? 32767 : (q < -32767 ? -32767 : q);
    if ((int)nr::quant1(nr::f2u(z), ib) != (int)q) qbad++;
  }
  printf("unit quant1 (%d values): %s\n", NQ, qbad ? "FAIL" : "exact");
  return (bad || qbad) ? 1 : 0;
}

int main(int argc, char** argv) {
  int fails = unit_tests() + bad_dims();
  for (int i = 1; i < argc; i++) fails += run_case(argv[i]);
  printf("%s: %d failing\n", fails ? "FAIL" : "ALL PASS", fails);
  return fails ? 1 : 0;
}
