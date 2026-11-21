// Unit checks of the int16 engine's arithmetic helpers (int16_linear.h). Linked into the int16 testbench
// (make test-int16 runs `build/tb_float_int16 --units`; one binary, because this laptop's application control
// blocked every freshly built stand-alone unit-test binary); stand-alone:
//   g++ -std=c++14 -O2 -DFSTREAM_ARITH=16 -DINT16_UNITS_MAIN -ffp-contract=off -o tb_int16_units tb_int16_units.cpp
// 1. i43_to_f32 == (float)(int64_t) (round to nearest even) on every |v| < 2^20, every power of two and its
//    neighbours, every tie pattern of each exponent, and 50 M random 43-bit values.
// 2. Never-reset 48-bit accumulators: (capture - previous capture) mod 2^43, sign-extended, is every block sum,
//    over random block lengths up to 4096 and extreme products (+-32767^2), with the 48-bit wrap of the DSP.
// 3. fmul_dq flushes denormal inputs and results to signed zero and is the IEEE product otherwise.
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <random>

#include "int16_linear.h"

using namespace fstream16;

static uint64_t field43(int64_t v) { return (uint64_t)v & ((1ULL << CW) - 1); }
static int64_t sext43(uint64_t f) { return (int64_t)(f << (64 - CW)) >> (64 - CW); }

static long fails = 0, checks = 0;
static void check_conv(int64_t v) {
  checks++;
  const float a = i43_to_f32(field43(v)), b = (float)v;
  if (memcmp(&a, &b, sizeof a)) {
    if (fails++ < 10) printf("i43_to_f32(%lld) = %.9g, (float) = %.9g\n", (long long)v, a, b);
  }
}

int int16_units_main() {
  // 1. conversion
  for (int64_t v = -(1 << 20); v <= (1 << 20); v++) check_conv(v);
  for (int e = 0; e < CW - 1; e++) {
    const int64_t p = 1LL << e;
    for (int64_t d = -3; d <= 3; d++) {
      check_conv(p + d);
      check_conv(-(p + d));
    }
    if (e > 24) {  // ties and near-ties: the bit below the 24 kept bits set, the rest zero / one
      const int sh = e - 23;
      for (int64_t m = (1LL << 23); m < (1LL << 23) + 64; m++) {
        const int64_t base = m << sh, half = 1LL << (sh - 1);
        for (int64_t d : {half - 1, half, half + 1}) {
          check_conv(base + d);
          check_conv(-(base + d));
        }
      }
      const int64_t allones = ((1LL << 24) - 1) << sh;  // rounds up into the next binade
      check_conv(allones + (1LL << (sh - 1)));
      check_conv(-(allones + (1LL << (sh - 1))));
    }
  }
  check_conv((1LL << 42) - 1);
  check_conv(-((1LL << 42) - 1));
  check_conv(-(1LL << 42));
  std::mt19937_64 rng(43);
  for (long i = 0; i < 50000000; i++) {
    const int64_t v = sext43(rng() & ((1ULL << CW) - 1));
    const int sh = (int)(rng() % 43);  // spread over all magnitudes
    check_conv(v >> sh);
  }
  printf("%s: i43_to_f32 vs (float)(int64): %ld values, %ld mismatches\n", fails ? "FAIL" : "PASS", checks, fails);
  const long conv_fails = fails;

  // 2. capture differences of never-reset accumulators
  long dfails = 0, blocks = 0;
  for (int trial = 0; trial < 2000; trial++) {
    Acc acc = 0;
    Cap prev = 0;
    for (int blk = 0; blk < 64; blk++) {
      const int K = trial < 1000 ? (int)(rng() % K_MAX) + 1 : K_MAX;
      const int mode = (int)(rng() % 3);  // random, all +max, all -max products
      int64_t exact = 0;
      for (int k = 0; k < K; k++) {
        int16_t x, w;
        if (mode == 0) {
          x = (int16_t)((int)(rng() % 65535) - 32767);
          w = (int16_t)((int)(rng() % 65535) - 32767);
        } else {
          x = 32767;
          w = mode == 1 ? 32767 : -32767;
        }
        acc_add(acc, x, w);
        exact += (int64_t)x * w;
      }
      const Cap c = cap_of(acc);
      const int64_t got = sext43(cap_diff(c, prev));
      prev = c;
      blocks++;
      if (got != exact && dfails++ < 10) printf("block sum %lld, from captures %lld\n", (long long)exact, (long long)got);
      if (acc >= (1LL << 47) || acc < -(1LL << 47)) dfails++;  // the model keeps a 48-bit value
    }
  }
  printf("%s: block sums from capture differences (48-bit wrap, 43-bit captures): %ld blocks, %ld mismatches\n",
         dfails ? "FAIL" : "PASS", blocks, dfails);

  // 3. denormal flush of the dequantization multiplies
  long ffails = 0;
  const float tiny = 1e-39f;  // denormal
  if (fmul_dq(tiny, 2.0f) != 0.0f || signbit(fmul_dq(-tiny, 2.0f)) == 0) ffails++;
  if (fmul_dq(1e-20f, 1e-20f) != 0.0f) ffails++;  // denormal result
  if (fmul_dq(3.0f, 0.5f) != 1.5f) ffails++;
  for (int i = 0; i < 1000000; i++) {
    float a, b;
    uint32_t ua = (uint32_t)rng(), ub = (uint32_t)rng();
    ua = (ua & 0x807FFFFFu) | ((64u + (ua >> 23) % 128u) << 23);  // normal, no overflow / underflow
    ub = (ub & 0x807FFFFFu) | ((64u + (ub >> 23) % 128u) << 23);
    memcpy(&a, &ua, 4);
    memcpy(&b, &ub, 4);
    const float p = fmul_dq(a, b), q = a * b;
    if (memcmp(&p, &q, 4)) ffails++;
  }
  printf("%s: fmul_dq (flush-to-zero FP32 multiply): %ld mismatches\n", ffails ? "FAIL" : "PASS", ffails);
  return (conv_fails || dfails || ffails) ? 1 : 0;
}

#ifdef INT16_UNITS_MAIN
int main() { return int16_units_main(); }
#endif
