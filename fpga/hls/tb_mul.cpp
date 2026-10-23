// Bit-exactness test for fstream::bf16_mul (float_linear.h): the one-DSP BF16 x FP32 multiplier must
// give the IEEE round-to-nearest-even FP32 product for every normal operand pair, i.e. the same bits as
// the hardware FP32 multiplier it replaces. Denormal inputs and denormal results are flushed to zero,
// as the Xilinx floating-point operator does; those cases are counted, not compared.
//   g++ -O2 -std=c++14 -ffp-contract=off -o build/tb_mul.exe tb_mul.cpp && build/tb_mul.exe
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "float_linear.h"

using namespace fstream;

static uint32_t f2u(float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  return u;
}
static float u2f(uint32_t u) {
  float f;
  memcpy(&f, &u, 4);
  return f;
}

static long tested = 0, flushed = 0, bad = 0;

static uint64_t rng_state = 0x9E3779B97F4A7C15ull;
static uint64_t rng() {  // xorshift64*: the MinGW 6.2 <random> header crashes that compiler
  rng_state ^= rng_state >> 12;
  rng_state ^= rng_state << 25;
  rng_state ^= rng_state >> 27;
  return rng_state * 2685821657736338717ull;
}

static void check(uint32_t xb, uint16_t wb) {
  const float x = u2f(xb), w = bf16_to_f32(wb);
  const int ex = (xb >> 23) & 0xFF, ew = (wb >> 7) & 0xFF;
  if (ex == 0xFF || ew == 0xFF) return;  // inf/NaN never occur in activations
  const float ref = x * w;               // SSE single precision: exact IEEE RNE product
  const uint32_t rb = f2u(ref);
  const uint32_t got = f2u(bf16_mul(x, wb));
  const bool in_denorm = ex == 0 || ew == 0;
  // the exact product (24 x 8 bits fits a double) below the normal range is flushed, even when IEEE
  // rounding would lift it to 2^-126
  const bool out_denorm = fabs((double)x * (double)w) < ldexp(1.0, -126);
  if (in_denorm || out_denorm) {  // flushed: the result must be a signed zero
    flushed++;
    if ((got & 0x7FFFFFFF) != 0 || (got >> 31) != ((xb >> 31) ^ (wb >> 15))) {
      if (bad++ < 10) printf("flush mismatch x=%08x w=%04x got=%08x\n", xb, wb, got);
    }
    return;
  }
  tested++;
  if (got != rb) {
    if (bad++ < 10) printf("MISMATCH x=%08x w=%04x ref=%08x got=%08x\n", xb, wb, rb, got);
  }
}

int main() {
  // every BF16 weight against a few activations, including all-ones mantissas (round-up to a power of 2)
  const uint32_t xs[] = {0x3F800000u, 0x3FFFFFFFu, 0x00800000u, 0x7F7FFFFFu, 0x3F7FFFFFu, 0x40490FDBu,
                         0xBF800001u, 0x3EAAAAABu, 0x00FFFFFFu, 0x5F000000u, 0x1F000000u, 0x3F000001u};
  for (uint32_t wb = 0; wb < 0x10000u; wb++)
    for (uint32_t xb : xs) check(xb, (uint16_t)wb);
  // random normal pairs over the whole exponent range, and pairs near the exponent limits
  for (long i = 0; i < 40000000L; i++) {
    uint32_t xb = (uint32_t)rng(), wb = (uint32_t)rng() & 0xFFFF;
    if (i % 4 == 0) {  // typical activation / weight magnitudes
      xb = (xb & 0x807FFFFFu) | ((uint32_t)(100 + rng() % 56) << 23);
      wb = (wb & 0x807Fu) | ((uint32_t)(110 + rng() % 36) << 7);
    }
    check(xb, (uint16_t)wb);
  }
  printf("bf16_mul: %ld pairs compared bit-exactly, %ld flushed (denormal in/out), %ld bad\n", tested,
         flushed, bad);
  return bad ? 1 : 0;
}
