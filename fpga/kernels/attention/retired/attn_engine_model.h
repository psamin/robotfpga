// C model of one head-batched call of the int16 linear engine as the attention path uses it
// (fpga/kernels/attention, README.md "Engine call"). Host-only. The per-head arithmetic and layouts are the
// engine's DRIVER CONTRACT (fpga/hls/int16_linear.h); the head loop is the one extension attention needs.
//
// Registers: M (output channels, a multiple of 32), K (reduction, 1..4096), N = n | (H - 1) << 16 (n tokens,
// H head iterations; H = 1 is today's engine), w0..w3, xmem, ymem. Np = 32 ceil(n / 32). Head h uses, with
// dense head strides (no stride register):
//   port p:  w_p  + h * WS, WS = M / 32 * K words: [M/32 groups][K] words, lane l = W[32 g + 8 p + l][k]
//   xmem:    xmem + h * XS, XS = M/4 + Np/4 + Np K / 8 words: [s_row M][s_tok Np][X: Np/32][K][32 tokens]
//   ymem:    ymem + h * YS, YS = M Np / 4 words: Y[M/32][Np][32] FP32
//   y[n][m] = ftz(ftz(float(sum_k X[n][k] W[m][k]) * s_row[m]) * s_tok[n]), the sum exact (int48 in the DSP)
// DDR here = an array of uint32, 4 per 128-bit word, word offsets everywhere.
#pragma once
#include <stdint.h>
#include <string.h>

#include <vector>

namespace attneng {

constexpr uint32_t TB = 32;

struct Call {
  uint32_t M, K, Nreg;     // registers
  uint32_t w[4], x, y;     // word offsets of w0..w3, xmem, ymem
};

static inline uint32_t n_of(uint32_t nreg) { return nreg & 0xFFFF; }
static inline uint32_t h_of(uint32_t nreg) { return ((nreg >> 16) & 0xFF) + 1; }
static inline uint32_t np_of(uint32_t n) { return (n + TB - 1) / TB * TB; }
static inline uint32_t ws_of(const Call& c) { return c.M / 32 * c.K; }
static inline uint32_t xs_of(const Call& c) {
  const uint32_t np = np_of(n_of(c.Nreg));
  return c.M / 4 + np / 4 + np * c.K / 8;
}
static inline uint32_t ys_of(const Call& c) { return c.M * np_of(n_of(c.Nreg)) / 4; }

static inline float u2f(uint32_t u) {
  float f;
  memcpy(&f, &u, sizeof f);
  return f;
}
static inline uint32_t f2u(float f) {
  uint32_t u;
  memcpy(&u, &f, sizeof u);
  return u;
}
static inline float ftz(float f) {  // the Xilinx FP32 cores flush denormals
  uint32_t u = f2u(f);
  if (((u >> 23) & 0xFF) == 0) u &= 0x80000000u;
  return u2f(u);
}
static inline int16_t i16_at(const std::vector<uint32_t>& m, uint64_t word, uint32_t lane) {
  return (int16_t)(uint16_t)(m[4 * word + (lane >> 1)] >> (16 * (lane & 1)));
}

static int engine_call(std::vector<uint32_t>& mem, const Call& c) {
  const uint32_t n = n_of(c.Nreg), H = h_of(c.Nreg);
  if (c.M % 32 || c.M < 32 || c.M > 8192 || c.K < 1 || c.K > 4096 || n < 1) return 1;
  const uint32_t np = np_of(n), WS = ws_of(c), XS = xs_of(c), YS = ys_of(c);
  std::vector<int16_t> xt((size_t)np * c.K), wt((size_t)c.M * c.K);
  for (uint32_t h = 0; h < H; h++) {
    const uint64_t xb = c.x + (uint64_t)h * XS, yb = c.y + (uint64_t)h * YS;
    const uint64_t sr = xb, st = xb + c.M / 4, xx = st + np / 4;
    for (uint32_t t = 0; t < np; t++)
      for (uint32_t k = 0; k < c.K; k++) {
        const uint32_t b = t / 32, a = t % 32;
        xt[(size_t)t * c.K + k] = i16_at(mem, xx + ((uint64_t)b * c.K + k) * 4 + a / 8, a % 8);
      }
    for (uint32_t m = 0; m < c.M; m++) {
      const uint32_t g = m / 32, p = (m % 32) / 8, l = m % 8;
      for (uint32_t k = 0; k < c.K; k++)
        wt[(size_t)m * c.K + k] = i16_at(mem, c.w[p] + (uint64_t)h * WS + (uint64_t)g * c.K + k, l);
    }
    for (uint32_t m = 0; m < c.M; m++) {
      const float srow = u2f(mem[4 * sr + m]);
      for (uint32_t t = 0; t < np; t++) {
        int64_t acc = 0;
        const int16_t* xr = &xt[(size_t)t * c.K];
        const int16_t* wr = &wt[(size_t)m * c.K];
        for (uint32_t k = 0; k < c.K; k++) acc += (int32_t)xr[k] * (int32_t)wr[k];
        const float y = ftz(ftz((float)acc * ftz(srow)) * ftz(u2f(mem[4 * st + t])));
        mem[4 * (yb + ((uint64_t)(m / 32) * np + t) * 8) + m % 32] = f2u(y);
      }
    }
  }
  return 0;
}

}  // namespace attneng
