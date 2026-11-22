// emit_act: GELU (tanh form) and SiLU(gate) x up inside the int16 GEMM's emit (phase B fusion, CONTRACT.md 1.0
// + the P_ACT record parameter of INTEGRATION.md). Header-only, included by fpga/hls/gemm_contract.h after
// fpga/kernels/contract.h. Builds in Vitis HLS 2022.2 and in g++ (the C model).
//
//   GEMM with p6 = P_ACT:
//     EPI_GELU       y[n][m] = gelu_tanh(x[n][m])              SigLIP fc1 (+ bias)           C: M columns
//     EPI_SILU       y[n][m] = silu(x[n][m])                   action_time_mlp_in (+ D row)  C: M columns
//     EPI_SILU_MUL   y[n][16 i + c] = silu(x[n][32 i + c]) * x[n][32 i + 16 + c]   (c < 16)  C: M / 2 columns
//   x = the contract epilogue ((fl(acc) s_row) s_tok [+ bias] [+ D]), bit for bit; the activation is the
//   gelu_silu kernel's front end (gs_emu / gs::gs_fe, NSB 10): h from the ROM, y = fmul(x, h), SILU_MUL
//   y = fmul(up, silu(gate)). So the GEMM's FP32 output is the bits that the unfused GEMM -> GELU / SILU_MUL
//   (FP32 out) pair writes, and QUANT_X of it gives the int16 + scale of the unfused F_QOUT command.
//
// SILU_MUL weight layout (offline packer): the gate|up blob is interleaved per 16 rows, blob row 32 i + c =
// gate row 16 i + c, blob row 32 i + 16 + c = up row 16 i + c (s_row likewise). One 32-row group's emit then
// leaves as (token t: gate half, up half), (token t + 1: ...): the up word of a token follows its gate word
// directly (int16_linear.h emit row r = 2 t + half), so silu(gate) waits 8 clocks in a 16-entry LUTRAM, not a
// tile in BRAM. A row permutation of an integer GEMM is exact.
//
// Datapath (in the dequantizer, gemm_contract.h dequant_ep): 0 DSP added.
//   - Act modes take 2 values a clock (8 clocks per emit word instead of 4). Lanes 0, 1 run the full epilogue
//     on values 4q + 2h + {0, 1} (h = the half of quarter q; h = 1 takes lanes 2, 3 of the quarter from a stash
//     register). Lanes 2, 3 lend their second multiplier: fmul(x, h(x)) or fmul(up, gb). The epilogue then
//     still has 2 fmul + 2 fadd per lane and no other FP operator (HLS: one core per source operation at II 1).
//   - h(x): one dual-port ROM (2 x 1024 x 36 bit, NSB 10: 2 BRAM36) for both act lanes, the shift as a mux tree
//     and the 15 x 12 interpolation multiply in fabric (gs::h_bits does both on DSPs).
//   - Per (group, block) the dequantizer then takes 512 clocks instead of 256. Every act GEMM of -12392 has
//     K >= 720 (compute16 takes K clocks per block), so the MACs never wait: time unchanged.
#pragma once
#include <ap_int.h>
#include <stdint.h>
#include <string.h>

#include "../../contract.h"

#ifndef EA_NSB
#define EA_NSB 10  // build/brief/integration_inputs.json decision 0: gelu_silu table NSB 10
#endif
#include "../../gelu_silu/gelu_silu_tables.h"

#define EA_CAT2(a, b) a##b
#define EA_CAT(a, b) EA_CAT2(a, b)
#define EA_TAB_INIT EA_CAT(GS_TAB_INIT_, EA_NSB)

namespace ea {

constexpr int NSB = EA_NSB;
constexpr int NSEG = 1 << NSB;
constexpr int FB = 12;
constexpr int TB = NSB + FB;
constexpr int VB = 21, DB = 15;

// the act mode of a GEMM record: ctr::EPI_* (0 = none)
constexpr int ACT_NONE = (int)ctr::EPI_NONE;
constexpr int ACT_GELU = (int)ctr::EPI_GELU;
constexpr int ACT_SILU_MUL = (int)ctr::EPI_SILU_MUL;
constexpr int ACT_SILU = (int)ctr::EPI_SILU;
constexpr int GB_DIST = 8;  // iterations between a gate value's write to gb and its up value's read

// the ROM (gelu_silu_tables.h, word V << 15 | D, address mode * NSEG + idx): one copy, 2 read ports
static inline ap_uint<36> rom(ap_uint<NSB + 1> a) {
#pragma HLS INLINE
  static const ap_uint<36> T[2 * NSEG] = EA_TAB_INIT;
#pragma HLS BIND_STORAGE variable = T type = rom_2p impl = bram
  return T[a];
}

// right shift of a W-bit value by a 5-bit amount, a 5-level mux tree (gs::rsh5)
template <int W>
static inline ap_uint<W> rsh5(ap_uint<W> v, ap_uint<5> s) {
#pragma HLS INLINE
  ap_uint<W> r = v;
  if (s[0]) r = r >> 1;
  if (s[1]) r = r >> 2;
  if (s[2]) r = r >> 4;
  if (s[3]) r = r >> 8;
  if (s[4]) r = r >> 16;
  return r;
}

// FP32 bits of h(x) = H 2^-21: gs::h_bits (fpga/kernels/gelu_silu/gelu_silu.h) with no DSP (the GS_SHIFT_LUT
// shift, the D f product in fabric). The testbench checks it against gs::h_bits bit for bit.
static inline ap_uint<32> h_bits(ap_uint<32> xb, bool tsilu) {
#pragma HLS INLINE
  const bool sx = xb[31];
  const ap_uint<8> ex = xb(30, 23);
  const bool xz = ex == 0;
  const ap_uint<24> mx = (ap_uint<1>(1), ap_uint<23>(xb(22, 0)));
  const ap_int<10> sh = (ap_int<10>)(150 + (tsilu ? 4 : 3) - TB) - (ap_int<10>)ex;
  const bool sat = !xz && sh < 24 - TB;
  const bool inr = !xz && sh >= 24 - TB && sh < 24;
  const ap_uint<TB> T = inr ? (ap_uint<TB>)rsh5<24>(mx, (ap_uint<5>)sh) : (ap_uint<TB>)0;
  const ap_uint<NSB> idx = T >> FB;
  const ap_uint<FB> f = T(FB - 1, 0);
  const ap_uint<36> w = rom(((ap_uint<NSB + 1>)(tsilu ? 1 : 0) << NSB) | idx);
  const ap_uint<VB> V = w(35, DB);
  const ap_uint<DB> D = w(DB - 1, 0);
  const ap_uint<DB + FB> df = D * f;
#pragma HLS BIND_OP variable = df op = mul impl = fabric
  const ap_int<VB + FB + 1> t = (ap_int<VB + FB + 1>)(((ap_uint<VB + FB>)V << FB) | (ap_uint<VB + FB>)(1 << (FB - 1))) -
                                (ap_int<VB + FB + 1>)df;
  const ap_uint<VB> g21 = (sat || t < 0) ? (ap_uint<VB>)0 : (ap_uint<VB>)(t >> FB);
  const ap_uint<VB + 1> H = sx ? (ap_uint<VB + 1>)g21 : (ap_uint<VB + 1>)((ap_uint<VB + 1>(1) << VB) - g21);
  const bool z4 = H(21, 6) == 0;
  const ap_uint<22> v4 = z4 ? (ap_uint<22>)(H << 16) : H;
  const bool z3 = v4(21, 14) == 0;
  const ap_uint<22> v3 = z3 ? (ap_uint<22>)(v4 << 8) : v4;
  const bool z2 = v3(21, 18) == 0;
  const ap_uint<22> v2 = z2 ? (ap_uint<22>)(v3 << 4) : v3;
  const bool z1 = v2(21, 20) == 0;
  const ap_uint<22> v1 = z1 ? (ap_uint<22>)(v2 << 2) : v2;
  const bool z0 = !v1[21];
  const ap_uint<22> v = z0 ? (ap_uint<22>)(v1 << 1) : v1;
  const ap_uint<5> npz = (ap_uint<1>(!z4), ap_uint<1>(!z3), ap_uint<1>(!z2), ap_uint<1>(!z1), ap_uint<1>(!z0));
  const ap_uint<32> hb = (ap_uint<1>(0), ap_uint<3>(3), npz, ap_uint<21>(v(20, 0)), ap_uint<2>(0));
  return v[21] ? hb : (ap_uint<32>)0;
}

static inline float hval(float x, bool tsilu) {
#pragma HLS INLINE
  uint32_t u;
  memcpy(&u, &x, 4);
  const uint32_t hb = (uint32_t)h_bits((ap_uint<32>)u, tsilu);
  float h;
  memcpy(&h, &hb, 4);
  return h;
}

// output columns of an act GEMM of Mp blob rows
static inline uint32_t out_cols(uint32_t act, uint32_t M) { return act == (uint32_t)ACT_SILU_MUL ? M / 2 : M; }

}  // namespace ea
