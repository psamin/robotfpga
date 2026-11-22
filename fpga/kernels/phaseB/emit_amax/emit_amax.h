// emit_amax (phase B fusion, fpga/kernels/phaseB/emit_amax/INTEGRATION.md): the GEMM writes a per-row absmax of
// its FP32 output next to Y, and the next GEMM's X loader quantizes those FP32 rows to int16 itself, in the one
// read it makes anyway. The QUANT_X command between the two disappears. Header-only, included by
// fpga/hls/gemm_contract.h after fpga/kernels/contract.h and int16_linear.h (it uses ctr:: and fstream:: names
// and does not include them itself). Builds in Vitis HLS 2022.2 and in g++ with -I<Vitis_HLS>/include.
//
// Producer: GEMM with p7 = P_AMX_SG != 0 (groups per segment; M / 32 / P_AMX_SG segments).
//   The store takes the absmax of every (row, segment) over the bits it writes (integer max of the 31 magnitude
//   bits, CONTRACT.md 2.1: denormals count, +-0 is 0), across all the 32-wide groups of the segment, and writes
//   it as FP32 bits to slot H:
//     part p of row t:  H + b H.stride + p H.pitch + 4 row        R = p2 = 0:  p = segment, row = t
//                                                                  R != 0:     p = t / R,    row = t % R
//   One word per 4 rows; lanes of rows >= N are 0. Segments are consecutive runs of P_AMX_SG groups; one segment
//   (P_AMX_SG = M / 32) is the whole row.
// Consumer: GEMM with p8 = P_XQ = parts (1..64).
//   A holds FP32 rows (pitch >= 4 K); F holds `parts` absmax vectors (part j at F + b F.stride + j F.pitch, pad4(N)
//   floats). The loader takes a = max over j, s = fl32(max(a, 1e-30) / 32767) (1.0 for a = 0), and
//   q = clamp(rne(fl32(x / s)), +-32767), the contract quantizer (quant_int16.h) bit for bit; s_tok = s 2^p4
//   (quant_int16.h stored_scale: zero rows 1.0, a non-finite absmax stored as itself).
//
// No DSP is added: the absmax is integer compares; the quantizer's products (X R 24 x 26, Qc S 25 x 17) are bound
// to fabric; the per-row divider is quant_int16.h's restoring divider (recip_unrolled, LUT only).
#pragma once
#include <ap_int.h>
#include <stdint.h>
#include <string.h>

#include "quant_int16.h"

namespace famx {

using fstream::F4;
using fstream::IoWord;

#ifndef AMX_QL
#define AMX_QL 4  // FP32 values the X loader quantizes per clock (one io word a clock at 4)
#endif
constexpr int QL = AMX_QL;
static_assert(QL == 1 || QL == 2 || QL == 4, "AMX_QL: 1, 2 or 4 quantizer lanes");
constexpr int QIT = F4 / QL;  // clocks per FP32 io word

typedef ap_uint<31> abits;  // |x| as bits

// ---------------------------------------------------------------- FP32 lanes of an io word as bits
#ifdef __VITIS_HLS__
static inline uint32_t lane_bits(const IoWord& w, int j) {
#pragma HLS INLINE
  return (uint32_t)w.range(32 * j + 31, 32 * j);
}
static inline void set_lane_bits(IoWord& w, int j, uint32_t v) {
#pragma HLS INLINE
  w.range(32 * j + 31, 32 * j) = v;
}
#else
static inline uint32_t lane_bits(const IoWord& w, int j) {
  uint32_t u;
  memcpy(&u, (const char*)&w + 4 * j, 4);
  return u;
}
static inline void set_lane_bits(IoWord& w, int j, uint32_t v) { memcpy((char*)&w + 4 * j, &v, 4); }
#endif

static inline abits amax2(abits a, abits b) {
#pragma HLS INLINE
  return a > b ? a : b;
}
// absmax of the 4 FP32 of a word
static inline abits word_amax(const IoWord& w) {
#pragma HLS INLINE
  return amax2(amax2(abits(lane_bits(w, 0) & 0x7FFFFFFFu), abits(lane_bits(w, 1) & 0x7FFFFFFFu)),
               amax2(abits(lane_bits(w, 2) & 0x7FFFFFFFu), abits(lane_bits(w, 3) & 0x7FFFFFFFu)));
}

// ---------------------------------------------------------------- per-row quantizer state (the loader's bank)
// qz::RowScale without s_bits (the bank never needs it): 84 bits a row.
struct XScale {
  ap_uint<24> S;  // significand of s
  ap_uint<26> R;  // min(floor(2^49 / S), 2^26 - 1)
  ap_uint<25> T;  // 2^25 - S
  ap_uint<9> B;   // 23 + Es
};

// One row's quantizer state from its absmax: exactly the row unit's path (quant_mode.h qz_ss / qz_sr): s =
// scale_bits(a), the restoring divider for R (here unrolled: one row a clock), stored scale with the shift.
static inline XScale xscale(abits a, ap_uint<32>& stored, ap_int<5> shift) {
#pragma HLS INLINE
  const ap_uint<32> s = qz::scale_bits(qz::abits_t(a));
  const qz::RowScale rs = qz::row_scale_from_s(s);
  XScale x;
  x.S = rs.S;
  x.R = qz::recip_unrolled(rs.S);
  x.T = rs.T;
  x.B = rs.B;
  stored = qz::stored_scale(qz::abits_t(a), s, shift);
  return x;
}

// q = clamp(rne(fl32(x / s)), +-32767): quant_int16.h quant_elem_recip, the same operations, with both products in
// fabric (the header's version puts X R on two DSPs and Qc S on one). tb_amax checks it against qz::quant_elem and
// ctr::ref::quant_elem on random and edge values.
static inline qz::q_t quant_elem_fab(qz::fbits_t xb, const XScale& rs) {
#pragma HLS INLINE
  ap_uint<8> E = xb(30, 23);
  bool neg = xb[31];
  ap_uint<24> X = (ap_uint<1>(1), ap_uint<23>(xb(22, 0)));
  ap_uint<50> P = X * rs.R;  // 24 x 26
#pragma HLS BIND_OP variable = P op = mul impl = fabric latency = 3
  ap_uint<26> Qc = P >> 24;  // floor(X 2^25 / S) - {0, 1}
  ap_uint<25> Qlo = Qc(24, 0);
  ap_uint<17> Slo = rs.S(16, 0);
  ap_uint<7> Shi = rs.S(23, 17);
  ap_uint<42> p1 = Qlo * Slo;  // 25 x 17 (only its low 25 bits are used)
#pragma HLS BIND_OP variable = p1 op = mul impl = fabric latency = 2
  ap_uint<8> qlo = Qc(7, 0);
  ap_uint<15> p2 = qlo * Shi;  // 8 x 7
#pragma HLS BIND_OP variable = p2 op = mul impl = fabric latency = 1
  ap_uint<25> lo = p1(24, 0) + (ap_uint<25>(p2(7, 0)) << 17);  // (Qc S) mod 2^25
  bool lz = lo == 0;
  bool ge = !lz && lo <= rs.T;
  bool st0 = ge ? lo != rs.T : !lz;
  ap_uint<26> Q = Qc + ap_uint<26>(ge);
  bool big = Q[25];
  ap_uint<26> Z = big ? Q : ap_uint<26>(Q << 1);
  ap_uint<25> M = Z >> 2;
  bool rb = Z[1], st = Z[0] | st0;
  if (rb & (st | M[0])) M += 1;
  ap_int<10> sh = ap_int<10>(rs.B) - ap_int<10>(E) + (big ? 0 : 1);
  qz::q_t q = qz::rne_sat(M, sh, neg);
  return E == 0 ? qz::q_t(0) : q;
}

}  // namespace famx
