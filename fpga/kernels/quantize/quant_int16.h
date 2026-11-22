// Per-token int16 activation quantizer: the reference arithmetic every int16 producer agrees with
// (the int16 engine's emit path, the norm / activation / attention / rope kernels and the standalone
// QUANT mode in quant_mode.h). Header only, no state: include it and call the functions.
//
// Definition (build/brief/precision_emulation.py quant_sym(x, 32767, dim=1), bit for bit):
//     a = max_j |x_j|                             exact (an integer max over the magnitude bits)
//     s = fl32(max(a, 1e-30f) / 32767)            one correctly rounded FP32 division per row
//     q_j = clamp(rne(fl32(x_j / s)), +-32767)    FP32 division, then round half to even
// with one convention from fpga/hls/pack_float.py (_quantize_int16, the int16 engine's host packer):
//     a == 0 (all-zero row, +0 or -0) -> s = 1.0f.  q = 0 either way, so y = acc * s_row * s_tok = 0
//     (quant_sym itself would store s = fl32(1e-30f / 32767) = 0x07A03A6C for such a row).
// Rows with 0 < a < 1e-30 (denormal / tiny) use the clamp exactly like quant_sym.
//
// No FP32 divider per element. Per row (row_scale):
//     s exactly from a: a 41-bit by 32767 division with shifts and adds (32767 = 2^15 - 1),
//     S = significand of s (24 bits), R = min(floor(2^49 / S), 2^26 - 1) by restoring division,
//     RECIP_BITS quotient bits per step (recip_init / recip_step / recip_result; recip_sig loops).
// Per element (quant_elem), x = X 2^(E-150), s = S 2^(Es-150), X, S 24-bit significands:
//     Qc = (X R) >> 24 is floor(X 2^25 / S) or one less (error < X / 2^24 < 1),
//     rem = X 2^25 - Qc S lies in [0, 2S) < 2^25, so only (Qc S) mod 2^25 is needed: a 25 x 17 DSP
//     product plus an 8 x 7 fabric product. rem >= S corrects Qc by one; rem != 0 is the sticky bit.
//     That gives fl32(X / S) exactly (24 bits + round + sticky); fl32(x / s) = fl32(X / S) 2^(E-Es)
//     is then rounded half to even to an integer: a 17-bit right shift and a prefix-OR sticky.
//     (Rounding fl32(x / s) and not the exact x / s matters: double-rounding cases are common,
//     see tb_quant's coverage counts.)
// Cost per lane: 3 DSP48E2 (X R is 24 x 26 = two DSPs, Qc S one), no BRAM, a few hundred LUT
// (README.md has the synthesised numbers). QZ_IMPL_FDIV instead uses the FP32 divider core per lane
// (x / s, IEEE round to nearest even; denormal x flushed to 0 cannot change q: |x / s| < 4e-4).
//
// This is the contract quantizer (fpga/kernels/CONTRACT.md 2.1, contract.h ctr::ref::quant_row,
// contract.py quant_rows): every int16 producer of float_top uses these functions. The STORED scale
// (stored_scale) is s * 2^shift (shift = record parameter p4 in [-8, 8], an exponent add; Q uses -3 for
// the attention's 1/8); an all-zero row stores 1.0 whatever the shift.
// Non-finite rows: q is undefined (finite garbage, no hang), but the stored scale is not hidden: a row whose
// absmax is +Inf stores +Inf, a NaN row stores that NaN (as ctr::ref / torch, whose s = amax / 32767 is
// non-finite too), so a divergence reaches the next GEMM's output instead of vanishing.
// Plain C++ on ap_int.h: the same code builds in Vitis HLS and in g++ with -I<Vitis_HLS>/include.
#pragma once
#include <ap_int.h>
#include <stdint.h>
#include <string.h>

namespace qz {

constexpr int QMAX = 32767;
constexpr uint32_t TINY_BITS = 0x0DA24260u;  // 1e-30f, precision_emulation.py TINY
constexpr uint32_t ONE_BITS = 0x3F800000u;   // 1.0f: the scale of an all-zero row (pack_float.py)
#ifndef QZ_RECIP_BITS
#define QZ_RECIP_BITS 1  // one conditional subtract a clock meets 250 MHz in the HLS estimate; 2+ do not
#endif
constexpr int RECIP_BITS = QZ_RECIP_BITS;                         // quotient bits per recip_step
constexpr int RECIP_STEPS = (25 + RECIP_BITS - 1) / RECIP_BITS;  // steps after recip_init

typedef ap_uint<32> fbits_t;  // an FP32 as its bit pattern
typedef ap_uint<31> abits_t;  // |x| as a bit pattern (non-negative floats order like unsigned ints)
typedef ap_int<16> q_t;

static inline uint32_t f2u(float f) {
#pragma HLS INLINE
  uint32_t u;
  memcpy(&u, &f, sizeof u);
  return u;
}
static inline float u2f(uint32_t u) {
#pragma HLS INLINE
  float f;
  memcpy(&f, &u, sizeof f);
  return f;
}

// ---------------------------------------------------------------- row statistics
// Running absmax of a row: integer max over the 31 magnitude bits (exact, no FP compare, denormals
// included; -0 and +0 both give 0).
static inline abits_t abs_bits(fbits_t xb) {
#pragma HLS INLINE
  return xb(30, 0);
}
static inline abits_t amax2(abits_t a, abits_t b) {
#pragma HLS INLINE
  return a > b ? a : b;
}

// Everything the per-element quantizer needs about one row.
struct RowScale {
  ap_uint<32> s_bits;  // the FP32 scale (what goes to DDR next to the int16 row)
  ap_uint<24> S;       // significand of s with the hidden bit
  ap_uint<26> R;       // min(floor(2^49 / S), 2^26 - 1)
  ap_uint<25> T;       // 2^25 - S (the remainder test without a negation)
  ap_uint<9> B;        // 23 + Es, Es = biased exponent of s (s is always normal: 12 <= Es <= 240)
};

// s = fl32(max(a, 1e-30f) / 32767) exactly, or 1.0f for a == 0. a_bits: absmax as bits (finite).
static inline ap_uint<32> scale_bits(abits_t a_bits) {
#pragma HLS INLINE
  abits_t a = a_bits > abits_t(TINY_BITS) ? a_bits : abits_t(TINY_BITS);  // normal from here on
  ap_uint<8> Ea = a(30, 23);
  ap_uint<24> A = (ap_uint<1>(1), ap_uint<23>(a(22, 0)));
  ap_uint<41> N = ap_uint<41>(A) << 17;
  // N / (2^15 - 1) = (N / 2^15) (1 + 2^-15 + 2^-30 + ...): q is the exact quotient or one less.
  ap_uint<42> sum = ap_uint<42>(N) + (N >> 15) + (N >> 30);
  ap_uint<27> q = sum >> 15;
  ap_int<43> rem = ap_int<43>(N) - ((ap_int<43>(q) << 15) - ap_int<43>(q));
  if (rem >= QMAX) {
    q += 1;
    rem -= QMAX;
  }
  bool st0 = rem != 0;
  // q = floor(A 2^17 / 32767) has 26 or 27 bits; keep 24, round half even (ties impossible).
  ap_uint<25> M;
  bool rb, st;
  ap_uint<9> Es;
  if (q[26]) {
    M = q >> 3;
    rb = q[2];
    st = q[1] | q[0] | st0;
    Es = ap_uint<9>(Ea) - 14;
  } else {
    M = q >> 2;
    rb = q[1];
    st = q[0] | st0;
    Es = ap_uint<9>(Ea) - 15;
  }
  if (rb & (st | M[0])) M += 1;
  if (M[24]) {  // rounded up to 2^24: renormalize
    M = ap_uint<25>(1) << 23;
    Es += 1;
  }
  ap_uint<32> s = (ap_uint<32>(Es(7, 0)) << 23) | ap_uint<32>(M(22, 0));
  return a_bits == 0 ? ap_uint<32>(ONE_BITS) : s;
}

// ---- R = min(floor(2^49 / S), 2^26 - 1), restoring division, RECIP_BITS bits per step.
// S > 2^23: 2^49 / S is in (2^25, 2^26), its top bit (2^25) is 1 with remainder 2^24 - S; the
// remaining 25 bits come one per conditional subtract. S == 2^23 (2^49 / S = 2^26) is clamped.
struct RecipState {
  ap_uint<25> r;  // partial remainder (< S)
  ap_uint<26> q;  // quotient bits so far (left-aligned as they are produced)
  ap_uint<5> n;   // quotient bits still to produce
  bool pow2;      // S == 2^23
};
// The start state depends on S only; recip_seed can sit in an earlier pipeline stage than recip_load.
struct RecipSeed {
  ap_uint<25> r0;  // 2^24 - S
  bool pow2;       // S == 2^23
};
static inline RecipSeed recip_seed(ap_uint<24> S) {
#pragma HLS INLINE
  RecipSeed sd;
  sd.r0 = (ap_uint<25>(1) << 24) - S;
  sd.pow2 = S == (ap_uint<24>(1) << 23);
  return sd;
}
static inline RecipState recip_load(const RecipSeed& sd) {
#pragma HLS INLINE
  RecipState st;
  st.r = sd.r0;
  st.q = 1;
  st.n = 25;
  st.pow2 = sd.pow2;
  return st;
}
static inline RecipState recip_init(ap_uint<24> S) {
#pragma HLS INLINE
  return recip_load(recip_seed(S));
}
static inline void recip_step(RecipState& st, ap_uint<24> S) {
#pragma HLS INLINE
  for (int k = 0; k < RECIP_BITS; k++) {
#pragma HLS UNROLL
    if (st.n != 0) {
      ap_uint<25> r2 = st.r << 1;  // st.r < S < 2^24, so no bit is lost
      bool ge = r2 >= S;
      st.r = ge ? ap_uint<25>(r2 - S) : r2;
      st.q = (st.q << 1) | ap_uint<26>(ge);
      st.n -= 1;
    }
  }
}
static inline ap_uint<26> recip_result(const RecipState& st) {
#pragma HLS INLINE
  return st.pow2 ? ap_uint<26>((1u << 26) - 1) : st.q;
}
// All 25 steps (a loop of RECIP_STEPS; for code where per-row latency does not matter).
static inline ap_uint<26> recip_sig(ap_uint<24> S) {
#pragma HLS INLINE
  RecipState st = recip_init(S);
recip:
  for (int i = 0; i < RECIP_STEPS; i++) {
#pragma HLS PIPELINE II = 1
    recip_step(st, S);
  }
  return recip_result(st);
}

// The same 25 steps fully unrolled: for a pipelined loop that needs one R per clock (the row unit's sc
// stage). 25 compare-subtract stages of 25 bits, about 1.2k LUT in all, no DSP; HLS registers the chain to
// meet the clock. Call it once per hardware instance: every call site inside a pipelined loop is its own
// 25-stage divider (an iterative recip_step engine is ~0.4k LUT but delivers one R per 27 clocks).
static inline ap_uint<26> recip_unrolled(ap_uint<24> S) {
#pragma HLS INLINE
  RecipState st = recip_init(S);
  for (int i = 0; i < RECIP_STEPS; i++) {
#pragma HLS UNROLL
    recip_step(st, S);
  }
  return recip_result(st);
}

// The scale written to DDR next to the int16 row (CONTRACT.md 2.1): s * 2^shift for a nonzero finite
// absmax (s is normal with a biased exponent in [12, 240], so shift in [-8, 8] keeps it normal and exact),
// 1.0 for an all-zero row, and the absmax itself (+Inf or the NaN) for a non-finite row.
static inline ap_uint<32> stored_scale(abits_t a_bits, ap_uint<32> s_bits, ap_int<5> shift) {
#pragma HLS INLINE
  const bool nonfinite = a_bits(30, 23) == 255;
  ap_uint<32> sh = s_bits;  // s is positive and normal: only its 8-bit exponent field moves (no carry out)
  sh(30, 23) = ap_uint<8>(s_bits(30, 23) + ap_uint<8>(ap_int<8>(shift)));
  return a_bits == 0 ? ap_uint<32>(ONE_BITS) : nonfinite ? ap_uint<32>(a_bits) : sh;
}

// Row parameters from s (R left out: fill it with recip_* from S).
static inline RowScale row_scale_from_s(ap_uint<32> s_bits) {
#pragma HLS INLINE
  RowScale rs;
  rs.s_bits = s_bits;
  rs.S = (ap_uint<1>(1), ap_uint<23>(s_bits(22, 0)));
  rs.R = 0;
  rs.T = (ap_uint<26>(1) << 25) - rs.S;
  rs.B = ap_uint<9>(s_bits(30, 23)) + 23;
  return rs;
}
static inline RowScale row_scale(abits_t a_bits) {
#pragma HLS INLINE
  RowScale rs = row_scale_from_s(scale_bits(a_bits));
  rs.R = recip_sig(rs.S);
  return rs;
}

// ---------------------------------------------------------------- per element
// Round M 2^-sh half to even (M <= 2^24), saturate at 32767, apply the sign. sh <= 8 means
// value >= 2^15 (saturate; unreachable when |x| <= the row's absmax), sh >= 26 means value < 0.5.
static inline q_t rne_sat(ap_uint<25> M, ap_int<10> sh, bool neg) {
#pragma HLS INLINE
  // t = sh - 1 in 8..24 is the position of the round bit; k = t - 8 in 0..16.
  const ap_uint<5> k = ap_uint<5>(sh - 9);
#ifdef QZ_SHIFT_DSP
  // The right shift by sh = 9 + k as one DSP product P = M 2^(16 - k) (25 x 17 bits, a one-hot multiplier):
  // floor(M / 2^sh) = P[40:25], the round bit M[t] = P[24], and every bit of M under it lands in P[23:0]
  // (the bits under P[16 - k] are 0), so the sticky bit is P[23:0] != 0. 1 DSP48E2 per lane instead of a
  // 5-stage sticky barrel shifter in fabric. Measured (csynth v5, 4 lanes): +4 DSP and +0.9k LUT, the deeper
  // pipeline costs more shift registers than the shifter saves, so the fabric shifter is the default.
  ap_uint<17> oh = 0;
  for (int i = 0; i <= 16; i++) {
#pragma HLS UNROLL
    oh[i] = k == ap_uint<5>(16 - i);
  }
  const ap_uint<41> P = ap_uint<41>(M) * oh;
#pragma HLS BIND_OP variable = P op = mul impl = dsp
  const bool rb = P[24];
  const bool st = P(23, 0) != 0;
  const ap_uint<16> ip = P(40, 25);  // floor(M / 2^sh) <= 2^15
#else
  // Constant shifts and explicit muxes only (a variable shift or bit index becomes a 32-bit barrel shifter in
  // HLS). A sticky barrel shifter: every stage ORs the bits it drops into st, so st ends as
  // OR(M[0 .. t-1]), any bit of M under the round bit.
  const ap_uint<17> Y0 = M >> 8;  // floor(M / 2^t) after the 5 stages; <= 2^16
  const bool s0 = M(7, 0) != 0;
  const ap_uint<17> Y1 = k[0] ? ap_uint<17>(Y0 >> 1) : Y0;
  const bool s1 = s0 | (k[0] & Y0[0]);
  const ap_uint<17> Y2 = k[1] ? ap_uint<17>(Y1 >> 2) : Y1;
  const bool s2 = s1 | (k[1] & (Y1(1, 0) != 0));
  const ap_uint<17> Y3 = k[2] ? ap_uint<17>(Y2 >> 4) : Y2;
  const bool s3 = s2 | (k[2] & (Y2(3, 0) != 0));
  const ap_uint<17> Y4 = k[3] ? ap_uint<17>(Y3 >> 8) : Y3;
  const bool s4 = s3 | (k[3] & (Y3(7, 0) != 0));
  const ap_uint<17> Y = k[4] ? ap_uint<17>(Y4 >> 16) : Y4;
  const bool st = s4 | (k[4] & (Y4(15, 0) != 0));
  const bool rb = Y[0];
  const ap_uint<16> ip = Y >> 1;  // floor(M / 2^sh) <= 2^15
#endif
  const ap_uint<17> v = ap_uint<17>(ip) + ap_uint<17>(rb & (st | ip[0]));
  ap_uint<16> mag = v > QMAX ? ap_uint<17>(QMAX) : v;
  if (sh < 9) mag = QMAX;   // value >= 2^15
  if (sh > 25) mag = 0;     // value < 0.5
  return neg ? q_t(-ap_int<17>(mag)) : q_t(mag);
}

// q = clamp(rne(fl32(x / s)), +-32767), integer datapath (QZ_IMPL_RECIP, the default).
static inline q_t quant_elem_recip(fbits_t xb, const RowScale& rs) {
#pragma HLS INLINE
  ap_uint<8> E = xb(30, 23);
  bool neg = xb[31];
  ap_uint<24> X = (ap_uint<1>(1), ap_uint<23>(xb(22, 0)));
  ap_uint<50> P = X * rs.R;                // 24 x 26: two DSPs
  ap_uint<26> Qc = P >> 24;                // floor(X 2^25 / S) - {0, 1}
  ap_uint<25> Qlo = Qc(24, 0);
  ap_uint<17> Slo = rs.S(16, 0);
  ap_uint<7> Shi = rs.S(23, 17);
  ap_uint<42> p1 = Qlo * Slo;              // 25 x 17: one DSP
  ap_uint<8> qlo = Qc(7, 0);
  ap_uint<15> p2 = qlo * Shi;              // 8 x 7: fabric (only its low 8 bits matter)
#pragma HLS BIND_OP variable = p2 op = mul impl = fabric
  ap_uint<25> lo = p1(24, 0) + (ap_uint<25>(p2(7, 0)) << 17);  // (Qc S) mod 2^25
  // rem = X 2^25 - Qc S = (2^25 - lo) mod 2^25 is in [0, 2S). rem >= S <=> lo != 0 && lo <= T,
  // and then rem - S = T - lo.
  bool lz = lo == 0;
  bool ge = !lz && lo <= rs.T;
  bool st0 = ge ? lo != rs.T : !lz;        // the exact quotient has more bits
  ap_uint<26> Q = Qc + ap_uint<26>(ge);    // floor(X 2^25 / S), 2^24 < Q < 2^26
  // Normalise to Z in [2^25, 2^26): X / S in [1, 2) has Q[25] = 1, X / S in (0.5, 1) does not.
  bool big = Q[25];
  ap_uint<26> Z = big ? Q : ap_uint<26>(Q << 1);
  // fl32(X / S) = M 2^-23 (M <= 2^24 after rounding); fl32(x / s) = M 2^-sh.
  ap_uint<25> M = Z >> 2;
  bool rb = Z[1], st = Z[0] | st0;
  if (rb & (st | M[0])) M += 1;
  ap_int<10> sh = ap_int<10>(rs.B) - ap_int<10>(E) + (big ? 0 : 1);
  q_t q = rne_sat(M, sh, neg);
  return E == 0 ? q_t(0) : q;              // zero or denormal x: |x / s| < 4e-4
}

// The same with the FP32 divider core (QZ_IMPL_FDIV): f = x / s, then rne on f's bits.
static inline q_t quant_elem_fdiv(fbits_t xb, const RowScale& rs) {
#pragma HLS INLINE
  float f = u2f(xb) / u2f(rs.s_bits);
  fbits_t fb = f2u(f);
  ap_uint<8> E = fb(30, 23);
  if (E == 0) return 0;
  ap_uint<25> M = (ap_uint<1>(1), ap_uint<23>(fb(22, 0)));
  return rne_sat(M, ap_int<10>(150) - ap_int<10>(E), fb[31]);
}

static inline q_t quant_elem(fbits_t xb, const RowScale& rs) {
#pragma HLS INLINE
#ifndef QZ_IMPL_FDIV
  return quant_elem_recip(xb, rs);
#else
  return quant_elem_fdiv(xb, rs);
#endif
}

// ---------------------------------------------------------------- inverse helpers
// Dequantize: fl32(q * s), what torch's q.float() * s gives (an int16 is exact in FP32).
static inline float dequant(q_t q, ap_uint<32> s_bits) {
#pragma HLS INLINE
  return float(int(q)) * u2f(s_bits);
}
// The int16 engine's output y = acc * s_row * s_tok is the engine's formula (int16_linear.h); this
// header only fixes the activation side (q, s_tok).

// ---------------------------------------------------------------- 128-bit words
// io word = 4 FP32 (value j in bits [32j+31:32j]) or 8 int16 (value j in bits [16j+15:16j]),
// the little-endian lane order of fpga/hls/float_linear.h.
typedef ap_uint<128> word_t;
static inline fbits_t word_f32(const word_t& w, int j) {
#pragma HLS INLINE
  return w(32 * j + 31, 32 * j);
}
static inline void word_set_q(word_t& w, int j, q_t q) {
#pragma HLS INLINE
  w(16 * j + 15, 16 * j) = ap_uint<16>(q);
}
static inline q_t word_q(const word_t& w, int j) {
#pragma HLS INLINE
  return q_t(ap_uint<16>(w(16 * j + 15, 16 * j)));
}
// absmax of the 4 FP32 of a word (as magnitude bits)
static inline abits_t word_amax(const word_t& w) {
#pragma HLS INLINE
  return amax2(amax2(abs_bits(word_f32(w, 0)), abs_bits(word_f32(w, 1))),
               amax2(abs_bits(word_f32(w, 2)), abs_bits(word_f32(w, 3))));
}
// 8 FP32 (two words, lo = values 0..3) -> one word of 8 int16 with one row's scale
static inline word_t quant8(const word_t& lo, const word_t& hi, const RowScale& rs) {
#pragma HLS INLINE
  word_t o = 0;
  for (int l = 0; l < 4; l++) {
#pragma HLS UNROLL
    word_set_q(o, l, quant_elem(word_f32(lo, l), rs));
    word_set_q(o, l + 4, quant_elem(word_f32(hi, l), rs));
  }
  return o;
}

}  // namespace qz
