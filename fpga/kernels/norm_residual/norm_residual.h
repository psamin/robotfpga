// norm_residual: fused residual add + LayerNorm / RMSNorm + per-row int16 quantization, the future NORM
// mode (sequencer OP_NORM 0x30) of float_top. Header-only core: the same code builds in Vitis HLS and in
// g++ (-I<Vitis_HLS>/include -ffp-contract=off). nr_emul.py is its bit-exact torch model.
//
// Per row y = A (+ B) (FP32, IEEE add), K values (K % 8 == 0, K <= KMAX):
//   stats    block floating point, exact integers: Emax = max exponent field; Y_i = round_half_away(
//            m_i * 2^(1 + e_i - Emax)) (|Y_i| < 2^25, so y_i ~ Y_i 2^(Emax-151)); S = sum Y, Q = sum Y^2;
//            num = K Q - S^2 (LayerNorm) or K Q (RMSNorm), exact (< 2^71). No cancellation, no order effects.
//   scalars  invK = recip_nr(K), invK2 = recip_nr(K^2) once per call;
//            var = ldexp(i2f(num) * invK2, 2 (Emax-151)), mean = ldexp(i2f(S) * invK, Emax-151) (LN; else 0),
//            rstd = rsqrt_nr(var + eps)
//   z        ((y - mean) * rstd) * g + b                          (FP32 ops in this order; RMSNorm b = 0)
//   quant    A = max |z|; inv = trunc17(fl(recip_nr(A) * 32767)); s = recip_nr(inv)
//            q = clamp(rne(z * inv), +-32767) (z * inv exact: 24 x 17 bits); the next linear sees q * s.
//            A < 2^-100 (all-zero row): inv = 0, s = 0, q = 0.
// recip_nr / rsqrt_nr: 128-entry seed ROM + 2 Newton-Raphson steps (FP32 mul / sub, no FMA).
//
// Outputs (flags as sequencer seq_format.h): D = y FP32 rows (F_SUM), C = q int16 rows (F_QOUT) with the scales
// in F, 4 per 128-bit word (last word of each batch item zero-padded), or C = z FP32 rows (no F_QOUT).
//
// Dataflow processes (4 lanes = one 128-bit FP32 word per clock each):
//   rdA / rdB : A rows over hp0, B rows over hp1; one burst per batch item when a row's pitch leaves a gap of
//               at most half a row (gap words read and dropped), else one burst per row        -> fa, fb
//   add   : 4 FP32 adds, running max exponent field per row                                    -> f1, fe
//   stats : Y_i, S, Q (4 x 25x25 squares)                                                      -> f2, fs
//   scal  : i2f, ldexp, rsqrt_nr per row (~126 clocks, one shared fmul / fadd / fsub)          -> fm
//   norm  : gamma / beta from hp2 into BRAM once per call; z, running max|z|                  -> fy, fz, fa
//   qscal : recip_nr, inv, s per row (~103 clocks)                                            -> fq
//   quant : q = rne(z * inv), 8 int16 per word (or z FP32 words)                              -> fo, fsc
//   wr    : the only io writer, chunks of <= 256 rows: one flat burst of records [D row | C row] (WM_REC)
//           or of C rows (WM_C), else two bursts per row (WM_ROW); then the chunk's scales in one burst.
// Every m_axi burst ends with a write-response wait (HLS schedules writeresp at the loop exit), so the record
// layout matters on the board: see README.md. Steady state: max(K/4, rdA words, out words) clocks per row.
#pragma once
#include <ap_int.h>
#include <hls_stream.h>
#include <stdint.h>

#include "nr_tables.h"

#ifndef NR_FADD_IMPL
#define NR_FADD_IMPL fabric  // FP32 add / sub cores in LUTs (fulldsp: 2 DSP each)
#endif
#ifndef NR_FMUL_IMPL
#define NR_FMUL_IMPL meddsp  // FP32 multiplier core (maxdsp / fulldsp / meddsp / fabric)
#endif
#ifndef NR_KMAX
#define NR_KMAX 1024
#endif

namespace nr {

constexpr int L = 4;                 // lanes: FP32 values per io word
constexpr int KMAX = NR_KMAX;        // longest row
constexpr int KW = KMAX / L;         // io words of an FP32 row
constexpr int FD = 2 * KW - 12;      // row FIFOs: two rows of K <= 976 (f2 and fz hide the per-row scalar latency)
constexpr int FO_DEPTH = KW / 2;     // quant -> wr: one int16 row (K / 8 words), LUTRAM
constexpr int QMAX = 32767;
constexpr uint32_t INV_MASK = 0xFFFFFF80u;
constexpr int TINY_E = 27;           // |z| max below 2^-100 (exponent field < 27): zero row

// flags (sequencer seq_format.h F_*)
constexpr uint32_t F_QOUT = 0x0002;
constexpr uint32_t F_RESID = 0x0004;
constexpr uint32_t F_SUM = 0x0008;
constexpr uint32_t F_LN = 0x0010;

typedef ap_uint<128> word_t;

// One call. Offsets, pitches and strides are in 128-bit words relative to the port's base address
// (sequencer mode: base 0, so offset = byte address >> 4).
struct Cfg {
  uint32_t flags;
  int K;          // row length (values)
  int N;          // rows per batch item
  int B;          // batch items
  uint32_t eps;   // FP32 bits
  uint32_t a_off, a_pitch, a_stride;  // A rows FP32 (hp0)
  uint32_t b_off, b_pitch, b_stride;  // B rows FP32 (hp1), F_RESID
  uint32_t e_off;                     // gamma K FP32, then beta K FP32 (F_LN) (hp2)
  uint32_t c_off, c_pitch, c_stride;  // C rows: int16 (F_QOUT) or FP32 z (io)
  uint32_t d_off, d_pitch, d_stride;  // D rows FP32 y = A + B (F_SUM) (io)
  uint32_t f_off, f_stride;           // F scales FP32, 4 per word (F_QOUT) (io)
  // derived once per call (set_derived): keeps the multipliers out of the processes
  int rows;     // B * N
  int T;        // rows * K / 4: io words of FP32 per call
  bool a_flat;  // A rows read as one burst per batch item (gap words read and dropped)
  bool b_flat;
  int a_len;    // words of that burst: (N - 1) * a_pitch + K / 4
  int b_len;
  int wmode;    // WM_*: how the outputs are written
  int rec;      // WM_REC / WM_C: words per output row in the flat burst
};

// Output write modes (one burst per chunk of up to SCH rows instead of two per row where the layout allows):
constexpr int WM_ROW = 0;  // per row: D row burst (F_SUM), then C row burst
constexpr int WM_REC = 1;  // F_SUM, records: C row r directly follows D row r (c_off = d_off + K/4,
                           // c_pitch = d_pitch = K/4 + K/8 [F_QOUT] or K/2): D and C in one flat burst
constexpr int WM_C = 2;    // no F_SUM, contiguous C rows (c_pitch = row words): one flat burst
constexpr int SCH = 256;   // rows per write chunk; their scales are buffered and written in one burst after it

// A row source can be read flat when its pitch leaves a gap of at most half a row (contiguous rows, or the
// records above: a residual stream written by WM_REC).
static inline bool flat_ok(uint32_t pitch, int KW_) {
#pragma HLS INLINE
  return pitch >= (uint32_t)KW_ && 2 * pitch <= 3 * (uint32_t)KW_;
}

static inline void set_derived(Cfg& c) {
#pragma HLS INLINE
  const int KW_ = c.K / L;
  const bool qout = c.flags & F_QOUT, sum = c.flags & F_SUM, resid = c.flags & F_RESID;
  const int CW_ = qout ? c.K / 8 : KW_;
  c.rows = (int)(ap_uint<16>(c.B) * ap_uint<16>(c.N));
  c.T = (int)(ap_uint<20>(c.rows) * ap_uint<9>(KW_));
  c.a_flat = flat_ok(c.a_pitch, KW_);
  c.b_flat = resid && flat_ok(c.b_pitch, KW_);
  c.a_len = c.a_flat ? (int)(ap_uint<16>(c.N - 1) * ap_uint<9>(c.a_pitch)) + KW_ : 0;
  c.b_len = c.b_flat ? (int)(ap_uint<16>(c.N - 1) * ap_uint<9>(c.b_pitch)) + KW_ : 0;
  if (sum && c.c_off == c.d_off + (uint32_t)KW_ && c.c_pitch == c.d_pitch && c.d_pitch == (uint32_t)(KW_ + CW_)) {
    c.wmode = WM_REC;
    c.rec = KW_ + CW_;
  } else if (!sum && c.c_pitch == (uint32_t)CW_) {
    c.wmode = WM_C;
    c.rec = CW_;
  } else {
    c.wmode = WM_ROW;
    c.rec = 0;
  }
}

struct Stat {
  ap_int<48> S;
  ap_uint<64> Q;
  ap_uint<8> emax;
};
struct MR {
  float mean, rstd;
};
struct IS {
  uint32_t inv, s;
};

// ------------------------------------------------------------------------------------------ scalar helpers
static inline uint32_t f2u(float f) {
#pragma HLS INLINE
  union {
    float f;
    uint32_t u;
  } c;
  c.f = f;
  return c.u;
}
static inline float u2f(uint32_t u) {
#pragma HLS INLINE
  union {
    float f;
    uint32_t u;
  } c;
  c.u = u;
  return c.f;
}
static inline float lane_f(const word_t& w, int l) {
#pragma HLS INLINE
  return u2f((uint32_t)w.range(32 * l + 31, 32 * l));
}
static inline void set_lane_f(word_t& w, int l, float f) {
#pragma HLS INLINE
  w.range(32 * l + 31, 32 * l) = f2u(f);
}

static inline float fadd(float a, float b) {
#pragma HLS INLINE
  float r = a + b;
#pragma HLS BIND_OP variable = r op = fadd impl = NR_FADD_IMPL
  return r;
}
static inline float fsub(float a, float b) {
#pragma HLS INLINE
  float r = a - b;
#pragma HLS BIND_OP variable = r op = fsub impl = NR_FADD_IMPL
  return r;
}
static inline float fmul(float a, float b) {
#pragma HLS INLINE
  float r = a * b;
#pragma HLS BIND_OP variable = r op = fmul impl = NR_FMUL_IMPL
  return r;
}

// 1 / a for a in [2^-125, 2^125] (a == +-0 -> 0)
// recip_nr / rsqrt_nr are inlined: the calling process (one row at a time, PIPELINE off) shares one fmul and
// one fadd/fsub core among all its scalar ops (ALLOCATION there).
static float recip_nr(float a) {
#pragma HLS INLINE
  static const uint32_t SEED[128] = NR_RECIP_SEED_INIT;
  const uint32_t b = f2u(a);
  if ((b & 0x7FFFFFFFu) == 0) return 0.0f;
  const int E = (int)((b >> 23) & 0xFF) - 127;
  const uint32_t idx = (b >> 16) & 0x7F;
  float y = u2f((uint32_t)((int32_t)SEED[idx] - E * (1 << 23)));
  for (int it = 0; it < 2; it++) {
    float t = fmul(a, y);
    t = fsub(2.0f, t);
    y = fmul(y, t);
  }
  return y;
}

// 1 / sqrt(v) for v in [2^-124, 2^124]
static float rsqrt_nr(float v) {
#pragma HLS INLINE
  static const uint32_t SEED[128] = NR_RSQRT_SEED_INIT;
  const uint32_t b = f2u(v);
  const int E = (int)((b >> 23) & 0xFF) - 127;
  const int p = E & 1;
  const int Eh = (E - p) / 2;  // exact: E - p is even
  const uint32_t idx = ((uint32_t)p << 6) | ((b >> 17) & 0x3F);
  float y = u2f((uint32_t)((int32_t)SEED[idx] - Eh * (1 << 23)));
  const float h = fmul(v, 0.5f);
  for (int it = 0; it < 2; it++) {
    float t = fmul(y, y);
    t = fmul(h, t);
    t = fsub(1.5f, t);
    y = fmul(y, t);
  }
  return y;
}

// exact unsigned integer -> FP32, round to nearest even, sign applied
template <int W>
static float i2f_rne(ap_uint<W> a, bool neg) {
#pragma HLS INLINE
  if (a == 0) return 0.0f;
  const int bl = W - (int)a.countLeadingZeros();  // bit length, >= 1
  ap_uint<25> m;
  int ef = 126 + bl;  // exponent field of a value in [2^(bl-1), 2^bl)
  if (bl <= 24) {
    m = ap_uint<25>(a);  // a < 2^24 here; widen before the shift (ap_uint shifts keep their width)
    m = m << (24 - bl);
  } else {
    const int sh = bl - 24;
    ap_uint<W> mm = a >> sh;
    ap_uint<W> rem = a & ((ap_uint<W>(1) << sh) - 1);
    ap_uint<W> half = ap_uint<W>(1) << (sh - 1);
    m = ap_uint<25>(mm);
    if (rem > half || (rem == half && m[0])) m = m + 1;
    if (m[24]) {  // rounded up to 2^24
      m = m >> 1;
      ef++;
    }
  }
  const uint32_t bits = ((uint32_t)neg << 31) | ((uint32_t)ef << 23) | ((uint32_t)m & 0x7FFFFFu);
  return u2f(bits);
}

// The same conversion, small and sequential (<= W/8 + 7 clocks): for the per-row scalars, where a 72-bit
// combinational normalizer would cost thousands of LUTs. Bit-identical to i2f_rne.
template <int W>
static float i2f_seq(ap_uint<W> a, bool neg) {
#pragma HLS INLINE
  if (a == 0) return 0.0f;
  ap_uint<W> x = a;
  int ef = 126 + W;  // exponent field once x[W-1] is the leading one
coarse:
  for (int i = 0; i < W / 8; i++) {
#pragma HLS PIPELINE off
    if (x(W - 1, W - 8) != 0) break;
    x = x << 8;
    ef -= 8;
  }
fine:
  for (int i = 0; i < 7; i++) {
#pragma HLS PIPELINE off
    if (x[W - 1]) break;
    x = x << 1;
    ef -= 1;
  }
  ap_uint<25> m = ap_uint<25>(x(W - 1, W - 24));
  const bool half = x[W - 25];
  const bool sticky = x(W - 26, 0) != 0;
  if (half && (sticky || m[0])) m = m + 1;
  if (m[24]) {
    m = m >> 1;
    ef++;
  }
  const uint32_t bits = ((uint32_t)neg << 31) | ((uint32_t)ef << 23) | ((uint32_t)m & 0x7FFFFFu);
  return u2f(bits);
}

// One shared instance for every row / call scalar conversion (K, K^2, num, S).
static float i2f72(ap_uint<72> a, bool neg) {
#pragma HLS INLINE off
  return i2f_seq<72>(a, neg);
}

// f * 2^k exactly; results below 2^-126 -> +0 (flush), overflow -> inf
static inline float ldexp_ftz(float f, int k) {
#pragma HLS INLINE
  const uint32_t b = f2u(f);
  const int e = (int)((b >> 23) & 0xFF);
  if (e == 0) return 0.0f;
  const int e2 = e + k;
  if (e2 <= 0) return 0.0f;
  if (e2 >= 255) return u2f((b & 0x80000000u) | 0x7F800000u);
  return u2f((b & 0x807FFFFFu) | ((uint32_t)e2 << 23));
}

// |Y| of one value against the row's max exponent field (stats), 25 bits
static inline ap_uint<25> yfix_mag(uint32_t b, ap_uint<8> emax) {
#pragma HLS INLINE
  const ap_uint<8> e = (b >> 23) & 0xFF;
  const ap_uint<24> m = e == 0 ? ap_uint<24>(0) : ap_uint<24>(0x800000u | (b & 0x7FFFFFu));
  const int sh = (int)emax - (int)e - 1;  // -1 .. 253
  ap_uint<25> mag;
  if (sh < 0) {
    mag = ap_uint<25>(m) << 1;
  } else if (sh > 24) {
    mag = 0;
  } else {
    const ap_uint<26> rnd = (ap_uint<26>(1) << sh) >> 1;  // 2^(sh-1), 0 for sh = 0
    mag = ap_uint<25>((ap_uint<26>(m) + rnd) >> sh);
  }
  return mag;
}

// q = clamp(rne(z * inv), +-32767); inv has a 17-bit significand
static inline ap_int<16> quant1(uint32_t zb, uint32_t ib) {
#pragma HLS INLINE
  const int ez = (zb >> 23) & 0xFF;
  const int ei = (ib >> 23) & 0xFF;
  const ap_uint<24> mz = 0x800000u | (zb & 0x7FFFFFu);
  const ap_uint<17> mi = 0x10000u | ((ib >> 7) & 0xFFFFu);
  const ap_uint<41> P = mz * mi;  // |z * inv| = P * 2^-sh, P in [2^39, 2^41)
  const int sh = 293 - ez - ei;
  ap_uint<17> mag;
  if (ez == 0 || ei == 0 || sh > 41) {
    mag = 0;  // zero, or |value| < 0.5
  } else if (sh < 24) {
    mag = QMAX;  // |value| >= 2^15: clamp
  } else {
    const int t = sh - 24;                            // 0 .. 17
    const ap_uint<18> qx = ap_uint<18>(P >> (23 + t));  // value * 2, truncated: the half bit is qx[0]
    // sticky = OR of P[22 + t : 0] = OR(P[22:0]) | OR(P[22 + t : 23]) via a prefix-OR chain over P[39:23]
    ap_uint<18> pre;
    pre[0] = 0;
    for (int i = 1; i < 18; i++) {
#pragma HLS UNROLL
      pre[i] = pre[i - 1] | P[22 + i];
    }
    const bool sticky = (P(22, 0) != 0) || pre[t];
    const ap_uint<17> qt = qx >> 1;
    const bool up = qx[0] && (sticky || qt[0]);
    const ap_uint<18> r = ap_uint<18>(qt) + (up ? 1 : 0);
    mag = r > QMAX ? ap_uint<17>(QMAX) : ap_uint<17>(r);
  }
  const ap_int<17> sm = (zb >> 31) ? ap_int<17>(-ap_int<17>(mag)) : ap_int<17>(mag);
  return ap_int<16>(sm);
}

// ------------------------------------------------------------------------------------------ processes
// One read burst of len words from src + off; keeps the words whose position in a pitch-word row is < KW_.
// Its own function (INLINE off) so HLS bursts this loop only, with no multipliers for a nest-wide burst.
static void rd_seg(const word_t* src, uint32_t off, int len, uint32_t pitch, int KW_, hls::stream<word_t>& fo) {
#pragma HLS INLINE off
  uint32_t jr = 0;
rd:
  for (int j = 0; j < len; j++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 180 max = 61440 avg = 12000
    const word_t w = src[off + j];
    if (jr < (uint32_t)KW_) fo.write(w);
    jr = jr == pitch - 1 ? 0u : jr + 1;
  }
}

// One row source (A over hp0, or B over hp1) -> FIFO of its K / 4 words per row, all rows of the call.
// Flat: one burst per batch item over (N - 1) * pitch + K / 4 words, the gap words dropped. Else one burst per row.
static void p_rdx(const word_t* src, uint32_t off, uint32_t pitch, uint32_t stride, bool en, bool flat, int len,
                  Cfg c, hls::stream<word_t>& fo) {
  if (!en) return;
  const int KW_ = c.K / L;
  uint32_t ib = off;
items:
  for (int b = 0; b < c.B; b++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 2 avg = 1
    if (flat) {
      rd_seg(src, ib, len, pitch, KW_, fo);
    } else {
      uint32_t rb = ib;
    rows:
      for (int n = 0; n < c.N; n++) {
#pragma HLS LOOP_TRIPCOUNT min = 50 max = 512 avg = 100
        rd_seg(src, rb, KW_, (uint32_t)KW_, KW_, fo);
        rb += pitch;
      }
    }
    ib += stride;
  }
}

// y = A (+ B), 4 FP32 adds per word, and each row's max exponent field.
static void p_add(Cfg c, hls::stream<word_t>& fa_, hls::stream<word_t>& fb_, hls::stream<word_t>& f1,
                  hls::stream<ap_uint<8> >& fe) {
  const int KW_ = c.K / L;
  const bool resid = c.flags & F_RESID;
  ap_uint<8> emax = 0;
  int jr = 0;
add:
  for (int j = 0; j < c.T; j++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 9000 max = 61440 avg = 20000
    const word_t wa = fa_.read();
    word_t wy = wa;
    if (resid) {
      const word_t wb = fb_.read();
      for (int l = 0; l < L; l++) {
#pragma HLS UNROLL
        set_lane_f(wy, l, fadd(lane_f(wa, l), lane_f(wb, l)));
      }
    }
    ap_uint<8> m = 0;
    for (int l = 0; l < L; l++) {
#pragma HLS UNROLL
      const ap_uint<8> e = wy.range(32 * l + 30, 32 * l + 23);
      m = e > m ? e : m;
    }
    emax = (jr == 0 || m > emax) ? m : emax;
    f1.write(wy);
    if (jr == KW_ - 1) {
      fe.write(emax);
      jr = 0;
    } else {
      jr++;
    }
  }
}

// All processes below run one flat pipelined loop over every word of the call (no drain between rows).
static void p_stats(Cfg c, hls::stream<word_t>& f1, hls::stream<ap_uint<8> >& fe, hls::stream<word_t>& f2,
                    hls::stream<Stat>& fs) {
  const int KW_ = c.K / L;
  const int T = c.T;
  ap_uint<8> emax = 0;
  ap_int<48> S = 0;
  ap_uint<64> Q = 0;
  int jr = 0;
acc:
  for (int j = 0; j < T; j++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 9000 max = 61440 avg = 20000
    if (jr == 0) emax = fe.read();
    const word_t w = f1.read();
    ap_int<28> s4 = 0;
    ap_uint<52> q4 = 0;
    for (int l = 0; l < L; l++) {
#pragma HLS UNROLL
      const uint32_t bits = (uint32_t)w.range(32 * l + 31, 32 * l);
      const ap_uint<25> mag = yfix_mag(bits, emax);
      s4 += (bits >> 31) ? ap_int<28>(-ap_int<27>(mag)) : ap_int<28>(mag);
      q4 += ap_uint<50>(mag * mag);
    }
    const ap_int<48> S1 = (jr == 0 ? ap_int<48>(0) : S) + s4;
    const ap_uint<64> Q1 = (jr == 0 ? ap_uint<64>(0) : Q) + q4;
    S = S1;
    Q = Q1;
    f2.write(w);
    if (jr == KW_ - 1) {
      Stat st;
      st.S = S1;
      st.Q = Q1;
      st.emax = emax;
      fs.write(st);
      jr = 0;
    } else {
      jr++;
    }
  }
}

static void p_scal(Cfg c, hls::stream<Stat>& fs, hls::stream<MR>& fm) {
  const bool ln = c.flags & F_LN;
#pragma HLS ALLOCATION operation instances = fmul limit = 1
#pragma HLS ALLOCATION operation instances = fsub limit = 1
  const ap_uint<11> Ku = c.K;
#pragma HLS ALLOCATION function instances = i2f72 limit = 1
  const float invK = recip_nr(i2f72(ap_uint<72>(Ku), false));
  const float invK2 = recip_nr(i2f72(ap_uint<72>(ap_uint<22>(Ku * Ku)), false));
  const float neg_eps = u2f(c.eps ^ 0x80000000u);  // var + eps computed as var - (-eps): identical in IEEE,
                                                    // and the process needs only its one fsub core
rows:
  for (int r = 0; r < c.rows; r++) {
#pragma HLS LOOP_TRIPCOUNT min = 50 max = 512 avg = 100
#pragma HLS PIPELINE off
    const Stat st = fs.read();
    const bool sneg = st.S < 0;
    const ap_uint<48> Sa = sneg ? ap_uint<48>(-st.S) : ap_uint<48>(st.S);
    const ap_uint<72> KQ = ap_uint<72>(st.Q) * Ku;
    const ap_uint<72> S2 = ln ? ap_uint<72>(ap_uint<36>(Sa) * ap_uint<36>(Sa)) : ap_uint<72>(0);
    const ap_uint<72> num = KQ - S2;
    const float numf = i2f72(num, false);
    const float Sf = i2f72(ap_uint<72>(Sa), sneg);
    const int k1 = (int)st.emax - 151;
    const float var = ldexp_ftz(fmul(numf, invK2), 2 * k1);
    MR o;
    o.mean = ln ? ldexp_ftz(fmul(Sf, invK), k1) : 0.0f;
    o.rstd = rsqrt_nr(fsub(var, neg_eps));
    fm.write(o);
  }
}

static void p_norm(const word_t* e, Cfg c, hls::stream<word_t>& f2, hls::stream<MR>& fm, hls::stream<word_t>& fy,
                   hls::stream<word_t>& fz, hls::stream<uint32_t>& fa) {
  static word_t gtab[KW], btab[KW];
  const int KW_ = c.K / L;
  const bool ln = c.flags & F_LN;
  const bool sum = c.flags & F_SUM;
load_g:
  for (int j = 0; j < KW_; j++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 180 max = 240 avg = 200
    gtab[j] = e[c.e_off + j];
  }
  if (ln) {
  load_b:
    for (int j = 0; j < KW_; j++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 180 max = 240 avg = 200
      btab[j] = e[c.e_off + KW_ + j];
    }
  } else {
  zero_b:
    for (int j = 0; j < KW_; j++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 180 max = 240 avg = 200
      btab[j] = 0;
    }
  }
  const int T = c.T;
  MR mr;
  mr.mean = 0.0f;
  mr.rstd = 0.0f;
  ap_uint<31> amax = 0;
  int jr = 0;
nz:
  for (int j = 0; j < T; j++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 9000 max = 61440 avg = 20000
    if (jr == 0) mr = fm.read();
    const word_t w = f2.read();
    const word_t g = gtab[jr], bb = btab[jr];
    word_t wz;
    ap_uint<31> m = 0;
    for (int l = 0; l < L; l++) {
#pragma HLS UNROLL
      float t = fsub(lane_f(w, l), mr.mean);
      t = fmul(t, mr.rstd);
      t = fmul(t, lane_f(g, l));
      const float z = fadd(t, lane_f(bb, l));
      set_lane_f(wz, l, z);
      const ap_uint<31> az = f2u(z) & 0x7FFFFFFFu;
      m = az > m ? az : m;
    }
    const ap_uint<31> a1 = (jr == 0 || m > amax) ? m : amax;
    amax = a1;
    if (sum) fy.write(w);
    fz.write(wz);
    if (jr == KW_ - 1) {
      fa.write((uint32_t)a1);
      jr = 0;
    } else {
      jr++;
    }
  }
}

static void p_qscal(Cfg c, hls::stream<uint32_t>& fa, hls::stream<IS>& fq) {
#pragma HLS ALLOCATION operation instances = fmul limit = 1
#pragma HLS ALLOCATION operation instances = fsub limit = 1
rows:
  for (int r = 0; r < c.rows; r++) {
#pragma HLS LOOP_TRIPCOUNT min = 50 max = 512 avg = 100
#pragma HLS PIPELINE off
    const uint32_t ab = fa.read();
    IS o;
    if ((int)(ab >> 23) < TINY_E) {
      o.inv = 0;
      o.s = 0;
    } else {
      const float rA = recip_nr(u2f(ab));
      const uint32_t inv = f2u(fmul(rA, (float)QMAX)) & INV_MASK;
      o.inv = inv;
      o.s = f2u(recip_nr(u2f(inv)));
    }
    fq.write(o);
  }
}

static void p_quant(Cfg c, hls::stream<word_t>& fz, hls::stream<IS>& fq, hls::stream<word_t>& fo,
                    hls::stream<uint32_t>& fsc) {
  const int KW_ = c.K / L;
  const int T = c.T;
  const bool qout = c.flags & F_QOUT;
  IS is;
  is.inv = 0;
  is.s = 0;
  ap_uint<64> lo = 0;
  int jr = 0;
q:
  for (int j = 0; j < T; j++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 9000 max = 61440 avg = 20000
    if (jr == 0) is = fq.read();
    const word_t w = fz.read();
    ap_uint<64> h;
    for (int l = 0; l < L; l++) {
#pragma HLS UNROLL
      h.range(16 * l + 15, 16 * l) = quant1((uint32_t)w.range(32 * l + 31, 32 * l), is.inv);
    }
    if (!qout) {
      fo.write(w);
    } else if (jr & 1) {
      word_t o;
      o.range(63, 0) = lo;
      o.range(127, 64) = h;
      fo.write(o);
    } else {
      lo = h;
    }
    if (jr == KW_ - 1) {
      if (qout) fsc.write(is.s);
      jr = 0;
    } else {
      jr++;
    }
  }
}

// Write bursts, each in its own function (INLINE off) so HLS bursts only that loop.
// Flat: len words of rows of rec words; the first y_words of a row come from fy, the rest from fo; at each
// row end the row's scale goes from fsc into sbuf[n] (F_QOUT).
static void wr_flat_seg(word_t* io, uint32_t off, int len, int rec, int y_words, bool qout, hls::stream<word_t>& fy,
                        hls::stream<word_t>& fo, hls::stream<uint32_t>& fsc, uint32_t sbuf[SCH]) {
#pragma HLS INLINE off
  int jr = 0, n = 0;
wr:
  for (int j = 0; j < len; j++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 4500 max = 61440 avg = 20000
    io[off + j] = jr < y_words ? fy.read() : fo.read();
    if (jr == rec - 1) {
      if (qout) sbuf[n] = fsc.read();
      n++;
      jr = 0;
    } else {
      jr++;
    }
  }
}

static void wr_seg(word_t* io, uint32_t off, int len, hls::stream<word_t>& src) {
#pragma HLS INLINE off
wr:
  for (int j = 0; j < len; j++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 90 max = 240 avg = 180
    io[off + j] = src.read();
  }
}

static void wr_scales(word_t* io, uint32_t off, int nr, const uint32_t sbuf[SCH]) {
#pragma HLS INLINE off
  const int nw = (nr + 3) >> 2;
wf:
  for (int w = 0; w < nw; w++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 13 max = 64 avg = 20
    word_t sw;
    for (int l = 0; l < 4; l++) {
#pragma HLS UNROLL
      sw.range(32 * l + 31, 32 * l) = 4 * w + l < nr ? sbuf[4 * w + l] : 0u;
    }
    io[off + w] = sw;
  }
}

// The only io writer. Rows go in chunks of up to SCH rows: the chunk's outputs (one flat burst in WM_REC / WM_C,
// two bursts per row in WM_ROW), then the chunk's scales (F_QOUT) from an on-chip buffer in one burst, so the
// write-response wait that ends every burst is paid twice per chunk instead of two or three times per row.
static void p_wr(word_t* io, Cfg c, hls::stream<word_t>& fy, hls::stream<word_t>& fo, hls::stream<uint32_t>& fsc) {
  const int KW_ = c.K / L;
  const bool qout = c.flags & F_QOUT;
  const bool sum = c.flags & F_SUM;
  const int CW_ = qout ? c.K / 8 : KW_;
  uint32_t sbuf[SCH];
#pragma HLS ARRAY_PARTITION variable = sbuf cyclic factor = 4
#pragma HLS BIND_STORAGE variable = sbuf type = ram_2p impl = lutram
  uint32_t ic = c.c_off, id = c.d_off, iF = c.f_off;
items:
  for (int b = 0; b < c.B; b++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 2 avg = 1
    if (c.wmode != WM_ROW) {
      uint32_t base = c.wmode == WM_REC ? id : ic;  // a record starts with its D row
      uint32_t rf = iF;
      int n0 = 0;
    chunks:
      while (n0 < c.N) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 2 avg = 1
        const int nr = c.N - n0 < SCH ? c.N - n0 : SCH;
        const int len = (int)(ap_uint<9>(nr) * ap_uint<9>(c.rec));  // <= 256 x 384: a small LUT multiplier
        wr_flat_seg(io, base, len, c.rec, c.wmode == WM_REC ? KW_ : 0, qout, fy, fo, fsc, sbuf);
        base += len;
        if (qout) {
          wr_scales(io, rf, nr, sbuf);
          rf += SCH / 4;
        }
        n0 += nr;
      }
    } else {
      // (row addresses are not live after this loop, so HLS does not build base + n * pitch multipliers)
      uint32_t rc = ic, rd = id, rf = iF;
      int n = 0;
    rows:
      for (int r = 0; r < c.N; r++) {
#pragma HLS LOOP_TRIPCOUNT min = 50 max = 512 avg = 100
        if (sum) wr_seg(io, rd, KW_, fy);
        wr_seg(io, rc, CW_, fo);
        if (qout) sbuf[n] = fsc.read();
        n++;
        if (n == SCH || r == c.N - 1) {
          if (qout) wr_scales(io, rf, n, sbuf);
          rf += SCH / 4;
          n = 0;
        }
        rc += c.c_pitch;
        rd += c.d_pitch;
      }
    }
    ic += c.c_stride;
    id += c.d_stride;
    iF += c.f_stride;
  }
}

// 0 = ok, 1 = bad dimensions (nothing read or written)
static inline int check_cfg(const Cfg& c) {
#pragma HLS INLINE
  return (c.K < 8 || c.K > KMAX || (c.K & 7) || c.N < 1 || c.B < 1) ? 1 : 0;
}

// The mode body: call from float_top (or the sequencer) with the four ports. a, b, e may alias (sequencer
// mode: all bases 0) but must be distinct m_axi bundles for full speed (hp0, hp1, hp2); io writes only.
// c: set_derived(c) must have been called (check_cfg passed).
static void norm_residual_core(const word_t* a, const word_t* b, const word_t* e, word_t* io, Cfg c) {
#pragma HLS DATAFLOW
#pragma HLS STABLE variable = c
  hls::stream<word_t, 32> fa_, fb_;
#pragma HLS BIND_STORAGE variable = fa_ type = fifo impl = srl
#pragma HLS BIND_STORAGE variable = fb_ type = fifo impl = srl
  hls::stream<word_t, FD> f1, f2, fy, fz;
#ifndef NR_FIFO_URAM  // explicit: HLS otherwise picks LUTRAM for some of them (and counts 0 LUT for it)
#pragma HLS BIND_STORAGE variable = f1 type = fifo impl = bram
#pragma HLS BIND_STORAGE variable = f2 type = fifo impl = bram
#pragma HLS BIND_STORAGE variable = fy type = fifo impl = bram
#pragma HLS BIND_STORAGE variable = fz type = fifo impl = bram
#else  // the four row FIFOs in URAM (2 x 72-bit URAM each) instead of BRAM
#pragma HLS BIND_STORAGE variable = f1 type = fifo impl = uram
#pragma HLS BIND_STORAGE variable = f2 type = fifo impl = uram
#pragma HLS BIND_STORAGE variable = fy type = fifo impl = uram
#pragma HLS BIND_STORAGE variable = fz type = fifo impl = uram
#endif
  hls::stream<word_t, FO_DEPTH> fo;
#pragma HLS BIND_STORAGE variable = fo type = fifo impl = lutram
  hls::stream<ap_uint<8>, 4> fe;
  hls::stream<Stat, 4> fs;
  hls::stream<MR, 4> fm;
  hls::stream<uint32_t, 4> fa;
  hls::stream<IS, 4> fq;
  hls::stream<uint32_t, 8> fsc;
  p_rdx(a, c.a_off, c.a_pitch, c.a_stride, true, c.a_flat, c.a_len, c, fa_);
  p_rdx(b, c.b_off, c.b_pitch, c.b_stride, (c.flags & F_RESID) != 0, c.b_flat, c.b_len, c, fb_);
  p_add(c, fa_, fb_, f1, fe);
  p_stats(c, f1, fe, f2, fs);
  p_scal(c, fs, fm);
  p_norm(e, c, f2, fm, fy, fz, fa);
  p_qscal(c, fa, fq);
  p_quant(c, fz, fq, fo, fsc);
  p_wr(io, c, fy, fo, fsc);
}

}  // namespace nr
