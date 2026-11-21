// NORM mode (opcode 0x30) of float_top under the integration contract 1.0 (fpga/kernels/CONTRACT.md):
// LayerNorm / RMSNorm of FP32 rows -> int16 rows + per-row scales (F_QOUT, the contract quantizer) or FP32 rows.
// Header only; the same code builds in Vitis HLS and in g++ (-I<Vitis_HLS>/include -ffp-contract=off).
// nr_emul.py is its bit-exact model (tb_nr.cpp checks whole DDR images word for word).
//
// Call from the sequencer's dispatch switch (one call site, INLINE off), after ctr::validate_hw(c) == ME_OK:
//     case ctr::OP_NORM: r = nr::norm_mode(c, w2, xmem, ymem); break;
// Ports: rows A are read only on io (xmem), gamma / beta (host constants) on one hp port (read only),
// every write goes out on io (ymem). No pointer argument, no register, no m_axi adapter of its own.
//
// Record (CONTRACT.md 3.3): K = values per row (8..1024, K % 8 == 0), N rows per item, B items;
//   A  FP32 rows   row n of item b at A.addr + b A.stride + n A.pitch
//   E  gamma [K] FP32 (shared by all items), beta [K] at E.addr + 4 K (F_LN)
//   C  F_QOUT: int16 rows (K / 8 words), rows [N, NP) written as zeros; else FP32 rows z (K / 4 words)
//   F  F_QOUT: scales, row n at F.addr + b F.stride + 4 n; pad4(NP) floats, pad lanes 1.0
//   p0 eps (FP32 bits), p4 scale shift (stored s * 2^p4, zero rows 1.0), p5 NP (0 = N)
//   flags F_QOUT, F_LN (F_RESID / F_SUM and the records layout are retired: the residual add is the GEMM epilogue)
// Every address is 64-bit in the record and 40-bit here (validate_hw rejects slot addresses >= 2^40); row and
// item offsets are formed by adding pitches / strides, never by a 32-bit product.
//
// Arithmetic (nr_emul.py has the same text):
//   stats    block floating point, exact integers: Emax = max exponent field; Y_i = round_half_away(
//            m_i 2^(1 + e_i - Emax)) (|Y_i| < 2^25); S = sum Y, Q = sum Y^2; num = K Q - S^2 (LN) or K Q (RMS)
//   scalars  mean = rn(S / K) 2^(Emax-151) (LN), var = rn(num / K^2) 2^(2 (Emax-151)): exact integer division,
//            correctly rounded (so an all-equal LayerNorm row gives y - mean = 0 and z = beta)
//            ve = max(fl(var + eps), 2^-126); rstd = rsqrt_nr(ve) (seed + 2 Newton), 0 for ve = inf
//   z        ((y - mean) * rstd) * g + b, FP32 cores (RMSNorm: b = 0)
//   q-out    the shared quantizer quantize/quant_int16.h (s = fl32(max(amax, 1e-30) / 32767), zero rows 1.0)
// Every FP32 operation flushes denormal inputs and results to zero with the sign kept (the Xilinx cores, contract
// 2.3); the g++ build models that with ctr::ref::ftz so the host C model is the hardware.
//
// DATAFLOW processes, one 128-bit word (4 FP32) per clock each:
//   ---- row-unit common parts (stand-ins for the shared reader / quantizer / writer of CONTRACT.md 4.1)
//   p_rd     A rows over io: one burst per item when the row gap is at most half a row (gap words dropped),
//            else one burst per row                                                          -> fa
//   ---- NORM front end (what the row unit takes from this kernel)
//   p_emax   per-row max exponent field                                                      -> f1 (row), fe
//   p_stats  Y_i, S, Q (4 squares a clock)                                                   -> f2 (row), fs
//   p_scal   num, then mean and var by one shared restoring divider (sequential, ~100 clocks a row) -> fv
//   p_rsqrt  ve, rstd (one shared fmul + fsub, ~70 clocks a row)                             -> fm
//   p_affine gamma / beta from hp into BRAM once per command; z = ((y - mean) rstd) g + b    -> fz
//   ---- row-unit common parts
//   p_qamax  row buffer + per-row absmax                                                     -> fq (row), fam
//   p_qscal  qz::row_scale (1 reciprocal bit a clock)                                        -> frs
//   p_quant  4 x qz::quant_elem a clock (or FP32 pass-through)                               -> fo, fsc
//   p_wr     C rows (one burst per <= 256-row chunk when C rows are contiguous, else per row), zero rows
//            [N, NP), then the chunk's scales in one burst (pad lanes 1.0)
// Steady state: max(K/4, A words read, C words written) clocks a row while the per-row scalar processes keep up,
// i.e. K >= 4 x 100 / ... : rows of K >= 512 run at the io rate; shorter rows are scalar-bound (README).
#pragma once
#include <ap_int.h>
#include <hls_stream.h>
#include <stdint.h>

#include "contract.h"
#include "nr_tables.h"
#include "quant_int16.h"

#ifdef NR_DFSIM
#include "nr_dfsim.h"
#endif

#ifndef NR_FADD_IMPL
#define NR_FADD_IMPL fulldsp  // FP32 add / sub cores of the normalize stage (fabric: 0 DSP, +140 LUT each)
#endif
#ifndef NR_FMUL_IMPL
#define NR_FMUL_IMPL fulldsp  // FP32 multipliers of the normalize stage: 2 DSP + 134 LUT each (meddsp: 1 DSP + 307 LUT)
#endif

namespace nr {

typedef ap_uint<128> word_t;
typedef ap_uint<40> baddr_t;  // byte address (validate_hw: < 2^40)
typedef ap_uint<36> waddr_t;  // 128-bit word index = byte address >> 4

constexpr int L = 4;                    // lanes: FP32 values per io word
constexpr int KMAX = ctr::NORM_K_MAX;   // 1024
constexpr int KWMAX = KMAX / L;         // 256 words in the longest row
#ifndef NR_RFD
#define NR_RFD 512
#endif
constexpr int RFD = NR_RFD;  // row FIFO depth (words): one BRAM depth. >= KWMAX + 1 or the region deadlocks
                             // (p_emax / p_qamax hold a whole row before its statistic); >= KWMAX + ~180 for
                             // full rate at K = 1024 (the per-row scalar latency)
constexpr int SCH = 256;                // writer chunk (rows): its scales are buffered, then written in one burst
constexpr uint32_t FLT_MIN_BITS = 0x00800000u;

#ifdef NR_DFSIM
template <class T>
using strm = dfsim::stream<T>;
#define NR_DECL(T, name, D) dfsim::stream<T> name(D, #name)
#define NR_RUN(call) dfsim::run([&] { call; })
#define NR_JOIN() dfsim::join_all()
#else
template <class T>
using strm = hls::stream<T>;
#define NR_DECL(T, name, D) hls::stream<T, D> name
#define NR_RUN(call) call
#define NR_JOIN()
#endif

#ifndef __SYNTHESIS__
extern long g_leftover;  // host only: streams left non-empty at the end of a call (tb_nr.cpp defines it)
#endif

// One command, decoded once by norm_mode (STABLE in the region; every process reads only what it needs).
struct Cfg {
  ap_uint<11> K;
  ap_uint<9> KW;    // K / 4: words of an FP32 row
  ap_uint<9> CW;    // words of an output row: K / 8 (F_QOUT) or K / 4
  ap_uint<17> N;    // rows per item (1..65535)
  ap_uint<17> NP;   // rows written per item (F_QOUT: N..65535; else N)
  ap_uint<17> B;    // items (1..65535)
  ap_uint<25> NKW;  // N * KW: words of an item (< 2^24)
  bool ln, qout;
  uint32_t neg_eps;  // -eps (FP32 bits): var + eps is computed as var - (-eps) on the one fsub core
  ap_int<5> shift;   // p4 (-8..8)
  // A (io read)
  waddr_t a_w;
  ap_uint<28> a_pw, a_sw;  // pitch, stride in words
  bool a_flat;
  ap_uint<26> a_len;  // flat: (N - 1) a_pw + KW words a burst
  // E (hp read)
  waddr_t e_w;
  // C, F (io write)
  waddr_t c_w, f_w;
  ap_uint<28> c_pw, c_sw, f_sw;
  bool c_flat;
  ap_uint<21> K2;  // K * K (p_scal left-aligns K and K^2 for the divider once per command)
};

struct Stat {
  ap_int<37> S;   // |S| < 2^35
  ap_uint<61> Q;  // < 2^60
  ap_uint<8> emax;
};
struct MV {
  uint32_t mean, var;
};
struct MR {
  float mean, rstd;
};
struct QS {  // per-row quantizer state
  qz::RowScale rs;
  bool zero;  // all-zero row: stored scale 1.0 whatever p4
};

// ------------------------------------------------------------------------------------------ FP32 cores
static inline float fp_mul(float a, float b) {
#pragma HLS INLINE
#ifdef __SYNTHESIS__
  float r = a * b;
#pragma HLS BIND_OP variable = r op = fmul impl = NR_FMUL_IMPL
  return r;
#else
  return ctr::ref::fmul_hw(a, b);
#endif
}
static inline float fp_add(float a, float b) {
#pragma HLS INLINE
#ifdef __SYNTHESIS__
  float r = a + b;
#pragma HLS BIND_OP variable = r op = fadd impl = NR_FADD_IMPL
  return r;
#else
  return ctr::ref::fadd_hw(a, b);
#endif
}
static inline float fp_sub(float a, float b) {
#pragma HLS INLINE
#ifdef __SYNTHESIS__
  float r = a - b;
#pragma HLS BIND_OP variable = r op = fsub impl = NR_FADD_IMPL
  return r;
#else
  return ctr::ref::fadd_hw(a, -b);  // a - b == a + (-b) exactly; ftz(-b) = -ftz(b)
#endif
}
// The per-row scalar process uses its own cores (one fmul, one fsub, shared by ALLOCATION there).
static inline float sc_mul(float a, float b) {
#pragma HLS INLINE
#ifdef __SYNTHESIS__
  float r = a * b;
#pragma HLS BIND_OP variable = r op = fmul impl = fulldsp
  return r;
#else
  return ctr::ref::fmul_hw(a, b);
#endif
}
static inline float sc_sub(float a, float b) {
#pragma HLS INLINE
#ifdef __SYNTHESIS__
  float r = a - b;
#pragma HLS BIND_OP variable = r op = fsub impl = fulldsp
  return r;
#else
  return ctr::ref::fadd_hw(a, -b);
#endif
}

static inline float lane_f(const word_t& w, int l) {
#pragma HLS INLINE
  return ctr::u2f((uint32_t)w.range(32 * l + 31, 32 * l));
}
static inline void set_lane_f(word_t& w, int l, float f) {
#pragma HLS INLINE
  w.range(32 * l + 31, 32 * l) = ctr::f2u(f);
}

// 1 / sqrt(v) for positive normal FP32 v (inlined: the caller shares one fmul and one fsub among its ops)
static inline float rsqrt_nr(float v) {
#pragma HLS INLINE
  static const uint32_t SEED[128] = NR_RSQRT_SEED_INIT;
  const uint32_t b = ctr::f2u(v);
  const int E = (int)((b >> 23) & 0xFF) - 127;
  const int p = E & 1;
  const int Eh = (E - p) / 2;  // exact: E - p is even
  const uint32_t idx = ((uint32_t)p << 6) | ((b >> 17) & 0x3F);
  float y = ctr::u2f((uint32_t)((int32_t)SEED[idx] - Eh * (1 << 23)));
  const uint32_t hb = ((b >> 23) & 0xFF) <= 1 ? (b & 0x80000000u) : b - (1u << 23);  // = fl(v * 0.5), flushed
  const float h = ctr::u2f(hb);
newton:
  for (int it = 0; it < 2; it++) {
#pragma HLS PIPELINE off
    float t = sc_mul(y, y);
    t = sc_mul(h, t);
    t = sc_sub(1.5f, t);
    y = sc_mul(y, t);
  }
  return y;
}

// ------------------------------------------------------------------------------------------ exact division
// rn(n / d) * 2^k as FP32 bits (half to even; biased exponent <= 0 -> +0, >= 255 -> inf), n >= 0, d > 0.
// d comes left-aligned (dn = d << (72 - ld), ld = bit length of d). Sequential: 8 + 7 normalizing shifts and
// 26 restoring-division steps, each one clock (a 73-bit subtract).
static inline uint32_t div_rn(ap_uint<72> n, ap_uint<72> dn, ap_uint<5> ld, int k, bool neg) {
#pragma HLS INLINE
  if (n == 0) return 0;
  ap_uint<72> x = n;
  ap_uint<7> lz = 0;
coarse:
  for (int i = 0; i < 8; i++) {
#pragma HLS PIPELINE II = 1
    if (x(71, 64) == 0) {
      x = x << 8;
      lz += 8;
    }
  }
fine:
  for (int i = 0; i < 7; i++) {
#pragma HLS PIPELINE II = 1
    if (!x[71]) {
      x = x << 1;
      lz += 1;
    }
  }
  // x / dn in (1/2, 2): q = floor(x / dn * 2^25), 26 bits
  ap_uint<73> R = x;
  ap_uint<26> q = 0;
divide:
  for (int i = 0; i < 26; i++) {
#pragma HLS PIPELINE II = 1
    const ap_uint<74> diff = ap_uint<74>(R) - ap_uint<74>(dn);
    const bool ge = !diff[73];
    const ap_uint<73> r1 = ge ? ap_uint<73>(diff(72, 0)) : R;
    R = r1 << 1;
    q = (q << 1) | ap_uint<26>(ge);
  }
  const bool rest = R != 0;
  // n / d = (x / dn) 2^(ln - ld), ln = 72 - lz
  int ex = (72 - (int)lz) - (int)ld;
  ap_uint<25> m;
  bool rb, st;
  if (q[25]) {
    m = q >> 2;
    rb = q[1];
    st = q[0] | rest;
  } else {
    m = q >> 1;
    rb = q[0];
    st = rest;
    ex -= 1;
  }
  if (rb & (st | m[0])) m += 1;
  if (m[24]) {
    m = ap_uint<25>(1) << 23;
    ex += 1;
  }
  const int field = ex + k + 127;
  const uint32_t sign = neg ? 0x80000000u : 0u;
  if (field <= 0) return 0;
  if (field >= 255) return sign | 0x7F800000u;
  return sign | ((uint32_t)field << 23) | ((uint32_t)m & 0x7FFFFFu);
}

// |Y| of one value against the row's max exponent field (stats), 25 bits
static inline ap_uint<25> yfix_mag(uint32_t b, ap_uint<8> emax) {
#pragma HLS INLINE
  const ap_uint<8> e = (b >> 23) & 0xFF;
  const ap_uint<24> m = e == 0 ? ap_uint<24>(0) : ap_uint<24>(0x800000u | (b & 0x7FFFFFu));
  // t = sh + 1 = emax - e in 0..255; w = floor(4 m / 2^t) = floor(m 2^(1 - sh)); |Y| = (w + 1) >> 1 rounds
  // m / 2^sh half away from zero (sh = -1: 2 m, sh = 0: m exactly; t >= 27: 0)
  const ap_uint<8> t = emax - e;
  const ap_uint<26> m4 = ap_uint<26>(m) << 2;
  const ap_uint<26> w = t > 26 ? ap_uint<26>(0) : ap_uint<26>(m4 >> ap_uint<5>(t));
  return ap_uint<25>((ap_uint<27>(w) + 1) >> 1);
}

// ------------------------------------------------------------------------------------------ processes
// One read burst of len words from word src + off; keeps the words whose position in a pw-word row is < KW.
static void rd_seg(const word_t* src, waddr_t off, ap_uint<26> len, ap_uint<28> pw, ap_uint<9> KW,
                   strm<word_t>& fo) {
#pragma HLS INLINE off
  ap_uint<28> jr = 0;
rd:
  for (ap_uint<26> j = 0; j < len; j++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 180 max = 131072 avg = 12000
    const word_t w = src[off + j];
    if (jr < KW) fo.write(w);
    jr = jr == pw - 1 ? ap_uint<28>(0) : ap_uint<28>(jr + 1);
  }
}

static void p_rd(const word_t* xmem, Cfg c, strm<word_t>& fa) {
  // flat: one segment per item over (N - 1) pitch + KW words; else N segments of KW words (one per row)
  const ap_uint<17> nseg = c.a_flat ? ap_uint<17>(1) : c.N;
  const ap_uint<26> len = c.a_flat ? c.a_len : ap_uint<26>(c.KW);
  const ap_uint<28> pw = c.a_flat ? c.a_pw : ap_uint<28>(c.KW);
  waddr_t ib = c.a_w;
items:
  for (ap_uint<17> b = 0; b < c.B; b++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 2 avg = 1
    waddr_t rb = ib;
  segs:
    for (ap_uint<17> n = 0; n < nseg; n++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 512 avg = 1
#pragma HLS LOOP_FLATTEN off
      rd_seg(xmem, rb, len, pw, c.KW, fa);
      rb += c.a_pw;
    }
    ib += c.a_sw;
  }
}

// per-row max exponent field; the row goes on to f1 (stats needs Emax before the row's first value)
static void p_emax(Cfg c, strm<word_t>& fa, strm<word_t>& f1, strm<ap_uint<8> >& fe) {
items:
  for (ap_uint<17> b = 0; b < c.B; b++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 2 avg = 1
    ap_uint<8> emax = 0;
    ap_uint<9> jr = 0;
  w:
    for (ap_uint<25> j = 0; j < c.NKW; j++) {
#pragma HLS LOOP_FLATTEN off
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 9000 max = 131072 avg = 20000
      const word_t w = fa.read();
      ap_uint<8> m = 0;
      for (int l = 0; l < L; l++) {
#pragma HLS UNROLL
        const ap_uint<8> e = w.range(32 * l + 30, 32 * l + 23);
        m = e > m ? e : m;
      }
      const ap_uint<8> e1 = (jr == 0 || m > emax) ? m : emax;
      emax = e1;
      f1.write(w);
      if (jr == c.KW - 1) {
        fe.write(e1);
        jr = 0;
      } else {
        jr++;
      }
    }
  }
}

static void p_stats(Cfg c, strm<word_t>& f1, strm<ap_uint<8> >& fe, strm<word_t>& f2, strm<Stat>& fs) {
items:
  for (ap_uint<17> b = 0; b < c.B; b++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 2 avg = 1
    ap_uint<8> emax = 0;
    ap_int<37> S = 0;
    ap_uint<61> Q = 0;
    ap_uint<9> jr = 0;
  acc:
    for (ap_uint<25> j = 0; j < c.NKW; j++) {
#pragma HLS LOOP_FLATTEN off
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 9000 max = 131072 avg = 20000
      if (jr == 0) emax = fe.read();
      const word_t w = f1.read();
      // sum of +-|Y|: a negative lane adds ~|Y| (its bits inverted) and 1 (counted in nneg): -x = ~x + 1
      ap_int<28> s4 = 0;
      ap_uint<3> nneg = 0;
      ap_uint<52> q4 = 0;
      for (int l = 0; l < L; l++) {
#pragma HLS UNROLL
        const uint32_t bits = (uint32_t)w.range(32 * l + 31, 32 * l);
        const ap_uint<25> mag = yfix_mag(bits, emax);
        const bool neg = bits >> 31;
        const ap_int<27> sm = ap_int<27>(ap_uint<27>(mag) ^ (neg ? ap_uint<27>(0x7FFFFFF) : ap_uint<27>(0)));
        s4 += sm;
        nneg += neg;
        q4 += ap_uint<50>(mag * mag);
      }
      s4 += nneg;
      const ap_int<37> S1 = (jr == 0 ? ap_int<37>(0) : S) + s4;
      const ap_uint<61> Q1 = (jr == 0 ? ap_uint<61>(0) : Q) + q4;
      S = S1;
      Q = Q1;
      f2.write(w);
      if (jr == c.KW - 1) {
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
}

// num = K Q - S^2 (LN) or K Q; var = rn(num / K^2) 2^(2 k1); mean = rn(S / K) 2^k1 (LN). One divider instance
// (one call site in a 2-iteration loop), ~100 clocks a LayerNorm row, ~50 an RMSNorm row.
static void p_scal(Cfg c, strm<Stat>& fs, strm<MV>& fv) {
  // divisors K and K^2 left-aligned to bit 71 (one shift a clock, once per command: no barrel shifter)
  ap_uint<72> kn = c.K, k2n = c.K2;
  ap_uint<5> lk = 0, lk2 = 0;  // bit lengths = 72 - shifts
  ap_uint<7> sk = 0, sk2 = 0;
align:
  for (int i = 0; i < 71; i++) {
#pragma HLS PIPELINE II = 1
    if (!kn[71]) {
      kn = kn << 1;
      sk++;
    }
    if (!k2n[71]) {
      k2n = k2n << 1;
      sk2++;
    }
  }
  lk = 72 - sk;
  lk2 = 72 - sk2;
items:
  for (ap_uint<17> b = 0; b < c.B; b++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 2 avg = 1
  rows:
    for (ap_uint<17> r = 0; r < c.N; r++) {
#pragma HLS LOOP_TRIPCOUNT min = 50 max = 512 avg = 100
#pragma HLS LOOP_FLATTEN off
#pragma HLS PIPELINE off
      const Stat st = fs.read();
      const bool sneg = st.S < 0;
      const ap_uint<36> Sa = sneg ? ap_uint<36>(-st.S) : ap_uint<36>(st.S);
      const ap_uint<72> KQ = ap_uint<72>(st.Q) * c.K;
      const ap_uint<72> S2 = c.ln ? ap_uint<72>(Sa * Sa) : ap_uint<72>(0);
      const ap_uint<72> num = KQ - S2;
      const int k1 = (int)st.emax - 151;
      uint32_t res[2] = {0, 0};
    div2:
      for (int it = 0; it < 2; it++) {
#pragma HLS PIPELINE off
        if (it == 1 && !c.ln) break;
        const ap_uint<72> nn = it == 0 ? num : ap_uint<72>(Sa);
        const ap_uint<72> dn = it == 0 ? k2n : kn;
        const ap_uint<5> ld = it == 0 ? lk2 : lk;
        res[it] = div_rn(nn, dn, ld, it == 0 ? 2 * k1 : k1, it == 1 && sneg);
      }
      MV o;
      o.var = res[0];
      o.mean = res[1];
      fv.write(o);
    }
  }
}

static void p_rsqrt(Cfg c, strm<MV>& fv, strm<MR>& fm) {
#pragma HLS ALLOCATION operation instances = fmul limit = 1
#pragma HLS ALLOCATION operation instances = fsub limit = 1
  const float neg_eps = ctr::u2f(c.neg_eps);
items:
  for (ap_uint<17> b = 0; b < c.B; b++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 2 avg = 1
  rows:
    for (ap_uint<17> r = 0; r < c.N; r++) {
#pragma HLS LOOP_TRIPCOUNT min = 50 max = 512 avg = 100
#pragma HLS LOOP_FLATTEN off
#pragma HLS PIPELINE off
      const MV mv = fv.read();
      uint32_t vb = ctr::f2u(sc_sub(ctr::u2f(mv.var), neg_eps));
      if (vb < FLT_MIN_BITS) vb = FLT_MIN_BITS;  // a denormal eps on an all-zero row (validate only asks eps > 0)
      MR o;
      o.mean = ctr::u2f(mv.mean);
      o.rstd = vb >= 0x7F800000u ? 0.0f : rsqrt_nr(ctr::u2f(vb));
      fm.write(o);
    }
  }
}

// gamma (and beta) from hp into BRAM once per command, then z = ((y - mean) rstd) g + b, 4 lanes a clock
static void p_affine(const word_t* hp, Cfg c, strm<word_t>& f2, strm<MR>& fm, strm<word_t>& fz) {
  word_t gtab[KWMAX], btab[KWMAX];
#ifdef NR_URAM
#pragma HLS BIND_STORAGE variable = gtab type = ram_s2p impl = uram
#pragma HLS BIND_STORAGE variable = btab type = ram_s2p impl = uram
#else
#pragma HLS BIND_STORAGE variable = gtab type = ram_s2p impl = bram
#pragma HLS BIND_STORAGE variable = btab type = ram_s2p impl = bram
#endif
  // one burst: gamma (KW words), then beta (KW words, F_LN) right behind it
  const ap_uint<10> nload = c.ln ? ap_uint<10>(2 * c.KW) : ap_uint<10>(c.KW);
  ap_uint<9> jl = 0;
  bool second = false;
load_gb:
  for (ap_uint<10> j = 0; j < nload; j++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 180 max = 512 avg = 200
    const word_t v = hp[c.e_w + j];
    if (second)
      btab[jl] = v;
    else
      gtab[jl] = v;
    if (jl == c.KW - 1) {
      jl = 0;
      second = true;
    } else {
      jl++;
    }
  }
items:
  for (ap_uint<17> b = 0; b < c.B; b++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 2 avg = 1
    MR mr;
    mr.mean = 0.0f;
    mr.rstd = 0.0f;
    ap_uint<9> jr = 0;
  nz:
    for (ap_uint<25> j = 0; j < c.NKW; j++) {
#pragma HLS LOOP_FLATTEN off
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 9000 max = 131072 avg = 20000
      if (jr == 0) mr = fm.read();
      const word_t w = f2.read();
      const word_t g = gtab[jr];
      const word_t bb = c.ln ? btab[jr] : word_t(0);
      word_t wz;
      for (int l = 0; l < L; l++) {
#pragma HLS UNROLL
        float t = fp_sub(lane_f(w, l), mr.mean);
        t = fp_mul(t, mr.rstd);
        t = fp_mul(t, lane_f(g, l));
        set_lane_f(wz, l, fp_add(t, lane_f(bb, l)));
      }
      fz.write(wz);
      jr = jr == c.KW - 1 ? ap_uint<9>(0) : ap_uint<9>(jr + 1);
    }
  }
}

// ---- quantizer stage (row unit common part): row buffer + absmax, scale, 4 x quant_elem a clock
static void p_qamax(Cfg c, strm<word_t>& fz, strm<word_t>& fq, strm<qz::abits_t>& fam) {
items:
  for (ap_uint<17> b = 0; b < c.B; b++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 2 avg = 1
    qz::abits_t a = 0;
    ap_uint<9> jr = 0;
  w:
    for (ap_uint<25> j = 0; j < c.NKW; j++) {
#pragma HLS LOOP_FLATTEN off
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 9000 max = 131072 avg = 20000
      const word_t w = fz.read();
      const qz::abits_t m = qz::word_amax(w);
      const qz::abits_t a1 = (jr == 0 || m > a) ? m : a;
      a = a1;
      fq.write(w);
      if (jr == c.KW - 1) {
        if (c.qout) fam.write(a1);
        jr = 0;
      } else {
        jr++;
      }
    }
  }
}

static void p_qscal(Cfg c, strm<qz::abits_t>& fam, strm<QS>& frs) {
  if (!c.qout) return;
items:
  for (ap_uint<17> b = 0; b < c.B; b++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 2 avg = 1
  rows:
    for (ap_uint<17> r = 0; r < c.N; r++) {
#pragma HLS LOOP_TRIPCOUNT min = 50 max = 512 avg = 100
#pragma HLS LOOP_FLATTEN off
#pragma HLS PIPELINE off
      const qz::abits_t a = fam.read();
      QS o;
      o.rs = qz::row_scale(a);
      o.zero = a == 0;
      frs.write(o);
    }
  }
}

static void p_quant(Cfg c, strm<word_t>& fq, strm<QS>& frs, strm<word_t>& fo, strm<uint32_t>& fsc) {
items:
  for (ap_uint<17> b = 0; b < c.B; b++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 2 avg = 1
    QS qs;
    qs.rs = qz::row_scale_from_s(ap_uint<32>(qz::ONE_BITS));
    qs.zero = true;
    ap_uint<64> lo = 0;
    ap_uint<9> jr = 0;
  q:
    for (ap_uint<25> j = 0; j < c.NKW; j++) {
#pragma HLS LOOP_FLATTEN off
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 9000 max = 131072 avg = 20000
      if (c.qout && jr == 0) qs = frs.read();
      const word_t w = fq.read();
      ap_uint<64> h;
      for (int l = 0; l < L; l++) {
#pragma HLS UNROLL
        h.range(16 * l + 15, 16 * l) = ap_uint<16>(qz::quant_elem(qz::word_f32(w, l), qs.rs));
      }
      if (!c.qout) {
        fo.write(w);
      } else if (jr[0]) {
        word_t o;
        o.range(63, 0) = lo;
        o.range(127, 64) = h;
        fo.write(o);
      } else {
        lo = h;
      }
      if (jr == c.KW - 1) {
        if (c.qout) {
          // stored scale: s 2^p4 (s is normal and stays normal for |p4| <= 8), all-zero rows 1.0
          const uint32_t sb = (uint32_t)qs.rs.s_bits + (uint32_t)((int)c.shift * (1 << 23));
          fsc.write(qs.zero ? qz::ONE_BITS : sb);
        }
        jr = 0;
      } else {
        jr++;
      }
    }
  }
}

// ---- writer (row unit common part)
// Flat: nr rows of CW words from word off; rows >= nv are zero rows (written here, not read from fo). At each row
// end the row's scale goes into sbuf.
static void wr_flat(word_t* io, waddr_t off, ap_uint<9> nr, ap_uint<9> nv, ap_uint<9> CW, bool qout,
                    strm<word_t>& fo, strm<uint32_t>& fsc, uint32_t sbuf[SCH]) {
#pragma HLS INLINE off
  const ap_uint<17> len = nr * CW;
  ap_uint<9> jr = 0, n = 0;
wr:
  for (ap_uint<17> j = 0; j < len; j++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 4500 max = 65536 avg = 20000
    const bool real = n < nv;
    io[off + j] = real ? fo.read() : word_t(0);
    if (jr == CW - 1) {
      if (qout) sbuf[n] = real ? fsc.read() : qz::ONE_BITS;
      n++;
      jr = 0;
    } else {
      jr++;
    }
  }
}

static void wr_row(word_t* io, waddr_t off, ap_uint<9> CW, bool real, strm<word_t>& fo) {
#pragma HLS INLINE off
wr:
  for (ap_uint<9> j = 0; j < CW; j++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 90 max = 256 avg = 120
    io[off + j] = real ? fo.read() : word_t(0);
  }
}

// nw scale words from sbuf (4 a word); lanes at or past nv are padding (1.0)
static void wr_scales(word_t* io, waddr_t off, ap_uint<9> nr, const uint32_t sbuf[SCH]) {
#pragma HLS INLINE off
  const ap_uint<7> nw = (nr + 3) >> 2;
wf:
  for (ap_uint<7> w = 0; w < nw; w++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 13 max = 64 avg = 20
    word_t sw;
    for (int l = 0; l < 4; l++) {
#pragma HLS UNROLL
      const ap_uint<9> i = 4 * w + l;
      sw.range(32 * l + 31, 32 * l) = i < nr ? sbuf[i] : qz::ONE_BITS;
    }
    io[off + w] = sw;
  }
}

static void p_wr(word_t* io, Cfg c, strm<word_t>& fo, strm<uint32_t>& fsc) {
  uint32_t sbuf[SCH];
#pragma HLS ARRAY_PARTITION variable = sbuf cyclic factor = 4
#pragma HLS BIND_STORAGE variable = sbuf type = ram_2p impl = lutram
  waddr_t ic = c.c_w, iF = c.f_w;
items:
  for (ap_uint<17> b = 0; b < c.B; b++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 2 avg = 1
    waddr_t rc = ic, rf = iF;
    ap_uint<17> n0 = 0;
  chunks:
    while (n0 < c.NP) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 2 avg = 1
      const ap_uint<17> left = c.NP - n0;
      const ap_uint<9> nr = left < SCH ? ap_uint<9>(left) : ap_uint<9>(SCH);
      const ap_uint<17> vleft = c.N > n0 ? ap_uint<17>(c.N - n0) : ap_uint<17>(0);
      const ap_uint<9> nv = vleft < nr ? ap_uint<9>(vleft) : nr;  // real (not zero-padding) rows of the chunk
      if (c.c_flat) {
        wr_flat(io, rc, nr, nv, c.CW, c.qout, fo, fsc, sbuf);
        rc += nr * c.CW;
      } else {
      rows:
        for (ap_uint<9> r = 0; r < nr; r++) {
#pragma HLS LOOP_TRIPCOUNT min = 50 max = 256 avg = 100
          const bool real = r < nv;
          wr_row(io, rc, c.CW, real, fo);
          if (c.qout) sbuf[r] = real ? fsc.read() : qz::ONE_BITS;
          rc += c.c_pw;
        }
      }
      if (c.qout) {
        wr_scales(io, rf, nr, sbuf);
        rf += SCH / 4;
      }
      n0 += nr;
    }
    ic += c.c_sw;
    iF += c.f_sw;
  }
}

// ------------------------------------------------------------------------------------------ the region
static void norm_df(const word_t* hp, const word_t* xmem, word_t* ymem, Cfg c) {
#pragma HLS DATAFLOW
#pragma HLS STABLE variable = c
  NR_DECL(word_t, fa, 32);
#pragma HLS BIND_STORAGE variable = fa type = fifo impl = srl
  NR_DECL(word_t, f1, RFD);
  NR_DECL(word_t, f2, RFD);
  NR_DECL(word_t, fq, RFD);
#ifdef NR_URAM
#pragma HLS BIND_STORAGE variable = f1 type = fifo impl = uram
#pragma HLS BIND_STORAGE variable = f2 type = fifo impl = uram
#else
#pragma HLS BIND_STORAGE variable = f1 type = fifo impl = bram
#pragma HLS BIND_STORAGE variable = f2 type = fifo impl = bram
#endif
#pragma HLS BIND_STORAGE variable = fq type = fifo impl = bram
  NR_DECL(word_t, fz, 32);
#pragma HLS BIND_STORAGE variable = fz type = fifo impl = srl
  NR_DECL(word_t, fo, 32);
#pragma HLS BIND_STORAGE variable = fo type = fifo impl = srl
  NR_DECL(ap_uint<8>, fe, 4);
  NR_DECL(Stat, fs, 4);
  NR_DECL(MV, fv, 4);
  NR_DECL(MR, fm, 4);
  NR_DECL(qz::abits_t, fam, 4);
  NR_DECL(QS, frs, 4);
  NR_DECL(uint32_t, fsc, 8);
  NR_RUN(p_rd(xmem, c, fa));
  NR_RUN(p_emax(c, fa, f1, fe));
  NR_RUN(p_stats(c, f1, fe, f2, fs));
  NR_RUN(p_scal(c, fs, fv));
  NR_RUN(p_rsqrt(c, fv, fm));
  NR_RUN(p_affine(hp, c, f2, fm, fz));
  NR_RUN(p_qamax(c, fz, fq, fam));
  NR_RUN(p_qscal(c, fam, frs));
  NR_RUN(p_quant(c, fq, frs, fo, fsc));
  NR_RUN(p_wr(ymem, c, fo, fsc));
  NR_JOIN();
#ifndef __SYNTHESIS__
  g_leftover += !fa.empty() + !f1.empty() + !f2.empty() + !fq.empty() + !fz.empty() + !fo.empty() + !fe.empty() +
                !fs.empty() + !fv.empty() + !fm.empty() + !fam.empty() + !frs.empty() + !fsc.empty();
#endif
}

static inline ap_uint<5> bitlen(uint32_t v) {
#pragma HLS INLINE
  ap_uint<5> n = 0;
  for (int i = 0; i < 21; i++) {
#pragma HLS UNROLL
    if ((v >> i) & 1) n = i + 1;
  }
  return n;
}

// The mode function: decode the command's fields once, then the DATAFLOW region. Returns ME_OK, or ME_DIMS for a
// command whose counters could not hold it (validate_hw already rejects it; this guard only keeps the region from
// hanging when called without it). hp: the hp port gamma / beta sit behind; xmem / ymem: the io bundle.
static uint32_t norm_mode(const ctr::Cmd& cmd, const word_t* hp, const word_t* xmem, word_t* ymem) {
#pragma HLS INLINE off
  const uint32_t K = cmd.K, N = cmd.N, B = cmd.B;
  const bool qout = cmd.flags & ctr::F_QOUT;
  const uint32_t NP = cmd.p[ctr::P_NP] ? cmd.p[ctr::P_NP] : N;
  if (cmd.op != ctr::OP_NORM || K < 8 || K > (uint32_t)KMAX || (K & 7) || N < 1 || N > ctr::N_MAX || B < 1 ||
      B > ctr::B_MAX || NP < N || NP > ctr::N_MAX || (!qout && NP != N))
    return ctr::ME_DIMS;
  Cfg c;
  c.K = K;
  c.KW = K / 4;
  c.CW = qout ? K / 8 : K / 4;
  c.N = N;
  c.NP = NP;
  c.B = B;
  c.NKW = ap_uint<25>(N) * ap_uint<9>(K / 4);
  c.ln = cmd.flags & ctr::F_LN;
  c.qout = qout;
  c.neg_eps = cmd.p[ctr::P_EPS] ^ 0x80000000u;
  c.shift = (int)cmd.p[ctr::P_SHIFT];
  const ctr::Slot& A = cmd.s[ctr::SLOT_A];
  const ctr::Slot& E = cmd.s[ctr::SLOT_E];
  const ctr::Slot& Cs = cmd.s[ctr::SLOT_C];
  const ctr::Slot& F = cmd.s[ctr::SLOT_F];
  c.a_w = baddr_t(A.addr) >> 4;
  c.a_pw = A.pitch >> 4;
  c.a_sw = A.stride >> 4;
  // one burst per item when the gap after each row is at most half a row (the expert's residual rows: 736 of 720)
  c.a_flat = N == 1 || (c.a_pw >= c.KW && 2 * c.a_pw <= 3 * c.KW);
  c.a_len = c.a_flat ? ap_uint<26>(ap_uint<17>(N - 1) * ap_uint<9>(N == 1 ? 0u : (uint32_t)c.a_pw) + c.KW)
                     : ap_uint<26>(0);
  c.e_w = baddr_t(E.addr) >> 4;
  c.c_w = baddr_t(Cs.addr) >> 4;
  c.c_pw = Cs.pitch >> 4;
  c.c_sw = Cs.stride >> 4;
  c.c_flat = NP == 1 || c.c_pw == c.CW;
  c.f_w = baddr_t(F.addr) >> 4;
  c.f_sw = F.stride >> 4;
  c.K2 = ap_uint<11>(K) * ap_uint<11>(K);
  norm_df(hp, xmem, ymem, c);
  return ctr::ME_OK;
}

}  // namespace nr
