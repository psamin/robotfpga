// Row softmax kernel for SmolVLA attention on the KR260 (K26), header-only core (README.md has the contract).
//
// One call = one 128-byte command record (the float_top sequencer's record format, opcode SOFTMAX 0x40) over
// B batch items of N rows x L keys of FP32 scores. Two layouts:
//   rows   (default): scores A[item][row][key] row-major with a pitch -> q C[item][row][key] int16 rows with a
//                     pitch + one FP32 scale per row F[item][row]; or FP32 probabilities C (F_QOUT clear)
//   engine (F_ENGINE): scores = the int16 engine's output tiling Y[Mk/32][Np][32] FP32 (Mk = pad32(L) keys as
//                     output channels, Np = pad32(N) rows as tokens) -> P as the engine's X operand of the P V
//                     call: X[Np/32][L][32 rows] int16 + s_tok[Np] FP32 (the xmem sections of int16_linear.h)
// q_j in [0, 32767], scale = RN(1 / sum_j q_j), so P_j = q_j * scale. Masks: key k visible to row n iff
// 0 <= E[k] <= D[n] (int32 rank vectors, F_MASK); a fully masked row is zero (or uniform with F_EMPTY_UNIFORM,
// torch's softmax of equal finfo.min scores).
//
// The arithmetic is integer from the score bits to the scale bits (fixed-point 2^x tables, an exact divider);
// smx_emu.py emulates it bit for bit. Structure: one DATAFLOW region, every loop pipelined at II = 1:
//   pass1     hp0: one burst per item in memory order; masks; row max on FP32 order keys       (4 lanes)
//   pass2     hp1: the same words again; 2^((x - max) c) in fixed point, q, row sums            (4 lanes)
//   recip     RN(1 / sum): normalised restoring division, two 1-bit units, 13 cycles per row
//   shaper    rows: 8 q per word / FP32 words; engine: 32 x 32 transposes (ping-pong, LUTRAM)
//   writer    io: data words in bursts, scales 4 per word
// Reading the scores twice (two ports) replaces a row buffer and makes the engine layout possible at all:
// there a row's keys are spread over Mk/32 groups of the item, so its max is known only after the whole item.
//
// Plain C++ on the host (g++ with the Vitis HLS include directory): the processes run one after the other on
// unbounded hls::stream FIFOs; every operation is ap_int except the FP32-output multiply and the descriptor's
// c = p0 * log2(e), which are IEEE single multiplies on both sides.
#pragma once
#include <ap_int.h>
#include <hls_stream.h>
#include <stdint.h>
#include <string.h>

#include "softmax_tables.h"

#ifndef SMX_L_MAX
#define SMX_L_MAX 256  // keys per row at most (E table, FP32-mode row FIFO)
#endif
#ifndef SMX_FP32_OUT
#define SMX_FP32_OUT 0  // 1 adds the FP32 output (rows layout): 4 fmul + a row FIFO behind the shaper
#endif
#ifndef SMX_ENGINE
#define SMX_ENGINE 1  // 0 removes the engine layout (row state RAMs, transposer)
#endif

namespace smx {

typedef ap_uint<128> Word;
typedef ap_uint<36> WAddr;  // 128-bit word index = byte address >> 4 (40-bit addresses)

constexpr int L_MAX = SMX_L_MAX;
static_assert(L_MAX % 32 == 0 && L_MAX >= 32 && L_MAX <= 2048, "SMX_L_MAX: multiple of 32 in [32, 2048]");
constexpr int R_MAX = 256;                               // rows per item: engine Np, masked rows layout
constexpr int EW = L_MAX / 4;                            // E table entries (4 ranks of 16 bits)
constexpr int DW = R_MAX / 4;                            // D table entries
constexpr int NSTEP = 13;                                // divider cycles per row and unit pair
constexpr int BCFIFO = SMX_FP32_OUT ? L_MAX / 4 + 64 : 32;
constexpr int BRW = 256;                                 // rows layout: rows per scale block
constexpr int KMIN = (int)0x80000000;                    // order key below every FP32 key

// ---- record (fpga/kernels/sequencer/seq_format.h layout)
constexpr uint32_t OP_SOFTMAX = 0x40;
constexpr uint32_t REC_MAGIC = 0xC5;
constexpr uint32_t F_QOUT = 0x0002;           // int16 q + FP32 scales (else FP32 probabilities, rows only)
constexpr uint32_t F_MASK = 0x0100;           // key k visible to row n iff 0 <= E[k] <= D[n]
constexpr uint32_t F_ENGINE = 0x1000;         // engine layouts (Y tiling in, X interleave + s_tok out)
constexpr uint32_t F_EMPTY_UNIFORM = 0x2000;  // fully masked row: uniform over its L keys (else zero)
constexpr uint32_t LOG2E_BITS = 0x3FB8AA3B;   // fl32(log2(e))
enum Status { ST_OK = 0, ST_DIMS = 1, ST_ALIGN = 2, ST_OPCODE = 3 };

struct Params {
  int B, N, L;                 // items, rows per item, keys per row
  bool engine, qout, mask, uniform;
  ap_uint<32> c_bits;          // c = fl32(p0 * fl32(log2 e)), positive normal
  int a_pitch, c_pitch;        // rows layout: row pitch of A / C in words
  WAddr a_off, c_off, d_off, e_off, f_off;
  WAddr a_stride, c_stride, d_stride, e_stride, f_stride;  // per item, words
  // derived once by decode() (so the processes need no multipliers of their own)
  int G;         // key groups: engine Mk/32, rows 1
  int NR;        // rows per group: engine Np, rows N
  int SW;        // words per row segment: engine 8, rows a_pitch
  int wlast;     // last data word of a segment: engine 7, rows ceil(L/4) - 1
  int nw;        // input words per item (one burst)
  int rows_all;  // rows through the divider: B * NR
  int ent_rows;  // rows layout: pass-2 entries of the call, B * N * ceil(L/4)
  int tiles;     // engine: 32 x 32 tiles per item, G * NR / 32
};

struct MX {  // pass1 -> pass2: one row's max
  ap_uint<32> bits;
  bool any;
};
struct DN {  // pass2 -> recip: the row sum S normalised, d = S << (34 - p), p = msb(S); zero = (S == 0)
  ap_uint<35> d;
  ap_uint<6> p;
  bool zero;
};
typedef ap_uint<92> EQ;  // pass2 -> shaper: 4 lanes of 23 bits (2^x values, or q in uint16 mode)

static inline float u2f(uint32_t u) {
#pragma HLS INLINE
  float f;
  memcpy(&f, &u, sizeof f);
  return f;
}
static inline uint32_t f2u(float f) {
#pragma HLS INLINE
  uint32_t u;
  memcpy(&u, &f, sizeof u);
  return u;
}

// ------------------------------------------------------------------ geometry of one item
static inline void derive(Params& p) {
#pragma HLS INLINE
  if (SMX_ENGINE && p.engine) {
    p.G = (p.L + 31) >> 5;
    p.NR = (p.N + 31) & ~31;
    p.SW = 8;
    p.wlast = 7;
  } else {
    p.G = 1;
    p.NR = p.N;
    p.SW = p.a_pitch;
    p.wlast = ((p.L + 3) >> 2) - 1;
  }
  p.nw = p.G * p.NR * p.SW;
  p.rows_all = p.B * p.NR;
  p.ent_rows = p.B * p.N * (p.wlast + 1);
  p.tiles = p.G * (p.NR >> 5);
}
struct Geo {
  int G, NR, SW, wlast, nw;
};
static inline Geo geometry(const Params& p) {
#pragma HLS INLINE
  Geo g;
  g.G = p.G;
  g.NR = p.NR;
  g.SW = p.SW;
  g.wlast = p.wlast;
  g.nw = p.nw;
  return g;
}

// ------------------------------------------------------------------ record decode and checks
static inline uint32_t field(const Word* rec, int j) {
#pragma HLS INLINE
  return (uint32_t)rec[j >> 2].range(32 * (j & 3) + 31, 32 * (j & 3));
}
// slot s (A 0 .. F 5): byte address (40 bits), pitch (bytes), batch stride (bytes) -> words; misaligned -> false
static inline bool slot(const Word* rec, int s, WAddr& off, int& pitch, WAddr& stride) {
#pragma HLS INLINE
  const uint32_t w0 = field(rec, 6 + 3 * s), w1 = field(rec, 7 + 3 * s), w2 = field(rec, 8 + 3 * s);
  const ap_uint<40> addr = ((ap_uint<40>)(w1 & 0xFF) << 32) | w0;
  const uint32_t pb = w1 >> 8;
  off = addr >> 4;
  pitch = (int)(pb >> 4);
  stride = (WAddr)(w2 >> 4);
  return (addr & 15) == 0 && (pb & 15) == 0 && (w2 & 15) == 0;
}
// decode the 8-word record at rec (opcode, magic and check word are the sequencer's business; checked here
// only for the opcode), then check the dimensions. Returns a Status.
static int decode(const Word* rec, Params& p) {
  const uint32_t f0 = field(rec, 0);
  const uint32_t flags = f0 >> 16;
  p.L = (int)field(rec, 3);
  p.N = (int)field(rec, 4);
  const uint32_t b = field(rec, 5) & 0xFFFF;
  p.B = b == 0 ? 1 : (int)b;
  p.engine = SMX_ENGINE && (flags & F_ENGINE);
  p.qout = flags & F_QOUT;
  p.mask = flags & F_MASK;
  p.uniform = flags & F_EMPTY_UNIFORM;
  const float c = u2f(field(rec, 24)) * u2f(LOG2E_BITS);  // one IEEE FP32 multiply per call
  p.c_bits = f2u(c);
  int pa, pc, pd, pe, pf;
  bool al = slot(rec, 0, p.a_off, pa, p.a_stride);
  al &= slot(rec, 2, p.c_off, pc, p.c_stride);
  al &= slot(rec, 3, p.d_off, pd, p.d_stride);
  al &= slot(rec, 4, p.e_off, pe, p.e_stride);
  al &= slot(rec, 5, p.f_off, pf, p.f_stride);
  p.a_pitch = pa;
  p.c_pitch = pc;
  if ((f0 & 0xFF) != OP_SOFTMAX) return ST_OPCODE;
  if (!al) return ST_ALIGN;
  const ap_uint<32> cb = p.c_bits;
  const int ce = cb.range(30, 23);
  if (cb[31] || ce == 0 || ce == 255) return ST_DIMS;  // c must be a positive normal
  if (p.L < 1 || p.L > L_MAX || p.N < 1) return ST_DIMS;
  const int L4 = (p.L + 3) >> 2;
  if ((flags & F_ENGINE) && !SMX_ENGINE) return ST_DIMS;
  if (p.engine) {
    if (((p.N + 31) & ~31) > R_MAX || !p.qout) return ST_DIMS;
  } else {
    if (p.a_pitch < L4 || p.c_pitch < (p.qout ? (p.L + 7) >> 3 : L4)) return ST_DIMS;
    if (!p.qout && !SMX_FP32_OUT) return ST_DIMS;
    if (p.mask && p.N > R_MAX) return ST_DIMS;
  }
  derive(p);
  return ST_OK;
}

// ------------------------------------------------------------------ arithmetic
// FP32 total-order integer key: monotonic in the float value; its own inverse on the bit pattern.
static inline ap_int<32> fkey(ap_uint<32> b) {
#pragma HLS INLINE
  ap_uint<32> k = b[31] ? (ap_uint<32>)(b ^ 0x7FFFFFFFu) : b;
  return (ap_int<32>)k;
}

// |x| * 2^ec on the Q.21 grid as a 47-bit magnitude, truncated toward zero, saturating at 2^47 - 1 (also inf/NaN);
// e == 0 (zero/denormal) -> 0.
static inline ap_uint<47> fixmag(ap_uint<32> b, ap_int<10> ec) {
#pragma HLS INLINE
  const ap_uint<8> e = b.range(30, 23);
  const ap_uint<48> m = ((ap_uint<48>)((ap_uint<24>)b.range(22, 0) | (ap_uint<24>)0x800000)) << 24;
  const ap_int<12> k = (ap_int<12>)153 - (ap_int<12>)e - (ap_int<12>)ec;  // right shift of M << 24
  ap_uint<47> v;
  if (e == 0 || k >= 48)
    v = 0;
  else if (k <= 0 || e == 255)
    v = (ap_uint<47>)0x7FFFFFFFFFFFULL;
  else
    v = (ap_uint<47>)(m >> (int)k);
  return v;
}
static inline ap_int<48> fix(ap_uint<32> b, ap_int<10> ec) {
#pragma HLS INLINE
  const ap_uint<47> v = fixmag(b, ec);
  return b[31] ? (ap_int<48>)(-(ap_int<48>)v) : (ap_int<48>)v;
}

// c = Mc * 2^ec, Mc = round(significand * 2^16) in [2^16, 2^17)
static inline void cparams(ap_uint<32> cb, ap_uint<17>& mc, ap_int<10>& ec) {
#pragma HLS INLINE
  const ap_uint<24> cm = (ap_uint<24>)cb.range(22, 0) | (ap_uint<24>)0x800000;
  const ap_uint<18> mcr = ((ap_uint<18>)(cm >> 6) + 1) >> 1;
  const bool cup = mcr[17];
  mc = cup ? (ap_uint<17>)(mcr >> 1) : (ap_uint<17>)mcr;
  ec = (ap_int<10>)((int)cb.range(30, 23) - 127 + (cup ? 1 : 0));
}

// 2^(-t) * 32767/32768 * 2^23 for t = u * Mc in Q.37 (u: Q.21 distance to the max, Mc: Q1.16 of c's significand).
// One table copy (1024 x 36 bits, one BRAM36, two ports) per pair of lanes.
template <int PAIR>
static inline void exp_pair(ap_uint<26> u0, ap_uint<26> u1, ap_uint<17> mc, ap_uint<23>& e0, ap_uint<23>& e1) {
#pragma HLS INLINE
  static const ap_uint<36> TAB[1024] = SMX_TAB_INIT;
#pragma HLS BIND_STORAGE variable = TAB type = rom_2p impl = bram
  ap_uint<26> u[2] = {u0, u1};
  ap_uint<23> e[2];
  for (int l = 0; l < 2; l++) {
#pragma HLS UNROLL
    const ap_uint<43> t = (ap_uint<43>)u[l] * mc;
    const ap_uint<6> I = t.range(42, 37);
    const ap_uint<10> ix = t.range(36, 27);
    const ap_uint<13> flo = t.range(26, 14);
    const ap_uint<36> te = TAB[ix];
    const ap_uint<23> T = te.range(35, 13);
    const ap_uint<13> S = te.range(12, 0);
    const ap_uint<26> corr = S * flo;
    const ap_uint<23> e23 = T - (ap_uint<23>)corr.range(25, 13);
    const ap_uint<24> h = ((ap_uint<24>)e23 << 1) >> (int)I;  // round(e23 / 2^I), half up
    e[l] = (I >= 24) ? (ap_uint<23>)0 : (ap_uint<23>)((h + 1) >> 1);
  }
  e0 = e[0];
  e1 = e[1];
}

// Exact FP32 bits of an integer 0 <= v < 2^23 (every such value is representable).
static inline ap_uint<32> u23_to_f32(ap_uint<23> v) {
#pragma HLS INLINE
  int msb = 0;
  for (int i = 0; i < 23; i++) {
#pragma HLS UNROLL
    if (v[i]) msb = i;
  }
  const ap_uint<23> mant = (ap_uint<23>)(v << (23 - msb));
  const ap_uint<8> ex = (ap_uint<8>)(127 + msb);
  return v == 0 ? (ap_uint<32>)0 : (ap_uint<32>)((ap_uint<9>)ex, mant);
}

// lane s (16 bits) of a 64-bit entry: a 4:1 mux of fixed slices (a variable range() costs a 64-bit shifter)
static inline ap_uint<16> lane16(ap_uint<64> x, ap_uint<2> s) {
#pragma HLS INLINE
  const ap_uint<16> a = x.range(15, 0), b = x.range(31, 16), c = x.range(47, 32), d = x.range(63, 48);
  return s[1] ? (s[0] ? d : c) : (s[0] ? b : a);
}

// ------------------------------------------------------------------ rank tables (per pass, from its own port)
// E: ranks of keys 4i..4i+3 in entry i (16 bits each), clamped: E < 0 or E >= 0x7FFF -> 0x7FFF (never visible).
// D: ranks of rows 4i..4i+3, clamped to [-1, 0x7FFE]. Visible iff (int16) E <= (int16) D.
static inline ap_uint<16> clamp_e(ap_uint<32> v) {
#pragma HLS INLINE
  const ap_int<32> e = (ap_int<32>)v;
  return (e < 0 || e >= 0x7FFF) ? (ap_uint<16>)0x7FFF : (ap_uint<16>)e;
}
static inline ap_uint<16> clamp_d(ap_uint<32> v) {
#pragma HLS INLINE
  const ap_int<32> d = (ap_int<32>)v;
  return d < 0 ? (ap_uint<16>)0xFFFF : d > 0x7FFE ? (ap_uint<16>)0x7FFE : (ap_uint<16>)d;
}
static void load_tables(const Word* mem, WAddr eoff, WAddr doff, int L, int N, ap_uint<64> etab[EW],
                        ap_uint<64> dtab[DW]) {
  const int ne = (L + 3) >> 2;
  const int nd = N < R_MAX ? (N + 3) >> 2 : DW;
ld_e:
  for (int i = 0; i < ne; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT max = 64
    const Word v = mem[eoff + i];
    ap_uint<64> t;
    for (int l = 0; l < 4; l++) {
#pragma HLS UNROLL
      t.range(16 * l + 15, 16 * l) = clamp_e(v.range(32 * l + 31, 32 * l));
    }
    etab[i] = t;
  }
ld_d:
  for (int i = 0; i < nd; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT max = 64
    const Word v = mem[doff + i];
    ap_uint<64> t;
    for (int l = 0; l < 4; l++) {
#pragma HLS UNROLL
      t.range(16 * l + 15, 16 * l) = clamp_d(v.range(32 * l + 31, 32 * l));
    }
    dtab[i] = t;
  }
}

// validity of the 4 lanes of data word w of row n in group g (keys kw * 4 + l, kw = engine ? 8 g + w : w)
static inline ap_uint<4> lanes_valid(const Params& p, ap_uint<9> kw, int n, bool rowv, const ap_uint<64>* etab,
                                     const ap_uint<64>* dtab) {
#pragma HLS INLINE
  const ap_uint<64> ev = etab[kw];
  const ap_uint<64> dv4 = dtab[(n >> 2) & (DW - 1)];
  const ap_int<16> dv = (ap_int<16>)lane16(dv4, (ap_uint<2>)(n & 3));
  ap_uint<4> val = 0;
  for (int l = 0; l < 4; l++) {
#pragma HLS UNROLL
    const bool vis = !p.mask || (ap_int<16>)ev.range(16 * l + 15, 16 * l) <= dv;
    val[l] = rowv && ((ap_uint<12>)(kw * 4 + l) < (ap_uint<12>)p.L) && vis;
  }
  return val;
}

// ------------------------------------------------------------------ pass 1: row max (hp0)
static void pass1(const Word* mem, hls::stream<MX>& mx, Params p) {
  ap_uint<64> etab[EW], dtab[DW];
#pragma HLS BIND_STORAGE variable = etab type = ram_s2p impl = lutram
#pragma HLS BIND_STORAGE variable = dtab type = ram_s2p impl = lutram
  ap_uint<33> mst[R_MAX];  // engine: (any, max key) of each row between its segments
#pragma HLS BIND_STORAGE variable = mst type = ram_s2p impl = bram
  const Geo geo = geometry(p);
  WAddr abase = p.a_off, ebase = p.e_off, dbase = p.d_off;
items1:
  for (int it = 0; it < p.B; it++) {
#pragma HLS LOOP_TRIPCOUNT max = 16
    if (p.mask && (it == 0 || p.e_stride != 0 || p.d_stride != 0)) load_tables(mem, ebase, dbase, p.L, p.N, etab, dtab);
    ap_uint<6> g = 0;
    int n = 0;
    ap_uint<24> w = 0;
    ap_int<32> km = KMIN;
    bool any = false;
  words1:
    for (int i = 0; i < geo.nw; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT max = 65536
#pragma HLS DEPENDENCE variable = mst inter false
      const Word v = mem[abase + i];
      if (w <= (ap_uint<24>)geo.wlast) {
        const ap_uint<9> kw = (p.engine ? (ap_uint<9>)(g * 8) : (ap_uint<9>)0) + (ap_uint<9>)w;
        const ap_uint<4> val = lanes_valid(p, kw, n, n < p.N, etab, dtab);
        ap_int<32> k[4];
        for (int l = 0; l < 4; l++) {
#pragma HLS UNROLL
          k[l] = val[l] ? fkey(v.range(32 * l + 31, 32 * l)) : (ap_int<32>)KMIN;
        }
        const ap_int<32> k01 = k[1] > k[0] ? k[1] : k[0], k23 = k[3] > k[2] ? k[3] : k[2];
        const ap_int<32> k4 = k23 > k01 ? k23 : k01;
        ap_int<32> kin = km;
        bool ain = any;
        if (w == 0) {
          kin = KMIN;
          ain = false;
#if SMX_ENGINE
          const ap_uint<33> s = mst[n & (R_MAX - 1)];
          if (g != 0) {
            kin = (ap_int<32>)s.range(31, 0);
            ain = s[32];
          }
#endif
        }
        const ap_int<32> kn = k4 > kin ? k4 : kin;
        const bool an = ain || val != 0;
        km = kn;
        any = an;
        if (w == (ap_uint<24>)geo.wlast) {
          if (g == geo.G - 1) {
            MX m;
            m.bits = (ap_uint<32>)fkey((ap_uint<32>)kn);  // the key's own bit pattern
            m.any = an;
            mx.write(m);
          } else {
#if SMX_ENGINE
            mst[n & (R_MAX - 1)] = ((ap_uint<1>)an, (ap_uint<32>)kn);
#endif
          }
        }
      }
      if (w == (ap_uint<24>)(geo.SW - 1)) {
        w = 0;
        if (n == geo.NR - 1) {
          n = 0;
          g++;
        } else {
          n++;
        }
      } else {
        w++;
      }
    }
    abase += p.a_stride;
    ebase += p.e_stride;
    dbase += p.d_stride;
  }
}

// ------------------------------------------------------------------ pass 2: exp, q, row sums (hp1)
static void pass2(const Word* mem, hls::stream<MX>& mx, hls::stream<EQ>& eq, hls::stream<DN>& sq, Params p) {
  ap_uint<64> etab[EW], dtab[DW];
#pragma HLS BIND_STORAGE variable = etab type = ram_s2p impl = lutram
#pragma HLS BIND_STORAGE variable = dtab type = ram_s2p impl = lutram
  ap_uint<84> rst[R_MAX];  // engine: (any, max on the Q.21 grid, partial sum) of each row between its segments
#pragma HLS BIND_STORAGE variable = rst type = ram_s2p impl = bram
  const Geo geo = geometry(p);
  const bool fp32 = SMX_FP32_OUT && !p.qout;
  ap_uint<17> mc;
  ap_int<10> ec;
  cparams(p.c_bits, mc, ec);
  WAddr abase = p.a_off, ebase = p.e_off, dbase = p.d_off;
items2:
  for (int it = 0; it < p.B; it++) {
#pragma HLS LOOP_TRIPCOUNT max = 16
    if (p.mask && (it == 0 || p.e_stride != 0 || p.d_stride != 0)) load_tables(mem, ebase, dbase, p.L, p.N, etab, dtab);
    ap_uint<6> g = 0;
    int n = 0;
    ap_uint<24> w = 0;
    ap_int<48> xm = 0;
    bool any = false;
    ap_uint<35> sum = 0;
  words2:
    for (int i = 0; i < geo.nw; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT max = 65536
#pragma HLS DEPENDENCE variable = rst inter false
      const Word v = mem[abase + i];
      if (w <= (ap_uint<24>)geo.wlast) {
        const bool rowv = n < p.N;
        const ap_uint<9> kw = (p.engine ? (ap_uint<9>)(g * 8) : (ap_uint<9>)0) + (ap_uint<9>)w;
        const ap_uint<4> val = lanes_valid(p, kw, n, rowv, etab, dtab);
        ap_int<48> xmi = xm;
        bool ai = any;
        ap_uint<35> si = sum;
        if (w == 0) {
          if (g == 0) {
            const MX m = mx.read();
            xmi = fix(m.bits, ec);
            ai = m.any;
            si = 0;
          } else {
#if SMX_ENGINE
            const ap_uint<84> s = rst[n & (R_MAX - 1)];
            ai = s[83];
            xmi = (ap_int<48>)s.range(82, 35);
            si = s.range(34, 0);
#endif
          }
        }
        ap_uint<26> u[4];
        for (int l = 0; l < 4; l++) {
#pragma HLS UNROLL
          const ap_uint<32> b = v.range(32 * l + 31, 32 * l);
          const ap_uint<47> mag = fixmag(b, ec);
          const ap_int<49> d = b[31] ? (ap_int<49>)((ap_int<49>)xmi + (ap_int<49>)mag)
                                     : (ap_int<49>)((ap_int<49>)xmi - (ap_int<49>)mag);
          u[l] = (!ai || d <= 0) ? (ap_uint<26>)0 : (d >= (1 << 26)) ? (ap_uint<26>)0x3FFFFFF : (ap_uint<26>)d;
        }
        ap_uint<23> e[4];
        exp_pair<0>(u[0], u[1], mc, e[0], e[1]);
        exp_pair<1>(u[2], u[3], mc, e[2], e[3]);
        EQ o = 0;
        ap_uint<26> s4 = 0;
        for (int l = 0; l < 4; l++) {
#pragma HLS UNROLL
          const bool inr = (ap_uint<12>)(kw * 4 + l) < (ap_uint<12>)p.L;
          const bool use = rowv && (ai ? (bool)val[l] : (inr && p.uniform));
          const ap_uint<23> el = use ? e[l] : (ap_uint<23>)0;
          const ap_uint<23> ql = (el + 128) >> 8;  // q in [0, 32767]
          const ap_uint<23> ol = fp32 ? el : ql;
          o.range(23 * l + 22, 23 * l) = ol;
          s4 += ol;
        }
        eq.write(o);
        const ap_uint<35> ns = si + s4;
        xm = xmi;
        any = ai;
        sum = ns;
        if (w == (ap_uint<24>)geo.wlast) {
          if (g == geo.G - 1) {
            int msb = 0;
            for (int bb = 0; bb < 35; bb++) {
#pragma HLS UNROLL
              if (ns[bb]) msb = bb;
            }
            DN dn;
            dn.d = ns << (34 - msb);
            dn.p = msb;
            dn.zero = ns == 0;
            sq.write(dn);
          } else {
#if SMX_ENGINE
            rst[n & (R_MAX - 1)] = ((ap_uint<1>)ai, (ap_uint<48>)xmi, ns);
#endif
          }
        }
      }
      if (w == (ap_uint<24>)(geo.SW - 1)) {
        w = 0;
        if (n == geo.NR - 1) {
          n = 0;
          g++;
        } else {
          n++;
        }
      } else {
        w++;
      }
    }
    abase += p.a_stride;
    ebase += p.e_stride;
    dbase += p.d_stride;
  }
}

// ------------------------------------------------------------------ RN(1 / S), FP32 bits (0 for S = 0)
// Restoring division of 2^34 by d = S << (34 - p): q = floor(2^(25 + p) / S) in (2^24, 2^25], 26 quotient bits,
// then round to nearest even with the remainder as sticky bit. Two units, one bit per cycle each, take the rows
// in turn (unit 0 loads at s = 0, unit 1 at s = 13 of a 26-cycle frame): one row every 13 cycles.
struct DivState {
  ap_uint<36> r;
  ap_uint<35> d;
  ap_uint<26> q;
  ap_uint<6> p;
  bool zero, act;
};
static inline ap_uint<32> div_finish(const DivState& u) {
#pragma HLS INLINE
  ap_uint<32> bits;
  if (u.zero) {
    bits = 0;
  } else if (u.q[25]) {  // S = 2^p
    bits = (ap_uint<32>)(127 - u.p) << 23;
  } else {
    const bool up = u.q[0] && (u.r != 0 || u.q[1]);
    const ap_uint<25> sig = (ap_uint<25>)(u.q >> 1) + (ap_uint<25>)up;
    const ap_uint<8> ex = (ap_uint<8>)(126 - u.p) + (ap_uint<8>)sig[24];
    bits = ((ap_uint<9>)ex, (ap_uint<23>)sig.range(22, 0));
  }
  return bits;
}
static inline void div_step(DivState& u) {
#pragma HLS INLINE
  const ap_int<37> t = (ap_int<37>)u.r - (ap_int<37>)u.d;
  const bool bit = t >= 0;
  const ap_uint<36> rr = bit ? (ap_uint<36>)t : u.r;
  u.q = (u.q << 1) | (ap_uint<26>)bit;
  u.r = rr << 1;
}
static void recip(hls::stream<DN>& sq, hls::stream<ap_uint<32> >& rq, Params p) {
  const int rows = p.rows_all;  // every row of every item (engine: padding rows too)
  const int total = (rows + 1) * NSTEP;
  DivState u0, u1;
  u0.act = false;
  u1.act = false;
  u0.r = u1.r = 0;
  u0.d = u1.d = 0;
  u0.q = u1.q = 0;
  u0.p = u1.p = 0;
  u0.zero = u1.zero = false;
  ap_uint<5> s = 0;
  int loaded = 0;
divide:
  for (int i = 0; i < total; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT max = 3342336
    const bool slot0 = s == 0, slot1 = s == 13;
    const bool ld = (slot0 || slot1) && loaded < rows;
    DN v;
    v.d = 0;
    v.p = 0;
    v.zero = true;
    if (ld) {
      v = sq.read();
      loaded++;
    }
    if (slot0) {
      u0.act = ld;
      u0.d = v.d;
      u0.p = v.p;
      u0.zero = v.zero;
      u0.r = (ap_uint<36>)1 << 34;
      u0.q = 0;
    }
    if (slot1) {
      u1.act = ld;
      u1.d = v.d;
      u1.p = v.p;
      u1.zero = v.zero;
      u1.r = (ap_uint<36>)1 << 34;
      u1.q = 0;
    }
    div_step(u0);
    div_step(u1);
    const bool f0 = s == 25 && u0.act, f1 = s == 12 && u1.act;
    if (f0 || f1) rq.write(div_finish(f0 ? u0 : u1));
    s = s == 25 ? (ap_uint<5>)0 : (ap_uint<5>)(s + 1);
  }
}

// ------------------------------------------------------------------ shaper: output words in DDR order
static void shaper(hls::stream<EQ>& eq, hls::stream<Word>& bc, Params p) {
  const Geo geo = geometry(p);
  if (!p.engine) {
    const bool fp32 = SMX_FP32_OUT && !p.qout;
    const int L4 = geo.wlast + 1;
    const int total = p.ent_rows;
    ap_uint<10> w = 0;
    ap_uint<64> pend = 0;
  rows_s:
    for (int i = 0; i < total; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT max = 262144
      const EQ e = eq.read();
      Word o = 0;
      ap_uint<64> q4 = 0;
      for (int l = 0; l < 4; l++) {
#pragma HLS UNROLL
        o.range(32 * l + 22, 32 * l) = e.range(23 * l + 22, 23 * l);
        q4.range(16 * l + 15, 16 * l) = e.range(23 * l + 15, 23 * l);
      }
      const bool last = w == (ap_uint<10>)(L4 - 1);
      if (fp32) {
        bc.write(o);
      } else if (w[0]) {
        bc.write((q4, pend));
      } else if (last) {
        bc.write((ap_uint<64>(0), q4));
      } else {
        pend = q4;
      }
      w = last ? (ap_uint<10>)0 : (ap_uint<10>)(w + 1);
    }
  }
#if SMX_ENGINE
  else {
    // tile (g, b): rows 32 b .. 32 b + 31, keys 32 g .. 32 g + Kg - 1, 256 entries in (row, word) order in;
    // out Kg x 4 words, word qd of key k = rows 8 qd .. 8 qd + 7 (one per bank) of key k.
    // Banks by row % 8, entry (pp, row / 8 % 4, word) holds the 4 keys of that word: 8 x 64 x 64 bits.
    ap_uint<64> xb[8][64];
#pragma HLS ARRAY_PARTITION variable = xb complete dim = 1
#pragma HLS BIND_STORAGE variable = xb type = ram_s2p impl = lutram
    const int tiles = p.tiles;
  items_s:
    for (int it = 0; it < p.B; it++) {
#pragma HLS LOOP_TRIPCOUNT max = 16
      const int total = (tiles + 1) * 256;
      int t = 0;
      ap_uint<8> j = 0;
      ap_uint<1> pp = 0;
      ap_uint<6> gd = 0;  // group of the tile being drained
      ap_uint<4> bd = 0;  // its row block
      int nout = 0;       // words of the tile being drained
    tiles_s:
      for (int i = 0; i < total; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT max = 41216
#pragma HLS DEPENDENCE variable = xb inter false
        if (t < tiles) {
          const EQ e = eq.read();
          ap_uint<64> q4;
          for (int l = 0; l < 4; l++) {
#pragma HLS UNROLL
            q4.range(16 * l + 15, 16 * l) = e.range(23 * l + 15, 23 * l);
          }
          const ap_uint<5> nl = j >> 3;  // row in the block
          const ap_uint<6> a = ((ap_uint<6>)pp << 5) | ((ap_uint<6>)(nl >> 3) << 3) | (ap_uint<6>)(j & 7);
          xb[nl & 7][a] = q4;
        }
        const ap_uint<7> kq = j >> 1;  // output word of the drained tile: key kq / 4, quad kq % 4
        if (t > 0 && j[0] && (int)kq < nout) {
          const ap_uint<5> kl = kq >> 2;
          const ap_uint<2> qd = kq & 3;
          const ap_uint<6> a = ((ap_uint<6>)(~pp) << 5) | ((ap_uint<6>)qd << 3) | (ap_uint<6>)(kl >> 2);
          Word o;
          for (int b8 = 0; b8 < 8; b8++) {
#pragma HLS UNROLL
            const ap_uint<64> x = xb[b8][a];
            o.range(16 * b8 + 15, 16 * b8) = lane16(x, (ap_uint<2>)(kl & 3));
          }
          bc.write(o);
        }
        if (j == 255) {
          // the tile just filled becomes the drained one: its group / block and key count
          const int kg = p.L - 32 * (int)gd;
          nout = 4 * (kg < 32 ? kg : 32);
          if ((int)bd == (geo.NR >> 5) - 1) {
            bd = 0;
            gd++;
          } else {
            bd++;
          }
          pp = ~pp;
          t++;
        }
        j++;
      }
    }
  }
#endif
}

// ------------------------------------------------------------------ writer (io)
static void writer(hls::stream<Word>& bc, hls::stream<ap_uint<32> >& rq, Word* io, Params p) {
  const Geo geo = geometry(p);
  const bool fp32 = SMX_FP32_OUT && !p.qout;
  const int L4 = (p.L + 3) >> 2;
  const ap_uint<10> wd = fp32 ? L4 : (p.L + 7) >> 3;  // data words per output row (rows layout)
  WAddr cbase = p.c_off, fbase = p.f_off;
items_w:
  for (int it = 0; it < p.B; it++) {
#pragma HLS LOOP_TRIPCOUNT max = 16
    if (!p.engine) {
      WAddr dbase = cbase, sbase = fbase;
    blocks_w:
      for (int r0 = 0; r0 < p.N; r0 += BRW) {
#pragma HLS LOOP_TRIPCOUNT max = 24
        const int rows = p.N - r0 < BRW ? p.N - r0 : BRW;
        const int total = rows * p.c_pitch;
        ap_uint<24> w = 0;
        ap_uint<32> rr = 0;
      rows_w:
        for (int i = 0; i < total; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT max = 65536
          if (fp32 && w == 0) rr = rq.read();
          Word o = 0;
          if (w < wd) {
            const Word v = bc.read();
            if (fp32) {
#if SMX_FP32_OUT
              const float sc = u2f((uint32_t)rr);
              for (int l = 0; l < 4; l++) {
#pragma HLS UNROLL
                const float f = u2f((uint32_t)u23_to_f32((ap_uint<23>)v.range(32 * l + 22, 32 * l)));
                o.range(32 * l + 31, 32 * l) = f2u(f * sc);
              }
#endif
            } else {
              o = v;
            }
          }
          io[dbase + i] = o;
          w = (w == (ap_uint<24>)(p.c_pitch - 1)) ? (ap_uint<24>)0 : (ap_uint<24>)(w + 1);
        }
        dbase += total;
        if (!fp32) {
          const int nsw = (rows + 3) >> 2;
        scales_w:
          for (int k = 0; k < nsw; k++) {
#pragma HLS PIPELINE II = 4
#pragma HLS LOOP_TRIPCOUNT max = 64
            Word o = 0;
            for (int l = 0; l < 4; l++) {
              if (4 * k + l < rows) o.range(32 * l + 31, 32 * l) = rq.read();
            }
            io[sbase + k] = o;
          }
          sbase += BRW / 4;
        }
      }
    }
#if SMX_ENGINE
    else {
      const int nb = geo.NR >> 5;
    groups_w:
      for (int g = 0; g < geo.G; g++) {
#pragma HLS LOOP_TRIPCOUNT max = 8
        const int kg = p.L - 32 * g < 32 ? p.L - 32 * g : 32;
        const int n = 4 * kg;
        WAddr tbase = cbase + (WAddr)(128 * g);
      blocks_e:
        for (int b = 0; b < nb; b++) {
#pragma HLS LOOP_TRIPCOUNT max = 8
        tile_w:
          for (int i = 0; i < n; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT max = 128
            io[tbase + i] = bc.read();
          }
          tbase += (WAddr)(4 * p.L);
        }
      }
    stok_w:
      for (int k = 0; k < (geo.NR >> 2); k++) {
#pragma HLS PIPELINE II = 4
#pragma HLS LOOP_TRIPCOUNT max = 64
        Word o;
        for (int l = 0; l < 4; l++) o.range(32 * l + 31, 32 * l) = rq.read();
        io[fbase + k] = o;
      }
    }
#endif
    cbase += p.c_stride;
    fbase += p.f_stride;
  }
}

// ------------------------------------------------------------------ the kernel
// hp0, hp1: read-only views of DDR (scores, rank tables); io: the output view. Word offsets in p.
static void softmax_core(const Word* hp0, const Word* hp1, Word* io, Params p) {
#pragma HLS DATAFLOW
  hls::stream<MX> mx("smx_mx");
  hls::stream<EQ> eq("smx_eq");
  hls::stream<DN> sq("smx_sq");
  hls::stream<ap_uint<32> > rq("smx_rq");
  hls::stream<Word> bc("smx_bc");
#pragma HLS STREAM variable = mx depth = 32
#pragma HLS BIND_STORAGE variable = mx type = fifo impl = srl
#pragma HLS STREAM variable = eq depth = 16
#pragma HLS BIND_STORAGE variable = eq type = fifo impl = srl
#pragma HLS STREAM variable = sq depth = 8
#pragma HLS STREAM variable = rq depth = 288
#pragma HLS BIND_STORAGE variable = rq type = fifo impl = bram
#pragma HLS STREAM variable = bc depth = BCFIFO
#if !SMX_FP32_OUT
#pragma HLS BIND_STORAGE variable = bc type = fifo impl = srl
#endif
  pass1(hp0, mx, p);
  pass2(hp1, mx, eq, sq, p);
  recip(sq, rq, p);
  shaper(eq, bc, p);
  writer(bc, rq, io, p);
}

// One SOFTMAX record at word offset rec_off of hp0's view: decode, check, run. Returns a Status.
static int softmax_record(const Word* hp0, const Word* hp1, Word* io, WAddr rec_off) {
  Word rec[8];
  for (int i = 0; i < 8; i++) {
#pragma HLS PIPELINE II = 1
    rec[i] = hp0[rec_off + i];
  }
  Params p;
  const int st = decode(rec, p);
  if (st == ST_OK) softmax_core(hp0, hp1, io, p);
  return st;
}

}  // namespace smx
