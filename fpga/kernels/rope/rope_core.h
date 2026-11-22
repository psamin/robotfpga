// RoPE front end of the row unit (fpga/kernels/CONTRACT.md 4.1, opcode ROPE 0x60): the rope-specific part of the
// mode, i.e. what the shared row unit gains for ROPE. It rotates FP32 rows of 64-value segments (head_dim 64,
// natural lerobot order: x1 = values 0..31, x2 = 32..63 of a segment) and emits the rotated FP32 values, 4 per clock,
// in natural order, plus each segment's absmax (as magnitude bits) for the shared contract quantizer.
//
//   rot_stage   xs (16 FP32 words per segment) + cs (its 16-word table row) -> rs (16 rotated groups)
//   emit_stage  rs -> fw (16 FP32 words per segment, natural order) + am (segment absmax, when quantizing)
//
// The FP32 values are, bit for bit, rope_emu.to_f32 (CONTRACT.md 2.2); everything after them (int16 + scales by
// the contract quantizer, the KW weight tiling, FP32 rows) is rope_mode.h. tb_rope.cpp checks every bit.
//
// Arithmetic (rope_emu.py has the same in int64 torch). Group k = 0..15 of a segment = the 4 values
// (x1[2k], x1[2k+1], x2[2k], x2[2k+1]), i.e. 2 rotation pairs:
//   e_j = FP32 biased exponents, E = max_j e_j; A_j = sign * ((1.frac << 2) >> (E - e_j)) (27-bit signed,
//   e_j = 0 (zero / denormal) -> 0); table NC = round(-cos * 2^17), NS = round(-sin * 2^17) (18-bit);
//   O0 = A2 NS0 - A0 NC0 (= y1[2k]), O1 = A3 NS1 - A1 NC1, O2 = -(A2 NC0 + A0 NS0) (= y2[2k]), O3 likewise:
//   27 x 18 products (8 DSP48E2 per group), |O| < 2^44, value O * 2^(E - 169).
//   Per group Lg = bitlen(max |O|), N_j = |O_j| >> (Lg - 26) (26 bits), Mg = E + Lg (0 for E = 0).
//   FP32: each sign * N * 2^(Mg - 195) rounded to 24 significant bits (RNE, unbounded exponent), then a result
//   below 2^-126 becomes +0 and one at or above 2^128 becomes inf (sign kept). Zero and denormal inputs count as 0;
//   NaN / Inf inputs are undefined (they give some finite or infinite value; control flow never depends on data).
// Data movement: one 128-bit input word per cycle. A segment arrives as 16 words; it is held in a ping-pong
// LUT-RAM (x1 half, x2 half) and rotated during the next segment's arrival, one group per cycle (group k needs
// x1 word k/2 and x2 word 8 + k/2); the segment's table row (16 words, same halves) is loaded with it. The
// rotated groups go to a second ping-pong (even / odd groups) from which the natural output order is read:
// output word c = (lanes of groups 2j, 2j+1), j = c % 8, lanes 0,1 (y1) for c < 8, lanes 2,3 (y2) for c >= 8.
// II = 1 everywhere.
//
// Counters are sized from the contract limits (validate_hw): segments per command B * NP * H < 2^38.
#pragma once
#include <stdint.h>

#include <ap_int.h>
#include <hls_stream.h>

namespace rope {

typedef ap_uint<128> Word;  // one io word: 4 x 32-bit lanes, lane j in bits [32j+31:32j]
typedef ap_uint<72> Tab;    // one table word cut to its 4 18-bit lanes
typedef ap_uint<40> rcount_t;  // segments per command: B * NP * H <= 65535 * 65535 * 64 < 2^38
typedef ap_uint<31> abits_t;   // |x| as FP32 magnitude bits (qz::abits_t)
constexpr int WORDS = 16;   // FP32 words per segment (64 values)
constexpr int HALF = 8;     // words per half segment (x1 | x2)
constexpr int EXP_OFF = 169;  // value = O * 2^(E - EXP_OFF)

typedef ap_int<27> A27;
typedef ap_int<18> C18;
typedef ap_int<45> O45;

struct Rot {  // one rotated group
  O45 o[4];
  ap_uint<8> e;
};

static inline ap_uint<32> lane(const Word& w, int j) {
#pragma HLS INLINE
  return w.range(32 * j + 31, 32 * j);
}

// ------------------------------------------------------------------------------------- rotation
// xg: the group's 4 FP32 (x1[2k], x1[2k+1], x2[2k], x2[2k+1]); nc / ns: -cos, -sin of its 2 pairs.
static inline Rot rotate_group(const ap_uint<32> xg[4], C18 nc0, C18 nc1, C18 ns0, C18 ns1) {
#pragma HLS INLINE
  ap_uint<8> e[4];
  ap_uint<1> sg[4];
  ap_uint<26> mag[4];
  for (int j = 0; j < 4; j++) {
#pragma HLS UNROLL
    e[j] = xg[j].range(30, 23);
    sg[j] = xg[j][31];
    mag[j] = ((ap_uint<26>)1 << 25) | ((ap_uint<26>)xg[j].range(22, 0) << 2);
  }
  const ap_uint<8> m01 = e[0] > e[1] ? e[0] : e[1];
  const ap_uint<8> m23 = e[2] > e[3] ? e[2] : e[3];
  const ap_uint<8> E = m01 > m23 ? m01 : m23;
  A27 a[4];
  for (int j = 0; j < 4; j++) {
#pragma HLS UNROLL
    const ap_uint<8> d = E - e[j];
    const ap_uint<26> s = (e[j] == 0 || d > 26) ? (ap_uint<26>)0 : (ap_uint<26>)(mag[j] >> d);
    a[j] = sg[j] ? (A27)(-(A27)s) : (A27)s;
  }
  Rot r;
  r.o[0] = (O45)(a[2] * ns0 - a[0] * nc0);     // x1 cos - x2 sin
  r.o[1] = (O45)(a[3] * ns1 - a[1] * nc1);
  r.o[2] = (O45)(-(a[2] * nc0 + a[0] * ns0));  // x2 cos + x1 sin
  r.o[3] = (O45)(-(a[3] * nc1 + a[1] * ns1));
  r.e = E;
  return r;
}

// Reachable ranges (keep the shifters small; rope_emu asserts them on every call it models): a group with
// E > 0 has a lane with |A| >= 2^25 and O0^2 + O2^2 = (A0^2 + A2^2)(NC0^2 + NS0^2) exactly with
// NC^2 + NS^2 > 2^34 (1 - 2^-16), so its max |O| >= 2^41.5: Lg = bitlen(max_j |O_j|) is in [42, 45] for
// every group with E > 0, and a group with E = 0 is all zero. Each group is therefore normalized once,
// with a 4-way shift: N_j = |O_j| >> (Lg - 26) (26 bits, the group max in [2^25, 2^26)), and its magnitude
// exponent Mg = E + Lg (0 for E = 0). The FP32 value of lane j is sign_j * N_j * 2^(Mg - 195).

// one stored group: 4 x {sign, N (26)} in bits [27j+26 : 27j], Mg in bits [117:108]
typedef ap_uint<118> Ent;

// FP32 element: (-1)^sg * n * 2^(mg - 195), rounded to 24 bits (RNE); zero and results below 2^-126 -> +0;
// results at or above 2^128 (after rounding) -> inf with the sign. rope_emu.to_f32 is the same.
static inline ap_uint<32> to_f32(ap_uint<1> sg, ap_uint<26> n, ap_uint<10> mg) {
#pragma HLS INLINE
  const ap_uint<5> lz = n.countLeadingZeros();  // 26 for n == 0
  const ap_uint<26> nrm = n << lz;               // leading one at bit 25
  const ap_uint<24> man = nrm.range(25, 2);
  const ap_uint<1> g = nrm[1];
  const bool st = nrm[0];
  const ap_uint<25> m25 = (ap_uint<25>)man + ((g && (st || man[0])) ? 1 : 0);
  // biased exponent: (25 - lz) + (mg - 195) + 127 + carry
  const ap_int<12> field = (ap_int<12>)mg - (ap_int<12>)(43 + lz) + (ap_int<12>)m25[24];
  ap_uint<32> b = 0;
  if (lz < 26 && field > 0) {
    b.range(22, 0) = (field > 254 || m25[24]) ? (ap_uint<23>)0 : (ap_uint<23>)m25.range(22, 0);
    b.range(30, 23) = field > 254 ? (ap_int<12>)255 : field;  // overflow -> inf, as torch's FP32 would
    b[31] = sg;
  }
  return b;
}

// ------------------------------------------------------------------------------------- stream processes
// rows segments, in the order the reader sends them. xs: 16 words per segment; cs: the segment's table row, 16
// words (int32 slots: NC[0..31] in words 0..7, NS[0..31] in words 8..15; only bits [17:0] are used), sent with
// every segment (the table read costs hp bandwidth only, and one table slot per x slot keeps the control trivial);
// rs: 16 groups per segment.
static void rot_stage(hls::stream<Word>& xs, hls::stream<Word>& cs, hls::stream<Rot>& rs, rcount_t rows) {
  Word x1b[2 * HALF], x2b[2 * HALF];
  Tab c1b[2 * HALF], c2b[2 * HALF];
#pragma HLS BIND_STORAGE variable = x1b type = ram_s2p impl = lutram
#pragma HLS BIND_STORAGE variable = x2b type = ram_s2p impl = lutram
#pragma HLS BIND_STORAGE variable = c1b type = ram_s2p impl = lutram
#pragma HLS BIND_STORAGE variable = c2b type = ram_s2p impl = lutram
#pragma HLS DEPENDENCE variable = x1b inter false
#pragma HLS DEPENDENCE variable = x2b inter false
#pragma HLS DEPENDENCE variable = c1b inter false
#pragma HLS DEPENDENCE variable = c2b inter false
  ap_uint<4> w = 0;
  ap_uint<1> sl = 0;  // slot being loaded (the other one is rotated)
  rcount_t r = 0;     // segment being loaded; r - 1 is rotated
  bool more = rows != 0;
rot:
  for (;;) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 32 max = 20000
    const ap_uint<3> wl = w.range(2, 0);
    if (more) {  // load word w of segment r and of its table row
      const Word x = xs.read();
      const Word c = cs.read();
      Tab t;
      for (int j = 0; j < 4; j++) {
#pragma HLS UNROLL
        t.range(18 * j + 17, 18 * j) = c.range(32 * j + 17, 32 * j);
      }
      if (w[3]) {
        x2b[sl * HALF + wl] = x;
        c2b[sl * HALF + wl] = t;
      } else {
        x1b[sl * HALF + wl] = x;
        c1b[sl * HALF + wl] = t;
      }
    }
    if (r != 0) {  // rotate group k = w of segment r - 1
      const ap_uint<3> j = w.range(3, 1);
      const ap_uint<4> a = (ap_uint<4>)((sl ^ 1) * HALF) + j;
      const Word X1 = x1b[a], X2 = x2b[a];
      const Tab C1 = c1b[a], C2 = c2b[a];
      const bool hi = w[0];
      ap_uint<32> xg[4];
      xg[0] = hi ? lane(X1, 2) : lane(X1, 0);
      xg[1] = hi ? lane(X1, 3) : lane(X1, 1);
      xg[2] = hi ? lane(X2, 2) : lane(X2, 0);
      xg[3] = hi ? lane(X2, 3) : lane(X2, 1);
      const C18 nc0 = hi ? C1.range(53, 36) : C1.range(17, 0), nc1 = hi ? C1.range(71, 54) : C1.range(35, 18);
      const C18 ns0 = hi ? C2.range(53, 36) : C2.range(17, 0), ns1 = hi ? C2.range(71, 54) : C2.range(35, 18);
      rs.write(rotate_group(xg, nc0, nc1, ns0, ns1));
    }
    if (w == WORDS - 1) {
      if (!more) break;  // the last segment has been rotated
      sl = ~sl;
      r++;
      more = r != rows;
    }
    w++;
  }
}

// Collects the 16 groups of a segment (normalized, as Ent) while emitting the previous segment in natural order:
// fw gets 16 FP32 words per segment; with quant, am gets each segment's absmax (an integer max over the 31
// magnitude bits of its 64 FP32 outputs, as qz::word_amax).
static void emit_stage(hls::stream<Rot>& rs, hls::stream<Word>& fw, hls::stream<abits_t>& am, rcount_t rows,
                       bool quant) {
  Ent ev[2 * HALF], od[2 * HALF];  // groups 2j / 2j + 1 of the segment in slot s at s * 8 + j
#pragma HLS BIND_STORAGE variable = ev type = ram_s2p impl = lutram
#pragma HLS BIND_STORAGE variable = od type = ram_s2p impl = lutram
#pragma HLS DEPENDENCE variable = ev inter false
#pragma HLS DEPENDENCE variable = od inter false
  ap_uint<4> w = 0;
  ap_uint<1> sl = 0;
  abits_t amx = 0;
  rcount_t r = 0;  // segment being collected; r - 1 is emitted
  bool more = rows != 0;
emit:
  for (;;) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 32 max = 20000
    if (more) {  // collect group w of segment r
      const Rot g = rs.read();
      ap_uint<1> sg[4];
      ap_uint<45> m[4];
      for (int j = 0; j < 4; j++) {
#pragma HLS UNROLL
        sg[j] = g.o[j][44];
        const ap_uint<45> ob = (ap_uint<45>)g.o[j];
        m[j] = (ob ^ (sg[j] ? ~(ap_uint<45>)0 : (ap_uint<45>)0)) + sg[j];  // |O|: one adder
      }
      const ap_uint<3> top = (m[0] | m[1] | m[2] | m[3]).range(44, 42);
      const ap_uint<2> lx = top[2] ? 3 : top[1] ? 2 : top[0] ? 1 : 0;  // Lg - 42
      const ap_uint<10> Mg = (g.e == 0) ? (ap_uint<10>)0 : (ap_uint<10>)(g.e + 42 + lx);
      Ent ent;
      for (int j = 0; j < 4; j++) {
#pragma HLS UNROLL
        ap_uint<26> nj;
        switch ((int)lx) {
          case 0: nj = m[j].range(41, 16); break;
          case 1: nj = m[j].range(42, 17); break;
          case 2: nj = m[j].range(43, 18); break;
          default: nj = m[j].range(44, 19); break;
        }
        ent.range(27 * j + 26, 27 * j) = ((ap_uint<27>)sg[j] << 26) | (ap_uint<27>)nj;
      }
      ent.range(117, 108) = Mg;
      if (w[0]) od[sl * HALF + w.range(3, 1)] = ent;
      else ev[sl * HALF + w.range(3, 1)] = ent;
    }
    if (r != 0) {  // emit output word w of segment r - 1
      const ap_uint<3> j = w.range(2, 0);
      const Ent A = ev[(sl ^ 1) * HALF + j], B = od[(sl ^ 1) * HALF + j];
      const bool y2 = w[3];  // lanes 2, 3 of the groups (y2) for output words 8..15
      ap_uint<27> v[4];
      v[0] = y2 ? A.range(80, 54) : A.range(26, 0);
      v[1] = y2 ? A.range(107, 81) : A.range(53, 27);
      v[2] = y2 ? B.range(80, 54) : B.range(26, 0);
      v[3] = y2 ? B.range(107, 81) : B.range(53, 27);
      const ap_uint<10> mA = A.range(117, 108), mB = B.range(117, 108);
      Word o;
      abits_t wm = 0;
      for (int l = 0; l < 4; l++) {
#pragma HLS UNROLL
        const ap_uint<32> f = to_f32(v[l][26], v[l].range(25, 0), l < 2 ? mA : mB);
        o.range(32 * l + 31, 32 * l) = f;
        const abits_t fa = f.range(30, 0);
        wm = fa > wm ? fa : wm;
      }
      fw.write(o);
      const abits_t prev = w == 0 ? (abits_t)0 : amx;
      amx = wm > prev ? wm : prev;
      if (quant && w == WORDS - 1) am.write(amx);
    }
    if (w == WORDS - 1) {
      if (!more) break;
      sl = ~sl;
      r++;
      more = r != rows;
    }
    w++;
  }
}

}  // namespace rope
