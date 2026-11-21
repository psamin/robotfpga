// RoPE kernel for SmolVLA on the KR260 (fpga/kernels/rope, README.md): rotates the Q or K rows of the LM
// prefix / action expert (head_dim 64, natural lerobot order: x1 = values 0..31, x2 = 32..63) and emits
//   MODE_F32   FP32 rows (row pitch programmable, in place allowed): the K that the KV cache / the K-blob
//              quantizer / the expert cross-attention k_proj read
//   MODE_INT16 int16 rows (8 words per (token, head), contiguous token-major) + one FP32 power-of-two
//              scale per (token, head), written head-major [h][ceil(T/4) words]: Q for the int16 Q K^T call
// Header only: the stream processes below become a mode of float_top; rope_top.cpp wraps them with DDR
// movers for csynth. Bit-exact model: rope_emu.py; tb_rope.cpp checks every output bit against it.
//
// Arithmetic (rope_emu.py has the same in int64 torch). Group k = 0..15 of a row = the 4 values
// (x1[2k], x1[2k+1], x2[2k], x2[2k+1]), i.e. 2 rotation pairs:
//   e_j = FP32 biased exponents, E = max_j e_j; A_j = sign * ((1.frac << 2) >> (E - e_j)) (27-bit signed,
//   e_j = 0 (zero / denormal) -> 0); table NC = round(-cos * 2^17), NS = round(-sin * 2^17) (18-bit);
//   O0 = A2 NS0 - A0 NC0 (= y1[2k]), O1 = A3 NS1 - A1 NC1, O2 = -(A2 NC0 + A0 NS0) (= y2[2k]), O3 likewise:
//   27 x 18 products (8 DSP48E2 per group), |O| < 2^44, value O * 2^(E - 169).
//   Per group Lg = bitlen(max |O|), N_j = |O_j| >> (Lg - 26) (26 bits), Mg = E + Lg (0 for E = 0).
//   MODE_F32: each sign * N * 2^(Mg - 195) rounded to FP32 (RNE), results below 2^-126 -> +0.
//   MODE_INT16: per row (64 values) Mtop = max over its groups of Mg;
//   q = rnd(|O| >> (Mtop - E - 15)) (half away from zero, clamp 32767, sign applied), s = 2^(Mtop - 184);
//   s below 2^-126 (incl. all-zero rows): q = 0, s = 0.
// Data movement: one 128-bit input word per cycle. A row arrives as 16 words; it is held in a ping-pong
// LUT-RAM (x1 half, x2 half) and rotated during the next row's arrival, one group per cycle (group k needs
// x1 word k/2 and x2 word 8 + k/2); the table row of a token (16 words, same halves) is loaded once and
// reused for the token's H heads. The rotated groups go to a second ping-pong (even / odd groups) from
// which the natural output order is read: output word c = (lanes of groups 2j, 2j+1), j = c % 8, lanes
// 0,1 (y1) for c < 8, lanes 2,3 (y2) for c >= 8. Every process runs at II = 1.
#pragma once
#include <stdint.h>

#include <ap_int.h>
#include <hls_stream.h>

namespace rope {

typedef ap_uint<128> Word;  // one io word: 4 x 32-bit lanes, lane j in bits [32j+31:32j]
typedef ap_uint<72> Tab;    // one table word cut to its 4 18-bit lanes
constexpr int WORDS = 16;   // FP32 words per (token, head) row
constexpr int HALF = 8;     // words per half row (x1 | x2)
constexpr int QWORDS = 8;   // int16 words per row
constexpr int MODE_INT16 = 0;
constexpr int MODE_F32 = 1;
constexpr int EXP_OFF = 169;    // value = O * 2^(E - EXP_OFF)
constexpr int SBANK = 512;      // MODE_INT16 limit: ceil(T / 4) * H <= SBANK (scale exponent buffer)

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
// exponent Mg = E + Lg (0 for E = 0). Both output modes are computed from (sign_j, N_j, Mg):
//   MODE_F32  : sign * N * 2^(Mg - 195) rounded to FP32 (RNE), results below 2^-126 -> +0
//   MODE_INT16: q = rnd(N >> (Mtop - Mg + 11)) (half away from zero, clamp 32767), s = 2^(Mtop - 184);
//               this equals rnd(|O| >> (Mtop - E - 15)) exactly, as the dropped bits lie below the round bit.

// one stored group: 4 x {sign, N (26)} in bits [27j+26 : 27j], Mg in bits [117:108]
typedef ap_uint<118> Ent;

// MODE_F32 element: (-1)^sg * n * 2^(mg - 195) to FP32 bits, RNE; zero and results below 2^-126 -> +0.
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

// MODE_INT16 element: a = clamp(Mtop - Mg + 10, 10, 26), rnd(n >> (a + 1)) = ((n >> a) + 1) >> 1
static inline ap_uint<16> q16(ap_uint<1> sg, ap_uint<26> n, ap_uint<5> a, bool dead) {
#pragma HLS INLINE
  const ap_uint<16> hi = n >> 10;
  const ap_uint<16> sh = hi >> (a - 10);  // a - 10 in [0, 16]; < 2^16 since Mtop >= Mg
  const ap_uint<17> r1 = ((ap_uint<17>)sh + 1) >> 1;
  const ap_uint<15> qm = r1 > 32767 ? (ap_uint<17>)32767 : r1;
  ap_int<16> q = sg ? (ap_int<16>)(-(ap_int<16>)qm) : (ap_int<16>)qm;
  if (dead) q = 0;
  return (ap_uint<16>)q;
}

static inline ap_uint<5> qshift(ap_uint<10> mtop, ap_uint<10> mg) {
#pragma HLS INLINE
  const ap_int<12> a = (ap_int<12>)mtop - (ap_int<12>)mg + 10;
  return a > 26 ? (ap_uint<5>)26 : a < 10 ? (ap_uint<5>)10 : (ap_uint<5>)a;
}

// ------------------------------------------------------------------------------------- stream processes
// rows = T * H, token-major (row r = token r / H, head r % H). xs: 16 words per row; cs: 16 words per token
// (int32 slots: NC[0..31] in words 0..7, NS[0..31] in words 8..15); rs: 16 groups per row.
static void rot_stage(hls::stream<Word>& xs, hls::stream<Word>& cs, hls::stream<Rot>& rs, int rows, int H) {
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
  ap_uint<1> xsl = 0;          // x slot being loaded (the other one is rotated)
  ap_uint<1> tsl = 0, tpr = 0;  // table slot of the row being loaded / rotated
  ap_uint<6> h = 0;
  int r = 0;
rot:
  for (int n = 0; n < (rows + 1) * WORDS; n++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT max = 20000
    const ap_uint<3> wl = w.range(2, 0);
    if (r < rows) {  // load word w of row r (and of its token's table row when h == 0)
      const Word x = xs.read();
      if (w[3]) x2b[xsl * HALF + wl] = x;
      else x1b[xsl * HALF + wl] = x;
      if (h == 0) {
        const Word c = cs.read();
        Tab t;
        for (int j = 0; j < 4; j++) {
#pragma HLS UNROLL
          t.range(18 * j + 17, 18 * j) = c.range(32 * j + 17, 32 * j);
        }
        if (w[3]) c2b[tsl * HALF + wl] = t;
        else c1b[tsl * HALF + wl] = t;
      }
    }
    if (r >= 1) {  // rotate group k = w of row r - 1
      const ap_uint<3> j = w.range(3, 1);
      const Word X1 = x1b[(xsl ^ 1) * HALF + j], X2 = x2b[(xsl ^ 1) * HALF + j];
      const Tab C1 = c1b[tpr * HALF + j], C2 = c2b[tpr * HALF + j];
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
      xsl = ~xsl;
      tpr = tsl;
      if (h == H - 1) {
        h = 0;
        tsl = ~tsl;
      } else {
        h++;
      }
      r++;
    }
    w++;
  }
}


// Collects the 16 groups of a row (normalized, as Ent) while emitting the previous row in natural order:
// qs gets 16 FP32 words per row (MODE_F32) or 8 int16 words per row (MODE_INT16). MODE_INT16 then sends
// the scales: H x ceil(T/4) words to ss, head-major, 4 consecutive tokens per word, lanes past T zero.
// nsw = H * tq scale words, tq = ceil(T / 4).
static void out_stage(hls::stream<Rot>& rs, hls::stream<Word>& qs, hls::stream<Word>& ss, int rows, int H, int tq,
                      int nsw, int mode) {
  Ent ev[2 * HALF], od[2 * HALF];  // groups 2j / 2j + 1 of the row in slot s at s * 8 + j
#pragma HLS BIND_STORAGE variable = ev type = ram_s2p impl = lutram
#pragma HLS BIND_STORAGE variable = od type = ram_s2p impl = lutram
#pragma HLS DEPENDENCE variable = ev inter false
#pragma HLS DEPENDENCE variable = od inter false
  // FP32 exponent fields of the row scales: byte t % 4 of sx[(t / 4) * H + h] (one BRAM18). Updated by
  // read-modify-write; two updates of one entry are >= 16 cycles apart (one row), beyond the pipeline depth.
  ap_uint<32> sx[SBANK];
#pragma HLS BIND_STORAGE variable = sx type = ram_s2p impl = bram
#pragma HLS DEPENDENCE variable = sx inter false
  const bool f32 = (mode == MODE_F32);
  ap_uint<10> cM = 0, eM = 0;  // Mtop of the row being collected / emitted (0 <=> all zero)
  ap_uint<64> qlo = 0;
  ap_uint<4> w = 0;
  ap_uint<1> sl = 0;
  ap_uint<6> h = 0;   // head and token % 4 of the row being emitted
  ap_uint<2> tb = 0;
  ap_uint<10> sidx = 0, sbase = 0;  // (token / 4) * H + h, (token / 4) * H
  int r = 0;
rows_loop:
  for (int n = 0; n < (rows + 1) * WORDS; n++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT max = 20000
    if (r < rows) {  // collect group w of row r
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
      const ap_uint<10> pM = (w == 0) ? (ap_uint<10>)0 : cM;
      cM = Mg > pM ? Mg : pM;
    }
    if (r >= 1) {  // emit output word w (MODE_F32) / half word w (MODE_INT16) of row r - 1
      const ap_uint<3> j = w.range(2, 0);
      const Ent A = ev[(sl ^ 1) * HALF + j], B = od[(sl ^ 1) * HALF + j];
      const bool y2 = w[3];  // lanes 2, 3 of the groups (y2) for output words 8..15
      ap_uint<27> v[4];
      v[0] = y2 ? A.range(80, 54) : A.range(26, 0);
      v[1] = y2 ? A.range(107, 81) : A.range(53, 27);
      v[2] = y2 ? B.range(80, 54) : B.range(26, 0);
      v[3] = y2 ? B.range(107, 81) : B.range(53, 27);
      const ap_uint<10> mA = A.range(117, 108), mB = B.range(117, 108);
      if (f32) {
        Word o;
        for (int l = 0; l < 4; l++) {
#pragma HLS UNROLL
          o.range(32 * l + 31, 32 * l) = to_f32(v[l][26], v[l].range(25, 0), l < 2 ? mA : mB);
        }
        qs.write(o);
      } else {
        const bool dead = eM <= 57;  // s = 2^(Mtop - 184) below 2^-126, or an all-zero row
        const ap_uint<5> aA = qshift(eM, mA), aB = qshift(eM, mB);
        ap_uint<64> h4;
        for (int l = 0; l < 4; l++) {
#pragma HLS UNROLL
          h4.range(16 * l + 15, 16 * l) = q16(v[l][26], v[l].range(25, 0), l < 2 ? aA : aB, dead);
        }
        if (w[0]) {
          Word o;
          o.range(63, 0) = qlo;
          o.range(127, 64) = h4;
          qs.write(o);
        } else {
          qlo = h4;
        }
        if (w == WORDS - 1) {
          const ap_uint<8> field = dead ? (ap_uint<8>)0 : (ap_uint<8>)(eM - 57);
          ap_uint<32> cur = (tb == 0) ? (ap_uint<32>)0 : sx[sidx];  // a new word starts with token % 4 == 0
          if (tb == 0) cur.range(7, 0) = field;
          if (tb == 1) cur.range(15, 8) = field;
          if (tb == 2) cur.range(23, 16) = field;
          if (tb == 3) cur.range(31, 24) = field;
          sx[sidx] = cur;
          if (h == H - 1) {  // next token: same group of 4 tokens, or the next one
            h = 0;
            if (tb == 3) sbase += H;
            sidx = sbase;
            tb++;
          } else {
            h++;
            sidx++;
          }
        }
      }
    }
    if (w == WORDS - 1) {
      eM = cM;
      sl = ~sl;
      r++;
    }
    w++;
  }
  if (f32) return;
  ap_uint<10> idx = 0, t4 = 0;
  ap_uint<6> hh = 0;
scales:
  for (int i = 0; i < nsw; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT max = 512
    const ap_uint<32> f = sx[idx];  // lanes past T were never written: 0
    Word o = 0;
    for (int b = 0; b < 4; b++) {
#pragma HLS UNROLL
      o.range(32 * b + 30, 32 * b + 23) = f.range(8 * b + 7, 8 * b);
    }
    ss.write(o);
    if (t4 == tq - 1) {  // next head
      t4 = 0;
      hh++;
      idx = hh;
    } else {
      t4++;
      idx += H;
    }
  }
}

}  // namespace rope
