// The GEMM unit of the integrated float_top: the int16 engine (int16_linear.h compute16, unchanged) behind the
// integration contract's record (fpga/kernels/CONTRACT.md 1.1, 2.3, 3.3, 5 and engine changes E-1..E-6, E-8,
// E-10). Header-only; builds in Vitis HLS 2022.2 and in g++ (the C model: one process after the other).
//
//   y[b][n][m] = ((fl(sum_k X[b][n][k] W[b][m][k]) * s_row[b][m]) * s_tok[b][n]) [+ bias[b][m]] [+ D[b][n][m] | + C]
//   every step one FP32 core operation (round to nearest even, denormals flushed to zero with the sign kept), the
//   sum exact (|sum| < 2^42 because every producer clamps to +-32767, CONTRACT.md 2.1; review R7).
//
// Slots (byte addresses, 64-bit, item b adds b * stride):
//   A  X int16 rows: row n at A + n * pitch, K values (K % 8 == 0)                         io read
//   B  weight blob, the engine's tiling: port p's [Mp/32][K] words at B + p * pitch          hp0..hp3
//   C  Y FP32 rows, exactly N rows x Mp columns; row t at C + (t / R) P1 + (t % R) pitch    io write
//      (R = p2 rows per row group, P1 = p3; R = 0: C + t pitch)
//   D  addend rows (F_ADD_D, pitch 0 = one row for every n), or C itself (F_ACC, C's addressing)  io read
//   E  s_row FP32 [Mp], F  s_tok FP32 [pad4(N)], G  bias FP32 [Mp] (F_BIAS)                  io read
//
// Processes of one item (one DATAFLOW region):
//   load_w   hp0..hp3 -> s0..s3        the group's weights, once per tile (as int16_linear.h load_w16)
//   load_io  io       -> xr, stok_s, srow_s, bias_s, ds   every io read (one reading process per bundle)
//                       X rows in 8-word units (64 k of one token, a 128-byte burst), tile by tile; tokens n >= N
//                       read row 0 and send zeros (no read outside the item's rows; a row's last unit may read up to
//                       7 words past its K values, inside the pitch or up to 112 bytes past the last row); the
//                       addend rows of (tile t, group g) interleaved with slices of X(t + 1) (E-4)
//   xpose    xr -> xs                   32 tokens x 64 k staging (LUTRAM, ping-pong) -> compute16's X words (8 tokens
//                                       at one k), one word in and one word out per clock (E-2)
//   compute16                           the MAC array (int16_linear.h), unchanged
//   dequant_ep  es + scales + ds -> ys  4 lanes (E-1): i43 -> FP32, * s_row, * s_tok, + bias, + D; rows n < N only
//                                       p6 = P_ACT (phase B emit_act): then GELU / SILU / SILU_MUL on y, 2 values
//                                       a clock (fpga/kernels/phaseB/emit_act/emit_act.h)
//   store    ys -> io                   row-major Y, one 128-byte burst per (token, group) (E-3); EPI_SILU_MUL:
//                                       64 bytes (16 output columns per group)
// The dequantizer and the store run 4 FP32 per clock: a block of 1,024 outputs takes 256 clocks, so blocks of
// K < 256 are bound by the store (attention Q.K^T, K = 64), as the contract's time model assumes. With P_ACT the
// dequantizer takes 512 clocks per block (every act GEMM of -12392 has K >= 720: hidden behind the MACs).
#pragma once
#include "int16_linear.h"
#include "../kernels/contract.h"
#include "../kernels/phaseB/emit_act/emit_act.h"

namespace fgemm {

using fstream::F4;
using fstream::IoWord;
using fstream::LANES;
using fstream::PORTS;
using fstream::WPort;
using fstream16::Cap;
using fstream16::EROWS;
using fstream16::EW;
using fstream16::EWord;
using fstream16::Fifo16;
using fstream16::M_MAX;
using fstream16::NBX;
using fstream16::NTT;
using fstream16::TB;
using fstream16::U43;

constexpr int CK = 64;           // k per staging chunk
constexpr int UW = CK / 8;       // io words per (token, chunk): one 128-byte unit
constexpr int DQL = 4;           // dequantizer lanes (E-1)
constexpr int DQS = EW / DQL;    // clocks per emit word
constexpr int YU = LANES / F4;   // io words per (token, group) of Y: 8

// ---------------------------------------------------------------- 16-bit lanes of an io word
#ifdef __VITIS_HLS__
static inline uint16_t h16(const IoWord& w, int j) {
#pragma HLS INLINE
  return (uint16_t)w.range(16 * j + 15, 16 * j);
}
static inline void set16(IoWord& w, int j, uint16_t v) {
#pragma HLS INLINE
  w.range(16 * j + 15, 16 * j) = v;
}
static inline IoWord zero_word() {
#pragma HLS INLINE
  return IoWord(0);
}
static inline IoWord ones_word() {  // four 1.0f
#pragma HLS INLINE
  IoWord w;
  for (int j = 0; j < F4; j++) w.range(32 * j + 31, 32 * j) = ctr::ONE_BITS;
  return w;
}
static inline float fadd_ep(float a, float b) {
#pragma HLS INLINE
  return a + b;  // FP32 adder core (denormals flush to zero)
}
template <class W>
static inline IoWord to_io(const W& w) {
#pragma HLS INLINE
  return (IoWord)w;
}
template <class W>
static inline W from_io(const IoWord& w) {
#pragma HLS INLINE
  return (W)w;
}
#else
static inline uint16_t h16(const IoWord& w, int j) {
  uint16_t v;
  memcpy(&v, (const char*)&w + 2 * j, 2);
  return v;
}
static inline void set16(IoWord& w, int j, uint16_t v) { memcpy((char*)&w + 2 * j, &v, 2); }
static inline IoWord zero_word() {
  IoWord w;
  memset(&w, 0, sizeof w);
  return w;
}
static inline IoWord ones_word() {
  IoWord w;
  for (int j = 0; j < F4; j++) w.v[j] = 1.0f;
  return w;
}
static inline float fadd_ep(float a, float b) { return ctr::ref::fadd_hw(a, b); }
template <class W>
static inline IoWord to_io(const W& w) {
  static_assert(sizeof(W) == 16, "io words are 128 bits");
  IoWord r;
  memcpy(&r, &w, 16);
  return r;
}
template <class W>
static inline W from_io(const IoWord& w) {
  static_assert(sizeof(W) == 16, "io words are 128 bits");
  W r;
  memcpy(&r, &w, 16);
  return r;
}
#endif

// ---------------------------------------------------------------- one item's addresses and sizes (64-bit)
struct Item {
  uint64_t w[PORTS];       // word index of port p's array
  uint64_t xa, ya, da;     // byte addresses: X row 0, Y row 0, addend row 0
  uint64_t ea, fa, ga;     // s_row, s_tok, bias
  uint64_t xp, yp, dp;     // pitches
  uint64_t yR, yP1, dR, dP1;  // row groups of Y and of the addend (F_ACC: C's), R = 0: none
  int N, K, groups, nblk, ntiles, nch;
  bool bias, addd;
  int act;  // p6 = P_ACT: ctr::EPI_* (0 = none)
  int yu;   // io words of Y per (token, group): YU, or YU / 2 for EPI_SILU_MUL
};

// rows per tile that hold tokens n < N
static inline int valid_rows(int N, int t) {
#pragma HLS INLINE
  const int rest = N - t * NTT;
  return rest < NTT ? rest : NTT;
}

// Row-group address walker: row t at base + (t / R) P1 + (t % R) pitch, advanced one row at a time (no divider).
struct RowWalk {
  uint64_t grp, row;  // byte offset of the current row group, of the row inside it
  uint64_t r;         // t % R
  void start(uint64_t t0, uint64_t R, uint64_t P1, uint64_t pitch) {
#pragma HLS INLINE
    if (R == 0) {
      grp = 0;
      row = t0 * pitch;
      r = 0;
    } else {
      grp = (t0 / R) * P1;
      r = t0 % R;
      row = r * pitch;
    }
  }
  void next(uint64_t R, uint64_t P1, uint64_t pitch) {
#pragma HLS INLINE
    if (R != 0 && r + 1 == R) {
      r = 0;
      row = 0;
      grp += P1;
    } else {
      r++;
      row += pitch;
    }
  }
  uint64_t off() const {
#pragma HLS INLINE
    return grp + row;
  }
};

// ---------------------------------------------------------------- process 1: weights
static void load_w(const WPort* w0, const WPort* w1, const WPort* w2, const WPort* w3, const Item it, Fifo16<WPort>& s0,
                   Fifo16<WPort>& s1, Fifo16<WPort>& s2, Fifo16<WPort>& s3) {
  const uint64_t words = (uint64_t)it.groups * it.K;
tiles_w:
  for (int t = 0; t < it.ntiles; t++) {
  load_w:
    for (uint64_t i = 0; i < words; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 256 max = 1048576
      s0.write(w0[it.w[0] + i]);
      s1.write(w1[it.w[1] + i]);
      s2.write(w2[it.w[2] + i]);
      s3.write(w3[it.w[3] + i]);
    }
  }
}

// ---------------------------------------------------------------- process 2: every io read
// HLS allows one reading process per bus bundle in a DATAFLOW region (HLS 200-1013), so this process makes every io
// read of the item, in an order that cannot deadlock and keeps the next tile's X ahead of the MACs:
//   s_tok(0), X(0) (compute16 needs it before its first MAC), s_row, bias, then per tile t:
//   s_tok(t) (t >= 1), and per group g of tile t: the addend rows of (t, g) (F_ADD_D / F_ACC), then the g-th of G
//   slices of X(t + 1). The slices are cut by credit counting (units * G against (g + 1) * units of the tile), no
//   divider. Consumers: the transposer (X: compute16 takes X(t + 1) while tile t computes), the dequantizer (D(t, g)
//   when group g of tile t comes out). Neither waits on a later item of this order, so no cycle.
struct XGen {  // the X unit walk of one item: unit = 8 words (64 k) of one token row; tokens n >= N read row 0
  int a, c, n0;
  uint64_t row, blk_row0;
};
template <class W>
static inline void x_unit(const W* io, const Item& it, XGen& x, Fifo16<IoWord>& xr) {
#pragma HLS INLINE
  const bool live = x.n0 + x.a < it.N;
  const uint64_t base = ((live ? x.row : it.xa) + (uint64_t)x.c * (CK * 2)) >> 4;
  for (int w = 0; w < UW; w++) {
#pragma HLS UNROLL
    const IoWord v = to_io(io[base + w]);
    xr.write(live ? v : zero_word());
  }
  if (x.a == TB - 1) {
    x.a = 0;
    if (x.c == it.nch - 1) {
      x.c = 0;
      x.blk_row0 += (uint64_t)TB * it.xp;
      x.n0 += TB;
    } else {
      x.c++;
    }
    x.row = x.blk_row0;
  } else {
    x.a++;
    x.row += it.xp;
  }
}
template <class W>
static void load_io(const W* io, const Item it, Fifo16<IoWord>& xr, Fifo16<IoWord>& stok_s, Fifo16<IoWord>& srow_s,
                    Fifo16<IoWord>& bias_s, Fifo16<IoWord>& ds) {
  const uint64_t f_words = (uint64_t)ctr::pad4((uint32_t)it.N) / F4;  // s_tok words the slot holds
  XGen x;
  x.a = 0;
  x.c = 0;
  x.n0 = 0;
  x.row = it.xa;
  x.blk_row0 = it.xa;
  const int G = it.groups;
stok0:
  for (int w = 0; w < NTT / F4; w++) {
#pragma HLS PIPELINE II = 1
    stok_s.write((uint64_t)w < f_words ? to_io(io[(it.fa >> 4) + w]) : ones_word());
  }
  const int units0 = fstream16::blocks_of_tile(it.nblk, 0) * it.nch * TB;
load_x0:
  for (int u = 0; u < units0; u++) {
#pragma HLS PIPELINE II = UW
#pragma HLS LOOP_TRIPCOUNT min = 32 max = 32768
    x_unit(io, it, x, xr);
  }
load_srow:
  for (int i = 0; i < G * (LANES / F4); i++) {
#pragma HLS PIPELINE II = 1
    srow_s.write(to_io(io[(it.ea >> 4) + i]));
  }
  if (it.bias) {
  load_bias:
    for (int i = 0; i < G * (LANES / F4); i++) {
#pragma HLS PIPELINE II = 1
      bias_s.write(to_io(io[(it.ga >> 4) + i]));
    }
  }
tiles_io:
  for (int t = 0; t < it.ntiles; t++) {
    if (t > 0) {
    stok_t:
      for (int w = 0; w < NTT / F4; w++) {
#pragma HLS PIPELINE II = 1
        const uint64_t wi = (uint64_t)t * (NTT / F4) + w;
        stok_s.write(wi < f_words ? to_io(io[(it.fa >> 4) + wi]) : ones_word());
      }
    }
    const int nv = valid_rows(it.N, t);
    const int unext = t + 1 < it.ntiles ? fstream16::blocks_of_tile(it.nblk, t + 1) * it.nch * TB : 0;
    uint64_t thr = 0, xg = 0;  // credit: (g + 1) * unext against units emitted * G
    RowWalk rw0;
    rw0.start((uint64_t)t * NTT, it.dR, it.dP1, it.dp);
  groups_io:
    for (int g = 0; g < G; g++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 256
      if (it.addd) {
        RowWalk rw = rw0;
      load_dr:
        for (int i = 0; i < nv; i++) {
#pragma HLS PIPELINE II = UW
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 64
          const uint64_t base = (it.da + rw.off() + (uint64_t)g * (LANES * 4)) >> 4;
          for (int w = 0; w < YU; w++) {
#pragma HLS UNROLL
            ds.write(to_io(io[base + w]));
          }
          rw.next(it.dR, it.dP1, it.dp);
        }
      }
      thr += (uint64_t)unext;
    load_xs:
      while (xg < thr) {
#pragma HLS PIPELINE II = UW
#pragma HLS LOOP_TRIPCOUNT min = 0 max = 1024
        x_unit(io, it, x, xr);
        xg += (uint64_t)G;
      }
    }
  }
}

// ---------------------------------------------------------------- process 3: rows -> compute16's X words
// In: per (block, chunk), 32 tokens x 8 words (word w = k 8w..8w+7 of the chunk, lane = k % 8). Out: per
// (block, chunk), for k < kc, q < 4: the word of tokens 8q..8q+7 at k (lane j = token 8q + j), as the per-call
// xmem image had them. Staging bank j holds tokens with a % 8 == j: one write and one 8-bank read per clock.
constexpr int XSETTLE = 4;  // clocks between a half's last write and its first read (pipeline distance)
// The read side keeps the word it offers in an output register (obuf): a word is loaded from the staging (LUTRAM
// read + lane select) whenever obuf is free or being taken this clock, so no path runs from the staging through
// the lane select into the FIFO handshake (csynth int16seq v1: 4.95 ns; the FIFO's full flag now only enables
// the register).
static void xpose(Fifo16<IoWord>& xr, Fifo16<IoWord>& xs, const Item it) {
  IoWord stage[8][2 * 32];  // [token % 8][half * 32 + (token / 8) * 8 + word]
#pragma HLS ARRAY_PARTITION variable = stage complete dim = 1
#pragma HLS BIND_STORAGE variable = stage type = ram_s2p impl = lutram
  const int halves_total = it.nblk * it.nch;
  const int klast = it.K - CK * (it.nch - 1);  // k of the last chunk of a block
  // write side
  int wh = 0, wa = 0, ww = 0, wdone = 0;  // half, token, word, halves written
  // read side: the load pointer (half, k, q, chunk of the block) and the halves fully loaded into obuf
  int rh = 0, rk = 0, rq = 0, rc = 0, rdone = 0;
  bool ov = false;  // obuf holds a word not yet taken by xs
  IoWord obuf = zero_word();
  int sent = 0;  // halves whose last word xs has taken
  bool last_in_obuf = false;  // obuf holds the last word of a half
  bool wd_dl[XSETTLE], rd_dl[XSETTLE];  // half-written / half-loaded events, delayed
#pragma HLS ARRAY_PARTITION variable = wd_dl complete
#pragma HLS ARRAY_PARTITION variable = rd_dl complete
  for (int i = 0; i < XSETTLE; i++) {
#pragma HLS UNROLL
    wd_dl[i] = false;
    rd_dl[i] = false;
  }
  int wseen = 0, rseen = 0;  // halves written (as the read side sees them), loaded (as the write side sees them)
xpose_main:
  while (true) {
#pragma HLS PIPELINE II = 1
#pragma HLS DEPENDENCE variable = stage inter false
#pragma HLS DEPENDENCE variable = stage intra false
#pragma HLS LOOP_TRIPCOUNT min = 256 max = 1048576
    const bool fin = sent == halves_total;  // exit test from the registered state (taken at the end)
    // read side, output: the registered word
    const bool taken = ov && xs.write_nb(obuf);
    if (taken && last_in_obuf) sent++;
    // read side, load: the next word into obuf while obuf is free (or taken now) and its half is written
    const int kc = rc == it.nch - 1 ? klast : CK;
    bool rfin = false;
    if ((!ov || taken) && wseen > rdone) {
      const int addr = rh * 32 + rq * 8 + (rk >> 3);
      IoWord o;
      for (int j = 0; j < 8; j++) {
#pragma HLS UNROLL
        set16(o, j, h16(stage[j][addr], rk & 7));
      }
      obuf = o;
      ov = true;
      last_in_obuf = rq == 3 && rk == kc - 1;
      if (rq == 3) {
        rq = 0;
        if (rk == kc - 1) {
          rk = 0;
          rfin = true;
          rh ^= 1;
          rc = rc == it.nch - 1 ? 0 : rc + 1;
        } else {
          rk++;
        }
      } else {
        rq++;
      }
    } else if (taken) {
      ov = false;
    }
    // write side: one input word into half wh while it is free (at most 2 halves written and not loaded)
    bool wfin = false;
    if (wdone < halves_total && wdone - rseen < 2) {
      IoWord v;
      if (xr.read_nb(v)) {
        stage[wa & 7][wh * 32 + (wa >> 3) * 8 + ww] = v;
        if (ww == UW - 1) {
          ww = 0;
          if (wa == TB - 1) {
            wa = 0;
            wfin = true;
            wh ^= 1;
          } else {
            wa++;
          }
        } else {
          ww++;
        }
      }
    }
    if (wfin) wdone++;
    if (rfin) rdone++;
    if (wd_dl[XSETTLE - 1]) wseen++;
    if (rd_dl[XSETTLE - 1]) rseen++;
    for (int i = XSETTLE - 1; i > 0; i--) {
#pragma HLS UNROLL
      wd_dl[i] = wd_dl[i - 1];
      rd_dl[i] = rd_dl[i - 1];
    }
    wd_dl[0] = wfin;
    rd_dl[0] = rfin;
    if (fin) break;
  }
}

// ---------------------------------------------------------------- process 6: block sums -> outputs (4 lanes)
// Without P_ACT: one quarter (4 values) of an emit word a clock, lane j the epilogue of value 4q + j.
// With P_ACT (emit_act.h): two clocks per quarter (h = 0, 1). Lanes 0, 1 run the epilogue of values 4q + 2h and
// 4q + 2h + 1 (h = 1: their inputs are lanes 2, 3 of the quarter, kept from h = 0 in the stash); lanes 2, 3 lend
// their second multiplier for the activation: fmul(x, h(x)) (GELU, SILU, SILU_MUL gate half) or fmul(up, gb)
// (SILU_MUL up half, gb = silu(gate) of the same column, written 8 iterations earlier). One io word of y leaves
// every second clock (SILU_MUL: of the up half only). Each FP32 operator stays one core per lane (2 fmul, 2 fadd).
static void dequant_ep(Fifo16<IoWord>& srow_s, Fifo16<IoWord>& bias_s, Fifo16<IoWord>& stok_s,
                       Fifo16<EWord>& es, Fifo16<IoWord>& ds, Fifo16<IoWord>& ys, const Item it) {
  float srow[M_MAX], bsv[M_MAX];
#pragma HLS ARRAY_PARTITION variable = srow cyclic factor = 4
#pragma HLS ARRAY_PARTITION variable = bsv cyclic factor = 4
  float stok[NTT];
  Cap prev[EROWS * DQS][DQL];  // [row * 4 + quarter][lane]
#pragma HLS ARRAY_PARTITION variable = prev complete dim = 2
#pragma HLS BIND_STORAGE variable = prev type = ram_2p impl = lutram
  float gb[2][2 * DQS];  // EPI_SILU_MUL: silu(gate) of the token's 16 gate columns, [act lane][2 q + h]
#pragma HLS ARRAY_PARTITION variable = gb complete dim = 1
#pragma HLS BIND_STORAGE variable = gb type = ram_s2p impl = lutram
prev_zero:
  for (int r = 0; r < EROWS * DQS; r++) {
#pragma HLS PIPELINE II = 1
    for (int j = 0; j < DQL; j++) {
#pragma HLS UNROLL
      prev[r][j] = 0;
    }
    if (r < 2 * DQS) {
      gb[0][r] = 0.0f;
      gb[1][r] = 0.0f;
    }
  }
srow_in:
  for (int i = 0; i < it.groups * (LANES / F4); i++) {
#pragma HLS PIPELINE II = 1
    const IoWord v = srow_s.read();
    IoWord bw = zero_word();
    if (it.bias) bw = bias_s.read();
    for (int j = 0; j < F4; j++) {
#pragma HLS UNROLL
      srow[F4 * i + j] = fstream::io_lane(v, j);
      bsv[F4 * i + j] = fstream::io_lane(bw, j);
    }
  }
  const bool am = it.act != ea::ACT_NONE;      // an activation follows the epilogue
  const bool tsilu = it.act != ea::ACT_GELU;   // h = sigmoid (else Phi_tanh)
  const bool mul = it.act == ea::ACT_SILU_MUL;  // gate | up halves
  const int sub = am ? 2 : 1;                   // clocks per quarter
  // the epilogue lanes' inputs for h = 1 (lanes 2, 3 of the quarter, kept from h = 0) and the h = 0 results
  float sd[2], ssr[2], sbs[2], sdv[2], ylo[2];
#pragma HLS ARRAY_PARTITION variable = sd complete
#pragma HLS ARRAY_PARTITION variable = ssr complete
#pragma HLS ARRAY_PARTITION variable = sbs complete
#pragma HLS ARRAY_PARTITION variable = sdv complete
#pragma HLS ARRAY_PARTITION variable = ylo complete
  for (int j = 0; j < 2; j++) {
#pragma HLS UNROLL
    sd[j] = 0.0f;
    ssr[j] = 0.0f;
    sbs[j] = 0.0f;
    sdv[j] = 0.0f;
    ylo[j] = 0.0f;
  }
tiles_dq:
  for (int t = 0; t < it.ntiles; t++) {
    const int nbt = fstream16::blocks_of_tile(it.nblk, t);
  stok_in:
    for (int i = 0; i < NTT / F4; i++) {
#pragma HLS PIPELINE II = 1
      const IoWord v = stok_s.read();
      for (int j = 0; j < F4; j++) {
#pragma HLS UNROLL
        stok[F4 * i + j] = fstream::io_lane(v, j);
      }
    }
    int g = 0, b = 0, r = 0, q = 0;
    bool h1 = false;  // the second clock of a quarter (P_ACT only)
    EWord cw = EWord();
  dq:
    for (int itr = 0; itr < it.groups * nbt * EROWS * DQS * sub; itr++) {
#pragma HLS PIPELINE II = 1
#pragma HLS DEPENDENCE variable = prev inter false
#pragma HLS DEPENDENCE variable = gb type = inter direction = RAW distance = 8 dependent = true
#pragma HLS LOOP_TRIPCOUNT min = 256 max = 1048576
      if (q == 0 && !h1) cw = es.read();
      const int tok = b * TB + (r >> 1);
      const bool valid = t * NTT + tok < it.N;
      const float st = stok[tok];
      IoWord dv = zero_word();
      if (it.addd && valid && !h1) dv = ds.read();
      // front: the quarter's 4 block sums (prev written once, at h = 0)
      float dfv[DQL], srv[DQL], bsj[DQL], dvj[DQL];
#pragma HLS ARRAY_PARTITION variable = dfv complete
#pragma HLS ARRAY_PARTITION variable = srv complete
#pragma HLS ARRAY_PARTITION variable = bsj complete
#pragma HLS ARRAY_PARTITION variable = dvj complete
      for (int j = 0; j < DQL; j++) {
#pragma HLS UNROLL
        const Cap c = fstream16::ew_get(cw, j);  // lanes 4q..4q+3: the word shifts down by 4 values per quarter
        const U43 d = fstream16::cap_diff(c, prev[r * DQS + q][j]);
        if (!h1) prev[r * DQS + q][j] = c;
        const int m = g * LANES + (r & 1) * EW + DQL * q + j;
        dfv[j] = fstream16::i43_to_f32(d);
        srv[j] = srow[m];
        bsj[j] = bsv[m];
        dvj[j] = fstream::io_lane(dv, j);
      }
      // lanes 0, 1 at h = 1: values 4q + 2, 4q + 3 from the stash
      float ind[DQL], ins[DQL], inb[DQL], ina[DQL];
#pragma HLS ARRAY_PARTITION variable = ind complete
#pragma HLS ARRAY_PARTITION variable = ins complete
#pragma HLS ARRAY_PARTITION variable = inb complete
#pragma HLS ARRAY_PARTITION variable = ina complete
      for (int j = 0; j < DQL; j++) {
#pragma HLS UNROLL
        const bool st_in = j < 2 && h1;
        ind[j] = st_in ? sd[j & 1] : dfv[j];
        ins[j] = st_in ? ssr[j & 1] : srv[j];
        inb[j] = st_in ? sbs[j & 1] : bsj[j];
        ina[j] = st_in ? sdv[j & 1] : dvj[j];
      }
      if (!h1) {
        for (int j = 0; j < 2; j++) {
#pragma HLS UNROLL
          sd[j] = dfv[2 + j];
          ssr[j] = srv[2 + j];
          sbs[j] = bsj[2 + j];
          sdv[j] = dvj[2 + j];
        }
      }
      float outv[DQL];
#pragma HLS ARRAY_PARTITION variable = outv complete
      float xv[2];
#pragma HLS ARRAY_PARTITION variable = xv complete
      for (int j = 0; j < 2; j++) {  // lanes 0, 1: the epilogue
#pragma HLS UNROLL
        float y = fstream16::fmul_dq(fstream16::fmul_dq(ind[j], ins[j]), st);
        if (it.bias) y = fadd_ep(y, inb[j]);
        if (it.addd) y = fadd_ep(y, ina[j]);
        xv[j] = y;
        outv[j] = y;
      }
      const bool uph = mul && (r & 1) != 0;  // SILU_MUL: the up half of the token's group
      const int gi = 2 * q + (h1 ? 1 : 0);
      for (int j = 2; j < DQL; j++) {  // lanes 2, 3: the epilogue, or the activation of lane j - 2's value
#pragma HLS UNROLL
        const int p = j - 2;
        const float m1 = fstream16::fmul_dq(ind[j], ins[j]);
        const float hv = ea::hval(xv[p], tsilu);
        const float gv = gb[p][gi];
        const float pr = fstream16::fmul_dq(am ? xv[p] : m1, am ? (uph ? gv : hv) : st);
        if (mul && !uph) gb[p][gi] = pr;
        float y = pr;
        if (it.bias) y = fadd_ep(y, inb[j]);
        if (it.addd) y = fadd_ep(y, ina[j]);
        outv[j] = am ? pr : y;
      }
      IoWord o;
      fstream::io_set(o, 0, am ? ylo[0] : outv[0]);
      fstream::io_set(o, 1, am ? ylo[1] : outv[1]);
      fstream::io_set(o, 2, outv[2]);
      fstream::io_set(o, 3, outv[3]);
      if (am && !h1) {
        ylo[0] = outv[2];
        ylo[1] = outv[3];
      }
      if (valid && (!am || (h1 && (!mul || uph)))) ys.write(o);
      const bool qend = !am || h1;  // the quarter is done
      h1 = am && !h1;
      if (qend) {
        cw = fstream16::ew_shift4(cw);
        if (q == DQS - 1) {
          q = 0;
          if (r == EROWS - 1) {
            r = 0;
            if (b == nbt - 1) {
              b = 0;
              g++;
            } else {
              b++;
            }
          } else {
            r++;
          }
        } else {
          q++;
        }
      }
    }
  }
}

// ---------------------------------------------------------------- process 7: row-major Y
// it.yu io words per (token, group): 8 (32 FP32 columns), or 4 for EPI_SILU_MUL (the group's 16 output columns).
template <class W>
static void store(Fifo16<IoWord>& ys, W* io, const Item it) {
  const uint64_t gstep = (uint64_t)it.yu * 16;  // bytes per group in a Y row: 128 or 64
tiles_st:
  for (int t = 0; t < it.ntiles; t++) {
    const int nv = valid_rows(it.N, t);
    RowWalk rw0;
    rw0.start((uint64_t)t * NTT, it.yR, it.yP1, it.yp);
    RowWalk rw = rw0;
    int i = 0;
    uint64_t gofs = 0;  // g * gstep
  store_y:
    for (int u = 0; u < it.groups * nv; u++) {
#pragma HLS PIPELINE II = YU
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 16384
      const uint64_t base = (it.ya + rw.off() + gofs) >> 4;
      for (int w = 0; w < YU; w++) {
#pragma HLS UNROLL
        if (w < it.yu) io[base + w] = from_io<W>(ys.read());
      }
      if (i == nv - 1) {
        i = 0;
        gofs += gstep;
        rw = rw0;
      } else {
        i++;
        rw.next(it.yR, it.yP1, it.yp);
      }
    }
  }
}

// ---------------------------------------------------------------- one item
template <class WP, class W>
static void gemm_item(const WP* w0, const WP* w1, const WP* w2, const WP* w3, const W* xmem, W* ymem, const Item it) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW
  Fifo16<WPort> s0, s1, s2, s3;
  Fifo16<IoWord> xr, xs, srow_s, stok_s, bias_s, ds, ys;
  Fifo16<EWord> es;
#pragma HLS STREAM variable = s0 depth = fstream16::WFIFO16
#pragma HLS STREAM variable = s1 depth = fstream16::WFIFO16
#pragma HLS STREAM variable = s2 depth = fstream16::WFIFO16
#pragma HLS STREAM variable = s3 depth = fstream16::WFIFO16
#if FSTREAM_NBX > 1
#pragma HLS BIND_STORAGE variable = s0 type = fifo impl = uram
#pragma HLS BIND_STORAGE variable = s1 type = fifo impl = uram
#pragma HLS BIND_STORAGE variable = s2 type = fifo impl = uram
#pragma HLS BIND_STORAGE variable = s3 type = fifo impl = uram
#endif
#pragma HLS STREAM variable = xr depth = 64
#pragma HLS STREAM variable = xs depth = 32
#pragma HLS STREAM variable = srow_s depth = 64
#pragma HLS STREAM variable = bias_s depth = 64
#pragma HLS STREAM variable = stok_s depth = 32
#pragma HLS STREAM variable = es depth = 4
#pragma HLS STREAM variable = ds depth = 64
#pragma HLS STREAM variable = ys depth = 32
#pragma HLS BIND_STORAGE variable = xr type = fifo impl = lutram
#pragma HLS BIND_STORAGE variable = xs type = fifo impl = srl
#pragma HLS BIND_STORAGE variable = srow_s type = fifo impl = srl
#pragma HLS BIND_STORAGE variable = bias_s type = fifo impl = srl
#pragma HLS BIND_STORAGE variable = stok_s type = fifo impl = srl
#pragma HLS BIND_STORAGE variable = es type = fifo impl = srl
#pragma HLS BIND_STORAGE variable = ds type = fifo impl = lutram
#pragma HLS BIND_STORAGE variable = ys type = fifo impl = srl
  load_w(w0, w1, w2, w3, it, s0, s1, s2, s3);
  load_io(xmem, it, xr, stok_s, srow_s, bias_s, ds);
  xpose(xr, xs, it);
  fstream16::compute16(s0, s1, s2, s3, xs, es, it.groups, it.K, it.nblk, it.ntiles);
  dequant_ep(srow_s, bias_s, stok_s, es, ds, ys, it);
  store(ys, ymem, it);
}

static inline Item plan_item(const ctr::Cmd& c, uint64_t b) {
#pragma HLS INLINE
  Item it;
  const ctr::Slot &A = c.s[ctr::SLOT_A], &Bw = c.s[ctr::SLOT_B], &C = c.s[ctr::SLOT_C], &D = c.s[ctr::SLOT_D];
  const ctr::Slot &E = c.s[ctr::SLOT_E], &F = c.s[ctr::SLOT_F], &G = c.s[ctr::SLOT_G];
  for (int p = 0; p < PORTS; p++) {
#pragma HLS UNROLL
    it.w[p] = (Bw.addr + (uint64_t)p * Bw.pitch + b * Bw.stride) >> 4;
  }
  it.xa = A.addr + b * A.stride;
  it.xp = A.pitch;
  it.ya = C.addr + b * C.stride;
  it.yp = C.pitch;
  it.yR = c.p[ctr::P_SEGC];
  it.yP1 = c.p[ctr::P_SEGF];
  const bool acc = (c.flags & ctr::F_ACC) != 0;
  it.addd = acc || (c.flags & ctr::F_ADD_D) != 0;
  it.da = acc ? it.ya : D.addr + b * D.stride;
  it.dp = acc ? it.yp : (uint64_t)D.pitch;
  it.dR = acc ? it.yR : 0;
  it.dP1 = acc ? it.yP1 : 0;
  it.ea = E.addr + b * E.stride;
  it.fa = F.addr + b * F.stride;
  it.ga = G.addr + b * G.stride;
  it.bias = (c.flags & ctr::F_BIAS) != 0;
  it.act = (int)c.p[ctr::P_ACT];
  it.yu = it.act == ea::ACT_SILU_MUL ? YU / 2 : YU;
  it.N = (int)c.N;
  it.K = (int)c.K;
  it.groups = (int)(c.M / LANES);
  it.nblk = (int)((c.N + TB - 1) / TB);
  it.ntiles = (it.nblk + NBX - 1) / NBX;
  it.nch = (int)((c.K + CK - 1) / CK);
  return it;
}

// The unit: one command (validate_hw passed: Mp 32..8192 % 32, K 8..4096 % 8, N, B 1..65535, slots aligned,
// addresses < 2^40). Items one after the other (each a DATAFLOW run; E-7 would overlap them). Returns ctr::ME_OK
// after the last write response of the last item (the store's writes end the region).
template <class WP, class W>
static uint32_t gemm_unit(const ctr::Cmd& c, const WP* hp0, const WP* hp1, const WP* hp2, const WP* hp3,
                          const W* io_rd, W* io_wr) {
#pragma HLS INLINE off
items:
  for (uint32_t b = 0; b < c.B; b++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 24
    const Item it = plan_item(c, b);
    gemm_item(hp0, hp1, hp2, hp3, io_rd, io_wr, it);
  }
  return ctr::ME_OK;
}

}  // namespace fgemm
