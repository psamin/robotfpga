// attn_pack: the attention operand packer for the int16 engine (fpga/kernels/attention, README.md).
//
// Turns FP32 rows of 64 values (one head of one token, as q_proj / k_proj / v_proj, the KV cache or the P V
// call leave them) into the int16 operands of the engine's attention calls, in the engine's DRIVER CONTRACT
// layouts (fpga/hls/int16_linear.h), each value group with one FP32 scale:
//   QX  queries -> X of the Q K^T call: per KV head j an xmem image, rows t = g * Tq + i (the G query heads
//       of KV head j stacked), X interleaved [Np/32][64][32 tokens], s_tok = per-row scale * scale_mul (1/8)
//   KW  keys    -> W of the Q K^T call: the four port arrays [Mk/32][64] (8 keys per word), s_row = per-key
//       scale, written into the s_row section of the Q K^T xmem image
//   VW  values  -> W of the P V call (V^T: 64 output channels, K = Tk): port p, group g, word k = channels
//       32 g + 8 p .. + 7 of key k; s_row = per-channel scale over the keys (into the P V xmem image)
//   OX  P V outputs -> X of the o_proj call: row i = the H heads' rows gathered, K = H * 64, X interleaved,
//       s_tok = one scale per query over all H * 64 values
// Quantizer: the project's shared one (fpga/kernels/quantize/quant_int16.h: s = fl32(max(amax, 1e-30) / 32767),
// s = 1.0 for an all-zero row, q = clamp(rne(fl32(x / s)), +-32767), no FP32 core). Rows that carry no data (t >= n_rows, or keys of rank
// KRANK_NEVER: padding keys no query may attend) are packed as zeros.
//
// Interface: every parameter lives in a 24-word descriptor in DDR (README.md "attn_pack descriptor"); one call
// runs n_desc consecutive descriptors: attn_pack(src, dst, desc_off, n_desc), src / dst = the same DDR space
// over one 128-bit 'io' m_axi bundle (float_top's xmem / ymem).
//
// Structure per KV-head image: [pre-pass: OX row amax / VW column amax over DDR] then one DATAFLOW region
// reader -> stats -> scaler -> quant -> writer at one input word (4 FP32) per clock. QX / KW: one read of the
// data (row scale computed on the fly, rows are 16 words); OX / VW: two reads (pre-pass + data).
// One row_scale unit (the scaler) and one transposer serve all four modes.
//
// Plain C++ on ap_int.h / hls_stream.h: the same code builds with Vitis HLS (attn_pack_top.cpp) and with g++
// (-I<Vitis_HLS>/include, tb_attn.cpp), where the processes run one after the other on unbounded FIFOs.
#pragma once
#include <ap_int.h>
#include <hls_stream.h>
#include <stdint.h>
#include <string.h>

#ifndef QZ_RECIP_BITS
#define QZ_RECIP_BITS 2  // 2 quotient bits per clock: 3 chained steps miss 250 MHz (csynth 5.5 ns), 2 still give a
                         // scale every 14 clocks, inside the 16 clocks of one 16-word row
#endif
#include "../quantize/quant_int16.h"

namespace attnpack {

static_assert(qz::RECIP_STEPS + 1 <= 16, "the scaler must deliver one row scale per 16-word row");

typedef ap_uint<128> Word;
using qz::RowScale;

constexpr int SEGW = 16;        // FP32 words per 64-value row segment (one head)
constexpr int TB = 32;          // engine tokens per block / keys per W group
constexpr int NMAX = 256;       // output rows per image (n_out)
constexpr int SMAX = 32;        // segment table entries per region
constexpr int OX_SMAX = 16;     // heads per OX row (K <= 1024)
constexpr int DESC_WORDS = 24;  // 96 x uint32
constexpr uint32_t NONE = 0xFFFFFFFFu;
constexpr uint16_t KRANK_NEVER = 0x7FFF;
enum Mode { QX = 0, KW = 1, VW = 2, OX = 3 };

static inline uint32_t wget(const Word& w, int j) {
#pragma HLS INLINE
  return (uint32_t)w.range(32 * j + 31, 32 * j);
}
static inline void wset(Word& w, int j, uint32_t v) {
#pragma HLS INLINE
  w.range(32 * j + 31, 32 * j) = v;
}
static inline uint16_t hget(const Word& w, int l) {
#pragma HLS INLINE
  return (uint16_t)w.range(16 * l + 15, 16 * l);
}
static inline void hset(Word& w, int l, uint16_t v) {
#pragma HLS INLINE
  w.range(16 * l + 15, 16 * l) = v;
}

struct Desc {
  uint32_t mode, n_img, n_rows, n_out, rps, spi, nseg, sm_bits, krank_off, n_split;
  uint32_t off[2], row_stride[2], half_stride[2];
  uint32_t out_off, out_img, out_q, sc_off, sc_img;
};

// descriptor: u[0..20] the fields in Desc order, u[32 + 32 r + h] = segment h's offset in region r
static void read_desc(const Word* src, uint32_t off, Desc& d, uint32_t segtab[2][SMAX]) {
  uint32_t u[24];
#pragma HLS ARRAY_PARTITION variable = u complete
rd:
  for (int i = 0; i < DESC_WORDS; i++) {
#pragma HLS PIPELINE II = 1
    const Word w = src[off + i];
    for (int j = 0; j < 4; j++) {
      const uint32_t v = wget(w, j);
      if (i < 6) u[4 * i + j] = v;
      if (i >= 8) segtab[(i - 8) >> 3][(4 * (i - 8) + j) & (SMAX - 1)] = v;
    }
  }
  d.mode = u[0];
  d.n_img = u[1];
  d.n_rows = u[2];
  d.n_out = u[3];
  d.rps = u[4];
  d.spi = u[5];
  d.nseg = u[6];
  d.sm_bits = u[7];
  d.krank_off = u[8];
  d.n_split = u[9];
  for (int r = 0; r < 2; r++) {
    d.off[r] = u[10 + 3 * r];
    d.row_stride[r] = u[11 + 3 * r];
    d.half_stride[r] = u[12 + 3 * r];
  }
  d.out_off = u[16];
  d.out_img = u[17];
  d.out_q = u[18];
  d.sc_off = u[19];
  d.sc_img = u[20];
}

static int check_desc(const Desc& d) {
  if (d.mode > OX) return 1;
  if (d.n_img < 1 || d.n_rows < 1 || d.n_rows > d.n_out || d.n_out > NMAX) return 2;
  if (d.mode == VW ? d.n_out != d.n_rows : (d.n_out % TB) != 0) return 3;
  if (d.nseg < 1 || (d.mode == OX ? d.nseg > OX_SMAX : d.nseg != 1)) return 4;
  if (d.rps < 1 || (d.rps < TB && d.rps != d.n_rows)) return 5;  // one wrap per +1 / +32 rows
  if ((d.n_img - 1) * d.spi + (d.n_rows - 1) / d.rps + d.nseg > SMAX) return 6;
  if (d.n_split > d.rps || d.row_stride[0] > 0xFFFF || d.row_stride[1] > 0xFFFF) return 7;
  const uint32_t e = (d.sm_bits >> 23) & 0xFF;
  if ((d.sm_bits & 0x807FFFFFu) || e < 100 || e > 127) return 8;  // scale_mul = 2^-27 .. 1
  return 0;
}

// valid[t]: output row t carries data (t < n_rows and, with a rank vector, key t is attended by some query)
static void load_valid(const Word* src, const Desc& d, bool valid[NMAX]) {
  Word w = 0;
lv:
  for (int t = 0; t < NMAX; t++) {
#pragma HLS PIPELINE II = 1
    bool v = (uint32_t)t < d.n_rows;
    if (d.krank_off != NONE) {
      if ((t & 7) == 0 && v) w = src[d.krank_off + (t >> 3)];
      v = v && hget(w, t & 7) != KRANK_NEVER;
    }
    valid[t] = v;
  }
}

// Row-position counters: row t of image j is segment h = j * spi + t / rps (+ s in OX), source row q = t % rps.
typedef ap_uint<9> row_t;  // row indices < NMAX = 256 (one spare bit for the wrap compare)
struct Pos {
  row_t div, mod;
};
static inline Pos pos_add1(Pos p, row_t rps) {  // rows past n_rows wrap at most once too (they carry no data)
#pragma HLS INLINE
  Pos o;
  const row_t m = p.mod + 1;
  const bool wrap = m == rps;
  o.mod = wrap ? row_t(0) : m;
  o.div = wrap ? row_t(p.div + 1) : p.div;
  return o;
}

static inline uint32_t src_addr(const Desc& d, const uint32_t segtab[2][SMAX], uint32_t hbase, Pos p, uint32_t s,
                                uint32_t w) {
#pragma HLS INLINE
  const bool r = p.mod >= row_t(d.n_split);
  const row_t ql = r ? row_t(p.mod - row_t(d.n_split)) : p.mod;
  const uint32_t h = (hbase + (uint32_t)p.div + s) & (SMAX - 1);
  const uint32_t seg = r ? segtab[1][h] : segtab[0][h];
  const uint32_t rs = r ? d.row_stride[1] : d.row_stride[0];
  const uint32_t hs = (w >> 3) ? (r ? d.half_stride[1] : d.half_stride[0]) : 0u;
  return (r ? d.off[1] : d.off[0]) + seg + (uint32_t)(ap_uint<8>(ql) * ap_uint<16>(rs)) + hs + (w & 7);
}

static inline qz::abits_t word_amax(const Word& x) {
#pragma HLS INLINE
  const qz::abits_t a0 = qz::amax2(qz::abs_bits(wget(x, 0)), qz::abs_bits(wget(x, 1)));
  const qz::abits_t a1 = qz::amax2(qz::abs_bits(wget(x, 2)), qz::abs_bits(wget(x, 3)));
  return qz::amax2(a0, a1);
}

// s * scale_mul for a power-of-two scale_mul (an exponent add; s >= 3e-35 keeps the result normal)
static inline uint32_t scaled(uint32_t s_bits, uint32_t sm_bits) {
#pragma HLS INLINE
  return s_bits + (sm_bits - 0x3F800000u);
}

// ---------------------------------------------------------------- the DATAFLOW processes
// iteration space of one image: blocks b (X / KW: n_out / 32; VW: 1) x segments s (nseg) x rows r (32; VW:
// n_rows) x words w (16)
struct Space {
  uint32_t blocks, rows, total;
};
static inline Space space_of(const Desc& d) {
#pragma HLS INLINE
  Space sp;
  sp.blocks = d.mode == VW ? 1u : d.n_out / TB;
  sp.rows = d.mode == VW ? d.n_rows : (uint32_t)TB;
  sp.total = sp.blocks * d.nseg * sp.rows * SEGW;
  return sp;
}

// Process 1: the FP32 words, zeros for rows without data
static void reader(const Word* src, const Desc& d, uint32_t j, const uint32_t segtab[2][SMAX], const bool valid[NMAX],
                   hls::stream<Word>& out) {
  const Space sp = space_of(d);
  const uint32_t hbase = j * d.spi;
  const row_t rps = d.rps, rows1 = sp.rows - 1;
  const ap_uint<5> nseg1 = d.nseg - 1;
  Pos blk = {0, 0}, p = {0, 0};
  row_t t = 0, tb = 0, r = 0;
  ap_uint<5> s = 0;
  ap_uint<4> w = 0;
rd:
  for (uint32_t i = 0; i < sp.total; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT max = 4096
    Word x = 0;
    if (valid[t]) x = src[src_addr(d, segtab, hbase, p, s, w)];
    out.write(x);
    if (w == SEGW - 1) {
      if (r == rows1) {  // next segment of the block, or the next block
        r = 0;
        if (s == nseg1) {
          s = 0;
          blk = pos_add1(p, rps);
          tb = t + 1;
          p = blk;
          t = t + 1;
        } else {
          s++;
          p = blk;
          t = tb;
        }
      } else {
        r++;
        p = pos_add1(p, rps);
        t++;
      }
    }
    w++;
  }
}

// Process 2: forwards the words; QX / KW: emits the amax of every 16-word row
static void stats(const Desc& d, hls::stream<Word>& in, hls::stream<Word>& out, hls::stream<qz::abits_t>& amax) {
  const Space sp = space_of(d);
  const bool grouped = d.mode == QX || d.mode == KW;
  qz::abits_t cur = 0;
  uint32_t w = 0;
st:
  for (uint32_t i = 0; i < sp.total; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT max = 4096
    const Word x = in.read();
    out.write(x);
    const qz::abits_t m = qz::amax2(word_amax(x), cur);
    const bool last = w == SEGW - 1;
    if (last && grouped) amax.write(m);
    cur = last ? qz::abits_t(0) : m;
    w = (w + 1) & (SEGW - 1);
  }
}

// Process 3: the only row-scale unit (quant_int16.h: s exactly by shifts and adds, then R = floor(2^49 / S) by
// restoring division, RECIP_BITS quotient bits per clock), one scale per RECIP_STEPS + 1 clocks (14 at 2 bits),
// inside the 16 clocks a row of 16 words takes:
//   QX / KW: per row from the stats amax; OX: per (block, segment, row) from the pre-pass row amax (the s_tok
//   of a row goes to the writer once, with segment 0); VW: the 64 channel scales from the pre-pass, first.
static void scaler(const Desc& d, const qz::abits_t am_row[NMAX], const qz::abits_t am_col[4][SEGW],
                   hls::stream<qz::abits_t>& amax, hls::stream<RowScale>& rs_out, hls::stream<uint32_t>& sc_out) {
  const Space sp = space_of(d);
  const bool grouped = d.mode == QX || d.mode == KW;
  const uint32_t n = d.mode == VW ? 64u : grouped ? d.n_out : sp.blocks * d.nseg * TB;
  const ap_uint<5> nseg1 = d.nseg - 1;
  ap_uint<5> r = 0, s = 0;
  ap_uint<6> c = 0;
  row_t tb = 0;
  ap_uint<4> phase = 0;
  RowScale rs = qz::row_scale_from_s(0x3F800000u);
  qz::RecipState st = qz::recip_init(rs.S);
sc:
  for (uint32_t i = 0; i < n * (qz::RECIP_STEPS + 1); i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT max = 5120
    if (phase == 0) {
      qz::abits_t a;
      if (grouped)
        a = amax.read();
      else if (d.mode == VW)
        a = am_col[c & 3][c >> 2];
      else
        a = am_row[(tb + r) & (NMAX - 1)];
      rs = qz::row_scale_from_s(qz::scale_bits(a));
      st = qz::recip_init(rs.S);
    } else {
      qz::recip_step(st, rs.S);
    }
    if (phase == qz::RECIP_STEPS) {
      rs.R = qz::recip_result(st);
      rs_out.write(rs);
      if (d.mode != OX || s == 0) sc_out.write(d.mode == QX ? scaled(rs.s_bits, d.sm_bits) : (uint32_t)rs.s_bits);
      if (r == TB - 1) {
        if (s == nseg1) {
          s = 0;
          tb += TB;
        } else {
          s++;
        }
      }
      r++;
      c++;
      phase = 0;
    } else {
      phase++;
    }
  }
}

// Process 4: 4 quantizations per clock; two input words (8 consecutive values of one row) make one int16 word.
// Row scale per 16-word row from the scaler; VW: per lane, channel 4 word + lane, the 64 scales read first.
static void quant(const Desc& d, hls::stream<Word>& in, hls::stream<RowScale>& rs_in, hls::stream<Word>& out) {
  const Space sp = space_of(d);
  const bool vw = d.mode == VW;
  RowScale rs_col[4][SEGW];
#pragma HLS ARRAY_PARTITION variable = rs_col complete dim = 1
#pragma HLS BIND_STORAGE variable = rs_col type = ram_s2p impl = lutram
  if (vw) {
  col:
    for (int c = 0; c < 64; c++) {
#pragma HLS PIPELINE II = 1
      rs_col[c & 3][c >> 2] = rs_in.read();
    }
  }
  RowScale cur = qz::row_scale_from_s(0x3F800000u);  // replaced before first use
  ap_uint<4> w = 0;
  ap_uint<64> lo = 0;
qt:
  for (uint32_t i = 0; i < sp.total; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT max = 4096
    if (w == 0 && !vw) cur = rs_in.read();
    const int wi = w;
    const Word x = in.read();
    ap_uint<64> h = 0;
    for (int j = 0; j < 4; j++) {
      const RowScale rs = vw ? rs_col[j][wi] : cur;
      h.range(16 * j + 15, 16 * j) = ap_uint<16>(qz::quant_elem(wget(x, j), rs));
    }
    if (w & 1) {
      out.write((Word(h) << 64) | Word(lo));
    } else {
      lo = h;
    }
    w++;
  }
}

// Process 5: the output layouts. X / KW: a 32-row x 64-value int16 tile is transposed so one word holds 8 rows
// (tokens / keys) at one k; VW: every int16 word goes straight to its port / group / key address.
static void writer(Word* dst, const Desc& d, uint32_t j, hls::stream<Word>& in, hls::stream<uint32_t>& sc) {
  const Space sp = space_of(d);
  const uint32_t base = d.out_off + j * d.out_img;
  const uint32_t sbase = d.sc_off + j * d.sc_img;
  if (d.mode == VW) {
    Word sw = 0;
  vsc:
    for (int i = 0; i < 64; i++) {  // the 64 channel scales = s_row of the P V call
#pragma HLS PIPELINE II = 1
      wset(sw, i & 3, sc.read());
      if ((i & 3) == 3) dst[sbase + (i >> 2)] = sw;
    }
    const uint32_t q = d.out_q, n = d.n_rows;
    const uint32_t pofs[8] = {0, q, 2 * q, 3 * q, n, q + n, 2 * q + n, 3 * q + n};  // chunk c: port c % 4, group c / 4
#pragma HLS ARRAY_PARTITION variable = pofs complete
    uint32_t t = 0;
    ap_uint<3> c = 0;
  vw:
    for (uint32_t i = 0; i < sp.total / 2; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT max = 2048
      dst[base + pofs[c] + t] = in.read();
      if (c == 7) t++;
      c++;
    }
    return;
  }
  // tile: bank l = row % 8, entry (row / 8) * 8 + chunk: the 8 rows of an output word sit in 8 banks
  Word tile[8][32];
#pragma HLS ARRAY_PARTITION variable = tile complete dim = 1
#pragma HLS BIND_STORAGE variable = tile type = ram_s2p impl = lutram
  const bool xm = d.mode != KW;
  const uint32_t q4[4] = {0, d.out_q, 2 * d.out_q, 3 * d.out_q};
#pragma HLS ARRAY_PARTITION variable = q4 complete
  uint32_t obase = base;  // + (b * nseg + s) * 64 * kstr
  const uint32_t ostep = xm ? 256u : 64u;
blocks:
  for (uint32_t b = 0; b < sp.blocks; b++) {
#pragma HLS LOOP_TRIPCOUNT max = 8
  segs:
    for (uint32_t s = 0; s < d.nseg; s++) {
#pragma HLS LOOP_TRIPCOUNT max = 16
    fill:
      for (int i = 0; i < TB * 8; i++) {
#pragma HLS PIPELINE II = 1
        tile[(i >> 3) & 7][((i >> 6) << 3) | (i & 7)] = in.read();  // row i / 8, chunk i % 8
      }
      // X: [b][k][4 words of 8 tokens], 256 consecutive words; KW: port qq [g][k], four runs of 64 words
    emit:
      for (int i = 0; i < 256; i++) {
#pragma HLS PIPELINE II = 1
        const int k = xm ? (i >> 2) : (i & 63), qq = xm ? (i & 3) : (i >> 6);
        Word o;
        const int e = qq * 8 + (k >> 3), m = k & 7;
        for (int l = 0; l < 8; l++) {
          const Word bw = tile[l][e];
          ap_uint<16> ln[8];
#pragma HLS ARRAY_PARTITION variable = ln complete
          for (int u = 0; u < 8; u++) ln[u] = bw.range(16 * u + 15, 16 * u);
          o.range(16 * l + 15, 16 * l) = ln[m];  // an 8:1 mux of 16 bits per bank
        }
        dst[obase + (xm ? (uint32_t)i : q4[qq] + k)] = o;
      }
      obase += ostep;
    }
    Word sw = 0;
  scw:
    for (int i = 0; i < TB; i++) {  // the block's 32 row scales, 4 per word
#pragma HLS PIPELINE II = 1
      wset(sw, i & 3, sc.read());
      if ((i & 3) == 3) dst[sbase + b * 8 + (i >> 2)] = sw;
    }
  }
}

static void run_pass(const Word* src, Word* dst, const Desc& d, uint32_t j, const uint32_t segtab[2][SMAX],
                     const bool valid[NMAX], const qz::abits_t am_row[NMAX], const qz::abits_t am_col[4][SEGW]) {
#pragma HLS DATAFLOW
#pragma HLS STABLE variable = segtab
#pragma HLS STABLE variable = valid
#pragma HLS STABLE variable = am_row
#pragma HLS STABLE variable = am_col
  hls::stream<Word> raw, fwd, q;
  hls::stream<qz::abits_t> amax;
  hls::stream<RowScale> rs;
  hls::stream<uint32_t> sc;
  // fwd covers one row (16 words) + the scaler's 14 clocks; q covers the writer's 256-clock emit at half rate
#pragma HLS STREAM variable = raw depth = 4
#pragma HLS STREAM variable = fwd depth = 48
#pragma HLS STREAM variable = q depth = 128
#pragma HLS STREAM variable = amax depth = 4
#pragma HLS STREAM variable = rs depth = 4
#pragma HLS STREAM variable = sc depth = 64
#pragma HLS BIND_STORAGE variable = raw type = fifo impl = srl
#pragma HLS BIND_STORAGE variable = fwd type = fifo impl = srl
#pragma HLS BIND_STORAGE variable = amax type = fifo impl = srl
#pragma HLS BIND_STORAGE variable = rs type = fifo impl = srl
#pragma HLS BIND_STORAGE variable = sc type = fifo impl = lutram
#pragma HLS BIND_STORAGE variable = q type = fifo impl = bram
  reader(src, d, j, segtab, valid, raw);
  stats(d, raw, fwd, amax);
  scaler(d, am_row, am_col, amax, rs, sc);
  quant(d, fwd, rs, q);
  writer(dst, d, j, q, sc);
}

// OX: amax of every row over its nseg segments -> am_row[t]; VW: amax of every channel over the valid keys ->
// am_col[lane][word] (channel 4 word + lane). One read of the image's data, no other output.
static void prepass(const Word* src, const Desc& d, uint32_t j, const uint32_t segtab[2][SMAX], const bool valid[NMAX],
                    qz::abits_t am_row[NMAX], qz::abits_t am_col[4][SEGW]) {
  const uint32_t hbase = j * d.spi;
  const bool ox = d.mode == OX;
  const uint32_t rows = ox ? d.n_out : d.n_rows;
  const uint32_t nseg = ox ? d.nseg : 1u;
init:
  for (int w = 0; w < SEGW; w++) {
#pragma HLS PIPELINE II = 1
    for (int l = 0; l < 4; l++) am_col[l][w] = 0;
  }
  Pos p = {0, 0};
  const row_t rps = d.rps;
  const ap_uint<5> nseg1 = nseg - 1;
  row_t t = 0;
  ap_uint<5> s = 0;
  ap_uint<4> w = 0;
  qz::abits_t cur = 0;
scan:
  for (uint32_t i = 0; i < rows * nseg * SEGW; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT max = 16384
#pragma HLS DEPENDENCE variable = am_col inter distance = 16 true
    Word x = 0;
    if (valid[t]) x = src[src_addr(d, segtab, hbase, p, s, w)];
    const qz::abits_t m = qz::amax2(word_amax(x), cur);
    const int wi = w;
    for (int l = 0; l < 4; l++) am_col[l][wi] = qz::amax2(am_col[l][wi], qz::abs_bits(wget(x, l)));
    const bool row_end = w == SEGW - 1 && s == nseg1;
    if (row_end) am_row[t] = m;
    cur = row_end ? qz::abits_t(0) : m;
    if (w == SEGW - 1) {
      if (s == nseg1) {
        s = 0;
        t++;
        p = pos_add1(p, rps);
      } else {
        s++;
      }
    }
    w++;
  }
}

// Runs n_desc consecutive descriptors at word offset desc_off of src. Returns 0, or 16 * index + error.
static int attn_pack(const Word* src, Word* dst, uint32_t desc_off, uint32_t n_desc) {
  static bool valid[NMAX];
  static uint32_t segtab[2][SMAX];
  static qz::abits_t am_row[NMAX];
  static qz::abits_t am_col[4][SEGW];
#pragma HLS ARRAY_PARTITION variable = segtab complete dim = 1
#pragma HLS ARRAY_PARTITION variable = am_col complete dim = 1
#pragma HLS BIND_STORAGE variable = am_col type = ram_2p impl = lutram
descs:
  for (uint32_t i = 0; i < n_desc; i++) {
#pragma HLS LOOP_TRIPCOUNT max = 4
    Desc d;
    read_desc(src, desc_off + i * DESC_WORDS, d, segtab);
    const int err = check_desc(d);
    if (err) return (int)(16 * i) + err;
    load_valid(src, d, valid);
  imgs:
    for (uint32_t j = 0; j < d.n_img; j++) {
#pragma HLS LOOP_TRIPCOUNT max = 12
      if (d.mode == OX || d.mode == VW) prepass(src, d, j, segtab, valid, am_row, am_col);
      run_pass(src, dst, d, j, segtab, valid, am_row, am_col);
    }
  }
  return 0;
}

}  // namespace attnpack
