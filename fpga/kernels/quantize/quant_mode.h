// QUANT_X mode core: B items x N rows x K FP32 (row-major, any row pitch / item stride) -> int16 rows
// + one FP32 scale per row, the arithmetic of quant_int16.h. Header only, so float_top can call
// qz::quant_rows(w0, w1, ymem, cmd) as one more mode on its existing bundles; quant_top.cpp is the
// standalone Vitis HLS top of the same core.
//
// Bundles: reads over two 128-bit read ports (even rows over the first, odd rows over the second:
// hp0 / hp1 in float_top), writes over one 128-bit port (io). All addresses are 128-bit word
// indices (byte address / 16) relative to the pointers passed in; in sequencer mode those are 0 and
// the indices are absolute.
//
// DDR layouts (all 16-byte aligned):
//   input   item b, row n: K FP32 at word in + b * in_bstride + n * in_pitch, 4 values a word
//           (value j in bits [32j+31:32j])
//   output  item b, row n: K int16 at word out + b * out_bstride + n * out_pitch, 8 values a word
//           (value j in bits [16j+15:16j], two's complement)
//   scales  item b: N FP32 at word scl + b * scl_bstride, 4 a word; the last word of an item is
//           padded with 1.0f (so an item's scale vector takes ceil(N / 4) words)
// Constraints: B >= 1, N >= 1, K a multiple of 16 in [16, K_MAX], in_pitch >= K / 4 and
// out_pitch >= K / 8 (when N > 1), scl_bstride >= ceil(N / 4) (when B > 1). quant_cmd_ok checks.
//
// Processes (one DATAFLOW region per call), 8 values per clock in steady state:
//   rd<0>, rd<1>  one per read port, alternate rows: each row is one burst of K / 4 words; the row's
//                 absmax goes to am<p>, its words, paired, to f<p> (a FIFO of 256-bit pairs).
//                 Rows longer than 8 * FIFO_DEPTH values take two passes over DDR (absmax, then data).
//   sc            absmax -> s (scale_bits), one row a clock; packs the scales 4 to a word for wr
//   rc            S -> R: NE restoring dividers (RECIP_BITS bits a clock each), rows round robin:
//                 a row every (2 + RECIP_STEPS) / NE clocks (13.5 at NE = 2, 1 bit)
//   qt            8 lanes of quant_elem a clock, one output word a clock (the rows in order)
//   wr            per item, segments of SEG rows: one burst of segment data when the output rows are
//                 contiguous (out_pitch == K / 8, else one burst per row), then one burst of its
//                 SEG / 4 scale words
#pragma once
#include <hls_stream.h>

#include "quant_int16.h"

namespace qz {

typedef ap_uint<36> waddr_t;  // a 128-bit word index: 40-bit byte addresses
typedef ap_uint<256> pair_t;  // two consecutive input words of one row (low word first)

#ifndef QZ_FIFO_DEPTH
#define QZ_FIFO_DEPTH 128  // pairs per row FIFO: rows of <= 1024 values in one DDR pass (LUTRAM)
#endif
constexpr int FIFO_DEPTH = QZ_FIFO_DEPTH;
constexpr int SINGLE_PASS_MAX = 8 * FIFO_DEPTH;  // K <= this: one DDR read per row
constexpr int K_MAX = 1 << 18;                   // longest row (values)
constexpr int SEG = 64;                          // rows per output segment (a multiple of 4)

struct QCmd {
  ap_uint<16> B;        // items
  ap_uint<24> N;        // rows per item
  ap_uint<20> K;        // values per row
  waddr_t in, out, scl;  // word index of item 0 row 0 (input, int16 output, scale vector)
  ap_uint<32> in_pitch, in_bstride;    // words between rows / items of the input
  ap_uint<32> out_pitch, out_bstride;  // words between rows / items of the output
  ap_uint<32> scl_bstride;             // words between the items' scale vectors
};

static inline bool quant_cmd_ok(const QCmd& c) {
#pragma HLS INLINE
  const ap_uint<32> wi = c.K >> 2, wo = c.K >> 3, ws = (ap_uint<32>(c.N) + 3) >> 2;
  return c.B >= 1 && c.N >= 1 && c.K >= 16 && c.K <= K_MAX && (c.K & 15) == 0 &&
         (c.N == 1 || c.in_pitch >= wi) && (c.N == 1 || c.out_pitch >= wo) &&
         (c.B == 1 || c.scl_bstride >= ws);
}

// ---------------------------------------------------------------- read (one per port)
template <int P>
static void qz_rd(const word_t* mem, const QCmd& c, hls::stream<pair_t>& f, hls::stream<abits_t>& am) {
  const ap_uint<18> W = c.K >> 2;  // words per row
  const bool two = c.K > SINGLE_PASS_MAX;
  waddr_t item = c.in;
  bool mine = P == 0;  // row 0 (global row index b N + n even) is port 0's
items:
  for (ap_uint<17> b = 0; b < c.B; b++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 16 avg = 2
    waddr_t row = item;
  rows:
    for (ap_uint<25> n = 0; n < c.N; n++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 1024 avg = 128
#pragma HLS LOOP_FLATTEN off
      if (mine) {
        abits_t m = 0;
        word_t lo = 0;
      scan:
        for (ap_uint<18> j = 0; j < W; j++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 4 max = 256 avg = 192
          word_t w = mem[row + j];
          m = amax2(m, word_amax(w));
          if (!two) {
            if (j[0]) f.write((pair_t(w) << 128) | pair_t(lo));
            else lo = w;
          }
        }
        am.write(m);
        if (two) {
        again:
          for (ap_uint<18> j = 0; j < W; j++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 4 max = 4096 avg = 3072
            word_t w = mem[row + j];
            if (j[0]) f.write((pair_t(w) << 128) | pair_t(lo));
            else lo = w;
          }
        }
      }
      mine = !mine;
      row += c.in_pitch;
    }
    item += c.in_bstride;
  }
}

// ---------------------------------------------------------------- row scales
// sc: absmax -> s, one row a clock, rows in order. Sends s to qt (which re-derives S, T, B), S and the divider's start state to rc, and packs the scales 4 to a word for wr (the last word
// of an item padded with 1.0f).
struct SeedS {  // sc -> rc
  ap_uint<24> S;
  RecipSeed sd;
};
static void qz_sc(const QCmd& c, hls::stream<abits_t>& am0, hls::stream<abits_t>& am1,
                  hls::stream<SeedS>& sdf, hls::stream<ap_uint<32>>& rpf, hls::stream<word_t>& sclf) {
  ap_uint<17> b = 0;
  ap_uint<25> n = 0;
  bool odd = false;
  ap_uint<32> w0 = ONE_BITS, w1 = ONE_BITS, w2 = ONE_BITS, w3 = ONE_BITS;  // the scale word being packed
rows:
  for (;;) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 8192 avg = 256
    abits_t a = odd ? am1.read() : am0.read();
    const ap_uint<32> s = scale_bits(a);
    const RowScale rs = row_scale_from_s(s);
    SeedS o;
    o.S = rs.S;
    o.sd = recip_seed(rs.S);
    sdf.write(o);
    rpf.write(s);
    const ap_uint<2> l = n(1, 0);
    w0 = l == 0 ? s : w0;
    w1 = l == 1 ? s : (l == 0 ? ap_uint<32>(ONE_BITS) : w1);
    w2 = l == 2 ? s : (l == 0 ? ap_uint<32>(ONE_BITS) : w2);
    w3 = l == 3 ? s : (l == 0 ? ap_uint<32>(ONE_BITS) : w3);
    const bool last = n == c.N - 1;
    if (l == 3 || last) sclf.write((word_t(w3) << 96) | (word_t(w2) << 64) | (word_t(w1) << 32) | word_t(w0));
    odd = !odd;
    if (last) {
      n = 0;
      if (++b == c.B) break;
    } else {
      n++;
    }
  }
}

// rc: NE reciprocal engines, rows dealt round robin: each takes 2 + RECIP_STEPS clocks per row (load,
// 25 / RECIP_BITS steps, hand over), so a row leaves every (2 + RECIP_STEPS) / NE clocks (13.5 at the
// defaults). Only S, the remainder and the quotient live here; R goes to qt in row order.
#ifndef QZ_RECIP_ENGINES
#define QZ_RECIP_ENGINES 2
#endif
constexpr int NE = QZ_RECIP_ENGINES;
static void qz_rc(const QCmd& c, hls::stream<SeedS>& sdf, hls::stream<ap_uint<26>>& rf) {
  ap_uint<24> S[NE];
  RecipState st[NE];
  bool busy[NE];
#pragma HLS ARRAY_PARTITION variable = S complete
#pragma HLS ARRAY_PARTITION variable = st complete
#pragma HLS ARRAY_PARTITION variable = busy complete
  for (int e = 0; e < NE; e++) {
#pragma HLS UNROLL
    busy[e] = false;
    st[e].n = 0;
    S[e] = 0;
  }
  ap_uint<17> bi = 0, bo = 0;  // item of the next row to load / to hand over
  ap_uint<25> ni = 0, no = 0;  // its row
  bool more = true;            // rows left to load
  ap_uint<4> ei = 0, eo = 0;   // engine to load next / to hand over next
steps:
  for (;;) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 27 max = 110592 avg = 3456
    // Both decisions use the state at the start of the clock (an engine freed by the hand-over is
    // reloaded one clock later), which keeps them off each other's path.
    const bool fin = busy[eo] && st[eo].n == 0;
    const bool can_load = more && !busy[ei];
    bool done = false;
    if (fin) {  // hand over: the oldest engine, once its 25 quotient bits are done
      rf.write(recip_result(st[eo]));
      busy[eo] = false;
      eo = eo == NE - 1 ? ap_uint<4>(0) : ap_uint<4>(eo + 1);
      if (no == c.N - 1) {
        no = 0;
        if (++bo == c.B) done = true;
      } else {
        no++;
      }
    }
    for (int e = 0; e < NE; e++) {  // RECIP_BITS quotient bits for every running engine
#pragma HLS UNROLL
      recip_step(st[e], S[e]);
    }
    if (can_load) {  // the next row into the next engine
      SeedS in;
      if (sdf.read_nb(in)) {
        S[ei] = in.S;
        st[ei] = recip_load(in.sd);
        busy[ei] = true;
        ei = ei == NE - 1 ? ap_uint<4>(0) : ap_uint<4>(ei + 1);
        if (ni == c.N - 1) {
          ni = 0;
          if (++bi == c.B) more = false;
        } else {
          ni++;
        }
      }
    }
    if (done) break;
  }
}

// ---------------------------------------------------------------- quantize, 8 lanes
static void qz_qt(const QCmd& c, hls::stream<pair_t>& f0, hls::stream<pair_t>& f1, hls::stream<ap_uint<32>>& rpf,
                  hls::stream<ap_uint<26>>& rf, hls::stream<word_t>& ow) {
  const ap_uint<17> H = c.K >> 3;  // pairs (= output words) per row
  RowScale rs;
  ap_uint<17> j = 0, b = 0;
  ap_uint<25> n = 0;
  bool odd = false;
words:
  for (;;) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 2 max = 1048576 avg = 24576
    if (j == 0) {
      rs = row_scale_from_s(rpf.read());  // S, T, B again from s: cheaper than a wide FIFO
      rs.R = rf.read();
    }
    pair_t p = odd ? f1.read() : f0.read();
    ow.write(quant8(p(127, 0), p(255, 128), rs));
    if (++j == H) {
      j = 0;
      odd = !odd;
      if (++n == c.N) {
        n = 0;
        if (++b == c.B) break;
      }
    }
  }
}

// ---------------------------------------------------------------- write
static void qz_wr(word_t* io, const QCmd& c, hls::stream<word_t>& ow, hls::stream<word_t>& sclf) {
  const ap_uint<17> Wo = c.K >> 3;
  const bool contig = c.N == 1 || c.out_pitch == Wo;
  waddr_t item = c.out, sitem = c.scl;
items:
  for (ap_uint<17> b = 0; b < c.B; b++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 16 avg = 2
#pragma HLS LOOP_FLATTEN off
    waddr_t row = item, sw = sitem;
  segs:
    for (ap_uint<25> n0 = 0; n0 < c.N; n0 += SEG) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 16 avg = 2
      const ap_uint<25> left = c.N - n0;
      const ap_uint<7> nr = left > SEG ? ap_uint<25>(SEG) : left;
      const ap_uint<7> runs = contig ? ap_uint<7>(1) : nr;
      // words of a run: one row (Wo) or the whole segment (nr * Wo, a small fabric multiplier)
      const ap_uint<24> Lm = nr * Wo;
#pragma HLS BIND_OP variable = Lm op = mul impl = fabric latency = 2
      const ap_uint<24> L = contig ? Lm : ap_uint<24>(Wo);
    runs_loop:
      for (ap_uint<7> r = 0; r < runs; r++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 64 avg = 1
#pragma HLS LOOP_FLATTEN off
      data:
        for (ap_uint<24> i = 0; i < L; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 2 max = 262144 avg = 6144
          io[row + i] = ow.read();
        }
        row += contig ? waddr_t(L) : waddr_t(c.out_pitch);
      }
      const ap_uint<5> ns = (nr + 3) >> 2;
    scales:
      for (ap_uint<5> k = 0; k < ns; k++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 16 avg = 8
        io[sw + k] = sclf.read();
      }
      sw += ns;
    }
    item += c.out_bstride;
    sitem += c.scl_bstride;
  }
}

// ---------------------------------------------------------------- the mode
// rd0 / rd1: the same DDR seen through two read bundles; io: the write bundle. Call only with
// quant_cmd_ok(c) (an invalid command can deadlock the dataflow region).
static void quant_rows(const word_t* rd0, const word_t* rd1, word_t* io, const QCmd& c) {
#pragma HLS DATAFLOW
  hls::stream<pair_t> f0("qz_f0"), f1("qz_f1");
#pragma HLS STREAM variable = f0 depth = QZ_FIFO_DEPTH
#pragma HLS STREAM variable = f1 depth = QZ_FIFO_DEPTH
#ifdef QZ_FIFO_URAM
#pragma HLS BIND_STORAGE variable = f0 type = fifo impl = uram
#pragma HLS BIND_STORAGE variable = f1 type = fifo impl = uram
#elif defined(QZ_FIFO_BRAM)
#pragma HLS BIND_STORAGE variable = f0 type = fifo impl = bram
#pragma HLS BIND_STORAGE variable = f1 type = fifo impl = bram
#else
#pragma HLS BIND_STORAGE variable = f0 type = fifo impl = lutram
#pragma HLS BIND_STORAGE variable = f1 type = fifo impl = lutram
#endif
  hls::stream<abits_t> am0("qz_am0"), am1("qz_am1");
  hls::stream<SeedS> sdf("qz_sdf");
  hls::stream<ap_uint<32>> rpf("qz_rpf");
  hls::stream<ap_uint<26>> rf("qz_rf");
  hls::stream<word_t> sclf("qz_sclf"), ow("qz_ow");
  // rpf bypasses rc: it holds the rows rc is working on (<= NE + 2 deep sdf/rf) plus slack.
  // sclf: wr takes a segment's SEG / 4 scale words after the segment's data, while sc may already be
  // a few dozen rows further (bounded by rpf + rf + the rows in ow).
#pragma HLS STREAM variable = am0 depth = 4
#pragma HLS STREAM variable = am1 depth = 4
#pragma HLS STREAM variable = sdf depth = 2
#pragma HLS STREAM variable = rf depth = 2
#pragma HLS STREAM variable = rpf depth = 16
#pragma HLS STREAM variable = sclf depth = 64
#pragma HLS STREAM variable = ow depth = 32
#pragma HLS BIND_STORAGE variable = sclf type = fifo impl = lutram
#pragma HLS BIND_STORAGE variable = ow type = fifo impl = lutram
  qz_rd<0>(rd0, c, f0, am0);
  qz_rd<1>(rd1, c, f1, am1);
  qz_sc(c, am0, am1, sdf, rpf, sclf);
  qz_rc(c, sdf, rf);
  qz_qt(c, f0, f1, rpf, rf, ow);
  qz_wr(io, c, ow, sclf);
}

}  // namespace qz
