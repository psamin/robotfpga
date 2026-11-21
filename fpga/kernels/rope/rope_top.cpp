// Standalone Vitis HLS top of the RoPE kernel (csynth only; the core is rope_core.h, README.md has the
// register map and the DDR layouts). Ports as float_top has them: x, y, s on the 128-bit m_axi bundle "io",
// the cos/sin table on the read-only 128-bit bundle "hp0" (the weight port the engine leaves idle in this
// mode), scalars on s_axi_control. Word offsets throughout (1 word = 16 bytes).
//   x  : row (t, h) at x[t * x_pitch + 16 h .. + 15], FP32, natural head_dim order
//   cs : T rows x 16 words, row t = the table row of token t's position (int32 slots, Q1.17 -cos | -sin)
//   y  : MODE_F32: row (t, h) at y[t * y_pitch + 16 h] (y == x and y_pitch == x_pitch: in place)
//        MODE_INT16: int16 row (t, h) at y[8 (t H + h) .. + 7] (contiguous, token-major)
//   s  : MODE_INT16 only: FP32 scale of row (t, h) in lane t % 4 of s[h * ceil(T/4) + t / 4]
#include "rope_core.h"

using namespace rope;

// Reads T token blocks of bw = H * 16 words at x[t * pitch]; one flat burst when the blocks are contiguous.
// (n = T * bw is passed in; offsets advance by addition, so no multiplier sits in the address path.)
static void read_rows(const Word* x, hls::stream<Word>& xs, int T, int bw, int n, int pitch) {
  if (pitch == bw) {
  rd_flat:
    for (int i = 0; i < n; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT max = 20000
      xs.write(x[i]);
    }
    return;
  }
  // narrow counters: the flattened trip count T * bw is a 16 x 10-bit product, not a 32 x 32 one
  const ap_uint<16> T16 = T;
  const ap_uint<10> bw10 = bw;
  int base = 0;
rd_tok:
  for (ap_uint<16> t = 0; t < T16; t++) {
#pragma HLS LOOP_TRIPCOUNT max = 128
  rd_row:
    for (ap_uint<10> i = 0; i < bw10; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT max = 256
      xs.write(x[base + i]);
    }
    base += pitch;
  }
}

static void read_tab(const Word* cs, hls::stream<Word>& out, int n) {
rd_tab:
  for (int i = 0; i < n; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT max = 2048
    out.write(cs[i]);
  }
}

static void write_out(hls::stream<Word>& qs, hls::stream<Word>& ss, Word* y, Word* s, int T, int bw, int rows,
                      int nsw, int y_pitch, int mode) {
  if (mode == MODE_F32) {
    if (y_pitch == bw) {
    wr_flat:
      for (int i = 0; i < rows * WORDS; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT max = 20000
        y[i] = qs.read();
      }
      return;
    }
    const ap_uint<16> T16 = T;
    const ap_uint<10> bw10 = bw;
    int base = 0;
  wr_tok:
    for (ap_uint<16> t = 0; t < T16; t++) {
#pragma HLS LOOP_TRIPCOUNT max = 128
    wr_row:
      for (ap_uint<10> i = 0; i < bw10; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT max = 256
        y[base + i] = qs.read();
      }
      base += y_pitch;
    }
    return;
  }
wq:
  for (int i = 0; i < rows * QWORDS; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT max = 10000
    y[i] = qs.read();
  }
ws:
  for (int i = 0; i < nsw; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT max = 512
    s[i] = ss.read();
  }
}

// Every scalar is computed by rope_top before the DATAFLOW region (keeps the region canonical).
static void rope_df(const Word* x, const Word* cs, Word* y, Word* s, int T, int H, int bw, int rows, int nx,
                    int ncs, int tq, int nsw, int x_pitch, int y_pitch, int mode) {
#pragma HLS DATAFLOW
  hls::stream<Word> xs("xs"), csr("csr"), qs("qs"), ss("ss");
  hls::stream<Rot> rs("rs");
#pragma HLS STREAM variable = xs depth = 16
#pragma HLS STREAM variable = csr depth = 16
#pragma HLS STREAM variable = rs depth = 4
#pragma HLS STREAM variable = qs depth = 16
#pragma HLS STREAM variable = ss depth = 4
#pragma HLS BIND_STORAGE variable = xs type = fifo impl = srl
#pragma HLS BIND_STORAGE variable = csr type = fifo impl = srl
#pragma HLS BIND_STORAGE variable = rs type = fifo impl = srl
#pragma HLS BIND_STORAGE variable = qs type = fifo impl = srl
#pragma HLS BIND_STORAGE variable = ss type = fifo impl = srl
  read_rows(x, xs, T, bw, nx, x_pitch);
  read_tab(cs, csr, ncs);
  rot_stage(xs, csr, rs, rows, H);
  out_stage(rs, qs, ss, rows, H, tq, nsw, mode);
  write_out(qs, ss, y, s, T, bw, rows, nsw, y_pitch, mode);
}

// T tokens (1..65535), H heads (1..63, 16 H <= 1008 words per token); MODE_INT16 needs ceil(T/4) * H <= 512.
void rope_top(const Word* x, const Word* cs, Word* y, Word* s, int T, int H, int x_pitch, int y_pitch, int mode) {
#pragma HLS INTERFACE mode = m_axi port = x offset = slave bundle = io max_read_burst_length = 256 num_read_outstanding = 16 depth = 65536
#pragma HLS INTERFACE mode = m_axi port = y offset = slave bundle = io max_write_burst_length = 256 num_write_outstanding = 16 depth = 65536
#pragma HLS INTERFACE mode = m_axi port = s offset = slave bundle = io max_write_burst_length = 256 num_write_outstanding = 16 depth = 4096
#pragma HLS INTERFACE mode = m_axi port = cs offset = slave bundle = hp0 max_read_burst_length = 256 num_read_outstanding = 16 depth = 4096
#pragma HLS INTERFACE mode = s_axilite port = T
#pragma HLS INTERFACE mode = s_axilite port = H
#pragma HLS INTERFACE mode = s_axilite port = x_pitch
#pragma HLS INTERFACE mode = s_axilite port = y_pitch
#pragma HLS INTERFACE mode = s_axilite port = mode
#pragma HLS INTERFACE mode = s_axilite port = return
  const int rows = (int)((ap_uint<16>)T * (ap_uint<6>)H);
  const int tq = (T + 3) / 4;
  const int nsw = (int)((ap_uint<14>)tq * (ap_uint<6>)H);  // scale words (MODE_INT16)
#pragma HLS BIND_OP variable = rows op = mul impl = fabric
#pragma HLS BIND_OP variable = nsw op = mul impl = fabric
  rope_df(x, cs, y, s, T, H, H * WORDS, rows, rows * WORDS, T * WORDS, tq, nsw, x_pitch, y_pitch, mode);
}
