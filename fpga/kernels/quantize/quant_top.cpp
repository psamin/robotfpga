// Standalone Vitis HLS top of the int16 per-token quantizer (quant_mode.h), the future QUANT_X mode
// of float_top: same bundle names and widths as float_top (hp0 / hp1 read, io write, s_axi_control).
//
// Registers (all word = 16-byte units; hp0 and hp1 must be given the same base address):
//   B, N, K                      items, rows per item, values per row
//   in_off, in_pitch, in_bstride     input FP32 rows, relative to hp0 / hp1
//   out_off, out_pitch, out_bstride  int16 rows, relative to io
//   scl_off, scl_bstride             FP32 scale vectors, relative to io
// Returns 0, or 1 (nothing read or written) when the command breaks quant_cmd_ok's constraints.
#include "quant_mode.h"

int quant_top(const qz::word_t* hp0, const qz::word_t* hp1, qz::word_t* io, unsigned B, unsigned N, unsigned K,
              unsigned in_off, unsigned in_pitch, unsigned in_bstride, unsigned out_off, unsigned out_pitch,
              unsigned out_bstride, unsigned scl_off, unsigned scl_bstride) {
#pragma HLS INTERFACE mode = m_axi port = hp0 offset = slave bundle = hp0 max_read_burst_length = 256 num_read_outstanding = 16 depth = 1048576
#pragma HLS INTERFACE mode = m_axi port = hp1 offset = slave bundle = hp1 max_read_burst_length = 256 num_read_outstanding = 16 depth = 1048576
#pragma HLS INTERFACE mode = m_axi port = io offset = slave bundle = io max_write_burst_length = 256 depth = 1048576
#pragma HLS INTERFACE mode = s_axilite port = B
#pragma HLS INTERFACE mode = s_axilite port = N
#pragma HLS INTERFACE mode = s_axilite port = K
#pragma HLS INTERFACE mode = s_axilite port = in_off
#pragma HLS INTERFACE mode = s_axilite port = in_pitch
#pragma HLS INTERFACE mode = s_axilite port = in_bstride
#pragma HLS INTERFACE mode = s_axilite port = out_off
#pragma HLS INTERFACE mode = s_axilite port = out_pitch
#pragma HLS INTERFACE mode = s_axilite port = out_bstride
#pragma HLS INTERFACE mode = s_axilite port = scl_off
#pragma HLS INTERFACE mode = s_axilite port = scl_bstride
#pragma HLS INTERFACE mode = s_axilite port = return
  if (B == 0 || B > 65535 || N == 0 || N > 0xFFFFFF || K > qz::K_MAX) return 1;
  qz::QCmd c;
  c.B = B;
  c.N = N;
  c.K = K;
  c.in = in_off;
  c.in_pitch = in_pitch;
  c.in_bstride = in_bstride;
  c.out = out_off;
  c.out_pitch = out_pitch;
  c.out_bstride = out_bstride;
  c.scl = scl_off;
  c.scl_bstride = scl_bstride;
  if (!qz::quant_cmd_ok(c)) return 1;
  qz::quant_rows(hp0, hp1, io, c);
  return 0;
}
