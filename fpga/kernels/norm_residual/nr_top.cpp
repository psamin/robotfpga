// Standalone Vitis HLS top of the norm_residual kernel (csynth; the g++ testbench calls the same function).
// Bundles as float_top: hp0 / hp1 / hp2 read-only 128-bit (A rows, B rows, gamma|beta), io 128-bit (writes:
// D sum rows, C output rows, F scales). Scalars: the Cfg fields of norm_residual.h, offsets in 128-bit words.
//   vitis_hls -f run_nr_hls.tcl
#include "norm_residual.h"

int nr_top(const nr::word_t* hp0, const nr::word_t* hp1, const nr::word_t* hp2, nr::word_t* io, unsigned flags,
           int K, int N, int B, unsigned eps, unsigned a_off, unsigned a_pitch, unsigned a_stride, unsigned b_off,
           unsigned b_pitch, unsigned b_stride, unsigned e_off, unsigned c_off, unsigned c_pitch, unsigned c_stride,
           unsigned d_off, unsigned d_pitch, unsigned d_stride, unsigned f_off, unsigned f_stride) {
#pragma HLS INTERFACE mode = m_axi port = hp0 offset = slave bundle = hp0 max_read_burst_length = 256 num_read_outstanding = 16 depth = 1048576
#pragma HLS INTERFACE mode = m_axi port = hp1 offset = slave bundle = hp1 max_read_burst_length = 256 num_read_outstanding = 16 depth = 1048576
#pragma HLS INTERFACE mode = m_axi port = hp2 offset = slave bundle = hp2 max_read_burst_length = 256 num_read_outstanding = 16 depth = 1048576
#pragma HLS INTERFACE mode = m_axi port = io offset = slave bundle = io max_write_burst_length = 256 num_write_outstanding = 16 depth = 1048576
#pragma HLS INTERFACE mode = s_axilite port = flags
#pragma HLS INTERFACE mode = s_axilite port = K
#pragma HLS INTERFACE mode = s_axilite port = N
#pragma HLS INTERFACE mode = s_axilite port = B
#pragma HLS INTERFACE mode = s_axilite port = eps
#pragma HLS INTERFACE mode = s_axilite port = a_off
#pragma HLS INTERFACE mode = s_axilite port = a_pitch
#pragma HLS INTERFACE mode = s_axilite port = a_stride
#pragma HLS INTERFACE mode = s_axilite port = b_off
#pragma HLS INTERFACE mode = s_axilite port = b_pitch
#pragma HLS INTERFACE mode = s_axilite port = b_stride
#pragma HLS INTERFACE mode = s_axilite port = e_off
#pragma HLS INTERFACE mode = s_axilite port = c_off
#pragma HLS INTERFACE mode = s_axilite port = c_pitch
#pragma HLS INTERFACE mode = s_axilite port = c_stride
#pragma HLS INTERFACE mode = s_axilite port = d_off
#pragma HLS INTERFACE mode = s_axilite port = d_pitch
#pragma HLS INTERFACE mode = s_axilite port = d_stride
#pragma HLS INTERFACE mode = s_axilite port = f_off
#pragma HLS INTERFACE mode = s_axilite port = f_stride
#pragma HLS INTERFACE mode = s_axilite port = return
  nr::Cfg c;
  c.flags = flags;
  c.K = K;
  c.N = N;
  c.B = B;
  c.eps = eps;
  c.a_off = a_off;
  c.a_pitch = a_pitch;
  c.a_stride = a_stride;
  c.b_off = b_off;
  c.b_pitch = b_pitch;
  c.b_stride = b_stride;
  c.e_off = e_off;
  c.c_off = c_off;
  c.c_pitch = c_pitch;
  c.c_stride = c_stride;
  c.d_off = d_off;
  c.d_pitch = d_pitch;
  c.d_stride = d_stride;
  c.f_off = f_off;
  c.f_stride = f_stride;
  if (nr::check_cfg(c)) return 1;
  nr::set_derived(c);
  nr::norm_residual_core(hp0, hp1, hp2, io, c);
  return 0;
}
