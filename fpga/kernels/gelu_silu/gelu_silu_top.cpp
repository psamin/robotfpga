// Standalone Vitis HLS top of the gelu_silu kernel, with float_top's bus shape: four read-only 128-bit m_axi
// bundles hp0..hp3 and the 128-bit 'io' bundle (written here), plus s_axi_control. In float_top the same core is
// one more mode (README.md, "Integration"), with the offsets taken from the sequencer command; this top exists for
// csynth and for the C testbench. Offsets / pitches in 16-byte words relative to the pointers (gelu_silu.h).
//   vitis_hls -f run_hls.tcl
#include "gelu_silu.h"

int gelu_silu_top(const gs::Word* w0, const gs::Word* w1, const gs::Word* w2, const gs::Word* w3, gs::Word* ymem,
                  int mode, int N, int K, unsigned a, unsigned pa, unsigned b, unsigned pb, unsigned c, unsigned pc,
                  unsigned f) {
#pragma HLS INTERFACE mode = m_axi port = w0 offset = slave bundle = hp0 max_read_burst_length = 256 num_read_outstanding = 16 depth = 1048576
#pragma HLS INTERFACE mode = m_axi port = w1 offset = slave bundle = hp1 max_read_burst_length = 256 num_read_outstanding = 16 depth = 1048576
#pragma HLS INTERFACE mode = m_axi port = w2 offset = slave bundle = hp2 max_read_burst_length = 256 num_read_outstanding = 16 depth = 1048576
#pragma HLS INTERFACE mode = m_axi port = w3 offset = slave bundle = hp3 max_read_burst_length = 256 num_read_outstanding = 16 depth = 1048576
#pragma HLS INTERFACE mode = m_axi port = ymem offset = slave bundle = io max_write_burst_length = 256 depth = 1048576
#pragma HLS INTERFACE mode = s_axilite port = mode
#pragma HLS INTERFACE mode = s_axilite port = N
#pragma HLS INTERFACE mode = s_axilite port = K
#pragma HLS INTERFACE mode = s_axilite port = a
#pragma HLS INTERFACE mode = s_axilite port = pa
#pragma HLS INTERFACE mode = s_axilite port = b
#pragma HLS INTERFACE mode = s_axilite port = pb
#pragma HLS INTERFACE mode = s_axilite port = c
#pragma HLS INTERFACE mode = s_axilite port = pc
#pragma HLS INTERFACE mode = s_axilite port = f
#pragma HLS INTERFACE mode = s_axilite port = return
  return gs::gelu_silu_core(w0, w1, w2, w3, ymem, mode, N, K, a, pa, b, pb, c, pc, f);
}
