// Vitis HLS top for the full-precision (BF16 weights, FP32 math) linear engine, issue #148.
// Four weight bundles map to S_AXI_HP0..HP3_FPD in the block design; X/Y share a fifth bundle
// (an HPC port, 128-bit words). Target 300 MHz.   cd fpga/hls && vitis_hls -f run_float_hls.tcl
#include "float_linear.h"

using namespace fstream;

int float_top(const WordBF* w0, const WordBF* w1, const WordBF* w2, const WordBF* w3, const WordF4* xmem,
              WordF4* ymem, int M, int K, int N) {
#pragma HLS INTERFACE mode = m_axi port = w0 offset = slave bundle = hp0 max_read_burst_length = 256 num_read_outstanding = 16 depth = 1048576
#pragma HLS INTERFACE mode = m_axi port = w1 offset = slave bundle = hp1 max_read_burst_length = 256 num_read_outstanding = 16 depth = 1048576
#pragma HLS INTERFACE mode = m_axi port = w2 offset = slave bundle = hp2 max_read_burst_length = 256 num_read_outstanding = 16 depth = 1048576
#pragma HLS INTERFACE mode = m_axi port = w3 offset = slave bundle = hp3 max_read_burst_length = 256 num_read_outstanding = 16 depth = 1048576
#pragma HLS INTERFACE mode = m_axi port = xmem offset = slave bundle = io max_read_burst_length = 256 depth = 1048576
#pragma HLS INTERFACE mode = m_axi port = ymem offset = slave bundle = io max_write_burst_length = 256 depth = 1048576
#pragma HLS INTERFACE mode = s_axilite port = M
#pragma HLS INTERFACE mode = s_axilite port = K
#pragma HLS INTERFACE mode = s_axilite port = N
#pragma HLS INTERFACE mode = s_axilite port = return
  // Token a of every block in its own memory (dim 1), 4 consecutive k in 4 memories (dim 3) so the
  // 128-bit X loads write 4 values per cycle.
  static float xbuf[TP][NB][K_MAX];
#pragma HLS BIND_STORAGE variable = xbuf type = ram_2p impl = uram
#pragma HLS ARRAY_PARTITION variable = xbuf complete dim = 1
#pragma HLS ARRAY_PARTITION variable = xbuf cyclic factor = 4 dim = 3
  return linear_bf16_core(w0, w1, w2, w3, xmem, ymem, M, K, N, xbuf);
}
