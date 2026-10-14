// Vitis HLS top level for the DDR weight-streaming engine (issue #148).
// All large buffers are m_axi masters into DDR (128-bit for weights); control via AXI-Lite.
//
//   mode 0: policy_stream  (table, n_layers, wmem, bmem, pkt -> out)
//   mode 1: linear_stream  (wmem, xmem -> ymem, M, K, N)
#include "stream_engine.h"

using namespace stream;

int stream_top(int mode, const LayerDesc* table, int n_layers, const Word* wmem, const int32_t* bmem,
               const uint8_t* pkt, int8_t* out, const int8_t* xmem, int32_t* ymem, int M, int K, int N) {
#pragma HLS INTERFACE mode = m_axi port = table offset = slave bundle = ctl depth = 16
#pragma HLS INTERFACE mode = m_axi port = wmem offset = slave bundle = wts max_read_burst_length = 256 num_read_outstanding = 8 depth = 1048576
#pragma HLS INTERFACE mode = m_axi port = bmem offset = slave bundle = ctl depth = 4096
#pragma HLS INTERFACE mode = m_axi port = pkt offset = slave bundle = io depth = 27658
#pragma HLS INTERFACE mode = m_axi port = out offset = slave bundle = io depth = 48
#pragma HLS INTERFACE mode = m_axi port = xmem offset = slave bundle = io depth = 262144
#pragma HLS INTERFACE mode = m_axi port = ymem offset = slave bundle = io depth = 262144
#pragma HLS INTERFACE mode = s_axilite port = mode
#pragma HLS INTERFACE mode = s_axilite port = n_layers
#pragma HLS INTERFACE mode = s_axilite port = M
#pragma HLS INTERFACE mode = s_axilite port = K
#pragma HLS INTERFACE mode = s_axilite port = N
#pragma HLS INTERFACE mode = s_axilite port = return
  if (mode == 0) return policy_stream(table, n_layers, wmem, bmem, pkt, out);
  if (mode == 1) return linear_stream(wmem, xmem, ymem, M, K, N);
  return 2;
}
