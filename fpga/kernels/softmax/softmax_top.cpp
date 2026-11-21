// Standalone Vitis HLS top of the softmax kernel, for csynth only. Same argument list, bundles and pragmas as
// fpga/hls/float_top.cpp (w0..w3 = hp0..hp3 read-only, xmem / ymem = io, M K N on s_axi_control), so the same
// s_axi_control map and ports; this top uses w0 (hp0), w1 (hp1) and ymem (io) only.
//   M = 0x5F000040 (M[31:24] = 0x5F kernel magic, M[7:0] = SOFTMAX 0x40), K = word index of the 128-byte SOFTMAX record (byte address >> 4) in w0's view,
//   N unused. Every address in the record is absolute (word index = byte address >> 4 from the pointer
//   registers, which the host sets to 0 as in the sequencer). Return value: the Status of smx::decode().
//   cd fpga/kernels/softmax && vitis_hls -f run_softmax_hls.tcl
#include "softmax_core.h"

int softmax_top(const smx::Word* w0, const smx::Word* w1, const smx::Word* w2, const smx::Word* w3,
                const smx::Word* xmem, smx::Word* ymem, int M, int K, int N) {
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
  (void)w2;
  (void)w3;
  (void)xmem;
  (void)N;
  if ((unsigned)M != 0x5F000040u) return smx::ST_OPCODE;
  return smx::softmax_record(w0, w1, ymem, (smx::WAddr)(unsigned)K);
}
