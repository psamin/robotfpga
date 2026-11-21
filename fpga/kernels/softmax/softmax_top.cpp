// Standalone top of the SOFTMAX unit for csynth and the host testbench. Same argument list, bundles and pragmas
// as fpga/hls/float_top.cpp (w0..w3 = hp0..hp3 read-only, xmem / ymem = io, M K N on s_axi_control); the unit
// uses xmem and ymem only. This top is a one-command stand-in for the sequencer (contract 3.1):
//   M = 0xA5 << 24 | index, K = list base >> 4, N unused; pointer registers 0 (absolute addresses).
// It fetches record `index` (256 bytes) over xmem, runs ctr::decode and ctr::validate_hw, and dispatches the
// SOFTMAX mode function from one call site. Return value: ST_NO_LEGACY without 0xA5; otherwise
// (me << 16) | st with st = decode's ST_* (ST_END = 0 when the record decodes, ST_MODE_ERROR when validate_hw
// or the mode fails, ST_BAD_OPCODE for an opcode other than SOFTMAX) and me = the ME_* code.
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
  (void)w0;
  (void)w1;
  (void)w2;
  (void)w3;
  (void)N;
  const uint32_t m = (uint32_t)M;
  if ((m >> 24) != ctr::MODE_MAGIC) return (int)ctr::ST_NO_LEGACY;
  const uint64_t rec = (uint64_t)(uint32_t)K + (uint64_t)(m & ctr::MAX_INDEX) * ctr::REC_WORDS;
  uint32_t f[ctr::REC_FIELDS];
#pragma HLS ARRAY_PARTITION variable = f complete
fetch:
  for (uint32_t w = 0; w < ctr::REC_WORDS; w++) {
#pragma HLS PIPELINE II = 1
    const smx::Word v = xmem[rec + w];
    for (int j = 0; j < 4; j++) {
#pragma HLS UNROLL
      f[4 * w + j] = (uint32_t)v.range(32 * j + 31, 32 * j);
    }
  }
  ctr::Cmd c;
  const uint32_t st = ctr::decode(f, c);
  if (st) return (int)st;
  if (c.op != ctr::OP_SOFTMAX) return (int)ctr::ST_BAD_OPCODE;
  uint32_t me = ctr::validate_hw(c);
  if (me == ctr::ME_OK) me = smx::softmax_mode(c, xmem, ymem);  // the dispatch site
  return me ? (int)((me << 16) | ctr::ST_MODE_ERROR) : (int)ctr::ST_END;
}
