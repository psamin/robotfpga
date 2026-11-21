// Standalone Vitis HLS top of the ROPE mode, for csynth only (the C model is checked by tb_rope.cpp with g++).
// It has float_top's bundles and pragmas for the ports the mode uses: w0 on hp0 (read only, the RoPE table) and
// xmem / ymem on io. It fetches one 256-byte contract record at word index K (as the sequencer does, list base >> 4),
// decodes it with ctr::decode and calls rope::rope_mode once, as the sequencer's dispatch switch does
// (`case ctr::OP_ROPE: return rope::rope_mode(c, w0, xmem, ymem);`). The record fetch, decode and the m_axi /
// s_axilite adapters belong to the sequencer and the top; the mode's own cost is the rope_mode instance in
// csynth.rpt.
#include "rope_mode.h"

using rope::Word;

int rope_top(const Word* w0, const Word* xmem, Word* ymem, unsigned K) {
#pragma HLS INTERFACE mode = m_axi port = w0 offset = slave bundle = hp0 max_read_burst_length = 256 num_read_outstanding = 16 depth = 1048576
#pragma HLS INTERFACE mode = m_axi port = xmem offset = slave bundle = io max_read_burst_length = 256 depth = 1048576
#pragma HLS INTERFACE mode = m_axi port = ymem offset = slave bundle = io max_write_burst_length = 256 depth = 1048576
#pragma HLS INTERFACE mode = s_axilite port = K
#pragma HLS INTERFACE mode = s_axilite port = return
  uint32_t f[ctr::REC_FIELDS];
#pragma HLS ARRAY_PARTITION variable = f complete
fetch:
  for (int i = 0; i < (int)ctr::REC_WORDS; i++) {
#pragma HLS PIPELINE II = 1
    const Word w = xmem[(ap_uint<36>)K + i];
    for (int j = 0; j < 4; j++) {
#pragma HLS UNROLL
      f[4 * i + j] = (uint32_t)w.range(32 * j + 31, 32 * j);
    }
  }
  ctr::Cmd c;
  const uint32_t st = ctr::decode(f, c);
  if (st != ctr::ST_END) return (int)st;
  return (int)rope::rope_mode(c, w0, xmem, ymem);
}
