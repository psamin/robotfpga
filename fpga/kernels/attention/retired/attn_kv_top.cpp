// Standalone csynth top of the QUANT_W / QUANT_WT mode (attn_kv.h): the io bundle exactly as float_top's (xmem read,
// ymem write, 128-bit, bursts <= 256, 64-bit addresses), one 256-byte record fetched from io and decoded with
// ctr::decode, then the mode function (its own instance in the report: attn::quant_kv, INLINE off). In float_top the
// sequencer does the fetch, decode and validate_hw, and calls attn::quant_kv from its dispatch switch.
//   cd fpga/kernels/attention && vitis_hls -f run_attn_hls.tcl
#include "attn_kv.h"

using attn::Word;

int attn_kv_top(const Word* xmem, Word* ymem, unsigned long long rec) {
#pragma HLS INTERFACE mode = m_axi port = xmem offset = slave bundle = io max_read_burst_length = 256 depth = 1048576
#pragma HLS INTERFACE mode = m_axi port = ymem offset = slave bundle = io max_write_burst_length = 256 depth = 1048576
#pragma HLS INTERFACE mode = s_axilite port = rec
#pragma HLS INTERFACE mode = s_axilite port = return
  uint32_t f[ctr::REC_FIELDS];
#pragma HLS ARRAY_PARTITION variable = f complete
fetch:
  for (uint32_t w = 0; w < ctr::REC_WORDS; w++) {
#pragma HLS PIPELINE II = 1
    const Word v = xmem[(rec >> 4) + w];
    for (int j = 0; j < 4; j++) {
#pragma HLS UNROLL
      f[4 * w + j] = (uint32_t)v.range(32 * j + 31, 32 * j);
    }
  }
  ctr::Cmd c;
  const uint32_t st = ctr::decode(f, c);
  if (st) return (int)(0x100 | st);
  if (c.op != ctr::OP_QUANT_W && c.op != ctr::OP_QUANT_WT) return 0x200;
  return (int)attn::quant_kv(c, xmem, ymem);
}
