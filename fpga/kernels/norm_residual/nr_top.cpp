// Standalone Vitis HLS top of the NORM mode, driven the way float_top's sequencer drives it: the record (256 bytes,
// contract 1.0) is fetched over io at the list base, decoded and checked with contract.h's decode + validate_hw,
// then nr::norm_mode is called from one dispatch site. Bundle names and pragmas are float_top's (hp2 read-only,
// io = xmem read + ymem write, s_axi_control M / K / N); the other hp ports are not used by NORM.
//   M = record index (list mode: 0xA5 << 24 | index; only the index is used here), K = list base >> 4.
//   return 0 = done; (status << 8) | ME code otherwise (ST_BAD_* from decode, ST_MODE_ERROR | ME_* from validate_hw).
// The g++ testbench (tb_nr.cpp) calls this same function on whole DDR images.
#include "norm_residual.h"

int nr_top(const nr::word_t* w2, const nr::word_t* xmem, nr::word_t* ymem, int M, int K, int N) {
#pragma HLS INTERFACE mode = m_axi port = w2 offset = slave bundle = hp2 max_read_burst_length = 256 num_read_outstanding = 16 depth = 1048576
#pragma HLS INTERFACE mode = m_axi port = xmem offset = slave bundle = io max_read_burst_length = 256 depth = 1048576
#pragma HLS INTERFACE mode = m_axi port = ymem offset = slave bundle = io max_write_burst_length = 256 depth = 1048576
#pragma HLS INTERFACE mode = s_axilite port = M
#pragma HLS INTERFACE mode = s_axilite port = K
#pragma HLS INTERFACE mode = s_axilite port = N
#pragma HLS INTERFACE mode = s_axilite port = return
  (void)N;
  const nr::waddr_t rec = nr::waddr_t((uint32_t)K) + nr::waddr_t((uint32_t)M & ctr::MAX_INDEX) * ctr::REC_WORDS;
  uint32_t f[ctr::REC_FIELDS];
#pragma HLS ARRAY_PARTITION variable = f complete
fetch:
  for (unsigned w = 0; w < ctr::REC_WORDS; w++) {
#pragma HLS PIPELINE II = 1
    const nr::word_t v = xmem[rec + w];
    for (int j = 0; j < 4; j++) {
#pragma HLS UNROLL
      f[4 * w + j] = (uint32_t)v.range(32 * j + 31, 32 * j);
    }
  }
  ctr::Cmd c;
  const uint32_t st = ctr::decode(f, c);
  if (st) return (int)(st << 8);
  if (c.op != ctr::OP_NORM) return (int)(ctr::ST_BAD_OPCODE << 8);
  const uint32_t me = ctr::validate_hw(c);
  if (me) return (int)((ctr::ST_MODE_ERROR << 8) | me);
  const uint32_t r = nr::norm_mode(c, w2, xmem, ymem);  // the single dispatch site
  return r ? (int)((ctr::ST_MODE_ERROR << 8) | r) : 0;
}
