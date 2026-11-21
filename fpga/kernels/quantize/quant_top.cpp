// Standalone Vitis HLS top of the row unit's QUANT_X / QUANT_W / QUANT_WT / COPY modes (quant_mode.h), for
// csynth only: it fetches one 256-byte contract record over io (as the sequencer does), decodes it with
// ctr::decode and calls qz::row_mode, so the csynth report shows row_mode as its own instance, exactly the
// function the sequencer's dispatch switch calls. validate_hw is the sequencer's (not instantiated here: the
// host tb runs every case through ctr::validate first).
//
// Bundles as float_top's io: xmem (read) and ymem (write) on m_axi "io", 128-bit, bursts <= 256, 64-bit
// addresses; s_axi_control. Difference to float_top.cpp: latency = 64 on io (no port change; float_top must
// add it, else the scheduler assumes 0 and every pipelined read / write unit waits for its own DDR round trip).
#include "quant_mode.h"

uint32_t quant_top(const qz::word_t* xmem, qz::word_t* ymem, unsigned long long rec) {
#pragma HLS INTERFACE mode = m_axi port = xmem offset = slave bundle = io max_read_burst_length = 256 num_read_outstanding = 16 latency = 64 depth = 1048576
#pragma HLS INTERFACE mode = m_axi port = ymem offset = slave bundle = io max_write_burst_length = 256 num_write_outstanding = 16 latency = 64 depth = 1048576
#pragma HLS INTERFACE mode = s_axilite port = rec
#pragma HLS INTERFACE mode = s_axilite port = return
  uint32_t f[ctr::REC_FIELDS];
#pragma HLS ARRAY_PARTITION variable = f complete
fetch:
  for (unsigned w = 0; w < ctr::REC_WORDS; w++) {
#pragma HLS PIPELINE II = 1
    const qz::word_t v = xmem[(rec >> 4) + w];
    for (int j = 0; j < 4; j++) {
#pragma HLS UNROLL
      f[4 * w + j] = (uint32_t)v.range(32 * j + 31, 32 * j);
    }
  }
  ctr::Cmd c;
  const uint32_t st = ctr::decode(f, c);
  if (st != ctr::ST_END) return 0x100 | st;
  return qz::row_mode(c, xmem, ymem);
}
