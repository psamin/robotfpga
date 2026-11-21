// Synthesis probe of the contract's hardware side: fetch one 256-byte record per command over io (one 16-beat
// burst), decode, validate_hw. Run: vitis_hls -f fpga/kernels/hls_probe/run_contract_hls.tcl (4.0 ns, xck26).
// The result (resources, estimated clock) is the sequencer's record cost in CONTRACT.md 4.
#include <ap_int.h>

#include "../contract.h"

typedef ap_uint<128> Word;

int contract_top(const Word* io, unsigned list_word, unsigned n, unsigned* first_bad) {
#pragma HLS INTERFACE mode = m_axi port = io offset = slave bundle = io max_read_burst_length = 256 depth = 4096
#pragma HLS INTERFACE mode = m_axi port = first_bad offset = slave bundle = io depth = 1
#pragma HLS INTERFACE mode = s_axilite port = list_word
#pragma HLS INTERFACE mode = s_axilite port = n
#pragma HLS INTERFACE mode = s_axilite port = return
  int ok = 0;
  unsigned bad = 0xFFFFFFFFu;
cmds:
  for (unsigned i = 0; i < n; i++) {
    uint32_t f[ctr::REC_FIELDS];
#pragma HLS ARRAY_PARTITION variable = f complete
  fetch:
    for (unsigned w = 0; w < ctr::REC_WORDS; w++) {
#pragma HLS PIPELINE II = 1
      const Word v = io[(uint64_t)list_word + (uint64_t)i * ctr::REC_WORDS + w];
      for (int j = 0; j < 4; j++) {
#pragma HLS UNROLL
        f[4 * w + j] = (uint32_t)v.range(32 * j + 31, 32 * j);
      }
    }
    ctr::Cmd c;
    const uint32_t st = ctr::decode(f, c);
    const uint32_t me = st ? st : ctr::validate_hw(c);
    if (me == 0)
      ok++;
    else if (bad == 0xFFFFFFFFu)
      bad = i;
  }
  *first_bad = bad;
  return ok;
}
