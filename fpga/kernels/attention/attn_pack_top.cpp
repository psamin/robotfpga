// Standalone Vitis HLS top of the attention operand packer (attn_pack.h), for csynth. In float_top the same core
// runs on the existing io bundle (src = xmem, dst = ymem) as a mode: README.md "Integration".
//   cd fpga/kernels/attention && vitis_hls -f run_attn_hls.tcl
#include "attn_pack.h"

using namespace attnpack;

int attn_pack_top(const Word* src, Word* dst, uint32_t desc_off, uint32_t n_desc) {
#pragma HLS INTERFACE mode = m_axi port = src offset = slave bundle = io max_read_burst_length = 256 num_read_outstanding = 16 depth = 1048576
#pragma HLS INTERFACE mode = m_axi port = dst offset = slave bundle = io max_write_burst_length = 256 num_write_outstanding = 16 depth = 1048576
#pragma HLS INTERFACE mode = s_axilite port = desc_off
#pragma HLS INTERFACE mode = s_axilite port = n_desc
#pragma HLS INTERFACE mode = s_axilite port = return
  return attn_pack(src, dst, desc_off, n_desc);
}
