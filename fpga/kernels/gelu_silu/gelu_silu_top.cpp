// Standalone Vitis HLS top of the GELU / SILU_MUL mode, for csynth and the testbench's pointer instantiation.
// It has only float_top's io bundle (xmem read, ymem write; the mode uses no hp port) and takes the record fields the
// mode reads as s_axi_control scalars. In float_top the sequencer decodes the 256-byte record (ctr::decode),
// checks it (ctr::validate_hw) and calls gs::act_mode from its dispatch switch; this top only builds the ctr::Cmd.
//   vitis_hls -f run_hls.tcl
#include "gelu_silu.h"

uint32_t gelu_silu_top(const gs::Word* xmem, gs::Word* ymem, uint32_t opflags, uint32_t K, uint32_t N, uint32_t B,
                       uint64_t a, uint32_t pa, uint32_t sa, uint64_t b, uint32_t pb, uint32_t sb, uint64_t c, uint32_t pc,
                       uint32_t sc, uint64_t f, uint32_t sf, uint32_t p4, uint32_t p5) {
#pragma HLS INTERFACE mode = m_axi port = xmem offset = slave bundle = io max_read_burst_length = 256 depth = 1048576
#pragma HLS INTERFACE mode = m_axi port = ymem offset = slave bundle = io max_write_burst_length = 256 depth = 1048576
#pragma HLS INTERFACE mode = s_axilite port = opflags
#pragma HLS INTERFACE mode = s_axilite port = K
#pragma HLS INTERFACE mode = s_axilite port = N
#pragma HLS INTERFACE mode = s_axilite port = B
#pragma HLS INTERFACE mode = s_axilite port = a
#pragma HLS INTERFACE mode = s_axilite port = pa
#pragma HLS INTERFACE mode = s_axilite port = sa
#pragma HLS INTERFACE mode = s_axilite port = b
#pragma HLS INTERFACE mode = s_axilite port = pb
#pragma HLS INTERFACE mode = s_axilite port = sb
#pragma HLS INTERFACE mode = s_axilite port = c
#pragma HLS INTERFACE mode = s_axilite port = pc
#pragma HLS INTERFACE mode = s_axilite port = sc
#pragma HLS INTERFACE mode = s_axilite port = f
#pragma HLS INTERFACE mode = s_axilite port = sf
#pragma HLS INTERFACE mode = s_axilite port = p4
#pragma HLS INTERFACE mode = s_axilite port = p5
#pragma HLS INTERFACE mode = s_axilite port = return
  ctr::Cmd cmd = {};
  cmd.op = opflags & 0xFF;
  cmd.flags = opflags >> 16;
  cmd.K = K;
  cmd.N = N;
  cmd.B = B == 0 ? 1 : B;
  cmd.s[ctr::SLOT_A] = ctr::Slot{a, pa, sa};
  cmd.s[ctr::SLOT_B] = ctr::Slot{b, pb, sb};
  cmd.s[ctr::SLOT_C] = ctr::Slot{c, pc, sc};
  cmd.s[ctr::SLOT_F] = ctr::Slot{f, 0, sf};
  cmd.p[ctr::P_SHIFT] = p4;
  cmd.p[ctr::P_NP] = p5;
  return gs::act_mode(cmd, xmem, ymem);
}
