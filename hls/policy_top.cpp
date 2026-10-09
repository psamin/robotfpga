// Vitis HLS top level: AXI-Stream in/out for the AXI DMA, AXI-Lite for control.
// Needs the Xilinx headers, so it only builds inside Vitis HLS (see run_hls.tcl).
//
// Register interface (s_axi_control; offsets are in the generated xpolicy_top_hw.h):
//   CTRL       ap_start / ap_done / ap_idle (standard HLS block control)
//   mode       0 = load weights.bin (568,240 bytes), 1 = one inference
//   shifts_lo  s1 | s2 << 8 | s3 << 16 | s4 << 24   (from manifest.json)
//   shifts_hi  s5 | s6 << 8 | s7 << 16 | s8 << 24
//   version    read-only, policy::VERSION
//   ap_return  status: 0 ok, 1 bad shift, 2 TLAST in the wrong place, 3 bad mode
//
// Inference: 27,658 bytes in (TLAST on the last), 48 bytes out (TLAST on the last).
#include <ap_axi_sdata.h>
#include <hls_stream.h>
#include <stdint.h>

typedef ap_axiu<8, 0, 0, 0> axis8_t;

static uint8_t read_byte(hls::stream<axis8_t>& s, bool& last) {
#pragma HLS INLINE
  axis8_t beat = s.read();
  last = beat.last;
  return beat.data;
}

static void write_byte(hls::stream<axis8_t>& s, uint8_t v, bool last) {
#pragma HLS INLINE
  axis8_t beat;
  beat.data = v;
  beat.keep = -1;
  beat.strb = -1;
  beat.last = last;
  s.write(beat);
}

#include "policy_kernel.h"

int policy_top(hls::stream<axis8_t>& in_stream, hls::stream<axis8_t>& out_stream, int mode,
               uint32_t shifts_lo, uint32_t shifts_hi, uint32_t* version) {
#pragma HLS INTERFACE mode = axis port = in_stream
#pragma HLS INTERFACE mode = axis port = out_stream
#pragma HLS INTERFACE mode = s_axilite port = mode
#pragma HLS INTERFACE mode = s_axilite port = shifts_lo
#pragma HLS INTERFACE mode = s_axilite port = shifts_hi
#pragma HLS INTERFACE mode = s_axilite port = version
#pragma HLS INTERFACE mode = s_axilite port = return
  *version = policy::VERSION;
  return policy::policy_kernel(mode, in_stream, out_stream, shifts_lo, shifts_hi);
}
