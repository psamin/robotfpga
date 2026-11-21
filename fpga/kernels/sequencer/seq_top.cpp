// Standalone Vitis HLS top of the float_top sequencer (contract 1.0), for its own csynth: the exact argument list,
// bundles and INTERFACE pragmas of fpga/hls/float_top.cpp (m_axi hp0..hp3 + io, one s_axi_control, int return),
// with null units in place of the engine, row unit and softmax.
//   cd fpga/kernels/sequencer && vitis_hls -f run_seq_hls.tcl        (SEQ_VARIANT, SEQ_PERIOD: see the tcl)
// SEQ_VARIANT:
//   default: walk + cycle counter (DATAFLOW region) + 3 null units; -DSEQ_NO_UNITS: no units (the sequencer alone)
//   (variants: run_seq_hls.tcl flags SEQ_UNITS, SEQ_CYCLES, SEQ_UNROLLED, SEQ_CHECK_INLINE)
// A null unit reads one word from each port its real unit reads, XORs every field its real unit consumes and
// writes one io word: enough that HLS keeps every decoded field and the port muxing, and cheap enough to subtract
// (its instance is reported on its own in csynth.rpt).
#include "seq_core.h"

#ifndef __SYNTHESIS__
namespace seq {
uint64_t g_host_cycles = 0;
}
#endif

#ifdef __VITIS_HLS__
typedef ap_uint<128> WPort;
typedef ap_uint<128> IoWord;
#else
typedef seq::Word WPort;
typedef seq::Word IoWord;
#endif

template <int UNIT>
static uint32_t null_unit(const ctr::Cmd& c, const WPort* w0, const WPort* w1, const WPort* w2, const WPort* w3,
                          const IoWord* xmem, IoWord* ymem) {
#pragma HLS INLINE off
  using namespace ctr;
  uint32_t x = c.op ^ c.flags ^ c.tag ^ c.M ^ c.K ^ c.N ^ c.B;
  uint32_t hi = 0;
  for (int s = 0; s < (int)NSLOTS; s++) {
#pragma HLS UNROLL
    x ^= (uint32_t)c.s[s].addr ^ c.s[s].pitch ^ c.s[s].stride;
    hi ^= (uint32_t)(c.s[s].addr >> 32);
  }
  for (int i = 0; i < (int)NPARAMS; i++) {
#pragma HLS UNROLL
    x ^= c.p[i];
  }
  const seq::Word a = seq::to_word(xmem[c.s[SLOT_A].addr >> 4]);
  uint32_t y = seq::wget(a, 0);
  if (UNIT == seq::U_GEMM) {  // weight blob on the four hp ports
    const uint64_t b = c.s[SLOT_B].addr, p = c.s[SLOT_B].pitch;
    y ^= seq::wget(seq::to_word(w0[b >> 4]), 0) ^ seq::wget(seq::to_word(w1[(b + p) >> 4]), 1) ^
         seq::wget(seq::to_word(w2[(b + 2 * p) >> 4]), 2) ^ seq::wget(seq::to_word(w3[(b + 3 * p) >> 4]), 3);
  } else {  // constants (gamma/beta, RoPE table, rank vectors) on hp0
    y ^= seq::wget(seq::to_word(w0[c.s[SLOT_E].addr >> 4]), 0);
  }
  seq::Word o;
  seq::wset(o, 0, x);
  seq::wset(o, 1, hi);
  seq::wset(o, 2, y);
  seq::wset(o, 3, UNIT);
  ymem[c.s[SLOT_C].addr >> 4] = seq::from_word<IoWord>(o);
  return ME_OK;
}

#ifdef SEQ_NO_UNITS  // variant 5: the sequencer alone (dispatch reaches no unit; ports unused but by fetch/status)
struct NullUnits {
  template <class WP, class W>
  static uint32_t gemm(const ctr::Cmd&, const WP*, const WP*, const WP*, const WP*, const W*, W*) {
    return ctr::ME_OK;
  }
  template <class WP, class W>
  static uint32_t row(const ctr::Cmd&, const WP*, const WP*, const WP*, const WP*, const W*, W*) {
    return ctr::ME_OK;
  }
  template <class WP, class W>
  static uint32_t softmax(const ctr::Cmd&, const WP*, const WP*, const WP*, const WP*, const W*, W*) {
    return ctr::ME_OK;
  }
};
#else
struct NullUnits {
  template <class WP, class W>
  static uint32_t gemm(const ctr::Cmd& c, const WP* w0, const WP* w1, const WP* w2, const WP* w3, const W* xr,
                       W* xw) {
#pragma HLS INLINE
    return null_unit<seq::U_GEMM>(c, w0, w1, w2, w3, xr, xw);
  }
  template <class WP, class W>
  static uint32_t row(const ctr::Cmd& c, const WP* w0, const WP* w1, const WP* w2, const WP* w3, const W* xr,
                      W* xw) {
#pragma HLS INLINE
    return null_unit<seq::U_ROW>(c, w0, w1, w2, w3, xr, xw);
  }
  template <class WP, class W>
  static uint32_t softmax(const ctr::Cmd& c, const WP* w0, const WP* w1, const WP* w2, const WP* w3, const W* xr,
                          W* xw) {
#pragma HLS INLINE
    return null_unit<seq::U_SMX>(c, w0, w1, w2, w3, xr, xw);
  }
};
#endif

int seq_top(const WPort* w0, const WPort* w1, const WPort* w2, const WPort* w3, const IoWord* xmem, IoWord* ymem,
            int M, int K, int N) {
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
  return (int)seq::top<NullUnits>(w0, w1, w2, w3, xmem, ymem, (uint32_t)M, (uint32_t)K, (uint32_t)N);
}
