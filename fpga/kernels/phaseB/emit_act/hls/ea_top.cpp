// csynth wrappers of the two GEMM processes emit_act changes, built against either tree (run_ea_hls.tcl):
//   base:    fpga/hls/gemm_contract.h as snapshotted (snap/, contract 1.0)
//   patched: patched/hls/gemm_contract.h (P_ACT) + emit_act.h
// EA_TOP=dq: fgemm::dequant_ep alone (the epilogue + activation datapath), its FIFOs as ap_fifo ports.
// EA_TOP=st: fgemm::store alone (row-major Y over an m_axi port).
// The Item comes in as scalars (s_axilite); the processes are INLINE off inside the wrapper, so the report's
// instance rows are exactly the merged top's dequant_ep / store.
#include "gemm_contract.h"

using namespace fgemm;

#if EA_TOP_DQ
void ea_top(Fifo16<IoWord>& srow_s, Fifo16<IoWord>& bias_s, Fifo16<IoWord>& stok_s, Fifo16<EWord>& es,
            Fifo16<IoWord>& ds, Fifo16<IoWord>& ys, int N, int groups, int nblk, int ntiles, bool bias, bool addd,
            int act) {
#pragma HLS INTERFACE mode = ap_fifo port = srow_s
#pragma HLS INTERFACE mode = ap_fifo port = bias_s
#pragma HLS INTERFACE mode = ap_fifo port = stok_s
#pragma HLS INTERFACE mode = ap_fifo port = es
#pragma HLS INTERFACE mode = ap_fifo port = ds
#pragma HLS INTERFACE mode = ap_fifo port = ys
#pragma HLS INTERFACE mode = s_axilite port = N
#pragma HLS INTERFACE mode = s_axilite port = groups
#pragma HLS INTERFACE mode = s_axilite port = nblk
#pragma HLS INTERFACE mode = s_axilite port = ntiles
#pragma HLS INTERFACE mode = s_axilite port = bias
#pragma HLS INTERFACE mode = s_axilite port = addd
#pragma HLS INTERFACE mode = s_axilite port = act
#pragma HLS INTERFACE mode = s_axilite port = return
  Item it = Item();
  it.N = N;
  it.groups = groups;
  it.nblk = nblk;
  it.ntiles = ntiles;
  it.bias = bias;
  it.addd = addd;
#if EA_PATCHED
  it.act = act;
  it.yu = act == ea::ACT_SILU_MUL ? YU / 2 : YU;
#else
  (void)act;
#endif
  dequant_ep(srow_s, bias_s, stok_s, es, ds, ys, it);
}
#else
void ea_top(Fifo16<IoWord>& ys, IoWord* io, uint64_t ya, uint64_t yp, uint64_t yR, uint64_t yP1, int N, int groups,
            int ntiles, int act) {
#pragma HLS INTERFACE mode = ap_fifo port = ys
#pragma HLS INTERFACE mode = m_axi port = io offset = slave bundle = io max_write_burst_length = 256 depth = 1048576
#pragma HLS INTERFACE mode = s_axilite port = ya
#pragma HLS INTERFACE mode = s_axilite port = yp
#pragma HLS INTERFACE mode = s_axilite port = yR
#pragma HLS INTERFACE mode = s_axilite port = yP1
#pragma HLS INTERFACE mode = s_axilite port = N
#pragma HLS INTERFACE mode = s_axilite port = groups
#pragma HLS INTERFACE mode = s_axilite port = ntiles
#pragma HLS INTERFACE mode = s_axilite port = act
#pragma HLS INTERFACE mode = s_axilite port = return
  Item it = Item();
  it.ya = ya;
  it.yp = yp;
  it.yR = yR;
  it.yP1 = yP1;
  it.N = N;
  it.groups = groups;
  it.ntiles = ntiles;
#if EA_PATCHED
  it.act = act;
  it.yu = act == ea::ACT_SILU_MUL ? YU / 2 : YU;
#else
  (void)act;
#endif
  store(ys, io, it);
}
#endif
