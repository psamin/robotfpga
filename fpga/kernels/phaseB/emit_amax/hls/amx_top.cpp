// csynth probe of emit_amax: the GEMM unit's two changed processes (load_io: the X loader with the on-load quantizer;
// store: Y + the absmax parts) in the DATAFLOW region they sit in, with the record fetched and decoded from DDR and
// plan_item / the item loop of gemm_unit, so every Item field is a run-time value as in float_top. The processes
// that do not change (load_w, xpose, compute16, dequant_ep) are cut out: their streams are top-level ports.
// Built twice by run_hls.tcl, from the same source:
//   AMX_BASELINE=1   the integrator's gemm_contract.h as snapshotted (snap/), plan_item(c, b)
//   (default)        patched/hls/gemm_contract.h (make_patch.py), plan_item(c, b, hoff)
// The two csynth reports differ only by the patch.
#include "float_linear.h"
#include "gemm_contract.h"

using fstream::IoWord;

template <class W>
static void region(const W* xmem, W* ymem, const fgemm::Item it, hls::stream<IoWord>& xr, hls::stream<IoWord>& stok_s,
                   hls::stream<IoWord>& srow_s, hls::stream<IoWord>& bias_s, hls::stream<IoWord>& ds,
                   hls::stream<IoWord>& ys) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW
  fgemm::load_io(xmem, it, xr, stok_s, srow_s, bias_s, ds);
  fgemm::store(ys, ymem, it);
}

int amx_top(const IoWord* xmem, IoWord* ymem, unsigned rec, hls::stream<IoWord>& xr, hls::stream<IoWord>& stok_s,
            hls::stream<IoWord>& srow_s, hls::stream<IoWord>& bias_s, hls::stream<IoWord>& ds,
            hls::stream<IoWord>& ys) {
#pragma HLS INTERFACE mode = m_axi port = xmem offset = slave bundle = io max_read_burst_length = 256 depth = 1048576
#pragma HLS INTERFACE mode = m_axi port = ymem offset = slave bundle = io max_write_burst_length = 256 depth = 1048576
#pragma HLS INTERFACE mode = s_axilite port = rec
#pragma HLS INTERFACE mode = s_axilite port = return
#pragma HLS INTERFACE mode = axis port = xr
#pragma HLS INTERFACE mode = axis port = stok_s
#pragma HLS INTERFACE mode = axis port = srow_s
#pragma HLS INTERFACE mode = axis port = bias_s
#pragma HLS INTERFACE mode = axis port = ds
#pragma HLS INTERFACE mode = axis port = ys
  uint32_t f[ctr::REC_FIELDS];
#pragma HLS ARRAY_PARTITION variable = f complete
fetch:
  for (int w = 0; w < (int)ctr::REC_WORDS; w++) {
#pragma HLS PIPELINE II = 1
    const IoWord v = xmem[(rec >> 4) + w];
    for (int j = 0; j < 4; j++) {
#pragma HLS UNROLL
      f[4 * w + j] = (uint32_t)v.range(32 * j + 31, 32 * j);
    }
  }
  ctr::Cmd c;
  const uint32_t st = ctr::decode(f, c);
  if (st != ctr::ST_END || c.op != ctr::OP_GEMM) return (int)st;
#ifndef AMX_BASELINE
  uint64_t hoff = 0;
#endif
items:
  for (uint32_t b = 0; b < c.B; b++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 24
#ifdef AMX_BASELINE
    const fgemm::Item it = fgemm::plan_item(c, b);
#else
    const fgemm::Item it = fgemm::plan_item(c, b, hoff);
    hoff += c.s[ctr::SLOT_H].stride;
#endif
    region(xmem, ymem, it, xr, stok_s, srow_s, bias_s, ds, ys);
  }
  return 0;
}
