// Vitis HLS top for the full-precision (BF16 weights, FP32 math) linear engine, issue #148.
// Four weight bundles map to S_AXI_HP0..HP3_FPD in the block design; X/Y share a fifth bundle
// (an HPC port, 128-bit words). Target 300 MHz.   cd fpga/hls && vitis_hls -f run_float_hls.tcl
#include "float_linear.h"
#if defined(FSTREAM_ARITH) && FSTREAM_ARITH == 16
#include "int16_linear.h"  // -DFSTREAM_ARITH=16: int16 W x int16 X -> int48, FP32 out (same interfaces)
#if defined(FSTREAM_SEQ) && FSTREAM_SEQ
#include "float_seq.h"  // -DFSTREAM_SEQ=1: the integrated top (contract 1.0 command lists; same interfaces)
#endif
#endif

using namespace fstream;

int float_top(const WPort* w0, const WPort* w1, const WPort* w2, const WPort* w3, const IoWord* xmem,
              IoWord* ymem, int M, int K, int N) {
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
#if defined(FSTREAM_ARITH) && FSTREAM_ARITH == 16 && defined(FSTREAM_SEQ) && FSTREAM_SEQ
  // list mode (fpga/kernels/CONTRACT.md 3.1): M = 0xA5 << 24 | step << 23 | first index, K = list >> 4,
  // N = status block >> 4, every pointer register 0; returns (next index << 8) | status (ST_NO_LEGACY for any
  // other M: nothing is read or written)
  return (int)seq::top<fseq::Units>(w0, w1, w2, w3, xmem, ymem, (uint32_t)M, (uint32_t)K, (uint32_t)N);
#elif defined(FSTREAM_ARITH) && FSTREAM_ARITH == 16
  // xmem = [s_row: M/4 words][s_tok: Np/4 words][X interleaved], int16_linear.h DRIVER CONTRACT
  const int np = ((N + fstream16::TB - 1) / fstream16::TB) * fstream16::TB;
  return fstream16::linear_int16_core(w0, w1, w2, w3, xmem, ymem, M, K, N, M / F4, (M + np) / F4);
#else
  // Token a of every block in its own memory (dim 1), 4 consecutive k in 4 memories (dim 3) so the
  // 128-bit X loads write 4 values per cycle.
  static float xbuf[TP][NB][K_MAX];
#pragma HLS BIND_STORAGE variable = xbuf type = ram_2p impl = uram
#pragma HLS ARRAY_PARTITION variable = xbuf complete dim = 1
#pragma HLS ARRAY_PARTITION variable = xbuf cyclic factor = 4 dim = 3
  return linear_bf16_core(w0, w1, w2, w3, xmem, ymem, M, K, N, xbuf);
#endif
}
