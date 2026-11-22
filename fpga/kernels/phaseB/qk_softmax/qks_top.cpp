// csynth tops of the qk_softmax fusion (run_qks_hls.tcl). float_top's exact ports and pragmas.
//
//   default      the fused GEMM unit (gemm_qks.h): GEMM and QK_SOFTMAX records
//   -DQKS_BASE   the integrator's GEMM unit as snapshotted (snap/hls/gemm_contract.h): GEMM records only
//   -DQKS_TAIL   the fused tail alone (sbuf, smx::compute, divider, the P writer) on stream inputs
//
// The first two share a one-command stand-in for the sequencer (fetch the 256-byte record at K + 16 index over
// xmem, ctr::decode, validate_hw, one dispatch site), so their difference is the fusion's cost inside the unit:
// the tail processes, the load_io / dequant_ep / store changes, the c prologue and the opcode's validate branch.
// Return: ST_NO_LEGACY without 0xA5; else (me << 16) | st.
#include "float_linear.h"
#if defined(QKS_BASE)
#include "gemm_contract.h"
#else
#include "gemm_qks.h"
#endif

using namespace fstream;

#if !defined(QKS_TAIL)
int float_top(const WPort* w0, const WPort* w1, const WPort* w2, const WPort* w3, const IoWord* xmem, IoWord* ymem,
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
  (void)N;
  const uint32_t m = (uint32_t)M;
  if ((m >> 24) != ctr::MODE_MAGIC) return (int)ctr::ST_NO_LEGACY;
  const uint64_t rec = (uint64_t)(uint32_t)K + (uint64_t)(m & ctr::MAX_INDEX) * ctr::REC_WORDS;
  uint32_t f[ctr::REC_FIELDS];
#pragma HLS ARRAY_PARTITION variable = f complete
fetch:
  for (uint32_t w = 0; w < ctr::REC_WORDS; w++) {
#pragma HLS PIPELINE II = 1
    const IoWord v = xmem[rec + w];
    for (int j = 0; j < 4; j++) {
#pragma HLS UNROLL
      f[4 * w + j] = (uint32_t)v.range(32 * j + 31, 32 * j);
    }
  }
  ctr::Cmd c;
#pragma HLS ARRAY_PARTITION variable = c.s complete
#pragma HLS ARRAY_PARTITION variable = c.p complete
  uint32_t st = ctr::decode(f, c);
#if !defined(QKS_BASE)
  if (st == ctr::ST_BAD_OPCODE && c.op == ctr::OP_QK_SOFTMAX) st = ctr::ST_END;  // contract 1.1: a known opcode
  if (st) return (int)st;
  if (c.op != ctr::OP_GEMM && c.op != ctr::OP_QK_SOFTMAX) return (int)ctr::ST_BAD_OPCODE;
  uint32_t me = c.op == ctr::OP_QK_SOFTMAX ? ctr::validate_qks_hw(c) : ctr::validate_hw(c);
#else
  if (st) return (int)st;
  if (c.op != ctr::OP_GEMM) return (int)ctr::ST_BAD_OPCODE;
  uint32_t me = ctr::validate_hw(c);
#endif
  if (me == ctr::ME_OK) me = fgemm::gemm_unit(c, w0, w1, w2, w3, xmem, ymem);  // the dispatch site
  return me ? (int)((me << 16) | ctr::ST_MODE_ERROR) : (int)ctr::ST_END;
}
#else
// -DQKS_TAIL: the tail processes alone. The scores and ranks arrive as AXI streams of 128-bit words (what
// dequant_ep and load_io write into ss / rs inside the GEMM region); the scalars come over s_axilite.
static void store_p_proc(smx::QStream& eq, smx::RStream& rq, IoWord* ymem, const smx::Params sp) {
  qks::store_p(eq, rq, ymem, sp);
}
static void tail_region(hls::stream<IoWord>& ss, hls::stream<IoWord>& rs, IoWord* ymem, const qks::Tail tl,
                        const smx::Params sp) {
#pragma HLS DATAFLOW
  smx::XStream sxs("qks_sxs");
  smx::MStream smx_s("qks_smx");
  smx::QStream eq("qks_eq");
  smx::SStream sq("qks_sq");
  smx::RStream rq("qks_rq");
#pragma HLS STREAM variable = sxs depth = smx::XS_DEPTH
#pragma HLS BIND_STORAGE variable = sxs type = fifo impl = lutram
#pragma HLS STREAM variable = smx_s depth = smx::MX_DEPTH
#pragma HLS STREAM variable = eq depth = smx::EQ_DEPTH
#pragma HLS BIND_STORAGE variable = eq type = fifo impl = srl
#pragma HLS STREAM variable = sq depth = smx::SQ_DEPTH
#pragma HLS STREAM variable = rq depth = smx::RQ_DEPTH
#pragma HLS BIND_STORAGE variable = rq type = fifo impl = lutram
  qks::sbuf(ss, rs, sxs, smx_s, tl);
  smx::compute(sxs, smx_s, eq, sq, sp);
  qks::divider_if(sq, rq, tl);
  store_p_proc(eq, rq, ymem, sp);
}
void qks_tail_top(hls::stream<IoWord>& ss, hls::stream<IoWord>& rs, IoWord* ymem, int on, int mask, int N, int G,
                  int ntiles, int W, uint32_t NP, uint32_t p0, uint32_t c_word, uint32_t c_pitch, uint32_t f_word) {
#pragma HLS INTERFACE mode = axis port = ss
#pragma HLS INTERFACE mode = axis port = rs
#pragma HLS INTERFACE mode = m_axi port = ymem offset = slave bundle = io max_write_burst_length = 256 depth = 1048576
#pragma HLS INTERFACE mode = s_axilite port = return
  qks::Tail tl;
  tl.on = on != 0;
  tl.mask = mask != 0;
  tl.N = N;
  tl.G = G;
  tl.ntiles = ntiles;
  tl.W = W;
  smx::Params p;
  p.B = 1;
  p.N = N;
  p.NP = NP;
  p.NP4 = (NP + 3) & ~3u;
  p.W = W;
  p.W2 = W >> 1;
  p.mask = mask != 0;
  ap_int<10> ec;
  smx::c_params(p0, p.mc, ec);
  p.kc = (ap_int<11>)153 - (ap_int<11>)ec;
  p.a = p.d = p.e = 0;
  p.c = c_word;
  p.f = f_word;
  p.ap = 0;
  p.cp = c_pitch;
  p.as = p.cs = p.ds = p.es = p.fs = 0;
  p.a_contig = false;
  p.c_contig = p.cp == p.W2;
  tail_region(ss, rs, ymem, tl, p);
}
#endif
