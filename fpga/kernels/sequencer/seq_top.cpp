// Standalone Vitis HLS top for the float_top command sequencer, with stub modes (seq_stubs.h).
// Same argument list, bundles and pragmas as fpga/hls/float_top.cpp, so the s_axi_control register
// map (0x00..0x74, ADDR_WIDTH 7) and the five m_axi bundles are unchanged.
//   vitis_hls -f run_seq_hls.tcl            (SEQ_VARIANT=0..5, SEQ_PERIOD=4.0|3.333)
// SEQ_VARIANT selects the dispatch experiment:
//   0  every mode function called from exactly one switch case (the design)
//   1  gemm_mode called from two mutually exclusive call sites (F_ACC and not), same arguments
//   2  variant 1 plus ALLOCATION function instances = 1
//   4  gemm_mode called from two SEQUENTIAL call sites (first half of the batch items, then the rest):
//      the case the front end cannot merge - does HLS duplicate the MAC array?
//   5  variant 4 plus ALLOCATION function instances = 1 - does HLS share it again?
//   3  no modes: one null mode consumes every decoded field (one io write per command) - the cost of the
//      sequencer itself (fetch, check, decode, FSM, status) without the mode multiplexing
#include "seq_stubs.h"

#ifndef SEQ_VARIANT
#define SEQ_VARIANT 0
#endif

using namespace seq;

#ifndef __SYNTHESIS__
namespace seq {
bool g_dry = false;
void (*g_trace)(const Cmd&) = nullptr;  // host only: called for every dispatched command
}  // namespace seq
#endif

#ifdef __VITIS_HLS__
typedef ap_uint<128> WPort;
typedef ap_uint<128> IoWord;
#else
typedef Word WPort;
typedef Word IoWord;
#endif

// The engine as one concrete function, so the ALLOCATION experiment can name it.
static uint32_t gemm_mode(const Cmd& c, const WPort* w0, const WPort* w1, const WPort* w2, const WPort* w3,
                          const IoWord* xmem, IoWord* ymem) {
#pragma HLS INLINE off
  return gemm_stub(c, w0, w1, w2, w3, xmem, ymem);
}

#if SEQ_VARIANT == 3
static uint32_t null_mode(const Cmd& c, IoWord* ymem) {
#pragma HLS INLINE off
  const Opnd* o[6] = {&c.a, &c.b, &c.c, &c.d, &c.e, &c.f};
  uint32_t x = c.op ^ c.flags ^ c.tag ^ c.M ^ c.K ^ c.N ^ c.B ^ c.G ^ c.p0 ^ c.p1 ^ c.p2 ^ c.p3 ^ c.p4 ^ c.p5 ^ c.p6;
  uint32_t hi = 0;
  for (int s = 0; s < 6; s++) {
#pragma HLS UNROLL
    x ^= (uint32_t)o[s]->addr ^ o[s]->pitch ^ o[s]->stride;
    hi ^= (uint32_t)(o[s]->addr >> 32);
  }
  Word w = zword();
  wset(w, 0, x);
  wset(w, 1, hi);
  ymem[c.c.addr >> 4] = w;  // IoWord == Word on both builds
  return c.op == OP_GEMM && c.M == 0 ? ME_DIMS : ME_OK;
}
#endif

struct Modes {
  static uint32_t dispatch(const Cmd& c, const WPort* w0, const WPort* w1, const WPort* w2, const WPort* w3,
                           const IoWord* xmem, IoWord* ymem) {
#pragma HLS INLINE
#if SEQ_VARIANT == 2 || SEQ_VARIANT == 5
#pragma HLS ALLOCATION function instances = gemm_mode limit = 1
#endif
#ifndef __SYNTHESIS__
    if (g_trace) g_trace(c);
#endif
    uint32_t r = ME_OK;
#if SEQ_VARIANT == 3
    if (c.op != OP_NOP) r = null_mode(c, ymem);
    return r;
#endif
    switch (c.op) {
      case OP_GEMM:
#if SEQ_VARIANT == 0
        r = gemm_mode(c, w0, w1, w2, w3, xmem, ymem);
#elif SEQ_VARIANT == 4 || SEQ_VARIANT == 5
        {  // synthesis experiment only: items [0, B1) then [B1, B) through a second call site
          Cmd c1 = c, c2 = c;
          c1.B = (c.B + 1) / 2;
          c2.B = c.B - c1.B;
          c2.a.addr += (addr_t)c1.B * c.a.stride;
          c2.c.addr += (addr_t)c1.B * c.c.stride;
          c2.d.addr += (addr_t)c1.B * c.d.stride;
          c2.f.addr += (addr_t)c1.B * c.f.stride;
          c2.b.addr += (addr_t)(c1.B / c.G) * c.b.stride;
          c2.e.addr += (addr_t)(c1.B / c.G) * c.e.stride;
          r = gemm_mode(c1, w0, w1, w2, w3, xmem, ymem);
          if (r == ME_OK && c2.B) r = gemm_mode(c2, w0, w1, w2, w3, xmem, ymem);
        }
#else
        if (c.flags & F_ACC) {
          r = gemm_mode(c, w0, w1, w2, w3, xmem, ymem);
          r = r ? r + 0x10 : 0;  // distinct epilogue: the front end cannot sink both calls into one
        } else {
          r = gemm_mode(c, w0, w1, w2, w3, xmem, ymem);
        }
#endif
        break;
      case OP_QUANT_X:
      case OP_QUANT_W:
      case OP_QUANT_WT:
        r = stream_stub<0>(c, xmem, ymem);
        break;
      case OP_NORM:
        r = stream_stub<1>(c, xmem, ymem);
        break;
      case OP_SOFTMAX:
        r = stream_stub<2>(c, xmem, ymem);
        break;
      case OP_GELU:
        r = stream_stub<3>(c, xmem, ymem);
        break;
      case OP_SILU_MUL:
        r = stream_stub<4>(c, xmem, ymem);
        break;
      case OP_ROPE:
        r = stream_stub<5>(c, xmem, ymem);
        break;
      case OP_COPY:
      case OP_GATHER:
      case OP_AXPY:
        r = stream_stub<6>(c, xmem, ymem);
        break;
      default:  // OP_NOP (decode already rejected unknown opcodes)
        r = ME_OK;
        break;
    }
    return r;
  }
};

// The legacy one-call protocol of the stub engine: X at the xmem register, Y at ymem, port p's weight
// array at register w<p>, the FP32 row scales right after port 0's array (w0 + groups K 16 bytes),
// X token scales right after X (xmem + N K 2 bytes). The real float_top keeps its own mapping.
static Cmd legacy_cmd(uint32_t M, uint32_t K, uint32_t N) {
  Cmd c;
  c.op = OP_GEMM;
  c.flags = 0;
  c.tag = 0;
  c.M = M;
  c.K = K;
  c.N = N;
  c.B = 1;
  c.G = 1;
  const Opnd z = {0, 0, 0};
  c.a = z;
  c.b = z;
  c.c = z;
  c.d = z;
  c.e = z;
  c.f = z;
  c.a.pitch = K * 2;
  c.c.pitch = M * 4;
  c.e.addr = (addr_t)(M / GLN) * K * 16;
  c.f.addr = (addr_t)N * K * 2;
  c.p0 = 0x3F800000;  // 1.0f
  c.p1 = c.p2 = c.p3 = c.p4 = c.p5 = c.p6 = 0;
  return c;
}

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
  const Cmd legacy = legacy_cmd((uint32_t)M, (uint32_t)K, (uint32_t)N);
  return (int)run<Modes>(w0, w1, w2, w3, xmem, ymem, (uint32_t)M, (uint32_t)K, (uint32_t)N, legacy);
}
