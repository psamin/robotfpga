// The integrated float_top (-DFSTREAM_ARITH=16 -DFSTREAM_SEQ=1): the sequencer's command walk as the body
// (fpga/kernels/sequencer/seq_core.h, CONTRACT.md 3 and E-9), dispatching every record to its unit:
//   GEMM, QK_SOFTMAX              fgemm::gemm_unit (gemm_contract.h: the int16 engine behind the contract record;
//                                 QK_SOFTMAX adds the score buffer and the softmax arithmetic, qks_tail.h)
//   QUANT_X / QUANT_W / QUANT_WT / COPY   qz::row_mode            (fpga/kernels/quantize/quant_mode.h)
//   NORM                          nr::norm_mode                    (fpga/kernels/norm_residual/norm_residual.h)
//   GELU / SILU_MUL               gs::act_mode                     (fpga/kernels/gelu_silu/gelu_silu.h)
//   ROPE                          rope::rope_mode                  (fpga/kernels/rope/rope_mode.h)
//   SOFTMAX                       smx::softmax_mode                (fpga/kernels/softmax/softmax_core.h), only with
//                                 -DFSEQ_SOFTMAX_UNIT=1 (bring-up builds); else ME_UNIT: the list builder
//                                 emits QK_SOFTMAX, whose softmax processes live in the GEMM unit
// Every mode function has exactly one call site (below) and is INLINE off: one hardware instance each. The row ops
// are four mode functions, not yet the contract's one shared row unit (CONTRACT.md 4.1): each carries its own
// reader / quantizer / writer (the kernels' state on 2026-11-21).
//
// Ports: every pointer register is 0 in list mode, so port[addr >> 4] is the byte address of the record.
//   hp0..hp3   weight blobs (GEMM), hp0 also the host constants: NORM gamma/beta, the RoPE tables
//   io         xmem = every activation read (records, X, scales, rows), ymem = every write
//   (HLS allows one io-reading process per DATAFLOW region: the GEMM unit's reader also fetches the addend rows)
// The C model (host) instantiates the same template with the host word types (fstream::WPort / IoWord, 16-byte
// structs); the kernels take their ap_uint<128> view of the same bytes.
#pragma once
#include "gemm_contract.h"

#include "../kernels/sequencer/seq_core.h"
#include "../kernels/quantize/quant_mode.h"
#include "../kernels/norm_residual/norm_residual.h"
#include "../kernels/gelu_silu/gelu_silu.h"
#include "../kernels/rope/rope_mode.h"
#include "../kernels/softmax/softmax_core.h"

#ifndef FSEQ_SOFTMAX_UNIT
#define FSEQ_SOFTMAX_UNIT 0  // phase B qk_softmax: the standalone SOFTMAX unit is not built
#endif

namespace fseq {

typedef ap_uint<128> KWord;  // the kernels' io / hp word

template <class T>
static inline const KWord* kw(const T* p) {
#pragma HLS INLINE
  static_assert(sizeof(T) == sizeof(KWord), "io words are 128 bits");
  return reinterpret_cast<const KWord*>(p);
}
template <class T>
static inline KWord* kw(T* p) {
#pragma HLS INLINE
  static_assert(sizeof(T) == sizeof(KWord), "io words are 128 bits");
  return reinterpret_cast<KWord*>(p);
}

// The row ops: one call site per mode function.
template <class WP, class W>
static uint32_t row_unit(const ctr::Cmd& c, const WP* hp0, const W* io_rd, W* io_wr) {
#pragma HLS INLINE off
  uint32_t r = ctr::ME_OK;
  switch (c.op) {
    case ctr::OP_NORM:
      r = nr::norm_mode(c, kw(hp0), kw(io_rd), kw(io_wr));
      break;
    case ctr::OP_GELU:
    case ctr::OP_SILU_MUL:
      r = gs::act_mode(c, kw(io_rd), kw(io_wr));
      break;
    case ctr::OP_ROPE:
      r = rope::rope_mode(c, kw(hp0), kw(io_rd), kw(io_wr));
      break;
    default:  // QUANT_X, QUANT_W, QUANT_WT, COPY
      r = qz::row_mode(c, kw(io_rd), kw(io_wr));
      break;
  }
  return r;
}

struct Units {
  template <class WP, class WW>
  static uint32_t gemm(const ctr::Cmd& c, const WP* hp0, const WP* hp1, const WP* hp2, const WP* hp3, const WW* io_rd,
                       WW* io_wr) {
#pragma HLS INLINE
    return fgemm::gemm_unit(c, hp0, hp1, hp2, hp3, io_rd, io_wr);
  }
  template <class WP, class WW>
  static uint32_t row(const ctr::Cmd& c, const WP* hp0, const WP*, const WP*, const WP*, const WW* io_rd, WW* io_wr) {
#pragma HLS INLINE
    return row_unit(c, hp0, io_rd, io_wr);
  }
  template <class WP, class WW>
  static uint32_t softmax(const ctr::Cmd& c, const WP*, const WP*, const WP*, const WP*, const WW* io_rd, WW* io_wr) {
#pragma HLS INLINE
#if FSEQ_SOFTMAX_UNIT
    return smx::softmax_mode(c, kw(io_rd), kw(io_wr));
#else
    (void)c;
    (void)io_rd;
    (void)io_wr;
    return ctr::ME_UNIT;  // no SOFTMAX unit in this build (nothing is touched)
#endif
  }
};

}  // namespace fseq
