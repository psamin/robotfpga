// hp_reads (phase-B fusion): how a contract record names an HP-read operand, and the rule that checks it.
// This header extends fpga/kernels/contract.h without editing it; INTEGRATION.md has the patch that moves these
// lines into contract.h / contract.py (and into ctr::Cmd, ctr::decode, ctr::encode, ctr::validate_t).
//
// Field 56 (the first of the reserved fields 56..62 of CONTRACT.md 3.2) becomes FLD_HPR:
//   [3:0]  A_PORTS: HP ports that read slot A in addition to io. Bit p = hp<p>. 0 = contract 1.0 (io only).
//          This build implements exactly one value besides 0: 1 << HPR_PORT (hp1), for every opcode in
//          HPR_OPS. Slot A's 8-word (128-byte) units then alternate: unit 2v of every row pass over io, unit
//          2v + 1 over hp1 (the unit grid of CONTRACT.md 1.1's readers: 8 words, the last one shifted back to end
//          at the row's end). The values, their order and every output byte are those of the io-only command.
//   [7:4]  B_PORTS: the same for slot B (SILU_MUL's up rows). Reserved: must be 0 (not built).
//   [31:8] 0.
// Ports: hp1 is free in every row / softmax command (hp0 carries NORM gamma/beta and the RoPE table; GEMM never
// overlaps a kernel command: R-ORD). The bus interfaces do not change: hp1 is READ_ONLY, 128-bit, bursts <= 256,
// as shipped (E-11); only its address / control use is new.
// Cross-master read-after-write: slot A was written over io (HPC) by an earlier command and is now read over HP.
// This is the same hazard as the runtime K / V^T blobs (CONTRACT.md 3.3), covered by R-ORD (a mode returns after
// its last BRESP) and by board test BT-1, which must then include an HPR read of a just-written row.
//
// Compatibility: contract.h 1.0 decode() does not copy fields 7 and 56..62 into ctr::Cmd and validate() cannot
// see them (CONTRACT.md 3.3 says "reserved fields must be 0 (ME_SLOT)": the code does not do that). So a 1.0
// bitstream ignores FLD_HPR and runs a list that sets it exactly as before, io only. HPR is therefore a pure
// performance field: no version change is needed (CONTRACT_VERSION stays 0x00010000).
#pragma once
#include <stdint.h>

#include "../../contract.h"

namespace hpr {

constexpr uint32_t FLD_HPR = 56;          // record field (was reserved)
constexpr uint32_t HPR_A_MASK = 0xF;      // [3:0] slot A ports
constexpr uint32_t HPR_B_MASK = 0xF0;     // [7:4] slot B ports (reserved, 0)
constexpr uint32_t HPR_PORT = 1;          // the HP port this build reads slot A through (hp1)
constexpr uint32_t HPR_A_HP1 = 1u << HPR_PORT;
constexpr uint32_t UNIT_WORDS = 8;        // the HPR unit: 8 io words = 128 B (32 FP32)

// opcodes whose unit implements A_PORTS = hp1 in this build (hpr_quant.h, hpr_softmax.h)
static inline bool hpr_op(uint32_t op) {
#pragma HLS INLINE
  return op == ctr::OP_QUANT_X || op == ctr::OP_QUANT_W || op == ctr::OP_QUANT_WT || op == ctr::OP_COPY ||
         op == ctr::OP_SOFTMAX;
}

// The extra check of FLD_HPR (validate and validate_hw both): ME_OK, or ME_SLOT (a reserved field that this
// build does not implement is not zero). Run after ctr::validate(c) == ME_OK.
static inline uint32_t validate_hpr(const ctr::Cmd& c, uint32_t hpr) {
#pragma HLS INLINE
  if (hpr == 0) return ctr::ME_OK;
  if (hpr != HPR_A_HP1) return ctr::ME_SLOT;
  if (!hpr_op(c.op)) return ctr::ME_SLOT;
  return ctr::ME_OK;
}

// Whether the unit really splits slot A over two ports for this command (else it runs io only, at 4 lanes).
// Row unit: rows of at least 16 words (K >= 64); below that the single-word path of quant_mode.h runs on io.
// Softmax: L % 64 == 0 (an even number of 8-word units per row; every -12392 softmax has L in {64, 128, 256}).
static inline bool hpr_active_row(const ctr::Cmd& c, uint32_t hpr) {
#pragma HLS INLINE
  return hpr == HPR_A_HP1 && c.K >= 64;
}
static inline bool hpr_active_smx(const ctr::Cmd& c, uint32_t hpr) {
#pragma HLS INLINE
  return hpr == HPR_A_HP1 && c.K % 64 == 0;
}

#ifndef __SYNTHESIS__
// host: encode a record with FLD_HPR (ctr::encode, then field 56 and the check word)
static inline void encode_hpr(const ctr::Cmd& c, uint32_t hpr, uint32_t f[ctr::REC_FIELDS]) {
  ctr::encode(c, f);
  f[FLD_HPR] = hpr;
  f[ctr::CHECK_FIELD] = ctr::rec_check(f);
}
#endif

}  // namespace hpr
