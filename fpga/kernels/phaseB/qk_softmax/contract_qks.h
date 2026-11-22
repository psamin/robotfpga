// Phase-B fusion qk_softmax: the record of the fused command QK_SOFTMAX (opcode 0x41), as an overlay of
// fpga/kernels/contract.h (contract 1.0). INTEGRATION.md gives the same lines as a patch of contract.h itself
// (version 1.1); until then this header adds them in namespace ctr without touching contract.h.
//
// QK_SOFTMAX = the contract's (GEMM Q.K^T, SOFTMAX) pair of one attention site as one command, run by the GEMM
// unit: the engine computes the scores of a 64-query-row tile, the epilogue dequantizes them exactly as the GEMM
// does (CONTRACT.md 2.3, no bias / D), they stay on chip (score buffer, 2 x 64 rows x 256 keys), and the softmax
// unit's arithmetic (CONTRACT.md 2.1 softmax rule, smx_emu) turns each row into int16 P + its 1/sum scale. Only
// P and the scales leave. Bit-exact with the unfused pair by construction (same FP32 scores, same row arithmetic).
//
//   op 0x41, flags F_QOUT (required) | F_MASK | F_YIELD
//   M = L keys = Mk (the KW blob's rows), 32..256, M % 32 == 0      (GEMM M and SOFTMAX K of the unfused pair)
//   K = head dim (the reduction), 8..4096, K % 8 == 0                (GEMM K)
//   N = query rows per item (G heads stacked, row t = g TqP + i), B = items (one per KV head)
//   p0 = score multiplier (FP32 bits, as SOFTMAX; 1.0 in the list: the 1/8 rides in Q's scale)
//   p5 = NP: P rows written, rows [N, NP) zero with scale 1.0 (0 = N)
//   slots (item b adds b * stride):
//     A  Q int16 rows (GEMM A)                 B  KW blob, port pitch (GEMM B)
//     C  P int16 rows out, L values (SOFTMAX C) D  rank_q int32 [pad4(N)] (F_MASK; SOFTMAX D)
//     E  rank_k int32 [L] (F_MASK; SOFTMAX E)   F  P scales out FP32 [pad4(NP)] (SOFTMAX F)
//     G  s_tok of Q, FP32 [pad4(N)] (GEMM F)    H  s_row of K, FP32 [M] (GEMM E)
//   Every slot keeps the role it has in SOFTMAX; the GEMM's two scale inputs move to the two slots contract 1.0
//   leaves free for this opcode (G is GEMM-only, H is unused in 1.0).
#pragma once
#include "../kernels/contract.h"

namespace ctr {

constexpr uint32_t OP_QK_SOFTMAX = 0x41;  // GEMM unit: Q.K^T + softmax -> int16 P + 1/sum scale (phase B)
constexpr uint32_t FLAGS_QKSMX = F_YIELD | F_QOUT | F_MASK;
constexpr uint32_t QKS_L_MAX = SMX_L_MAX;  // keys per row (256): the score buffer holds 2 x 64 rows of them
constexpr uint32_t QKS_VERSION = 0x00010100;  // the contract version that adds the opcode (1.1)

// validate / validate_hw of OP_QK_SOFTMAX (the same rules as the GEMM and SOFTMAX branches of validate_t,
// with the slot roles above). FULL adds the extent checks (host).
template <bool FULL>
static inline uint32_t validate_qks_t(const Cmd& c) {
  const uint64_t B = c.B, N = c.N, K = c.K, M = c.M;
  const uint32_t* p = c.p;
  if (c.flags & F_RETIRED) return ME_FLAGS;
  if (c.flags & ~FLAGS_QKSMX) return ME_FLAGS;
  if (!(c.flags & F_QOUT)) return ME_FLAGS;
  if (N < 1 || N > N_MAX || B < 1 || B > B_MAX) return ME_DIMS;
  if (M < ENGINE_LANES || M > QKS_L_MAX || M % ENGINE_LANES) return ME_DIMS;
  if (K < 8 || K > GEMM_K_MAX || K % 8) return ME_DIMS;
  const uint64_t NP = p[P_NP] ? p[P_NP] : N;
  if (NP < N || NP > N_MAX) return ME_PARAM;
  const float mul = u2f(p[P_EPS]);
  if (!(mul > 0.0f && mul < 1e30f)) return ME_PARAM;
  for (uint32_t i = 0; i < NPARAMS; i++)
    if (i != P_EPS && i != P_NP && p[i]) return ME_PARAM;
  uint32_t r, used = (1u << SLOT_A) | (1u << SLOT_B) | (1u << SLOT_C) | (1u << SLOT_F) | (1u << SLOT_G) |
                     (1u << SLOT_H);
  if ((r = check_slot<FULL>(c.s[SLOT_A], B, N, 2 * K, 2 * K, false))) return r;
  const uint64_t port_bytes = (M / ENGINE_LANES) * K * 16;
  if (!slot_aligned(c.s[SLOT_B])) return ME_ALIGN;
  if (c.s[SLOT_B].pitch < port_bytes) return ME_ALIGN;
  if (c.s[SLOT_B].addr >= ADDR_LIMIT || (FULL && slot_end(c.s[SLOT_B], B, ENGINE_PORTS, port_bytes) > ADDR_LIMIT))
    return ME_ADDR;
  if ((r = check_slot<FULL>(c.s[SLOT_C], B, NP, 2 * M, 2 * M, false))) return r;
  if ((r = check_slot<FULL>(c.s[SLOT_F], B, 1, 4 * pad4((uint32_t)NP), 0, true))) return r;
  if ((r = check_slot<FULL>(c.s[SLOT_G], B, 1, 4 * pad4(c.N), 0, true))) return r;
  if ((r = check_slot<FULL>(c.s[SLOT_H], B, 1, 4 * M, 0, true))) return r;
  if (c.flags & F_MASK) {
    used |= (1u << SLOT_D) | (1u << SLOT_E);
    if ((r = check_slot<FULL>(c.s[SLOT_D], B, 1, 4 * pad4(c.N), 0, true))) return r;
    if ((r = check_slot<FULL>(c.s[SLOT_E], B, 1, 4 * M, 0, true))) return r;
  }
  for (uint32_t s = 0; s < NSLOTS; s++)
    if (!(used & (1u << s)) && !slot_zero(c.s[s])) return ME_SLOT;
  return ME_OK;
}
static inline uint32_t validate_qks(const Cmd& c) { return validate_qks_t<true>(c); }
static inline uint32_t validate_qks_hw(const Cmd& c) { return validate_qks_t<false>(c); }

}  // namespace ctr
