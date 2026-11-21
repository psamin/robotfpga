// Command-list format for the float_top sequencer (fpga/kernels/sequencer).
//
// Single source of truth: seq_format.py parses the `constexpr uint32_t NAME = value;` lines below, so
// C++ and Python cannot drift. Keep one constant per line in that exact form.
//
// A command is one 128-byte record = 8 words of 128 bits = 32 little-endian uint32 fields f[0..31];
// field j sits in word j / 4, bits [32 (j % 4) + 31 : 32 (j % 4)] (the io word lane order of
// float_linear.h). Records are contiguous and 128-bit aligned; command i of a list starts at byte
// 128 i. All addresses are absolute physical byte addresses (40 bits): in sequencer mode the host
// programs every pointer register of float_top (w0..w3, xmem, ymem) to 0, so the AXI address of
// array index i is 16 i.
//
//   f0   [7:0] opcode  [15:8] SEQ_MAGIC  [31:16] flags
//   f1   tag (host-defined; reported in the status block on error / yield)
//   f2   M   (output columns / features)        f3  K (input columns / reduction length)
//   f4   N   (rows / tokens per batch item)     f5  [15:0] B batch items (0 = 1), [31:16] G divisor of
//                                                    batch index for operand B (GQA; 0 = 1)
//   f6..f23  six operand slots A, B, C, D, E, F; slot s = fields 6 + 3 s .. 8 + 3 s:
//            w0 = addr[31:0], w1 = addr[39:32] | pitch[23:0] << 8 (row pitch, bytes),
//            w2 = batch stride (bytes). Item b uses addr + b * stride (operand B: (b / G) * stride).
//   f24..f30 p0..p6 opcode parameters (FP32 bit patterns or integers)
//   f31  check = SEQ_CHECK_SEED ^ f0 ^ f1 ^ ... ^ f30
#pragma once
#include <stdint.h>

namespace seq {

// ---- record geometry
constexpr uint32_t REC_BYTES = 128;
constexpr uint32_t REC_WORDS = 8;
constexpr uint32_t REC_FIELDS = 32;
constexpr uint32_t SLOT_BASE = 6;
constexpr uint32_t SLOT_FIELDS = 3;
constexpr uint32_t NSLOTS = 6;
constexpr uint32_t PARAM_BASE = 24;
constexpr uint32_t NPARAMS = 7;
constexpr uint32_t CHECK_FIELD = 31;
constexpr uint32_t SEQ_MAGIC = 0xC5;
constexpr uint32_t SEQ_CHECK_SEED = 0x5EC0C5ED;

// ---- register protocol (float_top's existing s_axi_control map, nothing added)
// M[31:24] == MODE_MAGIC selects the sequencer; otherwise float_top runs one legacy call.
//   M[23] = single step (run one command, return STEP), M[22:0] = index of the first command
//   K     = list base >> 4 (physical, 16-byte aligned)
//   N     = status block address >> 4 (0: none)   ap_return = (next index << 8) | status code
constexpr uint32_t MODE_MAGIC = 0xA5;
constexpr uint32_t MODE_STEP_BIT = 23;
constexpr uint32_t MAX_INDEX = 0x7FFFFF;
constexpr uint32_t STATUS_WORDS = 2;
constexpr uint32_t STATUS_MAGIC = 0x53455153;

// ---- opcodes
constexpr uint32_t OP_NOP = 0x00;
constexpr uint32_t OP_END = 0x01;
constexpr uint32_t OP_GEMM = 0x10;      // int16 engine
constexpr uint32_t OP_QUANT_X = 0x20;   // FP32 rows -> int16 rows + per-row FP32 scale
constexpr uint32_t OP_QUANT_W = 0x21;   // FP32 rows -> weight blob (rows padded to 32)
constexpr uint32_t OP_QUANT_WT = 0x22;  // FP32 rows, transposed -> weight blob
constexpr uint32_t OP_NORM = 0x30;      // (residual add) + LayerNorm / RMSNorm (+ quantize)
constexpr uint32_t OP_SOFTMAX = 0x40;   // row softmax (+ block mask, + quantize)
constexpr uint32_t OP_GELU = 0x50;      // tanh-GELU (+ quantize)
constexpr uint32_t OP_SILU_MUL = 0x51;  // SiLU(A) * B or SiLU(A) (+ quantize)
constexpr uint32_t OP_ROPE = 0x60;      // rotary embedding by position table (+ quantize)
constexpr uint32_t OP_COPY = 0x70;      // 2-D strided copy of K bytes per row
constexpr uint32_t OP_GATHER = 0x71;    // table rows by int32 index (BF16 or FP32 table) * p0
constexpr uint32_t OP_AXPY = 0x80;      // C = A + p0 * B (FP32)

// ---- flags (f0[31:16]; bit i below is flag bit i of that 16-bit field)
constexpr uint32_t F_YIELD = 0x0001;    // stop after this command, return YIELD (resume at index + 1)
constexpr uint32_t F_QOUT = 0x0002;     // output int16 to C with per-row FP32 scales to F
constexpr uint32_t F_RESID = 0x0004;    // NORM: h = A + B before the norm
constexpr uint32_t F_SUM = 0x0008;      // NORM: write h to D
constexpr uint32_t F_LN = 0x0010;       // NORM: LayerNorm with gamma at E, beta at E + 4 K (else RMSNorm, gamma at E)
constexpr uint32_t F_BIAS = 0x0020;     // GEMM: the weight blob carries an FP32 bias row
constexpr uint32_t F_ACC = 0x0040;      // GEMM: C += result (reductions split over K)
constexpr uint32_t F_ADD_D = 0x0080;    // GEMM: + D (FP32 rows; pitch 0 broadcasts one row)
constexpr uint32_t F_MASK = 0x0100;     // SOFTMAX: key k visible to row n iff 0 <= E[k] <= D[n] (int32)
constexpr uint32_t F_NOMUL = 0x0200;    // SILU_MUL: no B operand
constexpr uint32_t F_BF16 = 0x0400;     // GATHER: BF16 table (widened exactly)

// ---- status codes (ap_return[7:0] and status word 0)
constexpr uint32_t ST_END = 0x00;
constexpr uint32_t ST_YIELD = 0x01;
constexpr uint32_t ST_STEP = 0x02;
constexpr uint32_t ST_LEGACY = 0x03;
constexpr uint32_t ST_BAD_MAGIC = 0x10;
constexpr uint32_t ST_BAD_CHECK = 0x11;
constexpr uint32_t ST_BAD_OPCODE = 0x12;
constexpr uint32_t ST_MODE_ERROR = 0x13;
constexpr uint32_t ST_LIMIT = 0x14;

// ---- mode error codes (returned by mode functions, reported as status word 4)
constexpr uint32_t ME_OK = 0;
constexpr uint32_t ME_DIMS = 1;
constexpr uint32_t ME_ALIGN = 2;

}  // namespace seq
