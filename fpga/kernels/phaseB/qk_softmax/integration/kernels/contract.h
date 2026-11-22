// Integration contract of the SmolVLA float_top: command record, opcodes, flags, status, limits, layouts,
// and the reference arithmetic (int16 quantizer, GEMM epilogue). fpga/kernels/CONTRACT.md is the text;
// this header is the single source of truth for every number in it: contract.py parses the
// `constexpr uint32_t|uint64_t NAME = expr;` lines below (one per line, expr of literals, earlier names, | << +),
// so C++ and Python cannot drift.
//
// Plain C++14, no Vitis types: builds in Vitis HLS (the decode / validate / address helpers are synthesizable,
// all arithmetic 64-bit) and in g++. The reference arithmetic in namespace ctr::ref is host only.
#pragma once
#include <math.h>
#include <stdint.h>
#include <string.h>

namespace ctr {

// ---------------------------------------------------------------- version
constexpr uint32_t CONTRACT_VERSION = 0x00010000;  // 1.0: f6 of every record and word 2 of the status block

// ---------------------------------------------------------------- record (256 bytes = 16 io words, CONTRACT.md 3)
constexpr uint32_t REC_BYTES = 256;
constexpr uint32_t REC_WORDS = 16;
constexpr uint32_t REC_FIELDS = 64;
constexpr uint32_t FLD_OP = 0;       // [7:0] opcode  [15:8] REC_MAGIC  [31:16] flags
constexpr uint32_t FLD_TAG = 1;      // host-defined, reported in the status block
constexpr uint32_t FLD_M = 2;        // output columns (GEMM: Mp, a multiple of 32)
constexpr uint32_t FLD_K = 3;        // input columns / reduction length / keys
constexpr uint32_t FLD_N = 4;        // rows (tokens) per item
constexpr uint32_t FLD_B = 5;        // items (0 means 1)
constexpr uint32_t FLD_VERSION = 6;  // must equal CONTRACT_VERSION
constexpr uint32_t FLD_RSV7 = 7;     // 0
constexpr uint32_t SLOT_BASE = 8;    // slot s = fields 8 + 4 s .. 11 + 4 s: addr[31:0], addr[63:32], pitch, stride
constexpr uint32_t SLOT_FIELDS = 4;
constexpr uint32_t NSLOTS = 8;
constexpr uint32_t PARAM_BASE = 40;  // p0..p15 = fields 40..55
constexpr uint32_t NPARAMS = 16;
constexpr uint32_t RSV_BASE = 56;    // fields 56..62: 0
constexpr uint32_t NRSV = 7;
constexpr uint32_t CHECK_FIELD = 63;  // CHECK_SEED ^ f0 ^ ... ^ f62
constexpr uint32_t REC_MAGIC = 0xC6;  // 0xC5 was the 128-byte draft (fpga/kernels/sequencer): rejected
constexpr uint32_t CHECK_SEED = 0x5EC0C6ED;

constexpr uint32_t SLOT_A = 0;  // main input rows
constexpr uint32_t SLOT_B = 1;  // GEMM weight blob / SILU_MUL up rows
constexpr uint32_t SLOT_C = 2;  // output
constexpr uint32_t SLOT_D = 3;  // GEMM addend rows / SOFTMAX rank_q
constexpr uint32_t SLOT_E = 4;  // GEMM s_row / NORM gamma,beta / ROPE table / SOFTMAX rank_k
constexpr uint32_t SLOT_F = 5;  // scales: GEMM s_tok in, q-out scales out
constexpr uint32_t SLOT_G = 6;  // GEMM bias / QK_SOFTMAX s_tok of Q
constexpr uint32_t SLOT_H = 7;  // QK_SOFTMAX s_row of K (unused by every other opcode: must be 0)

// parameter indices (p<i> = field PARAM_BASE + i)
constexpr uint32_t P_EPS = 0;     // NORM: eps, FP32 bits.  SOFTMAX: score multiplier, FP32 bits
constexpr uint32_t P_SEG = 1;     // row ops: values per segment (one scale per (row, segment)): 0 (= K) or SEG_HEAD
constexpr uint32_t P_SEGC = 2;    // row ops: bytes between segments in C.  GEMM: R, rows per C row group (0 = off)
constexpr uint32_t P_SEGF = 3;    // row ops: bytes between segments in F.  GEMM: P1, bytes between C row groups
constexpr uint32_t P_SHIFT = 4;   // row ops: stored scale = s * 2^p4 (int32, -8..8; the 1/8 of Q is -3)
constexpr uint32_t P_NP = 5;      // row ops / SOFTMAX: rows written incl. zero rows [N, NP) (0 = N);
                                  // QUANT_WT: keys of the V^T blob (its K)

// ---------------------------------------------------------------- registers (the shipped s_axi_control map)
constexpr uint32_t REG_AP_CTRL = 0x00;
constexpr uint32_t REG_GIE = 0x04;
constexpr uint32_t REG_IER = 0x08;
constexpr uint32_t REG_ISR = 0x0C;
constexpr uint32_t REG_AP_RETURN = 0x10;
constexpr uint32_t REG_W0 = 0x18;  // w0..w3, xmem, ymem: 64-bit, low word at the offset; all 0 in list mode
constexpr uint32_t REG_W1 = 0x24;
constexpr uint32_t REG_W2 = 0x30;
constexpr uint32_t REG_W3 = 0x3C;
constexpr uint32_t REG_XMEM = 0x48;
constexpr uint32_t REG_YMEM = 0x54;
constexpr uint32_t REG_M = 0x60;   // MODE_MAGIC << 24 | step << 23 | first index
constexpr uint32_t REG_K = 0x68;   // list base >> 4
constexpr uint32_t REG_N = 0x70;   // status block >> 4 (0 = none)
constexpr uint32_t CTRL_ADDR_BITS = 7;  // s_axi_control address width: must stay 7 (0x00..0x7F)
constexpr uint32_t MODE_MAGIC = 0xA5;
constexpr uint32_t MODE_STEP_BIT = 23;
constexpr uint32_t MAX_INDEX = 0x7FFFFF;

// ---------------------------------------------------------------- status block (3 io words = 48 bytes)
// word 0: code, next index, tag, commands done in this start
// word 1: mode error, STATUS_MAGIC, starts since configuration, last opcode
// word 2: CONTRACT_VERSION, index of the last finished command, cycles of this start [31:0], [63:32]
constexpr uint32_t STATUS_WORDS = 3;
constexpr uint32_t STATUS_BYTES = 48;
constexpr uint32_t STATUS_MAGIC = 0x53455153;

// ---------------------------------------------------------------- opcodes and the unit that runs each
constexpr uint32_t OP_NOP = 0x00;       // sequencer
constexpr uint32_t OP_END = 0x01;       // sequencer
constexpr uint32_t OP_GEMM = 0x10;      // engine (fpga/hls int16_linear.h + the Engine changes)
constexpr uint32_t OP_QUANT_X = 0x20;   // row unit: FP32 rows -> int16 rows (+ per-segment scales)
constexpr uint32_t OP_QUANT_W = 0x21;   // row unit: FP32 rows -> KW weight tiling (rows = blob rows)
constexpr uint32_t OP_QUANT_WT = 0x22;  // row unit: FP32 rows transposed -> VW weight tiling, column scales
constexpr uint32_t OP_NORM = 0x30;      // row unit: LayerNorm / RMSNorm (-> int16 or FP32)
constexpr uint32_t OP_SOFTMAX = 0x40;   // softmax unit: masked rows -> int16 P + 1/sum scale
constexpr uint32_t OP_QK_SOFTMAX = 0x41;  // engine (GEMM unit): Q.K^T + softmax in one command -> int16 P +
                                        // 1/sum scale; the scores stay on chip (phase B, qk_softmax)
constexpr uint32_t OP_GELU = 0x50;      // row unit: tanh-GELU -> int16
constexpr uint32_t OP_SILU_MUL = 0x51;  // row unit: SiLU(A) * B, or SiLU(A) with F_NOMUL -> int16
constexpr uint32_t OP_ROPE = 0x60;      // row unit: rotate (i, i+32) per 64-segment -> int16 / KW / FP32
constexpr uint32_t OP_COPY = 0x70;      // row unit: FP32 rows -> FP32 rows (pixel shuffle, constant rows)
// retired draft opcodes (BAD_OPCODE in 1.0): GATHER 0x71 (language rows are constants), AXPY 0x80 (Euler is
// the epilogue of action_out_proj)

// ---------------------------------------------------------------- flags (f0[31:16])
constexpr uint32_t F_YIELD = 0x0001;   // stop after this command with ST_YIELD (resume at index + 1)
constexpr uint32_t F_QOUT = 0x0002;    // row ops: int16 rows to C + scales to F (else FP32 rows to C)
constexpr uint32_t F_LN = 0x0010;      // NORM: LayerNorm, gamma at E, beta at E + 4 K (else RMSNorm, gamma at E)
constexpr uint32_t F_BIAS = 0x0020;    // GEMM: + bias[m] (slot G)
constexpr uint32_t F_ACC = 0x0040;     // GEMM: + C (read before written; D slot must be 0; C's addressing)
constexpr uint32_t F_ADD_D = 0x0080;   // GEMM: + D[n][m] (pitch 0 = one row broadcast; may alias C)
constexpr uint32_t F_MASK = 0x0100;    // SOFTMAX: key k visible to row n iff 0 <= E[k] <= D[n] (int32)
constexpr uint32_t F_NOMUL = 0x0200;   // SILU_MUL: no B operand
constexpr uint32_t F_KW = 0x0800;      // ROPE: write the KW weight tiling (as QUANT_W) instead of int16 rows
constexpr uint32_t F_RETIRED = 0xF40C; // RESID 0x4, SUM 0x8, BF16 0x400, ENGINE 0x1000, EMPTY_UNIFORM 0x2000,
                                       // 0x4000, 0x8000: must be 0
constexpr uint32_t FLAGS_GEMM = F_YIELD | F_BIAS | F_ACC | F_ADD_D;
constexpr uint32_t FLAGS_QUANT = F_YIELD | F_QOUT;
constexpr uint32_t FLAGS_NORM = F_YIELD | F_QOUT | F_LN;
constexpr uint32_t FLAGS_ACT = F_YIELD | F_QOUT | F_NOMUL;
constexpr uint32_t FLAGS_ROPE = F_YIELD | F_QOUT | F_KW;
constexpr uint32_t FLAGS_SOFTMAX = F_YIELD | F_QOUT | F_MASK;
constexpr uint32_t FLAGS_QKSMX = F_YIELD | F_QOUT | F_MASK;
constexpr uint32_t FLAGS_COPY = F_YIELD;

// ---------------------------------------------------------------- status codes (ap_return[7:0])
constexpr uint32_t ST_END = 0x00;
constexpr uint32_t ST_YIELD = 0x01;
constexpr uint32_t ST_STEP = 0x02;
constexpr uint32_t ST_NO_LEGACY = 0x03;  // M without MODE_MAGIC: nothing read or written (no per-call protocol)
constexpr uint32_t ST_BAD_MAGIC = 0x10;
constexpr uint32_t ST_BAD_CHECK = 0x11;
constexpr uint32_t ST_BAD_OPCODE = 0x12;
constexpr uint32_t ST_MODE_ERROR = 0x13;  // next index = the failing command; status word 1 lane 0 = ME_*
constexpr uint32_t ST_LIMIT = 0x14;       // index past MAX_INDEX
constexpr uint32_t ST_BAD_VERSION = 0x15;

// ---------------------------------------------------------------- mode errors (validate(); nothing is touched)
constexpr uint32_t ME_OK = 0;
constexpr uint32_t ME_DIMS = 1;   // M, K, N, B, segment or NP out of range
constexpr uint32_t ME_ALIGN = 2;  // a used slot's addr, pitch or stride not a multiple of 16; pitch too small
constexpr uint32_t ME_ADDR = 3;   // an extent ends at or past ADDR_LIMIT
constexpr uint32_t ME_FLAGS = 4;  // a flag the opcode does not take, or F_ACC with F_ADD_D
constexpr uint32_t ME_PARAM = 5;  // a parameter out of range, or non-zero where unused
constexpr uint32_t ME_SLOT = 6;   // an unused slot or reserved field not zero
constexpr uint32_t ME_UNIT = 7;   // the opcode's unit is not in this bitstream (a fused build without the
                                  // SOFTMAX unit returns it for OP_SOFTMAX)

// ---------------------------------------------------------------- limits
constexpr uint64_t ADDR_LIMIT = 1ull << 40;  // KR260 PS DDR: 0x0_0000_0000-0x0_7FFF_FFFF, 0x8_0000_0000-0x8_7FFF_FFFF
constexpr uint32_t ALIGN = 16;               // every used addr, pitch, stride
constexpr uint32_t N_MAX = 65535;            // rows per item, every op
constexpr uint32_t B_MAX = 65535;            // items, every op
constexpr uint32_t GEMM_M_MAX = 8192;        // Mp (s_row and bias on chip)
constexpr uint32_t GEMM_K_MAX = 4096;        // reduction per command (longer: K chunks + F_ACC)
constexpr uint32_t ROW_K_MAX = 16384;        // QUANT_X / COPY values per row (the connector is 12288)
constexpr uint32_t ROW_BUF = 3072;           // values held on chip per row: longer rows are read twice
constexpr uint32_t ACT_K_MAX = 3072;         // GELU / SILU_MUL values per row (one read)
constexpr uint32_t NORM_K_MAX = 1024;        // NORM values per row
constexpr uint32_t SEG_MAX = 64;             // segments per row
constexpr uint32_t SEG_HEAD = 64;            // the only segment size besides the whole row (no divider in hardware)
constexpr uint32_t ROPE_SEG = 64;            // ROPE: head dim (pairs i, i + 32)
constexpr uint32_t ROPE_TAB_BYTES = 256;     // ROPE: table row = 32 int32 -cos + 32 int32 -sin, Q1.17
constexpr uint32_t SMX_L_MAX = 256;          // SOFTMAX keys per row
constexpr uint32_t QKS_L_MAX = 256;          // QK_SOFTMAX keys per row (M): the score buffer holds 2 x 64 rows
constexpr uint32_t SHIFT_MAX = 8;            // |p4|

// ---------------------------------------------------------------- arithmetic constants
constexpr uint32_t QMAX = 32767;
constexpr uint32_t TINY_BITS = 0x0DA24260;  // 1e-30f, the scale floor (precision_emulation.quant_sym)
constexpr uint32_t ONE_BITS = 0x3F800000;   // 1.0f: the scale of an all-zero row, of pad rows, of pad lanes
constexpr uint32_t ENGINE_LANES = 32;       // output channels per group (weight tiling)
constexpr uint32_t ENGINE_PORTS = 4;        // hp0..hp3; port p holds rows 8p..8p+7 of each group
constexpr uint32_t ENGINE_TB = 32;          // tokens per engine block

// ---------------------------------------------------------------- decoded record
struct Slot {
  uint64_t addr;    // physical byte address
  uint32_t pitch;   // bytes between rows
  uint32_t stride;  // bytes between items
};

struct Cmd {
  uint32_t op, flags, tag;
  uint32_t M, K, N, B;  // B >= 1 after decode
  Slot s[NSLOTS];
  uint32_t p[NPARAMS];
};

static inline uint32_t f2u(float f) {
#pragma HLS INLINE
  uint32_t u;
  memcpy(&u, &f, sizeof u);
  return u;
}
static inline float u2f(uint32_t u) {
#pragma HLS INLINE
  float f;
  memcpy(&f, &u, sizeof f);
  return f;
}

static inline bool known_op(uint32_t op) {
#pragma HLS INLINE
  return op == OP_NOP || op == OP_END || op == OP_GEMM || op == OP_QUANT_X || op == OP_QUANT_W ||
         op == OP_QUANT_WT || op == OP_NORM || op == OP_SOFTMAX || op == OP_GELU || op == OP_SILU_MUL ||
         op == OP_ROPE || op == OP_COPY || op == OP_QK_SOFTMAX;
}

static inline uint32_t rec_check(const uint32_t f[REC_FIELDS]) {
  uint32_t x = CHECK_SEED;
  for (uint32_t j = 0; j < CHECK_FIELD; j++) {
#pragma HLS UNROLL
    x ^= f[j];
  }
  return x;
}

static inline void encode(const Cmd& c, uint32_t f[REC_FIELDS]) {
  for (uint32_t j = 0; j < REC_FIELDS; j++) f[j] = 0;
  f[FLD_OP] = (c.op & 0xFF) | (REC_MAGIC << 8) | (c.flags << 16);
  f[FLD_TAG] = c.tag;
  f[FLD_M] = c.M;
  f[FLD_K] = c.K;
  f[FLD_N] = c.N;
  f[FLD_B] = c.B;
  f[FLD_VERSION] = CONTRACT_VERSION;
  for (uint32_t s = 0; s < NSLOTS; s++) {
    f[SLOT_BASE + SLOT_FIELDS * s + 0] = (uint32_t)c.s[s].addr;
    f[SLOT_BASE + SLOT_FIELDS * s + 1] = (uint32_t)(c.s[s].addr >> 32);
    f[SLOT_BASE + SLOT_FIELDS * s + 2] = c.s[s].pitch;
    f[SLOT_BASE + SLOT_FIELDS * s + 3] = c.s[s].stride;
  }
  for (uint32_t i = 0; i < NPARAMS; i++) f[PARAM_BASE + i] = c.p[i];
  f[CHECK_FIELD] = rec_check(f);
}

// Unpack one record: ST_END (= 0, ok) or ST_BAD_MAGIC / ST_BAD_CHECK / ST_BAD_VERSION / ST_BAD_OPCODE.
// The opcode's own limits are validate()'s (ME_*).
static inline uint32_t decode(const uint32_t f[REC_FIELDS], Cmd& c) {
  if (((f[FLD_OP] >> 8) & 0xFF) != REC_MAGIC) return ST_BAD_MAGIC;
  if (rec_check(f) != f[CHECK_FIELD]) return ST_BAD_CHECK;
  if (f[FLD_VERSION] != CONTRACT_VERSION) return ST_BAD_VERSION;
  c.op = f[FLD_OP] & 0xFF;
  c.flags = f[FLD_OP] >> 16;
  c.tag = f[FLD_TAG];
  c.M = f[FLD_M];
  c.K = f[FLD_K];
  c.N = f[FLD_N];
  c.B = f[FLD_B] == 0 ? 1 : f[FLD_B];
  for (uint32_t s = 0; s < NSLOTS; s++) {
    c.s[s].addr = (uint64_t)f[SLOT_BASE + SLOT_FIELDS * s] | ((uint64_t)f[SLOT_BASE + SLOT_FIELDS * s + 1] << 32);
    c.s[s].pitch = f[SLOT_BASE + SLOT_FIELDS * s + 2];
    c.s[s].stride = f[SLOT_BASE + SLOT_FIELDS * s + 3];
  }
  for (uint32_t i = 0; i < NPARAMS; i++) c.p[i] = f[PARAM_BASE + i];
  if (!known_op(c.op)) return ST_BAD_OPCODE;
  // reserved fields travel through validate() as ME_SLOT: decode only reports the record-level faults
  return ST_END;
}

// ---------------------------------------------------------------- addressing (64-bit everywhere)
// Row n of item b: addr + b * stride + n * pitch.
static inline uint64_t row_addr(const Slot& s, uint64_t b, uint64_t n) {
#pragma HLS INLINE
  return s.addr + b * (uint64_t)s.stride + n * (uint64_t)s.pitch;
}
// GEMM output row t of item b with row groups (R = p2 > 0): addr + b stride + (t / R) P1 + (t % R) pitch.
static inline uint64_t row_addr2(const Slot& s, uint64_t b, uint64_t t, uint64_t R, uint64_t P1) {
#pragma HLS INLINE
  return R == 0 ? row_addr(s, b, t) : s.addr + b * (uint64_t)s.stride + (t / R) * P1 + (t % R) * (uint64_t)s.pitch;
}
// Segmented row ops: element (b, n, j-th segment, k-th value of it) of an int16 / FP32 row output, and its scale.
static inline uint64_t seg_addr(const Slot& s, uint64_t b, uint64_t n, uint64_t j, uint64_t seg_bytes) {
#pragma HLS INLINE
  return s.addr + b * (uint64_t)s.stride + j * seg_bytes + n * (uint64_t)s.pitch;
}
static inline uint64_t scale_addr(const Slot& f, uint64_t b, uint64_t j, uint64_t segf, uint64_t n) {
#pragma HLS INLINE
  return f.addr + b * (uint64_t)f.stride + j * segf + 4 * n;
}
// Weight tiling (engine DRIVER CONTRACT, kept): blob row m (output channel), reduction index k, K_blob words per
// group. The 128-bit word holding W[m][k] and its 16-bit lane. Port p's array starts at addr + p * pitch.
static inline uint64_t w_word_addr(const Slot& w, uint64_t b, uint64_t m, uint64_t k, uint64_t kblob) {
#pragma HLS INLINE
  return w.addr + b * (uint64_t)w.stride + ((m % ENGINE_LANES) / 8) * (uint64_t)w.pitch + ((m / ENGINE_LANES) * kblob + k) * 16;
}
static inline uint32_t w_lane(uint64_t m) {
#pragma HLS INLINE
  return (uint32_t)(m % 8);
}
static inline uint32_t pad32(uint32_t n) { return (n + 31) / 32 * 32; }
static inline uint32_t pad8(uint32_t n) { return (n + 7) / 8 * 8; }
static inline uint32_t pad4(uint32_t n) { return (n + 3) / 4 * 4; }

// ---------------------------------------------------------------- validation (CONTRACT.md 3.4)
// The sequencer calls validate_hw() on every decoded command before it dispatches it, and stops with
// ST_MODE_ERROR (nothing touched) when it is not ME_OK. The host runs validate() (the same rules plus the extent
// checks) on every list it builds; contract.py implements both: a list that passes contract.validate_list()
// never makes the IP return ST_MODE_ERROR.
static inline bool slot_zero(const Slot& s) { return s.addr == 0 && s.pitch == 0 && s.stride == 0; }
static inline bool slot_aligned(const Slot& s) {
  return s.addr % ALIGN == 0 && s.pitch % ALIGN == 0 && s.stride % ALIGN == 0;
}
// end (exclusive) of rows [0, rows) of items [0, B) of row_bytes each, 1-level addressing
static inline uint64_t slot_end(const Slot& s, uint64_t B, uint64_t rows, uint64_t row_bytes) {
  return s.addr + (B - 1) * (uint64_t)s.stride + (rows - 1) * (uint64_t)s.pitch + row_bytes;
}

template <bool FULL>
static inline uint32_t check_slot(const Slot& s, uint64_t B, uint64_t rows, uint64_t row_bytes, uint64_t min_pitch,
                                  bool pitch0_ok) {
  if (!slot_aligned(s)) return ME_ALIGN;
  if (rows > 1 && !(pitch0_ok && s.pitch == 0) && s.pitch < min_pitch) return ME_ALIGN;
  if (s.addr >= ADDR_LIMIT || (FULL && slot_end(s, B, rows, row_bytes) > ADDR_LIMIT)) return ME_ADDR;
  return ME_OK;
}

// OP_QK_SOFTMAX (phase B, fpga/kernels/phaseB/qk_softmax): M = L keys, K = head dim; slots A Q int16 rows, B KW
// blob, C P int16 rows, D rank_q, E rank_k (F_MASK), F P scales, G s_tok of Q, H s_row of K; p0 score
// multiplier, p5 NP. The same rules as the GEMM and SOFTMAX branches with these slot roles.
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
template <bool FULL>
static inline uint32_t validate_t(const Cmd& c) {
  const uint64_t B = c.B, N = c.N, K = c.K, M = c.M;
  uint32_t allowed = 0, used = 0;  // flags allowed, slots used (bit s)
  const uint32_t* p = c.p;
  if (c.flags & F_RETIRED) return ME_FLAGS;
  if (c.op == OP_QK_SOFTMAX) return validate_qks_t<FULL>(c);
  if (c.op == OP_NOP || c.op == OP_END) {
    allowed = F_YIELD;
  } else if (c.op == OP_GEMM) {
    allowed = FLAGS_GEMM;
  } else if (c.op == OP_QUANT_X || c.op == OP_QUANT_W || c.op == OP_QUANT_WT) {
    allowed = FLAGS_QUANT;
  } else if (c.op == OP_NORM) {
    allowed = FLAGS_NORM;
  } else if (c.op == OP_GELU || c.op == OP_SILU_MUL) {
    allowed = FLAGS_ACT;
  } else if (c.op == OP_ROPE) {
    allowed = FLAGS_ROPE;
  } else if (c.op == OP_SOFTMAX) {
    allowed = FLAGS_SOFTMAX;
  } else if (c.op == OP_COPY) {
    allowed = FLAGS_COPY;
  }
  if (c.flags & ~allowed) return ME_FLAGS;
  if (c.op == OP_NOP || c.op == OP_END) {
    for (uint32_t s = 0; s < NSLOTS; s++)
      if (!slot_zero(c.s[s])) return ME_SLOT;
    for (uint32_t i = 0; i < NPARAMS; i++)
      if (p[i]) return ME_PARAM;
    return ME_OK;
  }
  if (N < 1 || N > N_MAX || B < 1 || B > B_MAX) return ME_DIMS;
  uint32_t r;
  if (c.op == OP_GEMM) {
    const bool acc = c.flags & F_ACC, addd = c.flags & F_ADD_D, bias = c.flags & F_BIAS;
    if (acc && addd) return ME_FLAGS;
    if (M < ENGINE_LANES || M > GEMM_M_MAX || M % ENGINE_LANES || K < 8 || K > GEMM_K_MAX || K % 8) return ME_DIMS;
    const uint64_t R = p[P_SEGC], P1 = p[P_SEGF];
    if (R > N || (R == 0 && P1 != 0) || P1 % ALIGN) return ME_PARAM;
    for (uint32_t i = 0; i < NPARAMS; i++)
      if (i != P_SEGC && i != P_SEGF && p[i]) return ME_PARAM;
    used = (1u << SLOT_A) | (1u << SLOT_B) | (1u << SLOT_C) | (1u << SLOT_E) | (1u << SLOT_F);
    if ((r = check_slot<FULL>(c.s[SLOT_A], B, N, 2 * K, 2 * K, false))) return r;
    // weight blob: 4 port arrays of (M/32) K words each, port p at addr + p * pitch
    const uint64_t port_bytes = (M / ENGINE_LANES) * K * 16;
    if (!slot_aligned(c.s[SLOT_B])) return ME_ALIGN;
    if (c.s[SLOT_B].pitch < port_bytes) return ME_ALIGN;
    if (c.s[SLOT_B].addr >= ADDR_LIMIT || (FULL && slot_end(c.s[SLOT_B], B, ENGINE_PORTS, port_bytes) > ADDR_LIMIT))
      return ME_ADDR;
    // output rows, 1- or 2-level
    const Slot& y = c.s[SLOT_C];
    if (!slot_aligned(y) || (N > 1 && R != 1 && y.pitch < 4 * M)) return ME_ALIGN;
    const uint64_t in_grp = R ? (N < R ? N : R) : N, ngrp = R ? (N - 1) / R : 0;
    if (y.addr >= ADDR_LIMIT ||
        (FULL && y.addr + (B - 1) * (uint64_t)y.stride + ngrp * P1 + (in_grp - 1) * (uint64_t)y.pitch + 4 * M > ADDR_LIMIT))
      return ME_ADDR;
    if (addd) {
      used |= 1u << SLOT_D;
      if ((r = check_slot<FULL>(c.s[SLOT_D], B, N, 4 * M, 4 * M, true))) return r;
    }
    if ((r = check_slot<FULL>(c.s[SLOT_E], B, 1, 4 * M, 0, true))) return r;
    if ((r = check_slot<FULL>(c.s[SLOT_F], B, 1, 4 * pad4(c.N), 0, true))) return r;
    if (bias) {
      used |= 1u << SLOT_G;
      if ((r = check_slot<FULL>(c.s[SLOT_G], B, 1, 4 * M, 0, true))) return r;
    }
  } else if (c.op == OP_SOFTMAX) {
    if (K < 8 || K > SMX_L_MAX || K % 8) return ME_DIMS;
    const uint64_t NP = p[P_NP] ? p[P_NP] : N;
    if (NP < N || NP > N_MAX) return ME_PARAM;
    const float mul = u2f(p[P_EPS]);
    if (!(mul > 0.0f && mul < 1e30f)) return ME_PARAM;
    for (uint32_t i = 0; i < NPARAMS; i++)
      if (i != P_EPS && i != P_NP && p[i]) return ME_PARAM;
    if (!(c.flags & F_QOUT)) return ME_FLAGS;  // FP32 P is not an output of the integrated unit
    used = (1u << SLOT_A) | (1u << SLOT_C) | (1u << SLOT_F);
    if ((r = check_slot<FULL>(c.s[SLOT_A], B, N, 4 * K, 4 * K, false))) return r;
    if ((r = check_slot<FULL>(c.s[SLOT_C], B, NP, 2 * K, 2 * K, false))) return r;
    if ((r = check_slot<FULL>(c.s[SLOT_F], B, 1, 4 * pad4(NP), 0, true))) return r;
    if (c.flags & F_MASK) {
      used |= (1u << SLOT_D) | (1u << SLOT_E);
      if ((r = check_slot<FULL>(c.s[SLOT_D], B, 1, 4 * pad4(c.N), 0, true))) return r;
      if ((r = check_slot<FULL>(c.s[SLOT_E], B, 1, 4 * K, 0, true))) return r;
    }
  } else {  // row unit
    if (p[P_SEG] != 0 && p[P_SEG] != SEG_HEAD) return ME_DIMS;
    const uint64_t SEG = p[P_SEG] ? (uint64_t)SEG_HEAD : K;
    if (K < 8 || K % 8 || (p[P_SEG] && (K % SEG_HEAD || K / SEG_HEAD > SEG_MAX))) return ME_DIMS;
    const uint64_t nseg = p[P_SEG] ? K / SEG_HEAD : 1, NP = p[P_NP] ? p[P_NP] : N;
    const uint64_t segc = p[P_SEGC], segf = p[P_SEGF];
    const int32_t sh = (int32_t)p[P_SHIFT];
    if (sh < -(int32_t)SHIFT_MAX || sh > (int32_t)SHIFT_MAX) return ME_PARAM;
    if (segc % ALIGN || segf % ALIGN || (nseg == 1 && (segc || segf))) return ME_PARAM;
    for (uint32_t i = P_NP + 1; i < NPARAMS; i++)
      if (p[i]) return ME_PARAM;
    const bool q = c.flags & F_QOUT;
    used = (1u << SLOT_A) | (1u << SLOT_C);
    if (c.op == OP_COPY) {
      if (K % 4 || K > ROW_K_MAX || nseg != 1 || NP != N || sh || p[P_EPS]) return ME_DIMS;
      if ((r = check_slot<FULL>(c.s[SLOT_A], B, N, 4 * K, 4 * K, false))) return r;
      if ((r = check_slot<FULL>(c.s[SLOT_C], B, N, 4 * K, 4 * K, false))) return r;
    } else {
      if (p[P_EPS] && c.op != OP_NORM) return ME_PARAM;
      if (NP < N || NP > N_MAX) return ME_PARAM;
      if ((r = check_slot<FULL>(c.s[SLOT_A], B, N, 4 * K, 4 * K, false))) return r;
      const bool kw = c.op == OP_QUANT_W || (c.op == OP_ROPE && (c.flags & F_KW));
      if (c.op == OP_QUANT_W || c.op == OP_QUANT_WT) {
        if (q) return ME_FLAGS;  // their output format is the weight tiling
      }
      if (c.op == OP_ROPE && q && (c.flags & F_KW)) return ME_FLAGS;
      const bool scales = q || kw || c.op == OP_QUANT_WT;
      if (c.op == OP_QUANT_X && !q) return ME_FLAGS;
      if (c.op == OP_NORM) {
        if (K > NORM_K_MAX || nseg != 1) return ME_DIMS;
        const float eps = u2f(p[P_EPS]);
        if (!(eps > 0.0f && eps < 1.0f)) return ME_PARAM;
        used |= 1u << SLOT_E;
        if ((r = check_slot<FULL>(c.s[SLOT_E], 1, 1, (c.flags & F_LN ? 8 : 4) * K, 0, true))) return r;
      } else if (c.op == OP_GELU || c.op == OP_SILU_MUL) {
        if (K > ACT_K_MAX || nseg != 1) return ME_DIMS;
        if (c.op == OP_GELU && (c.flags & F_NOMUL)) return ME_FLAGS;
        if (c.op == OP_SILU_MUL && !(c.flags & F_NOMUL)) {
          used |= 1u << SLOT_B;
          if ((r = check_slot<FULL>(c.s[SLOT_B], B, N, 4 * K, 4 * K, false))) return r;
        }
      } else if (c.op == OP_ROPE) {
        if (SEG != ROPE_SEG) return ME_DIMS;
        used |= 1u << SLOT_E;
        if ((r = check_slot<FULL>(c.s[SLOT_E], B, N, ROPE_TAB_BYTES, ROPE_TAB_BYTES, false))) return r;
      } else if (K > ROW_K_MAX) {
        return ME_DIMS;
      }
      if (!scales && (sh || NP != N)) return ME_PARAM;
      const Slot& o = c.s[SLOT_C];
      if (!slot_aligned(o)) return ME_ALIGN;
      uint64_t end;
      if (kw) {  // KW: blob rows n < NP (NP % 32 == 0), K_blob = SEG; port pitch >= (NP / 32) SEG 16
        if (NP % ENGINE_LANES) return ME_DIMS;
        const uint64_t port = (NP / ENGINE_LANES) * SEG * 16;
        if (o.pitch < port || (nseg > 1 && segc < port)) return ME_ALIGN;
        end = o.addr + (B - 1) * (uint64_t)o.stride + (nseg - 1) * segc + 3 * (uint64_t)o.pitch + port;
      } else if (c.op == OP_QUANT_WT) {  // VW: blob of SEG channels x NP keys per segment
        if (SEG % ENGINE_LANES || NP % 8) return ME_DIMS;
        const uint64_t port = (SEG / ENGINE_LANES) * NP * 16;
        if (o.pitch < port || (nseg > 1 && segc < port)) return ME_ALIGN;
        end = o.addr + (B - 1) * (uint64_t)o.stride + (nseg - 1) * segc + 3 * (uint64_t)o.pitch + port;
      } else {  // rows: int16 (F_QOUT) or FP32; segment j of row n at C + n pitch + j segc (head-major when segc
                // is a whole head block, in-row when segc = SEG x element bytes; overlap is the builder's to avoid)
        const uint64_t eb = q ? 2 : 4, seg_row = nseg > 1 ? SEG * eb : K * eb;
        if (NP > 1 && o.pitch < seg_row) return ME_ALIGN;
        if (nseg > 1 && segc < seg_row) return ME_ALIGN;
        end = o.addr + (B - 1) * (uint64_t)o.stride + (nseg - 1) * segc + (NP - 1) * (uint64_t)o.pitch + seg_row;
      }
      if (o.addr >= ADDR_LIMIT || (FULL && end > ADDR_LIMIT)) return ME_ADDR;
      if (scales) {
        used |= 1u << SLOT_F;
        const uint64_t nsc = c.op == OP_QUANT_WT ? SEG : NP;  // QUANT_WT: one scale per channel
        const Slot& f = c.s[SLOT_F];
        if (!slot_aligned(f)) return ME_ALIGN;
        if (nseg > 1 && segf < 4 * pad4((uint32_t)nsc)) return ME_ALIGN;  // scale vectors never share a word
        if (f.addr >= ADDR_LIMIT ||
            (FULL && f.addr + (B - 1) * (uint64_t)f.stride + (nseg - 1) * segf + 4 * pad4((uint32_t)nsc) > ADDR_LIMIT))
          return ME_ADDR;
      } else if (segf) {
        return ME_PARAM;
      }
    }
  }
  for (uint32_t s = 0; s < NSLOTS; s++)
    if (!(used & (1u << s)) && !slot_zero(c.s[s])) return ME_SLOT;
  return ME_OK;
}

// validate(): the full rules (host: list builder, golden executor, contract.py). validate_hw(): the same without the
// extent sums (no 64-bit products: what the sequencer checks in hardware before each dispatch). A list that the host
// validated never fails validate_hw; validate_hw alone still catches every flag, dims, param, alignment, slot and
// address-bit fault of a corrupted or stale list.
static inline uint32_t validate(const Cmd& c) { return validate_t<true>(c); }
static inline uint32_t validate_hw(const Cmd& c) { return validate_t<false>(c); }

// list-mode start registers and ap_return
static inline uint32_t start_m(uint32_t first, bool step) {
  return (MODE_MAGIC << 24) | ((step ? 1u : 0u) << MODE_STEP_BIT) | (first & MAX_INDEX);
}

#ifndef __SYNTHESIS__
// ---------------------------------------------------------------- reference arithmetic (host only)
namespace ref {

// The int16 quantizer (CONTRACT.md 2), = fpga/kernels/quantize/quant_int16.h = pack_float._quantize_int16 =
// precision_emulation.quant_sym except all-zero rows (s = 1.0). Build with -ffp-contract=off (SSE, no x87).
static inline float scale_of_amax(uint32_t amax_bits) {
  if (amax_bits == 0) return 1.0f;
  const float a = u2f(amax_bits > TINY_BITS ? amax_bits : TINY_BITS);
  return a / 32767.0f;
}
static inline int16_t quant_elem(float x, float s) {
  float v = x / s;          // fl32
  v = nearbyintf(v);        // round half to even (default rounding mode)
  if (v > 32767.0f) v = 32767.0f;
  if (v < -32767.0f) v = -32767.0f;
  return (int16_t)v;
}
// one row (or segment) of n values: q[n], returns s; shift: the stored s is s * 2^shift (exact power of two)
static inline float quant_row(const float* x, uint32_t n, int16_t* q, int shift = 0) {
  uint32_t a = 0;
  for (uint32_t j = 0; j < n; j++) {
    const uint32_t m = f2u(x[j]) & 0x7FFFFFFFu;
    a = m > a ? m : a;
  }
  const float s = scale_of_amax(a);
  for (uint32_t j = 0; j < n; j++) q[j] = quant_elem(x[j], s);
  return a == 0 ? 1.0f : ldexpf(s, shift);  // zero rows store 1.0 whatever the shift
}

// The Xilinx FP32 cores flush denormal inputs and results to zero and keep the sign (int16_linear.h ftz).
static inline float ftz(float f) {
  uint32_t u = f2u(f);
  if (((u >> 23) & 0xFF) == 0) u &= 0x80000000u;
  return u2f(u);
}
static inline float fmul_hw(float a, float b) { return ftz(ftz(a) * ftz(b)); }
static inline float fadd_hw(float a, float b) { return ftz(ftz(a) + ftz(b)); }

// GEMM epilogue (CONTRACT.md 2.3): acc = exact sum_k Xq[n][k] Wq[m][k] (|acc| < 2^42)
//   y = ((fl(acc) * s_row[m]) * s_tok[n]) [+ bias[m]] [+ D[n][m] or + C[n][m] (F_ACC)], each step one FP32 op.
static inline float gemm_out(int64_t acc, float s_row, float s_tok, const float* bias, const float* addend) {
  float y = fmul_hw(fmul_hw((float)acc, s_row), s_tok);
  if (bias) y = fadd_hw(y, *bias);
  if (addend) y = fadd_hw(y, *addend);
  return y;
}

}  // namespace ref
#endif

}  // namespace ctr
