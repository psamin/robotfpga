// Command sequencer of float_top, integration contract 1.0 (fpga/kernels/CONTRACT.md 3 and 4.1).
// Header-only; builds in Vitis HLS 2022.2 and in g++ (host testbench). Every record, flag, status code and limit
// comes from ../contract.h: this file adds the walk, the status block, the cycle counter and decode_hw (a
// constant-index form of ctr::decode, checked equal to it; validate_hw is contract.h's, unchanged).
//
// The host writes one list of 256-byte records per arena and starts float_top with
//   M = 0xA5 << 24 | step << 23 | first index,  K = list base >> 4,  N = status block >> 4 (0 = none),
// every pointer register (w0..w3, xmem, ymem) at 0, so the AXI address of array index i is 16 i.
// Per command the IP
//   1. fetches the record through io (one 16-beat burst, 64-bit word index) and folds the check word on the fly,
//   2. decodes it (decode_hw: ctr::decode's rules with constant field indices; host builds check it against
//      ctr::decode on every record they fetch) -> ST_BAD_MAGIC / ST_BAD_CHECK / ST_BAD_VERSION / ST_BAD_OPCODE
//      (stop, nothing touched),
//   3. ctr::validate_hw -> ST_MODE_ERROR with the ME_* code in status word 1 (stop, nothing touched),
//   4. stops at END (next = END's index), else dispatches the command to its unit (engine, row unit, softmax:
//      exactly one call site each), which returns only after every write response of its writes (R-ORD),
//   5. stops after a command flagged F_YIELD (ST_YIELD), after one command in step mode (ST_STEP), or when the
//      next index passes MAX_INDEX (ST_LIMIT); a unit that returns non-zero stops it with ST_MODE_ERROR.
// Then it writes the 3-word status block (if N != 0) and returns (next index << 8) | status code.
// M without 0xA5 in [31:24] returns ST_NO_LEGACY: nothing is read or written (there is no per-call protocol).
//
// Units. The integrator supplies a policy class with three static functions of one signature:
//
//   struct Units {
//     template <class WP, class W>
//     static uint32_t gemm(const ctr::Cmd& c, const WP* hp0, const WP* hp1, const WP* hp2, const WP* hp3,
//                          const W* io_rd, W* io_wr);       // OP_GEMM
//     ... row(...)      same signature: OP_QUANT_X, QUANT_W, QUANT_WT, NORM, GELU, SILU_MUL, ROPE, COPY
//     ... softmax(...)  same signature: OP_SOFTMAX
//   };
//
//   - c has passed ctr::validate_hw; c.B >= 1. Addresses are absolute 64-bit byte addresses (< 2^40): index a
//     port as port[addr >> 4] (all address arithmetic in uint64_t or ap_uint<40>, never int).
//   - Each unit function is `#pragma HLS INLINE off` and is called from exactly one place (dispatch() below), so
//     it is one hardware instance. Do not add an ALLOCATION pragma (it broke timing: 5.66 ns).
//   - Return ctr::ME_OK, or an ME_* code for a fault the unit detects itself (the walk stops with ST_MODE_ERROR
//     at this command; the unit may already have written).
//   - Activations are read only through io_rd; hp0..hp3 only for weight blobs, host constants and runtime K/V^T
//     blobs (CONTRACT.md 3.3). Return only after the last write response.
//
// Use inside float_top (fpga/kernels/sequencer/seq_top.cpp is a complete standalone top):
//   int float_top(w0, w1, w2, w3, xmem, ymem, M, K, N) {  <the shipped INTERFACE pragmas>
//     return (int)seq::top<Units>(w0, w1, w2, w3, xmem, ymem, (uint32_t)M, (uint32_t)K, (uint32_t)N);
//   }
//
// Cycle counter (status word 2, lanes 2-3). In synthesis the body is a DATAFLOW region of two control-driven
// processes started together: the walk, and a counter that adds 1 per clock until the walk sends it a stop token
// (every exit path does, NO_LEGACY included), then returns the count; the walk writes it into the status block.
// So the count is the clocks from the start of the region to the end of the walk, within a few clocks. (An
// hls::task free-running counter is rejected by Vitis HLS 2022.2 inside an ap_ctrl_hs top: HLS 200-819.)
// On the host the count is g_host_cycles, which the host units advance (a model, not a measurement).
#pragma once
#include <stdint.h>
#include <string.h>
#ifndef __SYNTHESIS__
#include <stdio.h>
#endif

#ifdef SEQ_CONTRACT_INC  // synthesis experiment only: a patched copy of contract.h found through -I (run_seq_hls.tcl)
#include "contract.h"
#else
#include "../contract.h"
#endif

#ifdef __VITIS_HLS__
#include <ap_int.h>
#endif
#ifndef SEQ_CYCLES
#define SEQ_CYCLES 1  // 0: no cycle counter (status cycles read 0); its cost by difference (seq_top.cpp variant 1)
#endif
#if defined(__SYNTHESIS__) && SEQ_CYCLES
#include <hls_stream.h>
#endif

namespace seq {

// ---------------------------------------------------------------- io words (lane j = bits [32 j + 31 : 32 j])
#ifdef __VITIS_HLS__
typedef ap_uint<128> Word;
static inline uint32_t wget(const Word& w, int j) {
#pragma HLS INLINE
  return (uint32_t)w.range(32 * j + 31, 32 * j);
}
static inline void wset(Word& w, int j, uint32_t v) {
#pragma HLS INLINE
  w.range(32 * j + 31, 32 * j) = v;
}
template <class W>
static inline Word to_word(const W& w) {
#pragma HLS INLINE
  return (Word)w;
}
template <class W>
static inline W from_word(const Word& w) {
#pragma HLS INLINE
  return (W)w;
}
#else
struct Word {
  uint32_t u[4];
};
static inline uint32_t wget(const Word& w, int j) { return w.u[j]; }
static inline void wset(Word& w, int j, uint32_t v) { w.u[j] = v; }
template <class W>  // any 16-byte host word (fstream::IoWord, WPort) has the same bytes
static inline Word to_word(const W& w) {
  static_assert(sizeof(W) == 16, "io words are 128 bits");
  Word r;
  memcpy(&r, &w, 16);
  return r;
}
template <class W>
static inline W from_word(const Word& w) {
  static_assert(sizeof(W) == 16, "io words are 128 bits");
  W r;
  memcpy(&r, &w, 16);
  return r;
}
#endif

// ---------------------------------------------------------------- units
constexpr uint32_t U_NONE = 0;  // NOP (END never dispatches)
constexpr uint32_t U_GEMM = 1;
constexpr uint32_t U_ROW = 2;
constexpr uint32_t U_SMX = 3;

static inline uint32_t unit_of(uint32_t op) {
#pragma HLS INLINE
  if (op == ctr::OP_GEMM) return U_GEMM;
  if (op == ctr::OP_SOFTMAX) return U_SMX;
  if (op == ctr::OP_QUANT_X || op == ctr::OP_QUANT_W || op == ctr::OP_QUANT_WT || op == ctr::OP_NORM ||
      op == ctr::OP_GELU || op == ctr::OP_SILU_MUL || op == ctr::OP_ROPE || op == ctr::OP_COPY)
    return U_ROW;
  return U_NONE;
}

// The single dispatch site: one call per unit function.
template <class U, class WP, class W>
static inline uint32_t dispatch(const ctr::Cmd& c, const WP* w0, const WP* w1, const WP* w2, const WP* w3,
                                const W* xmem, W* ymem) {
#pragma HLS INLINE
  uint32_t r = ctr::ME_OK;
  switch (unit_of(c.op)) {
    case U_GEMM:
      r = U::gemm(c, w0, w1, w2, w3, xmem, ymem);
      break;
    case U_ROW:
      r = U::row(c, w0, w1, w2, w3, xmem, ymem);
      break;
    case U_SMX:
      r = U::softmax(c, w0, w1, w2, w3, xmem, ymem);
      break;
    default:  // OP_NOP
      break;
  }
  return r;
}

// ---------------------------------------------------------------- record fetch, decode, check
// One record = REC_WORDS io words at word index `word` (64-bit), shifted into rec[] (rec[w] = word w; constant
// indices only, no demultiplexer). xsum = the XOR of all 64 fields, folded 4 lanes per beat: the check word is
// right iff (CHECK_SEED ^ xsum) == 0, i.e. iff ctr::rec_check(f) == f[CHECK_FIELD] (4 XORs per beat instead of
// one 63-input tree after the fetch).
template <class W>
static void fetch(const W* io, uint64_t word, Word rec[ctr::REC_WORDS], uint32_t& xsum) {
#pragma HLS INLINE
  uint32_t x = 0;
fetch_rec:
  for (uint32_t w = 0; w < ctr::REC_WORDS; w++) {
#pragma HLS PIPELINE II = 1
    const Word v = to_word(io[word + w]);
    x ^= wget(v, 0) ^ wget(v, 1) ^ wget(v, 2) ^ wget(v, 3);
    rec[w] = v;
  }
  xsum = x;
}

// field j of a fetched record (j is a constant after unrolling: lane j % 4 of word j / 4)
static inline uint32_t fld(const Word rec[ctr::REC_WORDS], uint32_t j) {
#pragma HLS INLINE
  return wget(rec[j / 4], (int)(j % 4));
}

// ctr::decode() of a fetched record, as hardware: the same status in the same order (BAD_MAGIC, BAD_CHECK,
// BAD_VERSION, BAD_OPCODE, else ST_END = 0) and, when that is ST_END or ST_BAD_OPCODE, the same Cmd. Every field
// is unpacked first, whatever the status (so c.tag and c.op are the raw record's on every path). ctr::decode's
// field loops index the record with a loop variable, which HLS builds as 8:1 / 16:1 multiplexers plus one phi
// copy of the Cmd per loop (HLS of the earlier version: 2.0k LUT of XOR chain, 10.4k FF in the walk); here every
// index is a constant. Host builds compare it with ctr::decode on every record they fetch (check()).
static inline uint32_t decode_hw(const Word rec[ctr::REC_WORDS], uint32_t xsum, ctr::Cmd& c) {
#pragma HLS INLINE
  using namespace ctr;
  const uint32_t f0 = fld(rec, FLD_OP);
  c.op = f0 & 0xFF;
  c.flags = f0 >> 16;
  c.tag = fld(rec, FLD_TAG);
  c.M = fld(rec, FLD_M);
  c.K = fld(rec, FLD_K);
  c.N = fld(rec, FLD_N);
  c.B = fld(rec, FLD_B) == 0 ? 1u : fld(rec, FLD_B);
  for (uint32_t s = 0; s < NSLOTS; s++) {
#pragma HLS UNROLL
    c.s[s].addr = (uint64_t)fld(rec, SLOT_BASE + SLOT_FIELDS * s) |
                  ((uint64_t)fld(rec, SLOT_BASE + SLOT_FIELDS * s + 1) << 32);
    c.s[s].pitch = fld(rec, SLOT_BASE + SLOT_FIELDS * s + 2);
    c.s[s].stride = fld(rec, SLOT_BASE + SLOT_FIELDS * s + 3);
  }
  for (uint32_t i = 0; i < NPARAMS; i++) {
#pragma HLS UNROLL
    c.p[i] = fld(rec, PARAM_BASE + i);
  }
  if (((f0 >> 8) & 0xFF) != REC_MAGIC) return ST_BAD_MAGIC;
  if ((xsum ^ CHECK_SEED) != 0) return ST_BAD_CHECK;
  if (fld(rec, FLD_VERSION) != CONTRACT_VERSION) return ST_BAD_VERSION;
  if (!known_op(c.op)) return ST_BAD_OPCODE;
  return ST_END;
}

#ifndef __SYNTHESIS__
extern uint64_t g_decode_checked, g_decode_mismatch;  // host counters (defined in seq_top.cpp)
// Host builds: decode_hw against ctr::decode on one fetched record (the status, and every Cmd field when the
// record decodes or is BAD_OPCODE). Returns false and prints the record on any difference.
static inline bool decode_matches(const Word rec[ctr::REC_WORDS], uint32_t st_hw, const ctr::Cmd& hw) {
  uint32_t f[ctr::REC_FIELDS];
  for (uint32_t w = 0; w < ctr::REC_WORDS; w++)
    for (int j = 0; j < 4; j++) f[4 * w + j] = wget(rec[w], j);
  ctr::Cmd r;
  memset(&r, 0, sizeof r);
  const uint32_t st = ctr::decode(f, r);
  bool ok = st == st_hw;
  if (ok && (st == ctr::ST_END || st == ctr::ST_BAD_OPCODE)) {
    ok = r.op == hw.op && r.flags == hw.flags && r.tag == hw.tag && r.M == hw.M && r.K == hw.K && r.N == hw.N &&
         r.B == hw.B;
    for (uint32_t s = 0; s < ctr::NSLOTS; s++)
      ok = ok && r.s[s].addr == hw.s[s].addr && r.s[s].pitch == hw.s[s].pitch && r.s[s].stride == hw.s[s].stride;
    for (uint32_t i = 0; i < ctr::NPARAMS; i++) ok = ok && r.p[i] == hw.p[i];
  }
  ok = ok && hw.tag == f[ctr::FLD_TAG] && hw.op == (f[ctr::FLD_OP] & 0xFF);  // what the status block reports
  if (!ok) {
    printf("seq::decode_hw != ctr::decode: status %u vs %u, record", st_hw, st);
    for (uint32_t j = 0; j < ctr::REC_FIELDS; j++) printf(" %08x", f[j]);
    printf("\n");
  }
  return ok;
}
#endif

// decode + validate_hw of one fetched record: returns 0 (dispatch c) or the status code that stops the walk;
// me = the ME_* code when that status is ST_MODE_ERROR.
static inline uint32_t check(const Word rec[ctr::REC_WORDS], uint32_t xsum, ctr::Cmd& c, uint32_t& me) {
#pragma HLS INLINE
  me = ctr::ME_OK;
  const uint32_t st = decode_hw(rec, xsum, c);
#ifndef __SYNTHESIS__
  g_decode_checked++;
  if (!decode_matches(rec, st, c)) g_decode_mismatch++;
#endif
  if (st != ctr::ST_END) return st;
  me = ctr::validate_hw(c);
  return me != ctr::ME_OK ? ctr::ST_MODE_ERROR : 0u;
}

template <class W>
static void write_status(W* io, uint64_t word, const uint32_t s[4 * ctr::STATUS_WORDS]) {
#pragma HLS INLINE off
status_wr:
  for (uint32_t w = 0; w < ctr::STATUS_WORDS; w++) {
#pragma HLS PIPELINE II = 1
    Word v;
    for (int j = 0; j < 4; j++) {
#pragma HLS UNROLL
      wset(v, j, s[4 * w + j]);
    }
    io[word + w] = from_word<W>(v);
  }
}

// ---------------------------------------------------------------- cycle counter
#if defined(__SYNTHESIS__) && SEQ_CYCLES
struct CycleFifo {
  hls::stream<uint8_t, 2> stop;
  hls::stream<uint64_t, 2> count;
};
static void counter_proc(hls::stream<uint8_t, 2>& stop, hls::stream<uint64_t, 2>& count) {
#pragma HLS INLINE off
  uint64_t c = 0;
count_clk:
  while (true) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 200000000
    uint8_t t;
    if (stop.read_nb(t)) break;
    c++;
  }
  count.write(c);
}
static inline uint64_t cycles_stop(hls::stream<uint8_t, 2>& stop, hls::stream<uint64_t, 2>& count) {
#pragma HLS INLINE
  stop.write(1);
  return count.read();
}
#define SEQ_CYC_ARGS , hls::stream<uint8_t, 2>&cyc_stop, hls::stream<uint64_t, 2>&cyc_count
#define SEQ_CYC_PASS , cyc_stop, cyc_count
#define SEQ_CYC_STOP() cycles_stop(cyc_stop, cyc_count)
#elif defined(__SYNTHESIS__)
#define SEQ_CYC_ARGS
#define SEQ_CYC_PASS
#define SEQ_CYC_STOP() ((uint64_t)0)
#else
extern uint64_t g_host_cycles;  // host model of the clock: advanced by the walk (fetch) and by host units
#define SEQ_CYC_ARGS
#define SEQ_CYC_PASS
#define SEQ_CYC_STOP() (g_host_cycles - host_t0)
#endif
// host model only: clocks per record in csynth (cmd_loop iteration, sequencer alone at 4.0 ns: fetch 19,
// validate_hw 39, decode + walk + status the rest; README "Synthesis"). DDR read latency comes on top on the board.
constexpr uint64_t HOST_FETCH_CYCLES = 72;

// ---------------------------------------------------------------- the walk
template <class U, class WP, class W>
static uint32_t walk(const WP* w0, const WP* w1, const WP* w2, const WP* w3, const W* xmem, W* ymem, uint32_t M,
                     uint32_t K, uint32_t N SEQ_CYC_ARGS) {
#pragma HLS INLINE off
  static uint32_t starts = 0;  // list starts since configuration (a stale status block is visible)
#ifndef __SYNTHESIS__
  const uint64_t host_t0 = g_host_cycles;
#endif
  if ((M >> 24) != ctr::MODE_MAGIC) {  // no per-call protocol: touch nothing
    (void)SEQ_CYC_STOP();
    return ctr::ST_NO_LEGACY;
  }
  starts++;
  const bool step = (M >> ctr::MODE_STEP_BIT) & 1;
  const uint64_t list = K;  // list base, 16-byte words (the list sits below 64 GB)
  uint32_t idx = M & ctr::MAX_INDEX;
  uint32_t code = ctr::ST_END, tag = 0, done = 0, merr = ctr::ME_OK, lastop = 0, last_fin = 0xFFFFFFFFu;
cmd_loop:
  while (true) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 2701
    Word rec[ctr::REC_WORDS];
#pragma HLS ARRAY_PARTITION variable = rec complete
    uint32_t xsum;
    fetch(xmem, list + (uint64_t)idx * ctr::REC_WORDS, rec, xsum);
#ifndef __SYNTHESIS__
    g_host_cycles += HOST_FETCH_CYCLES;
#endif
    ctr::Cmd c;
#pragma HLS ARRAY_PARTITION variable = c.s complete
#pragma HLS ARRAY_PARTITION variable = c.p complete
    uint32_t me;
    const uint32_t st = check(rec, xsum, c, me);
    if (st != 0) {  // bad record or mode error: nothing dispatched, idx stays on it (tag, opcode: the raw record's)
      code = st;
      merr = me;
      tag = c.tag;
      lastop = c.op;
      break;
    }
    if (c.op == ctr::OP_END) {
      code = ctr::ST_END;
      tag = c.tag;
      lastop = c.op;
      break;
    }
    const uint32_t r = dispatch<U>(c, w0, w1, w2, w3, xmem, ymem);
    done++;
    tag = c.tag;
    lastop = c.op;
    if (r != ctr::ME_OK) {
      code = ctr::ST_MODE_ERROR;
      merr = r;
      break;
    }
    last_fin = idx;
    idx++;
    if (c.flags & ctr::F_YIELD) {
      code = ctr::ST_YIELD;
      break;
    }
    if (step) {
      code = ctr::ST_STEP;
      break;
    }
    if (idx > ctr::MAX_INDEX) {
      code = ctr::ST_LIMIT;
      break;
    }
  }
  const uint64_t dt = SEQ_CYC_STOP();
  if (N != 0) {
    uint32_t s[4 * ctr::STATUS_WORDS];
#pragma HLS ARRAY_PARTITION variable = s complete
    s[0] = code;
    s[1] = idx;
    s[2] = tag;
    s[3] = done;
    s[4] = merr;
    s[5] = ctr::STATUS_MAGIC;
    s[6] = starts;
    s[7] = lastop;
    s[8] = ctr::CONTRACT_VERSION;
    s[9] = last_fin;
    s[10] = (uint32_t)dt;
    s[11] = (uint32_t)(dt >> 32);
    write_status(ymem, (uint64_t)N, s);
  }
  return (idx << 8) | code;
}

// The body of float_top. Synthesis: a DATAFLOW region of the walk (the one process that touches the ports) and the
// cycle counter. Host (and SEQ_CYCLES = 0): the walk alone.
#if defined(__SYNTHESIS__) && SEQ_CYCLES
template <class U, class WP, class W>
static void walk_proc(const WP* w0, const WP* w1, const WP* w2, const WP* w3, const W* xmem, W* ymem, uint32_t M,
                      uint32_t K, uint32_t N, hls::stream<uint8_t, 2>& cyc_stop, hls::stream<uint64_t, 2>& cyc_count,
                      uint32_t& ret) {
#pragma HLS INLINE off
  ret = walk<U>(w0, w1, w2, w3, xmem, ymem, M, K, N, cyc_stop, cyc_count);
}
template <class U, class WP, class W>
static void region(const WP* w0, const WP* w1, const WP* w2, const WP* w3, const W* xmem, W* ymem, uint32_t M,
                   uint32_t K, uint32_t N, uint32_t& ret) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW
  hls::stream<uint8_t, 2> cyc_stop;
  hls::stream<uint64_t, 2> cyc_count;
  counter_proc(cyc_stop, cyc_count);
  walk_proc<U>(w0, w1, w2, w3, xmem, ymem, M, K, N, cyc_stop, cyc_count, ret);
}
#endif

template <class U, class WP, class W>
static uint32_t top(const WP* w0, const WP* w1, const WP* w2, const WP* w3, const W* xmem, W* ymem, uint32_t M,
                    uint32_t K, uint32_t N) {
#pragma HLS INLINE
#if defined(__SYNTHESIS__) && SEQ_CYCLES
  uint32_t ret;
  region<U>(w0, w1, w2, w3, xmem, ymem, M, K, N, ret);
  return ret;
#else
  return walk<U>(w0, w1, w2, w3, xmem, ymem, M, K, N);
#endif
}

}  // namespace seq
