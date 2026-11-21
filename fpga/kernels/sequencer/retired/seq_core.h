// Command sequencer core for float_top (header-only; Vitis HLS and g++).
//
// The ARM writes a command list (seq_format.h records) into CMA once, then starts float_top with
// M = MODE_MAGIC << 24 | first index, K = list base >> 4, N = status block >> 4 and every pointer
// register at 0. The IP fetches one 128-byte record through the io port (an 8-beat burst), checks it
// (magic, XOR check word, opcode), and calls the mode's function from a single dispatch site; it stops
// at END, after a command flagged F_YIELD, after one command in single-step mode, or at the first
// error, writes the status block and returns (next index << 8) | status code in ap_return. ap_done
// is the completion event (the HLS `interrupt` pin exists on the IP; see README for the BD).
//
// Usage inside float_top (see seq_top.cpp for a complete top with stub modes):
//
//   struct Modes {   // ONE call site per mode function: one hardware instance each
//     template <class WP, class W>
//     static uint32_t dispatch(const seq::Cmd& c, const WP* w0, ..., const W* xmem, W* ymem) {
//       switch (c.op) { case seq::OP_GEMM: return gemm_mode(c, ...); ... }
//     }
//   };
//   return seq::run<Modes>(w0, w1, w2, w3, xmem, ymem, M, K, N, legacy_cmd);
//
// The legacy (one-call) register protocol keeps working: when M[31:24] != MODE_MAGIC, run() executes
// the caller-built `legacy` command once through the same dispatch site and returns the mode's code.
#pragma once
#include <stdint.h>
#include <string.h>

#include "seq_format.h"

#ifdef __VITIS_HLS__
#include <ap_int.h>
#endif

namespace seq {

#ifdef __VITIS_HLS__
typedef ap_uint<128> Word;
typedef ap_uint<40> addr_t;
static inline uint32_t wget(const Word& w, int j) {
#pragma HLS INLINE
  return (uint32_t)w.range(32 * j + 31, 32 * j);
}
static inline void wset(Word& w, int j, uint32_t v) {
#pragma HLS INLINE
  w.range(32 * j + 31, 32 * j) = v;
}
static inline Word to_word(const Word& w) {
#pragma HLS INLINE
  return w;
}
static inline Word zword() {
#pragma HLS INLINE
  return Word(0);
}
#else
struct Word {
  uint32_t u[4];
};
typedef uint64_t addr_t;
static inline Word zword() { return Word{{0, 0, 0, 0}}; }
static inline uint32_t wget(const Word& w, int j) { return w.u[j]; }
static inline void wset(Word& w, int j, uint32_t v) { w.u[j] = v; }
template <class W>  // any 16-byte host word (e.g. fstream::IoWord = float[4]) has the same bytes
static inline Word to_word(const W& w) {
  static_assert(sizeof(W) == 16, "io words are 128 bits");
  Word r;
  memcpy(&r, &w, 16);
  return r;
}
template <class W>
static inline W from_word(const Word& w) {
  W r;
  memcpy(&r, &w, 16);
  return r;
}
#endif

struct Opnd {
  addr_t addr;      // physical byte address (40 bits)
  uint32_t pitch;   // row pitch in bytes (24 bits; 0 = broadcast one row where a mode allows it)
  uint32_t stride;  // batch stride in bytes
};

struct Cmd {
  uint32_t op, flags, tag;
  uint32_t M, K, N, B, G;
  Opnd a, b, c, d, e, f;
  uint32_t p0, p1, p2, p3, p4, p5, p6;
};

static inline bool valid_op(uint32_t op) {
#pragma HLS INLINE
  return op == OP_NOP || op == OP_END || op == OP_GEMM || op == OP_QUANT_X || op == OP_QUANT_W ||
         op == OP_QUANT_WT || op == OP_NORM || op == OP_SOFTMAX || op == OP_GELU || op == OP_SILU_MUL ||
         op == OP_ROPE || op == OP_COPY || op == OP_GATHER || op == OP_AXPY;
}

static inline Opnd slot(const uint32_t f[REC_FIELDS], int s) {
#pragma HLS INLINE
  Opnd o;
  const uint32_t w0 = f[SLOT_BASE + SLOT_FIELDS * s], w1 = f[SLOT_BASE + SLOT_FIELDS * s + 1];
#ifdef __VITIS_HLS__
  addr_t a = 0;
  a.range(31, 0) = w0;
  a.range(39, 32) = w1 & 0xFF;
  o.addr = a;
#else
  o.addr = (uint64_t)w0 | ((uint64_t)(w1 & 0xFF) << 32);
#endif
  o.pitch = w1 >> 8;
  o.stride = f[SLOT_BASE + SLOT_FIELDS * s + 2];
  return o;
}

// Unpack and check one record. Returns 0 or ST_BAD_MAGIC / ST_BAD_CHECK / ST_BAD_OPCODE.
static uint32_t decode(const Word rec[REC_WORDS], Cmd& c) {
#pragma HLS INLINE
  uint32_t f[REC_FIELDS];
#pragma HLS ARRAY_PARTITION variable = f complete
  for (int j = 0; j < (int)REC_FIELDS; j++) {
#pragma HLS UNROLL
    f[j] = wget(rec[j / 4], j % 4);
  }
  uint32_t x = SEQ_CHECK_SEED;
  for (int j = 0; j < (int)CHECK_FIELD; j++) {
#pragma HLS UNROLL
    x ^= f[j];
  }
  c.op = f[0] & 0xFF;
  c.flags = f[0] >> 16;
  c.tag = f[1];
  c.M = f[2];
  c.K = f[3];
  c.N = f[4];
  c.B = (f[5] & 0xFFFF) ? (f[5] & 0xFFFF) : 1;
  c.G = (f[5] >> 16) ? (f[5] >> 16) : 1;
  c.a = slot(f, 0);
  c.b = slot(f, 1);
  c.c = slot(f, 2);
  c.d = slot(f, 3);
  c.e = slot(f, 4);
  c.f = slot(f, 5);
  c.p0 = f[PARAM_BASE + 0];
  c.p1 = f[PARAM_BASE + 1];
  c.p2 = f[PARAM_BASE + 2];
  c.p3 = f[PARAM_BASE + 3];
  c.p4 = f[PARAM_BASE + 4];
  c.p5 = f[PARAM_BASE + 5];
  c.p6 = f[PARAM_BASE + 6];
  if (((f[0] >> 8) & 0xFF) != SEQ_MAGIC) return ST_BAD_MAGIC;
  if (x != f[CHECK_FIELD]) return ST_BAD_CHECK;
  if (!valid_op(c.op)) return ST_BAD_OPCODE;
  return 0;
}

// One record = one 8-beat burst on the io port.
template <class W>
static void fetch(const W* mem, addr_t word_index, Word rec[REC_WORDS]) {
#pragma HLS INLINE off
fetch_rec:
  for (int i = 0; i < (int)REC_WORDS; i++) {
#pragma HLS PIPELINE II = 1
    rec[i] = to_word(mem[word_index + i]);
  }
}

template <class W>
static void write_status(W* mem, addr_t word_index, uint32_t code, uint32_t next, uint32_t tag, uint32_t done,
                         uint32_t merr, uint32_t starts, uint32_t lastop) {
#pragma HLS INLINE off
  Word s0 = zword(), s1 = zword();
  wset(s0, 0, code);
  wset(s0, 1, next);
  wset(s0, 2, tag);
  wset(s0, 3, done);
  wset(s1, 0, merr);
  wset(s1, 1, STATUS_MAGIC);
  wset(s1, 2, starts);
  wset(s1, 3, lastop);
#ifdef __VITIS_HLS__
  mem[word_index] = s0;
  mem[word_index + 1] = s1;
#else
  mem[word_index] = from_word<W>(s0);
  mem[word_index + 1] = from_word<W>(s1);
#endif
}

// The sequencer. Returns ap_return: sequencer mode (next index << 8) | status code; legacy mode the
// mode's own return value. `legacy` is the command the caller decoded from its legacy registers.
template <class Modes, class WP, class W>
static uint32_t run(const WP* w0, const WP* w1, const WP* w2, const WP* w3, const W* xmem, W* ymem, uint32_t M,
                    uint32_t K, uint32_t N, const Cmd& legacy) {
  static uint32_t starts = 0;  // sequencer starts since configuration (lets the host spot a stale status)
  const bool seqmode = (M >> 24) == MODE_MAGIC;
  const bool step = (M >> MODE_STEP_BIT) & 1;
  uint32_t idx = M & MAX_INDEX;
  const addr_t list = K;  // list base in 16-byte words
  uint32_t code = ST_END, tag = 0, done = 0, merr = 0, lastop = 0, ret = 0;
  if (seqmode) starts++;
cmd_loop:
  while (true) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 4096
    Cmd c;
    if (seqmode) {
      Word rec[REC_WORDS];
#pragma HLS ARRAY_PARTITION variable = rec complete
      fetch(xmem, list + (addr_t)idx * REC_WORDS, rec);
      const uint32_t e = decode(rec, c);
      if (e) {
        code = e;
        tag = c.tag;
        break;
      }
    } else {
      c = legacy;
    }
    if (c.op == OP_END) {
      code = ST_END;
      tag = c.tag;
      break;
    }
    const uint32_t r = Modes::dispatch(c, w0, w1, w2, w3, xmem, ymem);  // the single dispatch site
    done++;
    tag = c.tag;
    lastop = c.op;
    ret = r;
    if (r) {
      code = ST_MODE_ERROR;
      merr = r;
      break;  // idx stays on the failing command
    }
    idx++;
    if (!seqmode) {
      code = ST_LEGACY;
      break;
    }
    if (c.flags & F_YIELD) {
      code = ST_YIELD;
      break;
    }
    if (step) {
      code = ST_STEP;
      break;
    }
    if (idx > MAX_INDEX) {
      code = ST_LIMIT;
      break;
    }
  }
  if (!seqmode) return ret;
  if (N) write_status(ymem, (addr_t)N, code, idx, tag, done, merr, starts, lastop);
  return (idx << 8) | code;
}

}  // namespace seq
