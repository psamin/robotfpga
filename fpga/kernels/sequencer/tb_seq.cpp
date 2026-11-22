// Host testbench of the float_top sequencer, contract 1.0 (seq_core.h + seq_top.cpp's top with host units).
//
//   export PATH="/c/Xilinx/Vitis_HLS/2022.2/tps/win64/msys64/mingw64/bin:/c/Xilinx/Vitis_HLS/2022.2/tps/win64/msys64/usr/bin:/c/Windows/System32:/c/Windows"
//   g++ -std=c++14 -O2 -ffp-contract=off -D__USE_MINGW_ANSI_STDIO=1 -Wall -Wno-unknown-pragmas -Wno-unused-label //       -static -o build/tb_seq.exe tb_seq.cpp
//   build/tb_seq.exe [--twin DIR] [CHUNK.bin TRACE.txt BASE_HEX ...]        (all of it: sh run_tests.sh)
//
// 1. Walk, every path, at memory base 0 and at 0x8_0000_0000 (64-bit word indices; status and list above 4 GB):
//    END, YIELD + resume, STEP, LIMIT (index 0x7FFFFF), NO_LEGACY (nothing read or written), BAD_MAGIC (incl. a
//    128-byte draft record), BAD_CHECK, BAD_VERSION, BAD_OPCODE (incl. the retired GATHER / AXPY), MODE_ERROR from
//    validate_hw (DIMS, ALIGN, ADDR, FLAGS, PARAM, SLOT: nothing dispatched, nothing written), MODE_ERROR from a
//    unit, no status block. Status words 0-2 checked field by field.
// 2. Dispatch: one record of every opcode goes to its unit exactly once, in list order, with every decoded field.
// 3. Data: QUANT_X -> GEMM (row groups, items, bias, D, ACC) -> COPY through host units written from the contract
//    (ctr::ref::quant_row, ctr::ref::gemm_out), compared with a direct computation, at both bases.
// 4. The -12392 chunk lists (CHUNK.bin from make_chunk.py, TRACE.txt = the Python decode of the same bytes):
//    one start (dry units) must reach END at the last index with every command dispatched, and every dispatched
//    command must print the line of TRACE.txt; then the same list in step mode (one start per command) and with
//    F_YIELD set on every 97th record (resealed) must dispatch the identical sequence.
// 5. decode_hw (the sequencer's hardware decode) against ctr::decode on 400,000 records: valid records of every
//    opcode with random fields, then 1-3 random bit flips in random fields, resealed or not, plus unknown and retired
//    opcodes and stale versions. Every record that any walk above fetches is also checked against ctr::decode
//    inside seq_core.h (g_decode_checked / g_decode_mismatch).
// --twin DIR writes every start of 1 and 3 (memory written before it, registers, result, dispatch trace, output
// bytes) for check_twin.py, which replays them through exec_ref.py's walk and units.
#if defined(__MINGW32__) && !defined(__USE_MINGW_ANSI_STDIO)
#define __USE_MINGW_ANSI_STDIO 1  // %llu / %llx in printf (MinGW's default msvcrt printf does not know them)
#endif
#include "seq_top.cpp"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <random>
#include <string>
#include <vector>

using namespace ctr;

static int g_fail = 0, g_checks = 0;
#define CHECK(cond, ...)                          \
  do {                                            \
    g_checks++;                                   \
    if (!(cond)) {                                \
      g_fail++;                                   \
      printf("FAIL %s:%d: ", __FILE__, __LINE__); \
      printf(__VA_ARGS__);                        \
      printf("\n");                               \
    }                                             \
  } while (0)

// ---------------------------------------------------------------- memory: AXI address = byte address
// g_mem holds [g_base, g_base + size); the port pointers are g_mem.data() - g_base / 16 words, so a unit's
// port[addr >> 4] lands in g_mem for every addr in that window (64-bit arithmetic all the way).
static std::vector<seq::Word> g_mem;
static uint64_t g_base = 0;
static uint8_t* bp(uint64_t a) {
  if (a < g_base || a >= g_base + g_mem.size() * 16) {
    printf("FATAL: host access 0x%llx outside [0x%llx, +0x%llx)\n", (unsigned long long)a,
           (unsigned long long)g_base, (unsigned long long)(g_mem.size() * 16));
    exit(2);
  }
  return (uint8_t*)g_mem.data() + (a - g_base);
}
static seq::Word* port() { return (seq::Word*)((uintptr_t)g_mem.data() - (uintptr_t)g_base); }

// twin log: every write of the test to memory before a start
struct Chunk {
  uint64_t addr;
  std::string bytes;
};
static std::vector<Chunk> g_writes;
static void put(uint64_t a, const void* p, size_t n) {
  memcpy(bp(a), p, n);
  g_writes.push_back({a, std::string((const char*)p, n)});
}
static void put_u32(uint64_t a, uint32_t v) { put(a, &v, 4); }
static void put_f32(uint64_t a, float v) { put(a, &v, 4); }
static void put_i16(uint64_t a, int16_t v) { put(a, &v, 2); }
static uint32_t get_u32(uint64_t a) {
  uint32_t v;
  memcpy(&v, bp(a), 4);
  return v;
}
static float get_f32(uint64_t a) {
  float v;
  memcpy(&v, bp(a), 4);
  return v;
}
static int16_t get_i16(uint64_t a) {
  int16_t v;
  memcpy(&v, bp(a), 2);
  return v;
}

// ---------------------------------------------------------------- dispatch trace (same text as check_twin.py)
static std::vector<std::string> g_trace;
static std::string trace_line(const Cmd& c, uint32_t unit) {
  char buf[1024];
  int n = snprintf(buf, sizeof buf, "op=%02x unit=%u flags=%04x tag=%08x M=%u K=%u N=%u B=%u", c.op, unit, c.flags,
                   c.tag, c.M, c.K, c.N, c.B);
  for (uint32_t s = 0; s < NSLOTS; s++)
    n += snprintf(buf + n, sizeof buf - n, " %llx/%x/%x", (unsigned long long)c.s[s].addr, c.s[s].pitch,
                  c.s[s].stride);
  for (uint32_t i = 0; i < NPARAMS; i++) n += snprintf(buf + n, sizeof buf - n, " %x", c.p[i]);
  return buf;
}

// ---------------------------------------------------------------- host units (contract arithmetic)
static bool g_dry = false;      // trace only
static uint32_t g_fault = 0;    // the next unit call returns this ME code (unit-fault path)
static const uint8_t* B8(const seq::Word* p, uint64_t a) { return (const uint8_t*)p + a; }
static uint8_t* B8w(seq::Word* p, uint64_t a) { return (uint8_t*)p + a; }
static float rf(const seq::Word* p, uint64_t a) {
  float v;
  memcpy(&v, B8(p, a), 4);
  return v;
}
static void wf(seq::Word* p, uint64_t a, float v) { memcpy(B8w(p, a), &v, 4); }

static uint32_t host_gemm(const Cmd& c, const seq::Word* const w[4], const seq::Word* xr, seq::Word* xw) {
  const Slot &A = c.s[SLOT_A], &Bw = c.s[SLOT_B], &Cy = c.s[SLOT_C], &D = c.s[SLOT_D], &E = c.s[SLOT_E],
             &F = c.s[SLOT_F], &G = c.s[SLOT_G];
  const uint64_t R = c.p[P_SEGC], P1 = c.p[P_SEGF];
  std::vector<float> y((size_t)c.N * c.M);
  for (uint64_t b = 0; b < c.B; b++) {
    for (uint64_t n = 0; n < c.N; n++)
      for (uint64_t m = 0; m < c.M; m++) {
        int64_t acc = 0;
        for (uint64_t k = 0; k < c.K; k++) {
          int16_t x, wv;
          memcpy(&x, B8(xr, row_addr(A, b, n) + 2 * k), 2);
          memcpy(&wv, B8(w[(m % ENGINE_LANES) / 8], w_word_addr(Bw, b, m, k, c.K) + 2 * w_lane(m)), 2);
          acc += (int64_t)x * wv;
        }
        const float srow = rf(w[0], E.addr + b * E.stride + 4 * m), stok = rf(xr, F.addr + b * F.stride + 4 * n);
        float bias = 0, add = 0;
        if (c.flags & F_BIAS) bias = rf(w[0], G.addr + b * G.stride + 4 * m);
        if (c.flags & F_ADD_D) add = rf(xr, row_addr(D, b, n) + 4 * m);
        if (c.flags & F_ACC) add = rf(xr, row_addr2(Cy, b, n, R, P1) + 4 * m);
        y[n * c.M + m] = ref::gemm_out(acc, srow, stok, (c.flags & F_BIAS) ? &bias : nullptr,
                                       (c.flags & (F_ADD_D | F_ACC)) ? &add : nullptr);
      }
    for (uint64_t n = 0; n < c.N; n++)  // write after every read of the item (C may alias D)
      for (uint64_t m = 0; m < c.M; m++) wf(xw, row_addr2(Cy, b, n, R, P1) + 4 * m, y[n * c.M + m]);
  }
  seq::g_host_cycles += 400 + (uint64_t)c.B * c.N * c.M * c.K / 1024;
  return ME_OK;
}

static uint32_t host_row(const Cmd& c, const seq::Word* xr, seq::Word* xw) {
  const Slot &A = c.s[SLOT_A], &Cs = c.s[SLOT_C], &F = c.s[SLOT_F];
  if (c.op == OP_COPY) {
    for (uint64_t b = 0; b < c.B; b++)
      for (uint64_t n = 0; n < c.N; n++) memcpy(B8w(xw, row_addr(Cs, b, n)), B8(xr, row_addr(A, b, n)), 4 * c.K);
  } else if (c.op == OP_QUANT_X) {
    const uint64_t seg = c.p[P_SEG] ? c.p[P_SEG] : c.K, nseg = c.K / seg, np = c.p[P_NP] ? c.p[P_NP] : c.N;
    const int shift = (int32_t)c.p[P_SHIFT];
    std::vector<float> x(seg);
    std::vector<int16_t> q(seg);
    for (uint64_t b = 0; b < c.B; b++)
      for (uint64_t j = 0; j < nseg; j++) {
        for (uint64_t n = 0; n < pad4((uint32_t)np); n++) {
          float s = 1.0f;
          if (n < c.N) {
            memcpy(x.data(), B8(xr, row_addr(A, b, n) + 4 * j * seg), 4 * seg);
            s = ref::quant_row(x.data(), (uint32_t)seg, q.data(), shift);
          } else {
            std::fill(q.begin(), q.end(), 0);
          }
          if (n < np) memcpy(B8w(xw, seg_addr(Cs, b, n, j, c.p[P_SEGC])), q.data(), 2 * seg);
          wf(xw, scale_addr(F, b, j, c.p[P_SEGF], n), s);
        }
      }
  } else {
    return ME_PARAM;  // the testbench runs the other row ops dry only
  }
  seq::g_host_cycles += 480 + (uint64_t)c.B * c.N * c.K / 4;
  return ME_OK;
}

struct HostUnits {
  template <class WP, class W>
  static uint32_t call(uint32_t unit, const Cmd& c, const WP* w0, const WP* w1, const WP* w2, const WP* w3,
                       const W* xr, W* xw) {
    g_trace.push_back(trace_line(c, unit));
    if (g_fault) {
      const uint32_t r = g_fault;
      g_fault = 0;
      return r;
    }
    if (g_dry) return ME_OK;
    const seq::Word* w[4] = {w0, w1, w2, w3};
    if (unit == seq::U_GEMM) return host_gemm(c, w, xr, xw);
    if (unit == seq::U_ROW) return host_row(c, xr, xw);
    return ME_PARAM;  // softmax: dry only
  }
  template <class WP, class W>
  static uint32_t gemm(const Cmd& c, const WP* w0, const WP* w1, const WP* w2, const WP* w3, const W* xr, W* xw) {
    return call(seq::U_GEMM, c, w0, w1, w2, w3, xr, xw);
  }
  template <class WP, class W>
  static uint32_t row(const Cmd& c, const WP* w0, const WP* w1, const WP* w2, const WP* w3, const W* xr, W* xw) {
    return call(seq::U_ROW, c, w0, w1, w2, w3, xr, xw);
  }
  template <class WP, class W>
  static uint32_t softmax(const Cmd& c, const WP* w0, const WP* w1, const WP* w2, const WP* w3, const W* xr,
                          W* xw) {
    return call(seq::U_SMX, c, w0, w1, w2, w3, xr, xw);
  }
};

// ---------------------------------------------------------------- records
static void store(uint64_t list, uint32_t idx, const Cmd& c) {
  uint32_t f[REC_FIELDS];
  encode(c, f);
  put(list + (uint64_t)REC_BYTES * idx, f, REC_BYTES);
}
static void reseal(uint64_t list, uint32_t idx) {  // recompute the check word after a deliberate edit
  uint32_t f[REC_FIELDS];
  memcpy(f, bp(list + (uint64_t)REC_BYTES * idx), REC_BYTES);
  f[CHECK_FIELD] = rec_check(f);
  put(list + (uint64_t)REC_BYTES * idx, f, REC_BYTES);
}
static void poke(uint64_t list, uint32_t idx, uint32_t field, uint32_t v, bool seal = true) {
  put_u32(list + (uint64_t)REC_BYTES * idx + 4 * field, v);
  if (seal) reseal(list, idx);
}
static Cmd simple(uint32_t op, uint32_t tag, uint32_t flags = 0) {
  Cmd c;
  memset(&c, 0, sizeof c);
  c.op = op;
  c.tag = tag;
  c.flags = flags;
  c.B = 0;  // encodes as 0 (= 1 item)
  return c;
}
static Cmd copy_cmd(uint32_t tag, uint64_t src, uint64_t dst, uint32_t K, uint32_t N, uint32_t flags = 0) {
  Cmd c = simple(OP_COPY, tag, flags);
  c.K = K;
  c.N = N;
  c.B = 1;
  c.s[SLOT_A] = {src, 4 * K, 0};
  c.s[SLOT_C] = {dst, 4 * K, 0};
  return c;
}

// ---------------------------------------------------------------- one start, logged for the twin replay
struct Status {
  uint32_t w[12];
  uint32_t code() const { return w[0]; }
  uint32_t next() const { return w[1]; }
  uint32_t tag() const { return w[2]; }
  uint32_t done() const { return w[3]; }
  uint32_t merr() const { return w[4]; }
  uint32_t magic() const { return w[5]; }
  uint32_t starts() const { return w[6]; }
  uint32_t lastop() const { return w[7]; }
  uint32_t version() const { return w[8]; }
  uint32_t last_fin() const { return w[9]; }
  uint64_t cycles() const { return w[10] | ((uint64_t)w[11] << 32); }
};
static Status status_at(uint64_t a) {
  Status s;
  memcpy(s.w, bp(a), sizeof s.w);
  return s;
}

static FILE* g_twin = nullptr;
static int g_twin_n = 0;
struct OutRange {
  uint64_t addr, len;
};

static uint32_t start_regs(uint32_t M, uint64_t list, uint64_t status, const std::vector<OutRange>& outs = {}) {
  g_trace.clear();
  const uint32_t K = (uint32_t)(list >> 4), N = (uint32_t)(status >> 4), fault0 = g_fault;
  const bool dry0 = g_dry;
  seq::Word* m = port();
  const uint32_t r = seq::top<HostUnits>(m, m, m, m, m, m, M, K, N);  // the core with the host units
  if (g_twin) {  // record: writes since the last start, registers, result, status, trace, outputs
    fprintf(g_twin, "START %d M=%x K=%x N=%x dry=%d fault=%u\n", g_twin_n++, M, K, N, dry0 ? 1 : 0, fault0);
    for (const Chunk& ch : g_writes) {
      fprintf(g_twin, "W %llx %zu ", (unsigned long long)ch.addr, ch.bytes.size());
      for (unsigned char b : ch.bytes) fprintf(g_twin, "%02x", b);
      fprintf(g_twin, "\n");
    }
    fprintf(g_twin, "RET %x\n", r);
    if (status) {
      const Status s = status_at(status);
      fprintf(g_twin, "STATUS");
      for (int i = 0; i < 10; i++) fprintf(g_twin, " %x", s.w[i]);
      fprintf(g_twin, "\n");
    }
    for (const std::string& t : g_trace) fprintf(g_twin, "T %s\n", t.c_str());
    for (const OutRange& o : outs) {
      fprintf(g_twin, "O %llx %llu ", (unsigned long long)o.addr, (unsigned long long)o.len);
      for (uint64_t i = 0; i < o.len; i++) fprintf(g_twin, "%02x", bp(o.addr)[i]);
      fprintf(g_twin, "\n");
    }
    fprintf(g_twin, "END\n");
  }
  g_writes.clear();
  return r;
}
static uint32_t start(uint64_t list, uint64_t status, uint32_t first, bool step = false,
                      const std::vector<OutRange>& outs = {}) {
  return start_regs(start_m(first, step), list, status, outs);
}

// ---------------------------------------------------------------- 1. the walk
static void test_walk(uint64_t base) {
  const uint64_t LIST = base + 0x2000, STAT = base + 0x1000, SRC = base + 0x20000, DST = base + 0x30000;
  const int f0 = g_fail;
  for (int i = 0; i < 64; i++) put_u32(SRC + 4 * i, 0x1000 + i);
  for (int i = 0; i < 64; i++) put_u32(DST + 4 * i, 0xDEADBEEF);
  store(LIST, 0, simple(OP_NOP, 0x100));
  store(LIST, 1, copy_cmd(0x101, SRC, DST, 16, 4));
  store(LIST, 2, simple(OP_NOP, 0x102, F_YIELD));
  store(LIST, 3, simple(OP_NOP, 0x103));
  store(LIST, 4, simple(OP_END, 0x104));
  const uint64_t cyc0 = seq::g_host_cycles;
  uint32_t r = start(LIST, STAT, 0, false, {{DST, 256}});
  Status st = status_at(STAT);
  CHECK(r == ((3u << 8) | ST_YIELD), "yield: return 0x%x", r);
  CHECK(st.code() == ST_YIELD && st.next() == 3 && st.tag() == 0x102 && st.done() == 3 && st.merr() == 0 &&
            st.magic() == STATUS_MAGIC && st.lastop() == OP_NOP && st.version() == CONTRACT_VERSION &&
            st.last_fin() == 2,
        "yield status code %u next %u tag %x done %u merr %u magic %x lastop %u version %x last %u", st.code(),
        st.next(), st.tag(), st.done(), st.merr(), st.magic(), st.lastop(), st.version(), st.last_fin());
  CHECK(st.cycles() == seq::g_host_cycles - cyc0 && st.cycles() > 0, "cycles %llu",
        (unsigned long long)st.cycles());
  CHECK(g_trace.size() == 1, "yield: %zu unit calls (NOP reaches no unit)", g_trace.size());
  int bad = 0;
  for (int i = 0; i < 64; i++) bad += get_u32(DST + 4 * i) != 0x1000u + i;
  CHECK(bad == 0, "copy: %d words wrong", bad);
  const uint32_t s0 = st.starts();
  r = start(LIST, STAT, 3);
  st = status_at(STAT);
  CHECK(r == ((4u << 8) | ST_END) && st.code() == ST_END && st.next() == 4 && st.tag() == 0x104 && st.done() == 1 &&
            st.starts() == s0 + 1 && st.lastop() == OP_END && st.last_fin() == 3,
        "resume: return 0x%x code %u next %u tag %x done %u starts %u", r, st.code(), st.next(), st.tag(), st.done(),
        st.starts());
  // single step: one command per start, END is not a command
  r = start(LIST, STAT, 0, true);
  CHECK(r == ((1u << 8) | ST_STEP) && g_trace.empty(), "step 0 (NOP): 0x%x", r);
  r = start(LIST, STAT, 1, true);
  st = status_at(STAT);
  CHECK(r == ((2u << 8) | ST_STEP) && g_trace.size() == 1 && st.done() == 1 && st.last_fin() == 1, "step 1 (COPY): 0x%x",
        r);
  r = start(LIST, STAT, 2, true);
  CHECK(r == ((3u << 8) | ST_YIELD), "step on a YIELD record: 0x%x (YIELD wins)", r);
  r = start(LIST, STAT, 4, true);
  st = status_at(STAT);
  CHECK(r == ((4u << 8) | ST_END) && st.done() == 0 && st.last_fin() == 0xFFFFFFFFu && g_trace.empty(),
        "step at END: 0x%x done %u last %x", r, st.done(), st.last_fin());
  // NO_LEGACY: an old per-call M (no magic) touches nothing, not even the status block
  for (int i = 0; i < 12; i++) put_u32(STAT + 4 * i, 0xEEEEEEEE);
  const uint32_t legacy_m[3] = {64, 0xA4000000u, 0x00A50000u};
  for (uint32_t m : legacy_m) {
    r = start_regs(m, LIST, STAT);
    CHECK(r == ST_NO_LEGACY && g_trace.empty() && get_u32(STAT) == 0xEEEEEEEEu, "no legacy M=%x: 0x%x", m, r);
  }
  // bad records: start right at the record, and from 0 through good commands (record 2's YIELD cleared first)
  poke(LIST, 2, FLD_OP, OP_NOP | (REC_MAGIC << 8));
  struct BadCase {
    const char* name;
    uint32_t field, value, want;
    bool seal;
  } bad_cases[] = {
      {"magic 0xC5 (the 128-byte draft)", FLD_OP, OP_NOP | (0xC5u << 8), ST_BAD_MAGIC, true},
      {"check word", PARAM_BASE + 3, 0x10, ST_BAD_CHECK, false},
      {"version 0.9", FLD_VERSION, 0x00000900, ST_BAD_VERSION, true},
      {"opcode 0x71 (GATHER, retired)", FLD_OP, 0x71 | (REC_MAGIC << 8), ST_BAD_OPCODE, true},
      {"opcode 0x80 (AXPY, retired)", FLD_OP, 0x80 | (REC_MAGIC << 8), ST_BAD_OPCODE, true},
      {"opcode 0x99", FLD_OP, 0x99 | (REC_MAGIC << 8), ST_BAD_OPCODE, true},
  };
  for (const BadCase& bc : bad_cases) {
    store(LIST, 3, simple(OP_NOP, 0x203));
    poke(LIST, 3, bc.field, bc.value, bc.seal);
    r = start(LIST, STAT, 3);
    st = status_at(STAT);
    CHECK(r == ((3u << 8) | bc.want) && st.code() == bc.want && st.next() == 3 && st.done() == 0 &&
              st.tag() == 0x203 && g_trace.empty(),
          "%s at 3: 0x%x tag %x done %u", bc.name, r, st.tag(), st.done());
    r = start(LIST, STAT, 0);
    st = status_at(STAT);
    CHECK(r == ((3u << 8) | bc.want) && st.done() == 3 && st.last_fin() == 2 && g_trace.size() == 1,
          "%s from 0: 0x%x done %u", bc.name, r, st.done());
  }
  // a 128-byte draft record (seq_format.h layout: magic 0xC5 in f0[15:8]) is BAD_MAGIC
  {
    uint32_t old[32] = {};
    old[0] = OP_GEMM | (0xC5u << 8);
    uint32_t x = 0x5EC0C5ED;
    for (int i = 0; i < 31; i++) x ^= old[i];
    old[31] = x;
    put(LIST + 3 * REC_BYTES, old, sizeof old);
    r = start(LIST, STAT, 3);
    CHECK(r == ((3u << 8) | ST_BAD_MAGIC), "128-byte draft record: 0x%x", r);
  }
  // mode errors from validate_hw: nothing dispatched, the output untouched
  struct MeCase {
    const char* name;
    uint32_t want;
  };
  for (int i = 0; i < 64; i++) put_u32(DST + 4 * i, 0xDEADBEEF);
  std::vector<std::pair<Cmd, MeCase>> me_cases;
  {
    Cmd g;
    memset(&g, 0, sizeof g);
    g.op = OP_GEMM;
    g.tag = 0x303;
    g.M = 32;
    g.K = 16;
    g.N = 4;
    g.B = 1;
    g.s[SLOT_A] = {SRC, 32, 0};
    g.s[SLOT_B] = {SRC, 1024, 0};
    g.s[SLOT_C] = {DST, 128, 0};
    g.s[SLOT_E] = {SRC, 0, 0};
    g.s[SLOT_F] = {SRC, 0, 0};
    Cmd t = g;
    me_cases.push_back({t, {"gemm ok (control: dispatched)", 0}});
    t = g, t.M = 33;
    me_cases.push_back({t, {"gemm M 33", ME_DIMS}});
    t = g, t.K = 4104;
    me_cases.push_back({t, {"gemm K 4104", ME_DIMS}});
    t = g, t.s[SLOT_A].addr += 8;
    me_cases.push_back({t, {"gemm A misaligned", ME_ALIGN}});
    t = g, t.s[SLOT_C].pitch = 64;
    me_cases.push_back({t, {"gemm C pitch < 4 M", ME_ALIGN}});
    t = g, t.s[SLOT_C].addr = 1ull << 40;
    me_cases.push_back({t, {"gemm C at 2^40", ME_ADDR}});
    t = g, t.flags = F_ACC | F_ADD_D, t.s[SLOT_D] = {SRC, 128, 0};
    me_cases.push_back({t, {"gemm ACC + ADD_D", ME_FLAGS}});
    t = g, t.flags = 0x0004;
    me_cases.push_back({t, {"gemm retired flag RESID", ME_FLAGS}});
    t = g, t.p[0] = 0x3E000000;
    me_cases.push_back({t, {"gemm p0 (output multiplier, retired)", ME_PARAM}});
    t = g, t.s[SLOT_H] = {16, 0, 0};
    me_cases.push_back({t, {"gemm stray slot H", ME_SLOT}});
    t = g, t.N = 0;
    me_cases.push_back({t, {"gemm N 0", ME_DIMS}});
    t = g, t.B = 65536;
    me_cases.push_back({t, {"gemm B 65536", ME_DIMS}});
    Cmd q = copy_cmd(0x304, SRC, DST, 16, 4);
    q.op = OP_QUANT_X;
    q.flags = F_QOUT;
    q.s[SLOT_C].pitch = 32;
    q.s[SLOT_F] = {DST + 512, 0, 0};
    me_cases.push_back({q, {"quant_x ok (control)", 0}});
    t = q, t.p[P_SHIFT] = 9;
    me_cases.push_back({t, {"quant_x shift 9", ME_PARAM}});
    t = q, t.p[P_SEG] = 32;
    me_cases.push_back({t, {"quant_x segment 32", ME_DIMS}});
    t = q, t.flags = 0;
    me_cases.push_back({t, {"quant_x without QOUT", ME_FLAGS}});
    t = simple(OP_NOP, 0x305), t.p[5] = 1;
    me_cases.push_back({t, {"nop with a parameter", ME_PARAM}});
    t = simple(OP_END, 0x306), t.s[SLOT_A] = {16, 0, 0};
    me_cases.push_back({t, {"end with a slot", ME_SLOT}});
  }
  for (auto& mc : me_cases) {
    store(LIST, 3, mc.first);
    g_dry = true;  // the control cases dispatch; their unit runs dry
    r = start(LIST, STAT, 3);
    g_dry = false;
    st = status_at(STAT);
    if (mc.second.want == 0) {
      CHECK((r & 0xFF) != ST_MODE_ERROR && g_trace.size() == 1, "%s: 0x%x, %zu dispatches", mc.second.name, r,
            g_trace.size());
      continue;
    }
    CHECK(r == ((3u << 8) | ST_MODE_ERROR) && st.merr() == mc.second.want && st.next() == 3 && st.done() == 0 &&
              st.tag() == mc.first.tag && st.lastop() == mc.first.op && g_trace.empty(),
          "%s: 0x%x merr %u (want %u) done %u tag %x", mc.second.name, r, st.merr(), mc.second.want, st.done(),
          st.tag());
  }
  int touched = 0;
  for (int i = 0; i < 64; i++) touched += get_u32(DST + 4 * i) != 0xDEADBEEFu;
  CHECK(touched == 0, "mode errors / dry units wrote %d words", touched);
  // a unit's own fault: MODE_ERROR at that command, counted as done (it ran)
  store(LIST, 3, simple(OP_NOP, 0x103));
  store(LIST, 3, copy_cmd(0x403, SRC, DST, 16, 4));
  g_fault = ME_ALIGN;
  r = start(LIST, STAT, 3);
  st = status_at(STAT);
  CHECK(r == ((3u << 8) | ST_MODE_ERROR) && st.merr() == ME_ALIGN && st.done() == 1 && st.lastop() == OP_COPY &&
            st.last_fin() == 0xFFFFFFFFu,
        "unit fault: 0x%x merr %u done %u", r, st.merr(), st.done());
  // no status block (N = 0): nothing written there
  for (int i = 0; i < 12; i++) put_u32(STAT + 4 * i, 0xEEEEEEEE);
  store(LIST, 3, simple(OP_NOP, 0x103));
  r = start(LIST, 0, 3);
  CHECK(r == ((4u << 8) | ST_END) && get_u32(STAT) == 0xEEEEEEEEu, "no status block: 0x%x", r);
  printf("walk @ 0x%llx: END / YIELD + resume / STEP / NO_LEGACY / BAD_MAGIC|CHECK|VERSION|OPCODE / MODE_ERROR "
         "(validate_hw, unit) / LIMIT / no status: %s\n",
         (unsigned long long)base, g_fail == f0 ? "ok" : "FAILED");
}

// ---------------------------------------------------------------- 2. dispatch routing
static void test_dispatch(uint64_t base) {
  const uint64_t LIST = base + 0x4000, STAT = base + 0x1000, A = base + 0x100000;
  const int f0 = g_fail;
  std::vector<Cmd> cs;
  std::vector<uint32_t> want_unit;
  Cmd c = simple(OP_NOP, 1);
  cs.push_back(c), want_unit.push_back(0);
  Cmd g;
  memset(&g, 0, sizeof g);
  g.op = OP_GEMM, g.tag = 2, g.M = 64, g.K = 64, g.N = 50, g.B = 5;
  g.flags = F_BIAS | F_ADD_D;
  g.s[SLOT_A] = {A, 128, 50 * 128};
  g.s[SLOT_B] = {A + 0x10000, 2048, 8192};
  g.s[SLOT_C] = {A + 0x20000, 256, 50 * 256};
  g.s[SLOT_D] = {A + 0x30000, 0, 0};
  g.s[SLOT_E] = {A + 0x40000, 0, 256};
  g.s[SLOT_F] = {A + 0x50000, 0, 208};
  g.s[SLOT_G] = {A + 0x60000, 0, 256};
  cs.push_back(g), want_unit.push_back(seq::U_GEMM);
  const uint32_t rops[8] = {OP_QUANT_X, OP_QUANT_W, OP_QUANT_WT, OP_NORM, OP_GELU, OP_SILU_MUL, OP_ROPE, OP_COPY};
  for (uint32_t op : rops) {
    Cmd r;
    memset(&r, 0, sizeof r);
    r.op = op, r.tag = 0x10 + op, r.K = 64, r.N = 32, r.B = 1;
    r.s[SLOT_A] = {A, 256, 0};
    r.s[SLOT_C] = {A + 0x20000, 256, 0};
    if (op == OP_QUANT_X || op == OP_GELU || op == OP_NORM || op == OP_SILU_MUL || op == OP_ROPE) {
      r.flags = F_QOUT;
      r.s[SLOT_C].pitch = 128;
      r.s[SLOT_F] = {A + 0x50000, 0, 0};
    }
    if (op == OP_NORM) r.p[P_EPS] = 0x3727C5AC, r.s[SLOT_E] = {A + 0x40000, 0, 0};  // 1e-5
    if (op == OP_SILU_MUL) r.s[SLOT_B] = {A + 0x8000, 256, 0};
    if (op == OP_ROPE) r.p[P_SEG] = 64, r.s[SLOT_E] = {A + 0x40000, 256, 0};
    if (op == OP_QUANT_W || op == OP_QUANT_WT) {
      r.p[P_SEG] = 64, r.p[P_NP] = 32;
      r.s[SLOT_C] = {A + 0x20000, 1024, 0};
      r.s[SLOT_F] = {A + 0x50000, 0, 0};
    }
    cs.push_back(r), want_unit.push_back(seq::U_ROW);
  }
  Cmd s;
  memset(&s, 0, sizeof s);
  s.op = OP_SOFTMAX, s.tag = 0x40, s.flags = F_QOUT | F_MASK, s.K = 128, s.N = 156, s.B = 5;
  s.s[SLOT_A] = {A, 512, 156 * 512};
  s.s[SLOT_C] = {A + 0x20000, 256, 156 * 256};
  s.s[SLOT_F] = {A + 0x50000, 0, 624};
  s.s[SLOT_D] = {A + 0x60000, 0, 0};
  s.s[SLOT_E] = {A + 0x61000, 0, 0};
  s.p[P_EPS] = 0x3F800000;
  cs.push_back(s), want_unit.push_back(seq::U_SMX);
  cs.push_back(simple(OP_END, 0xE));
  std::vector<std::string> want;
  for (size_t i = 0; i < cs.size(); i++) {
    CHECK(validate(cs[i]) == ME_OK, "dispatch case %zu (op %x) invalid: ME %u", i, cs[i].op, validate(cs[i]));
    store(LIST, (uint32_t)i, cs[i]);
    Cmd d;
    memset(&d, 0, sizeof d);
    uint32_t f[REC_FIELDS];
    encode(cs[i], f);
    decode(f, d);
    if (cs[i].op != OP_NOP && cs[i].op != OP_END) want.push_back(trace_line(d, want_unit[i]));
  }
  g_dry = true;
  const uint32_t r = start(LIST, STAT, 0);
  g_dry = false;
  CHECK(r == (((uint32_t)cs.size() - 1) << 8 | ST_END), "dispatch list: 0x%x", r);
  CHECK(g_trace == want, "dispatch trace differs (%zu vs %zu lines)", g_trace.size(), want.size());
  printf("dispatch @ 0x%llx: NOP + GEMM + 8 row ops + SOFTMAX -> units %s\n", (unsigned long long)base,
         g_fail == f0 ? "in order, once each, every field" : "FAILED");
}

// ---------------------------------------------------------------- 3. data through the units
static void test_data(uint64_t base, std::mt19937& rng) {
  const uint64_t LIST = base + 0x8000, STAT = base + 0x1000;
  const uint64_t XF = base + 0x200000, XQ = base + 0x300000, SX = base + 0x380000, W = base + 0x400000,
                 SR = base + 0x500000, BI = base + 0x510000, D = base + 0x520000, Y = base + 0x600000,
                 Y2 = base + 0x700000;
  const int f0 = g_fail;
  const uint32_t M = 64, Kd = 128, N = 12, NB = 2, R = 6;  // 2 items, C row groups of 6 rows
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::uniform_int_distribution<int> qd(-32767, 32767);
  // FP32 input rows [NB][N][Kd], row 5 of item 1 all zero (scale 1.0 path)
  std::vector<float> xf((size_t)NB * N * Kd);
  for (auto& v : xf) v = nd(rng) * 3.0f;
  for (uint32_t k = 0; k < Kd; k++) xf[((size_t)1 * N + 5) * Kd + k] = 0.0f;
  const uint32_t xf_pitch = 4 * Kd + 64, xq_pitch = 2 * Kd + 32;
  for (uint32_t b = 0; b < NB; b++)
    for (uint32_t n = 0; n < N; n++)
      for (uint32_t k = 0; k < Kd; k++)
        put_f32(XF + (uint64_t)(b * N + n) * xf_pitch + 4 * k, xf[((size_t)b * N + n) * Kd + k]);
  // weights: per item an [M][Kd] int16 blob in the port tiling, s_row, bias; one D row (pitch 0 broadcast)
  const uint32_t port_bytes = (M / 32) * Kd * 16, w_item = 4 * port_bytes + 4096;
  std::vector<int16_t> wq((size_t)NB * M * Kd);
  std::vector<float> sr(NB * M), bi(NB * M), dd(M);
  for (auto& v : wq) v = (int16_t)qd(rng);
  for (auto& v : sr) v = std::ldexp(1.0f + 0.25f * nd(rng), -17);
  for (auto& v : bi) v = nd(rng);
  for (auto& v : dd) v = nd(rng);
  for (uint32_t b = 0; b < NB; b++) {
    const Slot ws = {W, port_bytes, w_item};
    for (uint32_t m = 0; m < M; m++)
      for (uint32_t k = 0; k < Kd; k++)
        put_i16(w_word_addr(ws, b, m, k, Kd) + 2 * w_lane(m), wq[((size_t)b * M + m) * Kd + k]);
    for (uint32_t m = 0; m < M; m++) {
      put_f32(SR + b * 1024 + 4 * m, sr[b * M + m]);
      put_f32(BI + b * 1024 + 4 * m, bi[b * M + m]);
    }
  }
  for (uint32_t m = 0; m < M; m++) put_f32(D + 4 * m, dd[m]);
  // Y2 starts with known values (the F_ACC pass adds to it)
  std::vector<float> y2_0((size_t)NB * N * M);
  for (auto& v : y2_0) v = nd(rng);
  const uint32_t y_pitch = 4 * M + 16, P1 = 4 * M * 8;  // row t of item b at Y + b*stride + (t/R) P1 + (t%R) pitch
  const uint32_t y_stride = 2 * P1;  // item b's 2 row groups at Y + (2 b + g) P1: the COPY below walks them as items
  for (uint32_t b = 0; b < NB; b++)
    for (uint32_t t = 0; t < N; t++)
      for (uint32_t m = 0; m < M; m++)
        put_f32(row_addr2({Y2, y_pitch, y_stride}, b, t, R, P1) + 4 * m, y2_0[((size_t)b * N + t) * M + m]);
  // list: QUANT_X (shift -3, NP 16), GEMM bias + D -> Y, GEMM ACC -> Y2, COPY Y -> Y2 + 0x80000, END
  Cmd q;
  memset(&q, 0, sizeof q);
  q.op = OP_QUANT_X, q.flags = F_QOUT, q.tag = 1, q.K = Kd, q.N = N, q.B = NB;
  q.s[SLOT_A] = {XF, xf_pitch, N * xf_pitch};
  q.s[SLOT_C] = {XQ, xq_pitch, 16 * xq_pitch};
  q.s[SLOT_F] = {SX, 0, 64};
  q.p[P_SHIFT] = (uint32_t)-3;
  q.p[P_NP] = 16;
  Cmd g;
  memset(&g, 0, sizeof g);
  g.op = OP_GEMM, g.flags = F_BIAS | F_ADD_D, g.tag = 2, g.M = M, g.K = Kd, g.N = N, g.B = NB;
  g.s[SLOT_A] = {XQ, xq_pitch, 16 * xq_pitch};
  g.s[SLOT_B] = {W, port_bytes, w_item};
  g.s[SLOT_C] = {Y, y_pitch, y_stride};
  g.s[SLOT_D] = {D, 0, 0};
  g.s[SLOT_E] = {SR, 0, 1024};
  g.s[SLOT_F] = {SX, 0, 64};
  g.s[SLOT_G] = {BI, 0, 1024};
  g.p[P_SEGC] = R, g.p[P_SEGF] = P1;
  Cmd a = g;
  a.flags = F_ACC, a.tag = 3;
  a.s[SLOT_C] = {Y2, y_pitch, y_stride};
  a.s[SLOT_D] = {0, 0, 0};
  a.s[SLOT_G] = {0, 0, 0};
  Cmd cp = copy_cmd(4, Y, Y2 + 0x80000, M, R, 0);
  cp.B = 2 * NB;  // the row groups of Y as 4 items of R rows
  cp.s[SLOT_A] = {Y, y_pitch, P1};
  cp.s[SLOT_C] = {Y2 + 0x80000, 4 * M, R * 4 * M};
  const Cmd lst[5] = {q, g, a, cp, simple(OP_END, 5)};
  for (int i = 0; i < 5; i++) {
    CHECK(validate(lst[i]) == ME_OK, "data list record %d invalid: ME %u", i, validate(lst[i]));
    store(LIST, i, lst[i]);
  }
  const uint32_t r = start(LIST, STAT, 0, false,
                           {{XQ, (uint64_t)NB * 16 * xq_pitch}, {SX, 128}, {Y, (uint64_t)NB * y_stride},
                            {Y2, (uint64_t)NB * y_stride}, {Y2 + 0x80000, (uint64_t)2 * NB * R * 4 * M}});
  CHECK(r == ((4u << 8) | ST_END), "data list: 0x%x", r);
  // reference, straight from ctr::ref
  int bq = 0, bs = 0, by = 0, by2 = 0, bc = 0;
  for (uint32_t b = 0; b < NB; b++) {
    std::vector<int16_t> qr((size_t)16 * Kd, 0);
    std::vector<float> st(16, 1.0f);
    for (uint32_t n = 0; n < N; n++) st[n] = ref::quant_row(&xf[((size_t)b * N + n) * Kd], Kd, &qr[n * Kd], -3);
    for (uint32_t n = 0; n < 16; n++) {
      for (uint32_t k = 0; k < Kd; k++) bq += get_i16(XQ + (uint64_t)(b * 16 + n) * xq_pitch + 2 * k) != qr[n * Kd + k];
      bs += get_f32(SX + b * 64 + 4 * n) != st[n];
    }
    for (uint32_t t = 0; t < N; t++)
      for (uint32_t m = 0; m < M; m++) {
        int64_t acc = 0;
        for (uint32_t k = 0; k < Kd; k++) acc += (int64_t)qr[t * Kd + k] * wq[((size_t)b * M + m) * Kd + k];
        const float y = ref::gemm_out(acc, sr[b * M + m], st[t], &bi[b * M + m], &dd[m]);
        const float y2 = ref::gemm_out(acc, sr[b * M + m], st[t], nullptr, &y2_0[((size_t)b * N + t) * M + m]);
        const uint64_t at = row_addr2({Y, y_pitch, y_stride}, b, t, R, P1) + 4 * m;
        const uint64_t at2 = row_addr2({Y2, y_pitch, y_stride}, b, t, R, P1) + 4 * m;
        by += get_f32(at) != y;
        by2 += get_f32(at2) != y2;
        bc += get_f32(Y2 + 0x80000 + (uint64_t)(b * N + t) * 4 * M + 4 * m) != y;
      }
  }
  CHECK(bq == 0 && bs == 0, "quant_x: %d q, %d s mismatches", bq, bs);
  CHECK(by == 0 && by2 == 0 && bc == 0, "gemm: %d y, %d acc, %d copy mismatches", by, by2, bc);
  printf("data @ 0x%llx: QUANT_X (shift -3, NP pad, zero row) -> GEMM (2 items, row groups, bias, D) -> GEMM ACC -> "
         "COPY: %s\n",
         (unsigned long long)base, g_fail == f0 ? "bit-exact vs ctr::ref" : "FAILED");
}

// ---------------------------------------------------------------- 4. the -12392 chunk list
static void test_chunk(const std::string& bin, const std::string& trace, uint64_t base) {
  std::ifstream fb(bin, std::ios::binary);
  std::ifstream ft(trace);
  if (!fb || !ft) {
    printf("chunk %s: not found, skipped\n", bin.c_str());
    CHECK(false, "chunk list missing: %s / %s", bin.c_str(), trace.c_str());
    return;
  }
  const int f0 = g_fail;
  std::string blob((std::istreambuf_iterator<char>(fb)), std::istreambuf_iterator<char>());
  std::vector<std::string> want;
  for (std::string l; std::getline(ft, l);)
    if (!l.empty()) want.push_back(l);
  const uint32_t ncmd = (uint32_t)(blob.size() / REC_BYTES);
  const uint64_t LIST = base + 0x01000000, STAT = base + 0x1000;  // position independent: placed here
  memcpy(bp(LIST), blob.data(), blob.size());
  g_dry = true;
  uint32_t r = start(LIST, STAT, 0);
  Status st = status_at(STAT);
  CHECK(r == (((ncmd - 1) << 8) | ST_END) && st.done() == ncmd - 1 && st.last_fin() == ncmd - 2,
        "chunk: return 0x%x done %u (want next %u END)", r, st.done(), ncmd - 1);
  size_t diff = 0;
  for (size_t i = 0; i < want.size() || i < g_trace.size(); i++) {
    const std::string a = i < g_trace.size() ? g_trace[i] : "<none>", b = i < want.size() ? want[i] : "<none>";
    if (a != b && diff++ < 3) printf("  line %zu:\n    C++    %s\n    python %s\n", i, a.c_str(), b.c_str());
  }
  CHECK(diff == 0, "chunk trace: %zu of %zu lines differ", diff, want.size());
  const std::vector<std::string> one = g_trace;
  // step mode: one start per command
  std::vector<std::string> acc;
  uint32_t idx = 0, starts = 0;
  for (;;) {
    r = start(LIST, STAT, idx, true);
    acc.insert(acc.end(), g_trace.begin(), g_trace.end());
    starts++;
    if ((r & 0xFF) != ST_STEP) break;
    idx = r >> 8;
  }
  CHECK((r & 0xFF) == ST_END && acc == one && starts == ncmd, "step walk: %u starts, end 0x%x, %s", starts, r,
        acc == one ? "same" : "different");
  // F_YIELD on every 97th record (resealed copy): resume until END
  std::string y = blob;
  for (uint32_t i = 0; i + 1 < ncmd; i += 97) {
    uint32_t f[REC_FIELDS];
    memcpy(f, &y[(size_t)i * REC_BYTES], REC_BYTES);
    f[FLD_OP] |= F_YIELD << 16;
    f[CHECK_FIELD] = rec_check(f);
    memcpy(&y[(size_t)i * REC_BYTES], f, REC_BYTES);
  }
  memcpy(bp(LIST), y.data(), y.size());
  acc.clear();
  idx = 0, starts = 0;
  for (;;) {
    r = start(LIST, STAT, idx);
    for (std::string& l : g_trace) {  // the trace carries the flags: drop F_YIELD for the comparison
      const size_t p = l.find("flags=");
      unsigned fl = std::stoul(l.substr(p + 6, 4), nullptr, 16) & ~F_YIELD;
      char h[5];
      snprintf(h, sizeof h, "%04x", fl);
      l.replace(p + 6, 4, h);
    }
    acc.insert(acc.end(), g_trace.begin(), g_trace.end());
    starts++;
    if ((r & 0xFF) != ST_YIELD) break;
    idx = r >> 8;
  }
  CHECK((r & 0xFF) == ST_END && acc == one && starts == (ncmd - 2) / 97 + 2, "yield walk: %u starts, end 0x%x, %s",
        starts, r, acc == one ? "same" : "different");
  memcpy(bp(LIST), blob.data(), blob.size());
  g_dry = false;
  printf("chunk %s @ list 0x%llx: %u records, %u dispatched, trace vs python %s; step walk %u starts; yield walk: %s\n",
         bin.c_str(), (unsigned long long)LIST, ncmd, st.done(), diff ? "DIFFERENT" : "identical", ncmd,
         g_fail == f0 ? "ok" : "FAILED");
}

// ---------------------------------------------------------------- 5. decode_hw == ctr::decode
static void test_decode(std::mt19937& rng) {
  const int f0 = g_fail;
  const uint32_t ops[] = {OP_NOP, OP_END, OP_GEMM, OP_QUANT_X, OP_QUANT_W, OP_QUANT_WT, OP_NORM, OP_SOFTMAX,
                          OP_GELU, OP_SILU_MUL, OP_ROPE, OP_COPY, 0x71, 0x80, 0x02, 0xFF};
  std::uniform_int_distribution<uint32_t> u32;
  std::map<uint32_t, uint64_t> by_status;
  uint64_t bad = 0;
  const uint64_t n0 = seq::g_decode_mismatch;
  for (int it = 0; it < 400000; it++) {
    Cmd c;
    memset(&c, 0, sizeof c);
    c.op = ops[u32(rng) % 16];
    c.flags = u32(rng) & 0xFFFF;
    c.tag = u32(rng);
    c.M = u32(rng) >> (u32(rng) % 32);
    c.K = u32(rng) >> (u32(rng) % 32);
    c.N = u32(rng) >> (u32(rng) % 32);
    c.B = (it % 7 == 0) ? 0 : u32(rng) >> (u32(rng) % 32);  // 0 decodes as 1
    for (uint32_t s = 0; s < NSLOTS; s++)
      c.s[s] = {((uint64_t)u32(rng) << 32 | u32(rng)) >> (u32(rng) % 64), u32(rng), u32(rng)};
    for (uint32_t i = 0; i < NPARAMS; i++) c.p[i] = u32(rng) >> (u32(rng) % 32);
    uint32_t f[REC_FIELDS];
    encode(c, f);
    const uint32_t kind = it % 4;  // 0: as encoded, 1: flips not resealed, 2: flips resealed, 3: version / magic
    if (kind == 1 || kind == 2) {
      const int nflip = 1 + (int)(u32(rng) % 3);
      for (int k = 0; k < nflip; k++) f[u32(rng) % REC_FIELDS] ^= 1u << (u32(rng) % 32);
      if (kind == 2) f[CHECK_FIELD] = rec_check(f);
    } else if (kind == 3) {
      if (u32(rng) & 1)
        f[FLD_VERSION] = u32(rng) & 1 ? 0x00000900u : CONTRACT_VERSION + 1;
      else
        f[FLD_OP] = (f[FLD_OP] & 0xFFFF00FFu) | ((u32(rng) & 1 ? 0xC5u : u32(rng) & 0xFF) << 8);
      f[CHECK_FIELD] = rec_check(f);
    }
    seq::Word rec[REC_WORDS];
    uint32_t xsum = 0;
    for (uint32_t w = 0; w < REC_WORDS; w++) {
      for (int j = 0; j < 4; j++) seq::wset(rec[w], j, f[4 * w + j]);
      xsum ^= f[4 * w] ^ f[4 * w + 1] ^ f[4 * w + 2] ^ f[4 * w + 3];
    }
    Cmd hw;
    memset(&hw, 0xA5, sizeof hw);
    const uint32_t st = seq::decode_hw(rec, xsum, hw);
    by_status[st]++;
    if (!seq::decode_matches(rec, st, hw) && bad++ < 3) printf("  decode mismatch at record %d\n", it);
  }
  CHECK(bad == 0, "decode_hw: %llu of 400000 records differ from ctr::decode", (unsigned long long)bad);
  CHECK(seq::g_decode_mismatch == n0, "decode_hw mismatches inside the walks");
  printf("decode: 400000 records, decode_hw == ctr::decode (status and every field):");
  for (const auto& kv : by_status) printf(" st 0x%02x x %llu", kv.first, (unsigned long long)kv.second);
  printf(" -> %s\n", g_fail == f0 ? "identical" : "FAILED");
}

int main(int argc, char** argv) {
  std::vector<std::string> chunks;  // triples: bin, trace, base
  std::string twin;
  for (int i = 1; i < argc; i++) {
    const std::string a = argv[i];
    if (a == "--twin" && i + 1 < argc)
      twin = argv[++i];
    else
      chunks.push_back(a);
  }
  setvbuf(stdout, nullptr, _IONBF, 0);
  g_mem.assign((size_t)(512u << 20) / 16, seq::Word{{0, 0, 0, 0}});  // a 512 MB window
  std::mt19937 rng(12392);
  const uint64_t bases[2] = {0, 0x800000000ull};
  for (uint64_t base : bases) {
    g_base = base;  // the 512 MB window starts at the test base
    std::fill(g_mem.begin(), g_mem.end(), seq::Word{{0, 0, 0, 0}});
    if (!twin.empty()) {
      const std::string p = twin + (base ? "/twin_hi.txt" : "/twin_lo.txt");
      g_twin = fopen(p.c_str(), "w");
      g_twin_n = 0;
    }
    g_writes.clear();
    test_walk(base);
    test_dispatch(base);
    test_data(base, rng);
    if (g_twin) fclose(g_twin), g_twin = nullptr;
  }
  {  // LIMIT at a high base: window around the last record of a MAX_INDEX-long list
    g_base = 0x800000000ull;
    std::fill(g_mem.begin(), g_mem.end(), seq::Word{{0, 0, 0, 0}});
    const uint64_t rec = g_base + 0x1000000, lb = rec - (uint64_t)MAX_INDEX * REC_BYTES, STAT = g_base + 0x1000;
    store(lb, MAX_INDEX, simple(OP_NOP, 0x7FF));
    const uint32_t r = start(lb, STAT, MAX_INDEX);
    const Status st = status_at(STAT);
    CHECK(r == ((0x800000u << 8) | ST_LIMIT) && st.next() == 0x800000 && st.last_fin() == MAX_INDEX &&
              st.done() == 1 && (lb >> 4) <= 0xFFFFFFFFull,
          "limit: 0x%x next %x last %x", r, st.next(), st.last_fin());
    printf("limit: NOP at index 0x%x (list base 0x%llx) -> 0x%x (next 0x%x, LIMIT)\n", MAX_INDEX,
           (unsigned long long)lb, r, st.next());
  }
  for (size_t i = 0; i + 2 < chunks.size(); i += 3) {
    const uint64_t base = std::stoull(chunks[i + 2], nullptr, 16);
    g_base = base;
    std::fill(g_mem.begin(), g_mem.end(), seq::Word{{0, 0, 0, 0}});
    test_chunk(chunks[i], chunks[i + 1], base);
  }
  test_decode(rng);
  CHECK(seq::g_decode_mismatch == 0 && seq::g_decode_checked > 0, "decode_hw vs ctr::decode in the walks: %llu of %llu",
        (unsigned long long)seq::g_decode_mismatch, (unsigned long long)seq::g_decode_checked);
  printf("walks: %llu fetched records decoded by decode_hw and by ctr::decode, %llu differ\n",
         (unsigned long long)seq::g_decode_checked, (unsigned long long)seq::g_decode_mismatch);
  printf("%s (%d checks, %d failures)\n", g_fail ? "FAILED" : "PASSED", g_checks, g_fail);
  return g_fail ? 1 : 0;
}
