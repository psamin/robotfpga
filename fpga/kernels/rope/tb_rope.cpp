// Host testbench of the ROPE mode (rope_mode.h) under the integration contract (fpga/kernels/CONTRACT.md).
//
// For every vector case of gen_vectors.py and every layout below it builds a real 256-byte record (ctr::encode ->
// ctr::decode, which must give ST_END; ctr::validate and ctr::validate_hw must give ME_OK), fills a simulated DDR
// with a sentinel, places the inputs, runs it through seq::dispatch -> rope::rope_unit -> rope::rope_mode (all three
// ports see the same memory, as on the board), and compares EVERY word of the memory with the expected image: the
// outputs where the contract puts them (computed with contract.h's own address helpers and ctr::ref::quant_row, an
// independent path) and the sentinel / the inputs everywhere else (nothing may be written outside the declared
// extents). Every read unit must lie inside the declared inputs (rope_tb_read), and a rejected record reads nothing.
//   f32        FP32 rows, separate output, contiguous heads
//   f32_inpl   FP32 rows in place (C == A) inside a pitched row (the K part of a q|k|v buffer), table pitch 512
//   f32_hm     FP32 head-major (segments not adjacent: one burst per segment)
//   q_hm       int16 Q head-major like the -12392 list (pitch 128, p2 = one head block, p4 = -3, NP = pad4(T) + 4)
//   q_row      int16 in-row segments (p2 = 128), NP = T (scale pad lanes 1.0), p4 = +2
//   kw         KW blob (port pitch, p2 = item stride, NP = pad32(T) + 32: zero keys with scale 1.0)
//   kw_g1      KW written at group 1 of a blob (the expert suffix keys), p4 = -1
//   q_hm_b2, f32_b2, kw_b2   two items with item strides (item 1 = -x: exact, so its outputs are -y / -q)
// Every layout runs at a low base (0x4000_0000), and q_hm, kw_g1 also at 0x8_0000_0000 and 0xFF_0000_0000 (64-bit
// address arithmetic). Then malformed records: rope_mode must return the ME code and touch nothing.
//
// Build (Vitis MinGW g++, clean PATH; README.md):
//   g++ -std=c++14 -O2 -ffp-contract=off -I/c/Xilinx/Vitis_HLS/2022.2/include tb_rope.cpp -o build/tb_rope.exe
//   g++ ... -DROPE_DFSIM -Idfsim -I/c/Xilinx/Vitis_HLS/2022.2/include tb_rope.cpp -o build/tb_rope_dfsim.exe
//         (every DATAFLOW process a thread, FIFOs bounded at the pragma depths: the deadlock model)
//   ./build/tb_rope.exe build/vectors [--quick]
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <string>
#include <vector>

#define ROPE_TB_RDCHK  // rd reports every read unit (rope_tb_read below)
#include "rope_mode.h"
#include "../sequencer/seq_core.h"

using rope::Word;

// ---------------------------------------------------------------- the sequencer's dispatch switch
// Every layout run goes through seq::dispatch (the sequencer's single dispatch site) with a Units policy whose
// row() sends OP_ROPE to rope::rope_unit, exactly as float_top's row unit will. Other units must not be reached.
namespace seq {
uint64_t g_host_cycles = 0;
}
static long g_wrong_unit = 0;
struct RopeUnits {
  template <class WP, class W>
  static uint32_t gemm(const ctr::Cmd&, const WP*, const WP*, const WP*, const WP*, const W*, W*) {
    g_wrong_unit++;
    return ctr::ME_DIMS;
  }
  template <class WP, class W>
  static uint32_t softmax(const ctr::Cmd&, const WP*, const WP*, const WP*, const WP*, const W*, W*) {
    g_wrong_unit++;
    return ctr::ME_DIMS;
  }
  template <class WP, class W>
  static uint32_t row(const ctr::Cmd& c, const WP* hp0, const WP* hp1, const WP* hp2, const WP* hp3, const W* io_rd,
                      W* io_wr) {
    if (c.op != ctr::OP_ROPE) {
      g_wrong_unit++;
      return ctr::ME_DIMS;
    }
    return rope::rope_unit(c, hp0, hp1, hp2, hp3, io_rd, io_wr);
  }
};
static uint32_t run_dispatch(const ctr::Cmd& c, Word* port) {
  return seq::dispatch<RopeUnits>(c, (const Word*)port, (const Word*)nullptr, (const Word*)nullptr,
                                  (const Word*)nullptr, (const Word*)port, port);
}

// ---------------------------------------------------------------- read-extent check
// Every read unit of rd must lie inside the record's declared inputs: x unit = 16 words of head j of row n < N of
// item b (A), table unit = 16 words of table row n < N (E). g_rd_x / g_rd_e flag the allowed words of the current
// run (index = word - g_rd_base); with no run armed (the reject tests) any read is a failure.
static std::vector<uint8_t> g_rd_x, g_rd_e;
static uint64_t g_rd_base = 0;
static long g_rd_bad = 0, g_rd_units = 0;
// rd runs in one thread (also in ROPE_DFSIM), so no lock is needed
void rope::rope_tb_read(uint64_t xw, uint64_t ew) {
  g_rd_units++;
  for (uint64_t k = 0; k < rope::WORDS; k++) {
    const uint64_t xi = xw + k - g_rd_base, ei = ew + k - g_rd_base;
    const bool okx = xw + k >= g_rd_base && xi < g_rd_x.size() && g_rd_x[xi];
    const bool oke = ew + k >= g_rd_base && ei < g_rd_e.size() && g_rd_e[ei];
    if (!okx || !oke) {
      if (g_rd_bad < 4)
        fprintf(stdout, "  read outside the declared inputs: x word %#llx%s, table word %#llx%s\n",
                (unsigned long long)(xw + k), okx ? "" : " (bad)", (unsigned long long)(ew + k), oke ? "" : " (bad)");
      g_rd_bad++;
      return;
    }
  }
}

#ifdef ROPE_DFSIM
int rope_dfsim_depth(const std::string& n) {
  if (n == "rope_xs") return ROPE_D_XS;
  if (n == "rope_cs") return ROPE_D_CS;
  if (n == "rope_rs") return ROPE_D_RS;
  if (n == "rope_fe") return ROPE_D_FE;
  if (n == "rope_fw") return ROPE_D_FW;
  if (n == "rope_am") return ROPE_D_AM;
  if (n == "rope_sd") return ROPE_D_SD;
  if (n == "rope_sp") return ROPE_D_SP;
  if (n == "rope_rf") return ROPE_D_RF;
  if (n == "rope_ow") return ROPE_D_OW;
  if (n == "rope_ot") return ROPE_D_OT;
  if (n == "rope_ss") return ROPE_D_SS;
  if (n == "rope_hd") return ROPE_D_HD;
  if (n == "rope_dd") return ROPE_D_DD;
  fprintf(stderr, "unknown stream %s\n", n.c_str());
  exit(3);
}
#endif

template <class T>
static std::vector<T> load(const std::string& path, size_t n) {
  std::vector<T> v(n);
  FILE* f = fopen(path.c_str(), "rb");
  if (!f || fread(v.data(), sizeof(T), n, f) != n) {
    fprintf(stderr, "cannot read %s (%d items)\n", path.c_str(), (int)n);
    exit(2);
  }
  fclose(f);
  return v;
}

static uint32_t f2u(float f) { return ctr::f2u(f); }

// ---------------------------------------------------------------- simulated DDR
// MEM_WORDS 128-bit words at physical byte address `base`. The mode indexes its pointers with absolute word indices
// (addr >> 4), so it gets `mem - base / 16` as each port pointer: index base/16 + i is mem[i].
struct Ddr {
  uint64_t base;
  std::vector<uint32_t> u;  // 4 lanes per word
  explicit Ddr(uint64_t b, size_t words) : base(b), u(4 * words) {}
  size_t words() const { return u.size() / 4; }
  uint32_t& w32(uint64_t addr) {
    const uint64_t off = addr - base;
    if (addr < base || off / 4 >= u.size() || off % 4) {
      fprintf(stderr, "tb address %#llx outside the simulated DDR\n", (unsigned long long)addr);
      exit(4);
    }
    return u[off / 4];
  }
  uint16_t get16(uint64_t addr) {
    const uint32_t v = w32(addr & ~3ull);
    return (addr & 2) ? (uint16_t)(v >> 16) : (uint16_t)v;
  }
  void set16(uint64_t addr, uint16_t x) {
    uint32_t& v = w32(addr & ~3ull);
    v = (addr & 2) ? ((v & 0xFFFFu) | ((uint32_t)x << 16)) : ((v & 0xFFFF0000u) | x);
  }
  void sentinel() {
    for (size_t i = 0; i < u.size(); i++) u[i] = 0xDEAD0000u ^ (uint32_t)(i * 2654435761u);
  }
  std::vector<Word> words_out() const {
    std::vector<Word> w(words());
    for (size_t i = 0; i < w.size(); i++)
      for (int j = 0; j < 4; j++) w[i].range(32 * j + 31, 32 * j) = u[4 * i + j];
    return w;
  }
  void words_in(const std::vector<Word>& w) {
    for (size_t i = 0; i < w.size(); i++)
      for (int j = 0; j < 4; j++) u[4 * i + j] = (uint32_t)w[i].range(32 * j + 31, 32 * j);
  }
};

struct Bump {  // 4 KB aligned regions above the base
  uint64_t top;
  uint64_t take(uint64_t bytes) {
    const uint64_t a = (top + 4095) & ~4095ull;
    top = a + bytes;
    return a;
  }
};

struct Case {
  std::string name;
  int T, H;
  bool f32only;
  std::vector<uint32_t> x, tab, f32;
  std::vector<int16_t> q;
  std::vector<uint32_t> s;
};

enum Kind { K_F32, K_Q, K_KW };
struct Layout {
  const char* name;
  Kind kind;
  int B;
  bool inplace, headmajor, group1;
  int shift;
  int np_extra;  // Q: NP = pad4(T) + np_extra (or T when < 0); KW: NP = pad32(T) + np_extra
  uint32_t tab_pitch;
};

static const Layout LAYOUTS[] = {
    {"f32", K_F32, 1, false, false, false, 0, 0, 256},
    {"f32_inpl", K_F32, 1, true, false, false, 0, 0, 512},
    {"f32_hm", K_F32, 1, false, true, false, 0, 0, 256},
    {"q_hm", K_Q, 1, false, true, false, -3, 4, 256},
    {"q_row", K_Q, 1, false, false, false, 2, -1, 512},
    {"kw", K_KW, 1, false, false, false, 0, 32, 256},
    {"kw_g1", K_KW, 1, false, false, true, -1, 0, 256},
    {"q_hm_b2", K_Q, 2, false, true, false, -3, 0, 256},
    {"f32_b2", K_F32, 2, true, false, false, 0, 0, 512},
    {"kw_b2", K_KW, 2, false, false, false, 0, 0, 256},
};

static long g_fail = 0, g_checked = 0;

// The expected value of item b: item 1 is the negated input, which the arithmetic maps exactly to the negated
// output (a zero stays +0).
static uint32_t neg_f(uint32_t u) { return (u & 0x7FFFFFFFu) ? (u ^ 0x80000000u) : 0u; }

static void run_layout(const Case& cs, const Layout& L, uint64_t base) {
  const int T = cs.T, H = cs.H, K = 64 * H, B = L.B;
  const uint32_t rowb = 4 * K;
  int NP = T;
  if (L.kind == K_Q) NP = L.np_extra < 0 ? T : (int)ctr::pad4(T) + L.np_extra;
  if (L.kind == K_KW) NP = (int)ctr::pad32(T) + L.np_extra;
  Bump m{base + 4096};
  ctr::Cmd c;
  memset(&c, 0, sizeof c);
  c.op = ctr::OP_ROPE;
  c.tag = 0x60;
  c.K = K;
  c.N = T;
  c.B = B;
  c.p[ctr::P_SEG] = 64;
  c.p[ctr::P_SHIFT] = (uint32_t)L.shift;
  if (NP != T) c.p[ctr::P_NP] = NP;
  // A: FP32 rows (in place: pitch = row + 384 B gap, like K inside q|k|v)
  ctr::Slot A{0, L.inplace ? rowb + 384 : rowb, 0};
  A.stride = B > 1 ? (uint32_t)(((uint64_t)T * A.pitch + 4095) & ~4095ull) : 0;
  A.addr = m.take((uint64_t)B * (A.stride ? A.stride : (uint64_t)T * A.pitch));
  ctr::Slot E{0, L.tab_pitch, 0};
  E.addr = m.take((uint64_t)T * E.pitch);
  ctr::Slot Cs{0, 0, 0}, Fs{0, 0, 0};
  uint32_t segc = 0, segf = 0;
  if (L.kind == K_F32) {
    c.flags = 0;
    if (L.inplace) {
      Cs = A;
      segc = H > 1 ? 256 : 0;
    } else if (L.headmajor) {
      Cs.pitch = 272;  // 256 B segment + a 16 B gap per row
      segc = H > 1 ? (uint32_t)(((uint64_t)T * Cs.pitch + 255) & ~255ull) : 0;
      Cs.stride = B > 1 ? H * (segc ? segc : T * Cs.pitch) : 0;
      Cs.addr = m.take((uint64_t)B * H * (segc ? segc : (uint64_t)T * Cs.pitch));
    } else {
      Cs.pitch = rowb;
      segc = H > 1 ? 256 : 0;
      Cs.addr = m.take((uint64_t)B * T * rowb);
    }
  } else if (L.kind == K_Q) {
    c.flags = ctr::F_QOUT;
    if (L.headmajor) {
      Cs.pitch = 128;
      segc = H > 1 ? (uint32_t)NP * 128 : 0;
      Cs.stride = B > 1 ? (uint32_t)(H * NP * 128) : 0;
      Cs.addr = m.take((uint64_t)B * H * NP * 128);
    } else {
      Cs.pitch = 2 * K + 32;
      segc = H > 1 ? 128 : 0;
      Cs.stride = B > 1 ? (uint32_t)(NP * Cs.pitch) : 0;
      Cs.addr = m.take((uint64_t)B * NP * Cs.pitch);
    }
    segf = H > 1 ? 4 * ctr::pad4(NP) + 16 : 0;
    const uint64_t fitem = (uint64_t)H * (segf ? segf : 4 * ctr::pad4(NP));
    Fs.stride = B > 1 ? (uint32_t)((fitem + 15) & ~15ull) : 0;
    Fs.addr = m.take((uint64_t)B * (Fs.stride ? Fs.stride : fitem));
  } else {
    c.flags = ctr::F_KW;
    const int groups = NP / 32 + (L.group1 ? 1 : 0);  // the blob's own rows (+ one group before C)
    const uint32_t item = (uint32_t)groups * 64 * 16;     // one head's port array
    Cs.pitch = (uint32_t)H * item;                         // port arrays hold every head's item
    segc = H > 1 ? item : 0;
    const uint64_t blob = 4ull * Cs.pitch;
    Cs.stride = B > 1 ? (uint32_t)blob : 0;
    const uint64_t c0 = m.take((uint64_t)B * blob);
    Cs.addr = c0 + (L.group1 ? 64 * 16 : 0);
    const uint32_t vec = 4 * (uint32_t)(groups * 32);  // a head's s_row vector
    segf = H > 1 ? vec : 0;
    Fs.stride = B > 1 ? H * vec : 0;
    const uint64_t f0 = m.take((uint64_t)B * H * vec);
    Fs.addr = f0 + (L.group1 ? 4 * 32 : 0);
  }
  c.p[ctr::P_SEGC] = segc;
  c.p[ctr::P_SEGF] = segf;
  c.s[ctr::SLOT_A] = A;
  c.s[ctr::SLOT_C] = Cs;
  c.s[ctr::SLOT_E] = E;
  if (L.kind != K_F32) c.s[ctr::SLOT_F] = Fs;
  // record round trip and the contract's checks
  uint32_t f[ctr::REC_FIELDS];
  ctr::encode(c, f);
  ctr::Cmd d;
  const uint32_t st = ctr::decode(f, d), v = ctr::validate(d), vh = ctr::validate_hw(d);
  if (st != ctr::ST_END || v != ctr::ME_OK || vh != ctr::ME_OK) {
    printf("  %s %s @%#llx: record rejected (decode %#x, validate %u, validate_hw %u)\n", cs.name.c_str(), L.name,
           (unsigned long long)base, st, v, vh);
    g_fail++;
    return;
  }
  // memory: sentinel, inputs, expected image
  Ddr mem(base, (size_t)((m.top - base + 4095) / 16));
  mem.sentinel();
  for (int b = 0; b < B; b++)
    for (int n = 0; n < T; n++)
      for (int k = 0; k < K; k++) {
        const uint32_t xv = cs.x[(size_t)n * K + k];
        mem.w32(ctr::row_addr(A, b, n) + 4ull * k) = b ? (xv ^ 0x80000000u) : xv;
      }
  for (int n = 0; n < T; n++)
    for (int k = 0; k < 64; k++) mem.w32(E.addr + (uint64_t)n * E.pitch + 4ull * k) = cs.tab[(size_t)n * 64 + k];
  Ddr want = mem;
  std::vector<float> seg(64);
  std::vector<int16_t> qv(64);
  for (int b = 0; b < B; b++)
    for (int n = 0; n < NP; n++)
      for (int j = 0; j < H; j++) {
        const bool live = n < T;
        uint32_t ys[64];
        for (int k = 0; k < 64; k++) {
          const uint32_t y = live ? cs.f32[(size_t)n * K + 64 * j + k] : 0u;
          ys[k] = b ? neg_f(y) : y;
          seg[k] = ctr::u2f(ys[k]);
        }
        if (L.kind == K_F32) {
          for (int k = 0; k < 64; k++) want.w32(ctr::seg_addr(Cs, b, n, j, segc) + 4ull * k) = ys[k];
          continue;
        }
        const float sv = ctr::ref::quant_row(seg.data(), 64, qv.data(), L.shift);
        if (L.kind == K_Q) {
          for (int k = 0; k < 64; k++) want.set16(ctr::seg_addr(Cs, b, n, j, segc) + 2ull * k, (uint16_t)qv[k]);
        } else {
          ctr::Slot hs = Cs;
          hs.addr += (uint64_t)j * segc;
          for (int k = 0; k < 64; k++)
            want.set16(ctr::w_word_addr(hs, b, n, k, 64) + 2ull * ctr::w_lane(n), (uint16_t)qv[k]);
        }
        want.w32(ctr::scale_addr(Fs, b, j, segf, n)) = f2u(sv);
      }
  if (L.kind == K_Q)  // pad lanes of each head's last scale word
    for (int b = 0; b < B; b++)
      for (int j = 0; j < H; j++)
        for (int n = NP; n < (int)ctr::pad4(NP); n++) want.w32(ctr::scale_addr(Fs, b, j, segf, n)) = ctr::ONE_BITS;
  // run the mode on the image (hp, xmem and ymem are the same DDR)
  std::vector<Word> img = mem.words_out();
  // arm the read-extent check: A rows [0, N) x K values of every item, table rows [0, N)
  g_rd_base = base >> 4;
  g_rd_x.assign(img.size(), 0);
  g_rd_e.assign(img.size(), 0);
  for (int b = 0; b < B; b++)
    for (int n = 0; n < T; n++)
      for (int k = 0; k < K; k += 4) g_rd_x[(ctr::row_addr(A, b, n) + 4ull * k) / 16 - g_rd_base] = 1;
  for (int n = 0; n < T; n++)
    for (int k = 0; k < 64; k += 4) g_rd_e[(E.addr + (uint64_t)n * E.pitch + 4ull * k) / 16 - g_rd_base] = 1;
  const long rd_bad0 = g_rd_bad, rd_units0 = g_rd_units;
  Word* port = reinterpret_cast<Word*>(reinterpret_cast<uintptr_t>(img.data()) - (uintptr_t)(base >> 4) * sizeof(Word));
  const uint32_t r = run_dispatch(d, port);
  g_rd_x.clear();
  g_rd_e.clear();
  const long rd_bad = g_rd_bad - rd_bad0, rd_units = g_rd_units - rd_units0;
  const long want_units = (long)B * NP * H;  // one x + one table unit per segment
  mem.words_in(img);
  long bad = 0, outside = 0;
  for (size_t i = 0; i < mem.u.size(); i++)
    if (mem.u[i] != want.u[i]) {
      const bool was_sentinel = want.u[i] == (0xDEAD0000u ^ (uint32_t)(i * 2654435761u));
      if (bad < 4)
        printf("  %s %s @%#llx: byte %#llx got %08x want %08x%s\n", cs.name.c_str(), L.name, (unsigned long long)base,
               (unsigned long long)(base + 4 * i), mem.u[i], want.u[i], was_sentinel ? " (outside the outputs)" : "");
      bad++;
      outside += was_sentinel;
    }
  g_checked += (long)mem.u.size();
  if (r != ctr::ME_OK || bad || rd_bad || rd_units != want_units) {
    printf("  %-16s %-8s @%#011llx  FAIL: return %u, %ld words differ (%ld outside the outputs), %ld of %ld read units"
           " outside the inputs (want %ld units)\n",
           cs.name.c_str(), L.name, (unsigned long long)base, r, bad, outside, rd_bad, rd_units, want_units);
    g_fail++;
  }
}

// Malformed records: rope_mode returns the code and writes nothing (validate_hw rejects them first).
static void run_rejects() {
  struct R {
    const char* name;
    uint32_t K, N, B, flags, seg, np;
    uint32_t want;
  } rs[] = {
      {"K=0", 0, 4, 1, ctr::F_QOUT, 64, 0, ctr::ME_DIMS},
      {"K=4160 (H=65)", 4160, 4, 1, ctr::F_QOUT, 64, 0, ctr::ME_DIMS},
      {"K=96", 96, 4, 1, ctr::F_QOUT, 64, 0, ctr::ME_DIMS},
      {"SEG=32", 320, 4, 1, ctr::F_QOUT, 32, 0, ctr::ME_DIMS},
      {"N=0", 320, 0, 1, ctr::F_QOUT, 64, 0, ctr::ME_DIMS},
      {"N=65536", 320, 65536, 1, ctr::F_QOUT, 64, 0, ctr::ME_DIMS},
      {"NP<N", 320, 8, 1, ctr::F_QOUT, 64, 4, ctr::ME_PARAM},
      {"KW NP%32", 320, 8, 1, ctr::F_KW, 64, 40, ctr::ME_DIMS},
      {"F32 NP!=N", 320, 8, 1, 0, 64, 12, ctr::ME_PARAM},
      {"QOUT|KW", 320, 8, 1, ctr::F_QOUT | ctr::F_KW, 64, 32, ctr::ME_FLAGS},
      {"B=65536", 320, 8, 65536, ctr::F_QOUT, 64, 0, ctr::ME_DIMS},
  };
  std::vector<Word> img(1 << 12);
  for (size_t i = 0; i < img.size(); i++) img[i] = (Word)(0xC0FFEE00u + (uint32_t)i);
  const std::vector<Word> ref = img;
  for (const R& t : rs) {
    ctr::Cmd c;
    memset(&c, 0, sizeof c);
    c.op = ctr::OP_ROPE;
    c.K = t.K;
    c.N = t.N;
    c.B = t.B;
    c.flags = t.flags;
    c.p[ctr::P_SEG] = t.seg;
    c.p[ctr::P_NP] = t.np;
    c.s[ctr::SLOT_A] = {0x100, 2048, 0};
    c.s[ctr::SLOT_C] = {0x8000, 2048, 0};
    c.s[ctr::SLOT_E] = {0x10000, 256, 0};
    if (t.flags) c.s[ctr::SLOT_F] = {0x18000, 0, 0};
    const long units0 = g_rd_units;  // no run armed: any read is counted
    const uint32_t r = run_dispatch(c, img.data());
    const uint32_t vh = ctr::validate_hw(c);
    const bool same = img == ref, noread = g_rd_units == units0;
    const bool ok = r == t.want && vh != ctr::ME_OK && same && noread;
    printf("  reject %-14s rope_mode %u (want %u), validate_hw %u, memory %s, %s -> %s\n", t.name, r, t.want, vh,
           same ? "untouched" : "WRITTEN", noread ? "nothing read" : "READ", ok ? "ok" : "FAIL");
    if (!ok) g_fail++;
  }
}

int main(int argc, char** argv) {
  const std::string dir = argc > 1 ? argv[1] : "build/vectors";
  const bool quick = argc > 2 && !strcmp(argv[2], "--quick");
  FILE* cf = fopen((dir + "/cases.txt").c_str(), "r");
  if (!cf) {
    fprintf(stderr, "no %s/cases.txt (run gen_vectors.py)\n", dir.c_str());
    return 2;
  }
  char name[256];
  int ncases = 0, nruns = 0;
  long q_ref_bad = 0;
  while (fscanf(cf, "%255s", name) == 1) {
    Case cs;
    cs.name = name;
    const std::string d = dir + "/" + name;
    int f32only = 0;
    FILE* mf = fopen((d + "/meta.txt").c_str(), "r");
    if (!mf || fscanf(mf, "%d %d %d", &cs.T, &cs.H, &f32only) != 3) return 2;
    fclose(mf);
    cs.f32only = f32only;
    const size_t nv = (size_t)cs.T * cs.H * 64;
    cs.x = load<uint32_t>(d + "/x.bin", nv);
    cs.tab = load<uint32_t>(d + "/tab.bin", (size_t)cs.T * 64);
    cs.f32 = load<uint32_t>(d + "/f32.bin", nv);
    cs.q = load<int16_t>(d + "/q.bin", nv);
    cs.s = load<uint32_t>(d + "/s.bin", (size_t)cs.T * cs.H);
    // contract.py quant_rows == ctr::ref::quant_row on this case's FP32 rotation (shift 0)
    if (!cs.f32only) {
      std::vector<float> seg(64);
      std::vector<int16_t> qv(64);
      for (int r = 0; r < cs.T * cs.H; r++) {
        for (int k = 0; k < 64; k++) seg[k] = ctr::u2f(cs.f32[(size_t)r * 64 + k]);
        const float sv = ctr::ref::quant_row(seg.data(), 64, qv.data(), 0);
        if (f2u(sv) != cs.s[r]) q_ref_bad++;
        for (int k = 0; k < 64; k++) q_ref_bad += qv[k] != cs.q[(size_t)r * 64 + k];
      }
    }
    const long fail0 = g_fail;
    int runs = 0;
    for (const Layout& L : LAYOUTS) {
      if (cs.f32only && L.kind != K_F32) continue;
      if (quick && L.B > 1) continue;
      run_layout(cs, L, 0x40000000ull);
      runs++;
      if (!quick && (!strcmp(L.name, "q_hm") || !strcmp(L.name, "kw_g1"))) {
        run_layout(cs, L, 0x800000000ull);
        run_layout(cs, L, 0xFF00000000ull);
        runs += 2;
      }
    }
    printf("%-16s T=%3d H=%2d  %2d layouts  %s\n", name, cs.T, cs.H, runs, g_fail == fail0 ? "ok" : "FAIL");
    ncases++;
    nruns += runs;
  }
  fclose(cf);
  if (q_ref_bad) {
    printf("contract.py quant_rows != ctr::ref::quant_row: %ld values / scales\n", q_ref_bad);
    g_fail++;
  } else {
    printf("contract.py quant_rows == ctr::ref::quant_row on every segment of every case\n");
  }
  run_rejects();
  if (g_wrong_unit) {
    printf("seq::dispatch reached a unit other than row() / OP_ROPE %ld times\n", g_wrong_unit);
    g_fail++;
  }
  printf("TOTAL %d cases, %d layout runs, %ld memory words compared, %ld read units checked (%ld outside the inputs),"
         " %ld failures -> %s\n",
         ncases, nruns, g_checked, g_rd_units, g_rd_bad, g_fail, g_fail ? "FAIL" : "PASS");
  return g_fail ? 1 : 0;
}
