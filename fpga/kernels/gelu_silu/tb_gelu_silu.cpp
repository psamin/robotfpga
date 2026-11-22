// Host testbench of gs::act_mode (the GELU / SILU_MUL mode, C model of the HLS code; Vitis MinGW g++ with the HLS
// headers, -ffp-contract=off).
//   tb_gelu_silu [--plan LIST.bin] CASE_DIR [CASE_DIR ...]
// Build with -DGS_DFSIM -Idfsim (ahead of the Vitis include) for the threaded bounded-FIFO model: every DATAFLOW
// process a thread, every stream at its pragma depth (GS_D_*); a deadlock prints DEADLOCK and exits 7.
//
// Each case (make_vectors.py / policy_check.py --dump): meta.txt "mode N K nsb check", x.bin, [u.bin], y.bin, q.bin,
// s.bin. Every case runs as contract 1.0 records (ctr::encode -> ctr::decode -> ctr::validate == validate_hw ==
// ME_OK) through act_mode on a sparse 40-bit memory (only the declared buffers exist; any other access aborts), in 5
// layouts that between them take every reader and writer path (the expected path is checked through gs_plan):
//   plan    the -12392 list's layout (contract_plan.py): rows contiguous (SILU_MUL: the fused gate|up row, B = A + 4K),
//           q rows contiguous, F_QOUT, base 0x8_0000_0000 (high DDR)              -> dense read, dense write
//   padded  A pitch + 48 B, B in its own buffer, C pitch + 32 B, 2 items, NP = N + 3 zero rows, p4 = -3,
//           base 0xFF_0000_0000 (top of the 40-bit space)          -> dense read with gaps (SILU_MUL sparse), sparse write
//   wide    A pitch + 2 KB (fused gate|up for SILU_MUL), C contiguous, 2 items with gaps between them, NP = N + 5,
//           p4 = +2, base 0x2_0000_0000                            -> sparse read (N > 1), dense write with zero rows
//   fp32    FP32 rows out (no F_QOUT), A pitch + 48 B (fused), C pitch + 16 B, base 0x4000_0000 -> dense, sparse
//   fp32d   FP32 rows out, everything contiguous, 2 items                                       -> dense, dense
// Checks: y bits (FP32 layouts) == gs_emu; q and stored s == contract.quant_rows of gs_emu's y (q.bin, s.bin) ==
// ctr::ref::quant_row (C++ reference); zero rows / pad lanes; guards and inputs untouched; nothing outside the
// buffers touched. Cases with non-finite inputs (check 0): rows with only finite inputs are checked as above, rows
// with a NaN / Inf must store a non-finite scale. Then bad commands (act_mode returns non-zero and touches nothing),
// the csynth top's pointer instantiation, and with --plan the path census of every GELU / SILU_MUL record of the
// -12392 list (contract_plan.py --out): all must take the dense read and the dense write.
// Prints the error against a float64 activation in units of the token's step s. Exit 1 on failure.
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "gelu_silu.h"

uint32_t gelu_silu_top(const gs::Word* xmem, gs::Word* ymem, uint32_t opflags, uint32_t K, uint32_t N, uint32_t B,
                       uint64_t a, uint32_t pa, uint32_t sa, uint64_t b, uint32_t pb, uint32_t sb, uint64_t c, uint32_t pc,
                       uint32_t sc, uint64_t f, uint32_t sf, uint32_t p4, uint32_t p5);

#ifdef GS_DFSIM
int gs_dfsim_depth(const std::string& n) {
  static const std::map<std::string, int> d = {{"gs_xin", GS_D_XIN}, {"gs_yf", GS_D_YF}, {"gs_ys", GS_D_YS},
                                               {"gs_am", GS_D_AM},   {"gs_rsf", GS_D_RSF}, {"gs_sst", GS_D_SST},
                                               {"gs_ow", GS_D_OW}};
  auto it = d.find(n);
  if (it == d.end()) {
    fprintf(stderr, "DFSIM: unknown stream %s\n", n.c_str());
    exit(9);
  }
  return it->second;
}
#endif

using gs::Word;

// ------------------------------------------------------------------ sparse memory (word addressed)
struct Mem {
  struct Region {
    uint64_t base;  // word index
    std::vector<Word> w;
  };
  std::vector<Region> r;
  std::atomic<long> accesses{0};
  uint64_t add(uint64_t byte_addr, size_t words, const Word& fill) {
    r.push_back(Region{byte_addr >> 4, std::vector<Word>(words, fill)});
    return byte_addr;
  }
  Word& at(uint64_t wi) {
    accesses++;
    for (auto& g : r)
      if (wi >= g.base && wi < g.base + g.w.size()) return g.w[wi - g.base];
    fprintf(stderr, "FATAL: access outside every buffer: byte address 0x%llx\n", (unsigned long long)(wi << 4));
    exit(3);
  }
  Word& byte(uint64_t a) { return at(a >> 4); }
};
struct MemRef {  // what act_mode gets in place of a pointer
  Mem* m;
  Word& operator[](uint64_t wi) const { return m->at(wi); }
};

template <class T>
static std::vector<T> slurp(const std::string& p, size_t n) {
  std::vector<T> v(n);
  FILE* f = fopen(p.c_str(), "rb");
  if (!f || fread(v.data(), sizeof(T), n, f) != n) {
    fprintf(stderr, "cannot read %s\n", p.c_str());
    exit(2);
  }
  fclose(f);
  return v;
}
static float b2f(uint32_t b) {
  float f;
  memcpy(&f, &b, 4);
  return f;
}
static uint32_t f2b(float f) {
  uint32_t b;
  memcpy(&b, &f, 4);
  return b;
}
static bool finite_bits(uint32_t b) { return ((b >> 23) & 0xFF) != 0xFF; }
static double act_ref(int mode, double x, double u) {
  if (mode == 0) return 0.5 * x * (1.0 + std::tanh(std::sqrt(2.0 / 3.14159265358979323846) * (x + 0.044715 * x * x * x)));
  if (mode == 2) return x / (1.0 + std::exp(-x));
  return x / (1.0 + std::exp(-x)) * u;
}
static uint32_t lane32(const Word& w, int j) { return (uint32_t)w(32 * j + 31, 32 * j); }
static int16_t lane16(const Word& w, int j) { return (int16_t)(uint16_t)w(16 * j + 15, 16 * j); }
static const Word GUARD = (Word(0xDEADBEEFu) << 96) | (Word(0x12345678u) << 32) | Word(0x5A5A5A5Au);
static uint64_t al16(uint64_t v) { return (v + 15) / 16 * 16; }

struct Case {
  std::string d;
  int mode, N, K, nsb, check;
  std::vector<float> x, u;
  std::vector<uint32_t> y, s;
  std::vector<int16_t> q;
  std::vector<char> rowok;  // all inputs of the row finite
};

struct Layout {
  const char* name;
  bool qout;
  int items;      // 2 when N is even, else 1
  int np_extra;   // NP = N_item + np_extra
  int shift;      // p4
  uint64_t base;  // buffers from here
  uint32_t a_gap;  // bytes of pitch gap after each input row (A; fused SILU_MUL: after gate | up)
  bool fused;     // SILU_MUL: up at A + 4K in the same row (else its own buffer, pitch 4K + 80)
  uint32_t c_gap;  // bytes of pitch gap after each output row
};
static const Layout LAYOUTS[] = {
    {"plan", true, 1, 0, 0, 0x800000000ull, 0, true, 0},
    {"padded", true, 2, 3, -3, 0xFF00000000ull, 48, false, 32},
    {"wide", true, 2, 5, 2, 0x200000000ull, 2048, true, 0},
    {"fp32", false, 1, 0, 0, 0x40000000ull, 48, true, 16},
    {"fp32d", false, 2, 0, 0, 0x60000000ull, 0, true, 0},
};

static long g_paths[2][2];  // [rdense][wdense] commands run

static uint32_t run_cmd(const ctr::Cmd& c0, Mem& mem, bool* valid, gs::GsCmd* plan) {
  uint32_t f[ctr::REC_FIELDS];
  ctr::encode(c0, f);
  ctr::Cmd c;
  const uint32_t st = ctr::decode(f, c);
  const uint32_t v = ctr::validate(c), vh = ctr::validate_hw(c);
  *valid = st == ctr::ST_END && v == ctr::ME_OK && vh == ctr::ME_OK;
  if (!*valid) {
    fprintf(stderr, "  record invalid: decode %u validate %u validate_hw %u\n", st, v, vh);
    return 99;
  }
  *plan = gs::gs_plan(c);
  return gs::act_mode(c, MemRef{&mem}, MemRef{&mem});
}

// One layout of one case; returns the number of failures.
static long run_layout(const Case& cs, const Layout& L, double* emax, double* esum2, long* cnt) {
  const int N = cs.N, K = cs.K;
  const bool mul = cs.mode == 1, fp32 = !L.qout;
  const int items = (L.items == 2 && N % 2 == 0) ? 2 : 1;
  const int Ni = N / items;  // rows per item
  const int NP = L.qout ? Ni + L.np_extra : Ni;
  const int shift = L.qout ? L.shift : 0;
  const bool fused = mul && L.fused;
  Mem mem;
  ctr::Cmd c = {};
  c.op = cs.mode == 0 ? ctr::OP_GELU : ctr::OP_SILU_MUL;
  c.flags = (fp32 ? 0 : ctr::F_QOUT) | (cs.mode == 2 ? ctr::F_NOMUL : 0);
  c.tag = 7;
  c.K = K;
  c.N = Ni;
  c.B = items;
  c.p[ctr::P_SHIFT] = (uint32_t)shift;
  c.p[ctr::P_NP] = NP == Ni ? 0 : NP;
  const uint32_t rowb = 4 * K;
  // ---- inputs
  ctr::Slot &A = c.s[ctr::SLOT_A], &Bs = c.s[ctr::SLOT_B], &C = c.s[ctr::SLOT_C], &F = c.s[ctr::SLOT_F];
  A.addr = L.base + 0x1000;
  A.pitch = (fused ? 2 * rowb : rowb) + L.a_gap;
  A.stride = items > 1 ? (uint32_t)al16((uint64_t)Ni * A.pitch + 32) : 0;
  const uint64_t a_bytes = items > 1 ? (uint64_t)items * A.stride : (uint64_t)Ni * A.pitch;
  uint64_t b_bytes = 0;
  if (fused) {
    Bs = ctr::Slot{A.addr + rowb, A.pitch, A.stride};
  } else if (mul) {
    Bs.pitch = rowb + 80;
    Bs.stride = items > 1 ? (uint32_t)al16((uint64_t)Ni * Bs.pitch + 16) : 0;
    Bs.addr = L.base + 0x100000 + al16(a_bytes);
    b_bytes = items > 1 ? (uint64_t)items * Bs.stride : (uint64_t)Ni * Bs.pitch;
  }
  mem.add(A.addr, a_bytes / 16 + 4, GUARD);
  if (mul && !fused) mem.add(Bs.addr, b_bytes / 16 + 4, GUARD);
  for (int bi = 0; bi < items; bi++)
    for (int n = 0; n < Ni; n++)
      for (int k = 0; k < K; k++) {
        const size_t e = (size_t)(bi * Ni + n) * K + k;
        mem.byte(A.addr + (uint64_t)bi * A.stride + (uint64_t)n * A.pitch + 4 * k)(32 * (k % 4) + 31, 32 * (k % 4)) = f2b(cs.x[e]);
        if (mul)
          mem.byte(Bs.addr + (uint64_t)bi * Bs.stride + (uint64_t)n * Bs.pitch + 4 * k)(32 * (k % 4) + 31, 32 * (k % 4)) =
              f2b(cs.u[e]);
      }
  // ---- outputs
  const uint32_t orow = fp32 ? 4 * K : 2 * K;
  C.pitch = orow + L.c_gap;
  C.stride = items > 1 ? (uint32_t)al16((uint64_t)NP * C.pitch + 48) : 0;
  C.addr = L.base + 0x4000000;
  const uint64_t c_bytes = (uint64_t)(items - 1) * C.stride + (uint64_t)NP * C.pitch;
  mem.add(C.addr, c_bytes / 16 + 4, GUARD);
  const uint32_t fwords = (NP + 3) / 4;
  if (!fp32) {
    F.addr = C.addr + al16(c_bytes) + 0x10000;
    F.stride = items > 1 ? 16 * fwords + 32 : 0;
    F.pitch = 0;
    mem.add(F.addr, (uint64_t)items * (fwords + 2) + 4, GUARD);
  }
  // snapshot the inputs
  std::vector<Mem::Region> before = mem.r;
  bool valid;
  gs::GsCmd plan;
  mem.accesses = 0;
  const uint32_t rc = run_cmd(c, mem, &valid, &plan);
  long fails = 0;
  if (!valid || rc != 0) {
    fprintf(stderr, "%s [%s]: act_mode returned %u\n", cs.d.c_str(), L.name, rc);
    return 1;
  }
  // the path the layout is meant to take
  const bool exp_rd = (Ni == 1 || L.a_gap / 16 <= (uint32_t)gs::GAPW) && (!mul || fused);
  const bool exp_wr = NP == 1 || L.c_gap == 0;
  g_paths[plan.rdense][plan.wdense]++;
  if (plan.rdense != exp_rd || plan.wdense != exp_wr) {
    fprintf(stderr, "  [%s] path: rdense %d wdense %d, expected %d %d\n", L.name, (int)plan.rdense, (int)plan.wdense,
            (int)exp_rd, (int)exp_wr);
    fails++;
  }
  // inputs unchanged
  for (size_t i = 0; i < 1 + (mul && !fused ? 1u : 0u); i++)
    if (mem.r[i].w != before[i].w) {
      fprintf(stderr, "  input buffer %zu modified\n", i);
      fails++;
    }
  long bq = 0, bs = 0, by = 0, bg = 0, bnf = 0;
  for (int bi = 0; bi < items; bi++) {
    for (int n = 0; n < NP; n++) {
      const bool real = n < Ni;
      const size_t rowi = (size_t)bi * Ni + (real ? n : 0);
      const bool ok = !real || cs.rowok[rowi];
      const uint64_t row = C.addr + (uint64_t)bi * C.stride + (uint64_t)n * C.pitch;
      for (int k = 0; k < K && ok; k++) {
        const size_t e = rowi * K + k;
        if (fp32) {
          const uint32_t yv = lane32(mem.byte(row + 4 * k), k % 4);
          if (yv != cs.y[e]) {
            if (by < 5) fprintf(stderr, "  y mismatch item %d row %d k %d: %08x vs %08x (x %.9g)\n", bi, n, k, yv, cs.y[e], cs.x[e]);
            by++;
          }
        } else {
          const int16_t qv = lane16(mem.byte(row + 2 * k), k % 8);
          const int16_t qe = real ? cs.q[e] : 0;
          if (qv != qe) {
            if (bq < 5) fprintf(stderr, "  q mismatch item %d row %d k %d: %d vs %d (x %.9g)\n", bi, n, k, qv, qe, cs.x[e]);
            bq++;
          }
        }
      }
      // guard words between rows (pitch padding)
      for (uint64_t g = al16(orow); g < C.pitch; g += 16)
        if (mem.byte(row + g) != GUARD) bg++;
    }
    if (!fp32) {
      for (int n = 0; n < 4 * (int)fwords; n++) {
        const uint64_t fa = F.addr + (uint64_t)bi * F.stride + 4 * n;
        const uint32_t sv = lane32(mem.byte(fa), n % 4);
        if (n < Ni && !cs.rowok[bi * Ni + n]) {  // non-finite input: the stored scale must not hide it
          if (finite_bits(sv)) {
            if (bnf < 5) fprintf(stderr, "  item %d row %d: non-finite input, finite stored scale %08x\n", bi, n, sv);
            bnf++;
          }
          continue;
        }
        uint32_t se;
        if (n < Ni) {
          se = cs.s[bi * Ni + n];
          if (se != ctr::ONE_BITS) se = f2b(ldexpf(b2f(se), shift));
        } else {
          se = ctr::ONE_BITS;  // zero rows and pad lanes
        }
        if (sv != se) {
          if (bs < 5) fprintf(stderr, "  s mismatch item %d row %d: %08x vs %08x\n", bi, n, sv, se);
          bs++;
        }
      }
      if (items > 1 && bi == 0)
        for (uint64_t g = 16 * fwords; g < F.stride; g += 16)
          if (mem.byte(F.addr + g) != GUARD) bg++;
    }
  }
  // guard words after the output extents and between items
  for (auto& g : mem.r) {
    if (g.base == (C.addr >> 4) || (!fp32 && g.base == (F.addr >> 4))) {
      const bool isc = g.base == (C.addr >> 4);
      const uint64_t used = isc ? (c_bytes + 15) / 16 : (uint64_t)(items - 1) * (F.stride / 16) + fwords;
      for (uint64_t i = used; i < g.w.size(); i++)
        if (g.w[i] != GUARD) bg++;
      if (items > 1 && isc)
        for (uint64_t i = ((uint64_t)NP * C.pitch + 15) / 16; i < C.stride / 16; i++)
          if (g.w[i] != GUARD) bg++;
    }
  }
  // error against float64 (plan layout only: no shift)
  if (!strcmp(L.name, "plan") && cs.check)
    for (int n = 0; n < N; n++) {
      const double s = b2f(cs.s[n]);
      for (int k = 0; k < K; k++) {
        const size_t e = (size_t)n * K + k;
        const double xv = cs.x[e], uv = mul ? cs.u[e] : 0.0;
        if (!(std::fabs(xv) < 1e30 && std::fabs(uv) < 1e20)) continue;
        const double err = std::fabs(cs.q[e] * s - act_ref(cs.mode, xv, uv)) / s;
        if (err > *emax) *emax = err;
        *esum2 += err * err;
        (*cnt)++;
      }
    }
  if (bq || bs || by || bg || bnf)
    fprintf(stderr, "  [%s] q %ld s %ld y %ld guard %ld non-finite %ld mismatches\n", L.name, bq, bs, by, bg, bnf);
  return fails + bq + bs + by + bg + bnf;
}

static int run_case(const std::string& d) {
  Case cs;
  cs.d = d;
  FILE* f = fopen((d + "/meta.txt").c_str(), "r");
  if (!f || fscanf(f, "%d %d %d %d %d", &cs.mode, &cs.N, &cs.K, &cs.nsb, &cs.check) != 5) {
    fprintf(stderr, "bad meta in %s\n", d.c_str());
    return 1;
  }
  fclose(f);
  if (cs.nsb != GS_NSB) {
    fprintf(stderr, "%s: vectors for NSB %d, kernel built with %d\n", d.c_str(), cs.nsb, GS_NSB);
    return 1;
  }
  const size_t ne = (size_t)cs.N * cs.K;
  cs.x = slurp<float>(d + "/x.bin", ne);
  cs.u.assign(ne, 0.0f);
  if (cs.mode == 1) cs.u = slurp<float>(d + "/u.bin", ne);
  cs.y = slurp<uint32_t>(d + "/y.bin", ne);
  cs.q = slurp<int16_t>(d + "/q.bin", ne);
  cs.s = slurp<uint32_t>(d + "/s.bin", cs.N);
  cs.rowok.assign(cs.N, 1);
  int nbad = 0;
  for (int n = 0; n < cs.N; n++)
    for (int k = 0; k < cs.K; k++) {
      const size_t e = (size_t)n * cs.K + k;
      if (!finite_bits(f2b(cs.x[e])) || !finite_bits(f2b(cs.u[e]))) cs.rowok[n] = 0;
    }
  for (int n = 0; n < cs.N; n++) nbad += !cs.rowok[n];
  if (cs.check && nbad) {
    fprintf(stderr, "%s: check = 1 with %d non-finite rows\n", d.c_str(), nbad);
    return 1;
  }
  long fails = 0, rq = 0;
  // the C++ reference quantizer (ctr::ref) on gs_emu's y == contract.quant_rows (q.bin, s.bin), finite rows
  {
    std::vector<float> yr(cs.K);
    std::vector<int16_t> qr(cs.K);
    for (int n = 0; n < cs.N; n++) {
      if (!cs.rowok[n]) continue;
      for (int k = 0; k < cs.K; k++) yr[k] = b2f(cs.y[(size_t)n * cs.K + k]);
      const float s = ctr::ref::quant_row(yr.data(), cs.K, qr.data());
      if (f2b(s) != cs.s[n]) rq++;
      for (int k = 0; k < cs.K; k++) rq += qr[k] != cs.q[(size_t)n * cs.K + k];
    }
    if (rq) fprintf(stderr, "  ctr::ref::quant_row differs from contract.quant_rows: %ld\n", rq);
  }
  double emax = 0, esum2 = 0;
  long cnt = 0;
  for (const Layout& L : LAYOUTS) fails += run_layout(cs, L, &emax, &esum2, &cnt);
  printf("%-34s mode %d N %4d K %4d: %s; |q s - y_ref| / s max %.4f rms %.4f (%ld el.)%s\n", d.c_str(), cs.mode, cs.N,
         cs.K, (fails + rq) ? "FAIL" : "pass", emax, cnt ? std::sqrt(esum2 / cnt) : 0.0, cnt,
         nbad ? " [rows with NaN / Inf: scale non-finite checked]" : "");
  return (fails + rq) ? 1 : 0;
}

// Commands act_mode must refuse without touching memory (validate_hw rejects them too).
static int bad_commands() {
  int fails = 0;
  struct T {
    uint32_t op, flags, K, N, B, np, sh;
  } t[] = {
      {ctr::OP_GELU, ctr::F_QOUT, 12, 4, 1, 0, 0},                    // K % 8
      {ctr::OP_GELU, ctr::F_QOUT, 3080, 4, 1, 0, 0},                  // K > ACT_K_MAX
      {ctr::OP_GELU, ctr::F_QOUT, 0, 4, 1, 0, 0},                     // K = 0
      {ctr::OP_SILU_MUL, ctr::F_QOUT, 64, 0, 1, 0, 0},                // N = 0
      {ctr::OP_SILU_MUL, ctr::F_QOUT, 64, 70000, 1, 0, 0},            // N > N_MAX
      {ctr::OP_GELU, ctr::F_QOUT | ctr::F_NOMUL, 64, 4, 1, 0, 0},     // GELU + NOMUL
      {ctr::OP_NORM, ctr::F_QOUT, 64, 4, 1, 0, 0},                    // not this mode
      {ctr::OP_GELU, ctr::F_QOUT, 64, 4, 1, 3, 0},                    // NP < N
      {ctr::OP_GELU, ctr::F_QOUT, 64, 4, 1, 0, 9},                    // |shift| > 8
      {ctr::OP_GELU, ctr::F_QOUT, 64, 4, 1, 0, (uint32_t)-9},         // |shift| > 8
      {ctr::OP_GELU, 0, 64, 4, 1, 6, 0},                              // FP32 output with NP != N
      {ctr::OP_GELU, 0, 64, 4, 1, 0, 1},                              // FP32 output with a shift
  };
  for (const auto& b : t) {
    Mem mem;
    ctr::Cmd c = {};
    c.op = b.op, c.flags = b.flags, c.K = b.K, c.N = b.N, c.B = b.B;
    c.p[ctr::P_NP] = b.np, c.p[ctr::P_SHIFT] = b.sh;
    c.s[ctr::SLOT_A] = ctr::Slot{0x1000, 4 * b.K, 0};
    const uint32_t r = gs::act_mode(c, MemRef{&mem}, MemRef{&mem});
    if (r == 0 || mem.accesses != 0) {
      fprintf(stderr, "bad command op %#x flags %#x K %u N %u np %u sh %d: returned %u, %ld accesses\n", b.op, b.flags,
              b.K, b.N, b.np, (int)b.sh, r, (long)mem.accesses);
      fails++;
    }
  }
  printf("bad commands: %d of %zu not refused\n", fails, sizeof t / sizeof t[0]);
  return fails;
}

// The csynth top (plain pointers, as float_top passes them) on a small case: the pointer instantiation must equal
// the MemRef one.
static int top_check() {
  const int N = 3, K = 64;
  std::vector<Word> in(N * K / 2 + 8, GUARD), out(N * K / 8 + 8, GUARD);
  std::vector<float> xv(N * K), uv(N * K);
  for (int i = 0; i < N * K; i++) {
    xv[i] = std::sin(0.37f * i) * 6.0f;
    uv[i] = std::cos(0.11f * i) * 2.0f;
  }
  for (int n = 0; n < N; n++)
    for (int k = 0; k < K; k++) {
      in[(n * 2 * K + k) / 4](32 * (k % 4) + 31, 32 * (k % 4)) = f2b(xv[n * K + k]);
      in[(n * 2 * K + K + k) / 4](32 * (k % 4) + 31, 32 * (k % 4)) = f2b(uv[n * K + k]);
    }
  const uint32_t fw = N * K / 8;  // scale vector right after the q rows (word index)
  const uint32_t r = gelu_silu_top(in.data(), out.data(), ctr::OP_SILU_MUL | (ctr::F_QOUT << 16), K, N, 1, 0, 8 * K, 0,
                                   4 * K, 8 * K, 0, 0, 2 * K, 0, 16ull * fw, 0, 0, 0);
  // reference: the same command through MemRef
  Mem mem;
  mem.r.push_back(Mem::Region{0, in});
  mem.r.push_back(Mem::Region{0x100000, std::vector<Word>(out.size(), GUARD)});
  ctr::Cmd c = {};
  c.op = ctr::OP_SILU_MUL, c.flags = ctr::F_QOUT, c.K = K, c.N = N, c.B = 1;
  c.s[ctr::SLOT_A] = ctr::Slot{0, 8 * K, 0};
  c.s[ctr::SLOT_B] = ctr::Slot{4 * K, 8 * K, 0};
  c.s[ctr::SLOT_C] = ctr::Slot{0x1000000, 2 * K, 0};
  c.s[ctr::SLOT_F] = ctr::Slot{0x1000000 + 16ull * fw, 0, 0};
  const uint32_t r2 = gs::act_mode(c, MemRef{&mem}, MemRef{&mem});
  int bad = (r != 0 || r2 != 0) ? 1 : 0;
  for (size_t i = 0; i < out.size(); i++) bad += out[i] != mem.r[1].w[i];
  printf("csynth top (pointers) vs MemRef: %s\n", bad ? "FAIL" : "equal");
  return bad ? 1 : 0;
}

// Path census of the GELU / SILU_MUL records of an encoded list (contract_plan.py --out): every record must decode,
// validate and take the dense read and the dense write (the burst-per-item / per-chunk paths).
static int plan_census(const std::string& path) {
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) {
    fprintf(stderr, "cannot read %s\n", path.c_str());
    return 1;
  }
  std::vector<unsigned char> bin;
  unsigned char buf[ctr::REC_BYTES];
  while (fread(buf, 1, ctr::REC_BYTES, f) == ctr::REC_BYTES) bin.insert(bin.end(), buf, buf + ctr::REC_BYTES);
  fclose(f);
  const size_t n = bin.size() / ctr::REC_BYTES;
  int fails = 0;
  std::map<std::string, long> shapes;
  long cnt = 0, dense = 0;
  for (size_t i = 0; i < n; i++) {
    uint32_t fl[ctr::REC_FIELDS];
    memcpy(fl, &bin[i * ctr::REC_BYTES], ctr::REC_BYTES);
    ctr::Cmd c;
    if (ctr::decode(fl, c) != ctr::ST_END) {
      fails++;
      continue;
    }
    if (c.op != ctr::OP_GELU && c.op != ctr::OP_SILU_MUL) continue;
    cnt++;
    if (ctr::validate(c) != ctr::ME_OK || ctr::validate_hw(c) != ctr::ME_OK) {
      fprintf(stderr, "plan record %zu invalid\n", i);
      fails++;
      continue;
    }
    const gs::GsCmd p = gs::gs_plan(c);
    const bool ok = p.rdense && p.wdense;
    dense += ok;
    if (!ok) fails++;
    char key[160];
    snprintf(key, sizeof key, "%s%s K %u N %u B %u A.pitch %u C.pitch %u: rdense %d wdense %d qout %d",
             c.op == ctr::OP_GELU ? "GELU" : "SILU_MUL", (c.flags & ctr::F_NOMUL) ? "+NOMUL" : "", c.K, c.N, c.B,
             c.s[ctr::SLOT_A].pitch, c.s[ctr::SLOT_C].pitch, (int)p.rdense, (int)p.wdense, (int)p.qout);
    shapes[key]++;
  }
  printf("plan %s: %zu records, %ld GELU / SILU_MUL, %ld on the dense read + dense write paths\n", path.c_str(), n, cnt,
         dense);
  for (auto& s : shapes) printf("  %4ld x %s\n", s.second, s.first.c_str());
  return fails;
}

int main(int argc, char** argv) {
  int fails = 0;
#ifdef GS_DFSIM
  printf("threaded bounded-FIFO model (GS_DFSIM): every process a thread, every stream at its pragma depth "
         "(xin %d yf %d ys %d am %d rsf %d sst %d ow %d)\n",
         GS_D_XIN, GS_D_YF, (int)(GS_D_YS), GS_D_AM, GS_D_RSF, GS_D_SST, GS_D_OW);
#endif
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--plan") && i + 1 < argc) {
      fails += plan_census(argv[++i]);
      continue;
    }
    fails += run_case(argv[i]);
  }
  fails += bad_commands();
  fails += top_check();
  printf("paths run (rdense, wdense): 00 %ld, 01 %ld, 10 %ld, 11 %ld\n", g_paths[0][0], g_paths[0][1], g_paths[1][0],
         g_paths[1][1]);
  printf(fails ? "FAIL (%d)\n" : "PASS\n", fails);
  return fails ? 1 : 0;
}
