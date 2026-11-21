// Host testbench of the float_top command sequencer (seq_core.h + seq_top.cpp with the stub modes).
//
//   export PATH="/c/Xilinx/Vitis_HLS/2022.2/tps/win64/msys64/mingw64/bin:/c/Xilinx/Vitis_HLS/2022.2/tps/win64/msys64/usr/bin:/c/Windows/System32:/c/Windows"
//   g++ -std=c++14 -O2 -Wall -o build/tb_seq.exe tb_seq.cpp && build/tb_seq.exe [build/py]
//
// 1. GEMM stub arithmetic, bit-exact against a scalar reference (int16 x int16 exact sums, FP32 dequant,
//    bias, D, C accumulate, batch items, GQA weight sharing, partial token tiles).
// 2. The walk: END, YIELD + resume, single step, every error code, the status block, the legacy protocol.
// 3. If <dir>/chunk_list.bin exists (check_policy.py --dump): the whole -12392 chunk list is decoded by the
//    C++ sequencer (modes dry) and every dispatched command must print exactly the line of
//    <dir>/chunk_trace.txt (written by seq_format.Cmd.trace_line from the Python builder's objects).
#include "seq_top.cpp"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

static int g_fail = 0;
#define CHECK(cond, ...)                         \
  do {                                           \
    if (!(cond)) {                               \
      g_fail++;                                  \
      printf("FAIL %s:%d: ", __FILE__, __LINE__); \
      printf(__VA_ARGS__);                       \
      printf("\n");                              \
    }                                            \
  } while (0)

// ---- one flat memory: AXI address == byte offset (all pointer registers 0 in sequencer mode)
static std::vector<Word> g_mem;
static uint8_t* bp(uint64_t a) { return (uint8_t*)g_mem.data() + a; }
static void put_u32(uint64_t a, uint32_t v) { memcpy(bp(a), &v, 4); }
static uint32_t get_u32(uint64_t a) {
  uint32_t v;
  memcpy(&v, bp(a), 4);
  return v;
}
static void put_f32(uint64_t a, float v) { memcpy(bp(a), &v, 4); }
static float get_f32(uint64_t a) {
  float v;
  memcpy(&v, bp(a), 4);
  return v;
}
static void put_i16(uint64_t a, int16_t v) { memcpy(bp(a), &v, 2); }

struct Sl {
  uint64_t addr;
  uint32_t pitch, stride;
};
struct Rec {
  uint32_t f[32];
};

// The C++ twin of seq_format.Cmd.fields().
static Rec mk(uint32_t op, uint32_t flags, uint32_t tag, uint32_t M, uint32_t K, uint32_t N, uint32_t B, uint32_t G,
              const Sl s[6], const uint32_t p[7]) {
  Rec r;
  memset(&r, 0, sizeof r);
  r.f[0] = op | (SEQ_MAGIC << 8) | (flags << 16);
  r.f[1] = tag;
  r.f[2] = M;
  r.f[3] = K;
  r.f[4] = N;
  r.f[5] = (B & 0xFFFF) | ((G & 0xFFFF) << 16);
  for (int i = 0; i < 6; i++) {
    r.f[6 + 3 * i] = (uint32_t)s[i].addr;
    r.f[7 + 3 * i] = (uint32_t)(s[i].addr >> 32) | (s[i].pitch << 8);
    r.f[8 + 3 * i] = s[i].stride;
  }
  for (int i = 0; i < 7; i++) r.f[24 + i] = p[i];
  uint32_t x = SEQ_CHECK_SEED;
  for (int i = 0; i < 31; i++) x ^= r.f[i];
  r.f[31] = x;
  return r;
}
static Rec mk_simple(uint32_t op, uint32_t tag, uint32_t flags = 0) {
  Sl s[6] = {};
  uint32_t p[7] = {};
  return mk(op, flags, tag, 0, 0, 0, 1, 1, s, p);
}
static void store(uint64_t list, int idx, const Rec& r) { memcpy(bp(list + 128ull * idx), r.f, 128); }
static void reseal(uint64_t list, int idx) {  // recompute the check word after a deliberate edit
  uint32_t x = SEQ_CHECK_SEED;
  for (int i = 0; i < 31; i++) x ^= get_u32(list + 128ull * idx + 4 * i);
  put_u32(list + 128ull * idx + 124, x);
}

static uint32_t start(uint64_t list, uint64_t status, uint32_t first, bool step = false) {
  const uint32_t M = (MODE_MAGIC << 24) | ((uint32_t)step << MODE_STEP_BIT) | first;
  const Word* m = g_mem.data();
  return (uint32_t)seq_top(m, m, m, m, m, g_mem.data(), (int)M, (int)(uint32_t)(list >> 4), (int)(uint32_t)(status >> 4));
}

struct Status {
  uint32_t code, next, tag, done, merr, magic, starts, lastop;
};
static Status status_at(uint64_t a) {
  Status s;
  memcpy(&s, bp(a), 32);
  return s;
}

static uint32_t u32(float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  return u;
}

// ---- 1. GEMM arithmetic
// Weight blob of the engine (port p array at base + p * pitch: [group][k] words, lane l = row 32 g + 8 p + l),
// FP32 row scales at e, bias at e + 4 Mp.
static void put_weights(uint64_t base, uint32_t pitch, uint64_t e, const std::vector<int16_t>& W, const std::vector<float>& s,
                        const std::vector<float>& bias, int Mp, int K) {
  for (int g = 0; g < Mp / 32; g++)
    for (int k = 0; k < K; k++)
      for (int p = 0; p < 4; p++)
        for (int l = 0; l < 8; l++)
          put_i16(base + (uint64_t)p * pitch + ((uint64_t)g * K + k) * 16 + 2 * l, W[(size_t)(32 * g + 8 * p + l) * K + k]);
  for (int m = 0; m < Mp; m++) {
    put_f32(e + 4 * m, s[m]);
    put_f32(e + 4 * (Mp + m), bias.empty() ? 0.0f : bias[m]);
  }
}

static void test_gemm(std::mt19937& rng, int M, int K, int N, int B, int G, bool bias, bool acc, bool addd, float p0) {
  const uint64_t LIST = 0x1000, STAT = 0x800, XA = 0x10000, XS = 0x80000, WB = 0x100000, YC = 0x400000, DD = 0x7F0000;
  std::uniform_int_distribution<int> qd(-32767, 32767);
  std::uniform_real_distribution<float> fd(-1.0f, 1.0f);
  const int nw = (B + G - 1) / G;
  const uint32_t a_pitch = K * 2 + 32, a_stride = (uint32_t)(N * a_pitch + 64);
  const uint32_t c_pitch = M * 4 + 16, c_stride = (uint32_t)(N * c_pitch + 128);
  const uint32_t f_stride = (uint32_t)((N * 4 + 15) / 16 * 16 + 16);
  const uint32_t w_pitch = (uint32_t)(M / 32) * K * 16;  // per-port array
  const uint32_t w_stride = 4 * w_pitch + 4096;
  const uint32_t e_stride = w_stride;  // scales live after each item's data
  std::vector<std::vector<int16_t>> Ws(nw), Xs(B);
  std::vector<std::vector<float>> Ss(nw), Bs(nw), xs(B), C0(B);
  std::vector<float> D(M);
  for (int j = 0; j < nw; j++) {
    Ws[j].resize((size_t)M * K);
    for (auto& v : Ws[j]) v = (int16_t)qd(rng);
    Ss[j].resize(M);
    for (auto& v : Ss[j]) v = std::ldexp(1.0f + 0.5f * fd(rng), -20);
    if (bias) {
      Bs[j].resize(M);
      for (auto& v : Bs[j]) v = fd(rng);
    }
    const uint64_t wb = WB + (uint64_t)j * w_stride;
    put_weights(wb, w_pitch, wb + 4 * w_pitch, Ws[j], Ss[j], Bs[j], M, K);
  }
  for (auto& v : D) v = fd(rng);
  for (int m = 0; m < M; m++) put_f32(DD + 4 * m, D[m]);
  for (int b = 0; b < B; b++) {
    Xs[b].resize((size_t)N * K);
    xs[b].resize(N);
    C0[b].resize((size_t)N * M);
    for (int n = 0; n < N; n++) {
      xs[b][n] = std::ldexp(1.0f + 0.5f * fd(rng), -12);
      put_f32(XS + (uint64_t)b * f_stride + 4 * n, xs[b][n]);
      for (int k = 0; k < K; k++) {
        const int16_t v = (int16_t)qd(rng);
        Xs[b][(size_t)n * K + k] = v;
        put_i16(XA + (uint64_t)b * a_stride + (uint64_t)n * a_pitch + 2 * k, v);
      }
      for (int m = 0; m < M; m++) {
        C0[b][(size_t)n * M + m] = fd(rng);
        put_f32(YC + (uint64_t)b * c_stride + (uint64_t)n * c_pitch + 4 * m, C0[b][(size_t)n * M + m]);
      }
    }
  }
  const uint32_t fl = (bias ? F_BIAS : 0) | (acc ? F_ACC : 0) | (addd ? F_ADD_D : 0);
  Sl s[6] = {{XA, a_pitch, a_stride}, {WB, w_pitch, w_stride}, {YC, c_pitch, c_stride},
             {DD, 0, 0},              {WB + 4 * w_pitch, 0, e_stride}, {XS, 0, f_stride}};
  uint32_t p[7] = {u32(p0), (uint32_t)M, 0, 0, 0, 0, 0};
  store(LIST, 0, mk(OP_GEMM, fl, 0x77, M, K, N, B, G, s, p));
  store(LIST, 1, mk_simple(OP_END, 0x78));
  const uint32_t r = start(LIST, STAT, 0);
  CHECK(r == ((1u << 8) | ST_END), "gemm M%d K%d N%d B%d G%d: return 0x%x", M, K, N, B, G, r);
  int bad = 0;
  for (int b = 0; b < B; b++) {
    const int j = b / G;
    for (int n = 0; n < N; n++)
      for (int m = 0; m < M; m++) {
        int64_t a = 0;
        for (int k = 0; k < K; k++) a += (int64_t)Xs[b][(size_t)n * K + k] * Ws[j][(size_t)m * K + k];
        float y = (float)a * Ss[j][m] * xs[b][n] * p0 + (bias ? Bs[j][m] : 0.0f);
        if (addd) y += D[m];
        if (acc) y += C0[b][(size_t)n * M + m];
        const float got = get_f32(YC + (uint64_t)b * c_stride + (uint64_t)n * c_pitch + 4 * m);
        if (!(got == y) && bad++ < 5) printf("  b%d n%d m%d: got %.9g want %.9g\n", b, n, m, got, y);
      }
  }
  CHECK(bad == 0, "gemm M%d K%d N%d B%d G%d bias%d acc%d addd%d: %d mismatches", M, K, N, B, G, bias, acc, addd, bad);
  printf("gemm M=%d K=%d N=%d B=%d G=%d bias=%d acc=%d addd=%d p0=%g: %s\n", M, K, N, B, G, bias, acc, addd, p0,
         bad ? "MISMATCH" : "bit-exact");
}

// ---- 2. the walk
static void test_walk() {
  const uint64_t LIST = 0x2000, STAT = 0x1000, SRC = 0x20000, DST = 0x30000;
  // COPY stub (KIND 6, unary): dst word = src word ^ 0x07070707 per lane
  for (int i = 0; i < 64; i++) put_u32(SRC + 4 * i, 0x1000 + i);
  Sl s[6] = {{SRC, 64, 0}, {}, {DST, 64, 0}, {}, {}, {}};
  uint32_t p[7] = {};
  Rec cp = mk(OP_COPY, 0, 0x101, 0, 16, 4, 1, 1, s, p);
  store(LIST, 0, mk_simple(OP_NOP, 0x100));
  store(LIST, 1, cp);
  store(LIST, 2, mk_simple(OP_NOP, 0x102, F_YIELD));
  store(LIST, 3, mk_simple(OP_NOP, 0x103));
  store(LIST, 4, mk_simple(OP_END, 0x104));
  uint32_t r = start(LIST, STAT, 0);
  Status st = status_at(STAT);
  CHECK(r == ((3u << 8) | ST_YIELD), "yield return 0x%x", r);
  CHECK(st.code == ST_YIELD && st.next == 3 && st.tag == 0x102 && st.done == 3 && st.magic == STATUS_MAGIC &&
            st.lastop == OP_NOP && st.merr == 0,
        "yield status %u %u %x %u", st.code, st.next, st.tag, st.done);
  int bad = 0;
  for (int i = 0; i < 64; i++) bad += get_u32(DST + 4 * i) != ((0x1000u + i) ^ 0x07070707u);
  CHECK(bad == 0, "copy stub: %d words wrong", bad);
  const uint32_t starts0 = st.starts;
  r = start(LIST, STAT, 3);
  st = status_at(STAT);
  CHECK(r == ((4u << 8) | ST_END) && st.code == ST_END && st.next == 4 && st.tag == 0x104 && st.done == 1 &&
            st.starts == starts0 + 1,
        "resume: return 0x%x status %u %u %x %u starts %u", r, st.code, st.next, st.tag, st.done, st.starts);
  // single step
  r = start(LIST, STAT, 0, true);
  CHECK(r == ((1u << 8) | ST_STEP), "step 0: 0x%x", r);
  r = start(LIST, STAT, 1, true);
  CHECK(r == ((2u << 8) | ST_STEP), "step 1: 0x%x", r);
  r = start(LIST, STAT, 4, true);
  CHECK(r == ((4u << 8) | ST_END), "step at END: 0x%x", r);
  // bad magic
  put_u32(LIST + 128 * 3, get_u32(LIST + 128 * 3) ^ (0xFFu << 8));
  reseal(LIST, 3);
  r = start(LIST, STAT, 3);
  st = status_at(STAT);
  CHECK(r == ((3u << 8) | ST_BAD_MAGIC) && st.code == ST_BAD_MAGIC && st.next == 3 && st.done == 0,
        "bad magic: 0x%x", r);
  store(LIST, 3, mk_simple(OP_NOP, 0x103));
  // bad check: one flipped bit in a parameter
  put_u32(LIST + 128 * 3 + 4 * 27, get_u32(LIST + 128 * 3 + 4 * 27) ^ 0x10);
  r = start(LIST, STAT, 0);
  CHECK(r == ((3u << 8) | ST_BAD_CHECK), "bad check (after 3 good commands; F_YIELD at 2 stops first): 0x%x", r);
  r = start(LIST, STAT, 3);
  CHECK(r == ((3u << 8) | ST_BAD_CHECK), "bad check: 0x%x", r);
  store(LIST, 3, mk_simple(OP_NOP, 0x103));
  // bad opcode
  store(LIST, 3, mk_simple(0x99, 0x199));
  r = start(LIST, STAT, 3);
  st = status_at(STAT);
  CHECK(r == ((3u << 8) | ST_BAD_OPCODE) && st.tag == 0x199, "bad opcode: 0x%x tag %x", r, st.tag);
  // mode error: GEMM with M = 33 (not a multiple of 32) -> ME_DIMS; index stays on the failing command
  Sl z[6] = {};
  uint32_t pz[7] = {};
  store(LIST, 3, mk(OP_GEMM, 0, 0x203, 33, 64, 4, 1, 1, z, pz));
  r = start(LIST, STAT, 3);
  st = status_at(STAT);
  CHECK(r == ((3u << 8) | ST_MODE_ERROR) && st.merr == ME_DIMS && st.next == 3 && st.tag == 0x203 &&
            st.lastop == OP_GEMM && st.done == 1,
        "mode error dims: 0x%x merr %u", r, st.merr);
  // mode error: misaligned operand -> ME_ALIGN
  Sl al[6] = {{0x40008, 128, 0}, {}, {0x50000, 128, 0}, {}, {}, {}};
  store(LIST, 3, mk(OP_GEMM, 0, 0x303, 32, 64, 4, 1, 1, al, pz));
  r = start(LIST, STAT, 3);
  st = status_at(STAT);
  CHECK(r == ((3u << 8) | ST_MODE_ERROR) && st.merr == ME_ALIGN, "mode error align: 0x%x merr %u", r, st.merr);
  // no status block (N = 0): nothing written
  memset(bp(STAT), 0xEE, 32);
  store(LIST, 3, mk_simple(OP_NOP, 0x103));
  r = start(LIST, 0, 3);
  CHECK(r == ((4u << 8) | ST_END) && get_u32(STAT) == 0xEEEEEEEEu, "no status block: 0x%x", r);
  printf("walk: END / YIELD+resume / STEP / BAD_MAGIC / BAD_CHECK / BAD_OPCODE / MODE_ERROR(dims, align) / no status: %s\n",
         g_fail ? "see failures" : "ok");
}

// Legacy protocol: no magic in M -> one GEMM with the stub's legacy layout over separate port buffers.
static void test_legacy(std::mt19937& rng) {
  const int M = 64, K = 40, N = 6;  // N K 2 = 480: the token scales right after X stay 16-byte aligned
  std::uniform_int_distribution<int> qd(-32767, 32767);
  std::vector<Word> w[4], x(1024), y(1024);
  for (auto& v : w) v.assign(1024, zword());
  std::vector<int16_t> W((size_t)M * K), X((size_t)N * K);
  for (auto& v : W) v = (int16_t)qd(rng);
  for (auto& v : X) v = (int16_t)qd(rng);
  for (int g = 0; g < M / 32; g++)
    for (int k = 0; k < K; k++)
      for (int p = 0; p < 4; p++)
        for (int l = 0; l < 8; l++) {
          int16_t v = W[(size_t)(32 * g + 8 * p + l) * K + k];
          memcpy((char*)w[p].data() + ((size_t)g * K + k) * 16 + 2 * l, &v, 2);
        }
  const float sr = 1.0f / 1024, st = 1.0f / 4096;
  for (int m = 0; m < M; m++) memcpy((char*)w[0].data() + (size_t)(M / 32) * K * 16 + 4 * m, &sr, 4);
  memcpy((char*)x.data(), X.data(), X.size() * 2);
  for (int n = 0; n < N; n++) memcpy((char*)x.data() + N * K * 2 + 4 * n, &st, 4);
  const int r = seq_top(w[0].data(), w[1].data(), w[2].data(), w[3].data(), x.data(), y.data(), M, K, N);
  int bad = 0;
  for (int n = 0; n < N; n++)
    for (int m = 0; m < M; m++) {
      int64_t a = 0;
      for (int k = 0; k < K; k++) a += (int64_t)X[(size_t)n * K + k] * W[(size_t)m * K + k];
      const float want = (float)a * sr * st * 1.0f + 0.0f;
      float got;
      memcpy(&got, (char*)y.data() + ((size_t)n * M + m) * 4, 4);
      bad += !(got == want);
    }
  CHECK(r == 0 && bad == 0, "legacy: return %d, %d mismatches", r, bad);
  printf("legacy one-call protocol: return %d, %s\n", r, bad ? "MISMATCH" : "bit-exact");
}

// ---- 3. the -12392 chunk list, decoded by the C++ sequencer
static std::vector<std::string> g_lines;
static uint32_t g_tidx = 0;
static void trace(const Cmd& c) {
  char buf[512];
  int n = snprintf(buf, sizeof buf, "%u %02x %04x %08x %u %u %u %u %u", g_tidx++, c.op, c.flags, c.tag, c.M, c.K, c.N,
                   c.B, c.G);
  const Opnd* o[6] = {&c.a, &c.b, &c.c, &c.d, &c.e, &c.f};
  for (int s = 0; s < 6; s++)
    n += snprintf(buf + n, sizeof buf - n, " %llx/%x/%x", (unsigned long long)o[s]->addr, o[s]->pitch, o[s]->stride);
  const uint32_t p[7] = {c.p0, c.p1, c.p2, c.p3, c.p4, c.p5, c.p6};
  for (int i = 0; i < 7; i++) n += snprintf(buf + n, sizeof buf - n, " %08x", p[i]);
  g_lines.push_back(buf);
}

static void test_chunk(const std::string& dir) {
  std::ifstream fb(dir + "/chunk_list.bin", std::ios::binary);
  if (!fb) {
    printf("chunk list: %s/chunk_list.bin not found (run check_policy.py --dump), skipped\n", dir.c_str());
    return;
  }
  std::string blob((std::istreambuf_iterator<char>(fb)), std::istreambuf_iterator<char>());
  const uint32_t ncmd = (uint32_t)(blob.size() / 128);
  const uint64_t LIST = 0x100000, STAT = 0x80000;  // the list is position independent: relocate it
  memcpy(bp(LIST), blob.data(), blob.size());
  std::ifstream ft(dir + "/chunk_trace.txt");
  std::vector<std::string> want;
  for (std::string l; std::getline(ft, l);)
    if (!l.empty()) want.push_back(l);
  g_dry = true;
  g_trace = trace;
  g_lines.clear();
  g_tidx = 0;
  const uint32_t r = start(LIST, STAT, 0);
  g_dry = false;
  g_trace = nullptr;
  const Status st = status_at(STAT);
  CHECK(r == (((ncmd - 1) << 8) | ST_END), "chunk: return 0x%x (want next %u END)", r, ncmd - 1);
  CHECK(st.done == ncmd - 1, "chunk: done %u", st.done);
  size_t diff = 0;
  for (size_t i = 0; i < want.size() || i < g_lines.size(); i++) {
    const std::string a = i < g_lines.size() ? g_lines[i] : "<none>", b = i < want.size() ? want[i] : "<none>";
    if (a != b && diff++ < 3) printf("  line %zu:\n    C++    %s\n    python %s\n", i, a.c_str(), b.c_str());
  }
  CHECK(diff == 0 && g_lines.size() == want.size(), "chunk trace: %zu lines differ (%zu vs %zu)", diff, g_lines.size(),
        want.size());
  printf("chunk list: %u records (%zu bytes), %u dispatched, return 0x%x, decode trace vs python: %s\n", ncmd,
         blob.size(), st.done, r, diff ? "DIFFERENT" : "identical");
}

int main(int argc, char** argv) {
  g_mem.assign((size_t)1 << 20, zword());  // 16 MB
  std::mt19937 rng(12392);
  test_gemm(rng, 32, 8, 1, 1, 1, false, false, false, 1.0f);
  test_gemm(rng, 64, 48, 13, 1, 1, true, false, false, 1.0f);
  test_gemm(rng, 96, 64, 17, 3, 1, false, false, true, 0.125f);
  test_gemm(rng, 32, 64, 50, 6, 3, false, false, false, 0.125f);  // GQA: 6 heads over 2 KV heads
  test_gemm(rng, 64, 1024, 9, 2, 1, true, true, false, 31.0f);    // longest reduction, F_ACC
  test_walk();
  test_legacy(rng);
  test_chunk(argc > 1 ? argv[1] : "build/py");
  printf("%s (%d failures)\n", g_fail ? "FAILED" : "PASSED", g_fail);
  return g_fail ? 1 : 0;
}
