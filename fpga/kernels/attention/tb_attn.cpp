// Host testbench of the attention data path: attn_pack.h (the HLS core, compiled with g++) + the int16 engine
// (attn_engine_model.h, and with -DATTN_REAL_ENGINE also the engine's own C++ core fpga/hls/int16_linear.h) on DDR
// images that make_vectors.py writes from real SmolVLA -12392 attention calls. Every operand (int16 Q / K / V^T /
// o_proj input with their scales, in the engine's xmem / port layouts) and every engine output (scores, attention
// output) is compared bit for bit with the torch emulation (attn_int16.py).
//
//   python fpga/kernels/attention/make_vectors.py
//   export PATH=/c/Xilinx/Vitis_HLS/2022.2/tps/win64/msys64/mingw64/bin:/c/Xilinx/Vitis_HLS/2022.2/tps/win64/msys64/usr/bin:/c/Windows/System32:/c/Windows
//   g++ -O2 -std=c++14 -ffp-contract=off -I/c/Xilinx/Vitis_HLS/2022.2/include [-DATTN_REAL_ENGINE] -o tb_attn.exe tb_attn.cpp
//   ./tb_attn.exe vectors/vision vectors/prefix vectors/expert_self vectors/expert_cross
//
// script.txt, one command per line, run in order:
//   pack <desc_off> <n_desc>                    attn_pack(mem, mem, desc_off, n_desc)
//   engine M K Nreg w0 w1 w2 w3 xmem ymem       one head-batched engine call (Nreg = n | (H - 1) << 16)
//   check <name> <off> <n_words> <file> <f32>   compare mem[off .. off + n) with file (f32 = 1: also max |diff|)
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "attn_engine_model.h"
#include "attn_pack.h"
#ifdef ATTN_REAL_ENGINE
#include "../../hls/int16_linear.h"
#endif

using attnpack::Word;

static std::vector<uint32_t> load_u32(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) {
    fprintf(stderr, "cannot open %s\n", path.c_str());
    exit(2);
  }
  const size_t n = (size_t)f.tellg();
  std::vector<uint32_t> v(n / 4);
  f.seekg(0);
  f.read(reinterpret_cast<char*>(v.data()), (std::streamsize)(v.size() * 4));
  return v;
}

static void to_words(const std::vector<uint32_t>& m, std::vector<Word>& w) {
  w.resize(m.size() / 4);
  for (size_t i = 0; i < w.size(); i++)
    for (int j = 0; j < 4; j++) w[i].range(32 * j + 31, 32 * j) = m[4 * i + j];
}
static void from_words(const std::vector<Word>& w, std::vector<uint32_t>& m) {
  for (size_t i = 0; i < w.size(); i++)
    for (int j = 0; j < 4; j++) m[4 * i + j] = (uint32_t)w[i].range(32 * j + 31, 32 * j);
}

#ifdef ATTN_REAL_ENGINE
// the engine's own core, once per head (what float_top's head loop would do), on copies of the operands
static int real_engine(std::vector<uint32_t>& mem, const attneng::Call& c) {
  using fstream::IoWord;
  using fstream::WPort;
  const uint32_t n = attneng::n_of(c.Nreg), H = attneng::h_of(c.Nreg), np = attneng::np_of(n);
  const uint32_t WS = attneng::ws_of(c), XS = attneng::xs_of(c), YS = attneng::ys_of(c);
  for (uint32_t h = 0; h < H; h++) {
    std::vector<WPort> w[4];
    for (int p = 0; p < 4; p++) {
      w[p].resize(WS);
      memcpy(w[p].data(), &mem[4 * ((size_t)c.w[p] + (size_t)h * WS)], (size_t)WS * 16);
    }
    std::vector<IoWord> x(XS), y(YS);
    memcpy(x.data(), &mem[4 * ((size_t)c.x + (size_t)h * XS)], (size_t)XS * 16);
    const int st = fstream16::linear_int16_core(w[0].data(), w[1].data(), w[2].data(), w[3].data(), x.data(), y.data(),
                                                (int)c.M, (int)c.K, (int)n, (int)(c.M / 4), (int)((c.M + np) / 4));
    if (st) return st;
    memcpy(&mem[4 * ((size_t)c.y + (size_t)h * YS)], y.data(), (size_t)YS * 16);
  }
  return 0;
}
#endif

static int run_case(const std::string& dir) {
  std::vector<uint32_t> mem = load_u32(dir + "/mem.bin");
  std::ifstream sc(dir + "/script.txt");
  std::string line;
  int fails = 0, checks = 0;
  printf("== %s (%zu words of DDR image)\n", dir.c_str(), mem.size() / 4);
  while (std::getline(sc, line)) {
    std::istringstream is(line);
    std::string cmd;
    is >> cmd;
    if (cmd == "pack") {
      uint32_t off, n;
      is >> off >> n;
      std::vector<Word> w;
      to_words(mem, w);
      const auto t0 = std::chrono::steady_clock::now();
      const int st = attnpack::attn_pack(w.data(), w.data(), off, n);
      const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      from_words(w, mem);
      printf("  pack desc %u x %u: status %d (%.1f ms host)\n", off, n, st, ms);
      if (st) fails++;
    } else if (cmd == "engine") {
      attneng::Call c;
      is >> c.M >> c.K >> c.Nreg >> c.w[0] >> c.w[1] >> c.w[2] >> c.w[3] >> c.x >> c.y;
#ifdef ATTN_REAL_ENGINE
      const int st = real_engine(mem, c);
      const char* who = "int16_linear.h core";
#else
      const int st = attneng::engine_call(mem, c);
      const char* who = "C model";
#endif
      printf("  engine (%s) M %u K %u N %u H %u: status %d\n", who, c.M, c.K, attneng::n_of(c.Nreg),
             attneng::h_of(c.Nreg), st);
      if (st) fails++;
    } else if (cmd == "check") {
      std::string name, file;
      uint32_t off, n;
      int f32;
      is >> name >> off >> n >> file >> f32;
      const std::vector<uint32_t> ref = load_u32(dir + "/" + file);
      checks++;
      if (ref.size() != 4 * (size_t)n) {
        printf("  check %s: expected %u words, file has %zu\n", name.c_str(), n, ref.size() / 4);
        fails++;
        continue;
      }
      size_t bad = 0;
      double maxd = 0.0, maxr = 0.0;
      for (size_t i = 0; i < 4 * (size_t)n; i++) {
        const uint32_t a = mem[4 * (size_t)off + i];
        if (a != ref[i]) bad++;
        if (f32) {
          maxd = std::fmax(maxd, std::fabs((double)attneng::u2f(a) - (double)attneng::u2f(ref[i])));
          maxr = std::fmax(maxr, std::fabs((double)attneng::u2f(ref[i])));
        }
      }
      if (bad) fails++;
      if (f32)
        printf("  check %-6s %7u words: %s (%zu values differ, max |diff| %.3g of max |ref| %.3g)\n", name.c_str(), n,
               bad ? "FAIL" : "bit-exact", bad, maxd, maxr);
      else
        printf("  check %-6s %7u words: %s (%zu values differ)\n", name.c_str(), n, bad ? "FAIL" : "bit-exact", bad);
    }
  }
  printf("  %d checks, %d failures\n", checks, fails);
  return fails;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: tb_attn <case dir>...\n");
    return 2;
  }
  int fails = 0;
  for (int i = 1; i < argc; i++) fails += run_case(argv[i]);
  printf(fails ? "FAILED (%d)\n" : "ALL PASS\n", fails);
  return fails ? 1 : 0;
}
