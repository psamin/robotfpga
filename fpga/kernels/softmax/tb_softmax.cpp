// Host testbench of the SOFTMAX unit: runs the HLS source itself (softmax_top.cpp + softmax_core.h, g++ against
// the Vitis HLS headers) on the DDR images written by make_vectors.py and compares the whole image after the call,
// word for word, with the image the golden executor (smx_emu.run_cmd) predicts. Every byte not written by the
// unit holds 0xA5, so stray or missing writes fail too. One flat array serves as every port; it sits at the
// case's physical base address (pointer offset), so the 64-bit address path is exercised above 4 GB.
//
//   tb_softmax <case dir>...    params.txt: "<base hex> <record byte offset> <image bytes> <expected return>
//                               <direct>"; direct = 1 calls decode + validate_hw + softmax_mode without the top
//                               (for a list base above 64 GB, which the K register cannot hold)
//   tb_softmax --cparams        the prologue's integer c = fl32(p0 * fl32(log2 e)) against the host FPU
//   tb_softmax --divider        the divider on every row sum S in [0, 2^23) against the host's 1.0f / S
// Build: build.sh (sequential processes, unbounded FIFOs) and with -DSMX_DFSIM -Itb_dfsim (threads on FIFOs
// bounded at the pragma depths: the DATAFLOW deadlock model).
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <chrono>
#include <random>
#include <string>
#include <vector>

#include "softmax_top.cpp"

using smx::Word;

static std::vector<Word> load_words(const std::string& path, size_t nbytes) {
  std::vector<Word> v(nbytes / 16, 0);
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) {
    fprintf(stderr, "cannot open %s\n", path.c_str());
    exit(2);
  }
  std::vector<uint32_t> buf(nbytes / 4);
  if (fread(buf.data(), 1, nbytes, f) != nbytes) {
    fprintf(stderr, "short read %s\n", path.c_str());
    exit(2);
  }
  fclose(f);
  for (size_t i = 0; i < v.size(); i++) {
    Word w = 0;
    for (int j = 0; j < 4; j++) w.range(32 * j + 31, 32 * j) = buf[4 * i + j];
    v[i] = w;
  }
  return v;
}

static int run_case(const std::string& dir) {
  FILE* f = fopen((dir + "/params.txt").c_str(), "r");
  if (!f) {
    fprintf(stderr, "cannot open %s/params.txt\n", dir.c_str());
    return 2;
  }
  unsigned long long base, rec_off, nbytes;
  unsigned want;
  int direct;
  if (fscanf(f, "%llx %llu %llu %x %d", &base, &rec_off, &nbytes, &want, &direct) != 5) {
    fprintf(stderr, "bad params in %s\n", dir.c_str());
    return 2;
  }
  fclose(f);
  std::vector<Word> mem = load_words(dir + "/mem.bin", (size_t)nbytes);
  std::vector<Word> expect = load_words(dir + "/expect.bin", (size_t)nbytes);
  Word* p = mem.data() - (ptrdiff_t)(base >> 4);  // word index = physical byte address >> 4

  auto t0 = std::chrono::steady_clock::now();
  unsigned got;
  if (!direct) {
    const int M = (int)((ctr::MODE_MAGIC << 24) | 0u);
    const int K = (int)(uint32_t)((base + rec_off) >> 4);
    got = (unsigned)softmax_top(p, p, p, p, p, p, M, K, 0);
  } else {
    uint32_t fl[ctr::REC_FIELDS];
    const Word* r = mem.data() + rec_off / 16;
    for (uint32_t j = 0; j < ctr::REC_FIELDS; j++) fl[j] = (uint32_t)r[j / 4].range(32 * (j % 4) + 31, 32 * (j % 4));
    ctr::Cmd c;
    got = ctr::decode(fl, c);
    if (!got) {
      uint32_t me = ctr::validate_hw(c);
      if (!me) me = smx::softmax_mode(c, p, p);
      got = me ? ((me << 16) | ctr::ST_MODE_ERROR) : ctr::ST_END;
    }
  }
  const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

  long long bad = 0;
  for (size_t i = 0; i < mem.size(); i++) {
    if (mem[i] != expect[i]) {
      if (bad < 4)
        printf("  mismatch at 0x%llx: got %s want %s\n", (unsigned long long)(base + 16 * i),
               mem[i].to_string(16).c_str(), expect[i].to_string(16).c_str());
      bad++;
    }
  }
  const bool ok = bad == 0 && got == want;
  const char* name = strrchr(dir.c_str(), '/');
  printf("%-26s ret 0x%05x (want 0x%05x), %8zu words, %lld mismatches (%.1f ms) %s\n", name ? name + 1 : dir.c_str(),
         got, want, mem.size(), bad, ms, ok ? "PASS" : "FAIL");
  fflush(stdout);
  return ok ? 0 : 1;
}

// The prologue's integer RNE product against the host's IEEE single multiply (SSE): Mc and ec of c.
static int test_cparams() {
  std::mt19937_64 rng(12392);
  long long n = 0, bad = 0;
  auto one = [&](uint32_t pb) {
    float p0 = ctr::u2f(pb);
    if (!(p0 > 0.0f && p0 < 1e30f)) return;  // validate's range
    const float pc = p0 < ldexpf(1.0f, -126) ? ldexpf(1.0f, -126) : p0;
    volatile float a = pc, l2 = ctr::u2f(0x3FB8AA3B);
    const float cf = a * l2;
    const uint32_t cb = ctr::f2u(cf);
    // reference (Mc, ec) from the host product, as smx_emu.c_params
    const uint32_t m24 = (1u << 23) | (cb & 0x7FFFFF);
    uint32_t mc = (m24 + 64) >> 7;
    int ec = (int)((cb >> 23) & 0xFF) - 127;
    if (mc >> 17) {
      mc >>= 1;
      ec++;
    }
    ap_uint<17> hm;
    ap_int<10> he;
    smx::c_params(pb, hm, he);
    n++;
    if ((uint32_t)hm != mc || (int)he != ec) {
      if (bad < 5) printf("  c_params 0x%08x: hw (%u, %d) host (%u, %d)\n", pb, (unsigned)hm, (int)he, mc, ec);
      bad++;
    }
  };
  for (uint32_t e = 0; e < 255; e++)
    for (int i = 0; i < 8192; i++) one((e << 23) | (uint32_t)(rng() & 0x7FFFFF));
  for (uint32_t m = 0; m < (1u << 23); m += 7) one((127u << 23) | m);  // every 7th significand of [1, 2)
  for (uint32_t pb : {0x00000001u, 0x007FFFFFu, 0x00800000u, 0x3E000000u, 0x3F800000u, 0x7149F2C9u, 0x7149F2CAu})
    one(pb);
  printf("c_params: %lld p0 patterns, %lld mismatches %s\n", n, bad, bad ? "FAIL" : "PASS");
  return bad ? 1 : 0;
}

// The divider process on every row sum the unit can produce, S in [0, 2^23) (S <= 256 * 32767 < 2^23), against
// the host's correctly rounded single division 1.0f / S (S is exact in FP32); S = 0 must give 1.0.
static int test_divider() {
  const uint32_t per = 65536, items = (1u << 23) / per;
  smx::Params p;
  p.B = items;
  p.NP4 = per;
  smx::SStream sq("sq");
  smx::RStream rq("rq");
  for (uint32_t s = 0; s < (1u << 23); s++) sq.write(s);
  smx::divider(sq, rq, p);
  long long bad = 0;
  smx::Word w = 0;
  for (uint32_t s = 0; s < (1u << 23); s++) {
    if (s % 4 == 0) w = rq.read();  // the divider packs 4 rows' scales per word, row n in lane n % 4
    const uint32_t got = (uint32_t)w.range(32 * (s % 4) + 31, 32 * (s % 4));
    const uint32_t want = s ? ctr::f2u(1.0f / (float)s) : ctr::ONE_BITS;
    if (got != want) {
      if (bad < 5) printf("  divider S=%u: 0x%08x want 0x%08x\n", s, got, want);
      bad++;
    }
  }
  printf("divider: %u row sums, %lld mismatches %s\n", 1u << 23, bad, bad ? "FAIL" : "PASS");
  return bad ? 1 : 0;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s --cparams | <case dir>...\n", argv[0]);
    return 2;
  }
  if (!strcmp(argv[1], "--cparams")) return test_cparams();
#ifndef SMX_DFSIM
  if (!strcmp(argv[1], "--divider")) return test_divider();
#endif
  int fails = 0;
  for (int i = 1; i < argc; i++) fails += run_case(argv[i]) != 0;
#ifdef SMX_DFSIM
  const char* mode = "threads on bounded FIFOs (DATAFLOW deadlock model)";
#else
  const char* mode = "sequential processes";
#endif
  printf("%s: %d of %d cases passed\n", mode, argc - 1 - fails, argc - 1);
  return fails ? 1 : 0;
}
