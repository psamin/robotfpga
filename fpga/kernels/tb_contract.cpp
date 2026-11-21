// Host testbench of contract.h against contract.py and the HLS quantizer (CONTRACT.md 7, clauses C2 and C3).
//   .venv/Scripts/python.exe fpga/kernels/contract.py --vectors <dir>      (writes cases / quant / epilogue)
//   g++ -std=c++14 -O2 -ffp-contract=off -Wall -I<Vitis_HLS>/include -o tb_contract tb_contract.cpp
//   tb_contract <dir> [list.bin ...]
// 1. every record of cases.bin: decode status, validate() and validate_hw() codes equal to contract.py's;
// 2. every quant row: ctr::ref::quant_row == contract.py quant_rows == qz:: (quantize/quant_int16.h, the HLS
//    datapath) bit for bit, q and stored s;
// 3. the GEMM epilogue reference == contract.py gemm_out;
// 4. each extra list file (contract_plan.py output): decoded and validated by the C++ rules, all ME_OK.
#define __USE_MINGW_ANSI_STDIO 1  // %zu with MinGW
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <string>
#include <vector>

#include "contract.h"
#include "quantize/quant_int16.h"

static std::vector<unsigned char> slurp(const std::string& p) {
  std::vector<unsigned char> v;
  FILE* f = fopen(p.c_str(), "rb");
  if (!f) return v;
  fseek(f, 0, SEEK_END);
  v.resize((size_t)ftell(f));
  fseek(f, 0, SEEK_SET);
  if (!v.empty() && fread(v.data(), 1, v.size(), f) != v.size()) v.clear();
  fclose(f);
  return v;
}

static const char* st_name(uint32_t s) {
  switch (s) {
    case ctr::ST_END: return "END";
    case ctr::ST_BAD_MAGIC: return "BAD_MAGIC";
    case ctr::ST_BAD_CHECK: return "BAD_CHECK";
    case ctr::ST_BAD_VERSION: return "BAD_VERSION";
    case ctr::ST_BAD_OPCODE: return "BAD_OPCODE";
    default: return "?";
  }
}
static const char* me_name(uint32_t m) {
  static const char* n[] = {"OK", "DIMS", "ALIGN", "ADDR", "FLAGS", "PARAM", "SLOT"};
  return m < 7 ? n[m] : "?";
}

// cases.json is a flat list of {"name": ..., "status": ..., "me": ...}: pull the strings in order
static std::vector<std::string> json_field(const std::string& txt, const char* key) {
  std::vector<std::string> out;
  const std::string k = std::string("\"") + key + "\": \"";
  for (size_t p = txt.find(k); p != std::string::npos; p = txt.find(k, p + 1)) {
    const size_t a = p + k.size(), b = txt.find('"', a);
    out.push_back(txt.substr(a, b - a));
  }
  return out;
}

static int test_cases(const std::string& dir) {
  const std::vector<unsigned char> bin = slurp(dir + "/cases.bin"), js = slurp(dir + "/cases.json");
  if (bin.empty() || js.empty()) {
    printf("FAIL: no %s/cases.bin / cases.json\n", dir.c_str());
    return 1;
  }
  const std::string txt(js.begin(), js.end());
  const auto names = json_field(txt, "name"), sts = json_field(txt, "status"), mes = json_field(txt, "me"),
             hws = json_field(txt, "hw");
  const size_t nrec = bin.size() / ctr::REC_BYTES;
  int fails = 0;
  if (names.size() != nrec) {
    printf("FAIL: %zu records, %zu names\n", nrec, names.size());
    return 1;
  }
  for (size_t i = 0; i < nrec; i++) {
    uint32_t f[ctr::REC_FIELDS];
    memcpy(f, &bin[i * ctr::REC_BYTES], ctr::REC_BYTES);
    ctr::Cmd c;
    memset(&c, 0, sizeof c);
    const uint32_t st = ctr::decode(f, c);
    bool ok = sts[i] == st_name(st);
    std::string got = st_name(st);
    if (ok && st == ctr::ST_END) {
      const uint32_t me = ctr::validate(c), hw = ctr::validate_hw(c);
      got += std::string("/") + me_name(me) + "/hw " + me_name(hw);
      ok = mes[i] == me_name(me) && hws[i] == me_name(hw);
      uint32_t g[ctr::REC_FIELDS];
      ctr::encode(c, g);  // re-encode: identical bytes (B 0 -> 1 aside, never used by the vectors)
      if (memcmp(f, g, sizeof f)) {
        ok = false;
        got += " (re-encode differs)";
      }
    }
    if (!ok) {
      printf("FAIL case %-24s got %s, expected %s/%s\n", names[i].c_str(), got.c_str(), sts[i].c_str(), mes[i].c_str());
      fails++;
    }
  }
  printf("records: %zu cases, %d failures\n", nrec, fails);
  return fails;
}

static int test_quant(const std::string& dir) {
  const std::vector<unsigned char> b = slurp(dir + "/quant.bin");
  size_t p = 0;
  int rows = 0, fails = 0;
  long elems = 0;
  while (p + 8 <= b.size()) {
    uint32_t n;
    int32_t sh;
    memcpy(&n, &b[p], 4);
    memcpy(&sh, &b[p + 4], 4);
    p += 8;
    std::vector<float> x(n), sref(1);
    std::vector<int16_t> qref(n), q(n);
    memcpy(x.data(), &b[p], 4 * n);
    p += 4 * n;
    memcpy(qref.data(), &b[p], 2 * n);
    p += 2 * n;
    memcpy(sref.data(), &b[p], 4);
    p += 4;
    // C++ reference
    const float s = ctr::ref::quant_row(x.data(), n, q.data(), sh);
    bool ok = ctr::f2u(s) == ctr::f2u(sref[0]) && q == qref;
    // HLS datapath: absmax bits -> scale_bits / row_scale -> quant_elem
    qz::abits_t a = 0;
    for (uint32_t j = 0; j < n; j++) a = qz::amax2(a, qz::abs_bits(qz::fbits_t(ctr::f2u(x[j]))));
    const qz::RowScale rs = qz::row_scale(a);
    const uint32_t sb = (uint32_t)rs.s_bits;
    const float s_hls = a == 0 ? 1.0f : ldexpf(ctr::u2f(sb), sh);
    if (ctr::f2u(s_hls) != ctr::f2u(sref[0])) ok = false;
    for (uint32_t j = 0; j < n; j++)
      if ((int16_t)qz::quant_elem(qz::fbits_t(ctr::f2u(x[j])), rs) != qref[j]) ok = false;
    if (!ok) {
      printf("FAIL quant row %d (n %u, shift %d): s ref %08x py %08x hls %08x\n", rows, n, sh, ctr::f2u(s),
             ctr::f2u(sref[0]), ctr::f2u(s_hls));
      fails++;
    }
    rows++;
    elems += n;
  }
  printf("quantizer: %d rows, %ld values: ref == contract.py == quant_int16.h, %d failures\n", rows, elems, fails);
  return rows ? fails : 1;
}

static int test_epilogue(const std::string& dir) {
  const std::vector<unsigned char> b = slurp(dir + "/epilogue.bin");
  const int N = 8, M = 16;
  if (b.size() != (size_t)(8 * N * M + 4 * M + 4 * N + 4 * M + 4 * N * M + 4 * N * M)) {
    printf("FAIL: epilogue.bin size %zu\n", b.size());
    return 1;
  }
  const unsigned char* p = b.data();
  const int64_t* acc = (const int64_t*)p;
  const float* sr = (const float*)(p + 8 * N * M);
  const float* st = sr + M;
  const float* bias = st + N;
  const float* add = bias + M;
  const float* y = add + N * M;
  int fails = 0;
  for (int n = 0; n < N; n++)
    for (int m = 0; m < M; m++) {
      const float r = ctr::ref::gemm_out(acc[n * M + m], sr[m], st[n], &bias[m], &add[n * M + m]);
      if (ctr::f2u(r) != ctr::f2u(y[n * M + m])) fails++;
    }
  printf("epilogue: %d values, ref == contract.py gemm_out, %d failures\n", N * M, fails);
  return fails;
}

static int test_list(const std::string& path) {
  const std::vector<unsigned char> bin = slurp(path);
  const size_t n = bin.size() / ctr::REC_BYTES;
  int fails = 0, ops[256] = {0};
  for (size_t i = 0; i < n; i++) {
    uint32_t f[ctr::REC_FIELDS];
    memcpy(f, &bin[i * ctr::REC_BYTES], ctr::REC_BYTES);
    ctr::Cmd c;
    const uint32_t st = ctr::decode(f, c);
    const uint32_t me = st == ctr::ST_END ? ctr::validate(c) | ctr::validate_hw(c) : 0;
    if (st != ctr::ST_END || me != ctr::ME_OK) {
      if (fails < 10) printf("FAIL list %s record %zu: %s / %s\n", path.c_str(), i, st_name(st), me_name(me));
      fails++;
    } else {
      ops[c.op]++;
    }
  }
  const bool ends = n && ((bin[(n - 1) * ctr::REC_BYTES] & 0xFF) == ctr::OP_END);
  printf("list %s: %zu records, %d failures, ends with END: %s\n", path.c_str(), n, fails, ends ? "yes" : "NO");
  return fails + (ends ? 0 : 1) + (n ? 0 : 1);
}

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: tb_contract <vector dir> [list.bin ...]\n");
    return 2;
  }
  int fails = test_cases(argv[1]) + test_quant(argv[1]) + test_epilogue(argv[1]);
  for (int i = 2; i < argc; i++) fails += test_list(argv[i]);
  printf(fails ? "FAILED (%d)\n" : "ALL PASS\n", fails);
  return fails ? 1 : 0;
}
