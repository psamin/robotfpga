// Vitis HLS csim testbench for policy_top: drives the real AXI-Stream interface.
// The argument is the artifacts directory (run_hls.tcl passes it).
#include <ap_axi_sdata.h>
#include <hls_stream.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <fstream>
#include <iterator>
#include <string>
#include <vector>

typedef ap_axiu<8, 0, 0, 0> axis8_t;
int policy_top(hls::stream<axis8_t>& in_stream, hls::stream<axis8_t>& out_stream, int mode,
               uint32_t shifts_lo, uint32_t shifts_hi, uint32_t* version);

static std::vector<uint8_t> slurp(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return {};
  return std::vector<uint8_t>(std::istreambuf_iterator<char>(f), {});
}

static void push(hls::stream<axis8_t>& s, const std::vector<uint8_t>& bytes) {
  for (size_t i = 0; i < bytes.size(); i++) {
    axis8_t beat;
    beat.data = bytes[i];
    beat.keep = -1;
    beat.strb = -1;
    beat.last = i + 1 == bytes.size();
    s.write(beat);
  }
}

int main(int argc, char** argv) {
  std::string dir = argc > 1 ? argv[1] : "../artifacts/standin";
  std::vector<uint8_t> raw = slurp(dir + "/manifest.json");
  std::string text(raw.begin(), raw.end());
  std::vector<int> s;
  for (size_t p = text.find("\"shift\""); p != std::string::npos; p = text.find("\"shift\"", p + 1))
    s.push_back(atoi(text.c_str() + text.find(':', p) + 1));
  if (s.size() != 8) {
    printf("FAIL: could not read 8 shifts from %s/manifest.json\n", dir.c_str());
    return 1;
  }
  uint32_t lo = s[0] | s[1] << 8 | s[2] << 16 | s[3] << 24;
  uint32_t hi = s[4] | s[5] << 8 | s[6] << 16 | s[7] << 24;

  hls::stream<axis8_t> in, out;
  uint32_t version = 0;
  push(in, slurp(dir + "/weights.bin"));
  int status = policy_top(in, out, 0, lo, hi, &version);
  if (status != 0 || !in.empty()) {
    printf("FAIL: weight load status %d\n", status);
    return 1;
  }

  int failures = 0, n = 0;
  for (int v = 0; v < 1000; v++) {
    char name[64];
    snprintf(name, sizeof name, "/vectors/%03d_in.bin", v);
    std::vector<uint8_t> packet = slurp(dir + name);
    if (packet.empty()) break;
    snprintf(name, sizeof name, "/vectors/%03d_out.bin", v);
    std::vector<uint8_t> want = slurp(dir + name);
    n++;

    push(in, packet);
    status = policy_top(in, out, 1, lo, hi, &version);
    bool ok = status == 0 && in.empty();
    for (size_t i = 0; i < want.size(); i++) {
      if (out.empty()) {
        ok = false;
        break;
      }
      axis8_t beat = out.read();
      ok &= (uint8_t)beat.data == want[i] && (bool)beat.last == (i + 1 == want.size());
    }
    ok &= out.empty();
    if (!ok) {
      failures++;
      printf("FAIL vector %03d (status %d)\n", v, status);
      while (!out.empty()) out.read();
    }
  }
  printf("%s: %d vectors through policy_top, %d failure(s), version 0x%08x\n", failures || !n ? "FAIL" : "PASS",
         n, failures, version);
  return failures || !n ? 1 : 0;
}
