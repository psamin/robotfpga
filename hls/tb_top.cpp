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

static void push(hls::stream<axis8_t>& s, const std::vector<uint8_t>& bytes, int last_at = -1) {
  for (size_t i = 0; i < bytes.size(); i++) {
    axis8_t beat;
    beat.data = bytes[i];
    beat.keep = -1;
    beat.strb = -1;
    beat.last = last_at < 0 ? i + 1 == bytes.size() : i == (size_t)last_at;
    s.write(beat);
  }
}

static bool check_stream(hls::stream<axis8_t>& s, const std::vector<uint8_t>& want) {
  bool ok = true;
  for (size_t i = 0; i < want.size(); i++) {
    if (s.empty()) return false;
    axis8_t beat = s.read();
    ok &= (uint8_t)beat.data == want[i] && (bool)beat.last == (i + 1 == want.size());
    ok &= (bool)beat.keep && (bool)beat.strb;
  }
  ok &= s.empty();
  while (!s.empty()) s.read();
  return ok;
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
  for (int shift : s) if (shift < 1 || shift > 30) return 1;
  uint32_t lo = s[0] | s[1] << 8 | s[2] << 16 | s[3] << 24;
  uint32_t hi = s[4] | s[5] << 8 | s[6] << 16 | s[7] << 24;

  hls::stream<axis8_t> in, out;
  uint32_t version = 0;
  std::vector<uint8_t> weights = slurp(dir + "/weights.bin");
  if (weights.size() != 568240) {
    printf("FAIL: weights.bin must contain 568240 bytes\n");
    return 1;
  }
  push(in, weights);
  int status = policy_top(in, out, 0, lo, hi, &version);
  if (status != 0 || !in.empty() || !out.empty() || version != 0x00000100) {
    printf("FAIL: weight load status %d\n", status);
    return 1;
  }

  int failures = 0, n = 0;
  for (int v = 0; v < 1000; v++) {
    char name[64];
    snprintf(name, sizeof name, "/vectors/%03d_in.bin", v);
    std::vector<uint8_t> packet = slurp(dir + name);
    if (packet.empty()) {
      std::ifstream existing(dir + name, std::ios::binary);
      if (existing) {
        printf("FAIL vector %03d: empty input file\n", v);
        return 1;
      }
      break;
    }
    snprintf(name, sizeof name, "/vectors/%03d_out.bin", v);
    std::vector<uint8_t> want = slurp(dir + name);
    if (packet.size() != 27658 || want.size() != 48) {
      printf("FAIL vector %03d: incorrect input/output byte count\n", v);
      return 1;
    }
    n++;

    push(in, packet);
    status = policy_top(in, out, 1, lo, hi, &version);
    bool ok = status == 0 && in.empty() && version == 0x00000100;
    ok &= check_stream(out, want);
    if (!ok) {
      failures++;
      printf("FAIL vector %03d (status %d)\n", v, status);
      while (!out.empty()) out.read();
    }
  }
  if (!n) return 1;

  // Use actual golden vector 000 to check errors followed by valid recovery.
  const auto packet = slurp(dir + "/vectors/000_in.bin");
  const auto want = slurp(dir + "/vectors/000_out.bin");
  const std::vector<uint8_t> zeros(48, 0);
  for (int test = 0; test < 7; test++) {
    bool ok = true;
    if (test == 0) {
      push(in, packet);
      status = policy_top(in, out, 3, lo, hi, &version);
      ok = status == 3 && out.empty();
      ok &= check_stream(in, packet);  // Bad mode must not consume the stream.
    } else if (test <= 4) {
      const uint32_t bad_lo = (lo & ~0xffu) | (test == 2 ? 31u : 0u);
      const int last_at = test == 3 ? 0 : test == 4 ? (int)packet.size() : -1;
      push(in, packet, last_at);
      status = policy_top(in, out, 1, test <= 2 ? bad_lo : lo, hi, &version);
      ok = status == (test <= 2 ? 1 : 2) && in.empty();
      ok &= check_stream(out, test <= 2 ? zeros : want);
    } else {
      push(in, weights, test == 5 ? 0 : (int)weights.size());
      status = policy_top(in, out, 0, lo, hi, &version);
      ok = status == 2 && in.empty() && out.empty();
      // Restore a valid weight transaction before testing inference recovery.
      push(in, weights);
      status = policy_top(in, out, 0, lo, hi, &version);
      ok &= status == 0 && in.empty() && out.empty();
    }
    ok &= version == 0x00000100;
    if (!ok) {
      printf("FAIL error-path case %d\n", test);
      failures++;
    }
    while (!in.empty()) in.read();
    while (!out.empty()) out.read();
    push(in, packet);
    status = policy_top(in, out, 1, lo, hi, &version);
    ok = status == 0 && in.empty() && version == 0x00000100;
    ok &= check_stream(out, want);
    if (!ok) {
      printf("FAIL recovery after case %d\n", test);
      failures++;
    }
  }
  printf("Checked 7 stream error paths and 7 valid recoveries\n");
  printf("%s: %d vectors through policy_top, %d failure(s), version 0x%08x\n", failures || !n ? "FAIL" : "PASS",
         n, failures, version);
  return failures || !n ? 1 : 0;
}
