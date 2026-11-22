// Host C model of the integrated float_top (float_seq.h) as a DLL for the end-to-end check (e2e_seq.py):
// one flat DDR image [base, base + bytes) owned by the caller; seq_start() is one float_top start with every
// pointer register 0 (list mode), exactly the hardware's register protocol.
//   g++ -std=c++14 -O2 -DFSTREAM_ARITH=16 -ffp-contract=off -shared -static -I<Vitis_HLS>/include
//       -I../kernels -I../kernels/quantize -o build/seq_shim.dll seq_shim.cpp
#include "float_seq.h"

namespace seq {
uint64_t g_host_cycles = 0;
uint64_t g_decode_checked = 0, g_decode_mismatch = 0;
}  // namespace seq
namespace nr {
long g_leftover = 0;  // streams left non-empty after a NORM call (norm_residual.h, host only)
}  // namespace nr

using fstream::IoWord;
using fstream::WPort;

static uint8_t* g_img = nullptr;
static uint64_t g_base = 0, g_bytes = 0;

extern "C" {
__declspec(dllexport) void seq_image(uint8_t* img, uint64_t base, uint64_t bytes) {
  g_img = img;
  g_base = base;
  g_bytes = bytes;
}
// one start: M = MODE_MAGIC << 24 | step << 23 | first, K = list >> 4, N = status >> 4; returns ap_return
__declspec(dllexport) uint32_t seq_start(uint32_t M, uint32_t K, uint32_t N) {
  IoWord* io = (IoWord*)g_img - (g_base >> 4);
  const WPort* hp = (const WPort*)g_img - (g_base >> 4);
  return seq::top<fseq::Units>(hp, hp, hp, hp, (const IoWord*)io, io, M, K, N);
}
__declspec(dllexport) uint64_t seq_decode_mismatches() { return seq::g_decode_mismatch; }
__declspec(dllexport) long seq_leftover() { return nr::g_leftover; }
// first differing byte of [addr, addr + n) between the image and ref (or n if equal)
__declspec(dllexport) uint64_t seq_cmp(uint64_t addr, const uint8_t* ref, uint64_t n) {
  const uint8_t* a = g_img + (addr - g_base);
  if (memcmp(a, ref, n) == 0) return n;
  for (uint64_t i = 0; i < n; i++)
    if (a[i] != ref[i]) return i;
  return n;
}
}
