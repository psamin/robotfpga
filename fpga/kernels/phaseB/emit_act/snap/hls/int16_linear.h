// int16 linear engine for SmolVLA on the KR260: the -DFSTREAM_ARITH=16 variant of float_top.
//
// Y[n][m] = (float(sum_k Xq[n][k] * Wq[m][k]) * s_row[m]) * s_tok[n]      (FP32, round to nearest even)
//   Wq: int16 weights, per-output-row scale s_row[m] = rowmax|w| / 32767 (pack_float.quantize_int16_rows)
//   Xq: int16 activations, per-token scale s_tok[n] = tokmax|x| / 32767 (pack_float.quantize_int16_tokens)
//   Wq and Xq never hold -32768: every producer clamps to +-32767 (review R7). The 43-bit block capture below is
//   exact only then (|block sum| <= 4096 * 32767^2 < 2^42); a -32768 x -32768 block of K = 4096 reaches +2^42
//   and flips sign. tb_gemm.cpp asserts it on every operand it generates.
//   the sum is exact (integer); the dequantization is two FP32 multiplies (the Xilinx core: denormals flush to
//   zero); bias stays on the host.
//
// Same bus interfaces and s_axi_control register map as the FP32 engine (float_top.cpp; the generated
// xfloat_top_hw.h is identical): four 128-bit weight bundles hp0..hp3 in today's packing (pack_float.pack of the
// int16 bit patterns), X and Y over io. The scales travel in the X buffer: no register is added.
//
// Architecture (fpga/hls/model_int16.py models it per call):
//  - 32 tokens ("a block") x 32 output lanes = 1,024 int16 MACs per clock, one DSP48E2 each. The loop runs
//    (tile, group, block, k) with k innermost, so every (token, lane) accumulator serves exactly one block of K
//    products at a time: one accumulator per MAC, no partial-sum RAM, no fabric adder.
//  - A tile is NBX blocks (NBX = 2: 64 tokens). The weights of a group arrive once per tile over the four HP
//    ports (one 128-bit word per port per k) and are kept on chip (wbuf, URAM) while block 1 re-uses them, so
//    the weight stream runs at 1 / NBX of the MAC rate (8 GB/s at 250 MHz for full tiles, against ~14 GB/s of
//    DDR). The weight FIFOs hold a whole group (K_MAX words, URAM), so the next group streams in during the
//    re-use. NBX = 1 is the same engine without re-use (weight stream = MAC rate, DDR-bound).
//  - Accumulators are never zeroed. A block's sums are captured into a shadow register file (1,024 x 43 bit)
//    two clocks after the block's last MAC decision, from the accumulator registers (review R1: the MAC operands
//    sit in registers, the product in the DSP's M register, the accumulator is the register after the post-adder,
//    i.e. the P register, and the capture reads that register, never the adder output), and the dequantizer takes the
//    difference to the previous capture of the same accumulator. Two's-complement wrap makes that exact: one
//    block's sum is below K_MAX * 32767^2 < 2^42 in magnitude, so 43 low bits of the running 48-bit sums carry
//    it. The loop body stays a pure acc += x * w (no `first ? 0 : acc` mux) and one flat pipelined loop runs a
//    whole call without a drain per block.
//  - The loop has no blocking FIFO access, so it never stalls (no stall enable fanning out to the ~93k
//    accumulator/shadow flip-flops); bubbles multiply by zero. The MAC side, the emit side and the X fill side
//    each decide from their own registered state; the MAC side sees the other two through short delay lines.
//  - The shadow empties one 16-value word per clock into the emit FIFO (the MAC side sees it free EDONE = 6 clocks
//    after its last word) while the next block computes. The dequantizer takes 4 values per clock (review R2 /
//    CONTRACT E-1: 43-bit integer -> FP32 (i43_to_f32), * s_row, * s_tok), the rate of the 128-bit io store, so a
//    block of 1,024 outputs takes 256 clocks and calls with K < 256 are bound by the Y store (no SmolVLA linear
//    is; attention's Q.K^T with K = 64 is).
//  - X is double-buffered (URAM, 4 int16 per 64-bit word, 8 banks, 32 URAMs at NBX = 2): the next tile streams
//    in over io while the current one computes; the first tile of a call (4 * K * blocks clocks) is loaded
//    before the MACs start.
//
// Processes (one DATAFLOW region per call): load_w16 (hp0..3) -> 4 FIFOs -> compute16 <- xs <- load_io16 (io
// read: s_tok, X, s_row) ; compute16 -> es (16 x 43-bit captures) -> dequant16 -> ys (4 x FP32) -> store16 (io
// write). The integrated top's GEMM unit (gemm_contract.h) runs the same compute16 behind the contract record.
//
// DRIVER CONTRACT (float_top built with -DFSTREAM_ARITH=16; registers as fpga/board/runtime.py REG_*):
//   0x00 ap_ctrl (bit 0 ap_start, bit 1 ap_done, bit 2 ap_idle), 0x10 ap_return, 64-bit pointers (low word at the
//   offset, high at +4): 0x18 w0, 0x24 w1, 0x30 w2, 0x3c w3, 0x48 xmem, 0x54 ymem; 32-bit 0x60 M, 0x68 K, 0x70 N.
//   M  = output channels, a multiple of 32 (the host pads: zero weight rows, s_row 1.0), 32..8192 (M_MAX)
//   K  = reduction length of this call, 1..4096 (longer reductions: 4096-wide chunks, the host adds the FP32
//        outputs in chunk order; X is quantized once over the whole row, the same s_tok goes with every chunk)
//   N  = tokens, >= 1. Np = 32 * ceil(N / 32).
//   w0..w3: pack_float.pack(Wq.view(uint16)) for this K chunk: port p's array is [M/32 groups][K] 128-bit words,
//           lane l (bits 16l+15:16l) = Wq[32 g + 8 p + l][k] (two's complement); the four arrays are back to back
//           (w_p = chunk base + 16 * p * (M/32) * K bytes), as runtime.chunk_layout. 16-byte aligned. 2 M K bytes.
//   xmem: one buffer of 128-bit words, 16-byte aligned (page-aligned is best), 4 M + 4 Np + 2 Np K bytes:
//           words [0, M/4)                   s_row[m] FP32, value j of word i = s_row[4 i + j]
//           words [M/4, M/4 + Np/4)          s_tok[n] FP32 (n >= N: any finite value; 1.0)
//           words [(M + Np)/4, ... + Np K/4) Xq: for block b < Np/32, for k < K, 4 words; word q holds
//                                            Xq[32 b + 8 q + j][k] in lane j (bits 16j+15:16j), j < 8;
//                                            tokens n >= N are zero
//           (s_row is per layer: a host may keep one xmem buffer per layer with s_row written once.)
//   ymem: Y[g][Np][32] FP32 (the FP32 engine's Y_tiled), 16-byte aligned, 4 M Np bytes: y[n][32 g + l] =
//           Y[g][n][l]; rows n >= N and padded columns hold 0.
//   return value 0, or 1 when M, K, N are out of range (nothing is read or written then).
// pack_float.pack_xmem_int16 / unpack_y_int16 implement it; float_shim.cpp linear_int16_raw runs this core on
// such an image, and tb_float (int16 mode) checks the Python image against the shim's own, bit for bit.
#pragma once
#include "float_linear.h"

#ifndef __VITIS_HLS__
#include <vector>
#endif

namespace fstream16 {

using fstream::F4;
using fstream::FL;
using fstream::IoWord;
using fstream::K_MAX;
using fstream::LANES;
using fstream::PORTS;
using fstream::WPort;

#ifndef FSTREAM_NBX
#define FSTREAM_NBX 2
#endif
constexpr int TB = 32;                   // tokens per block: MAC rows (TB x LANES MACs per clock)
constexpr int NBX = FSTREAM_NBX;         // blocks per tile: the weights of a group are re-used NBX times
constexpr int NTT = TB * NBX;            // tokens per tile
constexpr int XB = 8;                    // X banks, 4 int16 tokens per 64-bit word: 32 tokens per clock
constexpr int XHALF = NBX * K_MAX;       // words per bank per X buffer half
constexpr int XW_K = TB / 8;             // io words of X per (block, k): 4
constexpr int EW = 16;                   // values per emit word (per clock)
constexpr int EROWS = TB * LANES / EW;   // emit words per block: 64
constexpr int CW = 43;                   // captured accumulator bits: |block sum| < 2^42
constexpr int KMIN = 8;                  // shortest block in clocks (URAM write -> read distance)
constexpr int SETTLE = 8;                // clocks between an X half's last write and first read (and back)
constexpr int EDONE = 6;                 // clocks from a block's last emitted row until the MAC side may capture
constexpr int M_MAX = 8192;              // longest output (s_row kept on chip)
constexpr int YW_BLK = TB * LANES / F4;  // io words of Y per (group, block): 256 (4 KB)
constexpr int WFIFO16 = NBX > 1 ? K_MAX : 512;  // weight words buffered per port: a whole group when re-used
constexpr int SFIFO = 64;                // io read FIFOs (X: LUTRAM; scales: shift registers)

static_assert(NBX == 1 || NBX == 2 || NBX == 4, "NBX: 1, 2 or 4 blocks per tile (X addresses are bit fields)");
static_assert((K_MAX & (K_MAX - 1)) == 0, "K_MAX must be a power of 2");

#ifdef __VITIS_HLS__
typedef ap_int<16> I16;
typedef ap_int<48> Acc;   // the DSP48E2 P register
typedef ap_uint<CW> Cap;  // low 43 bits of an accumulator
typedef ap_uint<64> XWord;
typedef ap_uint<EW * CW> EWord;  // 16 captures, value j in bits [43j+42:43j]
typedef ap_uint<CW> U43;
template <class T>
using Fifo16 = hls::stream<T>;

static inline I16 w_lane16(const WPort& w, int l) {
#pragma HLS INLINE
  return (I16)w.range(16 * l + 15, 16 * l);
}
static inline XWord io_half(const IoWord& w, int h) {
#pragma HLS INLINE
  return (XWord)w.range(64 * h + 63, 64 * h);
}
static inline I16 x_lane(const XWord& w, int j) {
#pragma HLS INLINE
  return (I16)w.range(16 * j + 15, 16 * j);
}
static inline Cap cap_of(const Acc& a) {
#pragma HLS INLINE
  return (Cap)a.range(CW - 1, 0);
}
static inline Cap ew_get(const EWord& e, int j) {
#pragma HLS INLINE
  return (Cap)e.range(CW * j + CW - 1, CW * j);
}
static inline void ew_set(EWord& e, int j, const Cap& c) {
#pragma HLS INLINE
  e.range(CW * j + CW - 1, CW * j) = c;
}
static inline EWord ew_shift4(const EWord& e) {  // value j + 4 -> j (the dequantizer's 4 lanes per clock)
#pragma HLS INLINE
  return e >> (4 * CW);
}
static inline void acc_add(Acc& a, const I16& x, const I16& w) {
#pragma HLS INLINE
  // review R1: a named product bound to the DSP with latency 2 gives the MAC core an M register; x, w come from
  // the operand registers and a is the register after the post-adder (the P register), so no path runs from the
  // URAM output through the multiplier and the adder in one clock (HLS probe: mac_muladd_..._2_1 with m_reg)
  ap_int<32> pr = x * w;
#pragma HLS BIND_OP variable = pr op = mul impl = dsp latency = 2
  a += pr;
}
static inline float fmul_dq(float a, float b) {
#pragma HLS INLINE
  return a * b;  // FP32 multiplier core (denormals flush to zero)
}
#else
typedef int16_t I16;
typedef int64_t Acc;   // holds a 48-bit two's-complement value (acc_add wraps like the DSP)
typedef uint64_t Cap;  // 43-bit field
typedef uint64_t XWord;
typedef uint64_t U43;
struct EWord {
  Cap v[EW];
};
// -DFSTREAM_CHAOS=p: every non-blocking FIFO operation fails with probability p % (data "not there yet", FIFO
// "full"), so the C model runs the stall paths of compute16 (late weights, late X, a full emit FIFO) and must still
// give the same bits (tb_float int16 mode is built that way once in make test-int16).
static inline bool fifo_chaos() {
#ifdef FSTREAM_CHAOS
  static thread_local uint64_t s = 0x9E3779B97F4A7C15ull;  // per OpenMP thread
  s ^= s << 13;
  s ^= s >> 7;
  s ^= s << 17;
  return (s % 100) < (uint64_t)(FSTREAM_CHAOS);
#else
  return false;
#endif
}
template <class T>
class Fifo16 {  // the host runs the processes one after another: an unbounded queue
 public:
  void write(const T& v) { q.push_back(v); }
  bool full() {  // a FIFO reported not full takes the write of the same clock (as the hardware's full_n)
    wseen = !fifo_chaos();
    return !wseen;
  }
  bool write_nb(const T& v) {
    if (!wseen && fifo_chaos()) return false;
    wseen = false;
    q.push_back(v);
    return true;
  }
  T read() { return q[r++]; }
  bool empty() {  // a FIFO reported non-empty is read successfully in the same clock (as the hardware's empty_n)
    seen = r < q.size() && !fifo_chaos();
    return !seen;
  }
  bool read_nb(T& v) {
    if (r >= q.size() || (!seen && fifo_chaos())) return false;
    seen = false;
    v = q[r++];
    return true;
  }

 private:
  std::vector<T> q;
  size_t r = 0;
  bool seen = false, wseen = false;
};
static inline I16 w_lane16(const WPort& w, int l) { return (I16)w.v[l]; }
static inline XWord io_half(const IoWord& w, int h) {
  XWord x;
  memcpy(&x, (const char*)&w + 8 * h, 8);  // little-endian: lane j of the word at bits [16j+15:16j]
  return x;
}
static inline I16 x_lane(const XWord& w, int j) { return (I16)(uint16_t)(w >> (16 * j)); }
static inline Cap cap_of(const Acc& a) { return (Cap)a & ((1ULL << CW) - 1); }
static inline Cap ew_get(const EWord& e, int j) { return e.v[j]; }
static inline void ew_set(EWord& e, int j, const Cap& c) { e.v[j] = c; }
static inline EWord ew_shift4(const EWord& e) {
  EWord r;
  for (int j = 0; j < EW; j++) r.v[j] = j + 4 < EW ? e.v[j + 4] : 0;
  return r;
}
static inline void acc_add(Acc& a, const I16& x, const I16& w) {
  const uint64_t s = (uint64_t)a + (uint64_t)(int64_t)((int32_t)x * (int32_t)w);
  a = (int64_t)(s << 16) >> 16;  // 48-bit wrap, as the P register
}
static inline float ftz(float f) {  // the Xilinx FP32 cores flush denormal inputs and results to zero
  uint32_t u;
  memcpy(&u, &f, sizeof u);
  if (((u >> 23) & 0xFF) == 0) u &= 0x80000000u;
  memcpy(&f, &u, sizeof f);
  return f;
}
static inline float fmul_dq(float a, float b) { return ftz(ftz(a) * ftz(b)); }
#endif

// Signed block sum from two captures of the same accumulator: (cap - prev) mod 2^43, sign-extended.
// In: 43-bit fields. Out: the 43-bit two's-complement difference (as a 43-bit field).
static inline U43 cap_diff(const Cap& cap, const Cap& prev) {
#pragma HLS INLINE
#ifdef __VITIS_HLS__
  return (U43)(cap - prev);  // ap_uint<43> arithmetic wraps mod 2^43
#else
  return (cap - prev) & ((1ULL << CW) - 1);
#endif
}

// 43-bit two's-complement integer -> FP32, round to nearest even (exact below 2^24); the same bits as
// (float)(int64_t) of the value (tb_int16_units.cpp checks it). A 6-stage logarithmic shifter normalizes (shift by
// 32, 16, 8, 4, 2, 1 while the top bits are zero; the stage bits are the leading-zero count), one increment rounds.
static inline float i43_to_f32(const U43& d) {
#pragma HLS INLINE
  const bool neg = (d >> (CW - 1)) & 1;
#ifdef __VITIS_HLS__
  const U43 mag = neg ? (U43)(-d) : d;  // |d| <= 2^42 fits 43 unsigned bits
#else
  const U43 mag = (neg ? (U43)(0 - d) : d) & ((1ULL << CW) - 1);
#endif
  U43 nrm = mag;  // shifted until the leading one is at bit 42
  uint32_t lz = 0;
  for (int st = 5; st >= 0; st--) {
#pragma HLS UNROLL
    const int sh = 1 << st;
    if ((nrm >> (CW - sh)) == 0) {
#ifdef __VITIS_HLS__
      nrm = nrm << sh;
#else
      nrm = (nrm << sh) & ((1ULL << CW) - 1);
#endif
      lz |= (uint32_t)sh;
    }
  }
  const uint32_t msb = (uint32_t)(CW - 1) - lz;  // position of the leading one (mag != 0)
  const uint32_t sig = (uint32_t)(nrm >> (CW - 24));                  // 24 bits with the leading one
  const uint32_t rem = (uint32_t)(nrm & ((1u << (CW - 24)) - 1));     // 19 dropped bits
  const uint32_t half = 1u << (CW - 25);
  const bool up = rem > half || (rem == half && (sig & 1));
  const uint32_t r = sig + (up ? 1u : 0u);  // 2^24 after a carry out of an all-ones significand
  const uint32_t carry = r >> 24;
  const uint32_t e = msb + 127u + carry;
  const uint32_t frac = (carry ? (r >> 1) : r) & 0x7FFFFFu;
  uint32_t out = (mag == 0) ? 0u : (((neg ? 1u : 0u) << 31) | (e << 23) | frac);
  float f;
  memcpy(&f, &out, sizeof f);
  return f;
}

static inline int blocks_of_tile(int nblk, int t) {
#pragma HLS INLINE
  const int rest = nblk - t * NBX;
  return rest < NBX ? rest : NBX;
}

// Process 1: the weights of every group, once per tile, the four ports in one loop.
static void load_w16(const WPort* w0, const WPort* w1, const WPort* w2, const WPort* w3, int words, int ntiles,
                     Fifo16<WPort>& s0, Fifo16<WPort>& s1, Fifo16<WPort>& s2, Fifo16<WPort>& s3) {
tiles_w:
  for (int t = 0; t < ntiles; t++) {
  load_w:
    for (int i = 0; i < words; i++) {
#pragma HLS PIPELINE II = 1
      s0.write(w0[i]);
      s1.write(w1[i]);
      s2.write(w2[i]);
      s3.write(w3[i]);
    }
  }
}

// Process 2: all reads of the io bundle. Tile 0's s_tok and X first (the MACs wait for them), then s_row (the
// dequantizer needs it after the first block), then the other tiles.
static void load_io16(const IoWord* xmem, int groups, int K, int nblk, int ntiles, int off_stok, int off_x,
                      Fifo16<IoWord>& srow_s, Fifo16<IoWord>& stok_s, Fifo16<IoWord>& xs) {
tiles_io:
  for (int t = 0; t < ntiles; t++) {
    const int nbt = blocks_of_tile(nblk, t);
    const int sbase = off_stok + t * NBX * (TB / F4);
  load_stok:
    for (int i = 0; i < nbt * (TB / F4); i++) {
#pragma HLS PIPELINE II = 1
      stok_s.write(xmem[sbase + i]);
    }
    const size_t xbase = (size_t)off_x + (size_t)t * NBX * K * XW_K;
  load_x:
    for (int i = 0; i < nbt * K * XW_K; i++) {
#pragma HLS PIPELINE II = 1
      xs.write(xmem[xbase + i]);
    }
    if (t == 0) {
    load_srow:
      for (int i = 0; i < groups * (LANES / F4); i++) {
#pragma HLS PIPELINE II = 1
        srow_s.write(xmem[i]);
      }
    }
  }
}

#ifndef __VITIS_HLS__
// C model only: clocks of compute16's main loop (one per pipelined iteration), and why the MACs idled.
struct SimStats {
  long clocks = 0, macs = 0, wait_x = 0, wait_w = 0, wait_cap = 0;
};
static SimStats sim_stats;
#endif

// Process 3: the MAC array. One flat loop; every clock it may (in this order) capture a finished block into the
// shadow, emit one shadow word, do one k of 1,024 MACs, and store one io word of the next X tile.
// Bubbles (no X or weights yet, shadow busy) multiply by zero, so the accumulators never need a reset or an
// enable. The loop has no blocking FIFO access (weights: empty() then read_nb, emit: write_nb, X: read_nb), so
// the pipeline never stalls: no stall enable fans out to the 49k accumulator and 44k shadow flip-flops, and the
// loop's control decisions (capture, exit) use only the state registered at the start of the clock.
static void compute16(Fifo16<WPort>& s0, Fifo16<WPort>& s1, Fifo16<WPort>& s2, Fifo16<WPort>& s3,
                      Fifo16<IoWord>& xs, Fifo16<EWord>& es, int groups, int K, int nblk, int ntiles) {
#ifdef __VITIS_HLS__
  XWord xbuf[XB][2 * XHALF];  // [bank][half][block][k]: 4 tokens per word
#pragma HLS BIND_STORAGE variable = xbuf type = ram_s2p impl = uram
#pragma HLS ARRAY_PARTITION variable = xbuf complete dim = 1
#if FSTREAM_NBX > 1
  WPort wbuf[PORTS][K_MAX];  // the current group's weights, for blocks 1..NBX-1
#pragma HLS BIND_STORAGE variable = wbuf type = ram_s2p impl = uram
#pragma HLS ARRAY_PARTITION variable = wbuf complete dim = 1
#endif
#else
  std::vector<XWord> xv((size_t)XB * 2 * XHALF);
  XWord(*xbuf)[2 * XHALF] = (XWord(*)[2 * XHALF])xv.data();
#if FSTREAM_NBX > 1
  std::vector<WPort> wv((size_t)PORTS * K_MAX);
  WPort(*wbuf)[K_MAX] = (WPort(*)[K_MAX])wv.data();
#endif
#endif
  Acc acc[TB][LANES];
#pragma HLS ARRAY_PARTITION variable = acc complete dim = 0
  Cap shadow[TB][LANES];
#pragma HLS ARRAY_PARTITION variable = shadow complete dim = 0
  WPort wcur[PORTS];
#pragma HLS ARRAY_PARTITION variable = wcur complete
zero_a:
  for (int a = 0; a < TB; a++) {
#pragma HLS UNROLL
    for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
      acc[a][l] = 0;  // the only reset: block sums are differences of captures
    }
  }
  for (int q = 0; q < PORTS; q++) {
#pragma HLS UNROLL
    wcur[q] = WPort();
  }

  // tile 0 before the MACs start
  const int nb0 = blocks_of_tile(nblk, 0);
  int pb = 0, pk = 0, pq = 0;
prefill:
  for (int i = 0; i < nb0 * K * XW_K; i++) {
#pragma HLS PIPELINE II = 1
    const IoWord v = xs.read();
    const int addr = pb * K_MAX + pk;
    xbuf[2 * pq][addr] = io_half(v, 0);
    xbuf[2 * pq + 1][addr] = io_half(v, 1);
    if (pq == XW_K - 1) {
      pq = 0;
      if (pk == K - 1) {
        pk = 0;
        pb++;
      } else {
        pk++;
      }
    } else {
      pq++;
    }
  }

  const int klast = (K < KMIN ? KMIN : K) - 1;  // blocks of K < KMIN pad with zero products
  const int kfill_last = K - 1, groups_m1 = groups - 1;
  const int last_nbt_m1 = blocks_of_tile(nblk, ntiles - 1) - 1;  // the last tile may have fewer blocks
  // Three sides, each deciding from its own registered state: the MACs (and the capture at a block's last MAC),
  // the emit (shadow -> es, paced by es), the X fill (xs -> the free half). The MAC side sees the other two
  // through delay lines (EDONE, SETTLE clocks), so no decision depends on a FIFO status of the same clock on
  // another side, and the shadow is read in program order after its write (no forwarding muxes).
  int g = 0, b = 0, k = 0;                       // MAC position in tile t
  bool kin = true, kend = klast == 0;            // k < K, k == klast (flags: no compare on the decision path)
  int tleft = ntiles - 1;                        // tiles after t
  int nbt_m1 = ntiles == 1 ? last_nbt_m1 : NBX - 1;  // blocks of tile t, minus one
  bool b0 = true, bend = nbt_m1 == 0;            // b == 0, b == nbt_m1
  bool gend = groups_m1 == 0, tlast = ntiles == 1;  // g == groups - 1, tleft == 0
  bool thalf = false;                            // X half of tile t (t & 1)
  bool done = false;
  bool sbusy = false;                            // MAC side: the shadow holds a block (until EDONE after its emit)
  bool edl[EDONE];                               // emit-done events of the last EDONE clocks
#pragma HLS ARRAY_PARTITION variable = edl complete
  bool emitting = false;                         // emit side: obuf holds shadow row er, still to go
  int er = 0;                                    // shadow row in obuf
  EWord obuf = EWord();                          // the emit word being offered to es (an output register)
  bool a1f = true, a2f = false;                  // fill side: ft - t >= 1, ft - t == 2
  bool am1 = true, am2 = false;                  // MAC side: the same, a fill counting SETTLE clocks after it ends
  bool fdl[SETTLE];                              // fill-done events of the last SETTLE clocks
#pragma HLS ARRAY_PARTITION variable = fdl complete
  int fleft = ntiles - 1;                        // tiles still to fill, from ft on
  bool fmore = ntiles > 1;                       // fleft > 0
  bool fklast = kfill_last == 0;                 // fk == kfill_last
  int fb = 0, fk = 0, fq = 0;                    // fill position in tile ft (block, k, io word)
  bool fhalf = true;                             // X half of tile ft
  int fnb_m1 = ntiles == 2 ? last_nbt_m1 : NBX - 1;  // blocks of tile ft, minus one
  bool fqlast = false, fblast = fnb_m1 == 0;     // fq == XW_K - 1, fb == fnb_m1
  int tage = SETTLE;                             // clocks since tile t started (saturating at SETTLE)
  bool tsettled = true;                          // tage >= SETTLE: the reads of tile t - 1 are done
  I16 xd[TB], wd[LANES];  // MAC operand registers (DSP A / B)
#pragma HLS ARRAY_PARTITION variable = xd complete
#pragma HLS ARRAY_PARTITION variable = wd complete
  for (int a = 0; a < TB; a++) {
#pragma HLS UNROLL
    xd[a] = 0;
  }
  for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
    wd[l] = 0;
  }
  bool cap1 = false, cap2 = false;  // a capture decided 1 / 2 clocks ago (its last products reach acc at cap1)
  for (int i = 0; i < EDONE; i++) {
#pragma HLS UNROLL
    edl[i] = false;
  }
  for (int i = 0; i < SETTLE; i++) {
#pragma HLS UNROLL
    fdl[i] = false;
  }
main:
  while (true) {
#pragma HLS PIPELINE II = 1
#pragma HLS DEPENDENCE variable = xbuf inter false
#pragma HLS DEPENDENCE variable = xbuf intra false
#if FSTREAM_NBX > 1
#pragma HLS DEPENDENCE variable = wbuf inter false
#pragma HLS DEPENDENCE variable = wbuf intra false
#endif
    // MAC side. The exit test is taken at the end of the body (a break here would put every register of the
    // body under !exit and turn the conditional capture into selects); in the exit clock nothing happens.
    const bool fin = done && !sbusy;
    // next values of the flags, from the registered counters, beside the decisions
    const bool kin_next = k + 1 < K, kend_next = k + 1 == klast;
    const bool bend_next = b + 1 == nbt_m1, gend_next = g + 1 == groups_m1;
    const int nbt_m1_next = tleft == 1 ? last_nbt_m1 : NBX - 1;  // of tile t + 1
#if FSTREAM_NBX > 1
    const bool need_w = kin && b0;  // block 0 takes the group's weights from the FIFOs, blocks 1.. from wbuf
#else
    const bool need_w = kin;  // every block reads the FIFOs (b0 unused)
    (void)b0;
#endif
    const bool wok = !need_w || (!s0.empty() && !s1.empty() && !s2.empty() && !s3.empty());
    // a block's last MAC also captures it, so it waits for a free shadow
    const bool mac = !done && am1 && wok && !(kend && sbusy);
    const bool live = mac && kin;
    const bool capture = mac && kend;
#ifndef __VITIS_HLS__
    sim_stats.clocks++;
    sim_stats.macs += mac;
    if (!done && !mac) {
      if (kend && sbusy)
        sim_stats.wait_cap++;
      else if (!am1)
        sim_stats.wait_x++;
      else
        sim_stats.wait_w++;
    }
#endif

    // 1. one k of the current block. X / weight buffer addresses are bit fields (half | block | k): no adder and
    //    no select on the k path; a padding step of a K < KMIN block (k >= K) reads whatever is at k, times zero.
    const int kk = k;
    XWord xr[XB];
    for (int u = 0; u < XB; u++) {
#pragma HLS UNROLL
      xr[u] = xbuf[u][(thalf ? XHALF : 0) | (b * K_MAX) | kk];
    }
#if FSTREAM_NBX > 1
    WPort wq[PORTS];
#pragma HLS ARRAY_PARTITION variable = wq complete
    if (live && need_w) {  // the four FIFOs are filled in lock step (load_w16) and all non-empty here
      s0.read_nb(wq[0]);
      s1.read_nb(wq[1]);
      s2.read_nb(wq[2]);
      s3.read_nb(wq[3]);
      for (int q = 0; q < PORTS; q++) {
#pragma HLS UNROLL
        wbuf[q][kk] = wq[q];
      }
    } else {
      for (int q = 0; q < PORTS; q++) {
#pragma HLS UNROLL
        wq[q] = wbuf[q][kk];
      }
    }
#else
    if (live) {
      s0.read_nb(wcur[0]);
      s1.read_nb(wcur[1]);
      s2.read_nb(wcur[2]);
      s3.read_nb(wcur[3]);
    }
    WPort wq[PORTS];
#pragma HLS ARRAY_PARTITION variable = wq complete
    for (int q = 0; q < PORTS; q++) {
#pragma HLS UNROLL
      wq[q] = wcur[q];
    }
#endif
    // 2. the MAC pipeline (review R1): this clock's operands go into registers (xd, wd: the DSP's A / B inputs
    //    come from flip-flops, not from the URAM output and the live select), the products of the operands
    //    registered one clock earlier go into acc (acc <= acc + xd * wd: the DSP's P register with its own Z = P
    //    feedback). A block is captured from the registered acc two clocks after its last MAC decision (cap2),
    //    i.e. from the P register's output (never from the adder output), before this clock's add.
    const int er1 = (er + 1) & (EROWS - 1);  // the emit's next row, read before this clock's capture (below)
    EWord onext;
    for (int j = 0; j < EW; j++) {
#pragma HLS UNROLL
      ew_set(onext, j, shadow[er1 >> 1][(er1 & 1) * EW + j]);
    }
    if (cap2) {
      for (int a = 0; a < TB; a++) {
#pragma HLS UNROLL
        for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
          shadow[a][l] = cap_of(acc[a][l]);
        }
      }
    }
    EWord orow0;  // the first emit word of a block captured now: token 0, lanes 0..15 (from the P registers)
    for (int j = 0; j < EW; j++) {
#pragma HLS UNROLL
      ew_set(orow0, j, cap_of(acc[0][j]));
    }
    for (int a = 0; a < TB; a++) {
#pragma HLS UNROLL
      for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
        acc_add(acc[a][l], xd[a], wd[l]);
      }
    }
    for (int a = 0; a < TB; a++) {
#pragma HLS UNROLL
      xd[a] = live ? x_lane(xr[a / 4], a % 4) : (I16)0;
    }
    for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
      wd[l] = w_lane16(wq[l / FL], l % FL);
    }
    const bool cap_now = cap2;
    cap2 = cap1;
    cap1 = capture;
    // 3. emit side: obuf (shadow row er; r = 2 * token + half of the 32 lanes) goes to es; its write's own
    //    result advances the emit (a full() test would be hoisted to the start of the clock and then has to follow
    //    the previous clock's late write: II = 2). The next row (onext) was read from the registered er before
    //    the capture in program order, so only a 2:1 select follows the FIFO handshake.
    bool edone = false;
    if (emitting) {
      if (es.write_nb(obuf)) {
        if (er == EROWS - 1) {
          emitting = false;
          edone = true;
        }
        er = er1;
        obuf = onext;
      }
    }
    if (cap_now) {  // never while emitting: the MAC side saw the shadow free, EDONE clocks after its last row
      obuf = orow0;  // row 0: token 0, lanes 0..15 of the block just captured
      emitting = true;
      er = 0;
    }
    const bool sfree = edl[EDONE - 1];
    for (int i = EDONE - 1; i > 0; i--) {
#pragma HLS UNROLL
      edl[i] = edl[i - 1];
    }
    edl[0] = edone;
    if (capture) {
      sbusy = true;
    } else if (sfree) {
      sbusy = false;
    }

    // MAC position
    bool tinc = false;
    if (mac) {
      if (kend) {
        k = 0;
        kin = true;
        kend = klast == 0;
        if (bend) {
          b = 0;
          b0 = true;
          if (gend) {
            g = 0;
            gend = groups_m1 == 0;
            if (tlast) {
              done = true;
            } else {
              tinc = true;
            }
          } else {
            g++;
            gend = gend_next;
          }
          bend = (tinc ? nbt_m1_next : nbt_m1) == 0;
        } else {
          b++;
          b0 = false;
          bend = bend_next;
        }
      } else {
        k++;
        kin = kin_next;
        kend = kend_next;
      }
    }
    if (tinc) {
      nbt_m1 = nbt_m1_next;
      tlast = tleft == 1;
      tleft--;
      thalf = !thalf;
    }

    // 4. fill side: one io word of tile ft: tile t + 1 (the other half) once tile t has run SETTLE clocks, or
    //    tile t itself when its X came later than the MACs of tile t - 1 ran
    const bool fill_ok = fmore && (!a1f || (!a2f && tsettled));
    const bool fklast_next = fk + 1 == kfill_last, fqlast_next = fq + 1 == XW_K - 1;
    const bool fblast_next = fb + 1 == fnb_m1;
    const int fnb_m1_next = fleft == 2 ? last_nbt_m1 : NBX - 1;  // of tile ft + 1
    const bool fill_go = fill_ok && !xs.empty();
    bool filled = false;
    if (fill_go) {
      IoWord v = IoWord();
      xs.read_nb(v);  // not empty (fill_go): succeeds
      const int addr = (fhalf ? XHALF : 0) | (fb * K_MAX) | fk;
      xbuf[2 * fq][addr] = io_half(v, 0);
      xbuf[2 * fq + 1][addr] = io_half(v, 1);
      if (fqlast) {
        fq = 0;
        fqlast = false;
        if (fklast) {
          fk = 0;
          fklast = kfill_last == 0;
          if (fblast) {
            fb = 0;
            fblast = fnb_m1_next == 0;
            filled = true;
          } else {
            fb++;
            fblast = fblast_next;
          }
        } else {
          fk++;
          fklast = fklast_next;
        }
      } else {
        fq++;
        fqlast = fqlast_next;
      }
    }
    if (filled) {
      fnb_m1 = fnb_m1_next;
      fmore = fleft > 1;
      fleft--;
      fhalf = !fhalf;
    }
    if (filled && !tinc) {  // fill side: ft - t + 1 (a fill starts only with ft - t <= 1)
      a2f = a1f;
      a1f = true;
    } else if (tinc && !filled) {  // ft - t - 1
      a1f = a2f;
      a2f = false;
    }
    const bool fseen = fdl[SETTLE - 1];  // MAC side: a tile is in SETTLE clocks after its last word
    for (int i = SETTLE - 1; i > 0; i--) {
#pragma HLS UNROLL
      fdl[i] = fdl[i - 1];
    }
    fdl[0] = filled;
    if (fseen && !tinc) {
      am2 = am1;
      am1 = true;
    } else if (tinc && !fseen) {
      am1 = am2;
      am2 = false;
    }
    tsettled = !tinc && tage >= SETTLE - 1;
    tage = tinc ? 0 : (tage < SETTLE ? tage + 1 : SETTLE);
    if (fin) break;
  }
}

// Process 4: block sums -> FP32, 4 values per clock (review R2 / CONTRACT E-1: the io store takes 4 FP32 per clock,
// so 16 dequantizer lanes bought nothing but 72 DSPs). One emit word (16 captures) is held for 4 clocks.
// Captures of one accumulator differ by exactly its block sum (mod 2^43).
constexpr int DQL = 4;         // dequantizer lanes
constexpr int DQS = EW / DQL;  // clocks per emit word
static void dequant16(Fifo16<IoWord>& srow_s, Fifo16<IoWord>& stok_s, Fifo16<EWord>& es, Fifo16<IoWord>& ys,
                      int groups, int nblk, int ntiles) {
  float srow[M_MAX];
#pragma HLS ARRAY_PARTITION variable = srow cyclic factor = 4
  float stok[NTT];
  Cap prev[EROWS * DQS][DQL];  // [row * 4 + quarter][lane]
#pragma HLS ARRAY_PARTITION variable = prev complete dim = 2
#pragma HLS BIND_STORAGE variable = prev type = ram_2p impl = lutram
prev_zero:
  for (int r = 0; r < EROWS * DQS; r++) {
#pragma HLS PIPELINE II = 1
    for (int j = 0; j < DQL; j++) {
#pragma HLS UNROLL
      prev[r][j] = 0;
    }
  }
srow_in:
  for (int i = 0; i < groups * (LANES / F4); i++) {
#pragma HLS PIPELINE II = 1
    const IoWord v = srow_s.read();
    for (int j = 0; j < F4; j++) {
#pragma HLS UNROLL
      srow[F4 * i + j] = fstream::io_lane(v, j);
    }
  }
tiles_dq:
  for (int t = 0; t < ntiles; t++) {
    const int nbt = blocks_of_tile(nblk, t);
  stok_in:
    for (int i = 0; i < nbt * (TB / F4); i++) {
#pragma HLS PIPELINE II = 1
      const IoWord v = stok_s.read();
      for (int j = 0; j < F4; j++) {
#pragma HLS UNROLL
        stok[F4 * i + j] = fstream::io_lane(v, j);
      }
    }
    int g = 0, b = 0, r = 0, q = 0;
    EWord cw = EWord();
  dq:
    for (int it = 0; it < groups * nbt * EROWS * DQS; it++) {
#pragma HLS PIPELINE II = 1
#pragma HLS DEPENDENCE variable = prev inter false
      if (q == 0) cw = es.read();
      const float st = stok[b * TB + (r >> 1)];
      IoWord o;
      for (int j = 0; j < DQL; j++) {
#pragma HLS UNROLL
        const Cap c = ew_get(cw, j);  // lanes 4q..4q+3 of the word: shifted down by 4 values per clock
        const U43 d = cap_diff(c, prev[r * DQS + q][j]);
        prev[r * DQS + q][j] = c;
        const float sr = srow[g * LANES + (r & 1) * EW + DQL * q + j];
        fstream::io_set(o, j, fmul_dq(fmul_dq(i43_to_f32(d), sr), st));
      }
      ys.write(o);
      cw = ew_shift4(cw);
      if (q == DQS - 1) {
        q = 0;
        if (r == EROWS - 1) {
          r = 0;
          if (b == nbt - 1) {
            b = 0;
            g++;
          } else {
            b++;
          }
        } else {
          r++;
        }
      } else {
        q++;
      }
    }
  }
}

// Process 5: one burst per (tile, group): Y[g][rows of the tile][32] FP32.
static void store16(Fifo16<IoWord>& ys, IoWord* ymem, int groups, int nblk, int ntiles) {
tiles_st:
  for (int t = 0; t < ntiles; t++) {
    const int nbt = blocks_of_tile(nblk, t);
  groups_st:
    for (int g = 0; g < groups; g++) {
      const size_t base = ((size_t)g * nblk + (size_t)t * NBX) * YW_BLK;
    store_y:
      for (int i = 0; i < nbt * YW_BLK; i++) {
#pragma HLS PIPELINE II = 1
        ymem[base + i] = ys.read();
      }
    }
  }
}

static void run_all16(const WPort* w0, const WPort* w1, const WPort* w2, const WPort* w3, const IoWord* xmem,
                      IoWord* ymem, int groups, int K, int nblk, int ntiles, int off_stok, int off_x) {
#pragma HLS DATAFLOW
  Fifo16<WPort> s0, s1, s2, s3;
  Fifo16<IoWord> xs, srow_s, stok_s;
  Fifo16<EWord> es;
  Fifo16<IoWord> ys;
#pragma HLS STREAM variable = s0 depth = WFIFO16
#pragma HLS STREAM variable = s1 depth = WFIFO16
#pragma HLS STREAM variable = s2 depth = WFIFO16
#pragma HLS STREAM variable = s3 depth = WFIFO16
#if FSTREAM_NBX > 1
#pragma HLS BIND_STORAGE variable = s0 type = fifo impl = uram
#pragma HLS BIND_STORAGE variable = s1 type = fifo impl = uram
#pragma HLS BIND_STORAGE variable = s2 type = fifo impl = uram
#pragma HLS BIND_STORAGE variable = s3 type = fifo impl = uram
#endif
#pragma HLS STREAM variable = xs depth = SFIFO
#pragma HLS STREAM variable = srow_s depth = SFIFO
#pragma HLS STREAM variable = stok_s depth = SFIFO
#pragma HLS STREAM variable = es depth = 4
#pragma HLS STREAM variable = ys depth = 16  // 4 FP32 per word
#pragma HLS BIND_STORAGE variable = xs type = fifo impl = lutram
#pragma HLS BIND_STORAGE variable = srow_s type = fifo impl = srl
#pragma HLS BIND_STORAGE variable = stok_s type = fifo impl = srl
#pragma HLS BIND_STORAGE variable = es type = fifo impl = srl
#pragma HLS BIND_STORAGE variable = ys type = fifo impl = srl
  load_w16(w0, w1, w2, w3, groups * K, ntiles, s0, s1, s2, s3);
  load_io16(xmem, groups, K, nblk, ntiles, off_stok, off_x, srow_s, stok_s, xs);
  compute16(s0, s1, s2, s3, xs, es, groups, K, nblk, ntiles);
  dequant16(srow_s, stok_s, es, ys, groups, nblk, ntiles);
  store16(ys, ymem, groups, nblk, ntiles);
}

// One call. w0..w3: the four port arrays of this K chunk ([M/32][K] words each). xmem: the buffer of the DRIVER
// CONTRACT with s_row at word 0, s_tok at word off_stok, X at word off_x (the HLS top passes M/4 and (M+Np)/4;
// the shim shifts them to run slices of groups). ymem: Y[M/32][Np][32] FP32.
static int linear_int16_core(const WPort* w0, const WPort* w1, const WPort* w2, const WPort* w3, const IoWord* xmem,
                             IoWord* ymem, int M, int K, int N, int off_stok, int off_x) {
  if (M % LANES || M < LANES || M > M_MAX || K < 1 || K > K_MAX || N < 1 || N > 65535) return 1;  // N + 31 in int
  const int groups = M / LANES;
  const int nblk = (N + TB - 1) / TB;
  const int ntiles = (nblk + NBX - 1) / NBX;
  run_all16(w0, w1, w2, w3, xmem, ymem, groups, K, nblk, ntiles, off_stok, off_x);
  return 0;
}

}  // namespace fstream16
