// Bit-exact int8 tiny-policy forward pass, written for Vitis HLS.
//
// Plain C++ types only, so the same code builds three ways:
//   - Vitis HLS synthesis, through the AXI-Stream wrapper in policy_top.cpp
//   - clang/gcc C model, through tb_kernel.cpp (runs on a laptop, no Xilinx tools)
//   - the ARM CPU baseline on the board (same loops, -O3)
//
// v0.1 goal is correctness (milestone M4), not speed: layers run one after
// another, about 1 multiply-add per clock. See plans/fpga.md for the M5 steps.
//
// Spec: ref/intref.py is the tiebreaker.
#pragma once
#include <stdint.h>
#ifndef __SYNTHESIS__
#include <string.h>
#endif

namespace policy {

constexpr int IMG_H = 96, IMG_W = 96, IMG_C = 3;
constexpr int IMG_BYTES = IMG_H * IMG_W * IMG_C;   // 27,648
constexpr int AUX_LEN = 10;
constexpr int PACKET_BYTES = IMG_BYTES + AUX_LEN;  // 27,658
constexpr int OUT_LEN = 48;
constexpr int N_LAYERS = 8;
constexpr int WEIGHT_FILE_BYTES = 568240;
constexpr uint32_t VERSION = 0x00000100;  // 0.1.0

constexpr int C1 = 16, C2 = 32, C3 = 64, C4 = 96, C5 = 128;
constexpr int FLAT = 3 * 3 * C5;              // 1,152
constexpr int FC6_IN = FLAT + AUX_LEN;        // 1,162
constexpr int FC6_OUT = 256, FC7_OUT = 256;

enum Mode { MODE_LOAD_WEIGHTS = 0, MODE_INFER = 1 };
enum Status { OK = 0, BAD_SHIFT = 1, BAD_TLAST = 2, BAD_MODE = 3 };

#ifndef __SYNTHESIS__
// C model only: if set, each layer's int8 output is copied here (layer k -> debug_layers[k-1]).
static int8_t* debug_layers[N_LAYERS] = {};
#endif

static inline void debug_dump(int k, const int8_t* a, int n) {
#ifndef __SYNTHESIS__
  if (debug_layers[k - 1]) memcpy(debug_layers[k - 1], a, n);
#endif
}

// y = (acc + 2^(s-1)) >> s, arithmetic shift, then saturate.
// ReLU layers clamp to [0, 127], the last layer to [-127, 127].
static inline int8_t requant(int32_t acc, int s, bool relu) {
#pragma HLS INLINE
  int32_t y = (acc + (1 << (s - 1))) >> s;  // >> on a negative int32 is arithmetic in gcc/clang/HLS
  int32_t lo = relu ? 0 : -127;
  if (y < lo) y = lo;
  if (y > 127) y = 127;
  return (int8_t)y;
}

// 3x3 conv, stride 2, pad 1, ReLU. in: HWC [H][W][CI]. out: HWC [H/2][W/2][CO].
// w: [CO][ky][kx][CI] flattened to [CO][9*CI].
template <int H, int W, int CI, int CO>
static void conv3x3s2(const int8_t in[H * W * CI], int8_t out[(H / 2) * (W / 2) * CO],
                      const int8_t w[CO][9 * CI], const int32_t b[CO], int s) {
  constexpr int HO = H / 2, WO = W / 2;
conv_y:
  for (int oy = 0; oy < HO; oy++) {
  conv_x:
    for (int ox = 0; ox < WO; ox++) {
    conv_o:
      for (int oc = 0; oc < CO; oc++) {
        int32_t acc = b[oc];
        int ky = 0, kx = 0, ic = 0;
      conv_k:
        for (int k = 0; k < 9 * CI; k++) {
#pragma HLS PIPELINE II = 1
          int iy = 2 * oy + ky - 1, ix = 2 * ox + kx - 1;
          bool inside = iy >= 0 && iy < H && ix >= 0 && ix < W;
          int8_t x = inside ? in[(iy * W + ix) * CI + ic] : (int8_t)0;  // zero padding
          acc += (int32_t)x * (int32_t)w[oc][k];
          if (++ic == CI) {
            ic = 0;
            if (++kx == 3) {
              kx = 0;
              ++ky;
            }
          }
        }
        out[(oy * WO + ox) * CO + oc] = requant(acc, s, true);
      }
    }
  }
}

// Fully connected. w: [OUT][IN].
template <int IN, int OUT>
static void fc(const int8_t in[IN], int8_t out[OUT], const int8_t w[OUT][IN], const int32_t b[OUT],
               int s, bool relu) {
fc_o:
  for (int o = 0; o < OUT; o++) {
    int32_t acc = b[o];
  fc_i:
    for (int i = 0; i < IN; i++) {
#pragma HLS PIPELINE II = 1
      acc += (int32_t)in[i] * (int32_t)w[o][i];
    }
    out[o] = requant(acc, s, relu);
  }
}

// weights.bin layout per layer: int8 weights (row-major), then int32 little-endian biases.
template <int ROWS, int COLS, class Src>
static bool load_layer(Src& src, int8_t w[ROWS][COLS], int32_t b[ROWS], int& n, int total) {
  bool last = false, bad = false;
load_w:
  for (int r = 0; r < ROWS; r++) {
    for (int c = 0; c < COLS; c++) {
#pragma HLS PIPELINE II = 1
      w[r][c] = (int8_t)read_byte(src, last);
      bad |= last != (++n == total);
    }
  }
load_b:
  for (int r = 0; r < ROWS; r++) {
    uint32_t u = 0;
    for (int byte = 0; byte < 4; byte++) {
#pragma HLS PIPELINE II = 1
      u |= (uint32_t)read_byte(src, last) << (8 * byte);
      bad |= last != (++n == total);
    }
    b[r] = (int32_t)u;
  }
  return !bad;
}

// The whole accelerator. The includer defines, before including this header,
//   uint8_t read_byte(Src&, bool& last)   and   void write_byte(Sink&, uint8_t, bool last)
// (AXI-Stream in policy_top.cpp, std::vector in tb_kernel.cpp). Weights live in
// static storage and persist between calls: load once (mode 0), then infer (mode 1).
//
// shifts_lo = s1 | s2 << 8 | s3 << 16 | s4 << 24, shifts_hi = s5..s8 the same way.
template <class Src, class Sink>
static int policy_kernel(int mode, Src& in, Sink& out, uint32_t shifts_lo, uint32_t shifts_hi) {
  // ---- parameters (on chip, persist across calls) ----
  static int8_t w1[C1][9 * IMG_C];
  static int8_t w2[C2][9 * C1];
  static int8_t w3[C3][9 * C2];
  static int8_t w4[C4][9 * C3];
  static int8_t w5[C5][9 * C4];
  static int8_t w6[FC6_OUT][FC6_IN];
  static int8_t w7[FC7_OUT][FC6_OUT];
  static int8_t w8[OUT_LEN][FC7_OUT];
  static int32_t b1[C1], b2[C2], b3[C3], b4[C4], b5[C5], b6[FC6_OUT], b7[FC7_OUT], b8[OUT_LEN];
  // FC6 (297 KB) and conv5 (111 KB) go to UltraRAM, packed 8 weights per 64-bit word.
  // Unpacked, FC6 alone would need 73 URAMs (there are 64) or 73 BRAM36s.
#pragma HLS BIND_STORAGE variable = w6 type = ram_1p impl = uram
#pragma HLS ARRAY_RESHAPE variable = w6 type = cyclic factor = 8 dim = 2
#pragma HLS BIND_STORAGE variable = w5 type = ram_1p impl = uram
#pragma HLS ARRAY_RESHAPE variable = w5 type = cyclic factor = 8 dim = 2

  // ---- activations, HWC ----
  static int8_t a0[IMG_BYTES];
  static int8_t aux[AUX_LEN];
  static int8_t a1[48 * 48 * C1];
  static int8_t a2[24 * 24 * C2];
  static int8_t a3[12 * 12 * C3];
  static int8_t a4[6 * 6 * C4];
  static int8_t a5[FLAT];
  static int8_t f6in[FC6_IN];
  static int8_t a6[FC6_OUT], a7[FC7_OUT], a8[OUT_LEN];

  if (mode == MODE_LOAD_WEIGHTS) {
    int n = 0;
    bool ok = load_layer<C1, 9 * IMG_C>(in, w1, b1, n, WEIGHT_FILE_BYTES);
    ok &= load_layer<C2, 9 * C1>(in, w2, b2, n, WEIGHT_FILE_BYTES);
    ok &= load_layer<C3, 9 * C2>(in, w3, b3, n, WEIGHT_FILE_BYTES);
    ok &= load_layer<C4, 9 * C3>(in, w4, b4, n, WEIGHT_FILE_BYTES);
    ok &= load_layer<C5, 9 * C4>(in, w5, b5, n, WEIGHT_FILE_BYTES);
    ok &= load_layer<FC6_OUT, FC6_IN>(in, w6, b6, n, WEIGHT_FILE_BYTES);
    ok &= load_layer<FC7_OUT, FC6_OUT>(in, w7, b7, n, WEIGHT_FILE_BYTES);
    ok &= load_layer<OUT_LEN, FC7_OUT>(in, w8, b8, n, WEIGHT_FILE_BYTES);
    return ok ? OK : BAD_TLAST;
  }
  if (mode != MODE_INFER) return BAD_MODE;

  int s[N_LAYERS];
  bool shifts_ok = true;
  for (int k = 0; k < N_LAYERS; k++) {
    s[k] = ((k < 4 ? shifts_lo : shifts_hi) >> (8 * (k % 4))) & 0xff;
    shifts_ok &= s[k] >= 1 && s[k] <= 30;
  }

  // Always consume the whole packet so the stream stays in sync, even on error.
  bool tlast_ok = true;
read_in:
  for (int i = 0; i < PACKET_BYTES; i++) {
#pragma HLS PIPELINE II = 1
    bool last;
    uint8_t v = read_byte(in, last);
    tlast_ok &= last == (i == PACKET_BYTES - 1);
    if (i < IMG_BYTES)
      a0[i] = (int8_t)((int)v - 128);  // pixel - 128; same as flipping the top bit
    else
      aux[i - IMG_BYTES] = (int8_t)v;
  }
  if (!shifts_ok) {  // still send 48 bytes so the receive DMA completes
  write_zeros:
    for (int i = 0; i < OUT_LEN; i++) write_byte(out, 0, i == OUT_LEN - 1);
    return BAD_SHIFT;
  }

  conv3x3s2<96, 96, IMG_C, C1>(a0, a1, w1, b1, s[0]);
  debug_dump(1, a1, sizeof(a1));
  conv3x3s2<48, 48, C1, C2>(a1, a2, w2, b2, s[1]);
  debug_dump(2, a2, sizeof(a2));
  conv3x3s2<24, 24, C2, C3>(a2, a3, w3, b3, s[2]);
  debug_dump(3, a3, sizeof(a3));
  conv3x3s2<12, 12, C3, C4>(a3, a4, w4, b4, s[3]);
  debug_dump(4, a4, sizeof(a4));
  conv3x3s2<6, 6, C4, C5>(a4, a5, w5, b5, s[4]);
  debug_dump(5, a5, sizeof(a5));

concat:
  for (int i = 0; i < FC6_IN; i++) {  // HWC flatten is a5 as-is, then the 10 aux bytes
#pragma HLS PIPELINE II = 1
    f6in[i] = i < FLAT ? a5[i] : aux[i - FLAT];
  }
  fc<FC6_IN, FC6_OUT>(f6in, a6, w6, b6, s[5], true);
  debug_dump(6, a6, sizeof(a6));
  fc<FC6_OUT, FC7_OUT>(a6, a7, w7, b7, s[6], true);
  debug_dump(7, a7, sizeof(a7));
  fc<FC7_OUT, OUT_LEN>(a7, a8, w8, b8, s[7], false);
  debug_dump(8, a8, sizeof(a8));

write_out:
  for (int i = 0; i < OUT_LEN; i++) {
#pragma HLS PIPELINE II = 1
    write_byte(out, (uint8_t)a8[i], i == OUT_LEN - 1);
  }
  return tlast_ok ? OK : BAD_TLAST;
}

}  // namespace policy
