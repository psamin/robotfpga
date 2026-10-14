// DDR weight-streaming int8 engine (issue #148).
//
// Weights never live on chip as a whole. They stay in DDR and stream in as 128-bit words, each
// holding the same reduction index k for 16 output channels ("lanes"). One group of 16 output
// channels is loaded into an on-chip tile buffer while the previous group computes (ping-pong
// double buffering, as in roofline-tiled FPGA CNN accelerators). Activations stay on chip.
//
// Two entry points:
//   policy_stream  a whole policy from a layer table in DDR (conv3x3/s2 + FC, any widths that are
//                  multiples of 16). Bit-exact with ref/intref.py: int32 accumulate, round-half-up
//                  arithmetic shift, clamp [0,127] for ReLU layers and [-127,127] for the last.
//   linear_stream  Y[n][m] = sum_k X[n][k] * W[m][k] as int32, for offloading transformer linear
//                  layers (SmolVLA). Scales and nonlinear ops are applied by the ARM afterwards.
//
// Plain C++ so the same code builds with g++ (tb_stream.cpp) and with Vitis HLS (stream_top.cpp).
#pragma once
#include <stdint.h>

namespace stream {

constexpr int LANES = 16;           // output channels per 128-bit weight word
constexpr int K_MAX = 4096;         // largest reduction length (9*Cin for conv, Cin for FC)
constexpr int ACT_MAX = 96 * 96 * 8;  // on-chip activation buffer (int8), covers image + any layer
constexpr int AUX_LEN = 10;
constexpr int IMG_BYTES = 96 * 96 * 3;
constexpr int N_TILE = 64;          // tokens kept on chip per pass in linear mode
constexpr int MAX_LAYERS = 16;

struct Word {  // one 128-bit DDR word: weight k for 16 consecutive output channels
  int8_t v[LANES];
};

enum LayerType : int32_t { CONV3S2 = 0, FC = 1 };

struct LayerDesc {  // 16 x int32 = 64 bytes, read from DDR
  int32_t type;     // CONV3S2 or FC
  int32_t h, w, cin;  // input shape (h = w = 1 for FC; cin = input length)
  int32_t cout;     // output channels, multiple of LANES
  int32_t shift;    // requantization shift, 1..30
  int32_t relu;     // 1: clamp [0,127], 0: clamp [-127,127]
  int32_t w_off;    // offset into the weight-word array
  int32_t b_off;    // offset into the bias array (int32)
  int32_t append_aux;  // 1: input is the previous output followed by the 10 aux bytes
  int32_t pad[6];
};

static inline int8_t requant(int32_t acc, int s, bool relu) {
#pragma HLS INLINE
  int32_t y = (acc + (1 << (s - 1))) >> s;  // arithmetic shift on signed int32
  int32_t lo = relu ? 0 : -127;
  if (y < lo) y = lo;
  if (y > 127) y = 127;
  return (int8_t)y;
}

// Burst-read one group's K weight words from DDR into an on-chip tile.
static void load_tile(const Word* wmem, int off, int k_len, Word tile[K_MAX]) {
load_w:
  for (int k = 0; k < k_len; k++) {
#pragma HLS PIPELINE II = 1
    tile[k] = wmem[off + k];
  }
}

// One group of 16 output channels for every output position of a layer.
static void compute_group(const LayerDesc& L, int g, const Word tile[K_MAX], const int32_t* bmem,
                          const int8_t* in, int8_t* out) {
  const bool conv = L.type == CONV3S2;
  const int ho = conv ? L.h / 2 : 1, wo = conv ? L.w / 2 : 1;
  const int k_len = conv ? 9 * L.cin : L.cin;
  int32_t bias[LANES];
#pragma HLS ARRAY_PARTITION variable = bias complete
  for (int l = 0; l < LANES; l++) bias[l] = bmem[L.b_off + g * LANES + l];
pos:
  for (int p = 0; p < ho * wo; p++) {
    int32_t acc[LANES];
#pragma HLS ARRAY_PARTITION variable = acc complete
    for (int l = 0; l < LANES; l++) acc[l] = bias[l];
    const int oy = p / wo, ox = p % wo;
    int ky = 0, kx = 0, ci = 0;
red:
    for (int k = 0; k < k_len; k++) {
#pragma HLS PIPELINE II = 1
      int8_t x;
      if (conv) {  // implicit im2col: 3x3 window, stride 2, zero padding 1, HWC input
        int iy = 2 * oy + ky - 1, ix = 2 * ox + kx - 1;
        bool inside = iy >= 0 && iy < L.h && ix >= 0 && ix < L.w;
        x = inside ? in[(iy * L.w + ix) * L.cin + ci] : (int8_t)0;
        if (++ci == L.cin) {
          ci = 0;
          if (++kx == 3) {
            kx = 0;
            ++ky;
          }
        }
      } else {
        x = in[k];
      }
      const Word wv = tile[k];
      for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
        acc[l] += (int32_t)x * (int32_t)wv.v[l];
      }
    }
    for (int l = 0; l < LANES; l++) out[p * L.cout + g * LANES + l] = requant(acc[l], L.shift, L.relu);
  }
}

// Whole policy. table/wmem/bmem/pkt/out are DDR buffers; activations ping-pong on chip.
// Returns 0 on success, 1 for a bad layer table.
static int policy_stream(const LayerDesc* table, int n_layers, const Word* wmem, const int32_t* bmem,
                         const uint8_t* pkt, int8_t* out, int8_t* layer_dump = nullptr) {
  static int8_t act[2][ACT_MAX];
  static Word tile[2][K_MAX];
#pragma HLS BIND_STORAGE variable = act type = ram_2p impl = uram
  if (n_layers < 1 || n_layers > MAX_LAYERS) return 1;
  int8_t aux[AUX_LEN];
read_pkt:
  for (int i = 0; i < IMG_BYTES + AUX_LEN; i++) {
#pragma HLS PIPELINE II = 1
    uint8_t v = pkt[i];
    if (i < IMG_BYTES)
      act[0][i] = (int8_t)((int)v - 128);  // pixel - 128
    else
      aux[i - IMG_BYTES] = (int8_t)v;
  }
  int cur = 0, dump_off = 0;
layers:
  for (int li = 0; li < n_layers; li++) {
    LayerDesc L = table[li];
    const bool conv = L.type == CONV3S2;
    const int k_len = conv ? 9 * L.cin : L.cin;
    const int n_out = conv ? (L.h / 2) * (L.w / 2) * L.cout : L.cout;
    if (L.cout % LANES || k_len > K_MAX || L.shift < 1 || L.shift > 30 || n_out > ACT_MAX) return 1;
    if (L.append_aux) {  // FC input = previous output (HWC flatten) + aux bytes
      for (int i = 0; i < AUX_LEN; i++) act[cur][L.cin - AUX_LEN + i] = aux[i];
    }
    const int groups = L.cout / LANES;
    // Ping-pong: while group g computes from tile[g & 1], group g + 1 streams into the other tile.
    load_tile(wmem, L.w_off, k_len, tile[0]);
  groups_loop:
    for (int g = 0; g < groups; g++) {
      if (g + 1 < groups) load_tile(wmem, L.w_off + (g + 1) * k_len, k_len, tile[(g + 1) & 1]);
      compute_group(L, g, tile[g & 1], bmem, act[cur], act[cur ^ 1]);
    }
    cur ^= 1;
    if (layer_dump) {
      for (int i = 0; i < n_out; i++) layer_dump[dump_off + i] = act[cur][i];
      dump_off += n_out;
    }
    if (li == n_layers - 1) {
      for (int i = 0; i < n_out; i++) out[i] = act[cur][i];
    }
  }
  return 0;
}

// Transformer linear layer: Y[n][m] = sum_k X[n][k] * W[m][k], int32 out, no requantization.
// X is [N][K] int8 in DDR; W is packed as groups of 16 rows (one Word per k per group).
// N_TILE tokens stay on chip per pass, so every weight word is read ceil(N / N_TILE) times.
static int linear_stream(const Word* wmem, const int8_t* xmem, int32_t* ymem, int M, int K, int N) {
  static int8_t xbuf[N_TILE][K_MAX];
  static Word tile[2][K_MAX];
  if (M % LANES || K > K_MAX || K < 1 || N < 1) return 1;
  const int groups = M / LANES;
tiles:
  for (int n0 = 0; n0 < N; n0 += N_TILE) {
    const int nt = (N - n0) < N_TILE ? (N - n0) : N_TILE;
  load_x:
    for (int i = 0; i < nt * K; i++) {
#pragma HLS PIPELINE II = 1
      xbuf[i / K][i % K] = xmem[(n0 + i / K) * K + i % K];
    }
    load_tile(wmem, 0, K, tile[0]);
    for (int g = 0; g < groups; g++) {
      if (g + 1 < groups) load_tile(wmem, (g + 1) * K, K, tile[(g + 1) & 1]);
      const Word* t = tile[g & 1];
      for (int n = 0; n < nt; n++) {
        int32_t acc[LANES];
#pragma HLS ARRAY_PARTITION variable = acc complete
        for (int l = 0; l < LANES; l++) acc[l] = 0;
        for (int k = 0; k < K; k++) {
#pragma HLS PIPELINE II = 1
          const int32_t x = xbuf[n][k];
          for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
            acc[l] += x * (int32_t)t[k].v[l];
          }
        }
        for (int l = 0; l < LANES; l++) ymem[(n0 + n) * M + g * LANES + l] = acc[l];
      }
    }
  }
  return 0;
}

}  // namespace stream
