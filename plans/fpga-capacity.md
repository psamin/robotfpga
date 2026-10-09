# FPGA capacity: the largest quantized policy the KR260 can run

For whoever trains the next policy. It gives the exact network to train so it runs on the board's
FPGA, and the limits behind it. Numbers come from the built 16-lane layer engine (#98): Vivado
post-route reports, the HLS report, and on-board timing. Check any other design with
[`hls/engine/capacity.py`](../hls/engine/capacity.py) before training it.

## 1. Answer

| Design | Parameters | FPGA time per inference | Hardware change |
|---|---|---|---|
| v3r2, running now | 565,552 | 8.03 ms measured | none |
| **TinyPolicy-L (section 2), recommended target** | **1,812,528 (3.2×)** | **~26 ms est.** | layer table + rebuild (~45 min), no new logic |
| Hard ceiling for this engine | ~2.0 M (126,976 weight words) | depends on shape | same as above |
| Engine v2: 64 lanes at 150 MHz | same ~2.0 M | ~5 ms est. for TinyPolicy-L | HLS change, not built |
| Engine v3: weights streamed from DDR | ~10–100 M est. | bandwidth-bound | new design, weeks, not built |

The limit is **weight memory, not compute**. The engine keeps every weight on chip in UltraRAM
(64 blocks of 4096 × 72 bit; a 128-bit weight word takes 2 side by side). 2 blocks go to other
buffers, which leaves 31 × 4096 = 126,976 words of 16 int8 weights. Compute has headroom: 21 of
1,248 DSPs are in use and the control loop allows ~33 ms per inference.

A much larger model (tens of millions of parameters) is possible on this chip, but only with an
engine that streams weights from DDR (section 6). That engine does not exist yet. SmolVLA
(~450 M) does not fit at control rates on any of these designs.

## 2. TinyPolicy-L: the exact network to train

Same input, output and layer pattern as v3r2; only the widths change. Every conv is 3×3, stride 2,
padding 1, NHWC; flatten is HWC order; the 10 aux bytes are appended after the flatten.

| # | Layer | In | Out | Weight shape (int8) | Bias (int32) | Weight words | MACs |
|---|---|---|---|---|---|---|---|
| 1 | conv1 + ReLU | 96×96×3 | 48×48×32 | [32, 3, 3, 3] | 32 | 54 | 1,990,656 |
| 2 | conv2 + ReLU | 48×48×32 | 24×24×64 | [64, 3, 3, 32] | 64 | 1,152 | 10,616,832 |
| 3 | conv3 + ReLU | 24×24×64 | 12×12×128 | [128, 3, 3, 64] | 128 | 4,608 | 10,616,832 |
| 4 | conv4 + ReLU | 12×12×128 | 6×6×192 | [192, 3, 3, 128] | 192 | 13,824 | 7,962,624 |
| 5 | conv5 + ReLU | 6×6×192 | 3×3×256 | [256, 3, 3, 192] | 256 | 27,648 | 3,981,312 |
| 6 | fc6 + ReLU | 2,304 + 10 = 2,314 | 384 | [384, 2314] | 384 | 55,536 | 888,576 |
| 7 | fc7 + ReLU | 384 | 384 | [384, 384] | 384 | 9,216 | 147,456 |
| 8 | fc8 (no ReLU) | 384 | 48 = 8 steps × 6 joints | [48, 384] | 48 | 1,152 | 18,432 |
| | **Total** | | | **1,811,040** | **1,488** | **113,190** | **36,222,720** |

- `weights.bin`: 1,816,992 bytes, layers in order, each layer's weights (int8) then biases (int32 LE).
- FPGA memory: 58 of 64 URAM, ~107 of 288 BRAM18, 21 DSP. That's 91% of UltraRAM, the near-capacity point
  with a small margin for the build.
- Input packet unchanged: 27,658 bytes (96×96×3 image + 10 aux). Output unchanged: 48 int8.
- `armlab` shapes: the widths are hard-coded in `armlab/policy/tiny.py` (`TinyPolicy.__init__`) and need
  to become these. Everything else (`pack_obs`,
  `unpack_actions`, chunking, eval, the remote backend, the real-arm controller) is unchanged.

**Variant W, front + wrist camera:** stack the two 96×96 RGB images as 6 input channels (conv1
weight [32, 3, 3, 6]). 1,813,392 parameters, ~27.6 ms, packet 55,306 bytes. That also changes
`pack_obs` and the packet size, so it needs its own sign-off.

## 3. Quantization rules (the engine computes exactly this)

Same as `ref/intref.py` today. The engine has no other modes.

- Weights int8 symmetric, biases int32, **one power-of-two scale per layer** (a shift `s`,
  1 ≤ s ≤ 31). Per-channel scales are not supported.
- Input: `x = pixel − 128` as int8; aux = 6 joints `floor(norm·127 + 0.5)` clamped to ±127, then a
  4-byte instruction one-hot (127 / 0).
- Accumulate in int32: `acc = bias + Σ w·x`. Worst case for TinyPolicy-L is fc6:
  2,314 × 128 × 128 = 37.9 M, far inside int32.
- Requantize: `y = (acc + (1 << (s−1))) >> s` (arithmetic shift), then ReLU clamp to [0, 127]
  for layers 1–7, and [−127, 127] for fc8. Output = fc8 / 127 → [8, 6] normalized joint targets.
- Train with the existing QAT (`armlab/policy/qat.py`, power-of-two scales). `armlab/policy/qat_train.py`
  writes `export/` (weights.bin + manifest.json). Check bit-exactness with `ref/intref.py` as for v3r2.

## 4. Rules any other design must follow

`python hls/engine/capacity.py --conv C1 C2 C3 C4 C5 --fc F [--image-channels 6]` checks all of these.

1. 5 convs (3×3, stride 2, pad 1) then 3 FCs; at most 8 layers (the weight header holds 8 shifts).
2. Every layer's output width is a multiple of 16; fc8 is 48.
3. Σ over layers of (out/16) × terms ≤ 126,976 weight words. Terms = 9·in for a conv, in for an FC.
4. Largest activation (96·96·image channels, or any conv output) ≤ ~150 KB.
5. FPGA time ≲ 30 ms: one inference must finish inside a 33 ms control tick. The checker's
   estimate matches v3r2 (8.1 ms est. vs 8.03 ms measured).

## 5. Training and handoff checklist

1. Train TinyPolicy-L in sim (same data, DAgger, QAT recipe as v3r2). Evaluate with
   `--latency-ms 33` too, because inference now takes most of one tick (~26 ms + ~1 ms network).
2. Export `weights.bin` + `manifest.json` + 100 golden vectors (`ref/vectors.py`) into
   `handoff/<name>/`.
3. FPGA side: set the layer table, `N_W_WORDS` = 113,190, `N_BIAS_WORDS` = 372, `ACT_SIZE` =
   73,728, bias groups = 93 in `hls/engine/policy.cpp` / `policy.h` / `hwpack.py`; rebuild with
   `board/engine/build.bat`; pass 100/100 vectors on the board.
4. Retrained weights after that need no rebuild: `package.py --weights-only` + `deploy.py`.

**Sign-off:** the architecture table in `plans/build-spec.md` and `LAYERS` in `ref/intref.py` are the
shared contract. Adopting TinyPolicy-L means an issue that both sides acknowledge before
either file changes (AGENTS.md).

## 6. Bigger models (not built; estimates only)

- **Engine v2, 64 lanes at 150 MHz.** Same 2 M ceiling (UltraRAM), TinyPolicy-L in ~5 ms. Uses ~80 DSPs.
  Output widths should be multiples of 64 to keep all lanes busy.
- **Engine v3, DDR streaming.** The KR260 has 4 GB DDR4 (~19 GB/s peak). Conv weights stay on chip
  per layer; FC weights stream once per inference. At ~6 GB/s through the PL ports, about 20 M
  FC parameters cost ~3 ms per inference, and conv compute (up to ~2,000 int8 MACs per clock on
  1,248 DSPs) stops being the limit. Practical range ~10–100 M parameters, e.g. a ResNet-18-class
  backbone (~11 M) at 96×96. This is a new accelerator design: weeks of FPGA work, not a rebuild.
- **AMD Vitis AI DPU.** An off-the-shelf int8 CNN accelerator for this chip (~1 TOPS class).
  It handles ResNet-scale CNNs, but uses a different toolchain and quantizer, not this engine.
