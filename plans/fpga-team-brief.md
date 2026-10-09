# FPGA brief: tiny policy accelerator

For the hardware side of the robot arm project. Everything you need to start without waiting on the software.

## 1. The short version

Praneeth is training a small neural network that looks at a 96x96 camera image and outputs joint targets for a robot arm. We need that network to run in the FPGA fabric, fast and at low power.

- **Input:** 27,658 bytes (image + arm state + instruction)
- **Output:** 48 bytes (8 future time steps x 6 joints)
- **Work per inference:** about 9.67 million multiply-adds, all int8 x int8
- **Weights:** 565,552 parameters, about 0.57 MB at int8
- **Target:** under 5 ms per inference end to end, with 33 ms as the hard ceiling

The software side ships a bit-exact Python reference and golden test vectors. Hardware is done when it matches those vectors byte for byte.

## 2. Board (please confirm)

We are assuming a Kria **KV260 or KR260**. Both use the K26 module, so the design is the same on either:

| Resource | K26 |
|---|---|
| CPU | 4x Cortex-A53 at 1.5 GHz, 2x Cortex-R5F |
| RAM | 4 GB DDR4 |
| Logic cells | 256K |
| DSP slices | 1,248 |
| Block RAM | 144 blocks (about 5.1 Mb) |
| UltraRAM | 64 blocks (about 18 Mb) |

Block RAM plus UltraRAM is roughly 2.9 MB, so the full weight set (0.57 MB) fits on chip with room for line buffers and activations. No DDR streaming of weights needed.

**If the board is something else, tell Praneeth right away.** On a small Artix-7 the network has to shrink and there is no ARM CPU to lean on.

## 3. How it fits together

```
 ARM CPU (Linux, Python/PYNQ)                 FPGA fabric
 ┌───────────────────────────┐   AXI-Stream   ┌──────────────────────────────┐
 │ camera + arm state        │ ── via DMA ──► │ in FIFO                      │
 │ pack 27,658 bytes         │                │ conv1 → conv2 → … → conv5    │
 │                           │                │ flatten + concat aux         │
 │ unpack 48 bytes           │ ◄── via DMA ── │ fc6 → fc7 → fc8 → out FIFO   │
 │ control loop at 30 Hz     │                │                              │
 │                           │ ── AXI-Lite ─► │ control/status registers     │
 └───────────────────────────┘                └──────────────────────────────┘
```

Suggested register map (AXI-Lite): `CTRL` (start, soft reset), `STATUS` (busy, done, error), `CYCLES` (clock cycles for the last inference), `VERSION`.

## 4. The network

All convs are 3x3, stride 2, padding 1, followed by ReLU.

| # | Layer | In | Out | Weights | Multiply-adds | Terms per output |
|---|---|---|---|---|---|---|
| 1 | conv | 96x96x3 | 48x48x16 | 432 | 995,328 | 27 |
| 2 | conv | 48x48x16 | 24x24x32 | 4,608 | 2,654,208 | 144 |
| 3 | conv | 24x24x32 | 12x12x64 | 18,432 | 2,654,208 | 288 |
| 4 | conv | 12x12x64 | 6x6x96 | 55,296 | 1,990,656 | 576 |
| 5 | conv | 6x6x96 | 3x3x128 | 110,592 | 995,328 | 864 |
| – | flatten + concat aux | 1,152 + 10 | 1,162 | – | – | – |
| 6 | FC + ReLU | 1,162 | 256 | 297,472 | 297,472 | 1,162 |
| 7 | FC + ReLU | 256 | 256 | 65,536 | 65,536 | 256 |
| 8 | FC | 256 | 48 | 12,288 | 12,288 | 256 |

96% of the multiply-adds are in the convs. Over half the weights are in FC6. So the convs decide your speed and FC6 decides your memory.

## 5. Number format (the contract)

Match this exactly. The Python file `ref/intref.py` is the tiebreaker.

- **Input packet:** bytes 0..27,647 are the image, uint8, HWC order (row, column, channel), row-major. Bytes 27,648..27,657 are aux, int8: 6 joint values then a 4-wide one-hot (hot = 127).
- **Image prep:** `x = pixel - 128` as int8. Do this in hardware.
- **Weights:** int8, range [-127, 127]. Conv layout `[out_ch][ky][kx][in_ch]`. FC layout `[out][in]`.
- **Biases:** int32, already scaled to the accumulator.
- **Accumulator:** int32 is safe. Worst case sum is 1,162 x 127 x 127, about 18.7 million, which fits in 26 bits signed. You can narrow it per layer if you want the area back.
- **Padding:** zeros (after the -128 offset, so the padded value is 0, not -128).
- **Requantize:** one shift per layer, no multiplier.
  `y = (acc + (1 << (s - 1))) >> s` using an arithmetic right shift.
- **Clamp:** ReLU layers saturate to [0, 127]. The final layer saturates to [-127, 127].
- **Flatten:** HWC order, so index = `(row * 3 + col) * 128 + channel`, then the 10 aux bytes appended.
- **Output packet:** 48 int8, index = `step * 6 + joint`.

Each layer's shift `s` comes from `manifest.json`. Make it a parameter, not a hard-coded constant, because it changes every time the model is retrained.

## 6. What you get from software

| File | Contents |
|---|---|
| `weights.bin` | all layers in order, weights (int8) then biases (int32 little-endian) per layer |
| `manifest.json` | per layer: type, shapes, shift, byte offsets, plus a SHA-256 of `weights.bin` |
| `vectors/NNN_in.bin` | 27,658-byte inputs, 100 of them |
| `vectors/NNN_out.bin` | the expected 48-byte outputs |
| `vectors/NNN_layerK.bin` | every layer's output for the first 5 vectors, for layer-by-layer debugging |
| `ref/intref.py` | the numpy reference, readable, no dependencies beyond numpy |

Until trained weights exist, you will get random weights with the same shapes. The hardware does not care what the weights mean.

## 7. Milestones

Each one has a clear pass condition. Do them in order.

**M0. Board bring-up.** Flash the Ubuntu image, install Kria-PYNQ (`sudo bash install.sh -b KV260` or `KR260`), run the PYNQ hello-world overlay from a notebook.
Pass: an overlay loads and runs from Python.

**M1. Loopback.** AXI DMA with a stream FIFO in the fabric. Send 27,658 bytes, get them back.
Pass: bytes match, and you have a measured round-trip time. This number is the floor for our latency, so write it down.

**M2. One FC layer.** FC8 (256 → 48) first because it is the smallest, then FC6. Weights in block RAM, the requantize and clamp logic from section 5.
Pass: matches `layer8` vectors byte for byte.

**M3. One conv layer.** Conv1 with a line buffer (two rows of storage, sliding 3x3 window, stride 2).
Pass: matches `layer1` vectors.

**M4. Full network.** Chain all eight layers.
Pass: all 100 output vectors match. Report cycles per inference and resource use (LUT, FF, DSP, BRAM, URAM).

**M5. Make it fast.** See section 8.
Pass: under 5 ms end to end from Python.

**M6. Measure power.** Board power idle and while running inference in a loop. `xmutil platformstats` on the board is the first thing to try. A USB or inline power meter on the supply is a good cross-check.
Pass: watts, inferences per second, and millijoules per inference.

## 8. Design choices (yours to make)

**HLS first, RTL where it matters.** Getting M4 working in Vitis HLS is the fastest route to a correct design. Then hand-write the conv engine in RTL if you want the speed and the learning.

**How much parallelism.** With about 9.67M multiply-adds:

| Parallel multipliers | Clock | Ideal time |
|---|---|---|
| 1 | 100 MHz | 97 ms |
| 16 | 100 MHz | 6.0 ms |
| 64 | 100 MHz | 1.5 ms |
| 256 | 100 MHz | 0.4 ms |

Real numbers will be worse because of memory access and control overhead. 64 multipliers is a sensible first target and uses 5% of the DSP slices. You can pack two int8 multiplies into one DSP48E2 if you want to go further.

**Two architectures to pick between:**

1. *Dataflow pipeline.* One hardware block per layer, streaming into the next. Fastest and simplest to verify. The network shape is baked in.
2. *Layer engine.* One conv/FC datapath, driven by a small table of layer descriptors (type, sizes, shift, weight address). Slower, but reprogrammable from software without a new bitstream.

Recommendation: do (1) to hit M4, then build (2) as the "our own GPU" track. A layer engine with a descriptor table is a small programmable tensor processor, which is the part of a GPU that matters for this workload. It also means Praneeth can change the network without waiting on synthesis.

**Things that will bite:**
- Stride-2 convs with padding 1: the window is centered on even rows and columns, starting at (0,0) with one row and one column of zero padding on the top and left only. Check against `layer1` vectors early.
- Rounding: the `+ (1 << (s - 1))` before the shift. Leaving it out gives outputs that are off by one.
- Arithmetic vs logical shift on negative accumulators.
- Byte order of the int32 biases.
- DMA transfer overhead can be larger than the compute. Measure M1 before optimizing anything.

## 9. What we report at the end

| Metric | FPGA fabric | ARM CPU on the same board | Laptop GPU |
|---|---|---|---|
| Latency p50 / p95 (ms) | | | |
| Power (W) | | | |
| Energy per inference (mJ) | | | |
| Robot task success (%) | | | |

The ARM CPU row matters. Beating the board's own CPU is the honest claim for the fabric.

## 10. Questions back to you

1. Which board exactly, and which Vivado/Vitis version do you have installed?
2. HLS or RTL for the first pass?
3. Who owns the compute datapath and who owns memory, DMA and board integration?
4. Any limit on how often you can rebuild a bitstream (machine time)?
