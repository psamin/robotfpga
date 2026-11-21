# Attention through the int16 engine (`fpga/kernels/attention`)

This directory covers every attention of SmolVLA checkpoint -12392 (SigLIP vision, the LM prefix, and the
action expert's self- and cross-attention) as two calls of the int16 linear engine
(`fpga/hls/int16_linear.h`, DRIVER CONTRACT), with the softmax in between. It contains:

- the layouts and descriptors those calls need;
- `attn_pack`, an HLS operand packer that becomes a mode of `float_top`;
- a torch emulation of the exact arithmetic, measured inside the full policy;
- a C testbench that runs the whole chain on real -12392 calls, bit for bit against the emulation and against
  the engine's own C++ core;
- a cost model.

| file | what |
|---|---|
| `attn_pack.h` | header-only HLS core of the packer (modes QX / KW / VW / OX), also compiles with g++ |
| `attn_pack_top.cpp`, `run_attn_hls.tcl` | standalone csynth top and script |
| `attn_engine_model.h` | C model of one head-batched engine call (the DRIVER CONTRACT plus the head loop) |
| `tb_attn.cpp` | host testbench; `-DATTN_REAL_ENGINE` runs `fstream16::linear_int16_core` instead of the model |
| `attn_int16.py` | torch emulation (quantizer, engine, attention), the DDR layouts, descriptor builder |
| `smx_bridge.py` | P from the softmax kernel's own arithmetic (`../softmax/smx_emu.py`, read only) |
| `check_policy.py` | full-policy check on -12392's 3 replay frames (monkeypatches every attention) |
| `make_vectors.py` | DDR images and expected bits from real calls (`vectors/`, git-ignored) |
| `count_cost.py` | engine calls, MACs and time per chunk (`results/cost.json`) |
| `results/` | `policy_check.{json,log}`, `cost.json`, `tb_attn*.log`, `hls_v*_p40.log` (v7 is the final design) |

## 1. Arithmetic

**Quantizer.** The project's shared one (`../quantize/quant_int16.h`, which is `precision_emulation.quant_sym`
plus pack_float's zero-row convention):

```
s = fl32(max(amax, 1e-30) / 32767), or 1.0 for an all-zero row
q = clamp(rne(fl32(x / s)), ±32767)
```

No FP32 core is involved; the header does it with an exact shift/add division and a restoring reciprocal.

**Engine.** `acc = Σ_k Xq[n][k]·Wq[m][k]` is exact (int48), and
`y = ftz(ftz(fl32(acc) · s_row[m]) · s_tok[n])`.

**One attention.** B = 1, H query heads, Hkv KV heads, G = H/Hkv, D = 64. For each KV head j, one head
iteration of each call:

| step | engine W (s_row) | engine X (s_tok) | output |
|---|---|---|---|
| Q Kᵀ | K_j: Tk keys × 64, scale per key; M = pad32(Tk) | the G query heads of KV head j stacked: row t = g·Tq + i, N = G·Tq, scale per (head, query) × 2⁻³ | scores Y[M/32][Np][32] FP32, already × 1/√64 |
| softmax | from the softmax kernel: int16 P ≥ 0 per row with an FP32 row scale | | |
| P V | V_jᵀ: 64 channels × Tk, scale per channel over the keys; M = 64, K = Tk (no padding) | P: N × Tk, row scale | O_j Y[2][Np][32], row t = g·Tq + i |
| o_proj input | | OX gathers row i of head h from O_{h/G}, row (h mod G)·Tq + i, into X: Tq × H·64, one scale per query over all H·64 values | |

- **Q's 1/8.** It is applied as an exact exponent add on s_tok, so the quantized Q is identical to
  quant_sym(Q).
- **GQA without copies.** K/V are never repeated: the G query heads of a KV head share one W by being stacked
  along the engine's token axis.

## 2. Engine call and head batching

`float_top` keeps its registers (w0..w3, xmem, ymem, M, K, N). Attention needs one addition: **H head
iterations per call, carried in the N register's upper bits**:

```
N = n | (H - 1) << 16        (n < 65536; H - 1 = 0 is today's single call)
```

The head strides are implicit and dense, so no stride register is needed. Head h uses:

- port p: `w_p + h·WS`, with WS = M/32·K words;
- xmem: `xmem + h·XS`, with XS = M/4 + Np/4 + Np·K/8 words;
- ymem: `ymem + h·YS`, with YS = M·Np/4 words;
- Np = pad32(n).

The `float_top` change is a loop around today's core, in the int16 branch:

```cpp
const int H = ((N >> 16) & 0xFF) + 1, n = N & 0xFFFF, np = (n + 31) / 32 * 32;
const size_t WS = M / 32 * K, XS = (M + np) / 4 + (size_t)np * K / 8, YS = (size_t)M * np / 4;
for (int h = 0; h < H; h++)
  r |= fstream16::linear_int16_core(w0 + h*WS, w1 + h*WS, w2 + h*WS, w3 + h*WS, xmem + h*XS, ymem + h*YS,
                                    M, K, n, M / 4, (M + np) / 4);
```

Per head, the layouts are exactly the DRIVER CONTRACT:

- **xmem image:** `[s_row: M FP32][s_tok: Np FP32][X: Np/32 blocks][K][32 tokens int16]`. Word
  (b·K + k)·4 + q holds tokens 32b + 8q .. +7 at k.
- **Port p:** `[M/32][K]` words; lane l = W[32g + 8p + l][k].
- **Y:** `[M/32][Np][32]` FP32.

The host allocates the port arrays as one buffer `[p][h][g][k]`, so w_p = base + p·H·WS.

`tb_attn -DATTN_REAL_ENGINE` runs the engine's own `linear_int16_core`, once per head with exactly these
offsets. It is bit-exact on all four sites, so the layouts are what the engine consumes.

### Calls per site (one layer instance; Tk / Tq as run, compacted prefix in brackets)

| site | Q Kᵀ: M × K, N, H | P V: M × K, N, H | instances / chunk |
|---|---|---|---|
| vision (2 images batched) | 256 × 64, 256, 24 | 64 × 256, 256, 24 | 12 |
| LM prefix | 96 [64] × 64, 243 [135], 5 | 64 × 81 [45], 243 [135], 5 | 16 |
| expert self (prefix KV + 50 actions) | 160 [96] × 64, 150, 5 | 64 × 131 [95], 150, 5 | 80 |
| expert cross (prefix KV) | 96 [64] × 64, 150, 5 | 64 × 81 [45], 150, 5 | 80 |

## 3. Masks

**Every SmolVLA mask has a two-vector form.** Each mask is `allowed(i, j) ⇔ krank[j] ≤ qrank[i]`, because
the rows' allowed key sets are nested. `check_policy.py` verifies this on every call: 6,336 masks checked,
0 failed.

**Keys no query may attend are zeroed by attn_pack before quantization.** These are the padding keys, with
krank = 0x7FFF. Masked keys then leave the K/V scales untouched, and nothing NaN can leak.

**Where the mask is applied: in the softmax kernel.** That kernel's options are n_valid / F_CAUSAL with
`causal_off` and `rows_per_head` / F_BITMAP. With the stacked rows, `rows_per_head = Tq`.

| site | prefix compacted to its valid tokens (45) | as run (81 tokens with padding holes) |
|---|---|---|
| vision | none (n_valid = 256) | same |
| prefix | bitmap: rows 0..43 see keys 0..43, the state row sees 0..44 (2 distinct rows, repeated per head) | bitmap |
| expert self | **causal**: n_valid = 95, causal_off = 45 (all prefix keys + j ≤ i in the suffix) | bitmap |
| expert cross | n_valid = 45 | bitmap (padding holes) |

**Compacting the prefix is recommended.** It means dropping the 36 padding language tokens before the LM.

- Valid tokens keep their RoPE positions (position_ids are the cumsum of the pad mask).
- On the attention side the result equals the full computation up to the last bit of P: `int16_compact`,
  528 calls, max |diff| 3.7e-5 = 1.2 LSB of P. This comes from torch's FP32 softmax summing a different
  number of zeros.
- It cuts the prefix and expert MACs by 30-60 %.

## 4. attn_pack (the packer, a future mode of float_top)

`attn_pack(src, dst, desc_off, n_desc)` runs n_desc consecutive 24-word descriptors.

- src and dst are the same DDR space over one 128-bit `io` bundle: reads on src (= xmem), writes on dst
  (= ymem). All offsets are 128-bit word offsets.
- It returns 0, or 16·index + error code (codes in `check_desc`).

| mode | input rows | output | scale |
|---|---|---|---|
| QX 0 | Q rows (stacked t = g·Tq + i per image j) | X section of the Q Kᵀ xmem image j, 32-row transposed tiles | s_tok = row scale × scale_mul (2⁻³) |
| KW 1 | K rows (keys), padding / rank-0x7FFF keys → 0 | the 4 port arrays of image j: 32 keys per group, 8 per word | s_row into the Q Kᵀ image's s_row section |
| VW 2 | V rows (keys) | port c%4, group c/4, word = key, chunk c = channels 8c..8c+7 (no transpose) | 64 channel scales over the keys → s_row of the P V image |
| OX 3 | P V outputs, H segments per query (segment table) | X section of the o_proj image (K = H·64) | s_tok per query over all segments |

### Descriptor (96 × uint32 = 24 words)

| u[i] | field | meaning |
|---|---|---|
| 0 | mode | 0 QX, 1 KW, 2 VW, 3 OX |
| 1 | n_img | images (KV heads) |
| 2 | n_rows | rows with data per image (QX G·Tq, KW/VW Tk, OX Tq) |
| 3 | n_out | output rows incl. padding (QX/KW/OX: multiple of 32, ≤ 256; VW: = n_rows) |
| 4 | rows_per_seg (rps) | segment h of output row t = j·spi + ⌊t/rps⌋ (+ s in OX), source row q = t mod rps; rps ≥ 32 or = n_rows |
| 5 | segs_per_img (spi) | QX G, KW/VW 1, OX 0 |
| 6 | nseg | segments per row (OX H ≤ 16, else 1) |
| 7 | scale_mul | FP32 bits of a power of two in [2⁻²⁷, 1] (QX: 0.125) |
| 8 | krank_off | word offset of the int16 key ranks (8 per word), 0xFFFFFFFF = none |
| 9 | n_split | source rows q ≥ n_split come from region 1, as q − n_split (expert self: prefix cache + suffix) |
| 10..12, 13..15 | region 0 / 1: off, row_stride, half_stride | FP32 word w (0..15) of segment h, row q: off + segtab[h] + q·row_stride + ⌊w/8⌋·half_stride + w mod 8 |
| 16..20 | out_off, out_img, out_q, sc_off, sc_img | output base of image 0, image stride, port stride (KW/VW; X modes 1), scale base of image 0, scale image stride |
| 32..63, 64..95 | segtab0[32], segtab1[32] | segment offsets in region 0 / 1 |

Source layouts are given by the region fields:

- engine output Y[2·nseg][Np][32]: row_stride 8, half_stride Np·8, segtab[h] = 2h·Np·8;
- row-major FP32 (the KV cache): row_stride 16·nseg, half_stride 8, segtab[h] = 16h.

`attn_int16.desc` / `region_*` build descriptors; `make_vectors.py` has every site's set.

### Data path and throughput

```
[pre-pass (OX, VW): amax over DDR] -> DATAFLOW: reader -> stats -> scaler -> quant (4 lanes) -> writer
```

- One input word (4 FP32) per clock, II = 1 everywhere.
- QX / KW read their data once (4 values/clock). OX / VW read twice (2 values/clock).
- One row_scale unit (the scaler: 2 quotient bits per clock, 14 clocks per scale) serves all modes.
- One 32 × 64 int16 transposer (8 LUTRAM banks) serves all modes.
- Fixed costs: per descriptor ≈ 340 clocks (read, 256-entry valid load, fill); VW adds 64 × 14 clocks for
  its channel scales.

### Resources (Vitis HLS 2022.2 csynth, xck26-sfvc784-2LV-c, 4.0 ns; `results/hls_v7_p40.log`)

**Total `attn_pack_top`: 28,252 LUT, 24,968 FF, 50 DSP, 13 BRAM18 (6.5 BRAM36), 0 URAM.** Estimated Fmax is
255.7 MHz, all loops II = 1 (read_desc II = 2, 24 words). 3.333 ns was not attempted: the writer's
estimated path is 3.91 ns.

| part | LUT | FF | DSP | BRAM18 |
|---|---|---|---|---|
| quant: 4 lanes of `qz::quant_elem` | 5,507 | 4,524 | 18 | 0 |
| writer: transposer, address generation | 6,414 | 5,096 | 9 | 0 |
| scaler: `row_scale`, iterative | 2,064 | 954 | 3 | 0 |
| reader + stats + prepass + read_desc + valid load | ≈ 6,000 | | 17 | |
| FIFOs (q 128 deep in BRAM, others SRL/LUTRAM) | | | | 8 |
| io m_axi + s_axi_control (exist once in float_top) | ≈ 3,600 | ≈ 1,100 | 0 | |

This is over the share a single kernel can take of the 35-45k LUT budget. Two things bring it down in
float_top:

1. **Share the quantizer and scaler with the QUANT mode.** Both are the same `qz::` functions, so use
   `#pragma HLS ALLOCATION function` / one shared sub-function instead of two copies. That saves ≈ 7.5k LUT
   and 21 DSP.
2. **Let the RoPE kernel emit Q/K int16 directly in the QX/KW layouts** for the LM sites. Its MODE_INT16
   already quantizes; `int16_ropepow2` shows its power-of-two scales cost nothing measurable. This removes
   pack time, not area.

The incremental attention-specific logic is then ≈ 15-17k LUT (address generators, transposer, descriptor),
by csynth estimates. csynth LUT figures usually exceed post-implementation; that is not measured here.

## 5. Accuracy

Full policy on -12392's 3 replay frames. The metric is max |chunk − ref_chunk_raw| (normalized actions;
gate 0.05). Every attention is re-routed through `attn_int16.attention`, everything else stays FP32.
`results/policy_check.json` has per-site detail.

| variant | frame 0 | frame 3563 | frame 7489 |
|---|---|---|---|
| LeRobot as loaded (laptop vs cluster) | 4.7e-6 | 3.6e-7 | 3.0e-7 |
| same path, FP32 operands | 3.6e-6 | 3.6e-7 | 4.2e-7 |
| **int16 Q/K/V/P, exact sums, FP32 softmax (the design)** | **3.2e-4** | **3.4e-5** | **3.3e-5** |
| int16, P from the softmax kernel's arithmetic (`int16_smx`) | 5.9e-5 | 2.7e-5 | 4.6e-5 |
| int16, Q/K power-of-two scales on LM sites (RoPE kernel QM = 0) | 6.3e-5 | 3.4e-5 | 1.7e-5 |
| int16, prefix compacted | 1.7e-4 | 3.2e-5 | 2.9e-5 |
| int16 Q Kᵀ only / P V only | 1.9e-4 / 4.7e-4 | 6.1e-6 / 2.6e-5 | 1.1e-5 / 2.1e-5 |
| int16 at vision / prefix / expert only | 8.7e-5 / 2.1e-4 / 5.7e-5 | 1.3e-5 / 1.2e-5 / 1.2e-5 | 7.6e-6 / 1.8e-5 / 1.3e-5 |
| int16, all scales powers of two | 1.0e-3 | 2.8e-5 | 2.7e-5 |
| **BF16 Q/K/V/P, FP32 sums** (comparison) | **2.0e-3** | **5.3e-4** | **9.0e-4** |

**Per call** (attention output vs FP32 attention, valid rows), max |diff| / max |ref|, worst over each
frame's calls:

| site | max \|diff\| / max \|ref\| | RMS rel |
|---|---|---|
| vision | 2.7-3.7e-4 | 2.6-2.9e-4 |
| prefix | 1.6-2.6e-4 | 1.0e-4 |
| expert self | 3.9-4.9e-4 | 1.3-1.5e-4 |
| expert cross | 1.8-2.0e-4 | 6.5-7.5e-5 |

**Frame 0 amplifies tiny perturbations** to the 1e-4 level: int16 variants spread over 4e-5 .. 4.7e-4 on it
for changes of one LSB. Frames 3563 / 7489 sit at 2-5e-5. Everything is 100-1500x under the gate and 3-30x
under BF16 operands.

## 6. Cost per 50-action chunk (`count_cost.py`, 1,024-MAC engine at 250 MHz)

| | prefix as run (81) | prefix compacted (45) |
|---|---|---|
| useful MACs (Q Kᵀ + P V) | 4.25 G | 3.55 G |
| ideal time (useful MACs / 1024 / 250 MHz) | 16.6 ms | 13.9 ms |
| engine MAC loop (padded blocks) | 17.8 ms | 14.6 ms |
| engine model (Y-store bound: 4 FP32/clock; + 200 clocks per head iteration, assumed) | 66.1 ms | 57.2 ms |
| Y words written (scores + outputs) | 12.9 M | 10.9 M |
| attn_pack (II = 1 loops, DDR efficiency not modelled) | 78.2 ms | 72.4 ms |
| softmax at 4 elements/clock (its kernel's rate) | 33.2 ms | 27.8 ms |
| engine calls (head-batched) / head iterations | 376 / 2,336 | 376 / 2,336 |
| all calls (376 engine + 188 softmax + 384 pack) | 948: 95 ms at 0.1 ms/call, 19 ms at 0.02 ms/call | same |

**Every attention call has K ≤ 256, so it is bound by the 4-FP32/clock Y store, not the MACs.** Without head
batching there would be 2,336 engine calls (234 ms at 0.1 ms/call).

## 7. What the neighbours have to provide (integration notes)

- **Softmax kernel.**
  - Input: the scores in the engine's Y layout. A row is M/32 chunks of 8 words with stride Np·8, so it needs
    a group-stride parameter; today it reads rows contiguously with `in_words` stride.
  - Output: P must go straight into the P V xmem image of KV head j, at words [16, 16 + Np/4) for the row
    scales and [16 + Np/4, ...) for the X-interleaved q (32-row transposed tiles, K = Tk). Today it writes
    row-major q.
  - Its masks map as in §3 (rows_per_head = Tq). Its own P format (q ∈ [0, 32767], scale 1/Σq) is fine for
    the engine and measured above as `int16_smx`.
- **RoPE kernel.** Either FP32 output in a region the QX/KW descriptors can read, or int16 straight into
  the QX/KW layouts.
- **Host / sequencer.**
  - Allocate per attention: Hkv xmem images and `[p][h][g][k]` port buffers for each call; the o_proj image's
    s_row section is written once per layer (OX only writes s_tok and X).
  - Pack the cross-attention K/V once per chunk (8 layers); pack the expert-self K/V every step (the per-channel
    V scale spans prefix + suffix).
  - Order: pack(QX, KW, VW) → Q Kᵀ → softmax → P V → pack(OX) → o_proj.
- **float_top mode.** Pick an unused register pattern, for example M = 0 (illegal for a linear) with K = mode id.
  Then N = n_desc, xmem = ymem = the arena base, and the descriptors start at xmem word 0 (desc_off = 0); every
  offset in them is relative to that base. `attn_pack` only touches `io`, so the bus interfaces do not change. It includes
  `../quantize/quant_int16.h` and sets `QZ_RECIP_BITS 2` unless already defined; the scaler needs
  RECIP_STEPS + 1 ≤ 16 (static_assert).

## 8. How to run

```sh
set HF_HUB_OFFLINE=1
C:\kp\lerobot-venv\Scripts\python.exe fpga\kernels\attention\check_policy.py --dump fpga\kernels\attention\vectors\calls.pt
C:\kp\lerobot-venv\Scripts\python.exe fpga\kernels\attention\make_vectors.py
C:\kp\lerobot-venv\Scripts\python.exe fpga\kernels\attention\count_cost.py
# g++ (Vitis MinGW), clean PATH:
export PATH=/c/Xilinx/Vitis_HLS/2022.2/tps/win64/msys64/mingw64/bin:/c/Xilinx/Vitis_HLS/2022.2/tps/win64/msys64/usr/bin:/c/Windows/System32:/c/Windows
g++ -O2 -std=c++14 -ffp-contract=off -I/c/Xilinx/Vitis_HLS/2022.2/include -o tb_attn.exe tb_attn.cpp
./tb_attn.exe vectors/vision vectors/prefix vectors/expert_self vectors/expert_cross
g++ ... -DFSTREAM_ARITH=16 -DATTN_REAL_ENGINE -o build/tb_attn_real.exe tb_attn.cpp   # the engine's own core
# csynth (PATH without Git):  /c/Xilinx/Vitis_HLS/2022.2/bin/vitis_hls.bat -f run_attn_hls.tcl
```

## 9. Not verified

- **No C/RTL co-simulation** (csim/cosim are broken here) and **no board run**.
  - DDR efficiency of the reader's per-row addresses (8-16-word runs) and the KW emit (64-word runs) is not
    measured; csynth reports inferred and missed bursts.
  - Pack times assume one word per clock.
- **The engine model** (Y-store bound, 200 clocks per head iteration) and the per-call host costs (0.1 / 0.02 ms)
  are assumptions.
- **The `float_top` head loop** (N[23:16]) and the attn_pack mode are proposals; neither exists in `fpga/hls/`.
- **csynth LUT/FF are estimates**; the shared-quantizer savings are not synthesised.
- **Prefix compaction** was measured on the attention side only; the LM linears on 45 tokens were not run.
- **Upstream files are in flux.** `quant_int16.h` (quantize agent) and `int16_linear.h` (engine agent) changed
  during this work; results are against their state of 2026-11-20 20:36.
