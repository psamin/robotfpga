# Attention under the integration contract (`fpga/kernels/attention`)

Status 2026-11-21, branch `fpga/int16-engine`, contract 1.0 (`fpga/kernels/CONTRACT.md`). Host runs only; nothing
here ran on the board, in co-simulation or in Vivado.

## 1. What attention is in the contract build

**Attention has no unit and no mode function of its own.** CONTRACT.md 5 maps it onto three units. Every record
goes through the sequencer's one dispatch switch to the unit that owns its opcode:

| step (per site, per KV head j = one GEMM item) | record | unit (mode function) |
|---|---|---|
| Q head-major `Q[h][TqP][64]` int16, per-(head, query) scale with p4 = -3 (the 1/8) | QUANT_X (vision) / ROPE + F_QOUT (LM, expert) | row unit (`qz::row_mode`, `quantize/quant_mode.h`; ROPE: `rope/rope_mode.h`) |
| K blob, KW tiling, per-key scale, Mk = pad32(keys) | QUANT_W (vision, LM odd layers, cross K') / ROPE + F_KW | row unit |
| scores = Q.K^T, already x 1/8 | GEMM, M = Mk, K = 64, N = 3 TqP, B = Hkv | engine |
| P int16 + 1/sum scale, int32 rank masks | SOFTMAX (+ F_MASK) | softmax unit (`smx::softmax_mode`, `softmax/softmax_core.h`) |
| V^T blob, VW tiling, per-channel scale (expert self: a prefix and a suffix blob) | QUANT_WT | row unit |
| O = P.V into `[Tq][H 64]` through the C row groups (R = TqP, P1 = 256 B); expert self: suffix part with F_ACC | GEMM | engine |
| o_proj input | QUANT_X | row unit |

So the attention-specific hardware is **0 LUT, 0 DSP**. Its costs sit in the engine, the row unit (cap 15k HLS
LUT for all 8 row opcodes) and the softmax unit (cap 8k). The previous plan, a separate packer (`attn_pack`), and
the first conform attempt, a separate QUANT_W / QUANT_WT mode `attn::quant_kv`, are in `retired/`. The csynth of
`attn::quant_kv` at 4.0 ns gave 26,241 LUT / 12 DSP / 16 BRAM18, 2.92 ns, every loop II = 1. That is 1.75 times the
whole row unit's cap. And the row unit already runs both opcodes bit-exactly on every attention command (section 3).

This directory owns:

- the C5 mapping as data and as a tested chain (`make_vectors.py`, `tb_attn.cpp`);
- the exact torch emulation used by the full-policy check (`attn_int16.attention_contract`, `check_policy.py`);
- the attention share of the chunk time (`attn_cost.py`).

## 2. Files

| file | role |
|---|---|
| `attn_int16.py` | torch emulation of the contract arithmetic for one attention call: quantizer (`quant_c` = `contract.quant_rows`), GEMM epilogue with sign-keeping FTZ (= `contract.gemm_out`), smx_emu P, rank masks (`contract_ranks`, checked against SmolVLA's mask on every call), key parts and blob rows (`site_geometry`); the pre-contract `attention()` for comparison |
| `check_policy.py` | full policy on -12392's 3 replay frames with every attention re-routed (variants below); `--dump` saves one real call per site (`vectors/calls.pt`) |
| `make_vectors.py` | contract records + DDR image + expected bytes per site (vision, prefix, expert_self, expert_cross) and an adversarial case (`edge`), at 40-bit bases up to 0xFF_C000_0000; expectations from contract.py's numpy reference and smx_emu, asserted bit-equal to `attn_int16.attention_contract` |
| `tb_attn.cpp` | the C5 clause test (section 3) |
| `dfsim/hls_stream.h` | bounded, threaded `hls::stream` that serves both the row unit's (named streams) and the softmax unit's (template depth) deadlock models in one build |
| `run_tests.sh` | builds and runs the four tb variants, logs in `results/` |
| `attn_cost.py` | attention's records in `contract_plan.build()` timed with contract_plan's own model (section 5) |
| `results/` | `tb_attn*.log`, `policy_check.json/.log`, `cost.json`, `hls_kv_final_p40.log` (the retired mode's csynth) |
| `retired/` | `attn_pack*`, `attn_kv*`, the per-call engine model, the old cost model and README (see `retired/README.md`) |

## 3. Tests (`sh fpga/kernels/attention/run_tests.sh`, Vitis MinGW g++ 6.2, clean PATH, `-ffp-contract=off -static`)

`tb_attn` decodes every record, validates it (`ctr::validate` and `ctr::validate_hw`) and dispatches it through a
switch shaped like the sequencer's:

- QUANT_X / QUANT_W / QUANT_WT go to `qz::row_mode`;
- SOFTMAX goes to `smx::softmax_mode`;
- GEMM goes to the contract model (`ctr::ref::gemm_out` with items, row groups, bias / D / F_ACC), or to the
  engine's own `fstream16::linear_int16_core`.

Those are the units' HLS sources, the same code csynth builds. The engine does not take the contract layouts yet
(CONTRACT.md 8: E-2, E-3, E-5), so its variant converts X and Y around the per-call layout.

What it checks:

- **Every intermediate is compared bit for bit**: Q and its scales, KW and its scales, VW and its scales, the
  scores, P and its scales (the softmax unit against smx_emu, the first time the softmax hardware code runs between
  the two GEMMs), O, the o_proj input and its scales.
- **Extents.** A row op's whole image must equal the reference row op's. A GEMM or SOFTMAX may change no word
  outside its declared outputs.
- **`--unit`**: 23 adversarial QUANT_X / QUANT_W / QUANT_WT commands plus 3 non-finite cases against a reference
  written from the contract text. The whole image, 1 MB guard zones included, is compared. The inputs mix zero,
  -0, denormal, 1e-32, 3e38, ties, outliers and wide exponents; shifts -8..+8; items 1-3; pitch and segment gaps;
  bases 0x4000_0000 / 0x8_0000_0000 / 0xFF_0000_0000.

Results, 2026-11-21. Units as of quant_mode.h sha256 edeb1bad…, softmax_core.h 8538c50d…, smx_emu.py a335642f…,
contract.h 70a114fe…, int16_linear.h 4a264c70…. These files belong to other agents and changed during the day. An
earlier pass on quant_mode.h 48950d5f… / softmax_core.h 42676228… gave the same result. The row unit's stream set
changed between the two: the dfsim depth table in tb_attn.cpp follows quant_mode.h's QZ_D_* macros, and a new
`qz_*` stream without a depth stops the run (exit 8).

| build | --unit | vision | prefix | expert_self | expert_cross | edge | result |
|---|---|---|---|---|---|---|---|
| `tb_attn` (units' HLS code, contract GEMM model) | 26 cases, 0 failures | 12/12 | 12/12 | 14/14 | 12/12 | 12/12 | **ALL PASS** |
| `tb_attn_real` (GEMM = engine core) | same | 12/12 | 12/12 | 14/14 | 12/12 | 12/12 | **ALL PASS** |
| `tb_attn_dfsim` (row unit and softmax unit threaded, FIFOs at the pragma depths) | same | 12/12 | 12/12 | 14/14 | 12/12 | 12/12 | **ALL PASS**, no deadlock |
| `tb_attn_models` (reference row ops, smx_emu bits) | same | 12/12 | 12/12 | 14/14 | 12/12 | 12/12 | **ALL PASS** |

- **Edge case.** The scores hold 168 expected -0.0 values, from tiny Q rows against tiny keys underflowing. They
  match bit for bit, so review defect 15 (ftz to +0) is fixed and tested.
- **Negative controls.**
  - One flipped bit in the expected scores, and one in the expected P, gives FAIL on exactly those 2 checks.
  - `-DSMX_XS_DEPTH=8` makes the threaded model report `DEADLOCK: write blocked on smx_xs` on the vision case.
- **Contract gap, reported to the row-unit owner, not used by attention.** `qz::row_mode` returns ME_DIMS (nothing
  touched) for QUANT_W with p1 = 0 and K != 64, for example K = 200. `ctr::validate` accepts that command, so a
  validated list could stop with MODE_ERROR. contract_plan never emits one: every QUANT_W there has p1 = 64.

## 4. Accuracy: full policy, -12392 frames 0 / 3563 / 7489 (`check_policy.py`, gate 0.05)

max |chunk - ref_chunk_raw|. Only the attention is changed; everything else is torch FP32.

| variant | frame 0 | frame 3563 | frame 7489 |
|---|---|---|---|
| LeRobot as loaded | 4.7e-6 | 3.6e-7 | 3.0e-7 |
| explicit FP32 path | 3.6e-6 | 3.6e-7 | 4.2e-7 |
| **contract (int16 Q/K/V^T/P, smx_emu P, rank masks, prefix compacted to 45, FTZ with sign)** | **4.81e-5** | **2.47e-5** | **4.39e-5** |
| contract at vision only | 1.29e-4 | 1.61e-5 | 1.84e-5 |
| contract at the LM prefix only | 1.14e-4 | 1.62e-5 | 1.24e-5 |
| contract at the expert only | 7.52e-5 | 2.27e-5 | 4.01e-5 |
| pre-contract int16, torch FP32 softmax + quantizer for P | 3.18e-4 | 3.37e-5 | 3.33e-5 |
| pre-contract int16 with smx P | 5.90e-5 | 2.69e-5 | 4.57e-5 |
| BF16 operands, FP32 sums | 1.96e-3 | 5.31e-4 | 8.96e-4 |

- **Rank masks.** The contract's int32 rank vectors reproduce SmolVLA's mask on all 1,056 masked calls checked.
- **Per call**, contract variant, worst over the frames: max |err| / max |ref| and RMS relative error.
  - vision: 7.0e-4 / 2.9e-4.
  - prefix: 2.5e-4 / 1.0e-4.
  - expert self: 4.9e-4 / 1.6e-4.
  - expert cross: 3.0e-4 / 1.0e-4.
- **Frame 0** amplifies small perturbations, as in every earlier study. Per site, the single-site errors are not
  additive.

## 5. Time per chunk (`attn_cost.py`: contract_plan's model, 250 MHz, DDR 14 GB/s; estimates)

| role | records | items | ms |
|---|---|---|---|
| GEMM Q.K^T | 187 | 1,163 | 38.3 |
| SOFTMAX | 187 | 1,163 | 34.2 |
| GEMM P.V | 279 | 1,563 | 32.9 |
| Q producer (QUANT_X / ROPE with p4 = -3, the rotation included) | 187 | 187 | 13.9 |
| o_proj input (QUANT_X) | 187 | 187 | 13.7 |
| V^T blob (QUANT_WT) | 115 | 127 | 13.3 |
| K blob (QUANT_W / ROPE F_KW) | 115 | 115 | 7.0 |
| **attention total** | | | **153.4 of F = 724.3 (21 %)** |

- Other clocks: 147.5 ms with E-7; 172.7 ms at 222 MHz; 191.7 ms at 200 MHz.
- The pre-contract design came to 157 ms at 250 MHz: engine 57.2 (61.8 with the engine's call model), packer 72.4
  (a single-beat burst estimate of 180-290), softmax 27.8. It also needed 948 host calls per chunk. The contract
  list has none.

## 6. Review defects (build/brief/integration_inputs.json, attention review) and their state

| # | defect | state |
|---|---|---|
| 1 | QZ_RECIP_BITS clash between attn_pack and the QUANT mode in one TU | gone: attn_pack retired; tb_attn includes quant_mode.h and softmax_core.h at their defaults (QZ_RECIP_BITS 1) in one TU |
| 2 | 28k LUT; ALLOCATION sharing impossible | no attention unit; QUANT_W / WT run on the row unit's one reader, quantizer and writer. The standalone version measured 26.2k LUT (retired) |
| 3 | no data-path bursts | attention has no DDR datapath; the row unit and softmax own their bursts. The retired quant_kv did infer bursts on all data loops |
| 4 | head batching N[23:16] and the M = 0 mode conflict with the sequencer | B items in the record (E-5); no own mode or descriptor; every record passes validate / validate_hw |
| 5 | version skew of the N encoding | f6 = CONTRACT_VERSION, BAD_VERSION, NO_LEGACY (contract 3.1) |
| 6 | P.V with K = 81 / 131 not % 8; int16 krank against int32 ranks; stale softmax notes | P.V K = pad32(keys of the part) with zero columns; int32 rank vectors only; README rewritten |
| 7 | ftz flushes to +0 in the emulation | `attn_int16.ftz` keeps the sign (copysign); the edge case's 168 -0.0 scores match bit for bit |
| 8 | scale_mul wrap below 2^-27 | p4 limited to ±8 by validate_hw; `--unit` runs shifts -8 / +8 on 1e-32 and denormal rows, bit-exact |
| 9 | rows_per_seg 9-bit truncation | no descriptor; the row unit's counters follow the contract limits (validate_hw) |
| 10 | headline row used torch FP32 softmax for P | the contract variant uses smx_emu P (the softmax unit's bits, checked against `smx::softmax_mode`) |
| 11 | the ropepow2 variant was not RoPE's arithmetic | removed; RoPE uses the contract quantizer on its FP32 rotation (contract 2.1) |
| 12 | timing nit (3.91 ns, negative slack in reader / scaler / writer) | no attention unit; the retired quant_kv estimated 2.92 ns |
| 13 | stale cost numbers (200 clocks per head iteration) | `attn_cost.py` uses contract_plan's model (the engine's L0 400 + tail 450 per item) |

## 7. How to run

```sh
set HF_HUB_OFFLINE=1
C:\kp\lerobot-venv\Scripts\python.exe fpga\kernels\attention\check_policy.py
C:\kp\lerobot-venv\Scripts\python.exe fpga\kernels\attention\check_policy.py --variants fp32 --frames 0 --dump fpga\kernels\attention\vectors\calls.pt
C:\kp\lerobot-venv\Scripts\python.exe fpga\kernels\attention\make_vectors.py
C:\kp\lerobot-venv\Scripts\python.exe fpga\kernels\attention\attn_cost.py
sh fpga/kernels/attention/run_tests.sh
```

## 8. Not verified / open

- **ROPE with F_KW and F_QOUT** (the K and Q producers at the LM and expert sites) is not in this chain. The vectors
  start from post-RoPE FP32 K and use QUANT_W. ROPE's own tb covers its writer.
- **The engine in the contract layouts** (row-major X / Y, items, row groups, bias / D / ACC) does not exist yet. The
  `_real` variant checks only that its arithmetic equals the contract model on these shapes.
- **Real -12392 inputs.** The vectors hold one real call per site (frame 0, layer 0 / step 0), plus the synthetic
  edge case. The full-policy check runs the torch emulation on every call of the 3 frames, and make_vectors asserts
  that emulation equal to the record chain on the dumped calls.
- **The policy check changes only the attention.** The whole-contract run (gate T-P1, golden executor) is not built.
- All times are model estimates: DDR rate, burst efficiency and per-record overheads have never been measured.
- **Environment.** At the time of these runs, the shared HF cache's `refs/main` for SmolVLM2-500M-Video-Instruct
  pointed at a snapshot with only `config.json` and an `.incomplete` blob, left by a download on 2026-11-20 22:55.
  An offline load fails with that cache. The check ran with `HF_HUB_CACHE` set to a private cache whose `refs/main`
  is the complete snapshot 7b375e1b (hard links, no copy). The shared cache was not changed.
