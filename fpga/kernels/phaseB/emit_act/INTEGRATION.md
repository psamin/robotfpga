# emit_act: GELU / SiLU(gate) x up inside the GEMM emit (phase B), integration plan

Status: 2026-11-21, branch `fpga/int16-engine`. Owner of this directory: the emit_act agent. Nothing here is
committed, nothing ran in Vivado or on the board. Sources are labelled as in CONTRACT.md: **measured** (run today,
command in section 8), **HLS** (Vitis HLS 2022.2 csynth, xck26-sfvc784-2LV-c), **est.**

## 1. What it does

A GEMM record with the new parameter `p6 = P_ACT` applies the activation to the epilogue's FP32 output before the
store. The separate GELU (12 records) and SILU_MUL (185 records) commands of the -12392 chunk go away.

| P_ACT | output (C) | -12392 use (records per chunk) |
|---|---|---|
| `EPI_GELU` = 1 | `gelu_tanh(y)`, M columns | SigLIP fc1 (+ bias): 12 |
| `EPI_SILU_MUL` = 2 | `silu(y[32i+c]) * y[32i+16+c]` at column `16i+c`, M/2 columns | LM gate_up (15), expert gate_up (160) |
| `EPI_SILU` = 3 | `silu(y)`, M columns | action_time_mlp_in (+ the step's D row): 10 |

`y` is the contract epilogue, unchanged: `((fl(acc) s_row) s_tok) [+ bias] [+ D]`. The activation is the gelu_silu
kernel's front end, bit for bit: h from the NSB 10 ROM (decision 0 of `build/brief/integration_inputs.json`),
`fmul(y, h)`, and for SILU_MUL `fmul(up, silu(gate))`.

**Exactness (measured).** The fused GEMM's FP32 output is byte-identical to the unfused GEMM -> GELU/SILU_MUL (FP32
out) pair. QUANT_X of it is byte-identical to the unfused act command with F_QOUT (int16 + scales). With emit_amax
merged (section 6), the next GEMM's output is byte-identical to the contract 1.0 chain.

## 2. Datapath, and why it adds 0 DSP

The change is in `dequant_ep` (process 6 of `gemm_contract.h`), plus a two-line change in `store`.

- **The pairing comes free from the emit order.** compute16 emits a block as rows `r = 2t + half`, each of 16
  columns. So the two 16-column halves of a 32-row group leave back to back for every token.
  - The packer interleaves the gate|up blob **per 16 rows**: blob row `32i + c` = gate row `16i + c`, and blob row
    `32i + 16 + c` = up row `16i + c`, for c < 16. `s_row` is interleaved the same way, and the bias too if one
    exists.
  - So a token's up word arrives 8 clocks after its gate word.
  - silu(gate) waits in a 2 x 8-entry LUTRAM (`gb`). This replaces a 64 x 32 FP32 BRAM tile buffer, which is what
    the brief's per-32-row-group interleave would have needed.
  - A row permutation of an integer GEMM is exact.
- **Two values a clock in act mode.** The emit word is worked as 4 quarters of 4 values; each quarter takes 2 clocks
  (h = 0, 1).
  - Lanes 0 and 1 run the full epilogue on values `4q + 2h + {0, 1}`. At h = 1 their inputs come from a stash
    register filled at h = 0 with lanes 2 and 3 of the quarter.
  - Lanes 2 and 3 lend their second FP32 multiplier: `fmul(x, h(x))`, or `fmul(up, gb)` for the up half of
    SILU_MUL.
  - In the source every lane keeps exactly 2 fmul + 2 fadd calls, and HLS instantiates one core per call at II 1.
    Normal mode puts operand muxes on lane 2/3's second multiplier and output muxes behind them.
- **h(x) without DSP.**
  - The shift uses gs's `GS_SHIFT_LUT` mux tree, and the 15 x 12 interpolation product is
    `BIND_OP impl = fabric`. `gs::h_bits` puts both on DSPs.
  - One dual-port ROM (2 x 1024 x 36 bit at NSB 10) serves both act lanes.
  - `ea::h_bits == gs::h_bits` on all 18.9 M checked inputs (measured): every table position of both functions,
    both signs, and every exponent.
- **Throughput.** In act mode the dequantizer takes 512 clocks per (group, block) instead of 256. compute16 needs
  K clocks per block, and every act GEMM of -12392 has K >= 720, so the MACs never wait. The last block's drain is
  256 clocks longer (counted in section 5). The store writes the same words or fewer: half as many for SILU_MUL,
  as 64-byte bursts.

## 3. Record change (CONTRACT.md reserved space)

- **The flags are full** (16 bits: 9 used, 7 retired), so no flag is added.
- **Parameter p6 of GEMM.** It is unused in 1.0 and must be 0 (ME_PARAM), so an old bitstream rejects a fused
  record and never silently runs it unfused. emit_amax already reserved p6 for P_ACT in a comment and uses p7 / p8.
- **New constants in `contract.h`:** `P_ACT = 6`, `EPI_NONE / EPI_GELU / EPI_SILU_MUL / EPI_SILU = 0..3`,
  `EPI_MAX = 3`.
- **New validation (contract.h `validate_t` and contract.py `validate`):**
  - `p6 > EPI_MAX` -> ME_PARAM.
  - `p6 != 0` with row groups (`p2 = R != 0`) -> ME_PARAM.
  - `p6 != 0` with F_ACC, or EPI_SILU_MUL with F_ADD_D -> ME_FLAGS.
  - C's minimum pitch and extent use `MO = M / 2` for EPI_SILU_MUL, otherwise M.
- **Unchanged:** registers, interfaces, slots and the record format. K and N limits are as for GEMM; no ACT_K_MAX
  applies, because there is no row buffer.

## 4. Patch plan (unified diffs in `diffs/`, against the snapshot `snap/` = the repo at 19:57)

Apply from the repo root with `git apply fpga/kernels/phaseB/emit_act/diffs/*.diff` (checked with
`git apply --check` against the live files at 20:1x, see section 8). The integrator edited `xpose` after the
snapshot, and the diffs do not touch it.

| file | hunks | change |
|---|---|---|
| `fpga/hls/gemm_contract.h` (`diffs/gemm_contract.diff`) | header comment; `#include "../kernels/phaseB/emit_act/emit_act.h"` | |
| | `struct Item` | + `int act` (p6), `int yu` (io words per (token, group): 8, or 4 for EPI_SILU_MUL) |
| | `dequant_ep` | `gb`, the stash, the 2-clock quarter in act mode, lanes 2/3 shared multiplier, output packing (section 2) |
| | `store` | the group's column offset is a running `gofs += yu * 16`; `if (w < it.yu)` on the 8 unrolled writes |
| | `plan_item` | `it.act = p[P_ACT]`, `it.yu` |
| `fpga/kernels/contract.h` (`diffs/contract_h.diff`) | 4 | P_ACT, EPI_*, validation (section 3) |
| `fpga/kernels/contract.py` (`diffs/contract_py.diff`) | 2 | the same rules (it parses the constants from contract.h) |
| new: `fpga/kernels/phaseB/emit_act/emit_act.h` | | `ea::h_bits` (0 DSP), the ROM, `ea::ACT_*`, `ea::out_cols` |

**Also needed outside fpga/hls, not in these diffs:**

1. **Packer** (`pack_float.py` / `contract.pack_w` users, a shared-contract file; both sides sign off).
   - For every LlamaMLP, pack `[gate; up]` as one blob of Mp = 2F rows, interleaved per 16 rows. Interleave
     s_row (and bias, if any) the same way.
   - F = 2560 and F = 2048 are multiples of 16. The rule is in `tb_emit_act.cpp nat_row` and in
     `check_policy.py interleave_idx`.
2. **List builder** (`contract_plan.py` -> the runtime builder). Per pair, replace (GEMM, act) by:
   - standalone: GEMM with `p6`, C = the FP32 rows (pitch >= 4 MO), followed by QUANT_X (K = the act's K, same
     C / F slots as the act record);
   - with emit_amax: GEMM with `p6` and `p7 = P_AMX_SG` (slot H), and the next GEMM with `p8 = P_XQ`.

   `savings.py fuse()` is exactly this transform. All 2,701 / 2,504 records pass the patched `contract.validate`
   (measured).
3. **Action-time MLP, emit_amax mode only.** The D row's pad columns (720..735 of E_DP = 736) must be 0.0, so
   silu(pad) = 0 and the producer's absmax over all 736 columns equals the consumer's over its 720. Standalone
   mode reads only K = 720 and does not depend on this. The pad contents the host writes today are unknown: check
   in the builder.

## 5. Time per -12392 chunk (est., `savings.py`, the contract's own model: contract_plan.py rules, DDR 14 GB/s)

| clock | F unfused | standalone: F, saved | + emit_amax: F, saved | T (+ 60 ms) unfused / standalone / + emit_amax |
|---|---|---|---|---|
| 250 MHz | 0.7243 s | 0.7064 s, **-17.9 ms** | 0.6684 s, **-55.9 ms** | 0.784 / 0.766 / 0.728 s |
| 214.29 MHz (300 x 5/7) | 0.8448 | 0.8239, **-20.9 ms** | 0.7795, **-65.3 ms** | 0.905 / 0.884 / 0.840 |
| 200 MHz | 0.9051 | 0.8827, -22.4 | 0.8352, -69.9 | 0.965 / 0.943 / 0.895 |
| 187.5 MHz (300 x 5/8) | 0.9655 | 0.9416, -23.9 | 0.8909, -74.6 | 1.026 / 1.002 / 0.951 |

In clocks per chunk: **-4.478 M standalone, -13.983 M with emit_amax**, at every clock. The 214 / 187.5 MHz steps
assume `set_pl_clock` divides from DIVISOR0 = 5; the bitstream's real divisor is unknown.

Where it comes from, at 250 MHz:

- **Removed:** GELU 19.0 ms + SILU_MUL 37.1 ms = 56.1 ms.
- **The 197 act GEMMs:** 199.75 ms before, 199.95 ms after (+0.2 ms, the longer drain). The 512-clock dequantizer
  stays below K.
- **Standalone adds 197 QUANT_X = 38.0 ms.** GELU's replacement costs about as much as GELU did, because both read
  the 3,072-wide rows once. SILU_MUL's replacement reads half as much as SILU_MUL. So standalone saves 18 ms;
  emit_act alone does not make the budget.
- **With emit_amax** the QUANT_X records go too: the brief's 56 ms. What P_AMX_SG / P_XQ cost on the GEMMs is
  emit_amax's to count; it is not included here.

## 6. Merge with phase B emit_amax (measured on its patched files of 20:10, sha256 429fc88a.. / b59dce42.. / 9177e3a0..)

Both patches start from the same snapshot. emit_amax does not touch `dequant_ep`. `merge_emit_amax.py` writes the
merged tree (`build/merge/`) from emit_amax's patched files with these edits:

1. **The emit_act parts unchanged:** the include, the Item fields, plan_item, dequant_ep (the whole process 6
   section), and the constants and validation in contract.h.
2. **emit_amax's word-level store loop:**
   - `if (w == 0) base = (it.ya + rw.off() + ((uint64_t)g << (it.yu == YU ? 7 : 6))) >> 4;` replaces the
     `g * (LANES * 4)` offset;
   - `if (w == it.yu - 1) {` replaces `YU - 1`.

   Its absmax then runs over exactly the activated values the store writes.
3. **The GEMM parameter whitelist** becomes `i != P_SEGC && i != P_SEGF && i != P_SHIFT && i != P_AMX_SG &&
   i != P_XQ && i != P_ACT`.

**Measured on the merged tree:**

- `tb_combo`: 124 of 124 chains are bit-exact. GEMM + ACT + AMX -> GEMM + XQ gives the same Z as
  GEMM -> ACT F_QOUT -> GEMM, including the four -12392 chains at full size (fc1 -> fc2, LM and expert gate_up ->
  down, time_in -> time_out).
- 120 of 120 emit_act cases with the emit_amax parameters at 0.
- emit_amax's own `tb_amax --quick`: PASS.

emit_amax's files are still changing: re-run `merge_emit_amax.py` and `tb_combo` after each change.

## 7. Resources and timing (HLS, csynth of `dequant_ep` / `store` alone, `hls/ea_top.cpp`)

RESOURCES_TABLE

## 8. Verification (all measured today)

| command | result |
|---|---|
| `sh fpga/kernels/phaseB/emit_act/run_tests.sh` | see below, ALL PASS |
| - diffs | snapshot + diffs == patched/ (3 files); the diffs apply cleanly to the live repo files |
| - integrator's `tb_gemm.cpp` on the patched header (P_ACT = 0) | 200/200, chaos 100/100 and NBX=1 100/100 bit-exact vs ctr::ref |
| - `tb_emit_act` h unit | 18,874,368 inputs, ea::h_bits == gs::h_bits |
| - `tb_emit_act` validation | the new rules: bad value, ACC, ADD_D with SILU_MUL, row groups, C pitch 4F |
| - `tb_emit_act` GEMM cases | 304/304: fused == unfused byte for byte over the whole image and O == a direct reference (ctr::ref::gemm_out + gs::h_bits + fmul_hw). Includes the 4 full-size -12392 GEMMs (fc1 3072x768x512, LM gu 5120x960x45, expert gu 4096x720x50, time_in 736x720x50 with D) and random shapes, items, bias, D, N % 32, K % 64, junk in every gap. Of the 5.4 M values checked, 0.79 M are 0 (0.67 M of them -0) and 1.18 M have \|x\| at or past the table range |
| - chaos / NBX=1 / base 0xFF_C000_0000 | 154/154, 154/154, 150/150 |
| - merged tree (section 6) | 124/124 chains, 120/120, tb_amax PASS |
| mutants (3: the gb index, the output lane order, the ROM rounding constant) | all caught |
| `check_policy.py` (frames 0 / 3563 / 7489) | E10 (fused) = U10 (unfused, NSB 10) exactly: max \|chunk - ref_chunk_raw\| **1.680e-3 / 1.588e-4 / 1.061e-4** (gate 0.05; T-P1: <= 0.006 frame 0, <= 2e-4 benign: met). Per frame, 210 act calls / 38.9 M values bit-equal to the unfused kernel's q and s, and 176 de-interleaved gate\|up GEMMs bit-equal to the separate gate and up GEMMs |
| `savings.py` | section 5 |

## 9. Unknown (not guessed)

- **Post-route timing and LUTs.** HLS estimates only; no Vivado run.
- **DDR efficiency of the 64-byte Y bursts** for SILU_MUL (BT-2). The bytes halve, but the bursts are half as long.
- **The PL clock steps.** They depend on the BD's PL0 divisor (`runtime.set_pl_clock`); 214.29 and 187.5 MHz
  assume DIVISOR0 = 5.
- **The host's pad columns of the time row** (section 4, item 3).
- **No cosim.** The act path's II and the gb distance are as HLS scheduled them (section 7), not simulated in RTL.
