# quantize: the row unit's backbone and the contract int16 quantizer (CONTRACT.md 1.0)

This directory holds two things for the float_top integration contract (`fpga/kernels/CONTRACT.md`, `contract.h`):

- `quant_int16.h`: **the contract quantizer (C2)** every int16 producer uses: `s = fl32(max(amax, 1e-30)/32767)`,
  an all-zero row stores `1.0`, `q = clamp(rne(fl32(x/s)), ±32767)`, stored scale `s · 2^p4`. No FP divider: an
  exact integer datapath, 3 DSP48E2 per lane, one 1-bit-a-clock reciprocal divider per row.
- `quant_mode.h`: **the row unit's reader → absmax + row buffer → quantizer → writer** (CONTRACT.md 4.1) with the
  identity front end, i.e. the opcodes QUANT_X, QUANT_W, QUANT_WT and COPY. NORM / GELU / SILU_MUL / ROPE are the
  same backbone with their FP32 front end in the marked slot (other kernels' code).

Everything here is bit-exact with `contract.h ctr::ref::quant_row` / `contract.py quant_rows`, and every output
image is compared word for word against an independent reference built from the contract's layout formulas.

## Files

| file | what |
|---|---|
| `quant_int16.h` | the arithmetic: `abs_bits`/`amax2`/`word_amax` (absmax), `scale_bits` (s), `recip_seed`/`recip_load`/`recip_step`/`recip_result`/`recip_sig` (R = floor(2^49/S)), `row_scale_from_s`/`row_scale`, `stored_scale` (p4 shift, zero rows 1.0, non-finite rows keep +Inf / NaN), `quant_elem` (one value), `quant8`, word lane helpers |
| `quant_mode.h` | `qz::row_mode(const ctr::Cmd&, const word_t* xmem, word_t* ymem)`: the mode function (one DATAFLOW region per command) |
| `quant_top.cpp` | csynth-only top: fetches one 256-byte record over io, `ctr::decode`, calls `row_mode` (so the report shows `row_mode` as its own instance) |
| `run_hls.tcl`, `hls_summary.py` | csynth (env `QZ_PERIOD`, `QZ_IMPL=fdiv`, `QZ_ENGINES`, `QZ_SMALL=0`, `QZ_TAG`); per-process summary of a report |
| `tb_quant.cpp`, `run_tests.sh`, `dfsim/hls_stream.h` | host testbench (g++); `--dfsim`: the threaded bounded-FIFO deadlock model |
| `qz_emul.py` | torch model of the integer datapath (`quant_kernel`), `quant_ref` / `quant_ref_engine`; `--selftest` |
| `check_policy.py` | full-policy check on -12392's 3 replay frames under the contract's arithmetic; `--dump` real activations for the tb |
| `workload.py` | this unit's time per chunk from the contract's list (`contract_plan.py --out`) and the cycle model |
| `results/` | `check_policy.log/.json`, `tb_full.log`, `tb_dfsim.log`, `workload.txt`, `captured.qz` (git-ignored) |

## Interface (CONTRACT.md 3, 4.1)

```
uint32_t qz::row_mode(const ctr::Cmd& c, const qz::word_t* xmem, qz::word_t* ymem);   // INLINE off
struct qz::RowUnit { template <class WP, class W> static uint32_t row(c, hp0, hp1, hp2, hp3, io_rd, io_wr); };
```

- Called from the sequencer's dispatch switch after `ctr::validate_hw(c) == ME_OK`, one call site for all four
  opcodes. Returns `ME_OK`, or `ME_DIMS` without touching memory for an opcode of another unit and for
  QUANT_W with p1 = 0 and K ≠ 64 (see Contract notes). With `QZ_SMALL=0` (area probe only) also for shapes that
  need single-word units.
- **Ports**: reads only `xmem`, writes only `ymem` (the io bundle); no hp port, no m_axi adapter, no decode, no
  register of its own. Every slot address is the absolute byte address (pointer registers 0 in list mode); all
  word indices are 36 bits (byte addresses < 2^40, validate_hw's rule).
- **R-ORD**: `row_mode` returns when the DATAFLOW region ends, i.e. after `wr`'s last write response.
- float_top's io m_axi needs `latency = 64` (quant_top has it): with the default 0 every pipelined burst unit waits
  for its own DDR round trip.

Per opcode (CONTRACT.md 1.1, 3.3; every address / pitch / stride a multiple of 16 B):

| op | A | C | F | params |
|---|---|---|---|---|
| QUANT_X (F_QOUT) | FP32 rows | int16 rows; segment j of row n at `C + b C.stride + n C.pitch + j p2` | scale of (n, j) at `F + b F.stride + j p3 + 4n`, pad4(NP) per vector, pad lanes 1.0 | p1 0 / 64, p2, p3, p4 shift, p5 NP (rows [N, NP) zero, scale 1.0) |
| QUANT_W | FP32 rows (keys) | KW tiling, word (n, d) at `C + b C.stride + j p2 + ((n%32)/8) C.pitch + ((n/32) 64 + d) 16`, lane n%8 | per (key, segment) | p1 = 64 (or K = 64), NP % 32 == 0 |
| QUANT_WT | FP32 rows (keys) | VW tiling, word (c, n) at `C + b C.stride + j p2 + ((c%32)/8) C.pitch + ((c/32) NP + n) 16`, lane c%8 | per channel over the item's N keys | p1 0 / 64, NP % 8 == 0 (keys [N, NP) zero) |
| COPY | FP32 rows | FP32 rows | | |

## Processes (quant_mode.h)

| process | what |
|---|---|
| rd | xmem → tokens in 16-word units (one 16-beat burst per unit, II = 16, so requests stay in flight across rows); a row not a multiple of 16 words ends with a shifted-back unit whose leading words go out as ignored tokens. QUANT_X rows above ROW_BUF = 3,072 values are read twice; QUANT_WT reads per 64-column chunk a column-absmax pass and a data pass, interleaved P1(0) P1(1) P2(0) P1(2) ... Rows of < 16 words use single words (`rd1`). |
| amx | per-segment absmax (QUANT_WT: per column) → `am`; data → `d`, the row buffer (1,024 × 128-bit BRAM FIFO) |
| ss | `scale_bits`, `recip_seed`, `stored_scale`, one a clock |
| sr | 2 restoring dividers, 1 quotient bit a clock, one scale per 13.5 clocks; never blocks (read_nb / write_nb) |
| qt | 4 lanes of `quant_elem` a clock (QUANT_WT: per-lane 16-entry column tables); the output word is written one clock later from a register |
| wb | 8-word units with addresses: rows (shifted-back last unit from an 8-word ring, zero rows), KW (8 × 8-word transpose block), VW (8 keys × 8 words reorder), scale vectors (32 a unit, pad lanes 1.0) |
| wr | units → ymem (one 8-beat burst per unit; `wr1` for single-word units) |

Deadlock freedom is checked by simulation, not argued: `--dfsim` runs every process as a thread with every stream
bounded at its pragma depth (dfsim/hls_stream.h); a stream blocked for 20 s aborts with DEADLOCK.

## Resources and timing (Vitis HLS 2022.2 csynth, xck26-sfvc784-2LV-c, 4.0 ns; build/qzm_recip_p40)

Final (v8): every loop at its target II (rd16 16, wr8 8, all others 1), **estimated 3.40 ns (294 MHz)**: sr
3.40 ns, qt 3.34 ns, everything else inside the 2.92 ns budget. `row_df` latency 176 clocks for a 1-row command.

| block | LUT | FF | DSP | BRAM18 |
|---|---|---|---|---|
| rd (rd16 + rd1) | 3,029 | 1,966 | 0 | 0 |
| amx | 1,243 | 707 | 0 | 0 |
| ss | 979 | 588 | 0 | 0 |
| sr (2 dividers) | 865 | 359 | 0 | 0 |
| qt (4 lanes) | 4,632 | 3,964 | 12 | 0 |
| wb | 3,640 | 5,975 | 0 | 0 |
| wr (wr8 + wr1) | 811 | 679 | 0 | 0 |
| FIFOs | 817 | 1,054 | 0 | 8 |
| **row_mode** (what float_top gains; + its plan logic) | **17,140** | **16,044** | **12** | **8 (4 BRAM36)** |
| quant_top standalone (+ io m_axi + s_axilite + fetch/decode) | 24,131 | 19,968 | 12 | 8 |

History of this conform (same tcl):

| version | row_mode LUT | DSP | est. clock | change |
|---|---|---|---|---|
| pre-contract (8 lanes, hp0/hp1 reads, own registers) | 16,282 (quant_rows) | 25 | 2.92 ns budget met | |
| v3 (crash WIP) | n/a | | 4.13 ns (242 MHz) | |
| v4 (crash WIP) | 18,041 | 12 | 3.78 ns (264 MHz) | contract layouts, 4 lanes, QUANT_W/WT/COPY, io only |
| v5 | 18,623 | 16 | 3.44 ns | sr: unconditional steps + captured result, non-blocking; qt DSP shifter (rejected: +0.9k LUT) |
| v6 | 18,127 | 12 | 3.44 ns | wb rewrite (one output select); **wb II = 3** |
| v7 | 16,851 | 12 | 3.40 ns | invariants precomputed in RP, unconditional rd tokens, no re-checks of validate_hw; **qt II = 2, wb II = 3** (not valid) |
| **v8** | **17,140** | **12** | **3.40 ns (294 MHz)** | one store site per LUTRAM (II = 1 everywhere) |
| v7 + `QZ_SMALL=0` probe | 15,389 | 12 | 3.40 ns | no single-word units: −1.46k (rd1, wr1, small-unit logic) |

**Against the budget (CONTRACT.md 4.2): not met.** The contract caps the whole row unit (this backbone plus the
NORM, GELU/SILU and ROPE front ends) at 15k HLS LUT / 50 DSP / 10 BRAM36. This backbone alone is 17.1k HLS LUT
(12 DSP, 4 BRAM36): 114 % of the cap before any front end. With the contract's 1.0× factor for integer fabric
logic it is ~17k post-route against the 12.8k the budget gives the row unit. It is 5 % below v4 and meets the clock
(v4 did not). What would close more of it, measured or estimated:

| option | saving | cost |
|---|---|---|
| contract pads rows to whole units (int16 / FP32 rows K % 64 == 0, scale vectors pad32): no single-word units | −1.46k (measured, v7 probe) | x_t (K = 32) rows padded to 64 columns with zero weights; a contract.h change |
| also no shifted-back tails (row words % 8 == 0, scale vectors % 32 == 0) | est. −0.5k (ring, pad lanes, rd skip) | K = 720 rows pad to 768; scale vectors pad32 |
| QUANT_WT moved out of the row unit (e.g. V written row-major and transposed by the engine's X stage) | est. −1.5k (amx column tables, qt lane tables, wb VW, rd P1/P2) | an engine change |
| merge ss into amx | est. −0.3k | |

None of these is done: each changes the contract or another owner's code.

## Speed (workload.py on the contract's list, 250 MHz)

| op | commands | M values | ms at DDR first-burst latency L = 0 / 64 / 128 clocks |
|---|---|---|---|
| QUANT_X | 236 | 18.76 | 19.40 / 19.46 / 19.52 |
| QUANT_W | 27 | 4.93 | 4.96 / 4.96 / 4.97 |
| QUANT_WT | 115 | 6.33 | 12.76 / 12.79 / 12.82 |
| COPY | 5 | 0.40 | 0.41 / 0.41 / 0.41 |
| **total** | 383 | | **37.5 / 37.6 / 37.7** (contract_plan.py budgets 38.8 ms for these four) |

Model: 1 io word read a clock (16-word bursts across rows), 13.5 clocks per scale minimum, 215 clocks per command.
It assumes the HP/HPC port sustains 16 B a clock at 250 MHz (4 GB/s) for 256-B bursts: not measured (BT-2).

## Accuracy

Testbench (`results/tb_full.log`): scale_bits on 8 exponents × all 2^23 significands + 2 M small / zero; R for all
2^23 S; 435 M elements (32 M ties, 13.9 M double-rounding cases) for the integer and the FP32-divider lanes;
exhaustive slices (1.34 G elements); stored_scale with every p4; 22 edge commands; 3,000 random contract-valid
commands (all four opcodes, segments, NP zero rows, shifts, items, pitch and item gaps, aliased inputs, 32-column
chunks, single-word units, bases up to 2^40), every one of this unit's 383 records of the -12392 list at two CMA
bases, rejects, ±Inf / NaN rows, 302 captured real-activation tensors. Memory is NaN-filled and compared word for
word against the reference image: 0 differences. The same runs in the bounded-FIFO model: no deadlock.

Full policy (`check_policy.py`, -12392, max |chunk − ref_chunk_raw| in normalized action units, gate 0.05):

| frame | fp32 | actq (quantizer only) | c_lin: contract GEMM for every linear | c_full: + contract attention (Q p4 = −3, K, V^T, smx P) | kernel == reference chunk |
|---|---|---|---|---|---|
| 0 | 5e-6 | 1.086e-3 | 1.734e-3 (0.051 deg) | 1.291e-3 (0.038 deg) | bitwise, both modes |
| 3563 | 0 | 1.74e-4 | 8.5e-5 | 1.53e-4 | bitwise, both modes |
| 7489 | 0 | 1.55e-4 | 1.66e-4 | 1.58e-4 | bitwise, both modes |

Every quantized operand of those runs was also compared with the contract reference: 638 M linear-input values
(653,574 rows), 40 M Q values with p4 = −3, 55 M K / V^T values: **0 q and 0 stored-scale mismatches**, 0 against
pack_float. Frame 0 is the sensitive frame (the precision study's I7e class allows <= 6e-3); c_lin's 1.7e-3 there
against the old float64-epilogue run's 4.6e-4 is the contract epilogue's two FP32 roundings, not the quantizer.
Not modelled here: the NORM / GELU / SILU / ROPE front-end emulators and the packer's folded constants (T-P1).

## Contract notes (for the contract owner)

- **QUANT_W with p1 = 0 and K ≠ 64**: `validate` accepts it, this unit returns ME_DIMS (one scale over a whole
  row of K > 64 values in the KW tiling would need 8 whole rows on chip). That breaks "a list that passed
  validate never stops the IP with a mode error". The -12392 list never builds it. Suggested contract.h line:
  QUANT_W (and ROPE F_KW) require `SEG == SEG_HEAD`, i.e. p1 = 64 or K = 64.
- The area cap above: the backbone does not fit 15k; options and numbers in Resources.

## Review defects (integration_inputs.json) and what happened

| defect | resolution |
|---|---|
| layout conflict with the engine's xmem image | contract C1: row-major everywhere; this unit writes exactly the contract layouts (tested) |
| activations read on hp0/hp1 | reads only on io (xmem) |
| 16.3k LUT for 8 lanes + 2 read ports | 4 lanes, one reader, no adapters, no decode: 17.1k for four opcodes incl. QUANT_W/WT/COPY; still over the cap (above) |
| check_policy covered only nn.Linear inputs (not the vision Q heads) | c_full quantizes every attention operand (Q with p4 = −3, K, V^T) with the kernel's datapath, compared with the reference |
| check_policy epilogue in float64 | `contract.gemm_out` order: fmul(fmul(fl(acc), s_row), s_tok), fadd bias, FTZ with sign |
| NaN / Inf turned into finite scales | stored scale is the row's +Inf / NaN (tested); q of such rows stays undefined (contract: runtime checks x_t) |
| exec_ref scale convention | contract C2 (contract.quant_rows); this unit equals it bit for bit |
| overlap of in / out / scale extents unchecked | R-ORD: overlap is a builder bug; validate / validate_hw are the checks |
| README 3.333 ns claims, divider cost, QZ_RBITS doc, "zero-padded" docstring | rewritten; sr (2 dividers) is 865 LUT measured; QZ_RBITS removed; docstring fixed |
| deadlock freedom only argued | bounded-FIFO threaded model (`--dfsim`) on every plan record, 1,000 random commands, edges |

## Tests (commands)

```
python ../contract_plan.py --out build/chunk_hi.bin                  # the contract list (0x8_0000_0000 base)
python ../contract_plan.py --out build/chunk_lo.bin --base 0x40000000
sh run_tests.sh --quick --plan build/chunk_hi.bin                     # ~20 s
sh run_tests.sh --exhaustive 8 --random 3000 --plan build/chunk_hi.bin --plan build/chunk_lo.bin --plan-all results/captured.qz
sh run_tests.sh --dfsim --random 1000 --plan build/chunk_hi.bin --plan build/chunk_lo.bin --plan-all results/captured.qz
C:\kp\lerobot-venv\Scripts\python.exe qz_emul.py --selftest
HF_HUB_OFFLINE=1 C:\kp\lerobot-venv\Scripts\python.exe check_policy.py --json results/check_policy.json --dump results/captured.qz
C:\kp\lerobot-venv\Scripts\python.exe workload.py
vitis_hls -f run_hls.tcl; python hls_summary.py build/qzm_recip_p40   (PATH without the Git directories)
```

## Not verified

Post-route area and timing (Vivado not run); RTL co-simulation (no csim / cosim here: the C model and the
bounded-FIFO model are the evidence); DDR latency and burst efficiency on the KR260 (the times assume 16 B a
clock); the FP32 divider core's real cost (`QZ_IMPL=fdiv`, HLS reports it as 0); the merged float_top (port
muxing with the engine and the softmax unit on the shared io adapter); nothing ran on the board.
