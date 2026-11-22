# gelu_silu: the GELU / SILU_MUL modes of float_top (integration contract 1.0)

The element-wise activations of SmolVLA -12392, as float_top modes that follow `fpga/kernels/CONTRACT.md` 1.0. Each
mode reads FP32 rows (the int16 engine's GEMM output) and writes int16 rows with one FP32 scale per row, quantized by
the contract quantizer (`quantize/quant_int16.h`). Without F_QOUT it writes FP32 rows instead.

| opcode | y | use per 50-action chunk (`contract_plan.py`) |
|---|---|---|
| `OP_GELU` (0x50) | tanh-GELU(x) (`gelu_pytorch_tanh`) | SigLIP fc1 -> fc2: 12 records, 512 x 3072 |
| `OP_SILU_MUL` (0x51) | silu(gate) * up | expert: 160 records, 50 x 2048; LM: 15 records, 45 x 2560 -> down_proj |
| `OP_SILU_MUL` + `F_NOMUL` | silu(x) | action-time MLP (`F.silu`): 10 records, 50 x 720 |

## Files

| file | what |
|---|---|
| `gelu_silu.h` | header-only HLS source: the mode function `gs::act_mode`, the front end `gs::gs_fe`, its private backbone; the arithmetic spec is in its top comment |
| `gelu_silu_tables.h`, `gs_tables.py` | the ROM (generated: `python gs_tables.py`), NSB 9 and 10 |
| `gelu_silu_top.cpp`, `run_hls.tcl` | standalone csynth top (the io bundle only) and the csynth script |
| `tb_gelu_silu.cpp`, `dfsim/hls_stream.h` | host testbench (links the HLS source); bounded `hls::stream` for the threaded deadlock model |
| `make_vectors.py`, `run_tests.sh` | vector generator and the test runner (everything below) |
| `gs_emu.py` | bit-exact torch emulation of the front end (the testbench's and the policy check's reference) |
| `policy_check.py` | full-policy effect on -12392's 3 replay frames, contract arithmetic |
| `time_model.py` | time per chunk over the real -12392 list, from the csynth schedule |

`cpp_mirror.py` (a Python transliteration of the old C++, written when the testbench could not run) is deleted: the
C++ testbench now runs.

## Contract conformance

- **Mode function.** `uint32_t gs::act_mode(const ctr::Cmd& c, RM io_rd, WM io_wr)` is `INLINE off` and has one call
  site in the dispatch switch: `case OP_GELU: case OP_SILU_MUL: return gs::act_mode(c, io_rd, io_wr);`.
  `gs::act_unit(c, hp0, hp1, hp2, hp3, io_rd, io_wr)` has the sequencer's unit signature (`seq_core.h`
  "Units"). c is a decoded record that passed `ctr::validate_hw`. The mode returns `ctr::ME_OK`.
- **No own decode, no own adapters, no register.** The mode takes float_top's io pointers as arguments. Activations
  are read and written only on io; the hp ports are not used (CONTRACT 3.3). Addresses are the records' absolute
  64-bit byte addresses, carried as 36-bit word indices (validate_hw: every address is below 2^40 and 16-byte
  aligned). All address sums are at least 36 bits wide; nothing is an `int`.
- **Slots (C1 row-major).**
  - A: x (SILU_MUL: the gate) rows, at `A + b A.stride + n A.pitch`.
  - B: up rows (SILU_MUL without F_NOMUL).
  - C: int16 rows (F_QOUT) or FP32 rows.
  - F: the scale vector of item b at `F + b F.stride`, with pad4(NP) floats; rows [N, NP) and the pad lanes hold
    1.0.
  - p4 shifts the stored scale (an all-zero row stores 1.0). p5 = NP: rows [N, NP) are written as zeros.
  - Items B >= 1 run in one command. Lanes are in natural order.
- **Quantizer (C2).** `quant_int16.h` throughout: `scale_bits`, `recip_sig` (1 bit a clock, QZ_RECIP_BITS = 1,
  enforced by `#error`), `quant_elem`, and `stored_scale`. stored_scale keeps a NaN / Inf row's scale non-finite.
  The kernel's own quantizer (s = 1/r, round half up, zero rows s = 0) is removed.
- **Front-end arithmetic (C2.2).** y is bit-exact with `gs_emu` (NSB 9, 4 lanes, natural column order).
- **Ordering (R-ORD).** HLS waits for the write responses of each burst loop before that loop ends, so the mode
  returns after its last BRESP.

## Arithmetic of the front end

Both activations are x h(x) with a sigmoid-like gate, h = Phi_tanh (GELU, range R = 8) or the logistic (SiLU,
R = 16). The table holds g(t) = 1 - h(t) for t = |x|:

1. `T = floor(t 2^(TB - log2 R))`, TB = NSB + 12. The segment index is `T >> 12` and the fraction is `f = T & 4095`.
   The shift is done on a DSP.
2. ROM word `V << 15 | D`: `g21 = max(0, (V << 12) + 2048 - D f) >> 12` in Q0.21, 0 for t >= R. V is the minimax
   offset over the 4096 sub-positions (`gs_tables.py`). NSB 9 means 512 segments per function, one
   1024 x 36 ROM (1 BRAM36) per lane pair.
3. `H = x < 0 ? g21 : 2^21 - g21`, and `h = H 2^-21`, normalised to an exact FP32 in 5 select stages.
4. `y = fmul(x, h)`. For SILU_MUL the gate words give silu(gate), which is stored in `gbuf` (one row). The up words
   then give `y = fmul(up, silu(gate))` with the same multiplier (the product is commutative). fmul is the Xilinx
   FP32 core: RNE, with denormal inputs and results flushed to zero and the sign kept (`contract.ftz`).

**Limits (review: tail resolution).** g has an absolute resolution of 2^-21, so the negative tails lose relative
precision:

- the relative error in y is 4.9e-4 at x = -8 for SiLU;
- SiLU gives -0 for x <= -15.25, where the true value is above -4e-6;
- GELU gives -0 for x <= -8, where the true value is above -1e-14;
- in a SILU_MUL row where a gate below -15.25 times a large up dominates, the absmax comes out wrong.

On the -12392 activations this is at most 0.037 step and has no policy effect (CONTRACT 9 #23). The old README
claimed that "small g keep their relative precision"; that claim was wrong and has been removed.

**NaN / Inf** inputs give an undefined q (finite garbage), but the row's stored scale is non-finite: Inf or the NaN,
from `stored_scale`. The divergence therefore reaches the next GEMM's output instead of being hidden. A run never
hangs and never writes outside its extents. The testbench checks all of this.

## Processes (one DATAFLOW region per command, every loop II = 1)

| process | what | in the integrated row unit (CONTRACT 4.1) |
|---|---|---|
| `gs_rd` | io reader. **Dense** (every GELU / SILU_MUL record of the -12392 list): one burst stream per item over rows [0, N); pitch gaps of up to GAPW = 64 words per row are read and dropped. SILU_MUL needs the fused gate\|up rows: B = A + 4K and B.pitch = A.pitch. Otherwise one burst per row (the A row, then the B row) | the shared reader |
| **`gs_fe`** | **the front end** (this kernel's own logic): 4 lanes of h, one fmul per lane, gbuf | **plugs between the reader and the row buffer** |
| `gs_am` | per-row absmax; y into the row buffer `ys` (one row + 128 words) | shared absmax + row buffer |
| `gs_sc` | absmax -> s, R (25-step restoring divider, 1 bit a clock), stored scale | shared quantizer |
| `gs_qt` | 4 lanes of `quant_elem`, or y passed through for FP32 output | shared quantizer |
| `gs_wr` | io writer. **Dense** (C.pitch = the row): one burst per chunk of up to WCH = 64 rows, then that chunk's scale words. Otherwise one burst per row. Zero rows [N, NP); pad lanes 1.0 | the shared writer |

**Front-end interface** (`gs_fe(const GsCmd&, stream<Word>& x_in, stream<Word>& y_out)`):

- GsCmd fields used: mode, k4 = K / 4, up0, rowits, n, nb.
- Input: per row, k4 words of x (GELU, SILU), or k4 words of gate followed by k4 words of up (SILU_MUL).
- Output: per row, k4 words of y.
- One input word a clock. SILU_MUL gives one output word per two input words, which is the reader's pace.

**Bursts (review: the rows loop was not flattened).** Each burst loop pays one DDR round trip. A csynth with
`-m_axi_latency 64` shows 76 clocks per reader run and 75 per writer run. Dense mode pays this once per item and once
per 64-row chunk; the per-row path pays it on every row. The writer's waits are hidden: it drains one word a clock
while int16 rows arrive at half a word a clock, and the row buffer absorbs the stall.

The testbench's `--plan` census decodes the real list (`contract_plan.py --out`): **all 197 GELU / SILU_MUL records
take the dense read and the dense write.**

**Deadlock (CONTRACT 7, the threaded bounded-FIFO model).**

- Rule: `ys` must hold one whole row, and the row's absmax token follows its last word. `static_assert(GS_D_YS >
  K4MAX)`; K > ACT_K_MAX is rejected.
- The writer pops one stored scale per real row even without F_QOUT. gs_sc produces one either way; otherwise sst
  would fill and stall gs_qt.
- The testbench built with `-DGS_DFSIM` runs every process as a thread on streams bounded at the pragma depths.
  - Pragma depths: no deadlock.
  - Minimum depths (every side FIFO 1, ys = 769): no deadlock.
  - Negative test (ys = 700 < 768): reports DEADLOCK, as it must.

## Verification (`sh fpga/kernels/gelu_silu/run_tests.sh`, all run 2026-11-21: ALL PASS)

`tb_gelu_silu.cpp` is built with the Vitis MinGW g++ on a clean PATH, with `-static -ffp-contract=off`. It links the
HLS source.

**Cases.** 79 cases, for NSB 9 and NSB 10:

- 75 synthetic from `make_vectors.py`:
  - random rows;
  - edge rows: zeros, -0, denormals, segment boundaries, +-R, 1e4, +-3e38, all-equal rows;
  - every small shape K 8/16/24/120 x N 1/2/3/5/7, including the SILU_MUL idle iterations;
  - writer chunks, N = 133 with a zero row;
  - tails, amax 2^-109..2^-125;
  - NaN / Inf rows.
- 4 real -12392 activations of frame 0 (`policy_check.py --dump`): vision GELU 256 x 3072, LM 81 x 2560, expert
  50 x 2048, action-time 50 x 720.

**Each case runs as contract records** (encode -> decode -> validate == validate_hw == ME_OK). It runs on a sparse
40-bit memory where any access outside the declared buffers aborts, in 5 layouts. The testbench checks the path
each layout takes:

| layout | path |
|---|---|
| plan | dense / dense, base 0x8_0000_0000 |
| padded | dense read with gaps (SILU_MUL: sparse) / sparse write; 2 items, NP = N + 3, p4 = -3, base 0xFF_0000_0000 |
| wide | sparse read / dense write with zero rows; 2 items, NP = N + 5, p4 = +2 |
| fp32 | dense / sparse, FP32 out |
| fp32d | dense / dense, FP32 out, 2 items |

Across the runs that covers all four (rdense, wdense) paths: 28 / 54 / 117 / 196.

**What is checked:**

- y bits against gs_emu;
- q and the stored s against `contract.quant_rows` of gs_emu's y, which equals `ctr::ref::quant_row`;
- zero rows, pad lanes and guard words;
- inputs untouched;
- the non-finite scale of NaN / Inf rows;
- 12 bad commands are refused with 0 memory accesses;
- the csynth top's pointer build equals the MemRef build.

**Results.**

- NSB 9 and NSB 10: 79 / 79 pass, with 0 q, 0 s, 0 y and 0 guard mismatches.
- Threaded model at the pragma depths and at minimum depths: 79 / 79 pass. The negative test exits 7.
- Error of q s against float64, in steps of s, on the real data (ideal rounding alone gives max 0.5, rms 0.289):

  | NSB | GELU | LM | expert | action-time |
  |---|---|---|---|---|
  | 9 | 0.69 | 0.87 | 0.86 | 0.61 |
  | 10 | 0.57 | 0.56 | 0.57 | 0.52 |

## Resources and timing (Vitis HLS 2022.2 csynth, xck26-sfvc784-2LV-c, 4.0 ns, NSB 9)

Every loop is II = 1. Estimated Fmax is 342 MHz (2.92 ns), against the contract's <= 3.6 ns. The pipeline depths are
rd 4, fe 21, am 4, qt 13, wr 4 and scales 3; the gs_sc row takes 34 clocks.

| block | LUT | FF | DSP | BRAM18 | URAM |
|---|---|---|---|---|---|
| **gs_fe (the front end)** | **3,889** | 5,563 | **20** (4 shift + 4 interpolation + 4 fmul x 3) | **12** (ROM 4 + gbuf 8) | 0 |
| gs_rd | 994 | 700 | 0 | 0 | 0 |
| gs_am | 533 | 275 | 0 | 0 | 0 |
| gs_sc | 1,293 | 565 | 0 | 0 | 0 |
| gs_qt | 3,673 | 2,906 | 12 | 0 | 0 |
| gs_wr | 1,865 | 1,532 | 1 | 0 | 0 |
| streams (ys = 8 BRAM18), dataflow start | 565 | | 0 | 8 | 0 |
| act_mode's own logic (gs_plan, its 16 x 11 multiply, the guard) | 1,075 (691 with GS_GUARD = 0) | | 1 | 0 | 0 |
| **act_mode, GS_GUARD = 1** | **13,916** | 13,042 | **34** | **20** | **0** |
| **act_mode, GS_GUARD = 0** (float_top: validate_hw already checks) | **13,532** | 13,030 | 34 | 20 | 0 |
| standalone top (+ io adapter, control) | 18,892 | 15,529 | 34 | 20 | 0 |
| NSB 10: act_mode, GS_GUARD = 1 (gs_fe 3,897 LUT, 16 BRAM18) | 13,924 | 13,044 | 34 | 24 | 0 |

**Against the budget** (CONTRACT 4.2: the whole row unit at most 15k HLS LUT, 50 DSP, 10 BRAM36, 0 URAM, for 8
opcodes):

- The front end alone is 26 % of the LUT cap, 40 % of the DSP cap and 60 % of the BRAM cap.
- This standalone mode, with its private backbone, is 90 % / 68 % / 100 %. In float_top the backbone (rd, am, sc,
  qt, wr: about 9.6k LUT and 13 DSP here) is the row unit's shared one and is counted once.

**Change against the review's build.** The old 8-lane DATAFLOW region was 23,607 LUT, 24 DSP and 11.5 BRAM36, with
36,811 LUT for its top with 5 adapters.

**BRAM options.**

- `GS_GBUF_URAM=1` moves gbuf to 2 URAM, -4 BRAM36. The pre-conform build measured gs_fe at 4 BRAM18 + 2 URAM. The
  row unit's URAM cap is 0; the device has about 16 URAM spare.
- **Proposal, not implemented (it needs a contract change):** the packer could interleave the gate_up blob's output
  channels in blocks, for example [gate 0..31 | up 0..31 | gate 32..63 | ...]. That is exact, a permutation of the
  GEMM's weight rows. gbuf would then shrink to one block, -4 BRAM36.

`GS_SHIFT_LUT` (-4 DSP, a mux tree instead) exists but was not synthesised.

## Time per chunk (`python time_model.py`, 250 MHz, every record of the -12392 list)

| op | K | N | records | this model ms | contract model ms |
|---|---|---|---|---|---|
| GELU | 3072 | 512 | 12 | 18.89 | 19.00 |
| SILU (F_NOMUL) | 720 | 50 | 10 | 0.38 | 0.39 |
| SILU_MUL | 2048 | 50 | 160 | 32.95 | 33.24 |
| SILU_MUL | 2560 | 45 | 15 | 3.47 | 3.50 |
| **total** | | | 197 | **55.70** | **56.13** |

- The DDR round trip is assumed at 100 clocks. At 0 clocks the total is 55.54 ms; at 300 clocks, 56.01 ms. Dense
  mode pays it once per item.
- SILU_MUL runs at 1 output word per 2 clocks, because it reads gate and up through one io port. GELU runs at 1 per
  clock.
- DDR traffic is 20-24 B a clock (5-6 GB/s), below the 14 GB/s assumed.
- All of these are model numbers: no cosim, no board.

## Accuracy: full policy, contract arithmetic (`policy_check.py`, frames 0 / 3563 / 7489)

Max |chunk - ref_chunk_raw| in normalized action units. The gate is 0.05. T-P1 (CONTRACT 7) asks for <= 0.006 on
frame 0 and <= 2e-4 on the benign frames, for the whole recipe.

| variant | frame 0 | frame 3563 | frame 7489 |
|---|---|---|---|
| F0: FP32 linears, torch activations (laptop vs reference) | 4.7e-6 | 3.6e-7 | 3.0e-7 |
| F1: FP32 linears, this kernel | 5.8e-4 | 5.1e-5 | 7.6e-5 |
| C0: contract int16 engine emulation (all linears incl. the 21 small ones, contract quantizer, epilogue, time-MLP D row), torch activations | 1.78e-3 | 8.6e-5 | 1.61e-4 |
| **C1: C0 + this kernel (NSB 9)** | **3.00e-3** | **2.12e-4** | **1.38e-4** |
| C1 - C0 (the kernel's own effect) | 1.22e-3 | 1.70e-4 | 1.93e-4 |
| C1 with NSB 10 | 1.68e-3 | 1.59e-4 | 1.06e-4 |

**Reading.**

- Everything is far inside the 0.05 gate.
- C1 at NSB 9 is 2.12e-4 on frame 3563. That is just above T-P1's 2e-4 for benign frames, and this emulation does
  not even include the other kernels.
- NSB 10 is lower than NSB 9 against the reference on all three frames. It halves the y error: rms 0.03 -> 0.015
  steps. It costs +2 BRAM36 (ROM 4 BRAM36), so the front end would hold 8 of the row unit's 10 BRAM36 unless gbuf
  moves.
- **The NSB choice is open for the integrator / psamin.** Both builds are tested bit for bit. The default stays NSB 9
  to keep the row unit's BRAM cap.

Not emulated here (they belong to the contract owner's T-P1): int16 attention and the softmax P, the folded constants,
the norm / rope kernels, the trimmed language tokens. This policy run pads the prefix to 81 rows; the list uses 45.

## Review defects and their resolution

| review item | resolution |
|---|---|
| own quantizer (s = 1/r, round half up, zero rows 0, pad lanes 0) | quant_int16.h; q and s bit-equal to contract.quant_rows and ctr::ref on every case; zero rows and pad lanes 1.0 |
| layout dependency on the engine | contract C1: row-major rows with pitch, scale vectors; slots as CONTRACT 3.3 |
| 8 lanes in permuted order by default | 4 lanes, natural order only; the 8-lane build and the permutation helpers are removed |
| area 23.6k region / 36.8k top | front end 3.9k; mode 13.5k with GS_GUARD = 0; no adapters of its own; backbone shared in float_top |
| tail resolution claim | documented limit (above); wrong claim removed |
| "testbench never ran" | it runs (above); cpp_mirror.py deleted |
| doc nits (BRAM count, 65 cycles per row) | numbers from this csynth (8 BRAM18 row buffer, 34-clock scale row) |
| NaN became a finite scale | stored_scale keeps it non-finite; checked |
| weak validation | validate_hw is the rule (pitches, alignment, extents); GS_GUARD re-checks only what could deadlock or corrupt (op, K, N, B, NP, p4) |
| cross-master read; rows loop not flattened | io only (the GEMM writes the input through io too); dense paths: one burst per item and per 64-row chunk (census: 197 / 197) |
| reviewer regenerated the tables | gelu_silu_tables.h is byte-identical to a18ee58 |

## Integration into float_top

1. `#include "gelu_silu/gelu_silu.h"` and dispatch OP_GELU / OP_SILU_MUL to `gs::act_mode(c, io_rd, io_wr)` (or
   `act_unit`) from one site. Build with `-DGS_GUARD=0` (-384 LUT).
2. In the shared row unit, take `gs_fe` as the GELU / SILU front end. Feed it from the shared reader, using the dense
   rule (gap-dropping, fused gate|up rows), and send its y to the shared absmax / row buffer / quantizer / writer.
3. Choose NSB (above).
4. Unknown (not measured): cosim, post-route LUT, board DDR latency and burst efficiency.
