# RoPE: the ROPE mode of float_top (fpga/kernels/rope)

Rotary position embedding for SmolVLA checkpoint -12392 on the KR260, as the ROPE opcode (0x60) of the integration
contract (`fpga/kernels/CONTRACT.md` 1.1, 2.1, 2.2, 3.3, 4.1). It applies lerobot's `apply_rope`
(`smolvlm_with_expert.py`) to Q and K after the q/k projections and writes one of three outputs: FP32 rows (in place
allowed), int16 rows + per-(row, head) scales through the contract quantizer, or the KW weight tiling of a key blob.

Status (2026-11-21 evening, conformed to contract 1.0; resumed after the 16:54 laptop crash, all runs repeated):

- **C model bit-exact** with the torch emulation and with `contract.h`'s own reference: `tb_rope` 0 failures on
  17 cases x 14 layouts, every word of the simulated DDR compared (13.9 M words), every read unit checked against
  the record's declared inputs (140,390 units, 0 outside). Every run goes through the sequencer's own
  `seq::dispatch` -> `rope::rope_unit`. The threaded bounded-FIFO model (deadlock check) passes the same.
  Mutations: 4 of 4 caught in the final run (10 in the earlier session: 9 caught, 1 equivalent).
- **Full policy** on -12392's 3 frames, contract deployment: max |err| 1.10e-4 / 1.41e-5 / 1.81e-5 (gate 0.05).
- **csynth at 4.0 ns**: estimated 3.064 ns (326 MHz), every loop at its II (1, or 16 / 8 for the burst units).
  rope_mode 18,540 LUT, 15,532 FF, 22 DSP, 0 BRAM. The rope-specific front end is 4,460 LUT and 8 DSP.
- No cosim, Vivado, board run or float_top integration (see "Not verified").

## What it computes

- `x` is `[B, L, H, 64]`, `x1 = x[..., :32]`, `x2 = x[..., 32:]`: `y1 = x1 cos - x2 sin`, `y2 = x2 cos + x1 sin`,
  with `cos/sin(pos / 10000^(2i/64))` from torch's FP32 (the host writes the table, `rope_emu.table_rows`).
- In the contract list (`contract_plan.py`) there are **271 ROPE records per chunk**:

  | shape (rope_cost.py) | records | (row, head) segments | ms at 250 MHz |
  |---|---|---|---|
  | Q int16, LM: N 45, NP 48, H 15, p4 = -3 | 15 | 10,800 | 0.72 |
  | Q int16, expert self and cross: N 50, NP 52, H 15, p4 = -3 | 160 | 124,800 | 8.33 |
  | KW, LM even layers: N 45 keys, NP 64, H 5 | 8 | 2,560 | 0.18 |
  | KW, expert self (suffix keys at blob row 64): N 50, NP 64, H 5 | 80 | 25,600 | 1.83 |
  | FP32 in place, LM odd layers + layer 15 K: N 45, H 5 | 8 | 1,800 | 0.13 |
  | **total** | **271** | **165,560** (153,725 live, 11,835 zero rows) | **11.19** |

  LM layer 15's Q is dead work and is skipped (CONTRACT.md 5). The hooked torch policy still computes it: 272
  `apply_rope` calls, 165,920 (token, head) rows per chunk. All 271 records pass `contract.validate`.
  The contract budgeted ROPE at 11.0 ms (section 6.2).

## Contract conformance

| clause | here |
|---|---|
| mode function | `uint32_t rope::rope_mode(const ctr::Cmd& c, const Word* hp, const Word* xmem, Word* ymem)`, INLINE off, one call site: `case ctr::OP_ROPE: return rope::rope_mode(c, w0, xmem, ymem);`. With the sequencer's unit signature (`seq_core.h` "Units"): `rope::rope_unit(c, hp0, hp1, hp2, hp3, io_rd, io_wr)` (INLINE; hp1..hp3 unused). tb_rope runs every record through `seq::dispatch` with a Units policy whose `row()` calls `rope_unit`; rope_top synthesises through `rope_unit`. No m_axi adapter, decode or register of its own |
| ports | x rows read on io (`xmem`), table rows on hp0 (`w0`, a host constant), every write on io (`ymem`). The standalone top has exactly float_top's `hp0` / `io` bundles, a 7-bit s_axi_control and an int return |
| record | `ctr::Cmd` from `ctr::decode` of a 256-byte record. K = 64 H, N, B; slots A, C, E, F; p1 = SEG = 64, p2 = SEGC, p3 = SEGF, p4 = SHIFT, p5 = NP; flags F_QOUT / F_KW |
| layouts (C1) | A: FP32 rows `A + b A.stride + n A.pitch + 256 j` (head j). E: table row n at `E + b E.stride + n E.pitch`. C int16 / FP32: segment j of row n at `C + b C.stride + n C.pitch + j p2`. F: `F + b F.stride + j p3 + 4n`, pad4(NP) floats per vector, pad lanes 1.0. KW: `C + b C.stride + j p2 + ((n%32)/8) C.pitch + ((n/32) 64 + d) 16`, lane n%8 |
| quantizer (C2) | the FP32 rotation (`rope_emu.to_f32`) per (row, head) through `quantize/quant_int16.h`: `scale_bits`, `recip_step` (QZ_RECIP_BITS = 1), `quant_elem`, `stored_scale` (s 2^p4; an all-zero segment stores 1.0; a non-finite segment stores its absmax) |
| rows [N, NP) | zero rows: q = 0 with scale 1.0. Q stacks heads, KW pads the blob to 32-key groups. They re-read row N - 1, so no read leaves the declared extents (tb: every read unit is checked against the A and E extents) |
| 64-bit addresses (C3) | each address is a 36-bit word index (byte address < 2^40 by validate_hw / validate). Running sums only, no `int`, no address multipliers in the movers (0 DSP in rd / pk / wr) |
| checks | rope_mode re-checks the fields that size its loops and returns ME_DIMS / ME_FLAGS / ME_PARAM with nothing read or written: op, p1 = 64, H = 1..64, N and B 1..65535, NP in [N, 65535], NP % 32 for KW, NP == N for FP32, QOUT with KW, \|p4\| <= 8. tb: 11 malformed records, memory untouched |
| ordering (R-ORD) | FP32 C == A in place is allowed: every segment is written after it was read. The mode returns after its writes (HLS waits for the B responses) |

## Defects of the review, and what changed

The integration inputs list these defects for this kernel (rope review + CONTRACT.md 9, #18-21):

| defect | fix |
|---|---|
| MODE_INT16 used its own rule: power-of-two scale, round half away from zero, zero row s = 0 (0 % of q equal to the contract quantizer) | retired. The int16 / KW outputs are the contract quantizer applied to the FP32 rotation. tb: q and s equal `ctr::ref::quant_row` (an independent C++ path) and `contract.quant_rows` (q.bin / s.bin) on every segment |
| table / record mismatch (position-indexed FP32 table + D positions) | E holds one Q1.17 table row per row (`rope_emu.table_rows`, CONTRACT.md 1.1); no D slot |
| Q written token-major; C.pitch and C.stride ignored | every output follows the record's slots: head-major Q (p2 = one head block) as the list uses, in-row segments, item strides, the KW tiling with port pitch, a blob group offset |
| 9-bit scale-buffer index aliased above 512 entries; H = 0 / H >= 64 deadlocked or misaligned | no scale buffer: scales leave in 8-word units through a 16-word ring. Counters are sized from the contract limits (segments 40-bit, H 7-bit, N / NP / B 17-bit). H = 64 is tested (synth6_40x64). Out-of-range records are rejected before anything runs |
| 2^-126 flush boundary: emulator and hardware disagreed (12 mismatches) | the emulator now rounds to 24 bits, then flushes (the hardware's order). The crafted boundary case (`boundary_crafted_40x2`) and a random one (`boundary_64x2`) pass bit-exactly |
| NaN / Inf inputs treated as finite | documented as undefined values. Control flow never depends on data, so no hang and no write outside the extents. A non-finite segment stores its absmax as the scale, so the divergence reaches the next GEMM |
| "no address multipliers" was not true (flattened loops made t·pitch multipliers) | now true: rd, pk and wr use running sums, 0 DSP. The 22 DSP are rot 8, qt 12, and 2 for the segment count in the mode's prologue |
| "core total" understated the marginal cost | resources below give the whole mode and separate the rope front end from the parts the shared row unit replaces |
| 272 calls stated; the list has 271 | 271 records (layer 15 Q skipped); the torch hook still counts 272 |
| per-token DDR latency on pitched rows | reads are fixed 16-word units (one (row, head) segment) at II = 16, and writes fixed 8-word units at II = 8. csynth infers 16-beat bursts on io and hp0 and 8-beat bursts on io, pipelined across units (no per-row latency in the schedule) |
| the WIP (paused) version: 32,018 LUT, 29 DSP, rc_stage at -0.33 ns; its KW transposer alone 9,257 LUT | rebuilt back end (below): 18,540 LUT, 22 DSP, 3.064 ns |

## Resumed after the 16:54 crash (2026-11-21)

Nothing was lost in this directory: the working tree equalled WIP commit 6c325c6, and no file was truncated or
zero-filled. Every test, the policy check, the cost model and csynth were re-run on the final sources (numbers
above and below are from those runs). Changes in this session:

- tb: a read-extent check (`ROPE_TB_RDCHK` hook in `rd`). The earlier tb could not see a read outside the inputs
  whose data is then zeroed; such a mutant (rows >= N re-reading past row N - 1) now fails.
- tb: every record runs through the sequencer's `seq::dispatch` with a Units policy; `rope::rope_unit` is the
  adapter with `seq_core.h`'s unit signature, and rope_top synthesises through it.
- The segment absmax left the front end (`emit_stage`) for a backbone stage (`am_stage`), matching the row unit's
  `[front end] -> amx` order (quant_mode.h) and gelu_silu's `gs_fe`. Front end 4,799 -> 4,460 LUT.

## Arithmetic (bit-exact model: `rope_emu.py`)

For group u of a segment, the values are `(x1[2u], x1[2u+1], x2[2u], x2[2u+1])`, lanes 0..3:

```
e_j = FP32 biased exponent (e = 0: zero / denormal -> 0), E = max_j e_j
A_j = sign_j * (((1 << 23 | frac_j) << 2) >> (E - e_j))           27-bit signed (0 if shift > 26)
O0 = A2 NS0 - A0 NC0   O1 = A3 NS1 - A1 NC1                          (y1[2u], y1[2u+1])
O2 = -(A2 NC0 + A0 NS0)   O3 = -(A3 NC1 + A1 NS1)                    (y2[2u], y2[2u+1])
value = O * 2^(E - 169),  |O| < 2^44,  NC = round(-cos 2^17), NS = round(-sin 2^17) (18-bit, the table)
Lg = bitlen(max_j |O_j|) in [42, 45] if E > 0 (O0^2 + O2^2 = (A0^2 + A2^2)(NC^2 + NS^2)), group all zero if E = 0
N_j = |O_j| >> (Lg - 26)  (26 bits),  Mg = E + Lg
FP32 rotation: sign * N * 2^(Mg - 195) rounded to 24 bits (RNE), then |y| < 2^-126 -> +0, >= 2^128 -> inf
int16 / KW:    contract quantizer over the segment's 64 FP32 values (CONTRACT.md 2.1), stored scale s 2^p4
```

The rotation is not bit-equal to torch FP32 (1.7 % of values are equal). The error comes from the Q1.17 table and
from the block alignment: within a group, the smaller lanes keep only the bits above the largest lane's alignment.
Relative to each row's max |result| that error is at most 1.05e-5 (below).

## Processes (rope_mode.h, one DATAFLOW region per command)

Segments are processed head-major, in the order (b, j, n). That is quant_mode.h's QUANT_W order: a head's rows are
consecutive, so its scales leave in units and its keys form KW blocks.

| process | what | II |
|---|---|---|
| `rd` | per segment one 16-word x unit (row min(n, N-1), head j; zeroed for n >= N) on io and one 16-word table unit on hp0 | 16 |
| `rot_stage` (rope_core.h) | x / table ping-pong, 8 DSP rotation per group | 1 |
| `emit_stage` (rope_core.h) | group normalization, FP32 in natural order | 1 |
| `am_stage` | segment absmax (quant_mode.h's `amx`); the words go on into the fw FIFO | 1 |
| `sc_stage` | `scale_bits`, `stored_scale`, divider seed | 1 |
| `rc_stage` | R = floor(2^49 / S): 2 iterative restoring dividers (`recip_step`, 1 bit a clock), round robin, one R per 13.5 clocks | 1 |
| `qt_stage` | 4 lanes of `quant_elem` (12 DSP): 16 FP32 words -> 8 int16 words + the stored scale; FP32 passes | 1 |
| `kw_stage` | F_KW: 8 keys x 8 int16 words -> 64 words of 8 keys, lock-step ping-pong (8 LUTRAM banks + 8:1 lane muxes); else pass-through | 1 |
| `pk_stage` | 8-word write units with addresses: data, then scale units (32 scales per unit; a vector that does not end on a unit gets a shift-back unit over its last 8 words; NP <= 28 Q vectors make the command single-word) | 1 |
| `wr_stage` | one 8-beat burst per unit, or single words | 8 / 1 |

The rope-specific part is `rot_stage` + `emit_stage` (the "front end" of CONTRACT.md 4.1). rd / am / sc / rc / qt /
kw / pk / wr are the reader, absmax, quantizer and writer that the shared row unit (`quantize/quant_mode.h`, owned by
the quantize agent) provides for every row op. Plugging the front end into its `[front end]` slot (ahead of `amx`)
needs:

- the 16-word segments in (b, j, n) order;
- the matching table unit from hp0 with every segment;
- 16 FP32 words per segment out, natural order (no absmax: that is the backbone's `amx`, as for gelu_silu's `gs_fe`).

FIFO depths are the `ROPE_D_*` macros. The deadlock model found these minimums (every other FIFO at depth 1, full
vector set): fw >= 16 (a segment's 16 words wait there for its scale; 15 deadlocks) and ss >= 8 (KW: a block's 8
scales; 7 deadlocks). The pragmas use 64 and 32.

## Resources (Vitis HLS 2022.2 csynth, xck26-sfvc784-2LV-c, `build/rope_p40`)

Final sources (rope_core.h 234cfbd2, rope_mode.h 19674fcf, rope_top.cpp 10760345; shared quantize/quant_int16.h
700be612, contract.h cedf10d5; sha1 prefixes). At 4.0 ns: estimated **3.064 ns** (326 MHz). The worst path is
rc_stage's FIFO read, -0.14 ns against the 2.92 ns HLS budget and under the contract's 3.6 ns gate. Every loop
reaches its II. Bursts inferred: 16-beat reads on io and hp0, 8-beat writes on io.

| block | LUT | FF | DSP | BRAM | latency / note |
|---|---|---|---|---|---|
| rot_stage | 1,801 | 2,341 | 8 | 0 | rope front end |
| emit_stage | 2,659 | 1,570 | 0 | 0 | rope front end |
| **rope front end** | **4,460** | **3,911** | **8** | **0** | what ROPE adds to the shared row unit |
| rd | 1,369 | 860 | 0 | 0 | depth 25 |
| am (segment absmax) | 574 | 418 | 0 | 0 | backbone (`amx`) |
| sc + rc | 983 + 1,165 | 509 + 407 | 0 | 0 | |
| qt (4-lane contract quantizer) | 3,709 | 2,856 | 12 | 0 | |
| kw (KW transposer) | 1,663 | 2,447 | 0 | 0 | |
| pk + wr | 1,752 + 707 | 1,175 + 473 | 0 | 0 | |
| FIFOs (15, LUTRAM / SRL) | 1,167 | 1,485 | 0 | 0 | |
| rope_df (dataflow region) | 17,612 | 14,546 | 20 | 0 | |
| **rope_mode** | **18,540** | **15,532** | **22** | **0** | prologue: checks + segment count (2 DSP) |
| rope_top (record fetch, decode, m_axi + s_axilite adapters) | 27,603 | 20,342 | 22 | 0 | the adapters are float_top's |

- Moving the absmax out of the front end (this session) took the front end from 4,799 to 4,460 LUT and the mode
  from 18,262 to 18,540 LUT (+278: one more process and FIFO in the standalone build; in the row unit the absmax
  is the shared `amx`, counted once). DSP, timing and II are unchanged.
- The shared quantizer header changed twice while this session ran (its owner's work). An intermediate version
  (DSP right shift in `rne_sat`, since made opt-in as QZ_SHIFT_DSP) gave qt 3,885 LUT / 16 DSP and the mode
  18,414 LUT / 26 DSP (`build/rope_p40_unit`). The numbers above use the fabric default.
- At 3.333 ns (`build/rope_p3333`, the sources before the absmax split, not re-run) the worst estimated path is
  2.86 ns (qt), -0.43 ns against the 2.43 ns HLS budget: 300 MHz is a nominal miss in HLS terms; 250 MHz is the
  runtime clock (set_pl_clock).
- The first rebuild (`build/rope_p40_v2`) was 21,442 LUT at 3.92 ns. Three changes took it to 18.3k:
  - pk's boundary decisions registered one iteration ahead (pk 3.92 ns -> within 2.92 ns; 2.54 ns in the 3.333 ns
    run);
  - STABLE scalars, which removed about 30 propagation FIFOs (2.6k -> 1.1k FIFO LUT, 3k FF);
  - iterative dividers instead of `qz::recip_unrolled` (3,614 -> 2,172 LUT).
- The paused WIP (`build/rope_p40_v1`) was 32,018 LUT / 29 DSP; the prototype before the contract 18,363 LUT incl.
  adapters with its power-of-two quantizer.

**Against the budget (CONTRACT.md 4.2).** The row unit, which holds all 8 row ops, is capped at **15k HLS LUT,
50 DSP and 10 BRAM36**.

- ROPE's own part (rot + emit) is 4.46k LUT, 8 DSP and 0 BRAM: 30 % of the LUT cap, 16 % of the DSP cap, 0 % BRAM.
- This build's equivalents of the shared parts add up to 14.1k LUT and 14 DSP (qt 12, prologue 2): reader,
  absmax, quantizer, writer, KW transposer, FIFOs and glue.
- So backbone + rope front end is 18.5k LUT here, **3.5k over the whole row-unit cap before NORM, GELU/SiLU or
  QUANT_WT are added**. The standalone ROPE mode does not fit the cap on its own; only the front end is meant to
  enter the row unit.
- With the rope front end at 4.46k, the cap leaves 10.5k LUT for the shared backbone and every other front end
  together (gelu_silu's front end is 3.9k). This build's backbone alone is 14.1k, and quantize's standalone builds
  are larger, so the cap needs a much smaller shared backbone than any built so far, or the contract's fallbacks
  (section 4.2).
- This is for the contract owner and the row-unit (quantize) agent to settle. Cuts left in the front end without
  changing the bit-exact arithmetic: merging rot and emit into one process (est. 0.3-0.5k LUT, not tried).

## Throughput and time

- About 16 clocks per (row, head) segment, 4 values a clock. rd's II = 16 unit is the bound: every later stage takes
  16 clocks per segment or less. pk writes per segment: 8 words for Q (plus a quarter scale word per row), 8.25 for
  KW, 16 for FP32.
- Per record: `80 (sequencer fetch) + about 250 (pipeline fill) + 64 (KW) + 2 x DDR latency + 16 x B x H x NP`
  clocks (`rope_cost.py`).
- **Per chunk: 11.19 ms at 250 MHz**, assuming 100 clocks of DDR latency (unmeasured). With zero latency it is
  10.98 ms. The 11,835 zero rows cost 0.76 ms of it. The packer could write them without reading or rotating, which
  has not been done.
- Traffic per chunk: 2.65 M x words + 2.65 M table words (hp0) read; about 1.3 M int16 / KW words + 29 k FP32 words
  written.

## Tests

| test | result (2026-11-21) |
|---|---|
| `rope_emu.py` selftest | FP32 max err / head amax 5.97e-6; contract-quantized 1.92e-5; 2^-126 boundary PASS |
| `tb_rope.exe build/vectors` | **PASS**: 17 cases x 14 layouts (228 runs) at bases 0x4000_0000, 0x8_0000_0000 and 0xFF_0000_0000 (two layouts), 13,893,896 memory words compared, 140,390 read units checked (0 outside the declared inputs), 0 failures, plus 11 rejected records with memory untouched and nothing read. Every run through `seq::dispatch` -> `rope_unit` |
| `tb_rope_dfsim.exe` (every process a thread, FIFOs bounded at the pragma depths) | **PASS** (same counts). Negative controls: `-DROPE_D_FW=8` and, with every other FIFO at 1, fw = 15 or ss = 7: DEADLOCK reported, as it must; fw = 16 / ss = 8 pass |
| mutations (scratch copies, `--quick`) | final sources: 4 of 4 caught: KW lane select, Q pad-lane scale, rows >= N re-read past row N - 1 (caught only by the new read check: its outputs were right), table row step. Earlier session, 9 of 10: tail unit, pad lanes, zero rows, KW lane select, KW group step, p4 shift, scale-vector step, scale ring index, FP32 rounding; the `recip_result` clamp mutant is equivalent (for S = 2^23 the restoring division already gives 2^26 - 1) |
| `check_policy.py` | below |
| `kernel_error.py 0` | below |
| `rope_cost.py` | 271 ROPE records, 0 invalid; the table above |

Each layout run builds a real 256-byte record (`ctr::encode` -> `ctr::decode`). It requires `ctr::validate` and
`ctr::validate_hw` to pass, fills the DDR with a sentinel, runs the record through `seq::dispatch`, and compares the
whole memory with the expected image:

- the outputs where `contract.h`'s address helpers put them;
- the sentinel or the inputs everywhere else, so a write outside the extents fails the test.

It also requires every read unit of `rd` (reported through the `ROPE_TB_RDCHK` hook, which synthesis never sees) to
lie inside the A rows [0, N) and E rows [0, N) of the record, and exactly one x and one table unit per segment.

Layouts:

- FP32 contiguous, in place inside a pitched q|k|v row, and head-major;
- Q head-major with stacked zero rows and p4 = -3, and Q in-row with p4 = +2 (pad lanes);
- KW, and KW at blob group 1 (the expert suffix keys);
- two-item versions of FP32, Q and KW.

Cases:

- 5 real calls of frame 0 (prefix L0 Q, L1 K, expert step 0 L0 Q / K, L1 cross Q);
- synthetic: exponent spread, zeros, -0, denormals, all-zero and tiny segments, T = 1 / 3 / 7 / 13 / 40 / 65 / 130,
  H = 1 / 2 / 3 / 5 / 15 / 64;
- all-equal rows; the 2^-126 boundary (random and crafted); power-of-two scales; FP32 overflow (FP32 only).

## Accuracy

Full policy (`check_policy.py`): `apply_rope` monkeypatched with the bit-exact model inside
`predict_action_chunk(batch, noise=noise)`, max |err| against `ref_chunk_raw` in normalized action units (gate 0.05),
`check_policy_results.jsonl`:

| variant | frame 0 | frame 3563 | frame 7489 |
|---|---|---|---|
| torch (laptop, unpatched) | 4.71e-6 | 3.58e-7 | 2.98e-7 |
| hook_exact (harness check) | 4.71e-6 | 3.58e-7 | 2.98e-7 |
| fmt_contract (exact torch RoPE, then the contract quantizer on Q and K: the int16 format alone) | 2.65e-5 | 5.25e-6 | 1.03e-5 |
| kernel_f32 (every RoPE = the FP32 rotation) | 5.04e-6 | 1.07e-6 | 5.96e-7 |
| **contract** (the deployment: Q = ROPE F_QOUT; K = ROPE F_KW or FP32 + QUANT_W, both the contract quantizer per (key, head); the cross k_proj reads the FP32 rotated K) | **1.10e-4** | **1.41e-5** | **1.81e-5** |

Element error over all 272 calls / 165,920 rows of frame 0 (`kernel_error.py`), relative to each row's max |torch|:

| output | max | rms |
|---|---|---|
| FP32 rotation | 1.05e-5 | 1.18e-6 |
| int16 (q · s, contract quantizer on the rotation) | 2.46e-5 | 8.82e-6 |
| format reference (torch FP32 -> contract quantizer) | 1.53e-5 | 8.74e-6 |

Not modelled in this check: the int16 linears, the int16 attention products, V and cross K' quantization, the folded
constants. Those belong to the other kernels and to the golden executor (T-P1, not built).

## Files and commands

| file | purpose |
|---|---|
| `rope_core.h` | the rope front end: `rotate_group`, `to_f32`, `rot_stage`, `emit_stage` |
| `rope_mode.h` | the ROPE mode: `rope::rope_mode`, `rope::rope_unit` (the sequencer's unit signature) and the reader / absmax / quantizer / packer / writer processes; `ROPE_DFSIM` threads; `ROPE_TB_RDCHK` read hook (tb only) |
| `rope_top.cpp` | standalone csynth top: float_top's hp0 / io bundles, one record fetch at K (list base >> 4) + `ctr::decode` + `rope_unit` -> `rope_mode` |
| `tb_rope.cpp`, `dfsim/hls_stream.h` | host testbench (includes `../sequencer/seq_core.h` for `seq::dispatch`); bounded blocking `hls::stream` for the deadlock model |
| `rope_emu.py` | bit-exact torch model (`rope_rows`, `to_f32`, `table_rows`, `quant_segments` = `contract.quant_rows`) |
| `gen_vectors.py` | test vectors (real + synthetic) into `build/vectors` |
| `check_policy.py`, `kernel_error.py` | full-policy and element-error checks (logs: `check_policy.log`, `check_policy_results.jsonl`, `kernel_error.log`) |
| `rope_cost.py` | ROPE records of the contract list, validated, with this mode's time |
| `run_rope_hls.tcl` | csynth (`ROPE_PERIOD`, default 4.0) into `build/rope_p<period>/` |

```sh
export HF_HUB_OFFLINE=1   # and HF_HUB_CACHE=<a cache with SmolVLM2-500M-Video-Instruct> (see Not verified)
PY=/c/kp/lerobot-venv/Scripts/python.exe
$PY fpga/kernels/rope/rope_emu.py && $PY fpga/kernels/rope/gen_vectors.py
$PY fpga/kernels/rope/check_policy.py --out <file.jsonl> ; $PY fpga/kernels/rope/kernel_error.py 0 ; $PY fpga/kernels/rope/rope_cost.py
# testbench: Vitis MinGW on a clean PATH, compile and run
export PATH="/c/Xilinx/Vitis_HLS/2022.2/tps/win64/msys64/mingw64/bin:/c/Xilinx/Vitis_HLS/2022.2/tps/win64/msys64/usr/bin:/c/Windows/System32:/c/Windows"
cd fpga/kernels/rope
g++ -std=c++14 -O2 -ffp-contract=off -static -I/c/Xilinx/Vitis_HLS/2022.2/include tb_rope.cpp -o build/tb_rope.exe
g++ -std=c++14 -O2 -ffp-contract=off -static -DROPE_DFSIM -Idfsim -I/c/Xilinx/Vitis_HLS/2022.2/include tb_rope.cpp -o build/tb_rope_dfsim.exe -pthread
./build/tb_rope.exe build/vectors && ./build/tb_rope_dfsim.exe build/vectors
# csynth (PATH without Git dirs)
ROPE_PERIOD=4.0 /c/Xilinx/Vitis_HLS/2022.2/bin/vitis_hls.bat -f run_rope_hls.tcl
```

## Not verified

- No C/RTL co-simulation (csim is broken here), Vivado or board run. Resources and timing are HLS estimates; the
  post-route LUT ratio of this mix is unknown.
- DDR latency and sustained bandwidth on io / hp0 are unknown. The 100-clock latency in `rope_cost.py` is an
  assumption. Burst efficiency of 16-beat reads and 8-beat writes on the KR260 is unmeasured (BT-2).
- Not integrated into float_top or the shared row unit. The front end's hook into `quant_mode.h` is described above
  but not built (that file belongs to the quantize agent).
- Reads and writes outside the declared extents are checked in the C model only (the tb's read hook,
  `ROPE_TB_RDCHK`, and the whole-memory compare). The RTL's bus behaviour is not simulated (no cosim).
- The base model's global HF cache entry (`~/.cache/huggingface/hub/models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct`)
  is broken on this laptop: its snapshot folder is empty and only a 395 MB `.incomplete` blob from 2026-11-20 22:55
  remains. The checks above ran with `HF_HUB_CACHE` pointed at a complete copy in the session scratchpad
  (model.safetensors is a hard link to the 2026-10-15 original). The global cache was not touched.
