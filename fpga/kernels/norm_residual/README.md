# norm_residual: the NORM front end of the row unit (opcode 0x30), integration contract 1.0

LayerNorm (vision, `F_LN`) or RMSNorm (LM prefix, action expert) of FP32 rows. Output is either int16 rows plus
one stored scale per row (`F_QOUT`) or FP32 rows (vision post_layernorm). The residual add is **not** here any
more: it is the GEMM epilogue (CONTRACT.md 2.3, `D = C = R`). The directory keeps its name.

Status (2026-11-21, conform pass 3, after the 16:54 crash):

- **Structure.** This kernel now contributes only the **NORM front end**, `nr::norm_front`, to the row unit
  (CONTRACT.md 4.1: "norm_residual: statistics + affine only"). The reader, absmax, quantizer and writer are the
  row unit's backbone (`quantize/quant_mode.h`), used unchanged. The kernel's own stand-ins for them are gone.
- **Tests.** The C++ is bit-exact against its torch model on 57 DDR-image cases (6 from real -12392 activations,
  21 rejected records). That holds in the sequential build and in a threaded bounded-FIFO build of the whole
  region, backbone included. With the front end switched off, 240 random QUANT_X / QUANT_W / QUANT_WT / COPY
  commands give byte-identical images to the backbone alone.
- **Policy.** The full-policy check passes on -12392's 3 frames (1.3e-3 / 1.4e-4 / 1.5e-4; gate 0.05).
- **csynth** at 4.0 ns estimates 3.48 ns (288 MHz). Every flat loop of the front end is II = 1.
- **Area.** The front end is 10,240 HLS LUT, 40 DSP and 16 BRAM18. That is **over any fair share of the row-unit
  cap** (section "Area").
- **Not done.** No Vivado run, no RTL co-simulation (csim and cosim are broken here), no board run.

## Files

| file | what |
|---|---|
| `norm_residual.h` | header-only HLS code: `nr::norm_front` (the front end), `nr::front_plan`, and `nr::norm_mode(cmd, hp, xmem, ymem)` (the backbone with the front end in its slot, for NORM alone) |
| `nr_top.cpp` | standalone csynth top: record fetch over io + `ctr::decode` + `ctr::validate_hw` + one `norm_mode` call site; float_top's bundle names |
| `run_nr_hls.tcl` | csynth script (env options `NR_PERIOD`, `NR_FMUL`, `NR_FADD`, `NR_SQ`, `NR_KFOLD`, `NR_URAM`) -> `build/nr6_*` |
| `tb_nr.cpp`, `run_tests.sh` | host testbench (g++): unit tests, whole-DDR-image comparison byte for byte, front-end pass-through check, threaded build |
| `dfsim/hls_stream.h` | threaded bounded-FIFO model: replaces `<hls_stream.h>` in the `-DNR_DFSIM` build, for this kernel's and the backbone's processes alike |
| `nr_emul.py` | bit-exact numpy model of the mode (`python nr_emul.py` runs its own definition checks) |
| `gen_vectors.py` | the DDR-image cases (synthetic, rejects, `--real` -12392 activations) |
| `policy_check.py`, `policy_check.json/.log` | full-policy check on -12392's 3 frames, norm counts and row statistics |
| `count_cost.py` | NORM time per chunk: the 386 NORM records of `contract_plan.py`, kernel model against the contract model |

## Contract conformance (CONTRACT.md 1.0)

| clause | how |
|---|---|
| unit (4.1) | `nr::norm_front(hp, FC, xs, out)` sits in the backbone's front-end slot (`qz_rd -> norm_front -> qz_amx`). It takes and gives the backbone's `qz::XT` tokens and passes their flags unchanged. With `FC.on = false` (any other opcode) every token passes through: measured identical images on 240 commands |
| mode function (3.3, 4.1) | `nr::norm_mode(const ctr::Cmd&, hp, xmem, ymem)`, `INLINE off`, one call site: `case ctr::OP_NORM: r = nr::norm_mode(c, w2, xmem, ymem);`. It maps NORM to the backbone's plan: F_QOUT = QUANT_X rows (p1 = 0), else COPY rows, through `qz::row_plan`. Fields used: K, N, B; slots A, C, E, F; p0 eps, p4 shift, p5 NP; flags F_QOUT and F_LN. It returns ME_OK, or ME_DIMS without touching memory when a counter could not hold the command (a guard behind `validate_hw`) |
| 64-bit addresses (3.2) | slot addresses are absolute. The backbone narrows them to 36-bit word indices after `validate_hw` (< 2^40), and gamma's index is the same 36 bits. Tested at bases 0xFFFF_0000 (across 4 GB), 0x8_7000_0000 (KR260 high DDR), 0xF_FFF8_0000, and 0x1_0000_0000 (pass-through) |
| layouts (1.1) | A: FP32 rows (pitch, stride). C: int16 rows (K/8 words) or FP32 rows. Rows [N, NP) are written as zeros with scale 1.0. F: one scale per row, `pad4(NP)` floats per item, pad lanes 1.0. E: gamma[K], with beta at E + 4K |
| ports (3.3) | activations are read **only on io** (xmem, by `qz_rd`). Gamma and beta (host constants) are read once per command on one hp port, read only. Every write is on io (ymem, by `qz_wr`). No pointer argument, no register, no m_axi adapter of its own |
| quantizer (2.1) | the backbone's: `quantize/quant_int16.h` through `qz_ss / qz_sr / qz_qt`, s = fl32(max(amax, 1e-30)/32767), q = clamp(rne(fl32(z/s)), ±32767), stored s·2^p4, all-zero rows 1.0. The own 1/trunc17 rule is gone |
| FP32 semantics (2.3) | every FP32 core operation flushes denormals with the sign kept. The g++ build uses `ctr::ref::fmul_hw / fadd_hw`, so the host model is the hardware |
| retired (2.3, 3.3) | F_RESID, F_SUM, the records layout and the hp0/hp1 activation reads are gone. Those flags give ME_FLAGS in `validate_hw` (tested) |
| limits (3.3) | K 8..1024 with K % 8 == 0; N and B 1..65535; NP in [N, 65535] with F_QOUT, NP = N otherwise; p1 = 0. Counters are sized from these |

## Integration into the row unit (for the integrator; `quant_mode.h` is the quantize kernel's file)

1. In `qz::row_plan`, map `OP_NORM` to `KX` (F_QOUT) or `KC` (no F_QOUT). This is exactly what `norm_mode` does
   by calling `row_plan` on a copy with `op = OP_QUANT_X / OP_COPY`.
2. In `qz::row_df`, replace the `xs` edge by `qz_rd -> xs -> nr::norm_front(hp, nr::front_plan(c), xs, xz) -> xz
   -> qz_amx`. Declare xz like xs (16 tokens, LUTRAM). `row_df` gains the hp pointer (`RowUnit::row` already
   receives hp0..hp3; NORM needs one of them, w2 in `nr_top`).
3. Nothing else changes. `norm_front` is a nested DATAFLOW region with its own FIFOs: f2 (512 tokens, BRAM),
   fs / fv / fm (4 each), and the gamma | beta table (512 x 128 bit, BRAM).
4. GELU, SILU and ROPE front ends can share the slot: chain them after `norm_front`, each passing through when it
   is not its opcode, or mux them.

Contract of the slot, as `norm_front` relies on it: for KX and KC, the tokens with `fwd = 1` are the row's K/4
words in row order, and `end` flags the command's last token (a row word). Tokens with `fwd = 0` pass through
unchanged and are not counted. `norm_front` counts K/4 data tokens per row itself, so it does not depend on `last`
or `amx`.

- The backbone changed this during the session. Until 17:59 `qz_rd` dropped the shift-back overlap of a row's
  last unit. Since then (`quant_mode.h` of 18:02) it emits all 16 tokens of that unit, with the overlap marked
  `fwd = 0, amx = 0`.
- Counting every token hung the region on the current backbone (`const_ln720`: K = 720 has a 4-word tail). The
  threaded model's watchdog showed `nr_f2` holding 108 tokens of an incomplete row. Counting only `fwd` tokens
  works with both versions.

## Arithmetic (exact definition; `nr_emul.py` is the reference, `tb_nr.cpp` checks the C++ against it)

Per row of K values:

1. **Statistics: one pass, running exponent, exact integers.** For each 4-value io word, `E = max(E, max
   exponent field of the word)`. When E grows by d, the sums are renormalized:
   `S := floor((S + 2^(d-1)) / 2^d)`, `Q := floor((Q + 2^(2d-1)) / 2^(2d))` (round half up). Then
   `Y_i = sign * round_half_away(m_i * 2^(1 + e_i - E))` (24-bit significand, |Y_i| < 2^25; a zero or denormal
   counts as 0), `S += Y_i`, `Q += Y_i^2`.
   - Without a renormalization (the row's first word holds its maximum exponent, as in every constant row), S
     and Q are the exact sums of the block-fixed values.
   - Each renormalization adds at most half a final LSB (2^-25 of the row max) to S. That is the size of the Y_i
     rounding itself, so it costs no measurable precision (the renorm cases are listed under "Tests").
   - Hardware: the renormalizing round and the accumulate are one add, `floor((y + 1)/2) + a = floor((y + 1 +
     2a)/2)`, and the word's d, sum and square sum pass a pipeline register first. So the loop-carried path is one
     shifter and one adder (3.48 ns estimate; 3.77 ns without this).
2. **Row scalars, correctly rounded, in integers.**
   - `num = max(K Q - S^2, 0)` (LN) or `K Q` (RMS).
   - `mean = rn(S / K) 2^(E-151)` (LN, else 0) and `var = rn(num / K^2) 2^(2 (E-151))`.
   - Results below 2^-126 flush to +0; above the FP32 range they give +inf.
   - Hardware: K Q - S^2 by one signed shift-add accumulator (37 clocks); then two bit-serial dividers with
     remainders of the divisor's width side by side (118 clocks, no dividend normalization). RMS divides Q by K
     (the same rational as K Q / K^2).
   - So a constant LayerNorm row gives `y - mean = 0` exactly and `z = beta`, as torch does (review defect #30).
3. **rstd, correctly rounded, in integers (no FP32 core, no seed ROM).**
   - `ve = max(fl(var + eps), 2^-126)`: an IEEE round-half-even add of two non-negative floats, inputs flushed,
     the smaller operand aligned 1 bit a clock.
   - `rstd = rn(1 / sqrt(ve))`: `D = floor(2^76 / M')` (54 steps), `R = isqrt(D)` (27 steps), with a sticky from
     both remainders; 0 if ve = inf.
   - Proven correctly rounded for every input: `tb_nr --units-exhaustive` checks every significand at both
     exponent parities against the exact midpoint condition in `__int128` (16,777,724 values, all correct, run
     today; the exponent only shifts the result). There are no ties: a midpoint would need 1/sqrt(ve) to be a
     dyadic rational with 25 bits, and 1/sqrt(ve) is dyadic only when M' is a power of 4, where it is an exact
     power of two.
4. **Normalize, FP32 cores, 4 lanes:** `z = ((y - mean) * rstd) * g + b` in that order (RMS: mean = 0, b = 0).
5. **q-out (F_QOUT):** the backbone's contract quantizer on z: `s = fl32(max(amax, 1e-30) / 32767)`,
   `q = clamp(rne(fl32(z / s)), +-32767)`; stored scale `s 2^p4`, all-zero rows 1.0.

Non-finite inputs give undefined values, but no hang and no write outside the declared extents. An exponent field
of 255 is a large finite value to the integer statistics, and a var above FP32 gives rstd = 0, i.e. z = beta. LN
rows whose variance exceeds FP32 (|y| above about 1e19) give z = beta, where torch gives NaN or inf. Real
activations are far from this.

## DATAFLOW region and speed

| process | role | per-row cost (csynth) |
|---|---|---|
| `qz_rd` (backbone) | A over io in 16-word units, kept in flight across rows | ceil(K/64) x 16 clocks |
| `fe_stats` | running exponent, S, Q (4 squares a clock); tokens go on to f2 | K/4 (pipeline depth 8) |
| `fe_scal` | num, mean, var (integers) | 167 clocks (LN), 127 (RMS) |
| `fe_rsqrt` | ve, rstd (integers) | 127 clocks |
| `fe_affine` | gamma \| beta into one dual-port BRAM once per command (512 clocks at most), then z, 4 lanes a clock | K/4 (depth 37) |
| `qz_amx`, `qz_ss`, `qz_sr`, `qz_qt`, `qz_wb`, `qz_wr` (backbone) | absmax + row buffer, scale, reciprocal, quantize, units, writes | K/4, or K/8 words written |

Steady state is `max(read units, K/4, words written, scalar clocks)` per row. All SmolVLA rows are io-bound: LN 768
is 192 clocks a row against 167; RMS 720 is 192 against 127 (the reader's 12 units for 180 words); RMS 960 is 240.
LN rows of K < 672 and RMS rows of K < 512 are scalar-bound.

`count_cost.py` on the 386 NORM records of `contract_plan.py` at 250 MHz:

| group | calls | rows | kernel model | contract model |
|---|---|---|---|---|
| vision LN 768, int16 | 24 | 12,288 | 9.54 ms | 9.69 ms |
| vision post_layernorm 768, FP32 | 1 | 512 | 0.40 | 0.40 |
| LM RMS 960 | 31 | 1,395 | 1.48 | 1.43 |
| expert RMS 720 | 330 | 16,500 | 13.97 | 12.86 |
| **total** | **386** | | **25.4 ms** | **24.4 ms** |

- +1.1 ms against the contract: the backbone reads a 720-value row as 12 units of 16 words (192 clocks for 180).
  A reader with a short last unit instead of the shift-back would recover it.
- With pessimistic assumptions (DDR latency 300, write response 150, call overhead 300 clocks) the total is
  26.1 ms.
- **Unmeasured:** DDR latency, the write-response wait and the call overhead are assumptions.

## Area (csynth, Vitis HLS 2022.2, xck26-sfvc784-2LV-c, 4.0 ns, `build/nr6_fulldsp_fulldsp_dsp_p40`)

| block | LUT | FF | DSP | BRAM18 |
|---|---|---|---|---|
| `fe_stats` | 2,589 | 2,384 | 8 | 0 |
| `fe_scal` | 1,863 | 1,077 | 0 | 0 |
| `fe_rsqrt` | 1,591 | 697 | 0 | 0 |
| `fe_affine` (16 FP32 cores = 2,976 LUT + 32 DSP; gamma \| beta table) | 3,901 | 6,206 | 32 | 8 |
| f2 row FIFO and small FIFOs | ~300 | ~460 | 0 | 8 |
| **NORM front end (`norm_front`)** | **10,240** | **10,825** | **40** | **16** |
| backbone as synthesized here (`qz_rd` 2,634, `qz_amx` 1,031, `qz_ss` 979, `qz_sr` 865, `qz_qt` 4,510 / 12 DSP, `qz_wb` 2,580, `qz_wr` 707, FIFOs) | ~14.3k | | 12 | 8 |
| **`norm_mode`** (backbone + front end) | 25,634 | 24,876 | 53 | 24 |
| standalone `nr_top` (+ fetch, decode, validate_hw, m_axi and s_axilite adapters) | 36,609 | 30,172 | 53 | 24 |

The backbone figures are the quantize kernel's `quant_mode.h` of 18:02 (md5 996dd98c..., it was being edited
during this session). It is counted once for the whole row unit, not per front end. Timing: 3.48 ns estimate (288
MHz); the slowest paths are `fe_stats` 3.48, `qz_sr` 3.40 and `qz_qt` 3.34 ns. Every flat loop is II = 1,
including the backbone's; the only II misses are `ctr::validate_t`'s two loops (II 2 and 4) in the standalone
top's `validate_hw`, which is the sequencer's, not this mode's.

Against the contract (CONTRACT.md 4.2: the **whole** row unit, 8 opcodes, at most 15k HLS LUT, 50 DSP, 10
BRAM36):

- The NORM front end alone takes 68 % of the LUT cap, 80 % of the DSP cap and 8 of the 10 BRAM36 (HLS counts).
- Backbone + NORM (`norm_df`) is 24.5k LUT, 52 DSP and 12 BRAM36: 1.6x the LUT cap, before the GELU / SILU / ROPE
  front ends. The backbone alone is about the cap (~14.3k).
- Row-unit HLS LUT is integer logic plus FP32 cores. With the contract's 0.85 factor, 24.5k is about 21k
  post-route, against the 12.8k booked.

Variants measured (front end only, same clock estimate):

| build | LUT | DSP | BRAM18 | what |
|---|---|---|---|---|
| default (fmul and fadd `fulldsp`), final source | 10,240 | 40 | 16 | |
| default, before the `fwd` fix ("Integration") | 9,951 | 40 | 16 | the reference for the two variants |
| `NR_FMUL=maxdsp` (before the `fwd` fix) | 10,047 | 48 | 16 | worse on both: not used |
| `NR_KFOLD=1` (before the `fwd` fix; not contract 1.0) | 7,956 | 24 | 8 | gamma / beta folded into the next linear: `fe_affine` 1,761 LUT / 16 DSP / 0 BRAM |

Ways down, for the integrator and the contract owner (none is in contract 1.0):

1. **Fold gamma / beta into the next linear** (`W' = W diag(g)`, `b' = b + W beta`; for the post_layernorm,
   through the pixel shuffle into the connector). It removes 2 of the 4 FP32 cores per lane, the table and its
   load. Measured (`build/nr6_fulldsp_fulldsp_dsp_p40_kfold`, same source before the `fwd` fix): the front end
   goes from 9,951 LUT / 40 DSP / 16 BRAM18 to 7,956 / 24 / 8 (-2.0k LUT, -16 DSP, -8 BRAM18). The
   activation-side policy effect was measured too (`kfold` in "Accuracy"); the re-quantization of `W diag(g)`
   was not. This changes NORM's E slot and `pack_float.py` (a shared-contract file), so it needs
   sign-off from both sides.
2. **Share the affine's FP32 cores** with the SILU_MUL / GELU front ends inside the row unit: one 4-lane
   `((x - a) r) g + b` stage that SILU_MUL can also use for its multiply. This is integration work.
3. **2 lanes** in the front end: about half the stats, affine and squares (est. -5k LUT, -20 DSP) for about
   +24 ms per chunk (NORM at K/2 clocks a row). That takes most of the +0.065 s margin. Not recommended.

HLS BRAM counts: HLS counts 8 BRAM18 for each 512 x 128 FIFO or dual-port table. That is 64 Kb, which 4 BRAM18
hold in 512 x 36 simple-dual-port mode. Vivado may use 4 for the FIFO; the true-dual-port gamma | beta table
needs 8 (TDP BRAM18 is at most 18 bits wide), or 2 URAM with `NR_URAM=1`. Not verified without Vivado.

## Accuracy

`policy_check.py --frames 0,1,2 --variants torch,zf32,kernel,torchq16,kfold` (`policy_check.log`, `.json`, run
today). All 91 norm modules use the mode's exact arithmetic (`nr_emul.py`); linears stay FP32, so only the norms
change. The table gives `predict_action_chunk(batch, noise)` against the cluster's `ref_chunk_raw`, max |chunk -
ref| in normalized action units (gate 0.05):

| frame | torch (laptop) | zf32: FP32 z everywhere | **kernel as deployed** (int16 into every linear, FP32 z for post_layernorm) | torchq16: torch norm + the same contract quantizer | kfold (option 1, activation side) |
|---|---|---|---|---|---|
| 0 | 5.13e-6 | 5.72e-6 | **1.34e-3** | 7.79e-4 | 7.83e-4 |
| 3563 | 4.62e-7 | 2.38e-7 | **1.43e-4** | 1.30e-4 | 1.61e-4 |
| 7489 | 4.77e-7 | 2.98e-7 | **1.52e-4** | 1.52e-4 | 1.67e-4 |

- **zf32.** The normalize arithmetic alone is at torch's own laptop-vs-cluster level.
- **The deployed error is the int16 quantization of the norm output.** torchq16 carries the same quantization
  with torch's norm in front. It is the same size on the benign frames (2e-4 class, contract T-P1). Kernel vs
  torchq16: 2.1e-3 / 6.8e-5 / 4.1e-5.
- **Frame 0 is the sensitive frame** (contract T-P1 class: <= 0.006). There, 1-ulp differences of z move the
  chunk by 1e-3. The pre-contract kernel's frame-0 figure was 1.45e-4, with the same per-call accuracy (below).
  This run is 37x under the gate.

Per norm call over every call of the 3 frames, against torch's FP32 norm:

| group | rows | z, of row max | int16 q * s, of row max | q differs from torch norm + contract quantizer | max q diff |
|---|---|---|---|---|---|
| vision LN 768 | 36,864 | 3.70e-7 | 1.54e-5 | 0.0097 % | 1 |
| vision post_layernorm | 1,536 | 3.52e-7 | 1.54e-5 | 0.012 % | 1 |
| LM RMS 960 | 7,776 | 2.85e-7 | 1.54e-5 | 0.0058 % | 1 |
| LM final (unused; skipped in the list) | 243 | 2.37e-7 | 1.53e-5 | 0.0034 % | 1 |
| expert RMS 720 | 48,000 | 3.01e-7 | 1.55e-5 | 0.0078 % | 1 |
| expert final | 1,500 | 2.03e-7 | 1.54e-5 | 0.0053 % | 1 |

The pre-contract kernel's own quantizer differed from the contract's on 1.4-2.9 % of q. With the shared
quantizer the remaining 0.003-0.012 % are elements whose z differs from torch's by an ulp at a rounding
boundary.

Synthetic cases (`gen_vectors.py`): z within 0.8e-7 to 1.7e-7 of a float64 norm, the same as torch's FP32 norm
(0.8e-7 to 2.2e-7). Constant LN rows are exact.

## Review defects and their resolution

| defect (review of the pre-contract kernel; CONTRACT.md 9 #28-31) | resolution | test |
|---|---|---|
| 28.5k LUT / 43 BRAM18; own quantizer next to the quantize kernel's; 4 row FIFOs | statistics + affine only; reader, quantizer and writer are the backbone's; one row FIFO (f2), the running exponent removed the exponent pass | csynth: front end 10,240 LUT / 16 BRAM18 (still over a fair share, see "Area") |
| 38 ms needs the records layout, which the builder does not emit; 53-75 ms otherwise | records layout, F_RESID and F_SUM retired (the residual add is the GEMM epilogue); NORM reads A once and writes C | `count_cost.py`: 25.4 ms on the contract's 386 records |
| `rec = 512` truncation at K = 1024 (hang) | records layout gone | `rms1024_f32`, `ln1024_f32_n260` (two 256-row writer chunks) |
| N >= 65,536 truncation (hang) | `validate_hw` rejects N, B > 65,535 (ME_DIMS); the mode's guard too; counters are 17 bits | `rej_n65536`, `rej_b65536`: rc 0x1301, image untouched |
| LayerNorm mean not correctly rounded: constant rows give ±32767 noise | mean and var correctly rounded in integers | `const_ln720/768/960/1024` (+ FP32): z = beta exactly; `nr_emul.py` self-check: all-equal rows give max\|z\| 0 |
| quantizer not the contract's (q ±1 on 1.4-2.9 %) | the backbone's `quant_int16.h` | all 57 images; policy: q differs from torch + contract quantizer on 0.003-0.012 % only |
| A and B read on hp0 / hp1 (cross-master RAW) | A on io only; B gone | the region reads A only through xmem |
| C and s_tok in the engine's interleaved layout | row-major everywhere (C1); the engine owes E-2 | — |
| test gaps: sequential DATAFLOW, no FIFO-depth check; real cases lack the fused paths | threaded bounded-FIFO build of the whole region (backbone included); fused paths retired | `tb_nr_df.exe`: 57/57 + 40 pass-through, FIFO peaks printed; f2 below one row deadlocks (measured 240 at K = 1024) |
| denormals differ between the HLS cores and the C model | the C model uses `ctr::ref::fmul_hw / fadd_hw` (flush, sign kept) | `eps_denormal`, `edge_*` |
| doc: 413 vs 388 vs 386 calls; behaviour above 1e19 | 386 NORM records (contract_plan); overflow documented under "Arithmetic" | — |

## Tests (`sh run_tests.sh [--exhaustive]`)

- Units:
  - `div_rn`: 400,000 random 72-bit dividends / K or K^2, ties, exact quotients and powers of two, against an
    `__int128` reference. Exact.
  - `fadd_pos`: 2,000,000 non-negative pairs against the host's IEEE add, inputs flushed. Exact.
  - `rsqrt_cr`: an `__int128` reference and the exact midpoint test. 300k random values by default; every
    significand at both parities with `--exhaustive` (16,777,724 values, all correctly rounded).
- DDR images (57): every output byte is compared, and every byte outside the outputs must keep its sentinel.
  - 21 records that `validate_hw` or `decode` rejects, with the image untouched: N = 0 / 65536, B = 65536, K =
    1032 / 12, retired flags, eps 0 / 1, shift 9, SEG 64, misaligned A, short C pitch, a used B slot, FP32 with NP
    or F, address >= 2^40, bad magic, check word and version.
  - LN and RMS; int16 and FP32 out; B = 2 with padded pitches and strides; N = 257 / 260 / 300 with zero rows
    [N, NP) and shifts -3 / +5 / -8.
  - K = 8 / 64 / 720 / 768 / 960 / 1024, so the backbone's single-word and 16-word reader paths and its small
    and 8-word writer paths are all used.
  - Constant LayerNorm rows at K = 720 / 768 / 960 / 1024 (z = beta exactly, q = 0 for beta = 0).
  - Renormalization-heavy rows (an exponent that grows every word, the max in the last word, a 2^60 jump).
  - K = 64 edge rows (zero, -0, denormal, tiny, huge, overflowing LN variance, one-hot), a denormal eps.
  - Bases across 4 GB and in high DDR.
  - 6 real -12392 calls: vision layer 5 ln2, post_layernorm (FP32), LM layers 7 and 15, expert layer 3, expert
    final.
- Pass-through (`--passthrough n`): random valid QUANT_X (whole rows; 64-value segments, head-major or in-row),
  QUANT_W, QUANT_WT and COPY commands above 4 GB, through the backbone with `norm_front` off, against
  `qz::row_mode`, every byte compared: 200 commands sequential, 40 threaded, identical.
- Two builds: sequential C simulation, and `-DNR_DFSIM` (`-Idfsim` first), where every process runs in a thread
  and every stream blocks at its declared depth (the name -> depth map is in tb_nr.cpp). A FIFO deadlock hangs
  there and the watchdog reports it. Both: 57 / 57.

## Running

- **HF cache.** `refs/main` of SmolVLM2-500M-Video-Instruct in the shared cache
  (`~/.cache/huggingface/hub`) points at snapshot `4cde9d2...`, which holds only config.json. Every offline load
  of the policy fails with "does not appear to have a file named model.safetensors".
- **Workaround used here.** `HF_HUB_CACHE` points at a hardlinked copy whose `refs/main` names the complete
  snapshot `7b375e1...`. The shared cache was not changed.
- **To fix for everyone.** Write `7b375e1b73b11138ff12fe22c8f2822d8fe03467` into that `refs/main`, or finish the
  download. This is the human's call.
