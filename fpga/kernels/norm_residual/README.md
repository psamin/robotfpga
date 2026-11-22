# norm_residual: the NORM mode (opcode 0x30) of float_top, integration contract 1.0

LayerNorm (vision, `F_LN`) or RMSNorm (LM prefix, action expert) of FP32 rows. Output is either int16 rows plus
one stored scale per row (`F_QOUT`, the contract quantizer) or FP32 rows (vision post_layernorm). The residual
add is **not** here any more: it is the GEMM epilogue (CONTRACT.md 2.3, `D = C = R`). The directory keeps its
name; the mode is `nr::norm_mode`.

Status (2026-11-21): the C++ is bit-exact against its torch model on 57 DDR-image cases (6 from real -12392
activations, 21 rejected records), in both the sequential build and a threaded bounded-FIFO build of the
DATAFLOW region. The full-policy check passes on -12392's 3 frames. csynth at 4.0 ns estimates 3.05 ns (328
MHz), with every flat loop at II = 1.
Not done: no Vivado run, no RTL co-simulation (csim/cosim are broken here), no board run.
**The area is over any fair share of the row-unit cap** (section "Area").

## Files

| file | what |
|---|---|
| `norm_residual.h` | header-only HLS code: `nr::norm_mode(cmd, hp, xmem, ymem)` and its DATAFLOW region |
| `nr_top.cpp` | standalone csynth top: record fetch over io + `ctr::decode` + `ctr::validate_hw` + one `norm_mode` call site; float_top's bundle names |
| `run_nr_hls.tcl` | csynth script (`NR_PERIOD`, `NR_FMUL`, `NR_FADD`, `NR_SQ`, `NR_KFOLD`, `NR_URAM` env options) |
| `tb_nr.cpp`, `nr_dfsim.h`, `run_tests.sh` | host testbench (g++): unit tests, whole-DDR-image comparison byte for byte, threaded bounded-FIFO model |
| `nr_emul.py` | bit-exact numpy model of the mode (`python nr_emul.py` runs its own definition checks) |
| `gen_vectors.py` | the DDR-image cases (synthetic, rejects, `--real` -12392 activations) |
| `policy_check.py`, `policy_check.json/.log` | full-policy check on -12392's 3 frames, norm counts and row statistics |
| `count_cost.py` | NORM time per chunk: the 386 NORM records of `contract_plan.py`, kernel model vs the contract model |

## Contract conformance (CONTRACT.md 1.0)

| clause | how |
|---|---|
| record (3.2, 3.3) | `nr::norm_mode(const ctr::Cmd&, hp, xmem, ymem)`, `INLINE off`, callable from the sequencer's dispatch switch after `validate_hw`: `case ctr::OP_NORM: r = nr::norm_mode(c, w2, xmem, ymem);`. Fields: K, N, B; A, C, E, F slots; p0 eps, p4 shift, p5 NP; flags `F_QOUT`, `F_LN`. Returns ME_OK, or ME_DIMS without touching memory when the counters could not hold the command (a guard behind `validate_hw`) |
| 64-bit addresses (3.2) | the slot addresses are absolute and narrowed to 40 bits (`validate_hw` rejects >= 2^40). Rows and items advance by adding pitches and strides to 36-bit word indices, never by a 32-bit product. Tested at bases 0xFFFF_0000 (across 4 GB), 0x8_7000_0000 (KR260 high DDR) and 0xF_FFF8_0000 (word indices past 2^32) |
| layouts (1.1) | A FP32 rows (pitch, stride); C int16 rows (K/8 words) or FP32 rows; rows [N, NP) are written as zeros with scale 1.0; F one scale per row, `pad4(NP)` floats per item, pad lanes 1.0; E gamma[K], beta at E + 4K |
| ports (3.3) | activations are read **only on io** (xmem); gamma / beta (host constants) on one hp port, read only; every write on io (ymem). No pointer argument, no register, no m_axi adapter of its own |
| quantizer (2.1) | `quantize/quant_int16.h`: `row_scale`, `quant_elem`, `stored_scale` (s 2^p4, zero rows 1.0, non-finite rows keep their absmax). The own 1/trunc17 rule is retired |
| FP32 semantics (2.3) | every FP32 core operation flushes denormals with the sign kept; the g++ build uses `ctr::ref::fmul_hw / fadd_hw`, so the host model is the hardware |
| retired (2.3, 3.3) | F_RESID, F_SUM, the records layout and the hp0/hp1 activation reads are gone; those flags give ME_FLAGS in `validate_hw` |
| limits (3.3) | K 8..1024, K % 8 == 0, N and B 1..65535, NP in [N, 65535] with F_QOUT, NP = N otherwise; counters sized from these |

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
   - This replaces the draft's separate exponent pass: one row FIFO (8 BRAM18 in HLS) and one process fewer.
2. **Row scalars, correctly rounded, in integers.**
   - `num = max(K Q - S^2, 0)` (LN) or `K Q` (RMS).
   - `mean = rn(S / K) 2^(E-151)` (LN, else 0) and `var = rn(num / K^2) 2^(2 (E-151))`.
   - Results below 2^-126 flush to +0; above the FP32 range they give +inf.
   - Hardware: K Q - S^2 by one signed shift-add accumulator (37 clocks); then two bit-serial dividers with
     remainders of the divisor's width side by side (118 clocks, no dividend normalization). RMS divides Q by K
     (the same rational as K Q / K^2).
   - So a constant LayerNorm row gives `y - mean = 0` exactly and `z = beta`, as torch (review defect #30).
3. **rstd, correctly rounded, in integers (no FP32 core, no seed ROM).**
   - `ve = max(fl(var + eps), 2^-126)`: an IEEE round-half-even add of two non-negative floats, inputs flushed,
     the smaller operand aligned 1 bit a clock.
   - `rstd = rn(1 / sqrt(ve))`: `D = floor(2^76 / M')` (54 steps), `R = isqrt(D)` (27 steps), with a sticky from
     both remainders; 0 if ve = inf.
   - Proven correctly rounded for every input: `tb_nr --units-exhaustive` checks every significand at both
     exponent parities against the exact midpoint condition in `__int128` (16,777,724 values; the exponent only
     shifts the result). There are no ties: a midpoint would need 1/sqrt(ve) to be a dyadic rational with 25 bits,
     and 1/sqrt(ve) is dyadic only when M' is a power of 4, where it is an exact power of two.
4. **Normalize, FP32 cores, 4 lanes:** `z = ((y - mean) * rstd) * g + b` in that order (RMS: mean = 0, b = 0).
5. **q-out (F_QOUT):** the contract quantizer on z: `s = fl32(max(amax, 1e-30) / 32767)`,
   `q = clamp(rne(fl32(z / s)), +-32767)`; stored scale `s 2^p4`, all-zero rows 1.0.

Non-finite inputs: undefined values, but no hang and no write outside the declared extents (an exponent field of
255 is a large finite value to the integer statistics; a var above FP32 gives rstd = 0, i.e. z = beta). LN rows
whose variance exceeds FP32 (|y| above about 1e19) give z = beta, where torch gives NaN or inf; real activations
are far from this.

## DATAFLOW region and speed

| process | role | per-row cost |
|---|---|---|
| `p_rd` | A over io: one burst per item when the row gap is at most half a row (gap words read and dropped), else one burst per row | K/4 (+gap) words |
| `p_stats` | running exponent, S, Q (4 squares a clock); the row goes on to f2 (512-word BRAM FIFO) | K/4 |
| `p_scal` | num, mean, var (integers) | 167 clocks (LN), 127 (RMS) (csynth) |
| `p_rsqrt` | ve, rstd (integers) | 127 clocks (csynth) |
| `p_affine` | gamma \| beta into one dual-port BRAM once per command, then z, 4 lanes a clock | K/4 |
| `p_qamax`, `p_qscal`, `p_quant`, `p_wr` | stand-ins for the row unit's shared absmax, quantizer and writer (see "Integration") | K/4, or K/8 written words |

Steady state: `max(K/4, A words read, C words written, scalar clocks)` a row. All SmolVLA rows are io-bound (LN
768: 192 clocks a row against 167; RMS 720 / 960: 180 / 240 against 127). Rows of K < 640 (LN) or K < 512 (RMS)
are scalar-bound.

`count_cost.py`, the 386 NORM records of `contract_plan.py` at 250 MHz:

| group | calls | rows | kernel model | contract model |
|---|---|---|---|---|
| vision LN 768, int16 | 24 | 12,288 | 9.59 ms | 9.69 ms |
| vision post_layernorm 768, FP32 | 1 | 512 | 0.40 | 0.40 |
| LM RMS 960 | 31 | 1,395 | 1.49 | 1.43 |
| expert RMS 720 | 330 | 16,500 | 13.63 | 12.86 |
| **total** | **386** | | **25.1 ms** | **24.4 ms** |

The 0.7 ms difference is the per-call fill of the 330 short (50-row) expert calls: the region's scalar latency
and the gamma load. With pessimistic assumptions (DDR latency 300, write response 150, call overhead 300 clocks)
the total is 26.3 ms. **Unmeasured:** DDR latency, write-response wait and call overhead are assumptions.

## Area (csynth, Vitis HLS 2022.2, xck26-sfvc784-2LV-c, 4.0 ns, `build/nr5_fulldsp_fulldsp_dsp_p40`)

The contract caps the **whole** row unit (8 opcodes) at 15k HLS LUT, 50 DSP and 10 BRAM36 (CONTRACT.md 4.2). It
gives no per-front-end split. The NORM front end is what the row unit takes from this kernel; the other
processes are stand-ins for parts the quantize kernel's backbone provides once.

| process | LUT | FF | DSP | BRAM18 | part of |
|---|---|---|---|---|---|
| `p_stats` | 3,100 | 1,800 | 8 | 0 | NORM front end |
| `p_scal` | 2,026 | 1,142 | 0 | 0 | NORM front end |
| `p_rsqrt` | 1,685 | 805 | 0 | 0 | NORM front end |
| `p_affine` (16 FP32 cores: fadd 238 LUT + 2 DSP, fmul 134 LUT + 2 DSP each) | 3,914 | 5,807 | 32 | 8 (gamma \| beta, 512 x 128 dual port) | NORM front end |
| f2 row FIFO (512 x 128) | ~420 | ~650 | 0 | 8 | NORM front end |
| **NORM front end** | **~11.1k** | **~10.2k** | **40** | **16** | |
| `p_rd`, `p_qamax`, `p_qscal`, `p_quant`, `p_wr`, fq FIFO, small FIFOs | ~10.0k | ~7.5k | 14 | 8 | stand-ins (shared backbone) |
| **norm_mode** | **21,882** | **18,246** | **57** | **24** | |

Against the draft found paused (WIP 52f4962, `build/nrc_fulldsp_fulldsp_p40`): its front end was 9.7k LUT,
50 DSP and 25 BRAM18 (an exponent pass and two row FIFOs; seed + Newton rsqrt on FP32 cores; a 74-bit divider).
This version saves 10 DSP and 9 BRAM18 and makes rstd correctly rounded, for about 1k LUT more. LUT is the
tight resource; the scalar units now hold ~3.7k LUT of narrow selects and adders, close to their floor.

**Reading.** At about 11k HLS LUT and 40 DSP, NORM alone takes 74 % of the row unit's LUT cap and 80 % of its
DSP. The 4-lane affine is 3.0k LUT of FP32 cores and 32 DSP by itself, and cannot get smaller inside contract
1.0. Ways down, for the integrator and the contract owner:

1. **Fold gamma / beta into the next linear** (`W' = W diag(g)`, `b' = b + W beta`; for the post_layernorm
   through the pixel shuffle into the connector). It removes 2 of the 4 FP32 cores a lane, the table and its
   load. The cost was measured by csynth with `NR_KFOLD=1` (`build/nr5_fulldsp_fulldsp_dsp_p40_kfold`): see
   the table below. The policy effect of the activation side was measured too (`kfold` in "Accuracy"); the
   re-quantization of `W diag(g)` was not. This is a contract change (NORM's E slot) and a `pack_float.py` change
   (a shared-contract file): it needs sign-off from both sides.
2. **Share the affine's FP32 cores** with the SILU_MUL / GELU front ends inside the row unit: one 4-lane
   `((x - a) r) g + b` stage that SILU_MUL can also use for its multiply. This is integration work.
3. **2 lanes** in the front end: about half the stats, affine and squares (est. -5k LUT, -20 DSP) for about
   +24 ms per chunk (NORM at K/2 clocks a row), which takes most of the +0.065 s margin. Not recommended.

| build | front-end LUT | DSP | BRAM18 | norm_mode LUT | est. clock |
|---|---|---|---|---|---|
| contract 1.0 (default) | ~11.1k | 40 | 16 | 21,882 | 3.05 ns |
| `NR_KFOLD=1` (not contract 1.0): `p_affine` 1,839 LUT, 16 DSP, 0 BRAM | ~9.1k | 24 | 8 | 19,764 | 3.05 ns |

HLS BRAM counts: HLS counts 8 BRAM18 for each 512 x 128 FIFO or dual-port table. That is 64 Kb, which 4 BRAM18
hold in 512 x 36 simple-dual-port mode. Vivado may use 4 for the FIFO; the true-dual-port gamma | beta table
needs 8 (TDP BRAM18 is at most 18 bits wide), or 2 URAM with `NR_URAM=1`. Not verified without Vivado.

## Accuracy

`policy_check.py --frames 0,1,2 --variants torch,zf32,kernel,torchq16,kfold` (`policy_check.log`, `.json`).
All 91 norm modules use the mode's exact arithmetic (`nr_emul.py`); linears stay FP32, so only the norms change.
The table gives `predict_action_chunk(batch, noise)` against the cluster's `ref_chunk_raw`, max |chunk - ref| in
normalized action units (gate 0.05):

| frame | torch (laptop) | zf32: FP32 z everywhere | **kernel as deployed** (int16 into every linear, FP32 z for post_layernorm) | torchq16: torch norm + the same contract quantizer | kfold (proposal 1, activation side) |
|---|---|---|---|---|---|
| 0 | 5.13e-6 | 5.72e-6 | **1.34e-3** | 7.79e-4 | 7.83e-4 |
| 3563 | 4.62e-7 | 2.38e-7 | **1.43e-4** | 1.30e-4 | 1.61e-4 |
| 7489 | 4.77e-7 | 2.98e-7 | **1.52e-4** | 1.52e-4 | 1.67e-4 |

- **zf32.** The normalize arithmetic alone is at torch's own laptop-vs-cluster level.
- **The deployed error is the int16 quantization of the norm output.** torchq16 carries the same quantization
  with torch's norm in front. It is the same size on the benign frames (2e-4 class, contract T-P1). Kernel vs
  torchq16: 2.1e-3 / 6.8e-5 / 4.1e-5.
- **Frame 0 is the sensitive frame** (contract T-P1 class: <= 0.006). There 1-ulp differences of z move the
  chunk by 1e-3. The draft's (WIP) frame-0 figure was 4.0e-4 and the pre-contract kernel's 1.45e-4, with the
  same per-call accuracy (below). This run is 37x under the gate.

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

## Integration notes for the row unit (CONTRACT.md 4.1)

- The NORM front end is the four processes `p_stats -> p_scal -> p_rsqrt -> p_affine` plus the f2 FIFO and the
  gamma | beta table.
- Its interface is a word stream in (FP32 rows, K/4 words a row, N rows per item, B items, in row order) and the
  same stream of z words out. Per command it needs `Cfg` (K, N, B, ln, eps, DV = K^2 or K, E's word address).
  It reads E on one hp port.
- In the quantize kernel's backbone (`quantize/quant_mode.h`, `qz::row_df`) it sits between `qz_rd` and
  `qz_amx`, as NORM = QUANT_X rows (F_QOUT) or COPY rows (FP32) with this front end. `qz_rd`'s 16-word units
  re-read the overlap of a row tail; the front end must take only the `fwd` tokens and re-emit the `last` / `end`
  flags. That adapter is not written here: quant_mode.h was being edited while this was done.
- The stand-ins (`p_rd`, `p_qamax`, `p_qscal`, `p_quant`, `p_wr`) are not meant for float_top. The backbone's
  reader uses fixed-size units so that DDR latency overlaps across rows; this stand-in reader pays it once per
  row when a row's gap is more than half a row (no NORM of `contract_plan.py` does that).
- f2 needs at least one row (K/4 words) or the region deadlocks. The bounded-FIFO model, measured: depth 240
  deadlocks at K = 1024 and 200 at K = 960, while 256 passes at K = 1024. 512 adds room for the scalar latency
  (about 300 clocks), so the stats keep streaming at full rate.

## Tests (`sh run_tests.sh [--exhaustive]`)

- Units:
  - `div_rn`: 400,000 random 72-bit dividends / K or K^2, ties, exact quotients and powers of two, against an
    `__int128` reference. Exact.
  - `fadd_pos`: 2,000,000 non-negative pairs against the host's IEEE add, inputs flushed. Exact.
  - `rsqrt_cr`: an `__int128` reference and the exact midpoint test. 300k random values by default; every
    significand at both parities with `--exhaustive`.
- DDR images (57): every output byte compared, and every byte outside the outputs must keep its sentinel.
  - 21 records `validate_hw` or `decode` rejects with the image untouched: N = 0 / 65536, B = 65536, K = 1032 /
    12, retired flags, eps 0 / 1, shift 9, SEG 64, misaligned A, short C pitch, a used B slot, FP32 with NP or F,
    address >= 2^40, bad magic, check word and version.
  - LN and RMS; int16 and FP32 out; B = 2 with padded pitches and strides; flat and per-row reads and writes;
    N = 257 / 260 / 300 with zero rows [N, NP) and shifts -3 / +5 / -8.
  - K = 8 / 64 / 720 / 768 / 960 / 1024, including K = 1024 FP32 over two 256-row writer chunks (65,536-word
    bursts).
  - Constant LayerNorm rows at K = 720 / 768 / 960 / 1024 (z = beta exactly, q = 0 for beta = 0).
  - Renormalization-heavy rows (an exponent that grows every word, the max in the last word, a 2^60 jump).
  - K = 64 edge rows (zero, -0, denormal, tiny, huge, overflowing LN variance, one-hot), a denormal eps.
  - Bases across 4 GB and in high DDR.
  - 6 real -12392 calls: vision layer 5 ln2, post_layernorm (FP32), LM layers 7 and 15, expert layer 3, expert
    final.
- Two builds: sequential C simulation, and `-DNR_DFSIM`, where every process runs in a thread and every stream
  blocks at its declared depth. A FIFO deadlock hangs there and the watchdog reports it. Both: 57 / 57.

## Running

- **HF cache.** `refs/main` of SmolVLM2-500M-Video-Instruct in the shared cache
  (`~/.cache/huggingface/hub`) points at snapshot `4cde9d2...`. That snapshot holds only config.json; a
  395 MB `.incomplete` blob from 2026-11-20 22:55 sits next to it. Every offline load of the policy fails with
  "does not appear to have a file named model.safetensors".
- **Workaround used here.** `HF_HUB_CACHE` points at a hardlinked copy whose `refs/main` names the complete
  snapshot `7b375e1...`. The shared cache was not changed.
- **To fix for everyone.** Write `7b375e1b73b11138ff12fe22c8f2822d8fe03467` into that `refs/main`, or finish the
  download. This is the human's call.
