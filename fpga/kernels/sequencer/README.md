# float_top command sequencer (contract 1.0)

The ARM writes one list of 256-byte command records per arena, once. Each chunk it writes the inputs and
starts `float_top` once. The IP walks the list through its `io` port and sends every command to one of three
units: the int16 engine (GEMM), the shared row unit, or the softmax unit. Every format, code, limit and check
comes from [`../contract.h`](../contract.h) and [`../CONTRACT.md`](../CONTRACT.md) (sections 3 and 4.1). This
directory holds the walk, its testbenches, and the golden executor of the contract's chunk list.

| file | what |
|---|---|
| `seq_core.h` | the sequencer, header only: `seq::top<Units>()` = fetch, `decode_hw`, `ctr::validate_hw`, one dispatch site per unit, END / YIELD / STEP / LIMIT / NO_LEGACY / BAD_* / MODE_ERROR, 3-word status block, cycle counter |
| `seq_top.cpp` | standalone HLS top with `float_top`'s exact argument list, bundles and INTERFACE pragmas, and null units |
| `run_seq_hls.tcl`, `run_hls_all.sh`, `hls_summary.py` | csynth (xck26-sfvc784-2LV-c) one run after another, and the summary of the reports |
| `hls_contract_unroll.py` | writes a copy of `contract.h` with its `validate_t` loops unrolled (a latency experiment, see "Synthesis") |
| `tb_seq.cpp` | host testbench (below) |
| `run_tests.sh` | the host checks in order: `make_chunk.py`, `tb_seq.exe`, `check_twin.py` |
| `make_chunk.py` | the -12392 chunk list (`contract_plan.build` via `build_smolvla.chunk_plan`) and its Python decode trace |
| `exec_ref.py` | golden executor of contract records in torch (gate T-P1): the walk of `seq_core.h`, every opcode's layouts and arithmetic |
| `build_smolvla.py` | fills every buffer the list addresses with -12392's weights (packed and folded as CONTRACT.md 2.4), constants and per-chunk inputs |
| `check_policy.py` | full policy through the list against `ref_chunk_raw`, in three modes |
| `check_twin.py` | replays every testbench start through `exec_ref.py` and compares it with the C++ run |

The 128-byte draft (`seq_format.h`, `seq_stubs.h`, `arena.py`, `count_cmds.py` and the old builder) is
retired by the contract. It is in git at a18ee58 (`git show a18ee58:fpga/kernels/sequencer/<file>`).

## Protocol

- **Registers:** the shipped `s_axi_control` map, unchanged. `M = 0xA5 << 24 | step << 23 | first index`,
  `K` = list base >> 4, `N` = status block >> 4 (0 = none). The pointers w0..w3, xmem and ymem are all 0. Records
  carry absolute byte addresses, so the AXI address of port index i is 16 i.
- **No magic in M:** the call returns `ST_NO_LEGACY` (0x03) and reads or writes nothing. There is no per-call
  protocol.
- **Return value:** `AP_RETURN = (next index << 8) | status`. It is u32: at `ST_LIMIT` (next = 0x800000) bit 31 is
  set.
- **Per command:**
  1. Fetch the record: 16 io words in one burst, at word index K + 16·index (64-bit).
  2. `decode_hw`, then `ctr::validate_hw`.
  3. Dispatch the command, then stop on END, F_YIELD, step mode or LIMIT.
- **Faults:** on BAD_* or MODE_ERROR nothing is dispatched and the next index stays on the failing record. The
  status block reports that record's raw tag and opcode.
- **Status block (N != 0):** 3 words at N << 4. The fields are exactly CONTRACT.md 3.5: code, next, tag, done |
  ME code, 0x53455153, starts since configuration, last opcode | CONTRACT_VERSION, index of the last finished
  command (0xFFFFFFFF = none), cycles [31:0], [63:32].
- **Cycle counter:** in synthesis it is a DATAFLOW region of two processes started together. One is the walk.
  The other counts clocks until the walk sends it a stop token, which every exit path does. On the host it is a
  model, `g_host_cycles`.

**decode_hw** is `ctr::decode`'s rules with constant field indices:

- the check word is folded 4 lanes per beat during the fetch: it is right iff `CHECK_SEED ^ xor(f0..f63) == 0`;
- the fields are unpacked from the fetched words by constant slices.

It returns the same status in the same order (MAGIC, CHECK, VERSION, OPCODE) and the same `Cmd`. Host builds
compare it with `ctr::decode` on every record any walk fetches, and `tb_seq` adds 400,000 fuzzed records.
`ctr::decode` itself costs 2.0k LUT of XOR chain plus about 7k FF of loop-phi copies (section "Synthesis").

## Mode function interface (what the engine, row unit and softmax provide)

```cpp
struct Units {
  template <class WP, class W>
  static uint32_t gemm(const ctr::Cmd& c, const WP* hp0, const WP* hp1, const WP* hp2, const WP* hp3,
                       const W* io_rd, W* io_wr);   // OP_GEMM
  // row(...)     same signature: QUANT_X, QUANT_W, QUANT_WT, NORM, GELU, SILU_MUL, ROPE, COPY
  // softmax(...) same signature: SOFTMAX
};
int float_top(...) { /* shipped pragmas */ return (int)seq::top<Units>(w0, w1, w2, w3, xmem, ymem, M, K, N); }
```

**What a unit may assume:** `c` has passed `ctr::validate_hw`, `c.B >= 1`, and every address is below 2^40.

**What a unit must do:**

- Index a port as `port[addr >> 4]`, with all address arithmetic in uint64 or ap_uint<40>, never int.
- Read activations only through `io_rd`. Use hp0..hp3 only for weights, host constants and the runtime K/V^T
  blobs.
- Return only after the last write response (R-ORD).
- Return `ME_OK`, or an ME code for a fault it detects itself. The walk then stops with MODE_ERROR and counts the
  command as done.
- Be `#pragma HLS INLINE off` with this one call site, so it is one hardware instance. No ALLOCATION pragma: it
  broke timing at 5.66 ns.
- Take the top's ports as arguments, with no m_axi adapters of its own.

## Tests (all run 2026-11-21; Vitis MinGW g++, clean PATH)

`sh run_tests.sh` gives ALL PASS. The parts:

**`tb_seq.exe`: PASSED, 150 checks, 0 failures.**

- **Walk paths:** at memory base 0 and at 0x8_0000_0000. The paths are END, YIELD + resume, STEP (including STEP
  on a YIELD record and at END), NO_LEGACY (three M values; nothing written, not even the status block) and
  BAD_MAGIC (including a 128-byte 0xC5 draft record). Then BAD_CHECK, BAD_VERSION and BAD_OPCODE (including the
  retired GATHER and AXPY). Then 16 validate_hw MODE_ERRORs with nothing dispatched or written, a unit's own
  fault, and no status block. Every status word is checked.
- **Defect 1 is fixed:** the from-0 bad-record cases clear record 2's F_YIELD first.
- **Dispatch:** NOP, GEMM, the 8 row ops and SOFTMAX each reach their unit once, in order, with every field.
- **Data:** QUANT_X (shift −3, NP padding, an all-zero row) → GEMM (2 items, C row groups, bias, D) → GEMM F_ACC
  → COPY. The host units are written from `ctr::ref`. The result is bit-exact.
- **LIMIT:** a NOP at index 0x7FFFFF with the list base below the record.
- **The -12392 list:** 2,701 records at CMA bases 0x4000_0000 and 0x8_0000_0000; at the high base, 12,285 slot
  addresses are at or above 4 GB.
  - One start reaches END with 2,700 dispatches.
  - The dispatch trace equals the Python decode line by line.
  - The same list in step mode (2,701 starts) and with YIELD on every 97th record dispatches the identical
    sequence.
- **decode_hw:** equal to `ctr::decode` on 400,000 fuzzed records:
  - by status: 147,643 END, 51,736 BAD_MAGIC, 98,201 BAD_CHECK, 49,659 BAD_OPCODE, 52,761 BAD_VERSION;
  - also on all 16,367 records the walks fetched.

**`check_twin.py`: PASS.** 2 × 44 testbench starts are replayed through `exec_ref.py`. Return value, status words
0-9, dispatch trace and output bytes are identical.

## Policy check (the golden executor, gate T-P1)

```
HF_HUB_OFFLINE=1 python check_policy.py <ckpt> build/ckpt/smolvla-12392/batches-12392 --modes exact,int16,int16k --out build/py/policy.json
```

The modes:

- **exact:** no quantization. FP32 torch front ends on the list's addresses, tilings and folded constants.
- **int16:** the contract quantizer and epilogue everywhere, with torch front ends.
- **int16k:** the contract quantizer and epilogue, with the kernels' bit-exact emulators (`nr_emul.kernel`,
  `gs_emu.act` NSB 9, `smx_emu.kernel` with zero rows 1.0, `rope_emu.rope_rows`). This is the golden executor of
  CONTRACT.md 7.

Max |chunk − ref| on -12392's frames, list started once per frame, one arena for all frames:

| mode | frame 0 | frame 3563 | frame 7489 | gate (CONTRACT.md 7) |
|---|---|---|---|---|
| torch FP32 policy, no list | 4.7e-6 | 3.6e-7 | 3.0e-7 | — |
| exact | 1.56e-5 | 1.01e-6 | 5.96e-7 | wiring <= 1e-5 (frame 0: see below) |
| int16 | 1.24e-3 | 1.35e-4 | 1.25e-4 | — |
| **int16k (T-P1)** | **1.00e-3** | **1.14e-4** | **1.29e-4** | <= 0.006 / <= 2e-4 / <= 2e-4: **pass** |

How to read it:

- Frame 0 is the onset frame. The precision study puts int16 attention there at 0.0017-0.0057 (I7e 0.0057).
- The exact mode's 1.56e-5 on frame 0 is above the 1e-5 wiring number. It comes from the folded constants:
  sqrt(960) in s_row, the Euler fold and the language rows are rounded differently from torch, and frame 0
  amplifies that. The benign frames are at 1e-6.
- Every run ended with ST_END, next 2,700 and done 2,700.

**Defect in `contract_plan.py` (the contract owner's file; reported, not edited).**

- The expert's `QKV_s` is allocated for 45 rows (`TQ_P * 1600 * 4`), but each Euler step writes 50 rows into it.
- The step qkv GEMM therefore writes 32,000 bytes over the next allocation: expert layer 0's self-attention K
  blob.
- Effect: int16 and int16k give NaN on every frame.
- The earlier exact mode missed it, because GEMMs read weights from the FP32 shadow, which FP32 byte writes never
  touched. `exec_ref.py` now poisons the other representation on every write. With that, exact mode on the list
  as built gives NaN (`--no-plan-fix`).
- The fix is one line: `QKV_s = m.alloc(max(TQ_P, TQ_E) * 1600 * 4)`.
- `build_smolvla.chunk_plan()` applies it until the owner does, by sizing every 45 × 1600 FP32 arena buffer for 50
  rows. No record changes except its addresses, and the result still validates.

**Checkpoint loading.** `config.json` of the -native checkpoint has `load_vlm_weights: true`. On this laptop the
HF cache's `refs/main` for SmolVLM2-500M-Video-Instruct was moved at 2026-11-20 22:55 to a snapshot that holds only
`config.json`, so offline loading fails. The runs above used a copy of the checkpoint (hardlinks) whose
`vlm_model_name` points at the complete snapshot `7b375e1b…`.

## Synthesis (Vitis HLS 2022.2 csynth, 4.0 ns, `sh run_hls_all.sh`, `python hls_summary.py`)

| build | est. | sequencer LUT | FF | DSP | BRAM | clocks / command | top total LUT (adapters) |
|---|---|---|---|---|---|---|---|
| **sequencer alone** (`SEQ_UNITS=0`: walk + decode_hw + validate_hw + status + cycle counter) | **2.92 ns** | **6,330** | **6,758** | **7** | **0** | **72** | 10,618 (4,219: io + control) |
| with 3 null units (`p40`: dispatch and port muxing) | 2.92 ns | 6,969 (13,814 − 6,845 null units) | 6,045 | 7 | 0 | 90 (incl. null units) | 27,452 (13,491) |
| same, `validate_t` loops unrolled (`SEQ_UNROLLED=1`, alone) | 2.92 ns, slack −0.00 | 8,055 | 6,773 | 7 | 0 | 47 | 12,343 |
| WIP before this conform (`ctr::decode`, alone) | 2.92 ns | 10,495 | 12,942 | 7 | 0 | 122 | 14,783 |

Notes:

- **Budget (CONTRACT.md 4.2): sequencer +8.5k LUT, +7 DSP, 0 BRAM.**
  - Alone: 6.3k LUT, 2.2k under, at a post-route factor of 1.0.
  - With dispatch to the three units: 7.0k LUT. The units' port muxing proper is its own budget line (+2.5k).
- **Pipelines:** every loop is II = 1 (fetch_rec, status_wr, count_clk). `cmd_loop` is the sequential walk, by
  design.
- **Per-command cost:** fetch 19 clocks, `validate_hw` 39, the rest 14, i.e. 72 clocks. The contract's time plan
  assumes 80, so 2,701 × 72 = 0.78 ms per chunk at 250 MHz. HLS assumes no DDR latency; the board adds the read
  latency once per record (unknown).
- **Unrolling `validate_t`** would save 25 clocks per command (0.27 ms per chunk) for 1.7k LUT, at slack −0.00. It
  is not worth it while LUT is the tight resource. That decision belongs to the contract owner.
- **The WIP version's cost:** `ctr::decode`'s rolled field loops over the partitioned record built 8:1 and 16:1
  multiplexers plus phi copies of the Cmd (10.4k FF). Its 63-input XOR check was estimated at 2,018 LUT.
- **Warnings:** HLS 200-657 reports a "backward" channel in the cycle-counter region (walk → counter stop token).
  That is inherent to a counter that runs beside the walk.

## Review defects (CONTRACT.md 9, sequencer rows) and what changed

| # | defect | now |
|---|---|---|
| 1 | tb_seq.cpp:248 expected BAD_CHECK where YIELD stops first | tb re-targeted to 256-B records; the from-0 cases clear F_YIELD first; the tb has run: PASSED |
| 2 | exec_ref quantizer s = amax/32767, zero rows 0 | `contract.quant_rows` everywhere (zero rows 1.0, 1e-30 floor, p4 shift), incl. the host inputs |
| 3 | GEMM needed bias/D/C/p0 | exec_ref and the list follow 2.3: no p0 (ME_PARAM), folds in the packer (`build_smolvla._weights`), bias / D / ACC in the epilogue (`contract.gemm_out`) |
| 4 | legacy path through the same dispatch | removed: NO_LEGACY, nothing touched (tested with 3 M values) |
| 5 | int index overflow above 2 GB; CMA plan above low DDR | 64-bit word indices in the walk, every test at 0x8_0000_0000 too; the policy check runs at base 0x8_0000_0000 |
| 6 | "3.239 ns (meets)" overclaim; II = 2 stubs | stubs retired; numbers above are csynth's own (2.92 ns at the 2.92 ns budget of a 4.0 ns target: 0 slack, 342 MHz est.) |
| 7 | weak stub checks, K > 1024 in the stub, the Python twin diverged | `validate_hw` is the only rule set; `exec_ref` runs the same walk (MODE_ERROR, STEP, LIMIT, raw tag on BAD_*, status words 0-2) and `check_twin.py` compares it with the C++ on every tb start |
| 8 | QUANT_W had no port pitch | KW / VW tilings use C.pitch as the port pitch (`exec_ref.put_tiled`); the GEMM reads B through its pitch |
| 9 | 2.5 ms per call host time, unmeasured | not used: one start per chunk; the board measures with the cycle counter (BT-3) |
| review | "status word 4" nit; K/V blobs row-major in the model | status block per 3.5; K/V blobs in the port tiling |

## Unknowns (not guessed)

- **The cycle counter is not simulated.** The DATAFLOW region with its stop and count streams has never run in RTL:
  csim and cosim are broken here, and the host build has no counter. `SEQ_CYCLES=0` drops it, and the status
  then reads 0 cycles.
- **DDR read latency per record fetch** on the board.
- **Cross-master ordering:** whether a write on io (HPC) is visible to a later hp read after BRESP. This matters
  for the runtime K/V blobs; it is board test BT-1.
- **The cost inside the merged float_top.** The real units replace the null units, and their port muxing has not
  been synthesised.
- **A frame-0 wiring number at or below 1e-5** would need the unfolded constants. The contract's folded recipe
  gives 1.56e-5.
