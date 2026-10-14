# Questions and context

The shared board for both teams. **Read this after every pull. Update it before every push** if you
asked, answered, or changed something the other side needs to know.

- Add a question under **Open** with who it is for. When answered, move it to **Answered** with the date
  and a link to the evidence.
- Write "unknown" or "not ready" rather than guessing. Say whether something was run or only planned.
- Keep **Current state** short and true; replace stale lines instead of appending.

## Current state (2026-10-14)

The task is the real one: SO-101 picks up a medicine bottle and places it on a plate, using the SmolVLA
fine-tuned on our demos. Goal: run it on the KR260 in full precision (BF16 weights in DDR, FP32 math,
linear layers on the FPGA). Cube-sorting work removed; preserved at tag `archive/cube-sorting`.

| Area | State | Where |
|---|---|---|
| Policy | Fine-tuned SmolVLA, checkpoint `smolvla-so101_medicine_bottle_pickplace-9604` on the cluster | [SETUP.md](SETUP.md) §3 |
| Real arm | Follower + leader calibrated; teleop / record / rollout toolkit works | [robot/](robot/README.md) |
| FPGA engine | `fpga/hls/` passes C simulation on real SmolVLA layers; whole SmolVLA through the engine code matches PyTorch FP32 to ~0.01° mean. Not synthesized | [docs/smolvla-fpga.md](docs/smolvla-fpga.md) |
| Estimated FPGA speed | ~3–4 s per 50-action chunk, compute-bound | [docs/smolvla-fpga.md](docs/smolvla-fpga.md) §Verification |
| KR260 | Ubuntu 22.04.4 + PYNQ 3.0.1, reachable over the laptop's Ethernet link | [SETUP.md](SETUP.md) §6 |
| Laptop ↔ board link | A board TCP service and laptop client exist from the earlier int8 work (outside git); the full-precision runtime is `fpga/board/runtime.py` | A-4 |

## Open

| # | Asked by | For | Question | Link |
|---|---|---|---|---|
| Q12 | psamin | FPGA (@athithan-elamaran1, @Dhyey234) | In `fpga/hls/`: `vitis_hls -f run_float_hls.tcl` then `vivado -mode batch -source build_bd.tcl` (writes `build/float.bit` + `.hwh`; untested). Report timing, II of the reduction loop, DSP/BRAM/URAM use, then the measured GEMM bandwidth on the board | [docs/smolvla-fpga.md](docs/smolvla-fpga.md) |
| Q13 | psamin | psamin | Speed vs precision: FP32 math gives ~3–4 s per chunk. Options are BF16 multiplies with FP32 accumulation (several times more MACs per DSP), running the vision encoder less often, or accepting a slower control rate | [docs/smolvla-fpga.md](docs/smolvla-fpga.md) §Verification |
| Q14 | psamin | @m-alan08 | The SGS pipeline was removed with the cube-sorting sim (recoverable at `archive/cube-sorting`). RL post-training of SmolVLA would need a pick-and-place sim, would start from the `-9604` checkpoint, and would write a new checkpoint. Any training run needs psamin's approval first | [CONTRIBUTING.md](CONTRIBUTING.md) |

## Answered

| # | Question | Answer | Date | Evidence |
|---|---|---|---|---|
| A-1 | KR260 boot, versions, endpoint | Ubuntu 22.04.4 (kernel 5.15.0-1027-xilinx-zynqmp), Python 3.10.12, PYNQ 3.0.1; private Ethernet link behind the controller laptop, SSH user `ubuntu` | 2026-10-09 | [#87](https://github.com/psamin/robotfpga/issues/87#issuecomment-6075687430) |
| A-2 | Full precision or int8? | Full precision: BF16 weights, FP32 math; linear layers on the FPGA, the rest on the ARM; int8 retired | 2026-10-13 | owner decision, [#151](https://github.com/psamin/robotfpga/pull/151) |
| A-3 | Which task and policy? | Only the medicine-bottle pick-and-place, with the SmolVLA fine-tuned on our demos; cube sorting removed | 2026-10-14 | owner decision |
| A-4 | Bridge split | Software owns the laptop client and the arm controller; FPGA owns the board service and overlays | 2026-10-09 | [#87](https://github.com/psamin/robotfpga/issues/87#issuecomment-6075858540) |
| A-5 | Cluster GPUs | `general` (1-day limit): H100 80 GB, H200 141 GB ×4, L40S 48 GB ×4, L4 24 GB ×4; `techfee` 3 days. Use GPU nodes even for CPU-only jobs | 2026-10-14 | `sinfo`, job logs |
