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
| Policy | Fine-tuned SmolVLA `smolvla-so101_medicine_bottle_pickplace-9604`, the best on this task (offline action error 0.080; every co-trained or augmented variant tried is 0.084–0.121). Download: GitHub release `policy-smolvla-9604` (checkpoint, replay batches, SHA256SUMS). Approved 2026-10-24: re-finetune at 256-px images for the KR260 (#215). Done: `-12392` (256 px), release `policy-smolvla-12392-256px`, offline error 0.0765 vs 0.0708 for `-9604`, waiting for the board and arm runs. Found then: the live client squashes frames to the config's 256×256 before the policy letterboxes them; `robot/toolkit/native_camera_shapes.py` writes a config with the recorded shapes | [robot/README.md](robot/README.md) §Running the best policy, #215 |
| Post-training (Alan) | Approved in #156. Built, all open PRs stacked from #172 to #200: randomized pick-and-place scene, reset families and buffer, CPU and Warp teacher envs, PPO + SGS training job, held-out eval, calibrated-camera rendering, SmolVLA-in-sim baseline runner. Scripted expert: 0.75 on 80 held-out rest starts. Waiting on: the first GPU training run, the camera photo session, then the SmolVLA baseline. No training run yet | [docs/smolvla-post-training.md](docs/smolvla-post-training.md) |
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
| Q16 | m-alan08 | psamin | Joint mapping, derived and checked against the demos, not measured on the arm. Body joints: sim radians = LeRobot degrees, same sign, 0 at mid travel (#171, #173). Gripper: reading 0 to 100 spans the calibrated 137.8 deg, 0 = jaw pads touching, clipped to the sim's 110 deg (#175); rest reading gives 3.6 mm, held-cap reading 21 mm vs 27 mm from the video. A jaw-gap check on the arm at a few readings would confirm | [policy/sim/so101.py](policy/sim/so101.py) |
| Q17 | psamin | FPGA (@athithan-elamaran1, @Dhyey234) | Board setup for `docs/run-smolvla-on-kr260.md`: does a Python 3.12 LeRobot venv (`uv`) install next to PYNQ's 3.10, is PYNQ's Python at `/usr/local/share/pynq-venv/bin/python3`, and is CMA large enough for the 0.71 GB of weights (`grep Cma /proc/meminfo`; how to raise it on this image is unknown)? Then `serve_policy_fpga.py --selftest` on the board: PASS? seconds per chunk, and how much of it is outside FPGA calls? | [docs/run-smolvla-on-kr260.md](docs/run-smolvla-on-kr260.md) §2–3 |
| Q18 | psamin | FPGA (@athithan-elamaran1, @Dhyey234) | The vision-encoder attention matmuls (Q·Kᵀ, A·V: 38.7 GMAC per chunk) are not `nn.Linear`, so they run on the ARM (likely seconds per chunk). Offload them (second HLS kernel with Q/K/V on chip, or `linear_bf16` with K packed per call)? | [docs/run-smolvla-on-kr260.md](docs/run-smolvla-on-kr260.md) §Expected speed |

| Q20 | psamin | FPGA (@athithan-elamaran1) | Run `-12392` (256 px) on the board: selftest PASS? seconds per chunk on `float_tp4_v4ip.bit` (estimate ~8 s, not measured)? Then the arm: does it pick up the bottle? Steps in #217 | #217 |

## Answered

| # | Question | Answer | Date | Evidence |
|---|---|---|---|---|
| A-9 | Q19: do the replay batches and `deploy_smolvla.py` work unchanged with the 256-px checkpoint? | The batches hold full-size frames (the policy resizes), but they carry `-9604`'s reference chunks, so a new checkpoint needs its own; the release has them. `deploy_smolvla.py` keeps `input_features` and only sets `load_vlm_weights` to false. It skips same-size files, so the board's old `batches/` must be deleted first | 2026-10-24 | code read; jobs 12404, 12405 |
| A-8 | Brief on #214, decision (b): re-finetune the expert at 256 px? | Yes. Same recipe as `-9604`, one changed flag, new checkpoint. Acceptance is arm trials against `-9604` with the squash fixed | 2026-10-24 | #215 |
| A-7 | Q15: measure the bottle, plate and layout? | Not needed. The scene randomizes sizes, mass, friction and positions across the desk (psamin's suggestion), over ranges centred on estimates from the demo joint data and front video (bottle start (0.25, +0.09) m, plate center (0.33, -0.19) m, plate about 19 cm, cap about 2.7 cm, body about 3.5 cm) | 2026-10-14 | [docs/smolvla-post-training.md](docs/smolvla-post-training.md), #172, #174, #176, #177 |
| A-6 | Q14: plan for RL post-training of SmolVLA on pick and place? | Teacher (PPO + SGS) in a new pick-and-place sim, then online DAgger into a new checkpoint from `-9604`; task-independent code restored from `archive/cube-sorting`. Scene blocked on Q15 | 2026-10-14 | [docs/smolvla-post-training.md](docs/smolvla-post-training.md), #158 |
| A-1 | KR260 boot, versions, endpoint | Ubuntu 22.04.4 (kernel 5.15.0-1027-xilinx-zynqmp), Python 3.10.12, PYNQ 3.0.1; private Ethernet link behind the controller laptop, SSH user `ubuntu` | 2026-10-09 | [#87](https://github.com/psamin/robotfpga/issues/87#issuecomment-6075687430) |
| A-2 | Full precision or int8? | Full precision: BF16 weights, FP32 math; linear layers on the FPGA, the rest on the ARM; int8 retired | 2026-10-13 | owner decision, [#151](https://github.com/psamin/robotfpga/pull/151) |
| A-3 | Which task and policy? | Only the medicine-bottle pick-and-place, with the SmolVLA fine-tuned on our demos; cube sorting removed | 2026-10-14 | owner decision |
| A-4 | Bridge split | Software owns the laptop client and the arm controller; FPGA owns the board service and overlays | 2026-10-09 | [#87](https://github.com/psamin/robotfpga/issues/87#issuecomment-6075858540) |
| A-5 | Cluster GPUs | `general` (1-day limit): H100 80 GB, H200 141 GB ×4, L40S 48 GB ×4, L4 24 GB ×4; `techfee` 3 days. Use GPU nodes even for CPU-only jobs | 2026-10-14 | `sinfo`, job logs |
