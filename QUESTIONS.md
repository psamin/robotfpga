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
| Policy | Fine-tuned SmolVLA `smolvla-so101_medicine_bottle_pickplace-9604`, the best on this task (offline action error 0.080; every co-trained or augmented variant tried is 0.084–0.121). Download: GitHub release `policy-smolvla-9604` (checkpoint, replay batches, SHA256SUMS) | [robot/README.md](robot/README.md) §Running the best policy |
| Post-training (Alan) | Approved in #156. Built, all open PRs stacked from #172 to #200: randomized pick-and-place scene, reset families and buffer, CPU and Warp teacher envs, PPO + SGS training job, held-out eval, calibrated-camera rendering, SmolVLA-in-sim baseline runner. Scripted expert: 0.75 on 80 held-out rest starts. Waiting on: the first GPU training run, the camera photo session, then the SmolVLA baseline. No training run yet | [docs/smolvla-post-training.md](docs/smolvla-post-training.md) |
| Real arm | Follower + leader calibrated; teleop / record / rollout toolkit works | [robot/](robot/README.md) |
| FPGA engine | **Runs SmolVLA on the KR260** (2026-10-22): `float_tp4.bit` (TP = 4, 250 MHz) with all 303 linear layers on the FPGA; `serve_policy_fpga.py --selftest` PASS, max 0.0043 vs PyTorch FP32 on the 3 release frames (same as job 10279). `check_engine.py` on the board: all 5 cases PASS | [docs/smolvla-fpga.md](docs/smolvla-fpga.md) §FPGA side, #214 |
| FPGA speed (measured) | 52–56 s per 50-action chunk on the board: ~32 s in the 1,419 engine calls including the socket copies, ~22 s on the ARM (attention, norms). Unprofiled split inside the 32 s; the FPGA compute alone is ~8 s by the model, so most of it is per-call overhead | #214 |
| KR260 | Ubuntu 22.04.4 + PYNQ 3.0.1 at 192.168.137.35 (laptop Internet Connection Sharing). CMA 1,000 MB; 4 GB swap added; Python 3.12 venv with `psamin/lerobot@e624f3f7` (torch 2.8 CPU) at `/home/ubuntu/lerobot-venv`; desktop off for RAM | `fpga/board/setup_board.sh` |
| Laptop ↔ board link | A board TCP service and laptop client exist from the earlier int8 work (outside git); the full-precision runtime is `fpga/board/runtime.py` | A-4 |

## Open

| # | Asked by | For | Question | Link |
|---|---|---|---|---|
| Q12 | psamin | FPGA (@athithan-elamaran1, @Dhyey234) | **Done on the board (2026-10-22, #214).** Bitstream `float_tp4.bit`: post-route WNS −0.026 ns at 300 MHz, run at 250 MHz; 644 DSP, 52.7k LUT, 74.5 BRAM tiles, 48 URAM; every loop II = 1. `check_engine.py` on the KR260: all 5 cases PASS with the C model's errors. Bandwidth: unmeasured as a clean figure (the small cases report 0.7–1.1 GB/s, dominated by per-call Python overhead) | [docs/smolvla-fpga.md](docs/smolvla-fpga.md) §FPGA side |
| Q13 | psamin | psamin | Speed vs precision: FP32 math gives ~3–4 s per chunk. Options are BF16 multiplies with FP32 accumulation (several times more MACs per DSP), running the vision encoder less often, or accepting a slower control rate | [docs/smolvla-fpga.md](docs/smolvla-fpga.md) §Verification |
| Q16 | m-alan08 | psamin | Joint mapping, derived and checked against the demos, not measured on the arm. Body joints: sim radians = LeRobot degrees, same sign, 0 at mid travel (#171, #173). Gripper: reading 0 to 100 spans the calibrated 137.8 deg, 0 = jaw pads touching, clipped to the sim's 110 deg (#175); rest reading gives 3.6 mm, held-cap reading 21 mm vs 27 mm from the video. A jaw-gap check on the arm at a few readings would confirm | [policy/sim/so101.py](policy/sim/so101.py) |
| Q17 | psamin | FPGA (@athithan-elamaran1, @Dhyey234) | **Done (2026-10-22).** A Python 3.12 venv installs next to PYNQ's 3.10 (offline kit, `deploy_smolvla.py --setup`); PYNQ's Python is `/usr/local/share/pynq-venv/bin/python3` (3.10.12); CmaTotal 1,024,000 kB. Needed two fixes: weights packed into 32 MB CMA slabs (one CMA buffer per layer wasted ~300 MB on alignment) and `load_vlm_weights=false` in the board's copy of the config (true downloads + loads the 2 GB SmolVLM2 backbone first: 3.19 vs 2.35 GB peak RSS, identical actions). Selftest: PASS 0.0043, 52–56 s per chunk, 32 s in FPGA calls | `fpga/board/deploy_smolvla.py`, #214 |
| Q18 | psamin | FPGA (@athithan-elamaran1, @Dhyey234) | Measured: ~22 s per chunk on the ARM (attention + non-linear ops), next to ~32 s in engine calls. Plan, in order of payoff per effort: cut the per-call overhead (engine in the policy process or shared memory, uncached buffers; software only), then a vision-attention HLS kernel, then BF16 multiplies (Q13). Not started | #214 |

## Answered

| # | Question | Answer | Date | Evidence |
|---|---|---|---|---|
| A-7 | Q15: measure the bottle, plate and layout? | Not needed. The scene randomizes sizes, mass, friction and positions across the desk (psamin's suggestion), over ranges centred on estimates from the demo joint data and front video (bottle start (0.25, +0.09) m, plate center (0.33, -0.19) m, plate about 19 cm, cap about 2.7 cm, body about 3.5 cm) | 2026-10-14 | [docs/smolvla-post-training.md](docs/smolvla-post-training.md), #172, #174, #176, #177 |
| A-6 | Q14: plan for RL post-training of SmolVLA on pick and place? | Teacher (PPO + SGS) in a new pick-and-place sim, then online DAgger into a new checkpoint from `-9604`; task-independent code restored from `archive/cube-sorting`. Scene blocked on Q15 | 2026-10-14 | [docs/smolvla-post-training.md](docs/smolvla-post-training.md), #158 |
| A-1 | KR260 boot, versions, endpoint | Ubuntu 22.04.4 (kernel 5.15.0-1027-xilinx-zynqmp), Python 3.10.12, PYNQ 3.0.1; private Ethernet link behind the controller laptop, SSH user `ubuntu` | 2026-10-09 | [#87](https://github.com/psamin/robotfpga/issues/87#issuecomment-6075687430) |
| A-2 | Full precision or int8? | Full precision: BF16 weights, FP32 math; linear layers on the FPGA, the rest on the ARM; int8 retired | 2026-10-13 | owner decision, [#151](https://github.com/psamin/robotfpga/pull/151) |
| A-3 | Which task and policy? | Only the medicine-bottle pick-and-place, with the SmolVLA fine-tuned on our demos; cube sorting removed | 2026-10-14 | owner decision |
| A-4 | Bridge split | Software owns the laptop client and the arm controller; FPGA owns the board service and overlays | 2026-10-09 | [#87](https://github.com/psamin/robotfpga/issues/87#issuecomment-6075858540) |
| A-5 | Cluster GPUs | `general` (1-day limit): H100 80 GB, H200 141 GB ×4, L40S 48 GB ×4, L4 24 GB ×4; `techfee` 3 days. Use GPU nodes even for CPU-only jobs | 2026-10-14 | `sinfo`, job logs |
