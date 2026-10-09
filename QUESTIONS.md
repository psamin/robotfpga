# Questions and context

The shared board for both teams. **Read this after every pull. Update it before every push**
if you asked, answered, or changed something the other side needs to know.

- Add a question under **Open** with who it is for. Answer it in place, then move it to
  **Answered** with the date and a link to the evidence (file, PR, log).
- Write "unknown" or "not ready" rather than guessing. Say whether something was run or only planned.
- Keep **Current state** short and true. Replace stale lines instead of appending.
- Long discussions belong in an issue; link the issue here.

## Current state (2026-10-09)

Full setup reference: [SETUP.md](SETUP.md).

| Area | State | Where |
|---|---|---|
| Trained policy | TinyPolicy int8 v3r2, sim only. Int8: Stage B 91%, Stage C 90%. Float: 100% / 98% | [handoff/v3r2/README.md](handoff/v3r2/README.md) |
| Policy code | Sim, expert, DAgger, training, QAT, eval merged to `main` | [armlab/](armlab/), [plans/tiny-policy-log.md](plans/tiny-policy-log.md) |
| FPGA handoff | Weights + 100 golden vectors pass the HLS C model (100/100) | [README.md](README.md), [board/demo/README.md](board/demo/README.md) |
| Bitstream | Board owner's 16-lane layer engine: `policy.bit` / `policy.hwh` built (Vivado 2022.2, 100 MHz, WNS +2.371 ns), not in git yet (#98). #82's DMA design: not built | [#87](https://github.com/psamin/robotfpga/issues/87#issuecomment-6075687430), #98 |
| Board (KR260) | Ubuntu 22.04.4 + PYNQ 3.0.1. **M4 passed:** v3r2 100/100 golden vectors on hardware, 8.03 ms p50 (ARM A53 numpy 70.8 ms). Sim Stage B with the FPGA in the loop: 37/40 | [#87](https://github.com/psamin/robotfpga/issues/87#issuecomment-6075687430), #15 |
| Real SO-101 arm | Follower + leader connected and calibrated (LeRobot); working teleop/record/SmolVLA rollout stack in psamin/roboticsexp. TinyPolicy not wired to it | [SETUP.md](SETUP.md) §2 |
| Laptop ↔ board bridge | Board TCP service (§5.6 contract, port 5555) and a laptop `PolicyBackend` client run outside git: 100/100 over Ethernet, 8.8 ms round trip. Ownership still Q4 | [#87](https://github.com/psamin/robotfpga/issues/87#issuecomment-6075687430) |

## Open

| # | Asked by | For | Question | Link |
|---|---|---|---|---|
| Q4 | Dhyey234, psamin | Both sides | Agree the bridge split. Proposed: software owns `remote_backend.py` and the SO-101 controller adapter; FPGA side owns the board inference service and overlay | #87 |
| Q5 | psamin | Software (psamin) | Verify LeRobot degrees → sim radians per joint (sign and zero offset; formula in SETUP.md §4.1) | [SETUP.md](SETUP.md) §4.1 |

## Answered

| # | Question | Answer | Date | Evidence |
|---|---|---|---|---|
| A1 | Real-arm launch command, model, joint order/units, camera/preprocessing | Real-arm stack exists in psamin/roboticsexp `so101/` (SmolVLA, different task); TinyPolicy v3r2 not wired to it. Units, ports, cameras, calibration in SETUP.md §2 and §4.1 | 2026-10-09 | [#87 correction](https://github.com/psamin/robotfpga/issues/87#issuecomment-6075239385), [SETUP.md](SETUP.md) |
| A2 | Software-side review of the golden-vector generator (#6) | Approved from software; used for the v3r2 vectors, HLS C model 100/100 | 2026-10-09 | [#6 comment](https://github.com/psamin/robotfpga/pull/6#issuecomment-6075156540) |
| A3 | Q1: KR260 boot, versions, endpoint, SSH user | Boots Ubuntu 22.04.4 (kernel 5.15.0-1027-xilinx-zynqmp), Python 3.10.12, PYNQ 3.0.1. Private LAN 192.168.137.35 behind the controller laptop, SSH user `ubuntu` | 2026-10-09 | [#87](https://github.com/psamin/robotfpga/issues/87#issuecomment-6075687430) |
| A4 | Q2: M0/M1 results | M0: boot, PYNQ and custom overlay load pass. `board/m0/` inventory not run. M1 DMA loopback not run (the built design uses `m_axi`, no DMA) | 2026-10-09 | [#87](https://github.com/psamin/robotfpga/issues/87#issuecomment-6075687430) |
| A5 | Q3: matching `policy.bit` / `policy.hwh`? | Yes, for the board owner's 16-lane layer engine (source not in git yet, #98). Vitis HLS / Vivado 2022.2, 100 MHz, WNS +2.371 ns; 100/100 v3r2 vectors on hardware, 8.03 ms | 2026-10-09 | [#87](https://github.com/psamin/robotfpga/issues/87#issuecomment-6075687430), #15 |
