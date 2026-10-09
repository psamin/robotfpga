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
| Bitstream | `policy.bit` / `policy.hwh` not built yet | — |
| Board (KR260) | No board result reported yet | #87 |
| Real SO-101 arm | Follower + leader connected and calibrated (LeRobot); working teleop/record/SmolVLA rollout stack in psamin/roboticsexp. TinyPolicy not wired to it | [SETUP.md](SETUP.md) §2 |
| Laptop ↔ board bridge | Not implemented; only the spec contract (27,658 bytes in, 48 + float32 out) | [plans/build-spec.md](plans/build-spec.md) §5.6 |

## Open

| # | Asked by | For | Question | Link |
|---|---|---|---|---|
| Q1 | Dhyey234 | Board owner (@athithan-elamaran1) | KR260 boot status, Ubuntu/Python/PYNQ versions, hostname and SSH user (no secrets) | #87 |
| Q2 | Dhyey234 | Board owner | M0/M1 results and where their logs are | #87 |
| Q3 | Dhyey234 | Board owner | Does a matching `policy.bit` / `policy.hwh` exist? Source revision, tool version, measured timing | #87 |
| Q4 | Dhyey234, psamin | Both sides | Agree the bridge split. Proposed: software owns `remote_backend.py` and the SO-101 controller adapter; FPGA side owns the board inference service and overlay | #87 |
| Q5 | psamin | Software (psamin) | Verify LeRobot degrees → sim radians per joint (sign and zero offset; formula in SETUP.md §4.1) | [SETUP.md](SETUP.md) §4.1 |

## Answered

| # | Question | Answer | Date | Evidence |
|---|---|---|---|---|
| A1 | Real-arm launch command, model, joint order/units, camera/preprocessing | Real-arm stack exists in psamin/roboticsexp `so101/` (SmolVLA, different task); TinyPolicy v3r2 not wired to it. Units, ports, cameras, calibration in SETUP.md §2 and §4.1 | 2026-10-09 | [#87 correction](https://github.com/psamin/robotfpga/issues/87#issuecomment-6075239385), [SETUP.md](SETUP.md) |
| A2 | Software-side review of the golden-vector generator (#6) | Approved from software; used for the v3r2 vectors, HLS C model 100/100 | 2026-10-09 | [#6 comment](https://github.com/psamin/robotfpga/pull/6#issuecomment-6075156540) |
