# FPGA status

Updated October 9, 2026 (America/New_York). Documentation issue
[#63](https://github.com/psamin/robotfpga/issues/63).
This is the current FPGA status; dated audits remain historical snapshots.

Remote main advanced to `014718f`: software training/QAT/evaluation, v3r2
handoff and #82 are merged. Fresh main preparation produced 262 pinned files.
[Bundle integrity, PR #86](https://github.com/psamin/robotfpga/pull/86), issue #85,
adds a verifier (263-file bundle): all hashes and 100 reference outputs/40
layer dumps passed; five negative/positive integrity tests passed.

User reports the working arm is USB-connected to another control computer,
with the KR260 connected to that computer by Ethernet. Proposed deployment:
keep camera/USB motor control on that computer, send observations to a KR260
inference service and return action chunks. The real controller command/model
is not identified yet; TCP/backend integration and board overlay remain pending.

[Friend handoff, PR #82](https://github.com/psamin/robotfpga/pull/82): clone
`board/81-fpga-handoff` and run `python3 board/demo/prepare_handoff.py`.
A fresh single-branch clone gathered 262 pinned files; all 100 trained outputs
and 40 layer dumps passed preflight. The runbook includes build/board commands;
`policy.bit`/`policy.hwh` still require Vivado integration and delivery.

## Board bring-up: M0 pending

- User reports that the KR260 has arrived.
- A teammate/friend is preparing the Ubuntu microSD card and handling setup.
- The last confirmed board state is not booted. Card preparation completion,
  image filename/version, firmware inventory and SSH access are unconfirmed.
- No M0 overlay execution, output comparison or reboot acceptance has run.

The selected M0 baseline is the official Kria K26 Ubuntu 22.04 arm64 image
with Kria-PYNQ, as researched in
[PR #18](https://github.com/psamin/robotfpga/pull/18). This is a selected
baseline, not a claim about the image currently being written. The existing
[inventory collector, PR #20](https://github.com/psamin/robotfpga/pull/20),
is the starting point once Linux access is established.

## Initial arm demo requirements (reported October 9)

- The first demo must use FPGA inference; CPU inference is not the requested fallback.
- User reports SmolVLA training on a GPU, with model readiness estimated in
  about 90 minutes at the time of the report. This is not a delivery guarantee.
- User confirms the FPGA model is the small INT8 convolution/FC policy.
  SmolVLA training is separate; build-spec section 9 excludes running it
  on this FPGA accelerator in the current project scope.
- Ubuntu is expected from the setup owner; PYNQ, SSH and a working overlay
  remain uncertain. The user identifies the arm as LeRobot SO-101 and confirms
  teleoperation and model/controller-driven motion work outside this audit.
- Software owner psamin is tracking a tiny-policy weights/vector release
  under [issue #73](https://github.com/psamin/robotfpga/issues/73). Its publication
  is published in [PR #80](https://github.com/psamin/robotfpga/pull/80), branch
  `handoff/v3r2`, commit `5ee0e03`; no GitHub release
  was found. Trained weights, 100 vectors, 40 layer dumps and eval JSONs are present.

Identify the existing real-arm controller entry point before connecting
FPGA actions to it. GPU training does not produce the
FPGA bitstream; HLS export, Vivado integration, board acceptance and golden
output checks remain necessary under the existing owners and milestone gates.

Remote branches refreshed and inspected October 9: `sw/16-final` contains
`armlab/backends/base.py`, `intref_backend.py` and `control/chunking.py`;
`eval/run.py` routes them to MuJoCo, not physical motors. `board-run-policy`
contains the prototype `board/run_policy.py` FPGA inference class. No
real-arm adapter, teleoperation entry point, `fpga_backend.py` or
`remote_backend.py` was found in the inspected remote branch heads.
The planned `armlab/robot/so101.py` remains absent there. The working
controller may be external or unpublished; its location is not established.

The [handoff checker, PR #77](https://github.com/psamin/robotfpga/pull/77), passed
the trained v3r2 manifest, weight hash/layout and vector sizes, then all 100
reference outputs and 40 layer dumps. The optimized HLS C model also matched
all 100 outputs/40 dumps; its real vendor-header stream test passed 100 vectors.
Trained-vector 000 Verilog/XSim co-simulation passed all 48 bytes/TLAST:
1,359,007 inference cycles (13.590 ms at simulated 100 MHz).
Weights SHA256: `c87abfae0cbedf771a9ef927f39302a43cdcbb954f7090aca4baa6bef350a732`.
Windows archive extraction changed `089_out.bin` to 51 bytes under `text=auto`;
the raw Git blob is correct (48 bytes). Exact blob extraction passed validation.
Do not use newline-converted binary copies. Published simulation results report
INT8 Stage C 180/200 success (90%) versus float 196/200 (98%); these are not
physical-arm or FPGA results. No new bitstream/controller was included.

## Executed host verification

[Stream error/recovery tests, PR #84](https://github.com/psamin/robotfpga/pull/84),
issue #83: real Vitis C simulation passed 100 trained vectors, seven error paths
and seven valid recoveries. Bad mode, shifts 0/31 and malformed input/weight
TLAST are checked; four malformed artifact fixtures failed promptly with exit 1.
This changes the testbench only; no additional RTL or board acceptance is claimed.

Vivado/XSim 2022.2, build 3671981. Target for standalone out-of-context
synthesis: `xck26-sfvc784-2LV-c`. These results were executed on the Windows
host and are documented in the corresponding PRs.

| Work | Issue / review | Executed evidence | Review state |
|---|---|---|---|
| Signed INT8 MAC | [#59](https://github.com/psamin/robotfpga/issues/59) / [PR #60](https://github.com/psamin/robotfpga/pull/60) | 338,736 cycle checks; synthesis: 94 LUTs, 32 registers, 0 DSPs | Open, not merged |
| Independent MAC lanes | [#61](https://github.com/psamin/robotfpga/issues/61) / [PR #62](https://github.com/psamin/robotfpga/pull/62) | 3,278,586 lane checks for 1/3/8 lanes; 8-lane synthesis: 745 LUTs, 256 registers, 0 DSPs | Open, stacked on #60 |
| Reproducible synthesis | [#65](https://github.com/psamin/robotfpga/issues/65) / [PR #66](https://github.com/psamin/robotfpga/pull/66) | Both tops passed at 10 ns with reports/checkpoints; invalid top/lane/period rejected | Open, stacked on #62 |
| Testbench fault detection | [#67](https://github.com/psamin/robotfpga/issues/67) / [PR #68](https://github.com/psamin/robotfpga/pull/68) | Clean baselines passed; four valid RTL mutants failed explicit scoreboard checks | Open, stacked on #66 |
| Standalone requantization | [#69](https://github.com/psamin/robotfpga/issues/69) / [PR #70](https://github.com/psamin/robotfpga/pull/70) | 261,663 XSim checks; synthesis: 154 LUTs, zero registers/latches/DSPs | Open, independent from main |
| Direct requant reference check | [#71](https://github.com/psamin/robotfpga/issues/71) / [PR #72](https://github.com/psamin/robotfpga/pull/72) | 281,652 actual NumPy cases matched RTL; malformed/partial/extra files failed | Open, stacked on #70 |
| Per-lane bias/output | [#74](https://github.com/psamin/robotfpga/issues/74) / [PR #75](https://github.com/psamin/robotfpga/pull/75) | 249,228 lane checks; eight-lane synthesis: 1,643 LUTs, zero registers/latches/DSPs | Open, stacked on #70 |
| Existing HLS accelerator | [#11](https://github.com/psamin/robotfpga/issues/11) / [PR #12](https://github.com/psamin/robotfpga/pull/12) | Real Vitis C simulation: 100 stand-in vectors passed; synthesis and IP export completed | Draft; latency/integration gates pending |

Vitis HLS 2022.2 completed the existing accelerator build after a Windows
project-path fix, pushed to #12. Estimates: 153 BRAM18, 14 URAM, 11 DSP,
5,472 registers and 17,102 LUTs; estimated period 6.598 ns against a 10 ns target.
Worst-case latency is 10,251,374 cycles (about 103 ms at 100 MHz), exceeding
the hard 33 ms ceiling in the baseline. Coordinated eight-channel optimization
[#78 / PR #79](https://github.com/psamin/robotfpga/pull/79) reduces the maximum
top estimate to 1,367,716 cycles (13.677 ms): all 100 final vectors and 40 layer
dumps match, all reduction loops achieve II=1, and IP export completed.
Estimated resources: 171 BRAM18, 24 URAM, 66 DSP, 8,918 FF and 38,649 LUT;
period 5.953 ns. XSim Verilog co-simulation passed one stand-in vector:
569,580 cycles for weight loading and 1,359,007 for inference (13.590 ms).
All 48 bytes/TLAST matched. Process-local removal of inherited `DEBUG=release`
fixed Vitis's generated compiler command; reproduction is documented in #79.
Still above 5 ms; full trained RTL vector coverage, system resources,
placement/routing and board latency remain unverified.

The array change also reran the original MAC test: 338,736 checks passed.
Simulation includes signed boundaries, reset/clear priority, enabled
accumulation, disabled hold, INT32 wrap and reproducible randomized tests.
Array tests verify lane isolation and non-default lane counts.
The fault harness detected zero extension, clear/enable priority inversion,
ignored enable and shared lane enables. Each mutant compiled/elaborated;
tool errors were not treated as detection. Production RTL was untouched.
Requantization tests cover shifts 1..30, both clipping modes, negative ties,
clamp transitions, INT32 extrema, invalid shifts and 100,000 random inputs.
The independent division/floor oracle follows the inspected integer reference.
The actual pinned `requant()` now also executed in Python 3.8.3/NumPy 1.18.4:
all 281,652 generated cases matched XSim (543,315 total combined checks).
Metadata records reference revision, hashes, versions, seed and count.
Whole-network/trained-model golden comparisons remain pending.
Rounding uses a 33-bit intermediate, without changing the network's separate
accumulator-domain checks. Parent timing constraints are required for this
combinational block; no layer or board interface has been added.
The October 9 bias/output stage adds each INT32 bias using an exact 33-bit
sum, then uses requantization. Per-lane overflow/reference-range errors and
shared invalid-shift flags force invalid outputs to zero. Consumers must
check flags and supply an unwrapped upstream product total. This preserves
the reference's conservative range restriction; it does not detect past
MAC overflow or define a complete layer/accelerator error protocol.

Timing reports are preliminary: the original scratch runs applied clocks
after synthesis. The tracked runner in #66 now reads clocks before synthesis
and reproduces the same resource counts. Interface delays are absent and
`HD.CLK_SRC` is unset in both workflows.
No placement/routing, board validation, full-network latency or power
measurement is established by these results. DSP mapping was not forced.

## Ownership and next steps

The friend handles current card/board setup; no completion is assumed.
This FPGA workstream owns issues #59/#61/#63/#65/#67/#69/#71/#74/#76/#78/#81/#83/#85, assigned to Dhyey234.
Existing reference, vectors, HLS and board-runner work retains its issue
owners. Do not duplicate it or change shared formats without joint review.
With owner coordination confirmed, #11 also assigns Dhyey234; its original
owner remains assigned. The Windows fix and build evidence use existing PR #12.

1. Confirm the card image and successful Linux boot with the setup owner.
2. Record board identity, OS, firmware, drivers and Python before selecting
   installation or firmware changes; establish the actual SSH endpoint.
3. Execute the existing M0 runbook's verified resizer/DMA checks, save
   concise evidence and repeat after reboot. Only then advance M1.
4. Review/merge #60 before #62, then #66 and #68; retarget each dependent PR to
   main after its base lands. A merge requires approval; issues close on merge.
   Requantization #70 is independently reviewable against main; #72 and #75 follow it.

Maintain this page and the root README in the same PR when new confirmed
status or verification changes their meaning. Record dates, issue/PR links,
test evidence and unresolved gates. Publish branches early and push verified
work promptly. Updates follow active work and reported facts; this page
does not imply unattended board monitoring.
