# FPGA status

Updated October 8, 2026 (America/New_York). Documentation issue
[#63](https://github.com/psamin/robotfpga/issues/63).
This is the current FPGA status; dated audits remain historical snapshots.

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

## Executed host verification

Vivado/XSim 2022.2, build 3671981. Target for standalone out-of-context
synthesis: `xck26-sfvc784-2LV-c`. These results were executed on the Windows
host and are documented in the corresponding PRs.

| Work | Issue / review | Executed evidence | Review state |
|---|---|---|---|
| Signed INT8 MAC | [#59](https://github.com/psamin/robotfpga/issues/59) / [PR #60](https://github.com/psamin/robotfpga/pull/60) | 338,736 cycle checks; synthesis: 94 LUTs, 32 registers, 0 DSPs | Open, not merged |
| Independent MAC lanes | [#61](https://github.com/psamin/robotfpga/issues/61) / [PR #62](https://github.com/psamin/robotfpga/pull/62) | 3,278,586 lane checks for 1/3/8 lanes; 8-lane synthesis: 745 LUTs, 256 registers, 0 DSPs | Open, stacked on #60 |
| Reproducible synthesis | [#65](https://github.com/psamin/robotfpga/issues/65) / [PR #66](https://github.com/psamin/robotfpga/pull/66) | Both tops passed at 10 ns with reports/checkpoints; invalid top/lane/period rejected | Open, stacked on #62 |
| Testbench fault detection | [#67](https://github.com/psamin/robotfpga/issues/67) / [PR #68](https://github.com/psamin/robotfpga/pull/68) | Clean baselines passed; four valid RTL mutants failed explicit scoreboard checks | Open, stacked on #66 |

The array change also reran the original MAC test: 338,736 checks passed.
Simulation includes signed boundaries, reset/clear priority, enabled
accumulation, disabled hold, INT32 wrap and reproducible randomized tests.
Array tests verify lane isolation and non-default lane counts.
The fault harness detected zero extension, clear/enable priority inversion,
ignored enable and shared lane enables. Each mutant compiled/elaborated;
tool errors were not treated as detection. Production RTL was untouched.

Timing reports are preliminary: the original scratch runs applied clocks
after synthesis. The tracked runner in #66 now reads clocks before synthesis
and reproduces the same resource counts. Interface delays are absent and
`HD.CLK_SRC` is unset in both workflows.
No placement/routing, board validation, full-network latency or power
measurement is established by these results. DSP mapping was not forced.

## Ownership and next steps

The friend handles current card/board setup; no completion is assumed.
This FPGA workstream owns issues #59/#61/#63/#65/#67, assigned to Dhyey234.
Existing reference, vectors, HLS and board-runner work retains its issue
owners. Do not duplicate it or change shared formats without joint review.

1. Confirm the card image and successful Linux boot with the setup owner.
2. Record board identity, OS, firmware, drivers and Python before selecting
   installation or firmware changes; establish the actual SSH endpoint.
3. Execute the existing M0 runbook's verified resizer/DMA checks, save
   concise evidence and repeat after reboot. Only then advance M1.
4. Review/merge #60 before #62, then #66 and #68; retarget each dependent PR to
   main after its base lands. A merge requires approval; issues close on merge.

Maintain this page and the root README in the same PR when new confirmed
status or verification changes their meaning. Record dates, issue/PR links,
test evidence and unresolved gates. Publish branches early and push verified
work promptly. Updates follow active work and reported facts; this page
does not imply unattended board monitoring.
