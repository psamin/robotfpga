# AXI-stream error and recovery tests

Issue #83, stacked on the eight-channel accelerator (#79). This extends
`hls/tb_top.cpp`, using the real Xilinx AXI-stream types and vendor headers.
The shared numerical reference, packet formats and production kernel are unchanged.

The testbench checks version `0x00000100`, exact weight/input/output byte
counts, all output bytes, output KEEP/STRB, TLAST and complete stream consumption.
An absent next vector ends a deliberately smaller fixture; an existing empty
input file fails. Use the #76 preflight for the mandatory 100-vector handoff.

| Error transaction | Expected result |
|---|---|
| Mode 3 | Status 3; input unchanged; no output |
| First-layer shift 0 or 31 | Status 1; input consumed; 48 zero bytes with final TLAST |
| Input TLAST on first byte or absent | Status 2; full input consumed; golden output/TLAST |
| Weight TLAST on first byte or absent | Status 2; full weights consumed; no output |

After each of these seven cases, a valid inference must match vector 000.
Malformed weight framing is followed by a valid weight reload before recovery.
The test supplies the full byte count even for malformed TLAST: a physically
short stream would block waiting for bytes and needs separate timeout testing.

## Executed October 9, 2026

Vitis HLS 2022.2 build 3670227, Windows, real vendor-header C simulation:

- All 100 trained v3r2 vectors passed; zero failures.
- Seven error paths and seven subsequent valid recoveries passed.
- Truncated weights, truncated input, truncated output and an empty input
  each exited 1 with an explicit byte-count/empty-file failure, without hanging.

Artifacts: handoff commit `5ee0e03`, weights SHA256
`c87abfae0cbedf771a9ef927f39302a43cdcbb954f7090aca4baa6bef350a732`.
Log: `hls/build/policy_hls/sol1/csim/report/policy_top_csim.log`.
The executed run performed C simulation only; synthesis was not repeated for
a testbench-only change. No additional RTL or board acceptance is claimed.

## Reproduce

Stage exact trained artifacts at `artifacts/standin/` as in the handoff runbook.
From `hls/`, the existing `vitis_hls -f run_hls.tcl` executes these tests before
synthesis/IP export. Require both the seven-case message and the 100-vector
zero-failure PASS message. A single-vector RTL fixture also runs the seven
error/recovery cases with this testbench, increasing simulation time.
Use an isolated tool project when running concurrently with board-owner work.
