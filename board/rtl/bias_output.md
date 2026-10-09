# Per-lane bias and output stage

Issue [#74](https://github.com/psamin/robotfpga/issues/74); stacked on
[requantization PR #70](https://github.com/psamin/robotfpga/pull/70).
Combinational, `LANES >= 1`, default 8. Packed lane 0 is least significant.
Inputs `totals[lane]` and `biases[lane]` are signed INT32 bit patterns;
`y[lane]` is signed INT8. `shift` and `relu` are shared across lanes.

Each exact 33-bit `sum = total + bias` feeds the unchanged requant primitive.
Per-lane `overflow` means the sum cannot fit signed INT32. `range_error`
matches the pinned reference's `_check_int32`: reject sums below `-2^31`
or at/above `2^30`, reserving the reference's conservative rounding headroom.
An INT32-fitting sum may therefore have `range_error=1` and `overflow=0`.
Shared `bad_shift` rejects shifts 0/31. Any flagged lane outputs zero;
consumers must check flags instead of treating error zeros as valid actions.
These local flags do not define an accelerator error protocol.

The input product total must not have wrapped during MAC accumulation.
This stage cannot reconstruct an earlier overflow. Bias is added exactly
once, after products; do not also preload it into the upstream MAC.
The reference contract is unchanged; this is not a complete FC/conv engine.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File ./board/rtl/run_bias_output_tests.ps1
```

Tests use widened integer addition and division/floor, covering 1/3/8 lanes,
signed bias/cancellation, both overflow directions, exact reference-domain
boundaries, all shifts/ReLU modes, invalid-shift recovery and 20,000 random
cases per configuration. Seeds: `74c0ffee XOR LANES`. Logs stay in fresh
`build/bias-output-<id>/` directories; generated artifacts must not be committed.
No clock/pipeline or parent input/output timing constraints are defined.

Executed October 9, 2026 on XSim 2022.2: 249,228 lane checks passed
(20,769 / 62,307 / 166,152 for 1/3/8 lanes), without compile/elaboration
warnings. Evidence: `build/bias-output-47e3bddaea014dc187b2dd19d369761e/runner-output.txt`.
Domain behavior follows `ref/intref.py` at revision
`2155ad6ec918d1cfd25329596f8361e7ebc4d4df`; no contract was changed.

To reproduce standalone synthesis, save the following in
`build/bias-output-synthesis/synth.tcl`, then run
`vivado -mode batch -source synth.tcl` from that directory:

```tcl
read_verilog -sv {../../board/rtl/requant.sv}
read_verilog -sv {../../board/rtl/bias_output.sv}
synth_design -top bias_output -part xck26-sfvc784-2LV-c -mode out_of_context
report_utilization -file utilization.rpt
write_checkpoint -force synth.dcp
exit
```

Eight-lane K26 synthesis passed: 1,643 LUTs, 160 CARRY8, zero registers,
latches and DSPs, no black boxes. Reports/checkpoint: `build/bias-output-synthesis/`.
Warnings concern parallel-synthesis criteria and flattened floorplanning.
No placement/routing, timing closure or board execution is claimed.
