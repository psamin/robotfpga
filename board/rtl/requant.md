# Standalone requantization

Issue [#69](https://github.com/psamin/robotfpga/issues/69). Combinational
INT32-to-INT8 conversion; no clock, pipeline, bias addition or layer interface.

For shift `s` in 1..30, compute `floor((acc + 2^(s-1)) / 2^s)`, then clamp
to `[0,127]` when `relu=1`, or `[-127,127]` otherwise. Halfway ties round
toward positive infinity, including negative ties: -3 / 2 becomes -1.
The signed addition uses 33 bits, enough to avoid rounding overflow for
every INT32 input. This matches `requant()` in the inspected reference at
[`2155ad6`](https://github.com/psamin/robotfpga/blob/2155ad6ec918d1cfd25329596f8361e7ebc4d4df/ref/intref.py).
It does not change the separate network accumulator-domain checks.

Ports: signed `acc[31:0]`, unsigned `shift[4:0]`, `relu`, signed `y[7:0]`,
and local `bad_shift`. Shift 0 or 31 gives `bad_shift=1` and `y=0`; this is
a local primitive interface, not a change to accelerator error/status rules.
Valid inputs give `bad_shift=0`. Allow combinational logic to settle before
sampling outputs; board/pipeline integration is outside this task.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File ./board/rtl/run_requant_tests.ps1
```

The runner retains logs in a fresh `build/requant-<id>/` directory and requires
successful exits and a PASS marker. The widened integer test oracle uses
division with explicit floor for negative values rather than mirroring the
RTL shift. Tests cover every shift, both ReLU modes, rounding/clamp boundaries,
INT32 extrema, 100,000 seeded random inputs and invalid-shift recovery.
Seed: `69c0ffee`. Generated artifacts must not be committed.

Executed October 8, 2026: XSim 2022.2 passed 261,663 checks without
compile/elaboration warnings. Evidence:
`build/requant-724adc09fdf640f39252c7cdc9e57f71/runner-output.txt`.
The oracle follows the inspected reference formula; it does not invoke
NumPy or establish whole-network golden-vector agreement.

For standalone synthesis, save this as `synth.tcl` in `build/requant-synthesis/`
and run `vivado -mode batch -source synth.tcl` from that directory:

```tcl
read_verilog -sv {../../board/rtl/requant.sv}
synth_design -top requant -part xck26-sfvc784-2LV-c -mode out_of_context
report_utilization -file utilization.rpt
write_checkpoint -force synth.dcp
exit
```

This combinational block has no clock. Its input/output timing budget must
be defined in a parent design before timing closure can be evaluated.
K26 synthesis executed successfully on Vivado 2022.2: 154 LUTs, 10 CARRY8,
zero registers, latches and DSPs; no black boxes. Utilization and checkpoint
are in `build/requant-synthesis/`. One warning states parallel synthesis
criteria were not met. No placement/routing or board execution has run.
