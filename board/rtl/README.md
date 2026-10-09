# Standalone INT8 MAC

Issue: https://github.com/psamin/robotfpga/issues/59

`mac.sv` multiplies signed `a` and `b` (-128 through 127), producing a signed
16-bit product. Sign extension preserves negative products when adding them
to the signed 32-bit `acc`. One product is added on each enabled rising edge;
there is no extra product pipeline register or vendor IP.

Controls are synchronous and active high: reset > clear > enable > hold.
Reset and clear both set `acc` to zero. Disabled cycles preserve the total.
INT32 overflow wraps modulo 2^32; the network must stay within its separately
approved reference-model domain. This primitive does not add bias loading,
requantization, ReLU, saturation, or a network interface.

From the repository root in PowerShell:

```powershell
& ./board/rtl/run_mac_tests.ps1
# Override the installation path when needed:
& ./board/rtl/run_mac_tests.ps1 -VivadoBin 'C:\Xilinx\Vivado\2022.2\bin'
# If local script execution is disabled, use an approved process-only override:
powershell.exe -NoProfile -ExecutionPolicy Bypass -File ./board/rtl/run_mac_tests.ps1
```

The runner compiles, elaborates and executes XSim, requiring both successful
tool exit codes and the testbench PASS marker. Each run keeps its evidence in
a new `build/mac-<id>/` directory; do not commit generated files.
Tests cover control priority, synchronous reset, hold, signed edges, every
operand pair, positive/negative INT32 wrap, and 10,000 random cycles generated
with a fixed xorshift seed `59c0ffee`. A widened integer reference checks each
cycle after the register update; any mismatch or timeout calls `$fatal`.

## Executed verification (October 8, 2026)

Vivado/XSim 2022.2, build 3671981: PASS, 338,736 cycle checks,
seed `59c0ffee`; compilation and elaboration completed without warnings.
Evidence: `build/mac-78120002531d4fac8871e7461013991e/runner-output.txt`.
An initial attempt caught unsized reference constants and a missing DUT
timescale; both were corrected before the passing run.

Standalone out-of-context synthesis for `xck26-sfvc784-2LV-c` succeeded:
94 LUTs, 32 registers, 12 CARRY8, zero DSPs, no black boxes. Vivado chose
fabric arithmetic automatically; DSP mapping is not forced. One warning
reported that parallel synthesis criteria were not met.
The generated script/reports are in `build/mac-synthesis/`. To reproduce,
put these Tcl commands in a scratch build directory (adjust the source path)
and run `vivado -mode batch -source synth.tcl` there:

```tcl
read_verilog -sv {../../mac.sv}
synth_design -top mac -part xck26-sfvc784-2LV-c -mode out_of_context
create_clock -name clk -period 10 [get_ports clk]
report_utilization -file utilization.rpt
report_timing_summary -file timing.rpt
exit
```

The clock is applied after synthesis; these are preliminary netlist reports,
not constrained implementation. Input/output delays, placement, routing and
board validation are still required before claiming timing closure.
The report has 19 inputs and 32 outputs without delay constraints and warns
that `HD.CLK_SRC` is unset; its +8.981 ns register-path WNS does not validate
the external operand-to-accumulator path or the final clock implementation.
This is standalone preparation, not M0 acceptance or accelerator integration.

## Independent MAC lanes (issue #61)

`mac_array.sv` instantiates `LANES` copies of `mac`, default 8, minimum 1.
`a[lane]` and `b[lane]` are signed INT8 bit patterns; `acc[lane]` is an
independent signed INT32 bit pattern. Packed buses place lane 0 in the least
significant bits. Interpret slices with `$signed` when doing arithmetic.
Reset/clear affect every lane; enable is per lane. There is no cross-lane
sum, extra latency, memory controller or board interface. Eight lanes can
accept eight operand pairs per enabled clock if upstream supplies them.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File ./board/rtl/run_mac_tests.ps1 -Test mac_array
```
The testbench checks lane isolation, signed edges, independent disabled hold,
shared reset/clear priority, both INT32 wrap boundaries and 10,000 randomized
cycles against widened integer references for lane counts 1, 3 and 8.
Seeds are `61c0ffee XOR LANES`; all configurations must pass.
XSim 2022.2 passed without compile/elaboration warnings: 273,200 checks for
1 lane, 819,618 for 3 lanes, 2,185,768 for 8 lanes (3,278,586 total).
Evidence: `build/mac_array-6353a7c9f9e142b3978311f25d67c562/runner-output.txt`.
The unchanged single-MAC test also passed 338,736 checks with the updated
runner: `build/mac-d4be1a9f6c5b46d0bdb950276cc9d590/runner-output.txt`.
PR is stacked on #60; review/merge the single MAC first.
Eight-lane K26 out-of-context synthesis succeeded: 745 LUTs, 256 registers,
96 CARRY8, zero DSPs and no black boxes. One warning notes parallel synthesis
criteria were not met. Evidence is in `build/mac-array-synthesis/`.
To reproduce with the Tcl recipe above, also read
`../../board/rtl/mac_array.sv` and change `-top mac` to `-top mac_array`.
The same preliminary timing limitations apply; no placement/routing or
board acceptance has run. This does not define a network scheduling scheme.

## Reproducible synthesis (issue #65)

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File ./board/rtl/run_mac_synthesis.ps1
powershell.exe -NoProfile -ExecutionPolicy Bypass -File ./board/rtl/run_mac_synthesis.ps1 -Top mac_array -Lanes 8 -PeriodNs 10
```

The tracked Tcl script targets K26 and reads the clock constraint before
synthesis. The runner accepts `-VivadoBin`, `-Top`, `-Lanes` (1..1024) and
`-PeriodNs` (0.1..1000). `-Lanes` only configures `mac_array`; `mac` always
contains one accumulator. Lane bounds are runner limits, not a resource-fit
guarantee. Every invocation uses a fresh `build/<top>-synth-<id>/` directory
and requires successful exit plus nonempty completion marker, utilization,
timing and checkpoint files. Do not commit these generated artifacts.
Input/output delays and the parent clock location remain undefined; these
reports do not prove operand-path timing closure or board performance.
The original scratch recipes above are historical evidence; use this runner
for subsequent builds so the clock constraint participates in synthesis.
Executed October 8, 2026 on Vivado 2022.2 at 10 ns: both tops passed with
all required artifacts. Resources remain 94 LUTs/32 registers/0 DSPs for
`mac` and 745 LUTs/256 registers/0 DSPs for the eight-lane array.
Evidence: `build/mac-synth-a091b6bbe731458ca89425c39b5f1c5d/` and
`build/mac_array-synth-87d46bae33a9450e8d2fbd9752ddb65a/`.
Invalid top, zero lanes and zero period were rejected with nonzero exits.
Vivado retained the parallel-synthesis and missing `HD.CLK_SRC` warnings.

## Testbench fault detection (issue #67)

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File ./board/rtl/run_mac_mutations.ps1
```

The harness first runs clean single-MAC and array baselines, then deliberately
breaks zero/sign extension, clear priority, enable hold and per-lane enables
in separate temporary source copies under `build/mac-faults-<id>/`.
Production RTL is untouched. Each fault must trigger an explicit scoreboard
mismatch and nonzero exit; compilation errors do not count as detection.
Logs are retained and a surviving fault makes the harness fail.
Executed October 8, 2026 on XSim 2022.2: both baselines passed; all four
faults compiled/elaborated and were detected by scoreboard mismatches.
Zero extension failed at MAC check 3, ignored enable at check 10, clear
priority at check 11, and shared enable at 3-lane check 17 (lane 1).
Evidence: `build/mac-faults-f745d0927ff242c78f106a0fc2f64b2e/summary.txt`
and each case's `harness-output.txt`. This demonstrates detection of these
specific defects, not exhaustive proof of every possible RTL fault.
