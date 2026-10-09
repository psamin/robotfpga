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
