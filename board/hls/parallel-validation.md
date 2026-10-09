# Eight-output-channel HLS validation

Issue #78; coordinated with the existing kernel owner. Based on `hls-top`
commit `4b65668`. This implements one step of the M5 parallelism proposal in
`origin/hls-v0.1:docs/plan.md`; that historical plan is absent from this branch.
It does not advance hardware milestones or change the shared contract.

Each convolution/FC reduction step updates eight independent accumulators.
The reduction order within each output stays unchanged. Eight cyclic weight
banks supply those lanes; activations are broadcast. All current output
dimensions divide by eight, enforced by compile-time assertions.
Bias initialization and output stores may take multiple memory cycles.

## Reproduce

Supply the canonical stand-in handoff in `artifacts/standin/`, then run from
`hls/` with Vitis HLS 2022.2:

```text
vitis_hls -f run_hls.tcl
g++ -std=c++14 -O2 -Wno-unknown-pragmas tb_kernel.cpp -o build/tb_kernel.exe
build/tb_kernel.exe ../artifacts/standin
```

On Windows, invoke the installed `vitis_hls.bat` and bundled MinGW `g++.exe`
by absolute path if they are not on PATH. Tool exit status alone is insufficient:
check PASS messages, synthesis reports and the exported IP files.

## Executed October 9, 2026

Windows, Vitis HLS 2022.2 build 3670227; K26 `xck26-sfvc784-2LV-c`, 10 ns clock.
Random stand-in weights SHA256:
`7fde407f816fb0a7820c9a970b1ef5a8e7af12810cf04762a4f05fc428051d71`.

- Real vendor-header C simulation: 100 final vectors, zero failures.
- Standalone GCC kernel test: 100 final vectors and 40 intermediate layer
  dumps (first five inputs), zero failures; existing error-path checks passed.
- Synthesis: all five convolution and three FC reduction loops achieved II=1.
- Vivado IP export completed: `hls/build/policy_hls/sol1/impl/export.zip`
  and `impl/ip/component.xml` exist. Generated files remain untracked.

| HLS estimate at 100 MHz | Baseline | Eight channels |
|---|---:|---:|
| Maximum top latency, cycles | 10,251,374 | 1,367,716 |
| Maximum top latency, ms | 102.514 | 13.677 |
| Estimated period, ns | 6.598 | 5.953 |
| BRAM18 | 153 | 171 |
| URAM | 14 | 24 |
| DSP | 11 | 66 |
| Registers | 5,472 | 8,918 |
| LUT | 17,102 | 38,649 |

The top estimate includes mode-dependent paths; it is not an RTL-measured
inference latency. The approximately 7.5x reduction is promising: below
33 ms in this estimate, above the 5 ms target. Resources fit the standalone
device estimate; the complete DMA/system design needs its own resource check.
Reports: `hls/build/policy_hls/sol1/syn/report/policy_top_csynth.rpt` and
`hls/build/policy_hls/sol1/csim/report/policy_top_csim.log`.

These are stand-in C-model checks, not trained-policy validation, RTL
co-simulation, placed/routed timing, board measurements or motor control.
The under-5-ms target includes transfers and software overhead; an HLS
compute estimate alone cannot establish end-to-end acceptance.

## RTL co-simulation smoke test

After synthesis, create `artifacts/cosim-one/vectors/` and copy the original
`manifest.json`, `weights.bin`, `vectors/000_in.bin` and `vectors/000_out.bin`
there, preserving the vectors subdirectory. Leave the canonical 100-vector
handoff untouched. This limits the existing testbench to one weight-load
transaction and one inference; it is not full RTL vector coverage.

Save this Tcl in `hls/build/` and run Vitis from that directory:

```tcl
unset -nocomplain env(DEBUG)
set here [file dirname [file normalize [info script]]]
cd $here
open_project policy_hls
open_solution sol1
cosim_design -rtl verilog -tool xsim -argv [file normalize "$here/../../artifacts/cosim-one"]
exit
```

The inherited Windows `DEBUG=release` environment variable initially caused
the generated Makefile to pass `release` as a compiler input filename.
Clearing it inside the tool process resolves compilation without altering
installed tools or persistent system settings. Vitis may exit zero after an
error: require a successful co-simulation report, not just an exit code.

Executed October 9: XSim 2022.2 Verilog co-simulation **PASS**, one golden
vector, zero failures. The test checked all 48 bytes, TLAST placement,
successful status and stream consumption. Transactions in
`hls/build/policy_hls/sol1/sim/verilog/policy_top.performance.result.transaction.xml`:

| Transaction | Measured RTL cycles | At 100 MHz |
|---|---:|---:|
| Initial weight load | 569,580 | 5.696 ms |
| One inference | 1,359,007 | 13.590 ms |

The weight load is separate and parameters persist for subsequent inference.
`hls/build/policy_hls/sol1/sim/report/policy_top_cosim.rpt` reports Verilog Pass.
These are simulated accelerator transactions, without a real DMA, DDR,
software scheduling, routed clock or trained weights. There is no latency
distribution or full-vector RTL claim from this single-vector smoke test.
