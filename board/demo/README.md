# Pullable KR260 handoff

This branch gathers exact reviewed-in-progress revisions; it does not merge
their PRs. Python 3.8+, Git and access to this repository are required.
Trained v3r2 weights are included through the pinned handoff in PR #80.
The preparation script copies raw Git bytes, avoiding Windows binary newline
conversion. It refuses an existing output directory and writes per-file
SHA256 provenance to `bundle.json`. It never accesses the board or motors.

## Your friend: get everything published so far

Use a fresh clone; these commands do not switch an existing checkout:

```sh
git clone --branch board/81-fpga-handoff https://github.com/psamin/robotfpga.git robotfpga-handoff
cd robotfpga-handoff
python3 board/demo/prepare_handoff.py
cd build/fpga-handoff
python3 -m pip install numpy
python3 board/demo/handoff_check.py artifacts/standin --reference ref/intref.py --verify-reference
```

On Windows use `python` instead of `python3` (or the installed Vivado Python).
The preparation command fetches missing pinned commits. The output contains:

- `hls/`: optimized eight-channel kernel, testbenches and Vitis build script.
- `artifacts/standin/`: **trained v3r2**, despite the legacy directory name;
  weights, manifest, 100 input/output vectors, 40 layer dumps and eval reports.
- `ref/intref.py`, `board/demo/handoff_check.py`: exact reference and preflight.
- `board/run_policy.py`, `board/loopback.py`: existing board-owner prototypes.
- `board/m0/`: inventory collector, version pins and first-boot runbook.

Expected trained weight SHA256:
`c87abfae0cbedf771a9ef927f39302a43cdcbb954f7090aca4baa6bef350a732`.
Preflight must report PASS, 100 reference calls and 40 layer dumps.
Keep generated bundles, logs and overlays out of commits.

## FPGA workstation: build the IP

Use Vivado/Vitis HLS **2022.2**, targeting `xck26-sfvc784-2LV-c`.
From the prepared directory, enter `hls/` and run:

```sh
vitis_hls -f run_hls.tcl
```

Windows: invoke `C:/Xilinx/Vitis_HLS/2022.2/bin/vitis_hls.bat` if not on PATH.
Require the 100-vector PASS message, successful synthesis, and
`hls/build/policy_hls/sol1/impl/export.zip` plus `impl/ip/component.xml`.
The export is an IP block, **not a runnable board overlay**.

## Missing deliverable: Vivado board overlay

No `policy.bit`/`policy.hwh` is published. The FPGA integration owner must:

1. Complete M0 acceptance using `board/m0/README.md`, then M1 DMA loopback.
2. Create a KR260 Vivado block design with the MPSoC board preset, 100 MHz
   PL clock, DDR access through `S_AXI_HP0_FPD`, and matching clock/reset wiring.
3. Add AXI DMA in simple mode, 8-bit streams, **26-bit buffer length**.
   Weight loading is 568,240 bytes; inference input is 27,658 bytes.
4. Add the exported HLS IP repository; instantiate `policy_top_0` and
   `axi_dma_0`. Connect DMA MM2S to policy input and policy output to DMA S2MM.
   Wire AXI-Lite control and DMA DDR masters through connection automation.
5. Validate, synthesize, implement, check timing/resources, generate the
   bitstream, and collect matching hardware handoff metadata.
6. Deliver `policy.bit` and `policy.hwh` with the same basename to the board.

These are integration requirements from the existing FPGA plan, not an
executed design. Keep M0/M1/M4 owners coordinated; no milestone is accepted
by preparing this directory. The board needs working Ubuntu, PYNQ, NumPy,
DMA allocation support and SSH/file transfer before the next commands.

## Connected board: verify before connecting the arm

Copy the prepared directory and matching overlay pair to the KR260. Follow
the M0 runbook and inventory first. Then confirm IP names and registers in
PYNQ: `Overlay('policy.bit').policy_top_0.register_map`.
Generated HLS offsets: mode `0x18`, shifts_lo `0x20`, shifts_hi `0x28`,
version `0x30`, status/ap_return `0x10`, control `0x00`.

From the prepared directory on the board:

```sh
python3 board/demo/handoff_check.py artifacts/standin --reference ref/intref.py --verify-reference
timeout 300s python3 -u board/run_policy.py policy.bit artifacts/standin
```

The prototype runner has unbounded internal waits and is untested on hardware;
the shell timeout prevents an indefinite process. A timeout needs operator
diagnosis/recovery, not automatic retries. The runner can exit zero after
mismatches: require **100 vectors, zero failures**, and save its latency log.
Measure end-to-end latency; 13.59 ms is one simulated RTL inference, not board
timing. Under 5 ms remains the target; 33 ms is the hard ceiling.

This verifies inference only. The real SO-101 controller, camera calibration
and action safety integration are still external/unpublished; these commands
do not move the arm.
