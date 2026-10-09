# KR260 handoff: step-by-step runbook

Start with [the root quick start](../../README.md). This is the detailed
sequence: prepare files → build IP → generate overlay → verify on the board.

The preparer on `main` gathers exact reviewed-in-progress revisions; it does not merge
their PRs. Python 3.8+, Git and access to this repository are required.
Trained v3r2 weights are included through the pinned handoff in PR #80.
The preparation script copies raw Git bytes, avoiding Windows binary newline
conversion. It refuses an existing output directory and writes per-file
SHA256 provenance to `bundle.json`. It never accesses the board or motors.

## 1. Your friend: pull and prepare the files

Use a fresh clone; these commands do not switch an existing checkout:

```sh
git clone --branch main https://github.com/psamin/robotfpga.git robotfpga-handoff
cd robotfpga-handoff
python3 board/demo/prepare_handoff.py
cd build/fpga-handoff
python3 board/demo/verify_bundle.py
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
The bundle verifier checks every recorded file, including sources and scripts;
run it again after copying to the board. It rejects changed/missing files and
unsafe provenance paths. It detects accidental corruption, not authenticity:
`bundle.json` is unsigned, and unlisted extra files are not verified.

## 2. FPGA workstation: build the IP

Use Vivado/Vitis HLS **2022.2**, targeting `xck26-sfvc784-2LV-c`.
From the prepared directory, enter `hls/` and run:

```sh
vitis_hls -f run_hls.tcl
```

Windows: invoke `C:/Xilinx/Vitis_HLS/2022.2/bin/vitis_hls.bat` if not on PATH.
Require the 100-vector PASS message, successful synthesis, and
`hls/build/policy_hls/sol1/impl/export.zip` plus `impl/ip/component.xml`.
The export is an IP block, **not a runnable board overlay**.

## 3. FPGA workstation: generate the missing board overlay

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

## 4. Connected board: verify before connecting the arm

Copy the prepared directory and matching overlay pair to the KR260. Follow
the M0 runbook and inventory first. Then confirm IP names and registers in
PYNQ: `Overlay('policy.bit').policy_top_0.register_map`.
Generated HLS offsets: mode `0x18`, shifts_lo `0x20`, shifts_hi `0x28`,
version `0x30`, status/ap_return `0x10`, control `0x00`.

From the prepared directory on the board:

```sh
python3 board/demo/verify_bundle.py
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

## Appendix: model handoff preflight (#76)

Issue [#76](https://github.com/psamin/robotfpga/issues/76). Run before handing
the small INT8 policy to the existing HLS/board runner; no FPGA/motors accessed.
This does not replace the software owner's artifact release (#73) or board
acceptance. It checks the existing format without changing shared contracts.

```bash
python board/demo/handoff_check.py artifacts/handoff --reference ref/intref.py
python board/demo/handoff_check.py artifacts/handoff --reference ref/intref.py --verify-reference
python board/demo/test_handoff_check.py ref/intref.py
```

Use the existing approved integer reference and Python with NumPy. The current
main branch may not contain `ref/` yet; pass its reviewed source path explicitly.
The first command validates structure, contiguous weight/bias layout, hashes,
shifts, weight range and exactly 100 inputs/outputs plus 40 layer dumps with
correct sizes. It reports `reference_verified=false`: metadata alone does not
prove inference. The second executes all 100 through the actual reference and
checks every output byte and the first five inputs' layer bytes.

Success prints JSON; any mismatch exits nonzero. Redirect stdout to an ignored
evidence directory when preserving reports. Reference timings are CPU diagnostics,
not FPGA latency. Never treat this report as permission to move the arm.
FPGA bitstream/HWH, PYNQ/DMA verification and the existing SO-101 controller
connection remain separate requirements for the requested FPGA-first demo.

Tests construct a complete zero-weight synthetic fixture using `Model.save`;
this is not the trained model. Generated model files stay in a temporary directory.

Executed October 9, 2026 with Python 3.8.3/NumPy 1.18.4: all five unit tests
passed (including complete zero-model reference inference and failure cases).
CLI structural and full-reference modes passed the randomized stand-in
generated by `ref-vectors` revision `883d6dff76d6ce34eb0e3004e55fb841ac989f79`,
seed 0: all 100 outputs and 40 layer dumps matched. Missing model path exits 1.
Stand-in weight SHA256: `7fde407f816fb0a7820c9a970b1ef5a8e7af12810cf04762a4f05fc428051d71`.
`handoff-v3r2` was not published when checked; trained-model and FPGA validation
remain pending. No motors or board hardware were accessed.
