# M0: KR260 Python overlay bring-up

**Hardware acceptance: NOT RUN.** The KR260 has not arrived (October 1, 2026).
No installation, firmware update or overlay execution has been performed.
This preparation does not pass M0. This workstream must not begin M1 until
the actual M0 acceptance below passes.

Scope: FPGA board bring-up only. Existing plan/HLS/reference/DMA work stays
with its current owners and PRs. This document supplements the FPGA plan;
it does not change the network, arithmetic, packet format or interface.

## Compatible workflow researched before installation

Use the **official Kria K26 Ubuntu Desktop 22.04 arm64** image with
**Kria-PYNQ**, then run the vendor **HelloWorld 3.0.0** resizer from Python.
The inspected installer accepts Ubuntu 22.04 and rejects other releases.
AMD's board support for Ubuntu 24.04 does not establish compatibility with
this installer. Generic Ubuntu installation media is not the selected image.

The installer's core constraints are Python **3.10.x**, PYNQ **3.0.1**,
NumPy **1.26.4**, pynqmetadata **0.1.2**, and pynqutils **0.1.1**. Board Python
is separate from the software team's Python 3.12 workstation environment.
Exact source revisions, artifact URLs and upstream hashes are in
[versions.json](versions.json).

AMD recommends **2022.1 K26 boot firmware or later, except 1.04**, for this
OS. Inspect the actual firmware before selecting any update. Do not choose
the newest firmware blindly: the 2026.1 release changed to board-specific
flat boot images. Firmware and Vivado release numbers need not match.

**Vivado, Vitis, Vitis HLS and PetaLinux are not required to run this prebuilt
M0 overlay.** Their later build versions remain unselected. No DPU/model
conversion is part of this acceptance test.

The full stack is not locked or hardware-validated yet: record the SD image
identity/hash, Python patch, kernel, XRT/zocl, xmutil and dfx-mgr versions on
the target. The vendor installer leaves some apt/pip dependencies unpinned.
Resolve this inventory before installing or configuring anything.

## Installation review points

- HelloWorld's README mentions PYNQ 2.7, but its package is 3.0.0 and the
  Kria-PYNQ installer includes it with PYNQ 3.0.1 on KR260. Keep this
  discrepancy visible until the pinned artifacts pass on hardware.
- Upstream explicitly maps KR260 to the `kv260`-named resizer bitstream and
  HWH. This does not authorize arbitrary KV260 overlays on KR260.
- Verify the downloaded 3.0.0 sdist's KR260 mappings against the inspected
  source, then verify the corresponding bitstream/HWH and record SHA256.
- The installer modifies apt sources, services and device-tree setup. Its
  KR260 branch removes `vitis-ai-runtime` before installing a DPU example.
  Review existing software first; document any departure from the recipe.
- The generated KR260 self-test runs DPU tests, not this resizer test.
  Successful imports or a notebook without output assertions do not pass M0.

## Incremental execution after board arrival

1. Confirm KR260 identity and boot Linux. Collect read-only OS, firmware,
   kernel, driver and Python inventory. Do not source the PYNQ profile during
   inventory: the profile can insert a device-tree overlay.
2. Compare inventory with the researched baseline. Resolve discrepancies;
   preserve working installations rather than reinstalling blindly.
3. Review and, if necessary, install the pinned vendor recipe for KR260.
   Capture resolved apt/pip versions and the exact commands used.
4. Verify the matching resizer artifacts, load from Python and run the test.
5. Save evidence, reboot cleanly and repeat the documented loading/test steps.

Read-only board queries, once SSH/console access exists:

```bash
cat /etc/os-release
uname -a
python3 --version
sudo xmutil boardid
sudo xmutil bootfw_status
```

Inspect an existing PYNQ environment with its interpreter directly, rather
than importing PYNQ or sourcing a profile during preflight:

```bash
/usr/local/share/pynq-venv/bin/python -m pip freeze
```

Missing commands or permission failures are unresolved observations, not
permission to assume compatible versions. Do not put passwords in logs.

## Acceptance on the actual board

- Board identity and Linux boot are confirmed; the complete runtime inventory
  and any departure from the baseline are recorded.
- Python loads the hash-verified resizer bitstream with matching HWH.
  `axi_dma_0` and `resize_accel_0` are present and usable.
- Resize deterministic 96x96 RGB images to 48x48 in fabric. Use spatially
  constant images with distinct channel values: bilinear interpolation has
  an exact constant expected result. Include 0 and 255 across at least four
  patterns; run the sequence three times and compare every output byte.
- Initialize each output buffer to a differing value to expose missing writes.
  Transfers finish within a bounded timeout with no hidden DMA errors.
- Save commands, artifact hashes, runtime versions, byte-comparison results,
  exit status and diagnostic durations; repeat after a clean reboot.

These example transfers do not define the neural-network interface. M1's
27,658-byte loopback remains a separate hardware milestone.

## Collaboration and reproducibility

Claim work through an assigned issue before code; use one issue per small PR.
Push reviewable checkpoints to your branch and record progress in the PR.
Do not push directly to main or duplicate another owner's open work.
Use `board/m0/` for this workstream. Do not edit shared contract files or the
existing FPGA plan without the required coordination.

Keep vendor downloads/builds/raw inventories in ignored `artifacts/m0/`.
Commit authored sources and reviewed concise evidence, not binaries or secrets.
This documentation can land before hardware arrives; landing it does not
close the hardware milestone.

## Sources

- [Pinned Kria-PYNQ installer](https://github.com/Xilinx/Kria-PYNQ/blob/8b975f43706490d179002900ff281480f7799ca5/install.sh)
- [AMD OS/firmware compatibility](https://xilinx-wiki.atlassian.net/wiki/spaces/A/pages/1641152513/Kria+SOMs+Starter+Kits)
- [HelloWorld package metadata](https://pypi.org/pypi/pynq-helloworld/3.0.0/json)
- [Pinned KR260 artifact mapping](https://github.com/Xilinx/PYNQ-HelloWorld/blob/9728edf2ea6b7bc157acb126f773e93c3e9561c2/pynq_helloworld/notebooks/edge/resizer.bit.link)
- [Matching HWH mapping](https://github.com/Xilinx/PYNQ-HelloWorld/blob/9728edf2ea6b7bc157acb126f773e93c3e9561c2/pynq_helloworld/notebooks/edge/resizer.hwh.link)
- [Vendor Python example](https://github.com/Xilinx/PYNQ-HelloWorld/blob/9728edf2ea6b7bc157acb126f773e93c3e9561c2/pynq_helloworld/notebooks/edge/resizer_pl.ipynb)
