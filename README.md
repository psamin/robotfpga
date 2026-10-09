# KR260 FPGA handoff — START HERE

**Use this branch: `board/81-fpga-handoff`.** It gathers the verified FPGA
sources, trained INT8 policy and board tooling into one directory.
Preparation and all 100 reference outputs/40 layer dumps passed from a fresh clone.

**Current blocker:** the board overlay (`policy.bit` + `policy.hwh`) still
needs to be built. Preparing this handoff does not load the FPGA or move the arm.

## 1. Pull and prepare

On Ubuntu, with Git and Python 3.8+ installed:

```sh
git clone --branch board/81-fpga-handoff https://github.com/psamin/robotfpga.git robotfpga-handoff
cd robotfpga-handoff
python3 board/demo/prepare_handoff.py
```

Expected: **Prepared 262 files** in `build/fpga-handoff`.
If that directory already exists, choose a fresh path with `--out NEW_DIRECTORY`.

## 2. Check the model

Use a Python environment with NumPy installed:

```sh
cd build/fpga-handoff
python3 board/demo/handoff_check.py artifacts/standin --reference ref/intref.py --verify-reference
```

Expected: **PASS**, `reference_calls: 100`, `layer_dumps: 40`.
The directory named `standin` contains the **trained v3r2 policy**.

## 3. Build the overlay, then run on the connected board

Follow [the detailed runbook](board/demo/README.md), sections 2–4.
It separates the Vivado workstation steps from the KR260 commands and lists
the required overlay files, IP names, DMA settings and success checks.
Do not skip M0/M1 board acceptance. [Review PR #82](https://github.com/psamin/robotfpga/pull/82).

## FPGA team status

Owners, verification evidence and open review links: [board/status.md](board/status.md).
