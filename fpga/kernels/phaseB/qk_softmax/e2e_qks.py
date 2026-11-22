"""End-to-end check of the qk_softmax fusion on SmolVLA -12392: the integrator's lockstep harness
(fpga/hls/e2e_seq.py, snapshot) run on the e2e tree build/e2e (e2e_build.sh: this fusion's integration/ files on
the snapshot), so the list builder, the golden executor and the C model are the patched ones of INTEGRATION.md.

    --fuse (default)  the list with every (GEMM qk, SOFTMAX) pair as one QK_SOFTMAX record (2,514 records); the C
                      model is seq_shim_qks.dll (the fused GEMM unit, no SOFTMAX unit); the golden executor runs
                      QK_SOFTMAX as exec_ref.op_qk_softmax (GEMM epilogue + smx_emu, scores never in memory)
    --control         the contract 1.0 list (2,701 records) on seq_shim_ctl.dll (same tree, SOFTMAX unit on)

After every command the C model's image must equal the golden memory byte for byte over every written region;
the C model's x_t then goes through the policy and is compared with ref_chunk_raw (e2e_seq.py's report).

    set HF_HUB_OFFLINE=1
    C:\\kp\\lerobot-venv\\Scripts\\python.exe fpga\\kernels\\phaseB\\qk_softmax\\e2e_qks.py [--control]
        [--frames 0,3563,7489] [--out results/e2e_fused.json]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
E = HERE / "build" / "e2e"
CKPT = ROOT / "build/ckpt/smolvla-12392/smolvla-so101_medicine_bottle_pickplace-12392-native"
BATCHES = ROOT / "build/ckpt/smolvla-12392/batches-12392"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", action="store_true")
    ap.add_argument("--frames", default="0,3563,7489")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    for p in (E / "hls", E / "kernels" / "sequencer", E / "kernels"):
        sys.path.insert(0, str(p))
    import contract  # the e2e tree's (integration) contract.py: parses its contract.h

    assert contract.__file__.startswith(str(E)), contract.__file__
    assert hasattr(contract.C, "OP_QK_SOFTMAX"), "e2e tree without the fusion: run e2e_build.sh"
    import contract_plan as CP

    assert CP.__file__.startswith(str(E)), CP.__file__
    CP.OPT["fuse_qk"] = not a.control
    import e2e_seq
    import torch

    assert e2e_seq.__file__.startswith(str(E)), e2e_seq.__file__
    xts: list = []  # the C model's x_t of every frame, saved next to the json for a bitwise compare across runs
    run0 = e2e_seq.LockstepProgram.run

    def run(self, *args, **kw):
        xt = run0(self, *args, **kw)
        xts.append(xt.clone())
        return xt

    e2e_seq.LockstepProgram.run = run
    dll = E / ("seq_shim_ctl.dll" if a.control else "seq_shim_qks.dll")
    out = a.out or str(HERE / "results" / ("e2e_control.json" if a.control else "e2e_fused.json"))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    print(f"e2e: {'control (contract 1.0 list)' if a.control else 'fused list (QK_SOFTMAX)'}, {dll.name}", flush=True)
    sys.argv = [
        "e2e_seq.py",
        str(CKPT),
        str(BATCHES),
        "--dll",
        str(dll),
        "--frames",
        a.frames,
        "--threads",
        str(a.threads),
        "--out",
        out,
    ]
    rc = e2e_seq.main()
    torch.save(xts, str(Path(out).with_suffix(".xt.pt")))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
