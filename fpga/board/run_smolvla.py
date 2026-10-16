"""Run the fine-tuned SmolVLA on the KR260 with every linear layer on the FPGA.

Replays preprocessed real frames saved on the cluster by
`policy/smolvla_fpga.py --dump-batches` (inputs, noise and the PyTorch FP32 reference chunk), so the
board needs the model and LeRobot but not the dataset or its preprocessing.

    # on the board, from the repo root
    python3 fpga/board/run_smolvla.py --bit float.bit --model <checkpoint dir> --batches <dir of frame_*.pt>
    # anywhere, without a board (same code path, host C core):
    python3 fpga/board/run_smolvla.py --mock layout --model <checkpoint dir> --batches <dir>

Prints, per frame, the max difference from the reference chunk (normalized action units) and the wall
time per 50-action chunk, split into FPGA calls and everything else. The arm is not touched.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fpga.board.runtime import Engine, fpga_linears  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--batches", required=True)
    ap.add_argument("--bit", help="overlay .bit (board)")
    ap.add_argument("--mock", choices=["shim", "layout"])
    ap.add_argument("--threads", type=int, default=4, help="ARM threads for the non-linear ops")
    ap.add_argument("--clock-mhz", type=float, default=None, help="PL clock to set after loading the overlay")
    args = ap.parse_args()

    import torch
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    torch.set_num_threads(args.threads)
    eng = Engine(args.bit, clock_mhz=args.clock_mhz) if args.bit else Engine(mock=args.mock or "layout")
    t0 = time.perf_counter()
    # Keep the checkpoint's BF16 while packing: .float() first would hold a 1.8 GB FP32 copy next to
    # the 0.71 GB DDR copy on a 4 GB board. Each linear is widened to FP32 only for its own packing and
    # then freed; the ~0.4 GB that stays in PyTorch (norms, embeddings, patch conv) is widened after.
    policy = SmolVLAPolicy.from_pretrained(str(Path(args.model).expanduser())).to("cpu").eval()
    n = fpga_linears(policy, eng)
    policy = policy.float()
    print(f"loaded and packed {n} linear layers in {time.perf_counter() - t0:.1f} s", flush=True)

    rows = []
    for f in sorted(Path(args.batches).glob("frame_*.pt")):
        d = torch.load(f, weights_only=False)
        eng.calls, eng.fpga_s = 0, 0.0
        policy.reset()
        t0 = time.perf_counter()
        with torch.no_grad():
            chunk = policy.predict_action_chunk(d["batch"], noise=d["noise"])
        wall = time.perf_counter() - t0
        diff = float((chunk.float() - d["ref_chunk_raw"].float()).abs().max())
        rows.append(
            {
                "frame": d["frame"],
                "max_abs_diff_normalized": diff,
                "seconds_per_chunk": round(wall, 3),
                "fpga_seconds": round(eng.fpga_s, 3),
                "fpga_calls": eng.calls,
            }
        )
        print(json.dumps(rows[-1]), flush=True)
    if not rows:
        sys.exit(f"no frame_*.pt in {args.batches}")
    print(
        json.dumps(
            {
                "frames": len(rows),
                "max_abs_diff_normalized": max(r["max_abs_diff_normalized"] for r in rows),
                "mean_seconds_per_chunk": sum(r["seconds_per_chunk"] for r in rows) / len(rows),
            }
        )
    )


if __name__ == "__main__":
    main()
