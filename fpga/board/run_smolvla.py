"""Run the fine-tuned SmolVLA on the KR260 with its linear layers on the FPGA.

Replays preprocessed real frames saved on the cluster by
`policy/smolvla_fpga.py --dump-batches` (inputs, noise and the PyTorch FP32 reference chunk), so the
board needs the model and LeRobot but not the dataset or its preprocessing.

    # on the board, from the repo root
    python3 fpga/board/run_smolvla.py --bit float.bit --model <checkpoint dir> --batches <dir of frame_*.pt>
    python3 fpga/board/run_smolvla.py --bit float_tp4_int16.bit --arith int16 --model ... --batches ...
    # anywhere, without a board (same code path, host C core; --mock ip: the board's buffer/register path):
    python3 fpga/board/run_smolvla.py --mock layout [--arith int16] --model <checkpoint dir> --batches <dir>

Prints, per frame, the max difference from the reference chunk (normalized action units), the wall
time per 50-action chunk, split into FPGA calls and everything else, the engine calls (gemm calls and
engine starts, one per 4096-wide K chunk) and, in int16, the ARM time of the per-token quantization.
The arm is not touched. --no-fuse / --no-memo / --no-keep-fp32 switch off the q/k/v + gate/up fusion,
the once-per-chunk cross-attention k/v and the FP32 layers on the ARM (runtime.fpga_linears).
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fpga.board.runtime import Engine, fpga_linears, memo_hits  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--batches", required=True)
    ap.add_argument("--bit", help="overlay .bit (board)")
    ap.add_argument("--mock", choices=["shim", "layout", "ip"])
    ap.add_argument("--arith", choices=["fp32", "int16"], default="fp32", help="the bitstream's arithmetic")
    ap.add_argument("--no-fuse", action="store_true", help="one engine call per nn.Linear")
    ap.add_argument("--no-memo", action="store_true", help="recompute the expert's prefix k/v at every step")
    ap.add_argument(
        "--no-keep-fp32", action="store_true", help="the checkpoint's FP32 layers on the engine too"
    )
    ap.add_argument("--frames", help="comma-separated indices into the sorted frame_*.pt (default: all)")
    ap.add_argument("--save-chunks", help="directory: save each frame's chunk as <frame>.pt")
    ap.add_argument("--json-out", help="write the rows and the summary here")
    ap.add_argument("--threads", type=int, default=4, help="ARM threads for the non-linear ops")
    ap.add_argument("--clock-mhz", type=float, default=None, help="PL clock to set after loading the overlay")
    args = ap.parse_args()

    import torch
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    torch.set_num_threads(args.threads)
    if args.bit:
        eng = Engine(args.bit, clock_mhz=args.clock_mhz, arith=args.arith)
    else:
        eng = Engine(mock=args.mock or "layout", arith=args.arith)
    t0 = time.perf_counter()
    # Keep the checkpoint's BF16 while packing: .float() first would hold a 1.8 GB FP32 copy next to
    # the 0.71 GB DDR copy on a 4 GB board. Each linear is widened to FP32 only for its own packing and
    # then freed; the ~0.4 GB that stays in PyTorch (norms, embeddings, patch conv) is widened after. The
    # layers kept FP32 on the ARM come from model.safetensors' dtypes (runtime.fp32_stored_linears).
    model_dir = Path(args.model).expanduser()
    policy = SmolVLAPolicy.from_pretrained(str(model_dir)).to("cpu").eval()
    info = {}
    n = fpga_linears(
        policy,
        eng,
        fuse=not args.no_fuse,
        memo=not args.no_memo,
        keep_fp32=not args.no_keep_fp32,
        checkpoint=model_dir,
        info=info,
    )
    policy = policy.float()
    print(
        f"loaded and packed {n} linear layers ({info['engine_layers']} engine layers, {info['fused_groups']}"
        f" fused groups, {len(info['kept_fp32'])} kept FP32 on the ARM, arith {args.arith}) in"
        f" {time.perf_counter() - t0:.1f} s",
        flush=True,
    )
    if info["kept_fp32"]:
        print("kept FP32:", " ".join(info["kept_fp32"]), flush=True)

    files = sorted(Path(args.batches).glob("frame_*.pt"))
    if args.frames:
        files = [files[int(i)] for i in args.frames.split(",")]
    rows = []
    for f in files:
        d = torch.load(f, weights_only=False)
        eng.calls, eng.runs, eng.fpga_s = 0, 0, 0.0
        eng.t = dict.fromkeys(eng.t, 0.0)
        hits0 = memo_hits(policy)
        policy.reset()
        t0 = time.perf_counter()
        with torch.no_grad():
            chunk = policy.predict_action_chunk(d["batch"], noise=d["noise"])
        wall = time.perf_counter() - t0
        diff = float((chunk.float() - d["ref_chunk_raw"].float()).abs().max())
        row = {
            "frame": d["frame"],
            "max_abs_diff_normalized": diff,
            "seconds_per_chunk": round(wall, 3),
            "fpga_seconds": round(eng.fpga_s, 3),
            "fpga_calls": eng.calls,
            "engine_runs": eng.runs,
            "memo_hits": memo_hits(policy) - hits0,
        }
        if args.arith == "int16":
            row["quant_ms_per_call"] = round(1e3 * eng.t["quant"] / max(eng.calls, 1), 4)
            row["quant_seconds"] = round(eng.t["quant"], 3)
        rows.append(row)
        print(json.dumps(row), flush=True)
        if args.save_chunks:
            Path(args.save_chunks).mkdir(parents=True, exist_ok=True)
            torch.save(chunk.detach().clone(), Path(args.save_chunks) / f"{int(d['frame']):05d}.pt")
    if not rows:
        sys.exit(f"no frame_*.pt in {args.batches}")
    summary = {
        "frames": len(rows),
        "arith": args.arith,
        "fuse": not args.no_fuse,
        "memo": not args.no_memo,
        "keep_fp32": len(info["kept_fp32"]),
        "max_abs_diff_normalized": max(r["max_abs_diff_normalized"] for r in rows),
        "mean_seconds_per_chunk": sum(r["seconds_per_chunk"] for r in rows) / len(rows),
    }
    print(json.dumps(summary))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps({"summary": summary, "rows": rows, "info": info}, indent=1))


if __name__ == "__main__":
    main()
