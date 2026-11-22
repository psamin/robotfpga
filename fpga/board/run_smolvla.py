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
the once-per-chunk cross-attention k/v and the FP32 layers on the ARM (runtime.fpga_linears). --lang-trim
(the real language tokens only: prefix 81 -> 45) and --skip-dead-layer (LM layer 15's unread q/o/MLP and the
final norm) are the design study's exact levers for this per-call path (runtime.trim_language /
skip_dead_lm_layer; both off by default).

--path list: the whole chunk as one command list (fpga/board/cmdlist.py, fpga/kernels/CONTRACT.md): weights
packed for the list at load, one IP start per chunk, two arenas; the language trim and the layer-15 skip are
part of the list. On the board it needs the list-mode float_top already loaded (load_overlay.py); without a
board, --mock emu walks the same list with the kernels' bit-exact emulators (fe kernels, the contract's T-P1
golden executor) behind the same registers:
    python fpga/board/run_smolvla.py --path list --mock emu --model <ckpt> --batches <dir> [--inflight 2]
        [--arm-fp32] [--list-fe torch] [--list-exact]
Per frame it prints the error, the time per chunk, the FPGA (emulator) time and the host time outside the
FPGA (prep, inputs, read, post, ARM layers); --inflight K re-runs the frames K at a time and checks that every
chunk equals its one-at-a-time chunk bit for bit.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fpga.board.runtime import (  # noqa: E402
    Engine,
    fpga_linears,
    memo_hits,
    skip_dead_lm_layer,
    trim_language,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--batches", required=True)
    ap.add_argument("--bit", help="overlay .bit (board)")
    ap.add_argument(
        "--mock",
        choices=["shim", "layout", "ip", "emu"],
        help="no board: per-call path shim / layout / ip; list path emu (the kernels' emulators)",
    )
    ap.add_argument(
        "--path", choices=["call", "list"], default="call", help="per-call engine or command list"
    )
    ap.add_argument(
        "--lang-trim", action="store_true", help="per-call path: real language tokens only (exact)"
    )
    ap.add_argument(
        "--skip-dead-layer", action="store_true", help="per-call path: skip LM layer 15's unread work (exact)"
    )
    ap.add_argument("--arm-fp32", action="store_true", help="list path: the 21 FP32-stored layers on the ARM")
    ap.add_argument("--arenas", type=int, default=2, help="list path: arenas (inferences in flight)")
    ap.add_argument("--inflight", type=int, default=0, help="list path: re-run the frames K at a time")
    ap.add_argument("--list-fe", choices=["kernels", "torch"], default="kernels", help="list mock front ends")
    ap.add_argument("--list-exact", action="store_true", help="list mock: no quantization (wiring check)")
    ap.add_argument("--nsb", type=int, default=10, help="list mock: gelu_silu table size")
    ap.add_argument("--low-ddr", action="store_true", help="list path: refuse buffers above 0x8000_0000")
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
    if args.path == "list":
        return run_list(args)
    if args.mock == "emu":
        sys.exit("--mock emu is the list path's mock (--path list)")
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
    skipped = skip_dead_lm_layer(policy) if args.skip_dead_layer else []
    if args.lang_trim:
        trim_language(policy)
    info = {"lang_trim": args.lang_trim, "skipped_dead": skipped}
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
        "lang_trim": args.lang_trim,
        "skip_dead_layer": args.skip_dead_layer,
        "max_abs_diff_normalized": max(r["max_abs_diff_normalized"] for r in rows),
        "mean_seconds_per_chunk": sum(r["seconds_per_chunk"] for r in rows) / len(rows),
    }
    print(json.dumps(summary))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps({"summary": summary, "rows": rows, "info": info}, indent=1))


def load_frames(args):
    import torch

    files = sorted(Path(args.batches).glob("frame_*.pt"))
    if args.frames:
        files = [files[int(i)] for i in args.frames.split(",")]
    if not files:
        sys.exit(f"no frame_*.pt in {args.batches}")
    return [torch.load(f, weights_only=False) for f in files]


def run_list(args):
    """--path list: fpga/board/cmdlist.py on the board (xrt) or the emulator mock."""
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    from fpga.board.cmdlist import ListRuntime, check_inflight, install, replay

    if args.mock not in (None, "emu"):
        sys.exit("--path list takes --mock emu (or no --mock: the board)")
    if args.bit:
        sys.exit("--path list: load the list-mode bitstream first (load_overlay.py), then run without --bit")
    if args.lang_trim or args.skip_dead_layer or args.no_fuse or args.no_memo or args.no_keep_fp32:
        sys.exit(
            "--path list: the language trim and the layer-15 skip are built into the list; use --arm-fp32"
        )
    t0 = time.perf_counter()
    policy = SmolVLAPolicy.from_pretrained(str(Path(args.model).expanduser())).to("cpu").eval()
    rt = ListRuntime(
        policy,
        backend="mock" if args.mock else "xrt",
        arenas=args.arenas,
        fe=args.list_fe,
        exact=args.list_exact,
        nsb=args.nsb,
        arm_fp32=args.arm_fp32,
        low_ddr=args.low_ddr,
        clock_mhz=args.clock_mhz,
    )
    install(policy, rt)
    print(f"list path ready in {time.perf_counter() - t0:.1f} s", flush=True)
    frames = load_frames(args)
    rows = replay(policy, frames)
    summary = {
        "path": "list",
        "backend": rt.backend,
        "fe": "exact" if args.list_exact else args.list_fe,
        "arm_fp32": args.arm_fp32,
        "arenas": args.arenas,
        "frames": len(rows),
        "records": rt.n_records,
        "max_abs_diff_normalized": max(r["max_abs_diff_normalized"] for r in rows),
        "mean_seconds_per_chunk": sum(r["seconds_per_chunk"] for r in rows) / len(rows),
        "mean_host_outside_fpga_ms": sum(r["host_outside_fpga_ms"] for r in rows) / len(rows),
        "memory": rt.bytes,
    }
    if args.inflight > 1:
        summary["inflight_check"] = check_inflight(policy, frames, [r["_chunk"] for r in rows], args.inflight)
    print(json.dumps(summary))
    if args.save_chunks:
        import torch

        Path(args.save_chunks).mkdir(parents=True, exist_ok=True)
        for r in rows:
            torch.save(r["_chunk"].detach().clone(), Path(args.save_chunks) / f"{r['frame']:05d}.pt")
    if args.json_out:
        out = [{k: v for k, v in r.items() if k != "_chunk"} for r in rows]
        Path(args.json_out).write_text(json.dumps({"summary": summary, "rows": out}, indent=1))
    rt.close()
    ok = summary["max_abs_diff_normalized"] < 0.05 and summary.get("inflight_check", {"ok": True})["ok"]
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
