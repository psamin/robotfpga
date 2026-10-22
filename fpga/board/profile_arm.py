"""Where does a chunk spend its ARM time? torch.profiler over one predict_action_chunk on the board.

    HF_HOME=... HF_HUB_OFFLINE=1 python fpga/board/profile_arm.py --ckpt <checkpoint> --batches <dir>

Runs like serve_policy_fpga.py --selftest (engine over the Unix socket), with every engine call wrapped
in a record_function("fpga_gemm") so the table separates FPGA + transfer time from the ARM ops.
"""

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fpga.board.runtime import Engine, RemoteEngine, fast_tanh_gelu, fpga_linears  # noqa: E402

GROUPS = {  # op name substring -> bucket
    "fpga_gemm": "fpga_gemm (engine + socket)",
    "bmm": "attention matmul",
    "baddbmm": "attention matmul",
    "matmul": "attention matmul",
    "mm": "attention matmul",
    "softmax": "softmax",
    "gelu": "gelu",
    "layer_norm": "layer_norm",
    "rms_norm": "rms_norm",
    "conv": "patch conv",
}


def bucket(name: str) -> str:
    for key, b in GROUPS.items():
        if key in name:
            return b
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--batches", required=True)
    ap.add_argument("--socket", default="/tmp/fpga-engine.sock")
    ap.add_argument("--mock", action="store_true", help="host C core instead of the board (laptop)")
    ap.add_argument("--engine", choices=["xrt", "socket"], default="xrt")
    ap.add_argument("--no-fast-gelu", action="store_true")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--frames", type=int, default=1)
    ap.add_argument("--rows", type=int, default=45)
    ap.add_argument(
        "--stack", action="store_true", help="also attribute copies/clones to their Python callers"
    )
    args = ap.parse_args()

    import torch
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from torch.profiler import ProfilerActivity, profile, record_function

    torch.set_num_threads(args.threads)
    if args.mock:
        eng = Engine(mock="layout")
    elif args.engine == "xrt":
        eng = Engine(backend="xrt")
    else:
        eng = RemoteEngine(args.socket)
    gemm = eng.gemm

    def timed_gemm(*a, **kw):
        with record_function("fpga_gemm"):
            return gemm(*a, **kw)

    eng.gemm = timed_gemm
    policy = SmolVLAPolicy.from_pretrained(args.ckpt).to("cpu").eval()
    n = fpga_linears(policy, eng)
    policy.float()
    g = 0 if args.no_fast_gelu else fast_tanh_gelu(policy)
    print(f"{n} linear layers on the engine, {g} GELUs on the sigmoid form", flush=True)
    files = sorted(Path(args.batches).glob("frame_*.pt"))[: args.frames]
    for f in files:
        d = torch.load(f, weights_only=False)
        policy.reset()
        t0 = time.perf_counter()
        with torch.no_grad(), profile(activities=[ProfilerActivity.CPU], with_stack=args.stack) as prof:
            chunk = policy.predict_action_chunk(d["batch"], noise=d["noise"])
        wall = time.perf_counter() - t0
        diff = float((chunk.float() - d["ref_chunk_raw"].float()).abs().max())
        print(f"frame {d['frame']}: {wall:.1f} s wall, max diff {diff:.4f}", flush=True)
        ka = prof.key_averages()
        buckets, calls = defaultdict(float), defaultdict(int)
        for ev in ka:
            b = bucket(ev.key)
            buckets[b] += ev.self_cpu_time_total / 1e6
            calls[b] += ev.count
        print("\nself CPU seconds by bucket (profiled region):")
        for b, s in sorted(buckets.items(), key=lambda kv: -kv[1]):
            print(f"  {b:32s} {s:7.2f} s  ({calls[b]} calls)")
        print(ka.table(sort_by="self_cpu_time_total", row_limit=args.rows))
        if args.stack:
            print("\ncopies and clones by caller (self CPU s):")
            rows = [
                ev
                for ev in prof.key_averages(group_by_stack_n=6)
                if ("copy_" in ev.key or "clone" in ev.key or "contiguous" in ev.key)
                and ev.self_cpu_time_total > 5e4
            ]
            for ev in sorted(rows, key=lambda e: -e.self_cpu_time_total)[:12]:
                frames = [f for f in (ev.stack or []) if "site-packages/torch/" not in f][:4]
                print(
                    f"  {ev.self_cpu_time_total / 1e6:6.2f} s {ev.count:6d}x {ev.key:24s} <- "
                    + " | ".join(frames)
                )


if __name__ == "__main__":
    main()
