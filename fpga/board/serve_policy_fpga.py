"""LeRobot policy server on the KR260 with every linear layer on the FPGA.

The laptop runs the arm and cameras exactly as with the cluster (robot/toolkit/policy.sh, BOARD=...); this
replaces the cluster's policy server. After LeRobot loads the policy, every nn.Linear is swapped for
runtime.FpgaLinear: its BF16 weights go to DDR through engine_server.py (PYNQ's Python) and each call runs
on the FPGA. The rest of SmolVLA (attention matmuls, norms, convolution, ...) runs in FP32 on the ARM.
Listens on 127.0.0.1 only: LeRobot's gRPC transport uses pickle, so reach it through an SSH tunnel.

    # board, terminal 1 (PYNQ's Python):
    sudo -E /usr/local/share/pynq-venv/bin/python3 fpga/board/engine_server.py --bit float.bit
    # board, terminal 2 (LeRobot venv, Python 3.12):
    python fpga/board/serve_policy_fpga.py --port 8080
    # check first, without the arm: load the checkpoint as the server does, replay the release's batches
    python fpga/board/serve_policy_fpga.py --selftest <checkpoint dir> --batches <dir of frame_*.pt>
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fpga.board.runtime import RemoteEngine, fpga_linears  # noqa: E402


def to_fpga(policy, eng):
    """Swap the linears (packing from the stored BF16, then freeing them), then FP32 for the rest."""
    t0 = time.perf_counter()
    n = fpga_linears(policy, eng)
    policy.float()
    left = sum(p.numel() for p in policy.parameters())
    print(
        f"{n} linear layers on the FPGA in {time.perf_counter() - t0:.1f} s; {left:,} params left on the ARM",
        flush=True,
    )
    return policy


def selftest(args, eng):
    import torch
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    torch.set_num_threads(args.threads)
    policy = to_fpga(SmolVLAPolicy.from_pretrained(args.selftest).to("cpu").eval(), eng)
    worst = 0.0
    files = sorted(Path(args.batches).glob("frame_*.pt"))
    if not files:
        sys.exit(f"no frame_*.pt in {args.batches}")
    for f in files:
        d = torch.load(f, weights_only=False)
        policy.reset()
        eng.calls, eng.fpga_s = 0, 0.0
        t0 = time.perf_counter()
        with torch.no_grad():
            chunk = policy.predict_action_chunk(d["batch"], noise=d["noise"])
        wall = time.perf_counter() - t0
        diff = float((chunk.float() - d["ref_chunk_raw"].float()).abs().max())
        worst = max(worst, diff)
        print(
            json.dumps(
                {
                    "frame": d["frame"],
                    "max_abs_diff_normalized": diff,
                    "seconds_per_chunk": round(wall, 2),
                    "fpga_and_transfer_s": round(eng.fpga_s, 2),
                    "engine_calls": eng.calls,
                }
            ),
            flush=True,
        )
    # FP32 rounding on the FPGA path vs PyTorch: the cluster host emulation measured 0.0043 (job 10083)
    ok = worst < 0.05
    print(
        f"{'PASS' if ok else 'FAIL'}: max |chunk - PyTorch FP32 reference| = {worst:.4f} (normalized units)"
    )
    sys.exit(0 if ok else 1)


def serve(args, eng):
    from lerobot.async_inference import policy_server as ps

    original = ps.PolicyServer.SendPolicyInstructions

    def instructions_on_fpga(self, request, context):
        import pickle
        import types

        specs = pickle.loads(request.data)  # nosec  (localhost-only server, same trust as LeRobot's)
        specs.device = "cpu"  # the board has no GPU, whatever the client asked for
        out = original(self, types.SimpleNamespace(data=pickle.dumps(specs)), context)
        to_fpga(self.policy, eng)
        return out

    ps.PolicyServer.SendPolicyInstructions = instructions_on_fpga
    sys.argv = [sys.argv[0], f"--host={args.host}", f"--port={args.port}", f"--fps={args.fps}"]
    print(f"SERVING with FPGA linears on {args.host}:{args.port} (localhost only)", flush=True)
    ps.serve()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--socket", default="/tmp/fpga-engine.sock", help="engine_server.py socket")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--selftest", metavar="CHECKPOINT", help="replay --batches instead of serving")
    ap.add_argument("--batches", help="dir of frame_*.pt (release asset smolvla-9604-replay-batches)")
    ap.add_argument("--threads", type=int, default=4, help="ARM threads for the non-linear ops")
    args = ap.parse_args()
    if args.host not in ("127.0.0.1", "localhost"):
        sys.exit(
            "refusing to listen beyond localhost: the gRPC transport unpickles requests; use an SSH tunnel"
        )
    eng = RemoteEngine(args.socket)
    selftest(args, eng) if args.selftest else serve(args, eng)


if __name__ == "__main__":
    main()
