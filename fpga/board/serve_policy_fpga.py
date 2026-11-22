"""LeRobot policy server on the KR260 with the linear layers on the FPGA.

The laptop runs the arm and cameras exactly as with the cluster (robot/toolkit/policy.sh, BOARD=...); this
replaces the cluster's policy server. After LeRobot loads the policy, every nn.Linear is swapped for
runtime.FpgaLinear: its weights go to DDR and each call runs on the FPGA, driven from this process
(--engine xrt, default) or through engine_server.py (--engine socket, FP32 only). --arith int16 drives the
int16 bitstream (float_tp?_int16.bit: int16 weights per output row, activations quantized per token on the
ARM, FP32 results). The rest of SmolVLA (attention, norms, convolution, ...) runs in FP32 on the ARM;
SigLIP's tanh-GELU uses its sigmoid form (same function, 3x faster on this CPU; --no-fast-gelu keeps
torch's kernel). The 21 layers the checkpoint stores in FP32 (state/action projections, action-time MLP,
the expert's cross-attention k/v) stay FP32 on the ARM (--no-keep-fp32: on the engine too, the board's
0.0043). Linear layers that take the same input (q/k/v, gate/up) share one engine call and the expert's
cross-attention k/v projections of the fixed prefix cache are computed once per chunk (runtime.fpga_linears
fuse/memo: same numbers; --no-fuse / --no-memo switch them off).
Listens on 127.0.0.1 only: LeRobot's gRPC transport uses pickle, so reach it through an SSH tunnel.

    # board, once (PYNQ's Python): load the bitstream
    sudo /usr/local/share/pynq-venv/bin/python3 fpga/board/load_overlay.py --bit float.bit --clock-mhz 250
    # board (LeRobot venv, Python 3.12, root: the engine maps CMA buffers and the IP registers itself)
    sudo env HF_HOME=... HF_HUB_OFFLINE=1 <venv>/bin/python fpga/board/serve_policy_fpga.py --port 8080
    # check first, without the arm: load the checkpoint as the server does, replay the release's batches
    python fpga/board/serve_policy_fpga.py --selftest <checkpoint dir> --batches <dir of frame_*.pt>
    # the same without a board (host C core; --mock ip runs the board's buffer/register code path too)
    python fpga/board/serve_policy_fpga.py --mock layout [--arith int16] --selftest <ckpt dir> --batches <dir>

--path list: the whole chunk as one command list (fpga/board/cmdlist.py, fpga/kernels/CONTRACT.md): every
weight packed at load (the 21 FP32-stored layers on the engine too; --arm-fp32 puts them back on the ARM),
one IP start per chunk, the language trim (12 real tokens) and LM layer 15's skip built into the list.
predict_action_chunk becomes re-entrant (each call its own tensors and arena, not the policy's shared
_queues); --inflight 2 lets LeRobot's server hold two observations and run two GetActions at once (two
arenas). Needs the list-mode float_top loaded (load_overlay.py). --selftest with --mock emu runs the same list
on the laptop with the kernels' bit-exact emulators and checks two-in-flight chunks against one-at-a-time
chunks bit for bit:
    python fpga/board/serve_policy_fpga.py --path list --mock emu --selftest <ckpt dir> --batches <dir>
--lang-trim / --skip-dead-layer: the same two exact levers for the per-call path (off by default).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fpga.board.runtime import (  # noqa: E402
    Engine,
    RemoteEngine,
    fast_tanh_gelu,
    fpga_linears,
    overlap_image_embedding,
    skip_dead_lm_layer,
    trim_language,
)

FAST_GELU = True
OVERLAP = True
FUSE = True  # q/k/v and gate/up of one input: one engine call (runtime.fusion_groups)
MEMO = True  # expert cross-attention k/v of the prefix cache: computed once per chunk (runtime.memo_modules)
KEEP_FP32 = True  # the checkpoint's FP32 layers stay FP32 on the ARM (runtime.fp32_stored_linears)
LANG_TRIM = False  # per-call path: real language tokens only (runtime.trim_language; exact)
SKIP_DEAD = False  # per-call path: LM layer 15's unread work skipped (runtime.skip_dead_lm_layer; bit-exact)
LIST = None  # --path list: the cmdlist.ListRuntime keyword arguments
_RT = None  # the current ListRuntime (one per loaded policy)


def checkpoint_file(name_or_path: str):
    """The checkpoint directory, or its model.safetensors from the local Hugging Face cache (offline)."""
    if os.path.isdir(name_or_path):
        return Path(name_or_path)
    try:
        from huggingface_hub import hf_hub_download

        return Path(hf_hub_download(name_or_path, "model.safetensors", local_files_only=True))
    except Exception as e:  # noqa: BLE001 - fpga_linears warns and keeps nothing
        print(f"checkpoint {name_or_path!r} not found locally ({type(e).__name__})", flush=True)
        return None


def to_fpga(policy, eng, checkpoint):
    """Swap the linears (packing from the stored BF16, then freeing them), then FP32 for the rest."""
    if LIST is not None:
        return to_list(policy)
    t0 = time.perf_counter()
    skipped = skip_dead_lm_layer(policy) if SKIP_DEAD else []
    if LANG_TRIM:
        trim_language(policy)
    info = {}
    n = fpga_linears(policy, eng, fuse=FUSE, memo=MEMO, keep_fp32=KEEP_FP32, checkpoint=checkpoint, info=info)
    policy.float()
    g = fast_tanh_gelu(policy) if FAST_GELU else 0
    if OVERLAP:
        overlap_image_embedding(policy, eng)
    left = sum(p.numel() for p in policy.parameters())
    print(
        f"{n} linear layers on the FPGA ({eng.arith}, {info['engine_layers']} engine layers) in"
        f" {time.perf_counter() - t0:.1f} s; {len(info['kept_fp32'])} FP32 layers on the ARM; {left:,} params"
        f" left on the ARM; {g} tanh-GELU modules on the sigmoid form; images"
        f" {'overlapped' if OVERLAP else 'sequential'}; fused calls {'on' if FUSE else 'off'}, k/v memo"
        f" {'on' if MEMO else 'off'}; language trim {'on' if LANG_TRIM else 'off'}; layer-15 dead work"
        f" {'skipped' if skipped else 'kept'}",
        flush=True,
    )
    return policy


def to_list(policy):
    """--path list: pack, place and validate the chunk's command list; predict_action_chunk runs it."""
    from fpga.board.cmdlist import ListRuntime, install

    global _RT
    if _RT is not None:  # a client reconnected: LeRobot loaded the policy again; free the old buffers first
        _RT.close()
        _RT = None
    t0 = time.perf_counter()
    rt = _RT = ListRuntime(policy, **LIST)
    install(policy, rt)
    print(f"command list ready in {time.perf_counter() - t0:.1f} s", flush=True)
    return policy


def selftest_list(args):
    import torch
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    from fpga.board.cmdlist import check_inflight, replay

    torch.set_num_threads(args.threads)
    policy = to_fpga(SmolVLAPolicy.from_pretrained(args.selftest).to("cpu").eval(), None, None)
    files = sorted(Path(args.batches).glob("frame_*.pt"))
    if not files:
        sys.exit(f"no frame_*.pt in {args.batches}")
    frames = [torch.load(f, weights_only=False) for f in files]
    rows = replay(policy, frames)
    worst = max(r["max_abs_diff_normalized"] for r in rows)
    k = max(2, args.inflight)
    inf = check_inflight(policy, frames, [r["_chunk"] for r in rows], k)
    print(json.dumps({"inflight_check": inf}), flush=True)
    ok = worst < 0.05 and inf["ok"]
    print(
        f"{'PASS' if ok else 'FAIL'}: max |chunk - PyTorch FP32 reference| = {worst:.4g} (normalized units);"
        f" {k} in flight bit-equal to one at a time: {inf['ok']}"
    )
    policy._fpga_list_runtime.close()
    sys.exit(0 if ok else 1)


def selftest(args, eng):
    import torch
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    torch.set_num_threads(args.threads)
    policy = to_fpga(
        SmolVLAPolicy.from_pretrained(args.selftest).to("cpu").eval(), eng, checkpoint_file(args.selftest)
    )
    worst = 0.0
    files = sorted(Path(args.batches).glob("frame_*.pt"))
    if not files:
        sys.exit(f"no frame_*.pt in {args.batches}")
    for f in files:
        d = torch.load(f, weights_only=False)
        policy.reset()
        eng.calls, eng.runs, eng.fpga_s = 0, 0, 0.0
        eng.t = dict.fromkeys(getattr(eng, "t", {}), 0.0)
        t0 = time.perf_counter()
        with torch.no_grad():
            chunk = policy.predict_action_chunk(d["batch"], noise=d["noise"])
        wall = time.perf_counter() - t0
        diff = float((chunk.float() - d["ref_chunk_raw"].float()).abs().max())
        worst = max(worst, diff)
        row = {
            "frame": d["frame"],
            "arith": eng.arith,
            "max_abs_diff_normalized": diff,
            "seconds_per_chunk": round(wall, 2),
            "fpga_and_transfer_s": round(eng.fpga_s, 2),
            "engine_calls": eng.calls,
            "engine_runs": eng.runs,
        }
        if eng.t:
            row["arm_s"] = {k: round(v, 3) for k, v in eng.t.items()}
        print(json.dumps(row), flush=True)
    # FP32 engine, all linears on it: 0.0043 on -9604 frame 0 (the board, job 10083); with the FP32 layers
    # on the ARM 3e-6 (laptop C model); int16 engine: 1.2e-3 on -12392 frame 0 (laptop C model)
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
        to_fpga(self.policy, eng, checkpoint_file(specs.pretrained_name_or_path))
        return out

    ps.PolicyServer.SendPolicyInstructions = instructions_on_fpga
    if args.inflight > 1:
        if LIST is None:
            sys.exit("--inflight > 1 needs --path list (the per-call path shares the policy's queues)")
        from fpga.board.cmdlist import patch_policy_server

        patch_policy_server(ps, args.inflight)
    sys.argv = [sys.argv[0], f"--host={args.host}", f"--port={args.port}", f"--fps={args.fps}"]
    what = (
        f"the command list ({args.inflight} in flight)" if LIST is not None else f"FPGA linears ({eng.arith})"
    )
    print(f"SERVING with {what} on {args.host}:{args.port} (localhost only)", flush=True)
    ps.serve()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--engine",
        choices=["xrt", "socket"],
        default="xrt",
        help="xrt: drive the FPGA from this process (root); socket: through engine_server.py (FP32 only)",
    )
    ap.add_argument(
        "--arith",
        choices=["fp32", "int16"],
        default="fp32",
        help="the loaded bitstream's arithmetic (int16: float_tp?_int16.bit); checked by a probe call",
    )
    ap.add_argument("--socket", default="/tmp/fpga-engine.sock", help="engine_server.py socket")
    ap.add_argument("--no-fast-gelu", action="store_true", help="keep torch's F.gelu(tanh) kernel")
    ap.add_argument("--no-overlap", action="store_true", help="encode the camera images one after the other")
    ap.add_argument(
        "--no-fuse", action="store_true", help="one engine call per nn.Linear (no q/k/v, gate/up fusion)"
    )
    ap.add_argument("--no-memo", action="store_true", help="recompute the expert's prefix k/v at every step")
    ap.add_argument(
        "--no-keep-fp32", action="store_true", help="the checkpoint's FP32 layers on the engine too"
    )
    ap.add_argument(
        "--mock",
        choices=["layout", "ip", "emu"],
        help="selftest without the board: per-call host C core (layout, ip) or the list's emulators (emu)",
    )
    ap.add_argument(
        "--path", choices=["call", "list"], default="call", help="per-call engine or command list"
    )
    ap.add_argument("--inflight", type=int, default=1, help="list path: inferences in flight (server queue)")
    ap.add_argument("--arenas", type=int, default=2, help="list path: arenas (>= --inflight)")
    ap.add_argument("--arm-fp32", action="store_true", help="list path: the 21 FP32-stored layers on the ARM")
    ap.add_argument("--list-fe", choices=["kernels", "torch"], default="kernels", help="list mock front ends")
    ap.add_argument("--low-ddr", action="store_true", help="list path: refuse buffers above 0x8000_0000")
    ap.add_argument("--clock-mhz", type=float, default=None, help="list path: PL clock to set (board)")
    ap.add_argument(
        "--lang-trim", action="store_true", help="per-call path: real language tokens only (exact)"
    )
    ap.add_argument(
        "--skip-dead-layer", action="store_true", help="per-call path: skip LM layer 15's unread work (exact)"
    )
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
    global FAST_GELU, OVERLAP, FUSE, MEMO, KEEP_FP32, LANG_TRIM, SKIP_DEAD, LIST
    FAST_GELU = not args.no_fast_gelu
    OVERLAP = not args.no_overlap
    FUSE, MEMO, KEEP_FP32 = not args.no_fuse, not args.no_memo, not args.no_keep_fp32
    LANG_TRIM, SKIP_DEAD = args.lang_trim, args.skip_dead_layer
    if args.path == "list":
        if args.mock not in (None, "emu") or args.engine != "xrt" or args.arith != "fp32":
            sys.exit("--path list: --mock emu or the board (--engine xrt); its arithmetic is int16 by build")
        if args.lang_trim or args.skip_dead_layer:
            sys.exit("--path list: the language trim and the layer-15 skip are built into the list")
        if args.arenas < args.inflight:
            sys.exit("--arenas must be >= --inflight")
        if args.mock and not args.selftest:
            sys.exit("--mock is for --selftest")
        LIST = dict(
            backend="mock" if args.mock else "xrt",
            arenas=args.arenas,
            fe=args.list_fe,
            arm_fp32=args.arm_fp32,
            low_ddr=args.low_ddr,
            clock_mhz=args.clock_mhz,
        )
        return selftest_list(args) if args.selftest else serve(args, None)
    if args.mock == "emu":
        sys.exit("--mock emu is the list path's mock (--path list)")
    if args.inflight > 1:
        sys.exit("--inflight > 1 needs --path list (the per-call path shares the policy's queues)")
    if args.mock:
        if not args.selftest:
            sys.exit("--mock is for --selftest")
        eng = Engine(mock=args.mock, arith=args.arith)
    elif args.engine == "xrt":
        eng = Engine(backend="xrt", arith=args.arith)
    else:
        if args.arith != "fp32":
            sys.exit("--engine socket is FP32 only (engine_server.py); use --engine xrt for --arith int16")
        eng = RemoteEngine(args.socket)
    selftest(args, eng) if args.selftest else serve(args, eng)


if __name__ == "__main__":
    main()
