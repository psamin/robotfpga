"""One command from a SmolVLA checkpoint to the model running on the KR260 FPGA engine.

    # laptop, repo root, LeRobot venv (needs torch, lerobot[smolvla], websockets):
    python fpga/board/deploy_smolvla.py --model <checkpoint dir or HF id> --bit fpga/hls/build/float_tp4.bit

Steps (each skippable with --skip-*):
  1. dump    policy/smolvla_fpga.py on the laptop: real frames from the dataset through the plain
             PyTorch FP32 model and the engine's C core; writes batches/frame_*.pt with inputs, noise
             and the reference action chunk (and checks the two agree).
  2. push    checkpoint, batches, bitstream (.bit + .hwh) and fpga/ code to the board's Jupyter
             folder <remote>/ (chunked uploads, any size).
  3. run     fpga/board/run_smolvla.py on the board as root through the Jupyter kernel: overlay,
             weights packed into DDR, every frame replayed; prints per-frame max difference vs the
             reference chunk and seconds per 50-action chunk.
Results are also saved to build/deploy/<name>/board_result.json.
"""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "fpga" / "board"))
import jb  # noqa: E402

CODE_FILES = [
    "fpga/board/runtime.py",
    "fpga/board/run_smolvla.py",
    "fpga/board/check_engine.py",
    "fpga/hls/pack_float.py",
]
CKPT_FILES = ["config.json", "model.safetensors", "policy_preprocessor.json", "policy_postprocessor.json"]


def resolve_model(model: str) -> Path:
    p = Path(model).expanduser()
    if p.is_dir():
        return p
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="checkpoint directory (pretrained_model) or HF repo id")
    ap.add_argument("--bit", required=True, help="float_*.bit from build_bd.tcl (.hwh beside it)")
    ap.add_argument("--frames", type=int, default=3)
    ap.add_argument("--name", default=None, help="build/deploy/<name>; default from the model name")
    ap.add_argument("--remote", default="smolvla", help="folder under the board's Jupyter root")
    ap.add_argument("--clock-mhz", type=float, default=None, help="PL clock on the board (default: as built)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--skip-dump", action="store_true")
    ap.add_argument("--skip-push", action="store_true")
    ap.add_argument("--skip-run", action="store_true")
    args = ap.parse_args()

    model = resolve_model(args.model)
    name = args.name or re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(args.model).name or "model")
    out = ROOT / "build" / "deploy" / name
    out.mkdir(parents=True, exist_ok=True)
    bit = Path(args.bit)
    hwh = bit.with_suffix(".hwh")
    need = [model / c for c in CKPT_FILES] + ([] if args.skip_push else [bit, hwh])
    for f in need:
        if not f.exists():
            sys.exit(f"missing {f}")
    extra = sorted(p for p in model.glob("policy_*processor*.safetensors"))

    if not args.skip_dump:
        print(f"== 1/3 dump {args.frames} frames through PyTorch and the engine C core (laptop)", flush=True)
        t0 = time.time()
        cmd = [sys.executable, str(ROOT / "policy" / "smolvla_fpga.py"), "--model", str(model)]
        cmd += ["--out", str(out), "--frames", str(args.frames), "--dump-batches"]
        r = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
        lines = [ln for ln in r.stdout.splitlines() if ln.startswith("{") or "loaded" in ln]
        print("\n".join(lines[-6:]))
        if r.returncode:
            print(r.stderr[-3000:])
            sys.exit("dump failed")
        print(f"   {time.time() - t0:.0f} s", flush=True)
    batches = sorted((out / "batches").glob("frame_*.pt"))
    if not batches:
        sys.exit(f"no batches in {out / 'batches'}")

    rem = args.remote.strip("/")
    if args.skip_push and args.skip_run:
        return
    jb.login()
    if not args.skip_push:
        print(f"== 2/3 push to the board: {rem}/", flush=True)
        t0 = time.time()
        for rel in CODE_FILES:
            jb.put(str(ROOT / rel), f"{rem}/{rel}")
        jb.put(str(bit), f"{rem}/{bit.name}")
        jb.put(str(hwh), f"{rem}/{hwh.name}")
        for b in batches:
            jb.put(str(b), f"{rem}/batches/{name}/{b.name}")
        for c in [model / c for c in CKPT_FILES] + extra:
            jb.put(str(c), f"{rem}/ckpt/{name}/{c.name}")
        print(f"   {time.time() - t0:.0f} s", flush=True)

    if args.skip_run:
        return
    print("== 3/3 run on the board", flush=True)
    jb.api("POST", f"kernels/{jb.kernel()}/restart")  # frees any overlay an earlier run holds
    time.sleep(3)
    root = jb.api("GET", "contents/")["path"] or ""
    folder = f"/root/jupyter_notebooks/{root}{rem}"
    clock = f"--clock-mhz {args.clock_mhz}" if args.clock_mhz else ""
    code = f"""
import os, subprocess, sys
os.chdir({folder!r})
cmd = [sys.executable, "-u", "fpga/board/run_smolvla.py", "--bit", {bit.name!r}, "--model", "ckpt/{name}",
       "--batches", "batches/{name}", "--threads", "{args.threads}"] + {clock!r}.split()
print(" ".join(cmd), flush=True)
p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
log = open("run_{name}.log", "w")
for line in p.stdout:
    print(line, end="", flush=True); log.write(line)
log.close()
print("EXIT", p.wait())
"""
    ok = jb.run(code, timeout=3600)
    log = jb.api("GET", f"contents/{rem}/run_{name}.log?type=file&format=text")["content"]
    rows = [json.loads(ln) for ln in log.splitlines() if ln.startswith("{")]
    (out / "board_result.json").write_text(json.dumps({"ok": ok, "rows": rows, "log": log}, indent=1))
    print(f"saved {out / 'board_result.json'}")
    if not ok or not rows:
        sys.exit("board run failed; see the log above")


if __name__ == "__main__":
    main()
