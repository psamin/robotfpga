"""The pick-and-place SmolVLA on the KR260 FPGA, from the laptop, through the board's Jupyter (root kernel).

    # LeRobot venv on the laptop (torch, lerobot[smolvla], websockets); repo root
    python fpga/board/deploy_smolvla.py --setup           # once: Python 3.12 + LeRobot on the board
    python fpga/board/deploy_smolvla.py --bit fpga/hls/build/float_tp4.bit \
        --model build/ckpt/smolvla-9604/smolvla-so101_medicine_bottle_pickplace-9604 \
        --batches build/ckpt/smolvla-9604/batches                    # push, engine check, selftest, serve

What it does on the board (docs/run-smolvla-on-kr260.md §1–3), everything under <remote>/ in the
Jupyter root (/root/jupyter_notebooks):
  push      fpga/ code, the bitstream (.bit + .hwh), the checkpoint, the replay batches, engine test cases
  engine    fpga/board/check_engine.py with the bitstream: every case PASS + weight bandwidth
  servers   engine_server.py (PYNQ's Python, root) in the background, socket handed to the policy user;
            serve_policy_fpga.py --selftest in the 3.12 venv: max |chunk - PyTorch| and seconds per chunk
  serve     with --serve: serve_policy_fpga.py --port 8080 in the background (left running); the laptop
            then starts the robot client through an SSH tunnel (guide §4)
Logs land in <remote>/logs/ on the board; the selftest rows in build/deploy/<name>/board_selftest.json.
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "fpga" / "board"))
import jb  # noqa: E402

CODE = [
    "fpga/board/runtime.py",
    "fpga/board/engine_server.py",
    "fpga/board/serve_policy_fpga.py",
    "fpga/board/check_engine.py",
    "fpga/board/run_smolvla.py",
    "fpga/board/setup_board.sh",
    "fpga/hls/pack_float.py",
]
CKPT = ["config.json", "model.safetensors", "policy_preprocessor.json", "policy_postprocessor.json"]
JUP = "/root/jupyter_notebooks"
PYNQ_PY = "/usr/local/share/pynq-venv/bin/python3"


def shell(folder: str, cmd: str, log: str, timeout: int = 1800, background: bool = False) -> bool:
    """Run a shell command on the board from `folder`; stream it here and into logs/<log>."""
    code = f"""
import os, subprocess
os.chdir({folder!r}); os.makedirs("logs", exist_ok=True)
log = os.path.join("logs", {log!r})
if {background!r}:
    p = subprocess.Popen({cmd!r}, shell=True, start_new_session=True,
                         stdout=open(log, "w"), stderr=subprocess.STDOUT)
    print("started pid", p.pid, "->", log, flush=True)
else:
    p = subprocess.Popen({cmd!r}, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    with open(log, "w") as f:
        for line in p.stdout:
            print(line, end="", flush=True)
            f.write(line)
    rc = p.wait()
    print("EXIT", rc, flush=True)
    if rc:
        raise SystemExit(rc)
"""
    return jb.run(code, timeout=timeout)


def wait_for(folder: str, log: str, marker: str, seconds: int) -> bool:
    code = f"""
import time
path = {folder!r} + "/logs/" + {log!r}
for _ in range({seconds}):
    try:
        text = open(path).read()
        if {marker!r} in text or "Error" in text or "Traceback" in text:
            break
    except FileNotFoundError:
        text = ""
    time.sleep(1)
print(text)
if {marker!r} not in text:
    raise SystemExit("no " + {marker!r} + " in " + path)
"""
    return jb.run(code, timeout=seconds + 30)


def push(rem: str, bit: Path, hwh: Path, model: Path, name: str, frames: list[Path]) -> None:
    for rel in CODE:
        jb.put(str(ROOT / rel), f"{rem}/{rel}")
    jb.put(str(bit), f"{rem}/{bit.name}")
    jb.put(str(hwh), f"{rem}/{hwh.name}")
    for p in frames:
        jb.put(str(p), f"{rem}/batches/{p.name}")
    ckpt = [model / c for c in CKPT] + sorted(model.glob("policy_*processor*.safetensors"))
    ckpt += [p for p in [model / "train_config.json"] if p.exists()]
    for p in ckpt:
        jb.put(str(p), f"{rem}/ckpt/{name}/{p.name}")
    if (model / "tokenizer").is_dir():
        for p in sorted((model / "tokenizer").glob("*")):
            jb.put(str(p), f"{rem}/ckpt/{name}/tokenizer/{p.name}")
    for d in sorted((ROOT / "build" / "stream" / "float").glob("case_*")):
        for p in sorted(d.glob("*")):
            jb.put(str(p), f"{rem}/build/stream/float/{d.name}/{p.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bit", help="float_*.bit from build_bd.tcl (.hwh beside it)")
    ap.add_argument("--model", help="checkpoint directory (release smolvla-so101-pickplace-9604.tar)")
    ap.add_argument("--batches", help="frame_*.pt directory (release smolvla-9604-replay-batches.tar.gz)")
    ap.add_argument("--remote", default="smolvla", help="folder under the board's Jupyter root")
    ap.add_argument("--user", default="ubuntu", help="board user that runs the policy server")
    ap.add_argument("--clock-mhz", type=float, default=None, help="PL clock after loading the overlay")
    ap.add_argument(
        "--setup", action="store_true", help="upload the offline Python 3.12 kit, run setup_board.sh"
    )
    ap.add_argument("--skip-push", action="store_true")
    ap.add_argument("--skip-engine-check", action="store_true")
    ap.add_argument("--skip-selftest", action="store_true")
    ap.add_argument("--serve", action="store_true", help="leave the policy server running on :8080")
    args = ap.parse_args()

    jb.login()
    rem = args.remote.strip("/")
    jroot = jb.api("GET", "contents/")["path"] or ""
    folder = f"{JUP}/{jroot}{rem}".rstrip("/")
    venv_py = f"/home/{args.user}/lerobot-venv/bin/python"

    if args.setup:
        print("== setup: Python 3.12 + LeRobot on the board (offline kit from build/board_py312)", flush=True)
        kit = ROOT / "build" / "board_py312"
        jb.put(str(ROOT / "fpga/board/setup_board.sh"), f"{rem}/fpga/board/setup_board.sh")
        for p in sorted(kit.glob("cpython-3.12.*aarch64*.tar.gz")):
            jb.put(str(p), f"{rem}/offline/{p.name}")
        for sub in ("wheels", "lerobot_fork"):
            for p in sorted((kit / sub).glob("*")):
                if p.is_file():
                    jb.put(str(p), f"{rem}/offline/{sub}/{p.name}")
        shell(
            folder,
            f"SUDO_USER={args.user} bash fpga/board/setup_board.sh {folder}/offline",
            "setup.log",
            7200,
        )
        return

    if not (args.bit and args.model and args.batches):
        sys.exit("--bit, --model and --batches are required (or --setup)")
    bit, model, batches = Path(args.bit), Path(args.model).expanduser(), Path(args.batches).expanduser()
    hwh = bit.with_suffix(".hwh")
    frames = sorted(batches.glob("frame_*.pt"))
    for f in [bit, hwh] + [model / c for c in CKPT]:
        if not f.exists():
            sys.exit(f"missing {f}")
    if not frames:
        sys.exit(f"no frame_*.pt in {batches}")
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", model.name)
    out = ROOT / "build" / "deploy" / name
    out.mkdir(parents=True, exist_ok=True)

    if not args.skip_push:
        print(f"== push to {folder}", flush=True)
        t0 = time.time()
        push(rem, bit, hwh, model, name, frames)
        print(f"   {time.time() - t0:.0f} s", flush=True)

    jb.api("POST", f"kernels/{jb.kernel()}/restart")  # releases an overlay or server a previous run left
    time.sleep(3)
    shell(
        folder, "pkill -f engine_server.py; pkill -f serve_policy_fpga.py; sleep 1; true", "cleanup.log", 60
    )
    clock = f" --clock-mhz {args.clock_mhz}" if args.clock_mhz else ""

    if not args.skip_engine_check:
        print("== engine check (PYNQ Python): every case + bandwidth", flush=True)
        shell(
            folder,
            f"{PYNQ_PY} fpga/board/check_engine.py --bit {bit.name} build/stream/float",
            "check_engine.log",
        )

    print("== engine server (background, root) + policy selftest (LeRobot venv)", flush=True)
    shell(folder, f"SUDO_USER={args.user} {PYNQ_PY} fpga/board/engine_server.py --bit {bit.name}{clock}",
          "engine_server.log", 60, background=True)  # fmt: skip
    if not wait_for(folder, "engine_server.log", "engine ready", 180):
        sys.exit("engine server failed")
    if not args.skip_selftest:
        cmd = f"sudo -u {args.user} {venv_py} fpga/board/serve_policy_fpga.py"
        cmd += f" --selftest ckpt/{name} --batches batches"
        shell(folder, cmd, "selftest.log", 3600)
        log = jb.api("GET", f"contents/{rem}/logs/selftest.log?type=file&format=text")["content"]
        rows = [json.loads(ln) for ln in log.splitlines() if ln.startswith("{")]
        (out / "board_selftest.json").write_text(json.dumps({"rows": rows, "log": log}, indent=1))
        print(f"saved {out / 'board_selftest.json'}")

    if args.serve:
        shell(folder, f"sudo -u {args.user} {venv_py} fpga/board/serve_policy_fpga.py --port 8080",
              "policy_server.log", 60, background=True)  # fmt: skip
        print(
            "policy server starting on the board's 127.0.0.1:8080. Laptop: ssh -N -L 8080:127.0.0.1:8080 "
            f"{args.user}@192.168.137.35, then the robot client (docs/run-smolvla-on-kr260.md §4)"
        )


if __name__ == "__main__":
    main()
