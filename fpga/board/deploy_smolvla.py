"""The pick-and-place SmolVLA on the KR260 FPGA, from the laptop, through the board's Jupyter (root kernel).

    # LeRobot venv on the laptop (torch, lerobot[smolvla], websockets); repo root
    python fpga/board/deploy_smolvla.py --setup           # once: Python 3.12 + LeRobot on the board, SSH key
    python fpga/board/deploy_smolvla.py --bit fpga/hls/build/float_tp4.bit --clock-mhz 250 \
        --model build/ckpt/smolvla-9604/smolvla-so101_medicine_bottle_pickplace-9604 \
        --batches build/ckpt/smolvla-9604/batches [--serve]   # push, engine check, selftest, serve

What it does on the board (docs/run-smolvla-on-kr260.md §1–3). Uploads land in the Jupyter root
(/root/jupyter_notebooks/<remote>/) and are mirrored to /home/<user>/smolvla/, owned by <user>, because the
policy server must run as that user while PYNQ needs root:
  push      fpga/ code, the bitstream (.bit + .hwh), the checkpoint, the replay batches, engine test cases
  engine    fpga/board/check_engine.py with the bitstream (root): every case PASS + weight bandwidth
  servers   engine_server.py (PYNQ's Python, root) in the background, socket handed to <user>;
            serve_policy_fpga.py --selftest as <user> (3.12 venv): max |chunk - PyTorch|, seconds per chunk
  serve     with --serve: serve_policy_fpga.py --port 8080 as <user> in the background (left running); then
            run fpga/board/laptop_client.ps1 on the laptop (SSH tunnel + robot client, guide §4)
Logs: /home/<user>/smolvla/logs/ on the board; selftest rows in build/deploy/<name>/board_selftest.json.
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
HF_HUB = Path.home() / ".cache" / "huggingface" / "hub"
HF_MAX_FILE = 50 << 20  # backbone config/tokenizer files only, never its 2 GB of weights
PYNQ_PY = "/usr/local/share/pynq-venv/bin/python3"


def shell(folder: str, cmd: str, log: str, timeout: int = 1800, background: bool = False, sink=None) -> bool:
    """Run a shell command on the board from `folder`; stream it here and into <folder>/logs/<log>."""
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
    return jb.run(code, timeout=timeout, sink=sink)


def wait_for(folder: str, log: str, marker: str, seconds: int) -> bool:
    code = f"""
import time
path = {folder!r} + "/logs/" + {log!r}
text = ""
for _ in range({seconds}):
    try:
        text = open(path).read()
        if {marker!r} in text or "Error" in text or "Traceback" in text:
            break
    except FileNotFoundError:
        pass
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
    ckpt = [model / c for c in CKPT if c != "config.json"] + sorted(
        model.glob("policy_*processor*.safetensors")
    )
    ckpt += [p for p in [model / "train_config.json"] if p.exists()]
    for p in ckpt:
        jb.put(str(p), f"{rem}/ckpt/{name}/{p.name}")
    cfg_path, vlm = board_config(model)
    jb.put(str(cfg_path), f"{rem}/ckpt/{name}/config.json")
    push_hf_cache(rem, vlm)
    if (model / "tokenizer").is_dir():
        for p in sorted((model / "tokenizer").glob("*")):
            if not p.name.startswith("."):  # skip macOS ._* metadata from tars made on a Mac
                jb.put(str(p), f"{rem}/ckpt/{name}/tokenizer/{p.name}")
    for d in sorted((ROOT / "build" / "stream" / "float").glob("case_*")):
        for p in sorted(d.glob("*")):
            jb.put(str(p), f"{rem}/build/stream/float/{d.name}/{p.name}")


def board_config(model: Path) -> tuple[Path, str]:
    """The checkpoint's config.json with load_vlm_weights off (written next to the deploy outputs).

    The released config has load_vlm_weights=true: SmolVLAPolicy.from_pretrained then downloads and loads the
    SmolVLM2 backbone (2 GB FP32) before the checkpoint overwrites every weight. Measured on the laptop:
    3.19 GB peak RSS vs 2.35 GB with false, identical action chunk (3.6e-7 from the release reference) and no
    network. The KR260 has ~2.9 GB for Linux once 1 GB is reserved as CMA."""
    cfg = json.loads((model / "config.json").read_text())
    cfg["load_vlm_weights"] = False
    out = ROOT / "build" / "deploy" / f"{model.name}_config.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cfg, indent=4))
    return out, cfg.get("vlm_model_name", "")


def push_hf_cache(rem: str, repo_id: str) -> None:
    """The backbone's config/processor/tokenizer from the laptop's HF cache, so the board stays offline."""
    if not repo_id:
        return
    src = HF_HUB / ("models--" + repo_id.replace("/", "--"))
    if not (src / "refs" / "main").exists():
        sys.exit(f"{repo_id} is not in {HF_HUB}: run the laptop selftest once online to cache its config")
    rev = (src / "refs" / "main").read_text().strip()
    dst = f"{rem}/hfcache/hub/{src.name}"
    jb.put(str(src / "refs" / "main"), f"{dst}/refs/main")
    for p in sorted((src / "snapshots" / rev).glob("*")):
        if p.is_file() and p.stat().st_size <= HF_MAX_FILE:
            jb.put(str(p), f"{dst}/snapshots/{rev}/{p.name}")


def mirror(src: str, home: str, user: str) -> bool:
    """Copy the Jupyter upload folder to the user's home (only newer files) and give it to the user."""
    return jb.run(
        f"""
import subprocess
subprocess.run(["mkdir", "-p", {home!r}], check=True)
subprocess.run(["cp", "-ru", {src!r} + "/.", {home!r} + "/"], check=True)
subprocess.run(["chown", "-R", {user!r} + ":" + {user!r}, {home!r}], check=True)
print("mirrored to", {home!r}, flush=True)
""",
        timeout=1800,
    )


def install_ssh_key(user: str) -> None:
    pub = next(iter(sorted(Path.home().glob(".ssh/id_*.pub"))), None)
    if pub is None:
        print("no ~/.ssh/id_*.pub on the laptop; skipping the SSH key (ssh-keygen -t ed25519)")
        return
    key = pub.read_text().strip()
    jb.run(
        f"""
import os, pwd, subprocess
pw = pwd.getpwnam({user!r}); d = os.path.join(pw.pw_dir, ".ssh"); f = os.path.join(d, "authorized_keys")
os.makedirs(d, mode=0o700, exist_ok=True)
have = open(f).read() if os.path.exists(f) else ""
if {key!r} not in have:
    open(f, "a").write({key!r} + "\\n")
os.chmod(f, 0o600); subprocess.run(["chown", "-R", {user!r} + ":" + {user!r}, d], check=True)
print("ssh key installed for", {user!r}, flush=True)
""",
        timeout=60,
    )


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
    ap.add_argument("--keep-desktop", action="store_true", help="do not stop the board's GNOME desktop")
    args = ap.parse_args()

    jb.login()
    rem = args.remote.strip("/")
    jroot = jb.api("GET", "contents/")["path"] or ""
    upload = f"{JUP}/{jroot}{rem}".rstrip("/")
    home = f"/home/{args.user}/smolvla"
    venv_py = f"/home/{args.user}/lerobot-venv/bin/python"

    if args.setup:
        print(
            "== setup: Python 3.12 + LeRobot on the board (offline kit build/board_py312), SSH key",
            flush=True,
        )
        kit = ROOT / "build" / "board_py312"
        jb.put(str(ROOT / "fpga/board/setup_board.sh"), f"{rem}/fpga/board/setup_board.sh")
        for p in sorted(kit.glob("cpython-3.12.*aarch64*.tar.gz")):
            jb.put(str(p), f"{rem}/offline/{p.name}")
        for sub in ("wheels", "lerobot_fork"):
            for p in sorted((kit / sub).glob("*")):
                if p.is_file():
                    jb.put(str(p), f"{rem}/offline/{sub}/{p.name}")
        install_ssh_key(args.user)
        shell(
            upload,
            f"SUDO_USER={args.user} bash fpga/board/setup_board.sh {upload}/offline",
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
        print(f"== push to {upload}, mirror to {home}", flush=True)
        t0 = time.time()
        push(rem, bit, hwh, model, name, frames)
        if not mirror(upload, home, args.user):
            sys.exit("mirror failed")
        print(f"   {time.time() - t0:.0f} s", flush=True)

    jb.api("POST", f"kernels/{jb.kernel()}/restart")  # releases an overlay or server a previous run left
    time.sleep(3)
    shell(
        home,
        "pkill -f '[e]ngine_server.py'; pkill -f '[s]erve_policy_fpga.py'; sleep 1; true",
        "cleanup.log",
        60,
    )
    clock = f" --clock-mhz {args.clock_mhz}" if args.clock_mhz else ""
    # the policy process: the user's 3.12 venv, Hugging Face fully offline from the pushed cache
    hf = f"HF_HOME={home}/hfcache HF_HUB_OFFLINE=1"
    policy = f"sudo -u {args.user} env {hf} {venv_py} fpga/board/serve_policy_fpga.py"
    if not args.keep_desktop:  # GNOME holds several hundred MB the model load needs
        shell(
            home,
            "xmutil desktop_disable || systemctl isolate multi-user.target; free -m | head -2",
            "desktop.log",
            120,
        )

    if not args.skip_engine_check:
        print("== engine check (PYNQ Python): every case + bandwidth", flush=True)
        shell(
            home,
            f"{PYNQ_PY} fpga/board/check_engine.py --bit {bit.name}{clock} build/stream/float",
            "check_engine.log",
        )

    print("== engine server (background, root) + policy selftest (LeRobot venv)", flush=True)
    shell(home, f"SUDO_USER={args.user} {PYNQ_PY} fpga/board/engine_server.py --bit {bit.name}{clock}",
          "engine_server.log", 60, background=True)  # fmt: skip
    if not wait_for(home, "engine_server.log", "engine ready", 180):
        sys.exit("engine server failed")
    if not args.skip_selftest:
        cmd = f"{policy} --selftest ckpt/{name} --batches batches"
        sink: list[str] = []
        ok = shell(home, cmd, "selftest.log", 3600, sink=sink)
        log = "".join(sink)
        rows = [json.loads(ln) for ln in log.splitlines() if ln.startswith("{")]
        (out / "board_selftest.json").write_text(json.dumps({"ok": ok, "rows": rows, "log": log}, indent=1))
        print(f"saved {out / 'board_selftest.json'}")
        if not ok:
            sys.exit("selftest failed")

    if args.serve:
        shell(home, f"{policy} --port 8080", "policy_server.log", 60, background=True)
        print(
            f"policy server starting on the board's 127.0.0.1:8080 (checkpoint {home}/ckpt/{name}).\n"
            "Laptop: fpga/board/laptop_client.ps1 opens the tunnel and starts the robot client (guide §4)."
        )


if __name__ == "__main__":
    main()
