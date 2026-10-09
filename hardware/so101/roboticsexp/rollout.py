"""Policy rollout session: keep the GPU server and tunnel up, start and rest the arm on command.

Commands (arrow keys / Voice Control phrases, or `./rollout.sh <command>` from any terminal):
  begin    Right  "robot begin"    start a rollout (the policy runs until rest, or MAX_S seconds)
  rest     Down   "robot finish"   stop the policy, glide back to the start pose, motors off
  success  Up     "robot capture"  mark the last rollout a success
  fail     Left   "robot scrap"    mark the last rollout a failure
  quit     Esc                     rest if needed, close the tunnel, cancel the server, print results

Start with the arm at rest: that pose is saved once and every rest returns to it.
The model runs on a Futurama GPU, never on this Mac. Results go to logs/rollouts-<time>.csv.
"""

import csv
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

SO101 = Path(__file__).resolve().parent
LEROBOT = Path.home() / "Documents/GitHub/lerobot"
PY = str(LEROBOT / ".venv/bin/python")
LOGIN = "pi34@planetexpress-login.cc.gatech.edu"
PORT = 8080
CMD_FILE = SO101 / "presets/rollout_cmd"
MAX_S = float(os.environ.get("MAX_S", 60))  # auto-rest after this long
# Motion settings, read at the start of every rollout so they can be tuned between rollouts.
# P16 I0 D32 are LeRobot's servo defaults. P12/D40 left the arm stuck in its folded rest pose:
# too little holding force to lift against gravity under the 50% torque limit.
SETTINGS = SO101 / "presets/rollout_settings.json"
DEFAULTS = {"P_GAIN": 16, "I_GAIN": 0, "D_GAIN": 32, "SMOOTH_ALPHA": 0.7, "MAX_STEP": 10}


def settings():
    try:
        saved = json.loads(SETTINGS.read_text())
    except (FileNotFoundError, ValueError):
        saved = {}
    s = {k: saved.get(k, DEFAULTS[k]) for k in DEFAULTS}
    SETTINGS.write_text(json.dumps(s, indent=2))  # always shows what the next rollout uses
    return s

env = dict(line.split("=", 1) for line in subprocess.run(
    ["bash", "-c", f"source {SO101}/env.sh && echo FOLLOWER_PORT=$FOLLOWER_PORT && echo TASK=$TASK"],
    capture_output=True, text=True, check=True).stdout.strip().splitlines())
# The policy server drops the camera rename map saved in training, so use the trained names directly.
CAMERAS = ("{ camera1: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, warmup_s: 3}, "
           "camera2: {type: opencv, index_or_path: 1, width: 480, height: 640, fps: 30, rotation: 90, warmup_s: 3}}")


def say(text):
    print(text, flush=True)
    subprocess.Popen(["say", text], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def ssh(cmd):
    return subprocess.run(["ssh", "-o", "BatchMode=yes", LOGIN, cmd], capture_output=True, text=True).stdout.strip()


def start_server():
    model = os.environ.get("MODEL") or ssh("ls -td /nethome/pi34/models/smolvla-* | head -1").replace(
        "/nethome/", "/nethome-instruction/", 1)
    job = ssh("squeue -u pi34 -n smolvla-serve -h -o %i | head -1")
    if not job:
        job = ssh(f"MODEL={model} sbatch --parsable ~/serve_policy.sbatch").splitlines()[-1]
        print(f"started policy server job {job} (1x H200)")
    print(f"model {model}\nwaiting for the policy server (job {job}) ...", flush=True)
    for _ in range(90):
        if ssh(f"grep -c SERVING /nethome/pi34/logs/smolvla-serve-{job}.log 2>/dev/null") not in ("", "0"):
            break
        time.sleep(5)
    node = ssh(f"scontrol show node $(squeue -j {job} -h -o %N) | grep -oE 'NodeHostName=\\S+' | cut -d= -f2")
    if not node:
        sys.exit(f"policy server job {job} is not running")
    time.sleep(5)
    tunnel = subprocess.Popen(["ssh", "-N", "-o", "BatchMode=yes", "-o", "ExitOnForwardFailure=yes", "-J", LOGIN,
                               "-L", f"{PORT}:127.0.0.1:{PORT}", f"pi34@{node}"])
    time.sleep(4)
    if tunnel.poll() is not None:
        sys.exit(f"SSH tunnel to {node} failed")
    return model, job, tunnel


class Session:
    def __init__(self, model, log_dir):
        self.model, self.client, self.watch, self.started = model, None, None, 0.0
        self.results, self.n = [], 0
        self.csv = log_dir / f"rollouts-{datetime.now():%Y%m%d-%H%M%S}.csv"
        self.log_dir = log_dir

    def begin(self):
        if self.client:
            return print("a rollout is already running", flush=True)
        self.n += 1
        cfg = settings()
        say(f"Rollout {self.n}")
        print(f"settings: {cfg}", flush=True)
        self.watch = subprocess.Popen(["nice", "-n", "10", PY, str(SO101 / "camera_presets.py"), "watch"],
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        log = open(self.log_dir / f"rollout-{datetime.now():%Y%m%d-%H%M%S}-{self.n}.log", "w")
        self.client = subprocess.Popen(
            [PY, str(SO101 / "smooth_client.py"), f"--server_address=127.0.0.1:{PORT}",
             "--robot.type=so101_follower", f"--robot.port={env['FOLLOWER_PORT']}", "--robot.id=my_follower",
             f"--robot.cameras={CAMERAS}", f"--robot.max_relative_target={cfg['MAX_STEP']}",
             "--robot.disable_torque_on_disconnect=false", f"--robot.position_p_coefficient={cfg['P_GAIN']}",
             f"--robot.position_i_coefficient={cfg['I_GAIN']}", f"--robot.position_d_coefficient={cfg['D_GAIN']}",
             f"--task={env['TASK']}", "--policy_type=smolvla",
             f"--pretrained_name_or_path={self.model}", "--policy_device=cuda", "--actions_per_chunk=50",
             "--chunk_size_threshold=0.5", "--aggregate_fn_name=weighted_average", "--fps=30"],
            stdout=log, stderr=subprocess.STDOUT, cwd=LEROBOT,
            env={**os.environ, "SMOOTH_ALPHA": str(cfg["SMOOTH_ALPHA"])})
        self.started = time.time()

    def rest(self):
        if self.client:
            self.client.send_signal(signal.SIGINT)  # client exits with the motors still holding the arm
            try:
                self.client.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.client.kill()
            self.client = None
            secs = time.time() - self.started
            self.results.append({"rollout": self.n, "seconds": round(secs, 1), "result": ""})
        say("Resting")
        subprocess.run([PY, str(SO101 / "home.py"), "go"])
        if self.watch:
            self.watch.terminate()
            self.watch = None
        if self.results and not self.results[-1]["result"]:
            say("Success or fail?")

    def mark(self, result):
        if not self.results:
            return print("no rollout to mark yet", flush=True)
        self.results[-1]["result"] = result
        say(f"Rollout {self.results[-1]['rollout']}: {result}")
        self.save()

    def save(self):
        with open(self.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["rollout", "seconds", "result"])
            w.writeheader()
            w.writerows(self.results)

    def summary(self):
        done = [r for r in self.results if r["result"]]
        wins = sum(r["result"] == "success" for r in done)
        return f"{wins}/{len(done)} successful rollouts" + (f" ({len(self.results) - len(done)} unmarked)"
                                                             if len(done) < len(self.results) else "")


def main():
    model, job, tunnel = start_server()
    log_dir = SO101 / "logs"
    log_dir.mkdir(exist_ok=True)
    subprocess.run([PY, str(SO101 / "home.py"), "save"], check=True)
    s = Session(model, log_dir)
    CMD_FILE.parent.mkdir(exist_ok=True)
    CMD_FILE.write_text("")
    pending = []

    sys.path.insert(0, str(LEROBOT / "src"))
    from lerobot.utils.keyboard_input import create_key_listener
    keymap = {"right": "begin", "down": "rest", "up": "success", "left": "fail", "esc": "quit"}
    listener = create_key_listener(lambda k: pending.append(keymap[k.lower()]) if k.lower() in keymap else None,
                                   controls_help="Right begin, Down rest, Up success, Left fail, Esc quit")
    say("Ready. Say robot begin to start a rollout.")
    try:
        while True:
            text = CMD_FILE.read_text().strip()
            if text:
                CMD_FILE.write_text("")
                pending.extend(text.split())
            if s.client and (s.client.poll() is not None or time.time() - s.started > MAX_S):
                print("rollout ended (client exited or time limit)", flush=True)
                s.rest()
            while pending:
                cmd = pending.pop(0)
                print(f"[rollout] {cmd}", flush=True)
                if cmd == "begin":
                    s.begin()
                elif cmd == "rest":
                    s.rest()
                elif cmd in ("success", "fail"):
                    s.mark(cmd)
                elif cmd == "quit":
                    raise KeyboardInterrupt
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        if s.client:
            s.rest()
        if listener is not None:
            listener.stop()
        tunnel.terminate()
        if not os.environ.get("KEEP_SERVER"):
            ssh(f"scancel {job}")
            print(f"policy server job {job} cancelled")
        s.save()
        say(s.summary())
        print(f"results: {s.csv}")


if __name__ == "__main__":
    main()
