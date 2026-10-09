"""Lock webcam focus and white balance for SO-101 recording; exposure stays automatic.

  lock   put the cameras on auto, let focus / white balance / exposure settle under the current
         lighting for a few seconds, then freeze focus (C920) and white balance (both).
         Saves presets/<timestamp>.json and presets/latest.json.
  watch  keep presets/latest.json applied until the running recorder exits
         (opening a video stream resets C920 settings, so this runs alongside LeRobot)
  apply  re-apply a saved preset once (default: presets/latest.json)
  auto   hand focus and white balance back to the cameras
  show   print the current values

Why exposure is not locked: on macOS the C920's manual exposure does not hold once a stream is
open (locked at brightness 97, the recorded image came out at 203-217 after the stream opened,
even with the value re-applied). Auto exposure under steady lighting barely moves, and training
uses brightness augmentation. The wrist camera's auto mode also uses a hidden gain that manual
mode can't reach. Focus is the setting that matters most, and it does hold.

Run with the LeRobot venv:  ~/Documents/GitHub/lerobot/.venv/bin/python ~/so101/camera_presets.py lock
`lock` refuses to run while a recorder is using the cameras.
"""

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

UVC = str(Path.home() / ".local/bin/uvc-util")
PRESETS = Path(__file__).parent / "presets"
LEROBOT_CMD = "/bin/lerobot-[a-z]|replay_record[.]py|record_session[.]py|demo_record[.]py|async_inference[.]robot_client|smooth_client[.]py"  # console scripts and our recorders
SETTLE_S = 4
MIN_BRIGHTNESS = 60  # mean pixel value (0-255) below which the scene is too dark to record well
ANTI_FLICKER = ("power-line-frequency", "2")  # 60 Hz mains (US)

# USB vendor:product IDs stay fixed even when OpenCV camera indexes shift.
CAMERAS = {
    "front": {"usb": "0x046d:0x08e5", "opencv": dict(index_or_path=0, width=640, height=480),
              "always": {"auto-exposure-mode": "8"},  # aperture-priority auto exposure
              "manual": {"auto-focus": "false", "auto-white-balance-temp": "false"},
              "auto": {"auto-focus": "true", "auto-white-balance-temp": "true"},
              "values": ["focus-abs", "white-balance-temp"]},
    # Fixed-focus lens: only white balance can be locked.
    "wrist": {"usb": "0x1e45:0x8022", "opencv": dict(index_or_path=1, width=480, height=640, rotation=90),
              "always": {},
              "manual": {"auto-white-balance-temp": "false"},
              "auto": {"auto-white-balance-temp": "true"},
              "values": ["white-balance-temp"]},
}


class UVCError(RuntimeError):
    pass


def uvc(cam, *args):
    out = subprocess.run([UVC, "-V", CAMERAS[cam]["usb"], *args], capture_output=True, text=True)
    if out.returncode != 0:
        raise UVCError(f"uvc-util failed on {cam} {args}: {out.stderr.strip()}")
    return out.stdout.strip()


def get(cam, ctrl):
    return uvc(cam, "-o", ctrl)


def set_(cam, ctrl, value):
    uvc(cam, "-s", f"{ctrl}={value}")


def current(cam):
    return {ctrl: get(cam, ctrl) for ctrl in CAMERAS[cam]["values"]}


def lerobot_running():
    return subprocess.run(["pgrep", "-f", LEROBOT_CMD], capture_output=True).returncode == 0


def open_cameras():
    from lerobot.cameras import Cv2Rotation
    from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig

    cams = {}
    for name, c in CAMERAS.items():
        kw = dict(c["opencv"])
        kw["rotation"] = Cv2Rotation(kw.get("rotation", 0))
        cams[name] = OpenCVCamera(OpenCVCameraConfig(fps=30, warmup_s=3, **kw))
        cams[name].connect()
    return cams


def stream(cams, seconds):
    """Stream for `seconds`; return mean brightness (0-255) and frame rate per camera."""
    frames, last = {n: 0 for n in cams}, {}
    t0 = time.time()
    while time.time() - t0 < seconds:
        for n, cam in cams.items():
            last[n] = cam.async_read(timeout_ms=2000)
            frames[n] += 1
    return {n: (float(last[n].mean()), frames[n] / seconds) for n in cams}


def apply(preset, check=True):
    for cam, values in preset["cameras"].items():
        for ctrl, value in [ANTI_FLICKER, *CAMERAS[cam]["always"].items(), *CAMERAS[cam]["manual"].items(),
                            *values.items()]:
            set_(cam, ctrl, value)
    if check:
        bad = {cam: (want, current(cam)) for cam, want in preset["cameras"].items() if current(cam) != want}
        if bad:
            sys.exit(f"Readback mismatch, settings did not stick: {bad}")


def lock():
    if lerobot_running():
        sys.exit("A recorder is running. Stop it first; opening the cameras now can crash it.")
    for cam in CAMERAS:  # start from full auto so focus and white balance settle on this scene
        for ctrl, value in [ANTI_FLICKER, *CAMERAS[cam]["always"].items(), *CAMERAS[cam]["auto"].items()]:
            set_(cam, ctrl, value)
    cams = open_cameras()
    try:
        print(f"Letting focus, white balance and exposure settle for {SETTLE_S}s ...")
        stream(cams, SETTLE_S)
        for cam in CAMERAS:  # freeze focus and white balance where auto put them
            for ctrl, value in CAMERAS[cam]["manual"].items():
                set_(cam, ctrl, value)
        measured = stream(cams, 2)
        preset = {"created": datetime.now().isoformat(timespec="seconds"),
                  "measured": {c: {"brightness": round(b, 1), "fps": round(f, 1)} for c, (b, f) in measured.items()},
                  "cameras": {c: current(c) for c in CAMERAS}}
    finally:
        for cam in cams.values():
            cam.disconnect()
    PRESETS.mkdir(exist_ok=True)
    path = PRESETS / f"{datetime.now():%Y%m%d-%H%M%S}.json"
    for p in (path, PRESETS / "latest.json"):
        p.write_text(json.dumps(preset, indent=2))
    print(f"Locked: {preset['cameras']}\nMeasured: {preset['measured']}\nSaved {path}")
    dim = [c for c, m in preset["measured"].items() if m["brightness"] < MIN_BRIGHTNESS]
    if dim:
        print(f"WARNING: {dim} look dark (brightness < {MIN_BRIGHTNESS}). Add light before recording.")


def watch(preset):
    """Re-apply the preset whenever it drifts, until the recorder exits.

    Only sends USB control requests; never opens a video stream, so it is safe next to LeRobot.
    A failed read or write (the camera can be briefly busy) is logged and retried next round."""
    started, t0, t_start = False, time.time(), 0.0
    while True:
        running = lerobot_running()
        if started and not running:
            return
        if not started and time.time() - t0 > 120:
            sys.exit("No recorder appeared within 2 minutes; stopping watch.")
        if running and not started:
            t_start = time.time()
        started |= running
        for cam, want in preset["cameras"].items():
            try:
                checks = {**CAMERAS[cam]["always"], **CAMERAS[cam]["manual"], **want}
                have = {k: get(cam, k) for k in checks}
                if have != checks:
                    drift = {k: f"{have[k]}->{v}" for k, v in checks.items() if have[k] != v}
                    apply({"cameras": {cam: want}}, check=False)
                    print(f"[camera_presets {datetime.now():%H:%M:%S}] {cam} drifted, re-applied {drift}", flush=True)
            except UVCError as e:
                print(f"[camera_presets {datetime.now():%H:%M:%S}] skipped a check: {e}", flush=True)
        # Fast checks while the recorder opens the streams (that's when the C920 resets), then light ones.
        time.sleep(1.0 if not started or time.time() - t_start < 20 else 10.0)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "show"
    latest = PRESETS / "latest.json"
    try:
        if cmd == "lock":
            lock()
        elif cmd == "watch":
            watch(json.loads(latest.read_text()))
        elif cmd == "apply":
            path = Path(sys.argv[2]) if len(sys.argv) > 2 else latest
            apply(json.loads(path.read_text()))
            print(f"Applied and verified {path}")
        elif cmd == "auto":
            for cam in CAMERAS:
                for ctrl, value in CAMERAS[cam]["auto"].items():
                    set_(cam, ctrl, value)
            print("Focus and white balance back on auto.")
        elif cmd == "show":
            for cam in CAMERAS:
                switches = {**CAMERAS[cam]["always"], **CAMERAS[cam]["manual"]}
                print(cam, current(cam), {k: get(cam, k) for k in switches})
        else:
            sys.exit(__doc__)
    except UVCError as e:
        sys.exit(str(e))
