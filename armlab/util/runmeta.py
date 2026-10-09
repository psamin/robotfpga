"""Every eval or training run writes a JSON with its config, git hash, seed and versions."""

import json
import platform
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

PACKAGES = ("armlab", "mujoco", "numpy", "torch", "lerobot")


def git_hash() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout
        return out.stdout.strip() + ("-dirty" if dirty.strip() else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def versions() -> dict:
    v = {"python": sys.version.split()[0], "platform": platform.platform()}
    for p in PACKAGES:
        try:
            v[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            pass
    return v


def write_runmeta(path: str | Path, config: dict, seed: int, results: dict | None = None) -> dict:
    meta = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git": git_hash(),
        "seed": seed,
        "config": config,
        "versions": versions(),
        "results": results or {},
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2, default=str) + "\n")
    return meta
