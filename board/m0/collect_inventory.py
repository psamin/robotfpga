"""Read-only KR260 inventory; never imports PYNQ or programs the fabric.

Run on the board with system Python, then with the existing PYNQ venv Python
if present. JSON is emitted to stdout. This is not an M0 acceptance test.
"""

import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def read_text(path):
    try:
        return Path(path).read_text().replace("\x00", "\n").strip()
    except (OSError, UnicodeError) as exc:
        return {"unavailable": str(exc)}


def command(argv):
    if shutil.which(argv[0]) is None:
        return {"argv": argv, "unavailable": "executable not found"}
    try:
        result = subprocess.run(
            argv, capture_output=True, text=True, timeout=15, check=False
        )
        return {
            "argv": argv,
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"argv": argv, "unavailable": str(exc)}


def main():
    packages = {}
    for name in ("pynq", "numpy", "pynqmetadata", "pynqutils", "pynq-helloworld"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None

    report = {
        "schema_version": 1,
        "kind": "inventory_only",
        "m0_acceptance": "not_run",
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "system": platform.system(),
        "machine": platform.machine(),
        "kernel": platform.release(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "board_environment": os.environ.get("BOARD"),
        "os_release": read_text("/etc/os-release"),
        "device_tree_model": read_text("/proc/device-tree/model"),
        "device_tree_compatible": read_text("/proc/device-tree/compatible"),
        "fpga_manager_state": read_text("/sys/class/fpga_manager/fpga0/state"),
        "python_packages": packages,
        "debian_packages": command([
            "dpkg-query", "-W", "-f=${Package}\t${Version}\n",
            "linux-image*", "xrt*", "*zocl*", "xmutil*", "dfx-mgr*",
            "xlnx-firmware*", "kria-k26-bootfw*",
        ]),
        "boardid": command(["sudo", "-n", "xmutil", "boardid"]),
        "bootfw_status": command(["sudo", "-n", "xmutil", "bootfw_status"]),
        "fabric_modules": command(["lsmod"]),
    }
    json.dump(report, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
