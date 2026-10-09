"""Gather pinned FPGA sources and trained artifacts without merging branches."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PINS = [
    ("3ec8fd35417f7a4b7c9989e0fbd22ecd094a4f8c", "hls/", "hls/"),
    ("5ee0e03758ee3aaf87a2a04d790779e3a45c9186", "handoff/v3r2/", "artifacts/standin/"),
    ("2155ad6ec918d1cfd25329596f8361e7ebc4d4df", "ref/intref.py", "ref/intref.py"),
    ("2b713c69dd256e37f34d65e013d5a6a2c67750af", "board/demo/handoff_check.py", "board/demo/handoff_check.py"),
    ("7cb91b0a502ad694ae6ff06756ff11f77acc49aa", "board/run_policy.py", "board/run_policy.py"),
    ("4155f1148a24dc4a622014becda1b28405f288d9", "board/loopback.py", "board/loopback.py"),
    ("efd3f9cabe67ee22b2087cc6119da29773b10970", "board/m0/", "board/m0/"),
]
WEIGHTS_HASH = "c87abfae0cbedf771a9ef927f39302a43cdcbb954f7090aca4baa6bef350a732"


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def prepare(destination):
    destination = destination.resolve()
    if destination.exists():
        raise ValueError("Destination already exists; choose a new directory: " + str(destination))
    for revision, _, _ in PINS:
        found = subprocess.run(["git", "cat-file", "-e", revision + "^{commit}"],
                               cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if found.returncode:
            subprocess.run(["git", "fetch", "--no-tags", "origin", revision], cwd=ROOT, check=True)
    files = {}
    for revision, source, target in PINS:
        paths = git("ls-tree", "-r", "--name-only", revision, source).decode().splitlines()
        if not paths:
            raise ValueError("Pinned source missing: " + source)
        for name in paths:
            relative = target + name[len(source):] if source.endswith("/") else target
            path = destination / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            # Never use git checkout/archive or text decoding for binary tensors.
            data = git("show", revision + ":" + name)
            path.write_bytes(data)
            files[relative] = hashlib.sha256(data).hexdigest()
    if files["artifacts/standin/weights.bin"] != WEIGHTS_HASH:
        raise ValueError("Trained weights do not match the pinned hash")
    provenance = {"pins": PINS, "sha256": files,
                  "model": "trained v3r2 (standin directory name retained for HLS script)",
                  "overlay_included": False}
    (destination / "bundle.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print("Prepared {} files in {}".format(len(files), destination))
    print("Next: run board/demo/handoff_check.py with --verify-reference (see README).")
    print("No bitstream included. Build/integrate HLS IP and complete board acceptance first.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "build" / "fpga-handoff")
    args = parser.parse_args()
    try:
        prepare(args.out)
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        parser.exit(1, "Handoff preparation failed: {}\n".format(exc))


if __name__ == "__main__":
    main()
