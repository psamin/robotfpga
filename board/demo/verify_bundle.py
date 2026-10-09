"""Read-only SHA256 integrity check for a prepared or transferred FPGA bundle."""
import argparse
import hashlib
import json
import re
from pathlib import Path, PurePosixPath


def verify(directory):
    base = directory.resolve()
    manifest = json.loads((base / "bundle.json").read_text())
    files = manifest["sha256"]
    if not isinstance(files, dict) or not files:
        raise ValueError("Missing per-file hash map")
    for name, expected in files.items():
        relative = PurePosixPath(name)
        if not name or relative.is_absolute() or ".." in relative.parts or "\\" in name or ":" in name:
            raise ValueError("Unsafe bundle path: " + name)
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected):
            raise ValueError("Invalid SHA256 for " + name)
        path = (base / name).resolve()
        path.relative_to(base)  # Also reject symlinks escaping the bundle.
        if not path.is_file():
            raise ValueError("Missing file: " + name)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected.lower():
            raise ValueError("SHA256 mismatch: " + name)
    return {"bundle": "PASS", "checked_files": len(files)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, nargs="?", default=Path.cwd())
    args = parser.parse_args()
    try:
        result = verify(args.directory)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        print(json.dumps({"bundle": "FAIL", "error": str(exc)}))
        return 1
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
