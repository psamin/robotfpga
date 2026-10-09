"""Generate standalone RTL test cases using the unchanged pinned NumPy reference."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import uuid

import numpy as np

PIN = "2155ad6ec918d1cfd25329596f8361e7ebc4d4df"
SEED = 0x71C0FFEE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", default=PIN)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    run = repo / "build" / ("requant-reference-" + uuid.uuid4().hex)
    run.mkdir(parents=True)
    revision = subprocess.check_output(
        ["git", "rev-parse", "--verify", args.revision + "^{commit}"], cwd=repo
    ).decode().strip()
    source = subprocess.check_output(["git", "show", revision + ":ref/intref.py"], cwd=repo)
    ref_path = run / "intref.py"
    ref_path.write_bytes(source)
    spec = importlib.util.spec_from_file_location("pinned_intref", ref_path)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    rng = np.random.RandomState(SEED)
    rows = []
    for shift in range(1, 31):
        divisor = 1 << shift
        values = list(range(-1024, 1024)) + [-(1 << 31), (1 << 31) - 1]
        for k in range(-129, 129):
            for delta in (-1, 0, 1):
                value = k * divisor - divisor // 2 + delta
                if -(1 << 31) <= value < (1 << 31):
                    values.append(value)
        values.extend(rng.randint(-(1 << 31), 1 << 31, size=2000, dtype=np.int64))
        inputs = np.asarray(values, dtype=np.int32)
        for relu in (0, 1):
            expected = reference.requant(inputs, shift, 0 if relu else -127, 127)
            for value, output in zip(inputs, expected):
                rows.append("{} {} {} {}\n".format(int(value), shift, relu, int(output)))
    vector_path = run / "vectors.txt"
    vector_path.write_text(str(len(rows)) + "\n" + "".join(rows), encoding="ascii")
    metadata = {
        "kind": "standalone_requant_reference_vectors",
        "revision": revision,
        "reference_sha256": hashlib.sha256(source).hexdigest(),
        "vectors_sha256": hashlib.sha256(vector_path.read_bytes()).hexdigest(),
        "python": sys.version.split()[0], "numpy": np.__version__,
        "seed": SEED, "count": len(rows),
    }
    (run / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("Generated {} actual-reference cases: {}".format(len(rows), vector_path))


if __name__ == "__main__":
    main()
