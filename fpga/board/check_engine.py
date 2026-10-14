"""Engine check: every test case through the runtime, compared with the float64 reference.

On the board (after build_bd.tcl):   python3 fpga/board/check_engine.py --bit float.bit build/stream/float
Without a board:                     python3 fpga/board/check_engine.py --mock layout build/stream/float
Cases come from `make -C fpga/hls test` (pack_float.py) or policy/smolvla_fpga.py layer dumps.
Reports PASS/FAIL per case and, on the board, the achieved weight bandwidth (target 16.2 GB/s).
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fpga.board.runtime import Engine, chunk_layout  # noqa: E402


def run_case(eng: Engine, d: Path, repeat: int):
    mp, k, n, m = (int(v) for v in (d / "dims.txt").read_text().split())
    packed = np.fromfile(d / "packed.bin", np.uint16)
    x = np.fromfile(d / "x.bin", np.float32).reshape(n, k)
    ref = np.fromfile(d / "y_ref.bin", np.float64).reshape(n, mp)
    sa = np.fromfile(d / "sumabs.bin", np.float64).reshape(n, mp)
    wbuf = eng.upload(packed)
    layout = chunk_layout(mp, k)
    y = eng.gemm(wbuf, layout, mp, x)
    t0 = time.perf_counter()
    for _ in range(repeat):
        eng.gemm(wbuf, layout, mp, x)
    dt = (time.perf_counter() - t0) / max(repeat, 1)
    worst = float((np.abs(y[:, :m] - ref[:, :m]) / (sa[:, :m] + 1e-30)).max())
    passes = -(-n // 32)
    gbps = passes * packed.nbytes / dt / 1e9 if repeat else float("nan")
    return worst <= 1e-5, worst, dt, gbps, (m, k, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cases", help="directory of case_* dirs, or one case dir")
    ap.add_argument("--bit", help="overlay .bit (board)")
    ap.add_argument("--mock", choices=["shim", "layout"], help="host-side C core instead of the board")
    ap.add_argument("--repeat", type=int, default=3)
    args = ap.parse_args()
    eng = Engine(args.bit) if args.bit else Engine(mock=args.mock or "layout")
    root = Path(args.cases)
    dirs = (
        [root]
        if (root / "dims.txt").exists()
        else sorted(p for p in root.iterdir() if (p / "dims.txt").exists())
    )
    ok_all = True
    for d in dirs:
        ok, worst, dt, gbps, (m, k, n) = run_case(eng, d, args.repeat)
        ok_all &= ok
        bw = f", {gbps:.2f} GB/s weights" if args.bit else ""
        status = "PASS" if ok else "FAIL"
        print(f"{status}: {d.name} M={m} K={k} N={n}, err/sum|xw| {worst:.2e}, {dt * 1e3:.2f} ms{bw}")
    print("ALL PASS" if ok_all else "FAILURES")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
