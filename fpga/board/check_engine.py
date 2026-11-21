"""Engine check: every test case through the runtime, compared with the float64 reference.

On the board:     python3 fpga/board/check_engine.py --bit float.bit build/stream/float
int16 bitstream: python3 fpga/board/check_engine.py --arith int16 --bit float_tp4_int16.bit build/stream/int16
Without a board: python3 fpga/board/check_engine.py --mock layout build/stream/float
                 python3 fpga/board/check_engine.py --mock ip --arith int16 build/stream/int16
Cases come from `make -C fpga/hls test` / `test-int16` (pack_float.py --tests / --tests-int16) or
policy/smolvla_fpga.py layer dumps. FP32 cases: PASS if |err| <= 1e-5 sum|x w| against the float64
product. int16 cases (dims.txt ends in "int16"): the stored quantized operands (xq, s_tok, s_row) go
through the runtime as they are; PASS if |err| <= 1e-6 sum|x w| against the exact dequantized integer
product (y_ref.bin), as in tb_float.cpp's int16 mode, and the padded columns are 0 (the runtime's own
quantizer is tested by fpga/board/test_runtime.py; a case stores no float x). Reports PASS/FAIL per case
and, on the board, the achieved weight bandwidth (target 16.2 GB/s).
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fpga.board.runtime import Engine, chunk_layout  # noqa: E402


def run_case(eng: Engine, d: Path, repeat: int):
    dims = (d / "dims.txt").read_text().split()
    mp, k, n, m = (int(v) for v in dims[:4])
    int16 = dims[4:] == ["int16"]
    if int16 != (eng.arith == "int16"):
        raise SystemExit(f"{d}: an {'int16' if int16 else 'FP32'} case, but the engine is {eng.arith}")
    packed = np.fromfile(d / "packed.bin", np.uint16)
    ref = np.fromfile(d / "y_ref.bin", np.float64).reshape(n, mp)
    sa = np.fromfile(d / "sumabs.bin", np.float64).reshape(n, mp)
    wbuf = eng.upload(packed)
    layout = chunk_layout(mp, k)
    if int16:
        srow = np.fromfile(d / "srow.bin", np.float32)
        xq = np.fromfile(d / "xq.bin", np.int16).reshape(n, k)
        stok = np.fromfile(d / "stok.bin", np.float32)

        def call():
            return eng.gemm_q16(wbuf, layout, mp, xq, stok, srow)

        tol = 1e-6
    else:
        x = np.fromfile(d / "x.bin", np.float32).reshape(n, k)

        def call():
            return eng.gemm(wbuf, layout, mp, x)

        tol = 1e-5
    y = call()
    t0 = time.perf_counter()
    for _ in range(repeat):
        call()
    dt = (time.perf_counter() - t0) / max(repeat, 1)
    worst = float((np.abs(y[:, :m] - ref[:, :m]) / (sa[:, :m] + 1e-30)).max())
    pad_zero = not np.any(y[:, m:])
    passes = -(-n // 32)
    gbps = passes * packed.nbytes / dt / 1e9 if repeat else float("nan")
    return worst <= tol and pad_zero, worst, dt, gbps, (m, k, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cases", help="directory of case_* dirs, or one case dir")
    ap.add_argument("--bit", help="overlay .bit (board)")
    ap.add_argument("--arith", choices=["fp32", "int16"], default="fp32", help="the bitstream's arithmetic")
    ap.add_argument("--clock-mhz", type=float, default=None, help="PL clock to set after loading the overlay")
    ap.add_argument("--mock", choices=["shim", "layout", "ip"], help="host-side C core instead of the board")
    ap.add_argument("--repeat", type=int, default=3)
    args = ap.parse_args()
    if args.bit:
        eng = Engine(args.bit, clock_mhz=args.clock_mhz, arith=args.arith)
    else:
        eng = Engine(mock=args.mock or "layout", arith=args.arith)
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
