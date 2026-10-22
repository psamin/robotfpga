"""Where does an engine call spend its time on the board? Run as root in PYNQ's Python.

    sudo -E /usr/local/share/pynq-venv/bin/python3 fpga/board/profile_engine.py --bit float_tp4.bit
    sudo <venv>/bin/python fpga/board/profile_engine.py          # XRT path on the loaded overlay

Times each stage of runtime.Engine.gemm for SmolVLA-shaped layers (ms per call; FPGA+regs is the
register writes, the run and the ap_done poll) and compares with the path over the Unix socket when
engine_server.py is reachable.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fpga.board.runtime import Engine, chunk_layout  # noqa: E402

SHAPES = [  # (M, K, N, calls per chunk) from fpga/hls/latency.py's layer list
    ("vision q/k/v/o", 768, 768, 1024, 96),
    ("vision fc1", 3072, 768, 1024, 24),
    ("vision fc2", 768, 3072, 1024, 24),
    ("connector", 960, 12288, 64, 2),
    ("LM qkvo", 960, 960, 177, 64),
    ("LM up/gate", 2560, 960, 177, 32),
    ("LM down", 960, 2560, 177, 16),
    ("expert qkvo", 736, 720, 50, 640),
    ("expert up/gate", 1920, 720, 50, 320),
    ("expert down", 736, 1920, 50, 160),
]


def profile(eng: Engine, mp: int, k: int, n: int, reps: int) -> dict:
    pf = eng.pf
    w = np.random.default_rng(0).normal(0, 0.05, (mp, k)).astype(np.float32)
    packed, mp2 = pf.pack(pf.to_bf16_bits(w))
    assert mp2 == mp
    wbuf = eng.upload(packed)
    lay = chunk_layout(mp, k)
    x = np.random.default_rng(1).normal(0, 1, (n, k)).astype(np.float32)
    eng.gemm(wbuf, lay, mp, x)  # warm up (scratch buffers)
    before = dict(eng.t)
    for _ in range(reps):
        eng.gemm(wbuf, lay, mp, x)
    eng.free_weights()
    return {key: 1e3 * (eng.t[key] - before[key]) / reps for key in eng.t}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bit", help="overlay to load (PYNQ's Python); without it: XRT on the loaded overlay")
    ap.add_argument("--clock-mhz", type=float, default=None)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--socket", default="/tmp/fpga-engine.sock", help="also time the socket path if it is up")
    args = ap.parse_args()
    eng = Engine(args.bit, clock_mhz=args.clock_mhz) if args.bit else Engine(backend="xrt")
    total_direct = total_fpga = 0.0
    print(
        f"{'layer':16s} {'MxKxN':>16s} {'calls':>5s} | copy_in flush  FPGA+regs inval copy_out"
        " | ms/call  s/chunk  FPGA s/chunk"
    )
    for name, m, k, n, calls in SHAPES:
        mp = -(-m // 32) * 32
        r = profile(eng, mp, k, n, args.reps)
        per_call = sum(r.values())
        total_direct += per_call * calls / 1e3
        total_fpga += r["fpga"] * calls / 1e3
        print(
            f"{name:16s} {f'{m}x{k}x{n}':>16s} {calls:5d} | {r['copy_in']:7.2f} {r['flush']:5.2f}"
            f" {r['fpga']:9.2f} {r['invalidate']:5.2f} {r['copy_out']:8.2f} |"
            f" {per_call:7.2f} {per_call * calls / 1e3:8.2f} {r['fpga'] * calls / 1e3:8.2f}"
        )
    print(f"direct path, all linear layers: {total_direct:.1f} s per chunk, FPGA busy {total_fpga:.1f} s")
    try:
        from fpga.board.runtime import RemoteEngine

        rem = RemoteEngine(args.socket)
        m, k, n = 768, 768, 1024
        w = np.random.default_rng(0).normal(0, 0.05, (m, k)).astype(np.float32)
        packed, mp = eng.pf.pack(eng.pf.to_bf16_bits(w))
        wid = rem.upload(packed)
        x = np.random.default_rng(1).normal(0, 1, (n, k)).astype(np.float32)
        rem.gemm(wid, chunk_layout(mp, k), mp, x)
        t0 = time.perf_counter()
        for _ in range(args.reps):
            rem.gemm(wid, chunk_layout(mp, k), mp, x)
        ms = 1e3 * (time.perf_counter() - t0) / args.reps
        print(f"socket path, vision 768x768x1024: {ms:.1f} ms/call")
    except Exception as e:  # noqa: BLE001
        print(f"remote path not measured: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
