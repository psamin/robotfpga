"""FPGA engine server for the KR260: owns the overlay and the DDR weight buffers; answers GEMM requests.

Runs in PYNQ's Python (3.10 on Kria-PYNQ 3.0.1). LeRobot needs Python >= 3.12, so the policy runs in another
process (fpga/board/serve_policy_fpga.py) and reaches this one through runtime.RemoteEngine over a Unix
socket that only this user can open (mode 0600). Requests (see runtime._send/_recv):
  upload  packed BF16 weights (uint16)  -> {"id"}   one DDR buffer each, freed when the client disconnects
  gemm    {"id", "mp", "k", "n"} + x (float32 n x k)  -> y (float32 n x mp), one engine call per K chunk
  stats   -> {"calls", "fpga_s", "buffers", "weight_bytes"}

    sudo -E /usr/local/share/pynq-venv/bin/python3 fpga/board/engine_server.py --bit float.bit
    python3 fpga/board/engine_server.py --mock layout     # host C core instead of the FPGA (no board)
"""

import argparse
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fpga.board.runtime import Engine, _recv, _send, chunk_layout  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bit", help="overlay .bit (board)")
    ap.add_argument("--clock-mhz", type=float, default=None, help="PL clock to set after loading the overlay")
    ap.add_argument("--mock", choices=["shim", "layout"], help="host-side C core instead of the board")
    ap.add_argument("--socket", default="/tmp/fpga-engine.sock")
    ap.add_argument(
        "--owner", default=os.environ.get("SUDO_USER"), help="user allowed to connect (sudo runs)"
    )
    args = ap.parse_args()
    if not args.bit and not args.mock:
        sys.exit("give --bit float.bit (board) or --mock layout")

    eng = Engine(args.bit, clock_mhz=args.clock_mhz) if args.bit else Engine(mock=args.mock)
    bufs, layouts = [], {}
    if os.path.exists(args.socket):
        os.unlink(args.socket)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    old = os.umask(0o177)  # socket created as 0600: only the owner can send requests
    srv.bind(args.socket)
    os.umask(old)
    if args.owner and os.geteuid() == 0:  # PYNQ needs root; hand the socket to the policy's user
        import pwd

        pw = pwd.getpwnam(args.owner)
        os.chown(args.socket, pw.pw_uid, pw.pw_gid)
    srv.listen(1)
    print(
        f"engine ready ({'board ' + args.bit if args.bit else 'mock ' + args.mock}) on {args.socket}",
        flush=True,
    )
    while True:
        conn, _ = srv.accept()
        print("policy connected", flush=True)
        try:
            while True:
                h, data = _recv(conn)
                try:
                    if h["op"] == "upload":
                        packed = np.frombuffer(data, np.uint16)
                        bufs.append(eng.upload(packed.copy() if eng.mock else packed))
                        _send(conn, {"id": len(bufs) - 1})
                    elif h["op"] == "gemm":
                        mp, k, n = h["mp"], h["k"], h["n"]
                        lay = layouts.setdefault((mp, k), chunk_layout(mp, k))
                        x = np.frombuffer(data, np.float32).reshape(n, k)
                        y = eng.gemm(bufs[h["id"]], lay, mp, x)
                        _send(conn, {"ok": True}, np.ascontiguousarray(y, np.float32).tobytes())
                    elif h["op"] == "stats":
                        wb = sum(int(b.nbytes) for b in bufs)
                        _send(
                            conn,
                            {
                                "calls": eng.calls,
                                "fpga_s": eng.fpga_s,
                                "buffers": len(bufs),
                                "weight_bytes": wb,
                            },
                        )
                    else:
                        _send(conn, {"error": f"unknown op {h['op']}"})
                except Exception as e:  # noqa: BLE001  report to the client, keep serving
                    _send(conn, {"error": f"{type(e).__name__}: {e}"})
        except ConnectionError:
            print(f"policy disconnected at {time.strftime('%H:%M:%S')}", flush=True)
        finally:
            conn.close()
            # A new policy load uploads every layer again; free this client's DDR first (CMA is small).
            for b in bufs:
                if hasattr(b, "freebuffer"):
                    b.freebuffer()
            bufs.clear()


if __name__ == "__main__":
    main()
