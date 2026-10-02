"""M4: run the policy accelerator against the golden vectors and time it.

    python3 run_policy.py policy.bit path/to/artifacts   # artifacts has manifest.json, weights.bin, vectors/

Loads weights once (mode 0), then runs every vector (mode 1) and compares all 48
output bytes. Reports end-to-end latency as seen from Python.

UNTESTED until the board is up. Register names come from the HLS-generated .hwh;
run `print(ol.policy_top_0.register_map)` once and fix the names here if needed.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from pynq import Overlay, allocate

PACKET_BYTES = 27_658
OUT_LEN = 48
AP_START, AP_IDLE = 0x1, 0x4
STATUS = {0: "ok", 1: "bad shift", 2: "TLAST in the wrong place", 3: "bad mode"}


class PolicyAccel:
    def __init__(self, bitfile: str, artifacts: Path):
        self.ol = Overlay(bitfile)
        self.dma = self.ol.axi_dma_0
        self.ip = self.ol.policy_top_0
        manifest = json.loads((artifacts / "manifest.json").read_text())
        s = [layer["shift"] for layer in manifest["layers"]]
        assert len(s) == 8 and all(1 <= x <= 30 for x in s), s
        self.regs = self.ip.register_map
        self.regs.shifts_lo = s[0] | s[1] << 8 | s[2] << 16 | s[3] << 24
        self.regs.shifts_hi = s[4] | s[5] << 8 | s[6] << 16 | s[7] << 24
        self.inbuf = allocate(shape=(PACKET_BYTES,), dtype=np.uint8)
        self.outbuf = allocate(shape=(OUT_LEN,), dtype=np.int8)
        self._load_weights((artifacts / manifest["weights_file"]).read_bytes())

    def _start(self, mode: int) -> None:
        self.regs.mode = mode
        self.ip.write(0x00, AP_START)  # CTRL register

    def _finish(self) -> None:
        while not self.ip.read(0x00) & AP_IDLE:
            pass
        status = int(self.regs.ap_return)
        if status:
            raise RuntimeError(f"accelerator status {status}: {STATUS.get(status, '?')}")

    def _load_weights(self, blob: bytes) -> None:
        buf = allocate(shape=(len(blob),), dtype=np.uint8)
        buf[:] = np.frombuffer(blob, np.uint8)
        buf.flush()
        self._start(0)
        self.dma.sendchannel.transfer(buf)
        self.dma.sendchannel.wait()
        self._finish()
        buf.freebuffer()

    def infer(self, packet: bytes) -> np.ndarray:
        self.inbuf[:] = np.frombuffer(packet, np.uint8)
        self.inbuf.flush()
        self._start(1)
        self.dma.recvchannel.transfer(self.outbuf)
        self.dma.sendchannel.transfer(self.inbuf)
        self.dma.sendchannel.wait()
        self.dma.recvchannel.wait()
        self.outbuf.invalidate()
        self._finish()
        return np.array(self.outbuf)


def main() -> None:
    bitfile = sys.argv[1] if len(sys.argv) > 1 else "policy.bit"
    artifacts = Path(sys.argv[2] if len(sys.argv) > 2 else "artifacts/standin")
    accel = PolicyAccel(bitfile, artifacts)

    vdir = artifacts / "vectors"
    failures, times = 0, []
    inputs = sorted(vdir.glob("*_in.bin"))
    for path in inputs:
        packet = path.read_bytes()
        want = np.frombuffer((vdir / path.name.replace("_in", "_out")).read_bytes(), np.int8)
        t0 = time.perf_counter()
        got = accel.infer(packet)
        times.append(time.perf_counter() - t0)
        if not np.array_equal(got, want):
            failures += 1
            first = int(np.flatnonzero(got != want)[0])
            print(f"FAIL {path.name}: first mismatch at {first} (got {got[first]} want {want[first]})")

    ms = np.array(times) * 1e3
    print(f"{'PASS' if not failures else 'FAIL'}: {len(inputs)} vectors, {failures} failure(s)")
    print(f"latency ms: p50 {np.percentile(ms, 50):.2f}  p95 {np.percentile(ms, 95):.2f}  max {ms.max():.2f}")


if __name__ == "__main__":
    main()
