"""M1: AXI DMA loopback through a stream FIFO in the fabric.

Sends random 27,658-byte packets (the real input size) out and back, checks every
byte, and reports the round-trip time. That time is the floor for our latency.

Run on the board with Kria-PYNQ. Copy loopback.bit and loopback.hwh (same base
name) next to this file, then either run it from a Jupyter cell with
`%run loopback.py loopback.bit`, or from a shell:

    sudo su
    source /etc/profile.d/pynq_venv.sh
    python3 loopback.py loopback.bit

UNTESTED until the board is up. If the DMA is not called axi_dma_0, check `ol.ip_dict`.
"""

import sys
import time

import numpy as np
from pynq import Overlay, allocate

PACKET_BYTES = 27_658
RUNS = 1000

ol = Overlay(sys.argv[1] if len(sys.argv) > 1 else "loopback.bit")
dma = ol.axi_dma_0
src = allocate(shape=(PACKET_BYTES,), dtype=np.uint8)
dst = allocate(shape=(PACKET_BYTES,), dtype=np.uint8)
rng = np.random.default_rng(0)

times = []
for i in range(RUNS):
    src[:] = rng.integers(0, 256, PACKET_BYTES, dtype=np.uint8)
    dst[:] = 0
    t0 = time.perf_counter()
    src.flush()
    dma.recvchannel.transfer(dst)
    dma.sendchannel.transfer(src)
    dma.sendchannel.wait()
    dma.recvchannel.wait()
    dst.invalidate()
    times.append(time.perf_counter() - t0)
    if not np.array_equal(src, dst):
        bad = np.flatnonzero(src != dst)
        sys.exit(f"FAIL run {i}: {bad.size} bytes differ, first at {bad[0]}")

ms = np.array(times) * 1e3
print(f"PASS: {RUNS} x {PACKET_BYTES} bytes round trip")
print(f"round trip ms: p50 {np.percentile(ms, 50):.3f}  p95 {np.percentile(ms, 95):.3f}  max {ms.max():.3f}")
