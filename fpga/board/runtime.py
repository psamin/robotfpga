"""KR260 runtime for the full-precision SmolVLA linear engine (fpga/hls).

SmolVLA runs on the board's ARM cores in PyTorch; every nn.Linear is swapped for FpgaLinear, which
keeps its BF16 weights in DDR (packed for the four HP ports by fpga/hls/pack_float.py) and sends
FP32 activations to the FPGA engine. The ARM adds the bias and runs everything nonlinear.

    from fpga.board.runtime import Engine, fpga_linears
    eng = Engine("float.bit")          # overlay from fpga/hls/run_float_hls.tcl + build_bd.tcl
    fpga_linears(policy, eng)          # policy = SmolVLAPolicy.from_pretrained(...).float()
    actions = policy.predict_action_chunk(batch)

Engine modes:
  Engine("float.bit")     the board (PYNQ)
  RemoteEngine(sock)      the board when the caller is not the PYNQ Python: talks to engine_server.py
  Engine(mock="shim")     the same C core as a host library, whole layer per call
  Engine(mock="layout")   the same C core, called once per 4096-wide K chunk with exactly the word
                          offsets the board path uses, so the DDR layout is tested without a board

Memory: the KR260 has 4 GB. After a layer's weights are packed into DDR, its PyTorch weight tensor is
freed, so the board holds one copy (0.71 GB of BF16 linear weights for SmolVLA). PYNQ allocates from
CMA; check `grep Cma /proc/meminfo` and raise the `cma=` boot argument if allocation fails (the needed
size on the board is UNKNOWN until measured).

UNTESTED ON HARDWARE until an overlay exists. Run `print(ol.float_top_0.register_map)` once and
adjust the pointer names below if Vivado renames them.
"""

from __future__ import annotations

import importlib.util
import time
from pathlib import Path

import numpy as np

K_MAX = 4096
LANES = 32
_PF = Path(__file__).resolve().parents[2] / "fpga" / "hls" / "pack_float.py"


def _pack_float():
    spec = importlib.util.spec_from_file_location("pack_float", _PF)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def chunk_layout(mp: int, k: int) -> list[tuple[int, int, int]]:
    """(k0, word offset of the chunk, words per port) for each K_MAX chunk; matches pack_float.pack."""
    groups, out, off = mp // LANES, [], 0
    for k0 in range(0, k, K_MAX):
        pw = groups * min(K_MAX, k - k0)
        out.append((k0, off, pw))
        off += 4 * pw
    return out


def set_pl_clock(mhz: float, built_mhz: float = 300.0) -> float:
    """Set PL clock 0. PYNQ's Clocks first; else scale DIVISOR0 of CRL_APB.PL0_REF_CTRL (ZU+) from the
    divisor the bitstream's block design programmed for `built_mhz`. Returns the frequency set."""
    try:
        from pynq.ps import Clocks

        Clocks.fclk0_mhz = mhz
        return float(Clocks.fclk0_mhz)
    except Exception as e:  # noqa: BLE001 - fall back to the register
        print(f"pynq Clocks failed ({e}); writing PL0_REF_CTRL directly", flush=True)
    import mmap
    import os

    PL0_REF_CTRL = 0xFF5E00C0
    fd = os.open("/dev/mem", os.O_RDWR | os.O_SYNC)
    try:
        m = mmap.mmap(fd, 4096, offset=PL0_REF_CTRL & ~0xFFF)
        off = PL0_REF_CTRL & 0xFFF
        reg = int.from_bytes(m[off : off + 4], "little")
        div0 = (reg >> 8) & 0x3F
        new = max(1, min(63, round(div0 * built_mhz / mhz)))
        m[off : off + 4] = ((reg & ~(0x3F << 8)) | (new << 8)).to_bytes(4, "little")
        m.close()
    finally:
        os.close(fd)
    return built_mhz * div0 / new


SLAB_BYTES = 32 << 20  # weights are packed into 32 MB CMA slabs (largest SmolVLA layer: 23.6 MB)
SLAB_ALIGN = 4096  # each layer starts on a 4 KB boundary inside its slab (AXI bursts never cross 4 KB)


class SlabWeights:
    """One layer's packed weights inside a shared CMA slab: what Engine.gemm needs from a weight buffer.

    One CMA allocation per layer wasted ~300 MB of the KR260's 1,000 MB CMA on alignment (the kernel aligns
    each allocation to up to 1 MB): 708 MB of SmolVLA weights did not fit. Slabs waste < 32 MB in total."""

    def __init__(self, slab, offset: int, nbytes: int):
        self.slab, self.offset, self.nbytes = slab, offset, nbytes
        self.physical_address = slab.physical_address + offset

    def freebuffer(self) -> None:  # the slab is freed by Engine.free_weights
        pass


class Engine:
    """PYNQ driver for float_top, or a host-side mock running the identical C core."""

    def __init__(self, bitfile: str | None = None, mock: str | bool = False, clock_mhz: float | None = None):
        self.pf = _pack_float()
        self.mock = {True: "shim"}.get(mock, mock) or None
        self.calls, self.fpga_s = 0, 0.0
        if self.mock:
            self.lib = self.pf.Lib()
            return
        from pynq import Overlay, allocate

        self.allocate = allocate
        self.ol = Overlay(bitfile)
        if clock_mhz:  # the bitstream is built for 300 MHz; run slower if timing did not close there
            set_pl_clock(clock_mhz)
        self.ip = self.ol.float_top_0
        self.regs = self.ip.register_map
        self._bx = self._by = None
        self._slabs, self._slab_off, self._dirty = [], SLAB_BYTES, set()

    def _set_ptr(self, name: str, addr: int) -> None:
        if hasattr(self.regs, name):
            setattr(self.regs, name, addr)
        else:  # HLS splits 64-bit pointers into _1 (low) and _2 (high)
            setattr(self.regs, f"{name}_1", addr & 0xFFFFFFFF)
            setattr(self.regs, f"{name}_2", addr >> 32)

    def _scratch(self, nx: int, ny: int):
        """Activation buffers, allocated once and grown on demand."""
        if self._bx is None or self._bx.size < nx:
            if self._bx is not None:
                self._bx.freebuffer()
            self._bx = self.allocate(shape=(nx,), dtype=np.float32)
        if self._by is None or self._by.size < ny:
            if self._by is not None:
                self._by.freebuffer()
            self._by = self.allocate(shape=(ny,), dtype=np.float32)
        return self._bx, self._by

    def upload(self, packed: np.ndarray):
        """Copy packed weights into DDR once; returns a SlabWeights (or the array when mocked).

        Layers are placed back to back (4 KB aligned) in 32 MB CMA slabs; a layer larger than a slab gets
        its own allocation. Slabs are flushed from the CPU cache before the next engine call."""
        if self.mock:
            return packed
        nbytes = packed.size * 2
        if nbytes > SLAB_BYTES:
            b = self.allocate(shape=(nbytes,), dtype=np.uint8)
            self._slabs.append(b)
            b.view(np.uint16)[:] = packed
            b.flush()
            return SlabWeights(b, 0, nbytes)
        if self._slab_off + nbytes > SLAB_BYTES:
            self._slabs.append(self.allocate(shape=(SLAB_BYTES,), dtype=np.uint8))
            self._slab_off = 0
        slab, off = self._slabs[-1], self._slab_off
        np.asarray(slab)[off : off + nbytes].view(np.uint16)[:] = packed
        self._dirty.add(id(slab))
        self._slab_off = off + -(-nbytes // SLAB_ALIGN) * SLAB_ALIGN
        return SlabWeights(slab, off, nbytes)

    def _flush_weights(self) -> None:
        if self._dirty:
            for slab in self._slabs:
                if id(slab) in self._dirty:
                    slab.flush()
            self._dirty.clear()

    def free_weights(self) -> None:
        """Release every weight slab (a new policy load uploads all layers again)."""
        if self.mock:
            return
        for slab in self._slabs:
            slab.freebuffer()
        self._slabs, self._slab_off, self._dirty = [], SLAB_BYTES, set()

    def gemm(self, wbuf, layout, mp: int, x: np.ndarray) -> np.ndarray:
        """y[N][Mp] = x @ W^T: one engine call per K chunk, summed in chunk order on the ARM."""
        n, k = x.shape
        t0 = time.perf_counter()
        if self.mock == "shim":
            y = self.lib(wbuf, mp, x)
        else:
            y = np.zeros((n, mp), np.float32)
            for k0, word_off, pw in layout:
                kc = min(K_MAX, k - k0)
                xc = np.ascontiguousarray(x[:, k0 : k0 + kc])
                if self.mock == "layout":  # one chunk = 4 port arrays back to back, as on the board
                    y += self.lib(wbuf[8 * word_off : 8 * (word_off + 4 * pw)], mp, xc)
                    continue
                self._flush_weights()
                bx, by = self._scratch(n * kc, n * mp)
                bx[: n * kc] = xc.reshape(-1)
                bx.flush()
                for p in range(4):  # one word = 8 x uint16 = 16 bytes
                    self._set_ptr(f"w{p}", wbuf.physical_address + 16 * (word_off + p * pw))
                self._set_ptr("xmem", bx.physical_address)
                self._set_ptr("ymem", by.physical_address)
                self.regs.M, self.regs.K, self.regs.N = mp, kc, n
                self.ip.write(0x00, 0x1)  # ap_start
                while not (self.ip.read(0x00) & 0x2):  # ap_done
                    pass
                if int(self.regs.ap_return):
                    raise RuntimeError(f"float engine status {int(self.regs.ap_return)}")
                by.invalidate()
                y += np.array(by[: n * mp]).reshape(n, mp)
        self.calls += 1
        self.fpga_s += time.perf_counter() - t0
        return y


def _send(sock, header: dict, payload: bytes = b"") -> None:
    import json
    import struct

    h = json.dumps(header).encode()
    sock.sendall(struct.pack("<QQ", len(h), len(payload)) + h + payload)


def _recv(sock) -> tuple[dict, bytes]:
    import json
    import struct

    def exact(n: int) -> bytearray:  # writable: torch.from_numpy warns on read-only arrays
        buf = bytearray()
        while len(buf) < n:
            part = sock.recv(min(n - len(buf), 1 << 22))
            if not part:
                raise ConnectionError("engine socket closed")
            buf += part
        return buf

    hn, pn = struct.unpack("<QQ", exact(16))
    return json.loads(exact(hn)), exact(pn)


class RemoteEngine:
    """Client for fpga/board/engine_server.py over a Unix socket: same upload/gemm interface as Engine.

    On the KR260, LeRobot needs Python >= 3.12 while PYNQ 3.0.1 runs on Python 3.10, so the policy and the
    FPGA driver live in two processes. Weights are packed here and stored in DDR by the server."""

    def __init__(self, path: str = "/tmp/fpga-engine.sock"):
        import socket
        import threading

        self.pf = _pack_float()
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.connect(path)
        self.lock = threading.Lock()  # the policy server answers gRPC calls from a thread pool
        self.calls, self.fpga_s = 0, 0.0

    def _call(self, header: dict, payload: bytes = b""):
        with self.lock:
            _send(self.sock, header, payload)
            h, data = _recv(self.sock)
        if "error" in h:
            raise RuntimeError(f"engine server: {h['error']}")
        return h, data

    def upload(self, packed: np.ndarray) -> int:
        h, _ = self._call(
            {"op": "upload", "n": int(packed.size)}, np.ascontiguousarray(packed, np.uint16).tobytes()
        )
        return h["id"]

    def gemm(self, wbuf: int, layout, mp: int, x: np.ndarray) -> np.ndarray:
        n, k = x.shape
        t0 = time.perf_counter()
        h, data = self._call(
            {"op": "gemm", "id": wbuf, "mp": mp, "k": k, "n": n},
            np.ascontiguousarray(x, np.float32).tobytes(),
        )
        self.calls += 1
        self.fpga_s += time.perf_counter() - t0
        return np.frombuffer(data, np.float32).reshape(n, mp)


class FpgaLinear:
    """Callable replacement for one nn.Linear: BF16 weights in DDR, FP32 math on the FPGA."""

    def __init__(self, weight: np.ndarray, bias: np.ndarray | None, eng: Engine):
        pf = eng.pf
        self.m, self.k = weight.shape
        packed, self.mp = pf.pack(pf.to_bf16_bits(weight.astype(np.float32)))
        self.layout = chunk_layout(self.mp, self.k)
        self.bias = bias
        self.eng = eng
        self.wbuf = eng.upload(packed)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        shape = x.shape
        x2 = np.ascontiguousarray(x.reshape(-1, self.k), np.float32)
        y = self.eng.gemm(self.wbuf, self.layout, self.mp, x2)[:, : self.m]
        if self.bias is not None:
            y = y + self.bias
        return y.reshape(*shape[:-1], self.m)


def fpga_linears(model, eng: Engine, free_weights: bool = True) -> int:
    """Swap every torch.nn.Linear in `model` for an FpgaLinear. Returns how many were swapped.

    With free_weights, each layer's PyTorch weight is released once it is packed, so only the DDR
    copy remains (the board has 4 GB)."""
    import types

    import torch
    from torch import nn

    count = 0
    for _, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        w = mod.weight.detach().float().cpu().numpy()
        b = None if mod.bias is None else mod.bias.detach().float().cpu().numpy()
        fl = FpgaLinear(w, b, eng)
        del w
        if free_weights:
            mod.weight.data = torch.empty(0, dtype=mod.weight.dtype, device=mod.weight.device)

        def fwd(self, x, _fl=fl):
            y = _fl(x.detach().float().cpu().numpy())
            return torch.from_numpy(y).to(x.device, x.dtype)

        mod.forward = types.MethodType(fwd, mod)
        count += 1
    return count
