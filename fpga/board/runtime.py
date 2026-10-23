"""KR260 runtime for the full-precision SmolVLA linear engine (fpga/hls).

SmolVLA runs on the board's ARM cores in PyTorch; every nn.Linear is swapped for FpgaLinear, which
keeps its BF16 weights in DDR (packed for the four HP ports by fpga/hls/pack_float.py) and sends
FP32 activations to the FPGA engine. The ARM adds the bias and runs everything nonlinear.

    from fpga.board.runtime import Engine, fpga_linears
    eng = Engine(backend="xrt")        # overlay already loaded (fpga/board/load_overlay.py); root
    fpga_linears(policy, eng)          # policy = SmolVLAPolicy.from_pretrained(...).float()
    actions = policy.predict_action_chunk(batch)

Engine modes:
  Engine(backend="xrt")   the board from any Python (LeRobot's 3.12 venv, as root): CMA buffers through
                          XRT and the IP's registers through /dev/mem (fpga/board/xrtmem.py); the
                          bitstream must already be loaded. Activations never leave the process.
  Engine("float.bit")     the board through PYNQ (its Python 3.10): loads the overlay, same engine code
  RemoteEngine(sock)      the board through engine_server.py over a Unix socket (the earlier two-process
                          design: ~9 s of copies per chunk; kept for comparison)
  Engine(mock="shim")     the same C core as a host library, whole layer per call
  Engine(mock="layout")   the same C core, called once per 4096-wide K chunk with exactly the word
                          offsets the board path uses, so the DDR layout is tested without a board
  Engine(mock="ip")       the board code path on host memory: slabs, scratch buffers, the pointer and
                          size registers and the tiled result buffer, with float_top emulated at the
                          register level by the C core (_FakeIp). Tests _gemm_hw without a board.

Buffers are cacheable CMA, flushed/invalidated over the bytes a call uses (PYNQ's default is
uncached memory, which reads at ~120 MB/s: 8.6 s of a chunk went into copying results out).

Memory: the KR260 has 4 GB. After a layer's weights are packed into DDR, its PyTorch weight tensor is
freed, so the board holds one copy (0.71 GB of BF16 linear weights for SmolVLA). Weights are packed
into 32 MB CMA slabs; check `grep Cma /proc/meminfo` (cma=1000M on the board).

ARM-side work per engine call is one copy of X into the CMA buffer and one pass over the result buffer
that un-tiles Y and adds the bias (gemm_parts). fpga_linears(fuse=True) gives Linear modules that take
the same input (q/k/v, gate/up) one shared call; memo=True skips the expert's cross-attention k/v
projections when their input (the prefix cache) is bit-identical to the previous step's.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

K_MAX = 4096
LANES = 32
NT = 32  # tokens per tile on chip; Y is written per tile-group (float_linear.h)
_PF = Path(__file__).resolve().parents[2] / "fpga" / "hls" / "pack_float.py"

# s_axi_control register map of float_top (Vitis HLS driver xfloat_top_hw.h; fixed by the argument list)
REG_CTRL, REG_RETURN = 0x00, 0x10
REG_W = (0x18, 0x24, 0x30, 0x3C)  # 64-bit pointers: low word at the offset, high word at +4
REG_XMEM, REG_YMEM = 0x48, 0x54
REG_M, REG_K, REG_N = 0x60, 0x68, 0x70
IP_BASE = 0xA0000000  # float_top_0 in build_bd.tcl's address map (the .hwh says so too)
IP_SPAN = 0x10000


def _pack_float():
    spec = importlib.util.spec_from_file_location("pack_float", _PF)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _host_lib(pf):
    """The C core as a host library (fpga/hls/build/libfloat_linear.so, or FSTREAM_LIB=<path> for another
    build of fpga/hls/float_shim.cpp)."""
    path = os.environ.get("FSTREAM_LIB")
    return pf.Lib(Path(path)) if path else pf.Lib()


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
        print(f"pynq Clocks unavailable ({type(e).__name__}); writing PL0_REF_CTRL directly", flush=True)
    import mmap

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


class _XrtMem:
    """A cacheable CMA buffer through fpga/board/xrtmem.py (any Python, root)."""

    def __init__(self, dev, nbytes: int):
        self.buf = dev.allocate(nbytes)
        self.array = self.buf.array
        self.physical_address = self.buf.physical_address

    def flush(self, off: int, n: int) -> None:
        self.buf.flush(off, n)

    def invalidate(self, off: int, n: int) -> None:
        self.buf.invalidate(off, n)

    def free(self) -> None:
        self.buf.free()


class _PynqMem:
    """The same through pynq.allocate (cacheable); slices sync only their own bytes."""

    def __init__(self, allocate, nbytes: int):
        self.buf = allocate(shape=(nbytes,), dtype=np.uint8, cacheable=True)
        self.array = np.asarray(self.buf)
        self.physical_address = self.buf.physical_address

    def flush(self, off: int, n: int) -> None:
        self.buf[off : off + n].flush()

    def invalidate(self, off: int, n: int) -> None:
        self.buf[off : off + n].invalidate()

    def free(self) -> None:
        self.buf.freebuffer()


class _FakeMem:
    """Host memory standing in for a CMA buffer (Engine(mock="ip")); the address is made up but unique."""

    def __init__(self, physical_address: int, nbytes: int):
        self.array = np.zeros(nbytes, np.uint8)
        self.physical_address = physical_address

    def flush(self, off: int, n: int) -> None:
        pass

    def invalidate(self, off: int, n: int) -> None:
        pass

    def free(self) -> None:
        pass


class _FakeIp:
    """float_top at the register level: ap_start reads the four weight pointers, X/Y pointers and M/K/N
    from the registers Engine wrote, resolves them into the fake buffers and runs the C core, writing
    Y_tiled[g][npad][32] as the hardware does. Registers keep their values between runs, as on the IP."""

    def __init__(self, lib):
        self.lib, self.regs, self.bufs, self._next = lib, {}, [], 0x1000_0000

    def allocate(self, nbytes: int) -> _FakeMem:
        b = _FakeMem(self._next, nbytes)
        self._next += -(-nbytes // SLAB_ALIGN) * SLAB_ALIGN + SLAB_ALIGN
        self.bufs.append(b)
        return b

    def _at(self, paddr: int, nbytes: int) -> np.ndarray:
        for b in self.bufs:
            off = paddr - b.physical_address
            if 0 <= off and off + nbytes <= b.array.size:
                return b.array[off : off + nbytes]
        raise ValueError(f"pointer {paddr:#x} ({nbytes} bytes) is outside every buffer")

    def _r64(self, off: int) -> int:
        return self.regs.get(off, 0) | (self.regs.get(off + 4, 0) << 32)

    def read(self, off: int) -> int:
        return 0x6 if off == REG_CTRL else self.regs.get(off, 0)  # ap_done | ap_idle: runs are synchronous

    def write(self, off: int, value: int) -> None:
        self.regs[off] = value
        if off == REG_CTRL and value & 0x1:
            self._start()

    def _start(self) -> None:
        mp, k, n = self.regs[REG_M], self.regs[REG_K], self.regs[REG_N]
        groups, npad = mp // LANES, -(-n // NT) * NT
        pw = groups * k  # words per port; the ports lie back to back in the packed layout
        packed = np.concatenate([self._at(self._r64(r), 16 * pw).view(np.uint16) for r in REG_W])
        x = self._at(self._r64(REG_XMEM), 4 * n * k).view(np.float32).reshape(n, k)
        y = self.lib(packed, mp, x)  # [n][mp] in the core's bits
        yt = self._at(self._r64(REG_YMEM), 4 * groups * npad * LANES).view(np.float32)
        yt = yt.reshape(groups, npad, LANES)
        yt[:, :n, :] = y.reshape(n, groups, LANES).transpose(1, 0, 2)
        yt[:, n:, :] = 0.0  # the zero padding tokens' sums
        self.regs[REG_RETURN] = 0


class SlabWeights:
    """One layer's packed weights inside a shared CMA slab: what Engine.gemm needs from a weight buffer.

    One CMA allocation per layer wasted ~300 MB of the KR260's 1,000 MB CMA on alignment (the kernel aligns
    each allocation to up to 1 MB): 708 MB of SmolVLA weights did not fit. Slabs waste < 32 MB in total."""

    def __init__(self, slab, offset: int, nbytes: int):
        self.slab, self.offset, self.nbytes = slab, offset, nbytes
        self.physical_address = slab.physical_address + offset


def _assign(dst, src, bias, accumulate: bool) -> None:
    """dst = src (+ bias), or dst += src; torch tensors or numpy arrays (views into the buffers are fine)."""
    if isinstance(dst, np.ndarray):
        if accumulate:
            np.add(dst, src, out=dst)
        elif bias is None:
            np.copyto(dst, src)
        else:
            np.add(src, bias, out=dst)
    elif accumulate:
        dst.add_(src)
    elif bias is None:
        dst.copy_(src)
    else:
        sys.modules["torch"].add(src, bias, out=dst)


def _untile(yt, g0: int, out, bias, accumulate: bool) -> None:
    """out[n][m] = Y_tiled[g0 : g0 + ceil(m / 32)][:n][:] un-tiled, plus the bias, in one pass over the
    result buffer (accumulate: out += ..., for the K chunks after the first; the bias then comes last).
    yt: (groups, npad, 32) view of the result buffer; out (n, m) and bias (m,) of the same kind."""
    n, m = out.shape
    gf, rem = divmod(m, LANES)
    if gf:
        src = yt[g0 : g0 + gf, :n, :]
        src = src.transpose(1, 0, 2) if isinstance(src, np.ndarray) else src.permute(1, 0, 2)
        dst = out if rem == 0 else out[:, : gf * LANES]  # (n, 32 gf): splitting its last axis is a view
        b = None if bias is None else bias[: gf * LANES].reshape(gf, LANES)
        _assign(dst.reshape(n, gf, LANES), src, b, accumulate)
    if rem:
        b = None if bias is None else bias[gf * LANES :]
        _assign(out[:, gf * LANES :], yt[g0 + gf, :n, :rem], b, accumulate)


class Engine:
    """Driver for float_top (XRT or PYNQ), or a host-side mock running the identical C core."""

    def __init__(
        self,
        bitfile: str | None = None,
        mock: str | bool = False,
        clock_mhz: float | None = None,
        backend: str | None = None,
        ip_base: int = IP_BASE,
        tiled_y: bool | None = None,
        cache_regs: bool = True,
    ):
        self.pf = _pack_float()
        self.mock = {True: "shim"}.get(mock, mock) or None
        self.calls, self.fpga_s = 0, 0.0
        self.lock = threading.Lock()  # one call at a time when several threads share the engine
        self.poll_sleep = 0.0  # > 0: yield the GIL while waiting for ap_done (overlap_image_embedding)
        self.t = dict.fromkeys(("copy_in", "flush", "fpga", "invalidate", "copy_out"), 0.0)
        # float_linear.h since v4 writes Y group-major (one 4 KB burst per tile-group); FSTREAM_LEGACY_Y=1
        # for bitstreams built from v3 (float_tp4.bit), which wrote Y row-major
        self.tiled_y = os.environ.get("FSTREAM_LEGACY_Y") != "1" if tiled_y is None else tiled_y
        # registers keep their values between runs: write only what changed (pointers of the scratch
        # buffers almost never, M/K/N rarely); cache_regs=False writes all 13 words every call
        self._regs = {} if cache_regs else None
        if self.mock in ("shim", "layout"):
            self.lib = _host_lib(self.pf)
            return
        if self.mock == "ip":
            self.backend = "ip"
            self.lib = _host_lib(self.pf)
            fake = _FakeIp(self.lib)
            self._alloc = fake.allocate
            self._rd, self._wr = fake.read, fake.write
        elif self.mock:
            raise ValueError(f"mock {mock!r}: shim, layout or ip")
        else:
            self.backend = backend or ("pynq" if bitfile else "xrt")
            if self.backend == "pynq":
                from pynq import Overlay, allocate

                self.ol = Overlay(bitfile)
                if clock_mhz:  # the bitstream is built for 300 MHz; run slower if timing did not close there
                    set_pl_clock(clock_mhz)
                self.ip = self.ol.float_top_0
                self._alloc = lambda n: _PynqMem(allocate, n)
                self._rd, self._wr = self.ip.read, self.ip.write
            elif self.backend == "xrt":
                sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
                from fpga.board.xrtmem import Mmio, XrtDevice

                if clock_mhz:
                    set_pl_clock(clock_mhz)
                self.dev = XrtDevice()
                self.mmio = Mmio(ip_base, IP_SPAN)
                self._alloc = lambda n: _XrtMem(self.dev, n)
                self._rd, self._wr = self.mmio.read, self.mmio.write
                if self._rd(REG_CTRL) & 0x4 == 0:
                    raise RuntimeError(
                        "float_top is not idle: is the overlay loaded (fpga/board/load_overlay.py)?"
                    )
            else:
                raise ValueError(f"backend {backend!r}: pynq or xrt")
        self._bx = self._by = None  # scratch buffers for X and Y_tiled
        self._bxt = self._byt = None  # their torch float32 views (made once per allocation)
        self._slabs, self._open = [], []  # all weight slabs; (slab, bytes used) for those with room

    def _wr_reg(self, off: int, value: int) -> None:
        if self._regs is None or self._regs.get(off) != value:
            if self._regs is not None:
                self._regs[off] = value
            self._wr(off, value)

    def _wr64(self, off: int, value: int) -> None:
        self._wr_reg(off, value & 0xFFFFFFFF)
        self._wr_reg(off + 4, (value >> 32) & 0xFFFFFFFF)

    def _scratch(self, nx: int, ny: int):
        """Activation buffers (bytes), allocated once and grown on demand."""
        if self._bx is None or self._bx.array.size < nx:
            if self._bx is not None:
                self._bx.free()
            self._bx, self._bxt = self._alloc(max(nx, 1 << 20)), None
        if self._by is None or self._by.array.size < ny:
            if self._by is not None:
                self._by.free()
            self._by, self._byt = self._alloc(max(ny, 1 << 20)), None
        return self._bx, self._by

    def _views(self, nxf: int, nyf: int, is_t: bool):
        """Flat float32 views (numpy, or torch when the activations are tensors) of the scratch buffers."""
        bx, by = self._scratch(4 * nxf, 4 * nyf)
        if not is_t:
            return bx.array.view(np.float32), by.array.view(np.float32)
        torch = sys.modules["torch"]
        if self._bxt is None:
            self._bxt = torch.from_numpy(bx.array.view(np.float32))
        if self._byt is None:
            self._byt = torch.from_numpy(by.array.view(np.float32))
        return self._bxt, self._byt

    def upload(self, packed: np.ndarray):
        """Copy packed weights into DDR once; returns a SlabWeights (or the array when mocked).

        Layers are placed back to back (4 KB aligned) in 32 MB CMA slabs; a layer larger than a slab gets
        its own allocation. Each layer is flushed from the CPU cache right after it is written."""
        if self.mock in ("shim", "layout"):
            return packed
        nbytes = packed.size * 2
        if nbytes > SLAB_BYTES:
            b = self._alloc(nbytes)
            self._slabs.append(b)
            b.array[:nbytes].view(np.uint16)[:] = packed
            b.flush(0, nbytes)
            return SlabWeights(b, 0, nbytes)
        need = -(-nbytes // SLAB_ALIGN) * SLAB_ALIGN
        for i, (slab, off) in enumerate(self._open):  # first slab with room (less end waste than last-only)
            if off + need <= SLAB_BYTES:
                self._open[i] = (slab, off + need)
                break
        else:
            slab, off = self._alloc(SLAB_BYTES), 0
            self._slabs.append(slab)
            self._open.append((slab, need))
        slab.array[off : off + nbytes].view(np.uint16)[:] = packed
        slab.flush(off, nbytes)
        return SlabWeights(slab, off, nbytes)

    def free_weights(self) -> None:
        """Release every weight slab (a new policy load uploads all layers again)."""
        if self.mock in ("shim", "layout"):
            return
        for slab in self._slabs:
            slab.free()
        self._slabs, self._open = [], []

    def _run(self, wbuf, word_off: int, pw: int, mp: int, kc: int, n: int) -> None:
        base = wbuf.physical_address + 16 * word_off  # one word = 8 x uint16 = 16 bytes
        for p, reg in enumerate(REG_W):
            self._wr64(reg, base + 16 * p * pw)
        self._wr64(REG_XMEM, self._bx.physical_address)
        self._wr64(REG_YMEM, self._by.physical_address)
        self._wr_reg(REG_M, mp)
        self._wr_reg(REG_K, kc)
        self._wr_reg(REG_N, n)
        self._wr(REG_CTRL, 0x1)  # ap_start
        while not (self._rd(REG_CTRL) & 0x2):  # ap_done
            if self.poll_sleep:
                time.sleep(self.poll_sleep)
        status = self._rd(REG_RETURN)
        if status:
            raise RuntimeError(f"float engine status {status} (M={mp} K={kc} N={n})")

    def gemm(self, wbuf, layout, mp: int, x):
        """y[N][Mp] = x @ W^T: one engine call per K chunk, summed in chunk order on the ARM.

        x: float32 (N, K), a torch tensor or a numpy array; the result has the same kind."""
        return self.gemm_parts(wbuf, layout, mp, x, [(0, mp, None)])[0]

    def gemm_parts(self, wbuf, layout, mp: int, x, parts):
        """Output columns [32 g0, 32 g0 + m) of x @ W^T plus bias for each part (g0, m, bias): a list of
        new (N, m) arrays of x's kind (torch or numpy; bias of that kind or None). One engine run per K
        chunk serves every part; on the board each part is un-tiled straight out of the result buffer
        with its bias added in the same pass. x: (N, K), any strides (and any dtype for tensors)."""
        k = x.shape[1]
        t0 = time.perf_counter()
        is_t = not isinstance(x, np.ndarray)
        if self.mock in ("shim", "layout"):
            if is_t:
                x = x.numpy()
            if self.mock == "shim":
                y = self.lib(wbuf, mp, x)
            else:  # one chunk = 4 port arrays back to back, as on the board; chunk sums in chunk order
                y = None
                for k0, word_off, pw in layout:
                    kc = min(K_MAX, k - k0)
                    part = self.lib(wbuf[8 * word_off : 8 * (word_off + 4 * pw)], mp, x[:, k0 : k0 + kc])
                    y = part if y is None else np.add(y, part, out=y)
            outs = []
            for g0, m, bias in parts:
                o = y if (g0 == 0 and m == mp) else y[:, LANES * g0 : LANES * g0 + m].copy()
                if bias is not None:
                    o += np.asarray(bias)
                outs.append(sys.modules["torch"].from_numpy(o) if is_t else o)
        else:
            with self.lock:
                outs = self._gemm_hw(wbuf, layout, mp, x, parts, is_t)
        self.calls += 1
        self.fpga_s += time.perf_counter() - t0
        return outs

    def _gemm_hw(self, wbuf, layout, mp: int, x, parts, is_t: bool):
        n, k = x.shape
        groups, npad = mp // LANES, -(-n // NT) * NT
        ny = groups * npad * LANES if self.tiled_y else n * mp
        if is_t:
            torch = sys.modules["torch"]
            outs = [torch.empty((n, m), dtype=torch.float32) for _, m, _ in parts]
        else:
            outs = [np.empty((n, m), np.float32) for _, m, _ in parts]
        last, t = len(layout) - 1, self.t
        for ci, (k0, word_off, pw) in enumerate(layout):
            kc = min(K_MAX, k - k0)
            xb, yb = self._views(n * kc, ny, is_t)
            t0 = time.perf_counter()
            xv = xb[: n * kc].reshape(n, kc)  # strided or non-float32 x: copy_ converts on the way in
            if is_t:
                xv.copy_(x[:, k0 : k0 + kc])
            else:
                np.copyto(xv, x[:, k0 : k0 + kc])
            t1 = time.perf_counter()
            self._bx.flush(0, 4 * n * kc)
            t2 = time.perf_counter()
            self._run(wbuf, word_off, pw, mp, kc, n)
            t3 = time.perf_counter()
            self._by.invalidate(0, 4 * ny)
            t4 = time.perf_counter()
            if self.tiled_y:  # Y_tiled[g][npad][32] -> out[n][m] for every part, bias in the same pass
                yt = yb[:ny].reshape(groups, npad, LANES)
                for (g0, _m, bias), out in zip(parts, outs, strict=True):
                    _untile(yt, g0, out, bias if last == 0 else None, ci > 0)
            else:
                yr = yb[:ny].reshape(n, mp)
                for (g0, m, bias), out in zip(parts, outs, strict=True):
                    _assign(out, yr[:, LANES * g0 : LANES * g0 + m], bias if last == 0 else None, ci > 0)
            t5 = time.perf_counter()
            t["copy_in"] += t1 - t0
            t["flush"] += t2 - t1
            t["fpga"] += t3 - t2  # registers + the FPGA run
            t["invalidate"] += t4 - t3
            t["copy_out"] += t5 - t4
        if last > 0:  # several K chunks: ((p0 + p1) + p2) + bias, the order the ARM always used
            for (_, _, bias), out in zip(parts, outs, strict=True):
                if bias is not None:
                    out += bias
        return outs


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

    The earlier design for the KR260's two Pythons (PYNQ on 3.10, LeRobot on 3.12): the FPGA driver in
    its own process, activations copied through the socket. Engine(backend="xrt") replaces it."""

    def __init__(self, path: str = "/tmp/fpga-engine.sock"):
        import socket

        self.pf = _pack_float()
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.connect(path)
        self.lock = threading.Lock()  # the policy server answers gRPC calls from a thread pool
        self.calls, self.fpga_s = 0, 0.0
        self.mock = "remote"  # FpgaLinear: numpy in, numpy out

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

    def gemm_parts(self, wbuf: int, layout, mp: int, x: np.ndarray, parts) -> list[np.ndarray]:
        y = self.gemm(wbuf, layout, mp, x)
        outs = []
        for g0, m, bias in parts:
            o = y[:, LANES * g0 : LANES * g0 + m].copy()
            if bias is not None:
                o += np.asarray(bias)
            outs.append(o)
        return outs


class FpgaLinear:
    """One engine layer for one nn.Linear, or for several that take the same input (q/k/v, gate/up):
    their weights stacked along the output axis, each padded to 32 rows, packed and uploaded once.

    Each output lane is the same independent dot product whichever matrix it sits in (same K order, same
    partial sums), so a fused layer gives every member exactly the bits of its own layer. The first
    member called with an input runs the one engine call and keeps the other members' outputs for their
    calls on that same tensor (the model calls q_proj(h), k_proj(h), v_proj(h) in a row); a member called
    with a different input runs the fused call itself. Takes and returns torch tensors; the in-process
    engine copies the activations straight into the CMA buffer (any strides) and un-tiles the result with
    the bias added in the same pass; mock and remote engines go through numpy.

    memo: when the input is bit-identical to the previous call's (the expert's cross-attention k/v
    projections read the fixed prefix cache at every denoising step), the previous output is returned."""

    def __init__(self, weights, biases, eng, memo: bool = False):
        import torch

        if isinstance(weights, np.ndarray):
            weights, biases = [weights], [biases]
        pf = eng.pf
        self.k = weights[0].shape[1]
        self.ms = [w.shape[0] for w in weights]
        mps = [-(-m // LANES) * LANES for m in self.ms]
        self.mp = sum(mps)
        g0s = [int(g) for g in np.cumsum([0] + mps[:-1]) // LANES]
        bits = np.zeros((self.mp, self.k), np.uint16)
        for w, g0, m in zip(weights, g0s, self.ms, strict=True):
            if w.shape[1] != self.k:
                raise ValueError("fused layers must share in_features")
            bits[LANES * g0 : LANES * g0 + m] = pf.to_bf16_bits(w)
        packed, _ = pf.pack(bits)
        self.layout = chunk_layout(self.mp, self.k)
        self.eng = eng
        self.numpy_io = getattr(eng, "mock", None) in ("shim", "layout", "remote")
        b_np = [None if b is None else np.ascontiguousarray(b, np.float32) for b in biases]
        b_t = [None if b is None else torch.from_numpy(b) for b in b_np]
        self.parts = list(zip(g0s, self.ms, b_np if self.numpy_io else b_t, strict=True))
        self.bias = b_t[0]  # of the first member (kept for callers that read it)
        self.m = self.ms[0]
        self.wbuf = eng.upload(packed)
        self.memo, self._memo, self.memo_hits = memo, None, 0
        self._stash = {}  # id(input) -> [input, its version, [outputs still to be collected]]
        self.lock = threading.Lock()

    def _compute(self, x):
        import torch

        x2 = x.reshape(-1, self.k)
        if x2.dtype != torch.float32:
            x2 = x2.to(torch.float32)
        memo = self._memo if self.memo else None
        if memo is not None and memo[0].shape == x2.shape and torch.equal(memo[0], x2):
            self.memo_hits += 1
            return [o.clone() for o in memo[1]]
        if self.numpy_io:
            outs = self.eng.gemm_parts(self.wbuf, self.layout, self.mp, x2.detach().numpy(), self.parts)
            outs = [torch.from_numpy(o) for o in outs]
        else:
            outs = self.eng.gemm_parts(self.wbuf, self.layout, self.mp, x2, self.parts)
        if self.memo:
            self._memo = (x2.clone(), [o.clone() for o in outs])
        return outs

    def __call__(self, x, i: int = 0):
        shape = x.shape
        if len(self.ms) == 1:
            y = self._compute(x)[0]
        else:
            with self.lock:
                ent = self._stash.get(id(x))
                hit = ent is not None and ent[0] is x and ent[1] == x._version and ent[2][i] is not None
                if hit:
                    y, ent[2][i] = ent[2][i], None
                    if all(o is None for o in ent[2]):
                        del self._stash[id(x)]
            if not hit:
                outs = self._compute(x)
                y, outs[i] = outs[i], None
                with self.lock:
                    self._stash[id(x)] = [x, x._version, outs]
                    while len(self._stash) > 4:  # members never called with this input: drop the oldest
                        del self._stash[next(iter(self._stash))]
        return y.reshape(*shape[:-1], self.ms[i])


def fusion_groups(model) -> list[list[str]]:
    """Names of nn.Linear modules that take the same input tensor, by module structure: q_proj/k_proj/
    v_proj with equal in_features (SigLIP, the LM, the expert's self-attention layers; the expert's
    cross-attention k/v take the prefix cache, so their in_features differ from q's) and gate_proj/
    up_proj of a gated MLP (one call, outputs un-tiled into q, k, v / gate, up)."""
    from torch import nn

    groups = []
    for name, mod in model.named_modules():
        for attrs in (("q_proj", "k_proj", "v_proj"), ("gate_proj", "up_proj")):
            lins = [getattr(mod, a, None) for a in attrs]
            if all(isinstance(li, nn.Linear) for li in lins) and len({li.in_features for li in lins}) == 1:
                groups.append([f"{name}.{a}" if name else a for a in attrs])
    return groups


def memo_modules(model) -> list[str]:
    """SmolVLA's expert k_proj/v_proj in cross-attention layers: they project the prefix K/V cache,
    which is the same tensor at every denoising step (smolvlm_with_expert.forward_cross_attn_layer)."""
    names = []
    for name, mod in model.named_modules():
        expert, mode = getattr(mod, "lm_expert", None), getattr(mod, "attention_mode", "")
        if expert is None or "cross" not in str(mode):
            continue
        every = int(getattr(mod, "self_attn_every_n_layers", -1))
        n_vlm, n_exp = int(mod.num_vlm_layers), int(mod.num_expert_layers)
        multiple_of = n_vlm // n_exp if n_exp else 0
        for j in range(n_exp):  # expert layer j runs at VLM layer index j * multiple_of (get_model_layers)
            layer_idx = j * multiple_of if multiple_of > 0 else j
            if every > 0 and layer_idx % every == 0:
                continue  # self-attention layer: k/v take the suffix
            prefix = f"{name}.lm_expert" if name else "lm_expert"
            names += [f"{prefix}.layers.{j}.self_attn.k_proj", f"{prefix}.layers.{j}.self_attn.v_proj"]
    return names


def fpga_linears(model, eng, free_weights: bool = True, fuse: bool = False, memo: bool = False) -> int:
    """Swap every torch.nn.Linear in `model` for an FpgaLinear. Returns how many were swapped.

    With free_weights, each layer's PyTorch weight is released once it is packed, so only the DDR
    copy remains (the board has 4 GB). fuse: modules that take the same input (fusion_groups) share one
    engine call. memo: the modules of memo_modules return their previous output on a bit-identical input.
    Both leave every number unchanged; both are off unless asked for."""
    import types

    import torch
    from torch import nn

    linears = {name: mod for name, mod in model.named_modules() if isinstance(mod, nn.Linear)}
    groups = [g for g in fusion_groups(model) if all(n in linears for n in g)] if fuse else []
    grouped = {n for g in groups for n in g}
    groups += [[n] for n in linears if n not in grouped]
    memo_names = set(memo_modules(model)) if memo else set()
    count = 0
    for names in groups:
        mods = [linears[n] for n in names]
        ws = [m.weight.detach().float().cpu().numpy() for m in mods]
        bs = [None if m.bias is None else m.bias.detach().float().cpu().numpy() for m in mods]
        fl = FpgaLinear(ws, bs, eng, memo=any(n in memo_names for n in names))
        del ws
        for i, mod in enumerate(mods):
            if free_weights:
                mod.weight.data = torch.empty(0, dtype=mod.weight.dtype, device=mod.weight.device)

            def fwd(self, x, _fl=fl, _i=i):
                return _fl(x, _i).to(x.dtype)

            mod.forward = types.MethodType(fwd, mod)
            count += 1
    return count


_GELU_C2 = 2 * 0.7978845608028654  # 2 * sqrt(2 / pi)


def fast_tanh_gelu(model) -> int:
    """Evaluate SigLIP's tanh-GELU as x * sigmoid(2 * sqrt(2/pi) * (x + 0.044715 x^3)): the same
    function (0.5 (1 + tanh u) == sigmoid 2u), in FP32, but torch 2.8's aarch64 kernel for
    F.gelu(approximate="tanh") runs at 12 M elements/s on the KR260 against ~34 M/s for this (measured:
    259 vs 91 ms per 1024 x 3072; max |difference| 4.8e-7). Returns how many modules were patched."""
    import types

    import torch
    from torch import nn

    def forward(self, x):
        u = x * x
        u.mul_(x).mul_(0.044715).add_(x).mul_(_GELU_C2)
        return torch.sigmoid_(u).mul_(x)

    count = 0
    for _, mod in model.named_modules():
        is_tanh_gelu = isinstance(mod, nn.GELU) and getattr(mod, "approximate", "") == "tanh"
        if is_tanh_gelu or type(mod).__name__ in ("PytorchGELUTanh", "GELUTanh"):
            mod.forward = types.MethodType(forward, mod)
            count += 1
    return count


def overlap_image_embedding(
    policy, eng, poll_sleep: float = 50e-6, threads_per_image: int | None = None
) -> None:
    """Encode the camera images in parallel threads, so one image's attention/GELU/norms on the ARM
    overlap the other image's linear layers on the FPGA. SmolVLA's embed_prefix calls embed_image
    once per camera, in order; this computes those calls up front (one thread each, the engine
    serialized by its lock) and hands the results to the unchanged embed_prefix. The numbers are the
    same: each image is encoded by the same code on the same data. While a thread waits for the FPGA
    it sleeps poll_sleep between register reads so the other thread's torch ops get the GIL."""
    import os
    import threading
    import types

    import torch

    model = policy.model
    vwe = model.vlm_with_expert
    orig_embed_prefix = model.embed_prefix
    orig_embed_image = vwe.embed_image
    eng.poll_sleep = poll_sleep
    total = torch.get_num_threads()
    # torch's OpenMP pool is per master thread: two image threads with the full count oversubscribe the
    # four cores (spin-waiting workers), so each worker gets its share
    per_image = threads_per_image or max(1, (os.cpu_count() or total) // 2)

    def embed_prefix(self, images, img_masks, lang_tokens, lang_masks, state=None):
        results, errors = [None] * len(images), []

        def work(i, img):
            try:
                torch.set_num_threads(per_image)
                with torch.no_grad():
                    results[i] = orig_embed_image(img)
            except Exception as e:  # noqa: BLE001 - re-raised in the caller
                errors.append(e)

        threads = [threading.Thread(target=work, args=(i, img)) for i, img in enumerate(images)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        torch.set_num_threads(total)
        if errors:
            raise errors[0]
        it = iter(results)
        vwe.embed_image = lambda img: next(it)
        try:
            return orig_embed_prefix(images, img_masks, lang_tokens, lang_masks, state=state)
        finally:
            vwe.embed_image = orig_embed_image

    model.embed_prefix = types.MethodType(embed_prefix, model)
