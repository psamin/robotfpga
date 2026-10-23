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

Buffers are cacheable CMA, flushed/invalidated over the bytes a call uses (PYNQ's default is
uncached memory, which reads at ~120 MB/s: 8.6 s of a chunk went into copying results out).

Memory: the KR260 has 4 GB. After a layer's weights are packed into DDR, its PyTorch weight tensor is
freed, so the board holds one copy (0.71 GB of BF16 linear weights for SmolVLA). Weights are packed
into 32 MB CMA slabs; check `grep Cma /proc/meminfo` (cma=1000M on the board).
"""

from __future__ import annotations

import importlib.util
import os
import sys
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


class SlabWeights:
    """One layer's packed weights inside a shared CMA slab: what Engine.gemm needs from a weight buffer.

    One CMA allocation per layer wasted ~300 MB of the KR260's 1,000 MB CMA on alignment (the kernel aligns
    each allocation to up to 1 MB): 708 MB of SmolVLA weights did not fit. Slabs waste < 32 MB in total."""

    def __init__(self, slab, offset: int, nbytes: int):
        self.slab, self.offset, self.nbytes = slab, offset, nbytes
        self.physical_address = slab.physical_address + offset


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
    ):
        import threading

        self.pf = _pack_float()
        self.mock = {True: "shim"}.get(mock, mock) or None
        self.calls, self.fpga_s = 0, 0.0
        self.lock = threading.Lock()  # one call at a time when several threads share the engine
        self.poll_sleep = 0.0  # > 0: yield the GIL while waiting for ap_done (overlap_image_embedding)
        self.t = dict.fromkeys(("copy_in", "flush", "fpga", "invalidate", "copy_out"), 0.0)
        # float_linear.h since v4 writes Y group-major (one 4 KB burst per tile-group); FSTREAM_LEGACY_Y=1
        # for bitstreams built from v3 (float_tp4.bit), which wrote Y row-major
        self.tiled_y = os.environ.get("FSTREAM_LEGACY_Y") != "1" if tiled_y is None else tiled_y
        if self.mock:
            self.lib = self.pf.Lib()
            return
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
        self._bx = self._by = None
        self._slabs, self._open = [], []  # all weight slabs; (slab, bytes used) for those with room

    def _wr64(self, off: int, value: int) -> None:
        self._wr(off, value & 0xFFFFFFFF)
        self._wr(off + 4, (value >> 32) & 0xFFFFFFFF)

    def _scratch(self, nx: int, ny: int):
        """Activation buffers (bytes), allocated once and grown on demand."""
        if self._bx is None or self._bx.array.size < nx:
            if self._bx is not None:
                self._bx.free()
            self._bx = self._alloc(max(nx, 1 << 20))
        if self._by is None or self._by.array.size < ny:
            if self._by is not None:
                self._by.free()
            self._by = self._alloc(max(ny, 1 << 20))
        return self._bx, self._by

    def upload(self, packed: np.ndarray):
        """Copy packed weights into DDR once; returns a SlabWeights (or the array when mocked).

        Layers are placed back to back (4 KB aligned) in 32 MB CMA slabs; a layer larger than a slab gets
        its own allocation. Each layer is flushed from the CPU cache right after it is written."""
        if self.mock:
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
        if self.mock:
            return
        for slab in self._slabs:
            slab.free()
        self._slabs, self._open = [], []

    def _run(self, wbuf, word_off: int, pw: int, bx, by, mp: int, kc: int, n: int) -> None:
        for p, reg in enumerate(REG_W):  # one word = 8 x uint16 = 16 bytes
            self._wr64(reg, wbuf.physical_address + 16 * (word_off + p * pw))
        self._wr64(REG_XMEM, bx.physical_address)
        self._wr64(REG_YMEM, by.physical_address)
        self._wr(REG_M, mp)
        self._wr(REG_K, kc)
        self._wr(REG_N, n)
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
        n, k = x.shape
        t0 = time.perf_counter()
        if self.mock == "shim":
            y = self.lib(wbuf, mp, x)
        elif self.mock == "layout":  # one chunk = 4 port arrays back to back, as on the board
            y = np.zeros((n, mp), np.float32)
            for k0, word_off, pw in layout:
                kc = min(K_MAX, k - k0)
                xc = np.ascontiguousarray(x[:, k0 : k0 + kc])
                y += self.lib(wbuf[8 * word_off : 8 * (word_off + 4 * pw)], mp, xc)
        else:
            with self.lock:
                y = self._gemm_hw(wbuf, layout, mp, x, n, k)
        self.calls += 1
        self.fpga_s += time.perf_counter() - t0
        return y

    def _gemm_hw(self, wbuf, layout, mp: int, x, n: int, k: int):
        torch = sys.modules.get("torch")
        is_t = torch is not None and isinstance(x, torch.Tensor)
        groups, npad = mp // LANES, -(-n // NT) * NT
        ny = groups * npad * LANES if self.tiled_y else n * mp
        t = self.t
        y = None
        for k0, word_off, pw in layout:
            kc = min(K_MAX, k - k0)
            bx, by = self._scratch(n * kc * 4, ny * 4)
            t0 = time.perf_counter()
            xv = bx.array[: n * kc * 4].view(np.float32).reshape(n, kc)
            if is_t:
                torch.from_numpy(xv).copy_(x[:, k0 : k0 + kc])
            else:
                np.copyto(xv, x[:, k0 : k0 + kc])
            t1 = time.perf_counter()
            bx.flush(0, n * kc * 4)
            t2 = time.perf_counter()
            self._run(wbuf, word_off, pw, bx, by, mp, kc, n)
            t3 = time.perf_counter()
            by.invalidate(0, ny * 4)
            t4 = time.perf_counter()
            yv = by.array[: ny * 4].view(np.float32)
            if self.tiled_y:  # Y_tiled[g][npad][32] -> [n][g * 32 + l], always a copy out of the buffer
                yv = yv.reshape(groups, npad, LANES)[:, :n, :]  # (reshape of a permuted view with one
                if is_t:  # group is a view, which the next call would overwrite)
                    part = torch.empty((n, mp), dtype=torch.float32)
                    part.view(n, groups, LANES).copy_(torch.from_numpy(yv).permute(1, 0, 2))
                else:
                    part = np.empty((n, mp), np.float32)
                    part.reshape(n, groups, LANES)[...] = yv.transpose(1, 0, 2)
            else:
                yv = yv.reshape(n, mp)
                part = torch.from_numpy(yv).clone() if is_t else yv.copy()
            if y is None:
                y = part
            elif is_t:
                y.add_(part)
            else:
                y += part
            t5 = time.perf_counter()
            t["copy_in"] += t1 - t0
            t["flush"] += t2 - t1
            t["fpga"] += t3 - t2  # registers + the FPGA run
            t["invalidate"] += t4 - t3
            t["copy_out"] += t5 - t4
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

    The earlier design for the KR260's two Pythons (PYNQ on 3.10, LeRobot on 3.12): the FPGA driver in
    its own process, activations copied through the socket. Engine(backend="xrt") replaces it."""

    def __init__(self, path: str = "/tmp/fpga-engine.sock"):
        import socket
        import threading

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


class FpgaLinear:
    """Callable replacement for one nn.Linear: BF16 weights in DDR, FP32 math on the FPGA.

    Takes and returns torch tensors. With the in-process engine the activations go straight from the
    tensor into the CMA buffer and the result comes back as a tensor (the bias is added in place by
    torch); mock and remote engines go through numpy."""

    def __init__(self, weight: np.ndarray, bias: np.ndarray | None, eng):
        import torch

        pf = eng.pf
        self.m, self.k = weight.shape
        packed, self.mp = pf.pack(pf.to_bf16_bits(weight.astype(np.float32)))
        self.layout = chunk_layout(self.mp, self.k)
        self.bias = None if bias is None else torch.from_numpy(np.ascontiguousarray(bias, np.float32))
        self.eng = eng
        self.numpy_io = bool(getattr(eng, "mock", None))
        self.wbuf = eng.upload(packed)

    def __call__(self, x):
        import torch

        shape = x.shape
        x2 = x.detach().to(torch.float32).reshape(-1, self.k).contiguous()
        if self.numpy_io:
            y = self.eng.gemm(self.wbuf, self.layout, self.mp, x2.numpy())
            y = torch.from_numpy(np.ascontiguousarray(y))
        else:
            y = self.eng.gemm(self.wbuf, self.layout, self.mp, x2)
        y = y[:, : self.m]
        if self.bias is not None:
            y = y.add_(self.bias) if self.mp == self.m else y + self.bias
        return y.reshape(*shape[:-1], self.m)


def fpga_linears(model, eng, free_weights: bool = True) -> int:
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
            return _fl(x).to(x.dtype)

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
