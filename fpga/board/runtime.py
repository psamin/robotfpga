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
            from pynq.ps import Clocks

            Clocks.fclk0_mhz = clock_mhz
        self.ip = self.ol.float_top_0
        self.regs = self.ip.register_map
        self._bx = self._by = None

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
        """Copy packed weights into a DDR buffer once; returns the buffer (or the array when mocked)."""
        if self.mock:
            return packed
        b = self.allocate(shape=(packed.size,), dtype=np.uint16)
        b[:] = packed
        b.flush()
        return b

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
