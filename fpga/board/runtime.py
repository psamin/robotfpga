"""KR260 runtime for the full-precision SmolVLA linear engine (fpga/hls, issue #148).

SmolVLA runs on the board's ARM cores in PyTorch; every nn.Linear is swapped for FpgaLinear, which
keeps its BF16 weights in DDR (packed for the four HP ports by fpga/hls/pack_float.py) and sends
FP32 activations to the FPGA engine. The ARM adds the bias and runs everything nonlinear.

    from board.stream.runtime import Engine, fpga_linears
    eng = Engine("float.bit")          # overlay built from fpga/hls/run_float_hls.tcl
    fpga_linears(policy, eng)          # policy = SmolVLAPolicy.from_pretrained(...).float()
    actions = policy.predict_action_chunk(batch)

Without a board, `Engine(mock=True)` runs the identical C core as a shared library
(fpga/hls/build/libfloat_linear.so), which is what policy/smolvla_fpga.py validates.
UNTESTED ON HARDWARE until an overlay exists. Run `print(ol.float_top_0.register_map)` once and
adjust the pointer names below if Vivado renames them.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

K_MAX = 4096
_PF = Path(__file__).resolve().parents[2] / "fpga" / "hls" / "pack_float.py"


def _pack_float():
    spec = importlib.util.spec_from_file_location("pack_float", _PF)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Engine:
    """PYNQ driver for float_top, or (mock=True) the same core as a host library."""

    def __init__(self, bitfile: str | None = None, mock: bool = False):
        self.pf = _pack_float()
        self.mock = mock
        if mock:
            self.lib = self.pf.Lib()
            return
        from pynq import Overlay, allocate

        self.allocate = allocate
        self.ol = Overlay(bitfile)
        self.ip = self.ol.float_top_0
        self.regs = self.ip.register_map

    def _set_ptr(self, name: str, addr: int) -> None:
        if hasattr(self.regs, name):
            setattr(self.regs, name, addr)
        else:  # HLS splits 64-bit pointers into _1 (low) and _2 (high)
            setattr(self.regs, f"{name}_1", addr & 0xFFFFFFFF)
            setattr(self.regs, f"{name}_2", addr >> 32)

    def upload(self, packed: np.ndarray):
        """Copy packed weights into a DDR buffer once; returns the buffer."""
        b = self.allocate(shape=(packed.size,), dtype=np.uint16)
        b[:] = packed
        b.flush()
        return b

    def gemm(self, wbuf, chunk_layout, mp: int, x: np.ndarray) -> np.ndarray:
        """y[N][Mp] = x @ W^T on the FPGA, one call per 4096-wide K chunk, summed on the ARM."""
        n, k = x.shape
        y = np.zeros((n, mp), np.float32)
        for k0, word_off, port_words in chunk_layout:
            kc = min(K_MAX, k - k0)
            bx = self.allocate(shape=(n * kc,), dtype=np.float32)
            by = self.allocate(shape=(n * mp,), dtype=np.float32)
            bx[:] = np.ascontiguousarray(x[:, k0 : k0 + kc]).reshape(-1)
            bx.flush()
            for p in range(4):  # word = 8 uint16 = 16 bytes
                self._set_ptr(f"w{p}", wbuf.physical_address + 16 * (word_off + p * port_words))
            self._set_ptr("xmem", bx.physical_address)
            self._set_ptr("ymem", by.physical_address)
            self.regs.M, self.regs.K, self.regs.N = mp, kc, n
            self.ip.write(0x00, 0x1)  # ap_start
            while not (self.ip.read(0x00) & 0x2):  # ap_done
                pass
            if int(self.regs.ap_return):
                raise RuntimeError(f"float engine status {int(self.regs.ap_return)}")
            by.invalidate()
            y += np.array(by).reshape(n, mp)
            bx.freebuffer()
            by.freebuffer()
        return y


class FpgaLinear:
    """Callable replacement for one nn.Linear: BF16 weights in DDR, FP32 math on the FPGA."""

    def __init__(self, module, eng: Engine):
        pf = eng.pf
        w = module.weight.detach().float().cpu().numpy()
        self.m, self.k = w.shape
        self.packed, self.mp = pf.pack(pf.to_bf16_bits(w))
        self.bias = None if module.bias is None else module.bias.detach().float().cpu().numpy()
        self.eng = eng
        groups = self.mp // 32
        self.layout, off = [], 0
        for k0 in range(0, self.k, K_MAX):
            pw = groups * min(K_MAX, self.k - k0)
            self.layout.append((k0, off, pw))
            off += 4 * pw
        self.wbuf = None if eng.mock else eng.upload(self.packed)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        shape = x.shape
        x2 = np.ascontiguousarray(x.reshape(-1, self.k), np.float32)
        if self.eng.mock:
            y = self.eng.lib(self.packed, self.mp, x2)
        else:
            y = self.eng.gemm(self.wbuf, self.layout, self.mp, x2)
        y = y[:, : self.m]
        if self.bias is not None:
            y = y + self.bias
        return y.reshape(*shape[:-1], self.m)


def fpga_linears(model, eng: Engine) -> int:
    """Swap every torch.nn.Linear in `model` for an FpgaLinear. Returns how many were swapped."""
    import types

    import torch
    from torch import nn

    count = 0
    for _, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            fl = FpgaLinear(mod, eng)

            def fwd(self, x, _fl=fl):
                y = _fl(x.detach().float().cpu().numpy())
                return torch.from_numpy(y).to(x.device, x.dtype)

            mod.forward = types.MethodType(fwd, mod)
            count += 1
    return count
