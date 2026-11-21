"""Golden executor of contract 1.0 command lists (fpga/kernels/CONTRACT.md), in torch + numpy.

The walk is seq_core.h's walk() line for line: fetch (256 B), contract.decode -> BAD_*, contract.validate(full=False)
(= ctr::validate_hw) -> MODE_ERROR, END / YIELD / STEP / LIMIT / NO_LEGACY, the 3-word status block (cycles 0).
check_twin.py replays every start of tb_seq.cpp through it and compares return, status words 0-9 and the dispatch
trace.

Every opcode follows the contract's layouts (C1: rows with pitch and item stride, scale vectors with pad lanes 1.0,
segments, NP zero rows, the KW / VW weight tilings, GEMM row groups) and arithmetic (C2):
  - int16 outputs: contract.quant_rows (the one quantizer; shift p4 on the stored scale; zero rows 1.0),
  - GEMM: exact integer sums (float64 holds |acc| < 2^53), then contract.gemm_out (FP32 steps, FTZ, sign kept),
  - the softmax's own P rule (fe="kernels") or the quantizer on FP32 torch P (fe="torch").
Front ends (the FP32 value that reaches the quantizer):
  fe="torch":   FP32 torch (LayerNorm / RMSNorm as the policy computes them, tanh-GELU, SiLU, softmax, rotation with
                the table's Q1.17 cos / sin);
  fe="kernels": the kernels' bit-exact emulators, as conformed to contract 1.0: norm_residual/nr_emul.kernel
                (statistics with correctly rounded mean / var + affine: its z), gelu_silu/gs_emu.act (NSB 9),
                softmax/smx_emu.kernel (q and 1/sum), rope/rope_emu.rope_rows (MODE_F32 rotation) - gate T-P1.
exact=True: no quantization at all. Every int16 buffer has an FP32 shadow (one float per int16 element, same
addresses); quantizers store x itself with scale 2^p4 (1.0 for zero rows), GEMM reads the shadows. With fe="torch"
this runs the policy through the list in FP32: it checks the list (addresses, pitches, tilings, order, aliasing,
the folded constants), not the arithmetic.
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

KDIR = Path(__file__).resolve().parents[1]
for sub in ("", "norm_residual", "gelu_silu", "softmax", "rope"):
    p = str(KDIR / sub) if sub else str(KDIR)
    if p not in sys.path:
        sys.path.insert(0, p)
import contract as K  # noqa: E402
from contract import C, SLOTS  # noqa: E402

U_NONE, U_GEMM, U_ROW, U_SMX = 0, 1, 2, 3
ROW_OPS = (C.OP_QUANT_X, C.OP_QUANT_W, C.OP_QUANT_WT, C.OP_NORM, C.OP_GELU, C.OP_SILU_MUL, C.OP_ROPE, C.OP_COPY)


def unit_of(op: int) -> int:
    if op == C.OP_GEMM:
        return U_GEMM
    if op == C.OP_SOFTMAX:
        return U_SMX
    return U_ROW if op in ROW_OPS else U_NONE


def trace_line(c: K.Cmd, unit: int) -> str:
    """The dispatch line tb_seq.cpp prints (trace_line there)."""
    s = [f"op={c.op:02x} unit={unit} flags={c.flags:04x} tag={c.tag:08x} M={c.M} K={c.K} N={c.N} B={c.B}"]
    for n in SLOTS:
        sl = c.slot(n)
        s.append(f"{sl.addr:x}/{sl.pitch:x}/{sl.stride:x}")
    s += [f"{c.par(i):x}" for i in range(C.NPARAMS)]
    return " ".join(s)


def _i32(v: int) -> int:
    v &= 0xFFFFFFFF
    return v - (1 << 32) if v >> 31 else v


# ---------------------------------------------------------------- memory
class Memory:
    """Physical memory as regions of bytes (uint8 tensors); optionally an FP32 shadow per region (exact mode):
    shadow element (a - base) // 2 belongs to the int16 element at byte address a."""

    def __init__(self, shadow: bool = False):
        self.regions: list[tuple[int, torch.Tensor, torch.Tensor | None]] = []
        self.shadow = shadow

    def add_region(self, base: int, size: int) -> None:
        size = (size + 15) // 16 * 16
        sh = torch.zeros(size // 2, dtype=torch.float32) if self.shadow else None
        self.regions.append((base, torch.zeros(size, dtype=torch.uint8), sh))

    def _find(self, addr: int, extent: int):
        for base, t, sh in self.regions:
            if base <= addr and addr + extent <= base + t.numel():
                return base, t, sh
        raise IndexError(f"access 0x{addr:x} + {extent} outside every region")

    def write(self, addr: int, raw: bytes) -> None:
        base, t, _ = self._find(addr, len(raw))
        t[addr - base : addr - base + len(raw)] = torch.frombuffer(bytearray(raw), dtype=torch.uint8)

    def read(self, addr: int, n: int) -> bytes:
        base, t, _ = self._find(addr, n)
        return bytes(t[addr - base : addr - base + n].numpy())

    def view(self, addr: int, shape, strides_b, dtype=torch.float32) -> torch.Tensor:
        """Strided view (byte strides, 0 = broadcast), bounds-checked."""
        isz = torch.empty(0, dtype=dtype).element_size()
        ext = isz + sum((n - 1) * s for n, s in zip(shape, strides_b))
        base, t, _ = self._find(addr, ext)
        off = addr - base
        assert off % isz == 0 and all(s % isz == 0 for s in strides_b), (hex(addr), strides_b, isz)
        return torch.as_strided(t.view(dtype), tuple(shape), tuple(s // isz for s in strides_b), off // isz)

    def sview(self, addr: int, shape, strides_b) -> torch.Tensor:
        """The FP32 shadow of int16 elements (byte strides of the int16 layout)."""
        ext = 2 + sum((n - 1) * s for n, s in zip(shape, strides_b))
        base, _, sh = self._find(addr, ext)
        assert sh is not None and (addr - base) % 2 == 0 and all(s % 2 == 0 for s in strides_b)
        return torch.as_strided(sh, tuple(shape), tuple(s // 2 for s in strides_b), (addr - base) // 2)


# ---------------------------------------------------------------- the tilings
def tile_words(w: torch.Tensor) -> torch.Tensor:
    """[Mp][K] -> the 4 port arrays [4][Mp/32][K][8]: port p, group g, word k, lane l = w[32 g + 8 p + l][k]."""
    mp, k = w.shape
    return w.reshape(mp // 32, 4, 8, k).permute(1, 0, 3, 2).contiguous()


def untile_words(a: torch.Tensor) -> torch.Tensor:
    """[4][G][K][8] -> [32 G][K]."""
    _, g, k, _ = a.shape
    return a.permute(1, 0, 3, 2).reshape(32 * g, k)


class Executor:
    """One instance = one IP: `starts` is the IP's sequencer start counter."""

    def __init__(self, mem: Memory, *, exact: bool = False, fe: str = "torch", dry: bool = False):
        assert fe in ("torch", "kernels")
        self.mem, self.exact, self.fe, self.dry = mem, exact, fe, dry
        self.starts = 0
        self.trace: list[str] | None = None  # set to a list to record every unit call
        self.fault = 0  # the next unit call returns this ME code (tb_seq twin replay)
        self.counts: dict[str, int] = {}
        if fe == "kernels":
            import gs_emu
            import nr_emul
            import rope_emu
            import smx_emu

            self.gs, self.nr, self.rope, self.smx = gs_emu, nr_emul, rope_emu, smx_emu

    # ------------------------------------------------------------ the walk (seq_core.h walk())
    def start(self, M: int, Kreg: int, N: int) -> int:
        M, Kreg, N = M & 0xFFFFFFFF, Kreg & 0xFFFFFFFF, N & 0xFFFFFFFF
        if M >> 24 != C.MODE_MAGIC:
            return C.ST_NO_LEGACY
        self.starts = (self.starts + 1) & 0xFFFFFFFF
        step = (M >> C.MODE_STEP_BIT) & 1
        lst = Kreg << 4
        idx = M & C.MAX_INDEX
        code, tag, done, merr, lastop, last_fin = C.ST_END, 0, 0, C.ME_OK, 0, 0xFFFFFFFF
        while True:
            raw = self.mem.read(lst + idx * C.REC_BYTES, C.REC_BYTES)
            f = struct.unpack("<64I", raw)
            st, c = K.decode(raw)
            me = C.ME_OK
            if st == C.ST_END:
                me = K.validate(c, full=False)
                st = C.ST_MODE_ERROR if me else 0
            if st:
                code, merr, tag, lastop = st, me, f[C.FLD_TAG], f[C.FLD_OP] & 0xFF
                break
            if c.op == C.OP_END:
                code, tag, lastop = C.ST_END, c.tag, c.op
                break
            r = self.dispatch(c)
            done += 1
            tag, lastop = c.tag, c.op
            if r:
                code, merr = C.ST_MODE_ERROR, r
                break
            last_fin = idx
            idx += 1
            if c.flags & C.F_YIELD:
                code = C.ST_YIELD
                break
            if step:
                code = C.ST_STEP
                break
            if idx > C.MAX_INDEX:
                code = C.ST_LIMIT
                break
        if N:
            w = [code, idx, tag, done, merr, C.STATUS_MAGIC, self.starts, lastop, C.CONTRACT_VERSION, last_fin, 0, 0]
            self.mem.write(N << 4, struct.pack("<12I", *[v & 0xFFFFFFFF for v in w]))
        return ((idx << 8) | code) & 0xFFFFFFFF

    def run_list(self, list_addr: int, status_addr: int, first: int = 0) -> tuple[int, int, int]:
        """Start at `first`, restart after every YIELD (as the ARM does); returns (code, next, starts)."""
        idx, starts = first, 0
        while True:
            r = self.start(K.start_regs(list_addr, status_addr, idx)[C.REG_M], list_addr >> 4, status_addr >> 4)
            starts += 1
            idx, code = r >> 8, r & 0xFF
            if code != C.ST_YIELD:
                return code, idx, starts

    def dispatch(self, c: K.Cmd) -> int:
        u = unit_of(c.op)
        if u == U_NONE:
            return C.ME_OK
        if self.trace is not None:
            self.trace.append(trace_line(c, u))
        if self.fault:
            r, self.fault = self.fault, 0
            return r
        if self.dry:
            return C.ME_OK
        name = C.opnames[c.op]
        self.counts[name] = self.counts.get(name, 0) + 1
        getattr(self, "op_" + name.lower())(c)
        return C.ME_OK

    # ------------------------------------------------------------ operand access
    def rows_f32(self, s: K.Slot, b: int, n: int, k: int) -> torch.Tensor:
        return self.mem.view(s.addr + b * s.stride, (n, k), (s.pitch, 4), torch.float32)

    def rows_i16(self, s: K.Slot, b: int, n: int, k: int) -> torch.Tensor:
        """int16 rows as float64 (exact mode: their FP32 shadow)."""
        if self.exact:
            return self.mem.sview(s.addr + b * s.stride, (n, k), (s.pitch, 2)).double()
        return self.mem.view(s.addr + b * s.stride, (n, k), (s.pitch, 2), torch.int16).double()

    def vec_f32(self, s: K.Slot, b: int, n: int) -> torch.Tensor:
        return self.mem.view(s.addr + b * s.stride, (n,), (4,), torch.float32)

    def vec_i32(self, s: K.Slot, b: int, n: int) -> torch.Tensor:
        return self.mem.view(s.addr + b * s.stride, (n,), (4,), torch.int32)

    def weight(self, s: K.Slot, b: int, mp: int, k: int) -> torch.Tensor:
        """[Mp][K] float64 from the port tiling (port p array at addr + p pitch + b stride)."""
        g = mp // 32
        arr = []
        for p in range(4):
            a = s.addr + b * s.stride + p * s.pitch
            if self.exact:
                arr.append(self.mem.sview(a, (g, k, 8), (k * 16, 16, 2)).double())
            else:
                arr.append(self.mem.view(a, (g, k, 8), (k * 16, 16, 2), torch.int16).double())
        return untile_words(torch.stack(arr, 0))

    def put_tiled(self, s: K.Slot, b: int, off: int, q: np.ndarray, x: torch.Tensor | None) -> None:
        """Write a [rows][Kb] blob (rows % 32 == 0) in the port tiling at s.addr + b stride + off (port p at
        + p pitch). q: int16 values (int16 mode); x: FP32 values (exact mode, the shadow)."""
        if self.exact:
            t = tile_words(x.float())
            for p in range(4):
                g, kb, _ = t[p].shape
                self.mem.sview(s.addr + b * s.stride + off + p * s.pitch, (g, kb, 8), (kb * 16, 16, 2)).copy_(t[p])
        else:
            t = tile_words(torch.from_numpy(np.ascontiguousarray(q)))
            for p in range(4):
                self.mem.write(s.addr + b * s.stride + off + p * s.pitch, t[p].numpy().tobytes())

    # ------------------------------------------------------------ quantizer + writers
    def quant(self, x: torch.Tensor, shift: int):
        """The contract quantizer over the last axis: (q int16 ndarray | None, stored scale float32 ndarray,
        the shadow values | None)."""
        xn = x.detach().float().contiguous().numpy()
        if self.exact:
            zero = np.abs(xn).max(axis=-1) == 0
            s = np.where(zero, np.float32(1.0), np.float32(2.0**shift)).astype(np.float32)
            return None, s, torch.from_numpy(xn)
        q, s = K.quant_rows(xn, shift)
        return q, s, None

    def put_scales(self, fsl: K.Slot, b: int, j: int, segf: int, s: np.ndarray, count: int) -> None:
        """Scales of rows [0, len(s)), then 1.0 up to pad4(count) (pad rows and pad lanes)."""
        v = np.ones(K.pad4(count), np.float32)
        v[: len(s)] = s
        self.mem.write(fsl.addr + b * fsl.stride + j * segf, v.tobytes())

    def put_qrows(self, c: K.Cmd, b: int, y: torch.Tensor) -> None:
        """y [N][K] FP32 -> the op's output: int16 rows + scales (F_QOUT) per segment with NP zero rows, the KW tiling
        (QUANT_W, ROPE + F_KW), or FP32 rows (segment j of row n at C + n pitch + j segc)."""
        N, Kc = y.shape
        seg = c.par(C.P_SEG) or Kc
        nseg = Kc // seg
        NP = c.par(C.P_NP) or N
        segc, segf, sh = c.par(C.P_SEGC), c.par(C.P_SEGF), _i32(c.par(C.P_SHIFT))
        cs, fs = c.slot("C"), c.slot("F")
        kw = c.op == C.OP_QUANT_W or (c.op == C.OP_ROPE and c.flags & C.F_KW)
        for j in range(nseg):
            yj = y[:, j * seg : (j + 1) * seg]
            if kw:
                q, s, x = self.quant(yj, sh)
                qp = np.zeros((NP, seg), np.int16)
                xp = torch.zeros(NP, seg)
                if q is not None:
                    qp[:N] = q
                else:
                    xp[:N] = x
                self.put_tiled(cs, b, j * segc, qp, xp)
                self.put_scales(fs, b, j, segf, s, NP)
            elif c.flags & C.F_QOUT:
                q, s, x = self.quant(yj, sh)
                a = cs.addr + b * cs.stride + j * segc
                if self.exact:
                    self.mem.sview(a, (NP, seg), (cs.pitch, 2)).zero_()
                    self.mem.sview(a, (N, seg), (cs.pitch, 2)).copy_(x)
                else:
                    qp = np.zeros((NP, seg), np.int16)
                    qp[:N] = q
                    self.mem.view(a, (NP, seg), (cs.pitch, 2), torch.int16).copy_(torch.from_numpy(qp))
                self.put_scales(fs, b, j, segf, s, NP)
            else:
                a = cs.addr + b * cs.stride + j * segc
                self.mem.view(a, (N, seg), (cs.pitch, 4), torch.float32).copy_(yj)

    # ------------------------------------------------------------ GEMM (engine)
    def op_gemm(self, c: K.Cmd) -> None:
        M, Kc, N = c.M, c.K, c.N
        R, P1 = c.par(C.P_SEGC), c.par(C.P_SEGF)
        A, Bw, Cy, D, E, Fs, G = (c.slot(n) for n in "ABCDEFG")
        for b in range(c.B):
            x = self.rows_i16(A, b, N, Kc)
            w = self.weight(Bw, b, M, Kc)
            acc = torch.matmul(x, w.T).numpy()  # exact integers (|acc| < 2^42 < 2^53) or FP64 sums (exact mode)
            srow = self.vec_f32(E, b, M).numpy()
            stok = self.vec_f32(Fs, b, N).numpy()
            bias = self.vec_f32(G, b, M).numpy() if c.flags & C.F_BIAS else None
            groups = [(0, N)] if R == 0 else [(g * R, min(N, (g + 1) * R)) for g in range((N + R - 1) // R)]

            def out_rows(t0, t1):
                a = Cy.addr + b * Cy.stride + ((t0 // R) * P1 if R else t0 * Cy.pitch)
                return self.mem.view(a, (t1 - t0, M), (Cy.pitch, 4), torch.float32)

            if c.flags & C.F_ADD_D:
                add = self.rows_f32(D, b, N, M).numpy()
            elif c.flags & C.F_ACC:
                add = np.concatenate([out_rows(t0, t1).numpy() for t0, t1 in groups], 0)
            else:
                add = None
            if self.exact:
                acc32 = acc.astype(np.float32)  # FP64 sums of FP32 products, rounded once
                y = K.ftz(K.ftz(acc32 * K.ftz(srow[None, :])) * K.ftz(stok[:, None]))
                if bias is not None:
                    y = K.ftz(y + K.ftz(bias[None, :]))
                if add is not None:
                    y = K.ftz(y + K.ftz(add))
            else:
                y = K.gemm_out(np.rint(acc).astype(np.int64), srow, stok, bias, add)
            for t0, t1 in groups:
                out_rows(t0, t1).copy_(torch.from_numpy(np.ascontiguousarray(y[t0:t1])))

    # ------------------------------------------------------------ row unit
    def op_quant_x(self, c: K.Cmd) -> None:
        for b in range(c.B):
            self.put_qrows(c, b, self.rows_f32(c.slot("A"), b, c.N, c.K).clone())

    op_quant_w = op_quant_x  # the KW tiling: put_qrows

    def op_quant_wt(self, c: K.Cmd) -> None:
        seg = c.par(C.P_SEG) or c.K
        NP, segc, segf, sh = c.par(C.P_NP) or c.N, c.par(C.P_SEGC), c.par(C.P_SEGF), _i32(c.par(C.P_SHIFT))
        cs, fs = c.slot("C"), c.slot("F")
        for b in range(c.B):
            x = self.rows_f32(c.slot("A"), b, c.N, c.K)
            for j in range(c.K // seg):
                xt = x[:, j * seg : (j + 1) * seg].T.contiguous()  # [channels][keys], scale per channel over N keys
                q, s, xs = self.quant(xt, sh)
                qp = np.zeros((seg, NP), np.int16)
                xp = torch.zeros(seg, NP)
                if q is not None:
                    qp[:, : c.N] = q
                else:
                    xp[:, : c.N] = xs
                self.put_tiled(cs, b, j * segc, qp, xp)
                self.put_scales(fs, b, j, segf, s, seg)

    def op_norm(self, c: K.Cmd) -> None:
        Kc, ln = c.K, bool(c.flags & C.F_LN)
        eps = K.bits_f32(c.par(C.P_EPS))
        e = c.slot("E")
        g = self.mem.view(e.addr, (Kc,), (4,), torch.float32).clone()
        be = self.mem.view(e.addr + 4 * Kc, (Kc,), (4,), torch.float32).clone() if ln else None
        for b in range(c.B):
            y = self.rows_f32(c.slot("A"), b, c.N, Kc).clone()
            if self.fe == "kernels" and not self.exact:  # nr_emul.kernel: stats, correctly rounded mean / var, affine
                gn, bn = g.numpy(), None if be is None else be.numpy()
                z = torch.from_numpy(np.ascontiguousarray(self.nr.kernel(y.numpy(), gn, bn, eps, ln)[0]))
            elif ln:
                z = F.layer_norm(y, (Kc,), g, be, eps)
            else:
                z = g * (y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + eps))
            self.put_qrows(c, b, z)

    def _act(self, c: K.Cmd, mode: int) -> None:
        for b in range(c.B):
            x = self.rows_f32(c.slot("A"), b, c.N, c.K).clone()
            u = self.rows_f32(c.slot("B"), b, c.N, c.K).clone() if mode == 1 else None
            if self.fe == "kernels" and not self.exact:
                y = self.gs.act(x, u, mode, 9)  # gs_emu: the table front end, NSB 9, natural order
            elif mode == 0:
                y = F.gelu(x, approximate="tanh")
            elif mode == 1:
                y = F.silu(x) * u
            else:
                y = F.silu(x)
            self.put_qrows(c, b, y)

    def op_gelu(self, c: K.Cmd) -> None:
        self._act(c, 0)

    def op_silu_mul(self, c: K.Cmd) -> None:
        self._act(c, 2 if c.flags & C.F_NOMUL else 1)

    def op_rope(self, c: K.Cmd) -> None:
        N, Kc = c.N, c.K
        nseg = Kc // C.ROPE_SEG
        e = c.slot("E")
        for b in range(c.B):
            x = self.rows_f32(c.slot("A"), b, N, Kc).clone().reshape(N, nseg, C.ROPE_SEG)
            tab = self.mem.view(e.addr + b * e.stride, (N, 64), (e.pitch, 4), torch.int32).clone()
            if self.fe == "kernels" and not self.exact:
                y = self.rope.rope_rows(x, tab)  # rope_emu: the MODE_F32 rotation from the Q1.17 table
            else:
                cos = (-tab[:, :32].double() / 2**17).float()[:, None, :]
                sin = (-tab[:, 32:].double() / 2**17).float()[:, None, :]
                x1, x2 = x[..., :32], x[..., 32:]
                y = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1)
            self.put_qrows(c, b, y.reshape(N, Kc))

    def op_copy(self, c: K.Cmd) -> None:
        for b in range(c.B):
            src = self.rows_f32(c.slot("A"), b, c.N, c.K).clone()
            self.rows_f32(c.slot("C"), b, c.N, c.K).copy_(src)

    # ------------------------------------------------------------ softmax unit
    def op_softmax(self, c: K.Cmd) -> None:
        N, L = c.N, c.K
        NP = c.par(C.P_NP) or N
        mul = K.bits_f32(c.par(C.P_EPS))
        cs, fs = c.slot("C"), c.slot("F")
        for b in range(c.B):
            x = self.rows_f32(c.slot("A"), b, N, L).clone()
            if c.flags & C.F_MASK:
                rq = self.vec_i32(c.slot("D"), b, N).long()
                rk = self.vec_i32(c.slot("E"), b, L).long()
                valid = (rk[None, :] >= 0) & (rk[None, :] <= rq[:, None])
            else:
                valid = torch.ones(N, L, dtype=torch.bool)
            a = cs.addr + b * cs.stride
            if self.fe == "kernels" and not self.exact:
                q, s = self.smx.kernel(x, self.smx.c_from_mult(mul), valid, empty="zero")
                s = torch.where(q.sum(-1, keepdim=True) == 0, torch.ones_like(s), s)  # contract: zero rows 1.0
                assert bool((q >= 0).all() & (q <= C.QMAX).all())
                qp = np.zeros((NP, L), np.int16)
                qp[:N] = q.numpy().astype(np.int16)
                self.mem.view(a, (NP, L), (cs.pitch, 2), torch.int16).copy_(torch.from_numpy(qp))
                self.put_scales(fs, b, 0, 0, s.reshape(-1).numpy().astype(np.float32), NP)
                continue
            p = torch.softmax(torch.where(valid, x * mul, torch.tensor(float("-inf"))), -1)
            p = torch.nan_to_num(p, nan=0.0)  # a fully masked row: P = 0 (scale 1.0)
            q, s, xs = self.quant(p, 0)
            if self.exact:
                self.mem.sview(a, (NP, L), (cs.pitch, 2)).zero_()
                self.mem.sview(a, (N, L), (cs.pitch, 2)).copy_(xs)
            else:
                qp = np.zeros((NP, L), np.int16)
                qp[:N] = q
                self.mem.view(a, (NP, L), (cs.pitch, 2), torch.int16).copy_(torch.from_numpy(qp))
            self.put_scales(fs, b, 0, 0, s, NP)
