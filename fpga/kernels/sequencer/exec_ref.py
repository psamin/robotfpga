"""Reference executor of float_top command lists in PyTorch: the sequencer's walk (fetch, decode, check,
dispatch, stop at END / YIELD / error) over a byte-accurate model of the CMA regions, and each mode's
semantics as the builder's contract defines them (README.md, "Mode contracts").

Arithmetic of the modes:
  qbytes = 4  "exact": no quantization (scales 1), every op in FP32 as the policy does it. Running the
              policy through the list this way checks the list itself: addresses, pitches, strides,
              aliasing between roles, command order.
  qbytes = 2  int16 emulation: QUANT_* and every F_QOUT output quantize per row, s = amax|x| / 32767,
              q = round_half_even(x / s) (s = 0 -> q = 0); GEMM sums int16 x int16 exactly (float64 holds
              the int48 sums exactly) and outputs fp32(acc) * s_row * s_tok * p0 (+ bias) (+ D) (+ C).
              Weights quantized the same way per output row. The other modes stay FP32 (their kernels'
              own error is their owners' to report).
Activations live in real bytes (one uint8 tensor per arena and one for consts + lists); weights and the
token table are kept as tensors in a registry keyed by their reserved address (no 0.9 GB byte copy).
"""

from __future__ import annotations

import struct

import torch
import torch.nn.functional as F

from seq_format import C, Cmd, bits_f32

QMAX = 32767.0


def quant_rows(x: torch.Tensor, qbytes: int):
    """Per-row (last dim) int16 quantization; returns (q, s). qbytes 4: identity, s = 1."""
    if qbytes == 4:
        return x.float(), torch.ones(x.shape[:-1])
    amax = x.abs().amax(-1)
    s = amax / QMAX
    safe = torch.where(s > 0, s, torch.ones_like(s))
    q = torch.round(x / safe[..., None]).clamp_(-QMAX, QMAX)
    q = torch.where(s[..., None] > 0, q, torch.zeros_like(q))
    return q.to(torch.int16), s


class Memory:
    def __init__(self):
        self.regions = []   # (base, uint8 tensor)
        self.blobs = {}     # addr -> registry entry (W, bias, blob)
        self.tables = {}
        self._qcache = {}

    def add_region(self, base: int, size: int):
        self.regions.append((base, torch.zeros(size, dtype=torch.uint8)))

    def _find(self, addr: int, extent: int):
        for base, t in self.regions:
            if base <= addr and addr + extent <= base + t.numel():
                return base, t
        raise IndexError(f"access 0x{addr:x}+{extent} outside every region")

    def write(self, addr: int, raw: bytes):
        base, t = self._find(addr, len(raw))
        t[addr - base: addr - base + len(raw)] = torch.frombuffer(bytearray(raw), dtype=torch.uint8)

    def read(self, addr: int, n: int) -> bytes:
        base, t = self._find(addr, n)
        return bytes(t[addr - base: addr - base + n].numpy())

    def view(self, addr: int, shape, strides_b, dtype) -> torch.Tensor:
        """Strided view (strides in bytes; 0 = broadcast) into a region, bounds-checked."""
        isz = torch.empty(0, dtype=dtype).element_size()
        ext = isz + sum((n - 1) * s for n, s in zip(shape, strides_b))
        base, t = self._find(addr, ext)
        off = addr - base
        assert off % isz == 0 and all(s % isz == 0 for s in strides_b), (hex(addr), strides_b, isz)
        typed = t.view(dtype)
        return torch.as_strided(typed, tuple(shape), tuple(s // isz for s in strides_b), off // isz)

    # weight operands
    def weight(self, addr: int, scale_addr: int, mp: int, k: int, qbytes: int, bias: bool):
        if addr in self.blobs:
            key = (addr, qbytes)
            if key not in self._qcache:
                e = self.blobs[addr]
                W = torch.zeros(mp, k)
                W[: e["W"].shape[0]] = e["W"]
                q, s = quant_rows(W, qbytes)
                b = torch.zeros(mp)
                if e["bias"] is not None:
                    b[: e["bias"].shape[0]] = e["bias"]
                self._qcache[key] = (q, s, b)
            q, s, b = self._qcache[key]
            return q, s, (b if bias else None)
        qdt = torch.int16 if qbytes == 2 else torch.float32
        q = self.view(addr, (mp, k), (k * qbytes, qbytes), qdt)
        s = self.view(scale_addr, (mp,), (4,), torch.float32)
        b = self.view(scale_addr + 4 * mp, (mp,), (4,), torch.float32) if bias else None
        return q, s, b


class Executor:
    def __init__(self, mem: Memory, qbytes: int, gelu=None):
        self.mem, self.qb = mem, qbytes
        self.qdt = torch.int16 if qbytes == 2 else torch.float32
        self.gelu = gelu or (lambda x: F.gelu(x, approximate="tanh"))
        self.counts = {}

    # ---- views
    def v3(self, s, B, N, K, dtype=torch.float32, esz=4):
        return self.mem.view(s.addr, (B, N, K), (s.stride, s.pitch, esz), dtype)

    def vq(self, s, B, N, K):
        return self.v3(s, B, N, K, self.qdt, self.qb)

    def vvec(self, s, B, N, dtype=torch.float32):
        return self.mem.view(s.addr, (B, N), (s.stride, 4), dtype)

    def out(self, c: Cmd, y: torch.Tensor):
        B, N, K = y.shape
        assert N == 1 or c.c.pitch >= K * (self.qb if c.flags & C.F_QOUT else 4), ("output rows overlap", c.note)
        if c.flags & C.F_QOUT:
            q, s = quant_rows(y, self.qb)
            self.vq(c.c, B, N, K).copy_(q)
            self.vvec(c.f, B, N).copy_(s)
        else:
            self.v3(c.c, B, N, K).copy_(y)

    # ---- the sequencer walk (seq_core.h run())
    def run(self, list_addr: int, first: int, status_addr: int, max_cmds: int = 1 << 20):
        idx, done, tag = first, 0, 0
        while True:
            raw = self.mem.read(list_addr + idx * C.REC_BYTES, C.REC_BYTES)
            c = Cmd.decode(raw)
            f = struct.unpack("<32I", raw)
            x = C.SEQ_CHECK_SEED
            for v in f[: C.CHECK_FIELD]:
                x ^= v
            if (f[0] >> 8) & 0xFF != C.SEQ_MAGIC:
                code = C.ST_BAD_MAGIC
                break
            if x != f[C.CHECK_FIELD]:
                code = C.ST_BAD_CHECK
                break
            if c.op not in C.opnames:
                code = C.ST_BAD_OPCODE
                break
            if c.op == C.OP_END:
                code = C.ST_END
                break
            self.dispatch(c)
            done += 1
            tag = c.tag
            idx += 1
            if c.flags & C.F_YIELD:
                code = C.ST_YIELD
                break
            assert done < max_cmds
        if status_addr:
            st = [code, idx, tag, done, 0, C.STATUS_MAGIC, 0, 0]
            self.mem.write(status_addr, b"".join(v.to_bytes(4, "little") for v in st))
        return code, idx, done

    def dispatch(self, c: Cmd):
        name = C.opnames[c.op]
        self.counts[name] = self.counts.get(name, 0) + 1
        getattr(self, "op_" + name.lower())(c)

    # ---- modes
    def op_nop(self, c):
        pass

    def op_gemm(self, c: Cmd):
        B, G, N, K, M = c.B, c.G, c.N, c.K, c.M
        X = self.vq(c.a, B, N, K)
        xs = self.vvec(c.f, B, N)
        p0 = bits_f32(c.p[0])
        ys = []
        for j in range((B + G - 1) // G):
            q, s, b = self.mem.weight(c.b.addr + j * c.b.stride, c.e.addr + j * c.e.stride, M, K, self.qb,
                                      bool(c.flags & C.F_BIAS))
            Xj = X[j * G: (j + 1) * G]
            if self.qb == 2:
                acc = torch.matmul(Xj.double(), q.double().T).float()
            else:
                acc = torch.matmul(Xj, q.T)
            y = acc * s[None, None, :] * xs[j * G: (j + 1) * G, :, None] * p0
            if b is not None:
                y = y + b
            ys.append(y)
        y = torch.cat(ys, 0)
        if c.flags & C.F_ADD_D:
            y = y + self.v3(c.d, B, N, M)
        if c.flags & C.F_ACC:
            y = y + self.v3(c.c, B, N, M)
        assert N == 1 or c.c.pitch >= 4 * M, ("GEMM output rows overlap", c.note)
        self.v3(c.c, B, N, M).copy_(y)

    def op_quant_x(self, c):
        x = self.v3(c.a, c.B, c.N, c.K)
        q, s = quant_rows(x, self.qb)
        self.vq(c.c, c.B, c.N, c.K).copy_(q)
        self.vvec(c.f, c.B, c.N).copy_(s)

    def _blob_out(self, c, w, b):
        mp, k = w.shape
        wp = torch.zeros(-(-mp // 32) * 32, k)
        wp[:mp] = w
        q, s = quant_rows(wp, self.qb)
        mp = wp.shape[0]
        self.mem.view(c.c.addr + b * c.c.stride, (mp, k), (k * self.qb, self.qb), self.qdt).copy_(q)
        self.mem.view(c.f.addr + b * c.f.stride, (mp,), (4,), torch.float32).copy_(s)

    def op_quant_w(self, c):
        x = self.v3(c.a, c.B, c.N, c.K)
        for b in range(c.B):
            self._blob_out(c, x[b], b)

    def op_quant_wt(self, c):
        x = self.v3(c.a, c.B, c.N, c.K)
        red = c.p[0]
        assert red >= c.N
        for b in range(c.B):
            w = torch.zeros(c.K, red)
            w[:, : c.N] = x[b].T
            self._blob_out(c, w, b)

    def op_norm(self, c):
        B, N, K = c.B, c.N, c.K
        h = self.v3(c.a, B, N, K).clone()
        if c.flags & C.F_RESID:
            h = h + self.v3(c.b, B, N, K)
        if c.flags & C.F_SUM:
            self.v3(c.d, B, N, K).copy_(h)
        eps = bits_f32(c.p[0])
        if c.flags & C.F_LN:
            g = self.mem.view(c.e.addr, (K,), (4,), torch.float32)
            be = self.mem.view(c.e.addr + 4 * K, (K,), (4,), torch.float32)
            y = F.layer_norm(h, (K,), g, be, eps)
        else:
            g = self.mem.view(c.e.addr, (K,), (4,), torch.float32)
            y = g * (h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps))
        self.out(c, y)

    def op_softmax(self, c):
        B, N, K = c.B, c.N, c.K
        s = self.v3(c.a, B, N, K).clone()
        p0 = bits_f32(c.p[0])
        if p0 != 1.0:
            s = s * p0
        if c.flags & C.F_MASK:
            kinfo = self.vvec(c.e, B, K, torch.int32)
            qinfo = self.vvec(c.d, B, N, torch.int32)
            vis = (kinfo[:, None, :] >= 0) & (kinfo[:, None, :] <= qinfo[:, :, None])
            s = s.masked_fill(~vis, float("-inf"))
        p = torch.softmax(s, -1)
        p = torch.nan_to_num(p, nan=0.0)  # contract: a fully masked row outputs zeros
        self.out(c, p)

    def op_gelu(self, c):
        self.out(c, self.gelu(self.v3(c.a, c.B, c.N, c.K)))

    def op_silu_mul(self, c):
        y = F.silu(self.v3(c.a, c.B, c.N, c.K))
        if not c.flags & C.F_NOMUL:
            y = y * self.v3(c.b, c.B, c.N, c.K)
        self.out(c, y)

    def op_rope(self, c):
        B, N, K = c.B, c.N, c.K
        x = self.v3(c.a, B, N, K).clone()
        pos = self.vvec(c.d, B, N, torch.int32).long()
        tab = self.mem.view(c.e.addr, (256, K), (c.e.pitch, 4), torch.float32)
        h = K // 2
        cos, sin = tab[pos][..., :h], tab[pos][..., h:]
        x1, x2 = x[..., :h], x[..., h:]
        y = torch.empty_like(x)
        y[..., :h] = x1 * cos - x2 * sin
        y[..., h:] = x2 * cos + x1 * sin
        self.out(c, y)

    def op_copy(self, c):
        src = self.mem.view(c.a.addr, (c.B, c.N, c.K), (c.a.stride, c.a.pitch, 1), torch.uint8).clone()
        self.mem.view(c.c.addr, (c.B, c.N, c.K), (c.c.stride, c.c.pitch, 1), torch.uint8).copy_(src)

    def op_gather(self, c):
        ids = self.vvec(c.d, 1, c.N, torch.int32)[0].long()
        tab = self.mem.tables[c.a.addr]
        y = tab[ids] * bits_f32(c.p[0])
        self.v3(c.c, 1, c.N, c.K).copy_(y[None])

    def op_axpy(self, c):
        a = self.v3(c.a, c.B, c.N, c.K).clone()
        b = self.v3(c.b, c.B, c.N, c.K)
        self.v3(c.c, c.B, c.N, c.K).copy_(a + bits_f32(c.p[0]) * b)
