"""Python side of the float_top command-list format. The constants come from seq_format.h (parsed),
so the C++ and Python definitions cannot drift.

    from seq_format import C, Cmd, Slot, encode_list, decode_list
    cmd = Cmd(C.OP_GEMM, M=960, K=720, N=50, a=Slot(xq, pitch=1440), ...)
    blob = encode_list([cmd, Cmd(C.OP_END)])          # bytes, 128 per command
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass, field
from pathlib import Path

_H = Path(__file__).with_name("seq_format.h")


class _Consts:
    def __init__(self, path: Path):
        txt = path.read_text()
        self.values: dict[str, int] = {}
        for m in re.finditer(r"constexpr\s+uint32_t\s+(\w+)\s*=\s*(0x[0-9A-Fa-f]+|\d+)\s*;", txt):
            self.values[m.group(1)] = int(m.group(2), 0)
        for k, v in self.values.items():
            setattr(self, k, v)
        self.opnames = {v: k[3:] for k, v in self.values.items() if k.startswith("OP_")}
        self.stnames = {v: k[3:] for k, v in self.values.items() if k.startswith("ST_")}
        self.flagnames = {v: k[2:] for k, v in self.values.items() if k.startswith("F_")}


C = _Consts(_H)
SLOTS = ("a", "b", "c", "d", "e", "f")


@dataclass
class Slot:
    addr: int = 0     # physical byte address (40 bits)
    pitch: int = 0    # row pitch, bytes (24 bits)
    stride: int = 0   # batch stride, bytes (32 bits)

    def pack(self) -> tuple[int, int, int]:
        assert 0 <= self.addr < (1 << 40), hex(self.addr)
        assert 0 <= self.pitch < (1 << 24), self.pitch
        assert 0 <= self.stride < (1 << 32), self.stride
        return (self.addr & 0xFFFFFFFF, (self.addr >> 32) | (self.pitch << 8), self.stride)

    @staticmethod
    def unpack(w0: int, w1: int, w2: int) -> "Slot":
        return Slot(w0 | ((w1 & 0xFF) << 32), w1 >> 8, w2)

    def __add__(self, off: int) -> "Slot":  # same pitch/stride, address moved by off bytes
        return Slot(self.addr + off, self.pitch, self.stride)


def f32bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


def bits_f32(u: int) -> float:
    return struct.unpack("<f", struct.pack("<I", u & 0xFFFFFFFF))[0]


@dataclass
class Cmd:
    op: int
    flags: int = 0
    tag: int = 0
    M: int = 0
    K: int = 0
    N: int = 0
    B: int = 1
    G: int = 1
    a: Slot = field(default_factory=Slot)
    b: Slot = field(default_factory=Slot)
    c: Slot = field(default_factory=Slot)
    d: Slot = field(default_factory=Slot)
    e: Slot = field(default_factory=Slot)
    f: Slot = field(default_factory=Slot)
    p: list = field(default_factory=lambda: [0] * 7)  # p0..p6 as uint32 (use f32bits for floats)
    note: str = ""  # builder annotation, not encoded

    def fields(self) -> list[int]:
        assert 0 <= self.op < 256 and 0 <= self.flags < (1 << 16)
        assert 1 <= self.B < (1 << 16) and 1 <= self.G < (1 << 16)
        f = [0] * C.REC_FIELDS
        f[0] = self.op | (C.SEQ_MAGIC << 8) | (self.flags << 16)
        f[1] = self.tag & 0xFFFFFFFF
        f[2], f[3], f[4] = self.M, self.K, self.N
        f[5] = (self.B & 0xFFFF) | ((self.G & 0xFFFF) << 16)
        for s, name in enumerate(SLOTS):
            f[C.SLOT_BASE + 3 * s : C.SLOT_BASE + 3 * s + 3] = getattr(self, name).pack()
        assert len(self.p) == C.NPARAMS
        for i, v in enumerate(self.p):
            f[C.PARAM_BASE + i] = v & 0xFFFFFFFF
        x = C.SEQ_CHECK_SEED
        for v in f[: C.CHECK_FIELD]:
            x ^= v
        f[C.CHECK_FIELD] = x
        return f

    def encode(self) -> bytes:
        return struct.pack("<32I", *self.fields())

    @staticmethod
    def decode(rec: bytes) -> "Cmd":
        f = struct.unpack("<32I", rec)
        c = Cmd(f[0] & 0xFF, f[0] >> 16, f[1], f[2], f[3], f[4], (f[5] & 0xFFFF) or 1, (f[5] >> 16) or 1)
        for s, name in enumerate(SLOTS):
            setattr(c, name, Slot.unpack(*f[C.SLOT_BASE + 3 * s : C.SLOT_BASE + 3 * s + 3]))
        c.p = list(f[C.PARAM_BASE : C.PARAM_BASE + C.NPARAMS])
        return c

    def trace_line(self, idx: int) -> str:
        """The line tb_seq.cpp prints for a dispatched command (decode equivalence check)."""
        s = [f"{idx} {self.op:02x} {self.flags:04x} {self.tag:08x} {self.M} {self.K} {self.N} {self.B} {self.G}"]
        for name in SLOTS:
            sl = getattr(self, name)
            s.append(f"{sl.addr:x}/{sl.pitch:x}/{sl.stride:x}")
        s.append(" ".join(f"{v:08x}" for v in self.p))
        return " ".join(s)


def encode_list(cmds: list[Cmd]) -> bytes:
    return b"".join(c.encode() for c in cmds)


def decode_list(blob: bytes) -> list[Cmd]:
    return [Cmd.decode(blob[i : i + C.REC_BYTES]) for i in range(0, len(blob), C.REC_BYTES)]


def start_regs(list_phys: int, status_phys: int, first: int = 0, step: bool = False) -> dict[str, int]:
    """float_top register values for one sequencer start (all pointer registers 0)."""
    assert list_phys % 16 == 0 and status_phys % 16 == 0 and 0 <= first <= C.MAX_INDEX
    return dict(w0=0, w1=0, w2=0, w3=0, xmem=0, ymem=0,
                M=(C.MODE_MAGIC << 24) | (int(step) << C.MODE_STEP_BIT) | first,
                K=list_phys >> 4, N=status_phys >> 4)


def parse_return(ret: int) -> tuple[str, int]:
    return C.stnames.get(ret & 0xFF, f"0x{ret & 0xFF:02x}"), ret >> 8
