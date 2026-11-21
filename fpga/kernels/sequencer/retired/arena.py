"""CMA layout for the sequencer: one shared region (weights + constant tables + command lists) and two
per-inference activation arenas, so the ARM can fill arena B (images, tokens, state, noise) and read
arena A's actions while the FPGA runs the other one.

    plan = CmaPlan(base=0x6000_0000)                 # physical base of one big CMA allocation
    w = plan.weights.alloc("lm.0.q", nbytes)          # shared, read-only during a run
    a = plan.arena(0).alloc("x_t", 50 * 32 * 4)       # per inference

Every buffer is 64-byte aligned (one A53 cache line; flush/invalidate never touch a neighbour); weight
blobs are 4 KB aligned (AXI bursts never cross 4 KB, and HLS can then issue maximal bursts).
Scratch buffers that live only inside one command sequence are 'roles' (Region.role): one buffer per
role, sized by the largest request, reused by every stage (vision, LM, expert) - commands run in
order, so reuse is safe as long as a role's data is consumed before the role is written again; the
torch executor (exec_ref.py) catches violations as wrong actions.
"""

from __future__ import annotations

from dataclasses import dataclass


def align_up(x: int, a: int) -> int:
    return (x + a - 1) // a * a


@dataclass
class Buf:
    name: str
    addr: int
    nbytes: int

    def __add__(self, off: int) -> int:
        return self.addr + off


class Region:
    def __init__(self, name: str, base: int, size: int, align: int = 64):
        assert base % 4096 == 0
        self.name, self.base, self.size, self.align = name, base, size, align
        self.top = base
        self.bufs: dict[str, Buf] = {}
        self.roles: dict[str, int] = {}  # role -> max bytes requested (sizing pass)
        self.role_bufs: dict[str, Buf] = {}

    def alloc(self, name: str, nbytes: int, align: int | None = None) -> Buf:
        assert name not in self.bufs, f"{self.name}: {name} allocated twice"
        a = align or self.align
        addr = align_up(self.top, a)
        nbytes = max(int(nbytes), 16)
        if addr + nbytes > self.base + self.size:
            raise MemoryError(f"{self.name}: {name} ({nbytes} B) does not fit ({self.size} B region)")
        self.top = addr + nbytes
        b = Buf(name, addr, nbytes)
        self.bufs[name] = b
        return b

    def role(self, name: str, nbytes: int) -> Buf:
        """A reusable scratch buffer. After freeze_roles() it returns the one buffer for the role."""
        if name in self.role_bufs:
            b = self.role_bufs[name]
            assert nbytes <= b.nbytes, f"{self.name}: role {name} needs {nbytes} > {b.nbytes} B (re-size)"
            return b
        self.roles[name] = max(self.roles.get(name, 0), int(nbytes))
        return Buf(name, 0, nbytes)  # sizing pass: placeholder address

    def freeze_roles(self, sizes: dict[str, int]) -> None:
        for name, n in sorted(sizes.items()):
            self.role_bufs[name] = self.alloc("role:" + name, n, 4096)

    @property
    def used(self) -> int:
        return self.top - self.base

    def table(self) -> list[tuple[str, int, int]]:
        return [(b.name, b.addr, b.nbytes) for b in sorted(self.bufs.values(), key=lambda b: b.addr)]


class CmaPlan:
    """[shared: weights | consts | lists] [arena 0] [arena 1], one physically contiguous CMA block."""

    def __init__(self, base: int = 0x6000_0000, weights_size: int = 1 << 30, consts_size: int = 8 << 20,
                 lists_size: int = 4 << 20, arena_size: int = 96 << 20, n_arenas: int = 2):
        self.base = base
        off = base
        self.weights = Region("weights", off, weights_size, 4096)
        off += weights_size
        self.consts = Region("consts", off, consts_size)
        off += consts_size
        self.lists = Region("lists", off, lists_size, 128)
        off += lists_size
        self.arenas = []
        for i in range(n_arenas):
            self.arenas.append(Region(f"arena{i}", off, arena_size))
            off += arena_size
        self.end = off

    def arena(self, i: int) -> Region:
        return self.arenas[i]

    def summary(self) -> str:
        rows = [f"{r.name:8s} base 0x{r.base:010x} used {r.used / 2**20:8.2f} MB of {r.size / 2**20:.0f} MB"
                for r in (self.weights, self.consts, self.lists, *self.arenas)]
        return "\n".join(rows)
