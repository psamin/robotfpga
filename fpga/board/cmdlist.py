"""Command-list runtime for SmolVLA -12392 on the KR260 (fpga/kernels/CONTRACT.md 1.0, section 10).

One IP start per 50-action chunk instead of ~1,400 engine calls: at policy load every weight is quantized and
packed (contract.quant_rows, the port tiling, folded constants: sequencer/build_smolvla.Program), the constant
tables are written, and the chunk's 2,701-record list (contract_plan.py through build_smolvla.chunk_plan) is
relocated into real buffers and validated. Per chunk the ARM resizes/pads the images (LeRobot's own
prepare_images), im2cols the patches and quantizes them with the contract quantizer, writes the state, the
12 language rows and the noise into a free arena, flushes them, starts the list, polls done, checks the
status block and reads x_t (50 x 32 FP32).

    from fpga.board.cmdlist import ListRuntime, install
    rt = ListRuntime(policy, backend="xrt")          # the list-mode bitstream already loaded; root
    install(policy, rt)                              # policy.predict_action_chunk now runs the list
    chunk = policy.predict_action_chunk(batch)       # re-entrant: two callers may be in flight at once

Memory (CONTRACT.md 1.3). The plan's bump allocator gives every allocation (weight blob, s_row, bias, constant
table, activation buffer, KV blob) a virtual address; Placement packs the allocations into physically
contiguous buffers (32 MB slabs, each allocation at its own alignment, as runtime.py packs the per-call
weights) and every slot address of every record is relocated into them (64-bit). Shared: the weights and the
constants (0.66 GB). Per arena (one per inference in flight, default 2): the activations, the KV blobs, the
status block, the 12 language rows and the list itself (~40 MB). Nothing assumes low or high DDR: every
address comes from the buffer's physical address; low_ddr=True refuses buffers that end above 0x8000_0000.

Backends:
  "xrt"   the board: XRT buffer objects (fpga/board/xrtmem.py) and the IP's registers over /dev/mem. Needs the
          list-mode float_top (the integrator's build, not shipped yet); a probe list (one END record) at
          start-up refuses the shipped per-call bitstream (it returns 1 for M = 0xA5...).
  "mock"  the same code path on the laptop: host buffers at made-up physical addresses (from 0x4000_0000), a
          register-level fake IP whose ap_start walks the list with the golden executor
          (fpga/kernels/sequencer/exec_ref.py, the sequencer's walk; fe="kernels": the kernels' bit-exact
          emulators nr_emul / gs_emu / smx_emu / rope_emu + contract quantizer + contract GEMM epilogue).
          strict=True (default) gives every allocation its own buffer with a guard gap, so any access that
          leaves its allocation stops with an error. The integrated C shim (fpga/hls built for list mode) does
          not exist yet; when it does it replaces the executor behind the same registers.

arm_fp32=True puts the 21 layers the checkpoint stores in FP32 (state_proj, action_in/out_proj,
action_time_mlp_in/out, the expert's 16 cross-attention k/v projections: 57 records per chunk) back on the
ARM: their GEMM records become NOP + YIELD, and at each such yield the ARM computes the layer in FP32 from the
record's own int16 input (dequantized: q * s_tok), with the layer's FP32 weights, the record's epilogue (+ D,
+= C) and, for action_out_proj, torch's Euler step x_t + dt * v_t; then the list resumes. The default (False)
runs them on the engine (design study section 2.2, CONTRACT.md 6).

Exact levers in the list (always on, built into contract_plan.py): the language prefix is the 12 real tokens
(tokenizer padding "longest": prefix 45 rows, not 81) and LM layer 15's q / attention / o / MLP and the final
norm are skipped (only its K / V reach the expert). A task whose tokenizer gives another count needs a new
list (it is refused per chunk).
"""

from __future__ import annotations

import bisect
import json
import math
import queue
import struct
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
KDIR = ROOT / "fpga" / "kernels"
for _p in (KDIR, KDIR / "sequencer", KDIR / "rope"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import build_smolvla as BS  # noqa: E402
import contract as K  # noqa: E402
import contract_plan as CP  # noqa: E402
from contract import C  # noqa: E402
from exec_ref import Executor, Memory  # noqa: E402

VBASE = 0x40_0000_0000  # the plan's virtual base (2^38): far from any physical address, so a missed
# relocation cannot alias real memory (it falls outside every buffer)
SLAB_BYTES = 32 << 20
BUF_ALIGN = 4096
LOW_DDR_END = 0x8000_0000
MOCK_BASE = 0x4000_0000  # mock physical addresses start in the KR260's low DDR (the contract's test base)
MOCK_GUARD = 64 << 10  # strict mock: unmapped bytes between buffers
# The 21 FP32-stored layers as list records (phase, note name): 1 + 4 x 10 steps + 16 = 57 GEMM records
ARM_FP32 = {
    ("connector", "state"),
    ("steps", "a_in"),
    ("steps", "t_in"),
    ("steps", "t_out"),
    ("steps", "a_out+euler"),
    ("cross_kv", "kproj"),
    ("cross_kv", "vproj"),
}
DT = -1.0 / 10  # euler_integrate's dt for num_steps 10 (checked against the config at load)


# ---------------------------------------------------------------- plan + allocations
@dataclass
class Alloc:
    addr: int  # virtual
    nbytes: int
    align: int
    region: str  # contract_plan region: weights / consts / arena (or "list", added here)


def build_plan(vbase: int = VBASE):
    """contract_plan's chunk (through build_smolvla.chunk_plan, which fixes the QKV_s sizing) at vbase, and
    every allocation the bump allocator made, in order."""
    allocs: list[Alloc] = []
    orig = CP.Mem.alloc

    def rec(self, nbytes: int, align: int = 64) -> int:
        a = orig(self, nbytes, align)
        allocs.append(Alloc(a, nbytes, align, self.cur))
        return a

    CP.Mem.alloc = rec
    try:
        plan = BS.chunk_plan(vbase, True)
    finally:
        CP.Mem.alloc = orig
    bad = K.validate_list(plan.cmds)
    assert not bad, bad[:5]
    return plan, allocs


# ---------------------------------------------------------------- physical buffers
@dataclass
class Buf:
    paddr: int
    nbytes: int
    array: np.ndarray  # uint8, the CPU view
    sync: object = None  # board: the XRT buffer (flush / invalidate); mock: None

    def flush(self, off: int, n: int) -> None:
        if self.sync is not None:
            self.sync.flush(off, n)

    def invalidate(self, off: int, n: int) -> None:
        if self.sync is not None:
            self.sync.invalidate(off, n)

    def free(self) -> None:
        if self.sync is not None:
            self.sync.free()
            self.sync = None


class FakeCma:
    """Host memory standing in for CMA: 4 KB-aligned buffers at increasing made-up physical addresses, with a
    guard gap between them."""

    def __init__(self, base: int = MOCK_BASE, guard: int = MOCK_GUARD):
        self.next, self.guard = base, guard

    def allocate(self, nbytes: int) -> Buf:
        b = Buf(self.next, nbytes, np.zeros(nbytes, np.uint8))
        self.next += -(-nbytes // BUF_ALIGN) * BUF_ALIGN + self.guard
        return b


class XrtCma:
    """Cacheable CMA buffers through XRT (fpga/board/xrtmem.py; root)."""

    def __init__(self):
        from fpga.board.xrtmem import XrtDevice

        self.dev = XrtDevice()

    def allocate(self, nbytes: int) -> Buf:
        x = self.dev.allocate(nbytes)
        return Buf(x.physical_address, nbytes, x.array, x)


class PhysMemory(Memory):
    """exec_ref.Memory over physical buffers (sorted, bisect lookup); flush / invalidate by address."""

    def __init__(self, shadow: bool = False):
        super().__init__(shadow=shadow)
        self.bases: list[int] = []
        self.bufs: list[Buf] = []

    def add_buffer(self, b: Buf) -> None:
        assert b.nbytes % 16 == 0 and b.paddr % 16 == 0
        i = bisect.bisect_left(self.bases, b.paddr)
        if i > 0:
            assert self.bases[i - 1] + self.bufs[i - 1].nbytes <= b.paddr, "buffers overlap"
        if i < len(self.bases):
            assert b.paddr + b.nbytes <= self.bases[i], "buffers overlap"
        sh = torch.zeros(b.nbytes // 2, dtype=torch.float32) if self.shadow else None
        self.bases.insert(i, b.paddr)
        self.bufs.insert(i, b)
        self.regions.insert(i, (b.paddr, torch.from_numpy(b.array), sh))

    def _index(self, addr: int, extent: int) -> int:
        i = bisect.bisect_right(self.bases, addr) - 1
        if i < 0 or addr + extent > self.bases[i] + self.bufs[i].nbytes:
            raise IndexError(f"access 0x{addr:x} + {extent} outside every buffer")
        return i

    def _find(self, addr: int, extent: int):
        return self.regions[self._index(addr, extent)]

    def flush(self, addr: int, n: int) -> None:
        i = self._index(addr, n)
        self.bufs[i].flush(addr - self.bases[i], n)

    def invalidate(self, addr: int, n: int) -> None:
        i = self._index(addr, n)
        self.bufs[i].invalidate(addr - self.bases[i], n)

    def flush_all(self) -> None:
        for b in self.bufs:
            b.flush(0, b.nbytes)


class Placement:
    """Allocations -> physical addresses. Shared allocations once; arena allocations once per arena."""

    def __init__(self, allocs: list[Alloc], cma, mem: PhysMemory, strict: bool, slab: int = SLAB_BYTES):
        self.allocs, self.cma, self.mem, self.strict, self.slab = allocs, cma, mem, strict, slab
        self.starts = [a.addr for a in allocs]
        assert self.starts == sorted(self.starts)
        for a, b in zip(allocs, allocs[1:], strict=False):
            assert a.addr + a.nbytes <= b.addr, "plan allocations overlap"
        self.buffers: list[Buf] = []

    def place(self, idx: list[int]) -> dict[int, int]:
        """Pack allocations idx (in order) into buffers; returns {alloc index: physical address}."""
        groups: list[list[tuple[int, int]]] = []
        size = 0
        for i in idx:
            a = self.allocs[i]
            n = -(-max(a.nbytes, 16) // 16) * 16
            al = max(a.align, 16)
            off = -(-size // al) * al
            if not groups or self.strict or off + n > self.slab:
                groups.append([])
                off = 0
            groups[-1].append((i, off))
            size = off + n
        phys = {}
        for g in groups:
            i_last, off_last = g[-1]
            n = off_last + -(-max(self.allocs[i_last].nbytes, 16) // 16) * 16
            b = self.cma.allocate(-(-n // BUF_ALIGN) * BUF_ALIGN)
            assert b.paddr % BUF_ALIGN == 0, hex(b.paddr)
            self.mem.add_buffer(b)
            self.buffers.append(b)
            for i, off in g:
                phys[i] = b.paddr + off
        return phys

    def relocator(self, phys: dict[int, int]):
        starts, allocs = self.starts, self.allocs

        def reloc(addr: int) -> int:
            i = bisect.bisect_right(starts, addr) - 1
            if i < 0 or addr >= starts[i] + max(allocs[i].nbytes, 1) or i not in phys:
                raise ValueError(f"slot address 0x{addr:x} is in no placed allocation")
            return phys[i] + (addr - starts[i])

        return reloc


def relocate(cmds: list[K.Cmd], reloc) -> list[K.Cmd]:
    out = []
    for c in cmds:
        s = {n: K.Slot(reloc(sl.addr), sl.pitch, sl.stride) for n, sl in c.s.items() if not sl.zero()}
        out.append(K.Cmd(c.op, c.flags, c.tag, c.M, c.K, c.N, c.B, s, dict(c.p), c.note))
    return out


# ---------------------------------------------------------------- one arena's program
class ArenaProgram(BS.Program):
    """build_smolvla.Program on given (relocated) records, memory and executor; load() (weights + constants,
    once for all arenas) only on the first."""

    def __init__(self, policy, cmds: list[K.Cmd], mem: PhysMemory, ex: Executor, exact: bool):
        self.p, self.exact, self.cmds, self.mem, self.ex = policy, exact, cmds, mem, ex
        m = policy.model
        vwe = m.vlm_with_expert
        self.vlm = vwe.get_vlm_model()
        self.vis, self.text, self.exp = self.vlm.vision_model, self.vlm.text_model, vwe.lm_expert
        assert (
            self.vlm.connector.scale_factor == 4
            and vwe.num_vlm_layers == 16
            and vwe.self_attn_every_n_layers == 2
        )
        self._identify()

    def load(self) -> None:
        with torch.no_grad():
            self._weights()
            self._consts()

    def lang_rows(self, lang_tokens: torch.Tensor, lang_masks: torch.Tensor) -> torch.Tensor:
        """The real language tokens' embeddings x sqrt(960), in FP32 whatever the module's dtype (torch FP32
        policy: embedding lookup, then lang_emb * math.sqrt(dim) in FP32)."""
        ids = lang_tokens[0][lang_masks[0].bool()]
        if ids.numel() != CP.TQ_P - 2 * 16 - 1:
            raise ValueError(
                f"the list is built for {CP.TQ_P - 33} real language tokens, this observation has"
                f" {ids.numel()} (another task string needs another list)"
            )
        w = self.text.get_input_embeddings().weight
        emb = w[ids].float()
        return emb * math.sqrt(emb.shape[-1])


# ---------------------------------------------------------------- the IP at the register level
class FakeListMmio:
    """float_top in list mode at the register level (mock): ap_start runs exec_ref's walk (= seq_core.h
    walk()) with M / K / N from the registers; the pointer registers must be 0 (CONTRACT.md 3.1)."""

    def __init__(self, ex: Executor):
        self.ex, self.regs, self.done, self.starts = ex, {}, False, 0

    def read(self, off: int) -> int:
        if off == C.REG_AP_CTRL:
            v = 0x4 | (0x2 if self.done else 0)  # ap_idle; ap_done is clear-on-read
            self.done = False
            return v
        return self.regs.get(off, 0)

    def write(self, off: int, value: int) -> None:
        self.regs[off] = value & 0xFFFFFFFF
        if off == C.REG_AP_CTRL and value & 1:
            for r in (C.REG_W0, C.REG_W1, C.REG_W2, C.REG_W3, C.REG_XMEM, C.REG_YMEM):
                assert self.regs.get(r, 0) == 0 and self.regs.get(r + 4, 0) == 0, hex(r)
            self.starts += 1
            ret = self.ex.start(
                self.regs.get(C.REG_M, 0), self.regs.get(C.REG_K, 0), self.regs.get(C.REG_N, 0)
            )
            self.regs[C.REG_AP_RETURN] = ret
            self.done = True


class ListIp:
    """One list start: the registers of contract.start_regs, ap_start, poll ap_done, ap_return."""

    def __init__(self, mmio, poll_sleep: float = 200e-6, timeout: float = 30.0):
        self.mmio, self.poll_sleep, self.timeout = mmio, poll_sleep, timeout
        self.lock = threading.Lock()

    def run(self, list_addr: int, status_addr: int, first: int = 0, step: bool = False) -> int:
        with self.lock:
            m = self.mmio
            if not m.read(C.REG_AP_CTRL) & 0x4:
                raise RuntimeError("float_top is not idle")
            for off, v in K.start_regs(list_addr, status_addr, first, step).items():
                m.write(off, v)
            m.write(C.REG_AP_CTRL, 1)
            t0 = time.perf_counter()
            while not m.read(C.REG_AP_CTRL) & 0x2:
                if time.perf_counter() - t0 > self.timeout:
                    raise TimeoutError(f"no ap_done after {self.timeout} s (list at 0x{list_addr:x})")
                time.sleep(self.poll_sleep)
            return m.read(C.REG_AP_RETURN)


def probe_list_ip(ip: ListIp, mem: PhysMemory, list_addr: int, status: int) -> dict:
    """A one-record list (END, tag 0xE0D) at list_addr: a list-mode float_top returns (0 << 8) | END and
    writes the status block (magic, version, tag); the shipped per-call bitstream returns 1 (M = 0xA5...
    is out of its range) and touches nothing. The list buffer's first record is restored and the status
    block zeroed around it. Returns the parsed status block."""
    saved = mem.read(list_addr, C.REC_BYTES)
    mem.write(list_addr, K.encode(K.Cmd(C.OP_END, tag=0xE0D)))
    mem.write(status, bytes(C.STATUS_BYTES))
    mem.flush(list_addr, C.REC_BYTES)
    mem.flush(status, C.STATUS_BYTES)
    try:
        r = ip.run(list_addr, status)
        mem.invalidate(status, C.STATUS_BYTES)
        st = K.parse_status(mem.read(status, C.STATUS_BYTES))
    finally:
        mem.write(list_addr, saved)
        mem.flush(list_addr, C.REC_BYTES)
    if r == 1:
        raise RuntimeError(
            "the loaded float_top is the per-call engine (ap_return 1): load the list-mode build"
        )
    if r != C.ST_END or not st["magic_ok"] or st["version"] != C.CONTRACT_VERSION or st["tag"] != 0xE0D:
        raise RuntimeError(f"list probe: ap_return 0x{r:x}, status {st}")
    return st


# ---------------------------------------------------------------- jobs
@dataclass
class Arena:
    k: int
    prog: ArenaProgram
    run_cmds: list[K.Cmd]
    list_addr: int
    status: int
    arm_ops: dict[int, tuple] = field(default_factory=dict)  # record index -> (original record, W, b, kind)
    inputs: list[tuple[int, int]] = field(default_factory=list)  # (addr, nbytes) the ARM writes per chunk
    x_t: tuple[int, int] = (0, 0)


@dataclass
class Job:
    arena: Arena
    done: threading.Event = field(default_factory=threading.Event)
    error: BaseException | None = None
    t: dict = field(default_factory=dict)
    status: dict = field(default_factory=dict)


def _extent(s: K.Slot, b: int, rows: int, row_bytes: int) -> int:
    return (b - 1) * s.stride + (rows - 1) * s.pitch + row_bytes


class ListRuntime:
    """Weights, constants, arenas, lists and the engine thread for one policy."""

    def __init__(
        self,
        policy,
        *,
        backend: str = "mock",
        arenas: int = 2,
        fe: str = "kernels",
        exact: bool = False,
        nsb: int = 10,
        arm_fp32: bool = False,
        strict: bool = True,
        low_ddr: bool = False,
        release_weights: bool = True,
        clock_mhz: float | None = None,
        verbose: bool = True,
    ):
        assert backend in ("mock", "xrt"), backend
        assert arenas >= 1
        t0 = time.perf_counter()
        cfg = policy.config
        assert cfg.n_obs_steps == 1 and cfg.num_steps == 10 and cfg.chunk_size == CP.TQ_E, (
            "the list is built for n_obs_steps 1, 10 steps, chunk 50"
        )
        self.policy, self.backend, self.exact, self.arm_fp32 = policy, backend, exact, arm_fp32
        if backend == "xrt":
            assert not exact, "exact (FP32 shadow) mode is a mock-only list check"
            self.cma, strict = XrtCma(), False
        else:
            self.cma = FakeCma()
        self.mem = PhysMemory(shadow=exact)
        # the executor: the IP's model (mock) or only its operand helpers (board: weight tiling, ARM layers)
        self.ex = Executor(self.mem, exact=exact, fe=fe if backend == "mock" else "torch", nsb=nsb)
        plan, allocs = build_plan()
        self.n_records = len(plan.cmds)
        # the language rows are written per chunk: each arena gets its own copy, so two chunks in flight never
        # share a host-written buffer (contract_plan puts them with the constants)
        lang_v = next(c for c in plan.cmds if c.note == "connector:lang").slot("A").addr
        lang_i = bisect.bisect_right([a.addr for a in allocs], lang_v) - 1
        list_bytes = len(plan.cmds) * C.REC_BYTES
        allocs.append(Alloc(allocs[-1].addr + allocs[-1].nbytes + BUF_ALIGN, list_bytes, BUF_ALIGN, "list"))
        allocs[-1].addr = -(-allocs[-1].addr // BUF_ALIGN) * BUF_ALIGN
        self.place = Placement(allocs, self.cma, self.mem, strict)
        shared = [i for i, a in enumerate(allocs) if a.region in ("weights", "consts") and i != lang_i]
        per_arena = [i for i, a in enumerate(allocs) if a.region in ("arena", "list") or i == lang_i]
        phys0 = self.place.place(shared)
        self.arenas: list[Arena] = []
        for k in range(arenas):
            phys = dict(phys0) | self.place.place(per_arena)
            cmds = relocate(plan.cmds, self.place.relocator(phys))
            prog = ArenaProgram(policy, cmds, self.mem, self.ex, exact)
            list_addr, status = phys[len(allocs) - 1], self.place.relocator(phys)(plan.layout["status"])
            self.arenas.append(Arena(k, prog, cmds, list_addr, status))
        self.t_place = time.perf_counter() - t0
        hi = max(b.paddr + b.nbytes for b in self.place.buffers)
        if low_ddr and hi > LOW_DDR_END:
            raise MemoryError(f"buffers end at 0x{hi:x}, above low DDR (0x{LOW_DDR_END:x}); low_ddr=True")
        # weights + constants (shared), once
        t1 = time.perf_counter()
        self.arenas[0].prog.load()
        self.t_load = time.perf_counter() - t1
        self.n_blobs = self.arenas[0].prog.n_blobs
        for a in self.arenas:
            self._finish_arena(a)
        self.mem.flush_all()
        self.bytes = {
            "shared": sum(allocs[i].nbytes for i in shared),
            "per_arena": sum(allocs[i].nbytes for i in per_arena),
            "buffers": len(self.place.buffers),
            "buffer_bytes": sum(b.nbytes for b in self.place.buffers),
            "lowest": min(b.paddr for b in self.place.buffers),
            "highest_end": hi,
        }
        if release_weights:
            self.released = release_policy_weights(policy)
        # the IP
        if backend == "xrt":
            from fpga.board.runtime import IP_BASE, IP_SPAN, set_pl_clock
            from fpga.board.xrtmem import Mmio

            if clock_mhz:
                self.clock_mhz = set_pl_clock(clock_mhz)
            self.ip = ListIp(Mmio(IP_BASE, IP_SPAN))
        else:
            self.ip = ListIp(FakeListMmio(self.ex), poll_sleep=0)
        self.probe()
        self.free: queue.Queue[Arena] = queue.Queue()
        for a in self.arenas:
            self.free.put(a)
        self.jobs: queue.Queue[Job | None] = queue.Queue()
        self.thread = threading.Thread(target=self._engine, name="float_top", daemon=True)
        self.thread.start()
        self.t_init = time.perf_counter() - t0
        if verbose:
            b = self.bytes
            print(
                f"list runtime ({backend}{', exact' if exact else ''}, fe {self.ex.fe}"
                f"{'' if backend == 'xrt' else ' ' + '+'.join(sorted(self.ex.kops))}, 21 FP32 layers on the"
                f" {'ARM' if arm_fp32 else 'engine'}): {self.n_records} records x {arenas} arena(s),"
                f" {self.n_blobs} weight blobs, shared {b['shared'] / 2**20:.1f} MB + {arenas} x"
                f" {b['per_arena'] / 2**20:.1f} MB in {b['buffers']} buffers at 0x{b['lowest']:x}.."
                f"0x{b['highest_end']:x}; placed {self.t_place:.1f} s, packed {self.t_load:.1f} s",
                flush=True,
            )

    # ------------------------------------------------------------ load-time helpers
    def _finish_arena(self, a: Arena) -> None:
        """The arena's run list (with the ARM layers as NOP + YIELD when arm_fp32), validated and written; the
        input and output extents."""
        prog, cmds = a.prog, a.run_cmds
        run = list(cmds)
        if self.arm_fp32:
            for i, (c, (ph, name, li, _)) in enumerate(zip(cmds, prog.roles, strict=True)):
                if c.op == C.OP_GEMM and (ph, name) in ARM_FP32:
                    W, b = prog._module_weight(ph, name, li)
                    a.arm_ops[i] = (
                        c,
                        W.detach().float().clone(),
                        None if b is None else b.detach().float().clone(),
                        name,
                    )
                    run[i] = K.Cmd(C.OP_NOP, C.F_YIELD, c.tag, note=c.note + "@arm")
            assert len(a.arm_ops) == 57, len(a.arm_ops)
        bad = K.validate_list(run)
        assert not bad, bad[:5]
        a.run_cmds = run
        blob = K.encode_list(run)
        self.mem.write(a.list_addr, blob)
        # per-chunk inputs: patches + scales, state + scale, language rows, x_t (noise in, actions out)
        pg = prog.find("vision", "patch", C.OP_GEMM)
        st = prog.find("connector", "state", C.OP_GEMM)
        lg = prog.find("connector", "lang", C.OP_COPY)
        xa = prog.find("steps", "x_t", C.OP_QUANT_X).slot("A")
        A, F = pg.slot("A"), pg.slot("F")
        a.inputs = [
            (A.addr, _extent(A, pg.B, pg.N, 2 * pg.K)),
            (F.addr, _extent(F, pg.B, 1, 4 * K.pad4(pg.N))),
            (st.slot("A").addr, 2 * st.K),
            (st.slot("F").addr, 16),
            (lg.slot("A").addr, lg.N * lg.slot("A").pitch),
            (xa.addr, CP.TQ_E * CP.A_D * 4),
        ]
        a.x_t = (xa.addr, CP.TQ_E * CP.A_D * 4)

    def probe(self) -> None:
        a = self.arenas[0]
        self.probe_status = probe_list_ip(self.ip, self.mem, a.list_addr, a.status)

    # ------------------------------------------------------------ per chunk
    def sample_actions(self, images, img_masks, lang_tokens, lang_masks, state, noise=None) -> tuple:
        """VLAFlowMatching.sample_actions on the FPGA: (x_t [1][50][32], timings). Thread-safe: each call
        takes a free arena (blocks while all are in flight) and queues its list for the engine thread."""
        t = {}
        t0 = time.perf_counter()
        if len(images) != 2 or not all(bool(mk.all()) for mk in img_masks):
            raise ValueError("the list is built for 2 present cameras")
        if state.shape[0] != 1:
            raise ValueError("batch size 1 only")
        if noise is None:
            noise = self.policy.model.sample_noise(
                (1, CP.TQ_E, self.policy.config.max_action_dim), state.device
            )
        a = self.free.get()
        t["wait_arena"] = time.perf_counter() - t0
        try:
            t1 = time.perf_counter()
            self._write_inputs(a, images, lang_tokens, lang_masks, state, noise)
            t["inputs"] = time.perf_counter() - t1
            job = Job(a)
            job.t["submit"] = time.perf_counter()
            self.jobs.put(job)
            job.done.wait()
            if job.error is not None:
                raise job.error
            t3 = time.perf_counter()
            self.mem.invalidate(*a.x_t)
            x = self.mem.view(
                a.x_t[0], (1, CP.TQ_E, CP.A_D), (CP.TQ_E * 4 * CP.A_D, 4 * CP.A_D, 4), torch.float32
            ).clone()
            t["read"] = time.perf_counter() - t3
        finally:
            self.free.put(a)
        if not torch.isfinite(x).all():
            raise FloatingPointError("x_t has NaN/Inf (CONTRACT.md review #27)")
        t |= job.t
        t["arena"] = a.k
        t["status"] = job.status
        return x, t

    def _write_inputs(self, a: Arena, images, lang_tokens, lang_masks, state, noise) -> None:
        prog = a.prog
        with torch.no_grad():
            pt = torch.stack(
                [
                    im[0].float().reshape(3, 16, 16, 16, 16).permute(1, 3, 0, 2, 4).reshape(256, 768)
                    for im in images
                ]
            )
            prog._put_qin(prog.find("vision", "patch", C.OP_GEMM), "A", "F", pt)
            prog._put_qin(
                prog.find("connector", "state", C.OP_GEMM), "A", "F", state.float().reshape(1, 1, -1)
            )
            lr = prog.lang_rows(lang_tokens, lang_masks)
            self.mem.write(prog.find("connector", "lang", C.OP_COPY).slot("A").addr, lr.numpy().tobytes())
            self.mem.write(a.x_t[0], noise[0].float().contiguous().numpy().tobytes())
        for addr, n in a.inputs:
            self.mem.flush(addr, n)

    def _engine(self) -> None:
        while True:
            job = self.jobs.get()
            if job is None:
                return
            try:
                self._run(job)
            except BaseException as e:  # noqa: BLE001 - handed to the caller
                job.error = e
            job.done.set()

    def _run(self, job: Job) -> None:
        a, t = job.arena, job.t
        t["wait_engine"] = time.perf_counter() - t.pop("submit")
        t["fpga"], t["arm_gemm"], starts, idx, done = 0.0, 0.0, 0, 0, 0
        while True:
            t0 = time.perf_counter()
            r = self.ip.run(a.list_addr, a.status, idx)
            t["fpga"] += time.perf_counter() - t0
            starts += 1
            nxt, code = r >> 8, r & 0xFF
            if code == C.ST_YIELD:  # the status block's done count is per start: sum it
                self.mem.invalidate(a.status, C.STATUS_BYTES)
                done += K.parse_status(self.mem.read(a.status, C.STATUS_BYTES))["done"]
            if code == C.ST_YIELD:
                if nxt - 1 in a.arm_ops:
                    t1 = time.perf_counter()
                    arm_gemm(self.mem, self.ex, *a.arm_ops[nxt - 1])
                    t["arm_gemm"] += time.perf_counter() - t1
                idx = nxt
                continue
            break
        self.mem.invalidate(a.status, C.STATUS_BYTES)
        st = K.parse_status(self.mem.read(a.status, C.STATUS_BYTES))
        st["list_starts"] = starts
        st["done_all_starts"] = done + st["done"]
        job.status = st
        ok = (
            code == C.ST_END
            and st["done_all_starts"] == len(a.run_cmds) - 1
            and nxt == len(a.run_cmds) - 1
            and st["magic_ok"]
            and st["version"] == C.CONTRACT_VERSION
            and st["code"] == "END"
        )
        if not ok:
            raise RuntimeError(f"list stopped: ap_return 0x{r:x} ({K.parse_return(r)}), status {st}")

    def close(self) -> None:
        self.jobs.put(None)
        self.thread.join(timeout=5)
        for b in self.place.buffers:
            b.free()


def arm_gemm(
    mem: PhysMemory, ex: Executor, c: K.Cmd, W: torch.Tensor, b: torch.Tensor | None, name: str
) -> None:
    """One GEMM record on the ARM in FP32: x = q * s_tok (its int16 input as the engine sees it), y = x W^T
    + b with the layer's FP32 weights (rows padded to M with zeros); then + D / += C as the record says.
    action_out_proj ("a_out+euler"): x_t += dt * v_t, torch's Euler step (the record's -0.1 fold is the
    engine's)."""
    A, Cy, D, Fs = c.slot("A"), c.slot("C"), c.slot("D"), c.slot("F")
    assert c.B == 1 and c.par(C.P_SEGC) == 0, "the 21 FP32 layers are single-item, no row groups"
    M, Kc, N = c.M, c.K, c.N
    for s, rows, rb in ((A, N, 2 * Kc), (Fs, 1, 4 * K.pad4(N)), (Cy, N, 4 * M)):
        mem.invalidate(s.addr, _extent(s, 1, rows, rb))
    if c.flags & C.F_ADD_D:
        mem.invalidate(D.addr, _extent(D, 1, N, 4 * M))
    with torch.no_grad():
        q = ex.rows_i16(A, 0, N, Kc).float()
        s = ex.vec_f32(Fs, 0, N).clone()
        x = q * s[:, None]
        Wp = torch.zeros(M, Kc)
        Wp[: W.shape[0]] = W
        bp = None
        if b is not None:
            bp = torch.zeros(M)
            bp[: b.shape[0]] = b
        y = torch.nn.functional.linear(x, Wp, bp)
        out = mem.view(Cy.addr, (N, M), (Cy.pitch, 4), torch.float32)
        if name == "a_out+euler":
            y = out + DT * y
        elif c.flags & C.F_ADD_D:
            y = y + mem.view(D.addr, (N, M), (D.pitch, 4), torch.float32)
        elif c.flags & C.F_ACC:
            y = out + y
        mem.wview(Cy.addr, (N, M), (Cy.pitch, 4), torch.float32).copy_(y)
    mem.flush(Cy.addr, _extent(Cy, 1, N, 4 * M))


def release_policy_weights(policy) -> int:
    """Free every parameter the list path no longer needs (the board has 4 GB): all of them but the language
    embedding table (lang_rows). Returns the number of values released."""
    keep = policy.model.vlm_with_expert.get_vlm_model().text_model.get_input_embeddings().weight
    n = 0
    for p in policy.parameters():
        if p is keep or p.numel() == 0:
            continue
        n += p.numel()
        p.data = torch.empty(0, dtype=p.dtype, device=p.device)
    return n


def predict_action_chunk(policy, rt: ListRuntime, batch: dict, noise=None, timings: dict | None = None):
    """SmolVLAPolicy.predict_action_chunk through the list, without the policy's shared observation queues
    (populate_queues / self._queues): every call keeps its own tensors, so two calls can be in flight at once
    (design study 6, 3). Same numbers as the policy's own for n_obs_steps 1 (the queue holds just this batch:
    prepare_images / prepare_state then take [:, -1] of a 1-step stack, i.e. the tensor itself)."""
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    t0 = time.perf_counter()
    with torch.no_grad():
        b = policy._prepare_batch(dict(batch))
        images, img_masks = policy.prepare_images(b)
        state = policy.prepare_state(b)
        lang_tokens, lang_masks = b[OBS_LANGUAGE_TOKENS], b[OBS_LANGUAGE_ATTENTION_MASK]
        t_prep = time.perf_counter() - t0
        actions, t = rt.sample_actions(images, img_masks, lang_tokens, lang_masks, state, noise)
        t1 = time.perf_counter()
        actions = actions[:, :, : policy.config.action_feature.shape[0]]
        if policy.config.adapt_to_pi_aloha:
            actions = policy._pi_aloha_encode_actions(actions)
    t["prep"] = t_prep
    t["post"] = time.perf_counter() - t1
    t["total"] = time.perf_counter() - t0
    # host time outside the FPGA: everything but the IP runs and the waits for another inference (a free
    # arena, the engine busy with another arena's list)
    t["host_outside_fpga"] = t["total"] - t["fpga"] - t["wait_engine"] - t["wait_arena"]
    if timings is not None:
        timings.update(t)
    return actions


def install(policy, rt: ListRuntime) -> None:
    """policy.predict_action_chunk -> the list (what LeRobot's policy server calls); the last call's timings
    in last_timings(policy) (per thread)."""
    import types

    local = threading.local()

    def fwd(self, batch, noise=None, **kw):
        t = {}
        out = predict_action_chunk(self, rt, batch, noise=noise, timings=t)
        local.t = t
        return out

    policy.predict_action_chunk = types.MethodType(fwd, policy)
    policy._fpga_list_runtime = rt
    policy._fpga_list_local = local


def last_timings(policy) -> dict:
    return getattr(policy._fpga_list_local, "t", {})


def status_words(raw: bytes) -> list[int]:
    return list(struct.unpack("<12I", raw[: C.STATUS_BYTES]))


# ---------------------------------------------------------------- replay and server helpers
def replay(policy, frames: list[dict], verbose: bool = True) -> list[dict]:
    """Each replay frame (fpga/board/run_smolvla.py's frame_*.pt: batch, noise, ref_chunk_raw) through the
    installed list path, one after the other: the chunk, its max |chunk - ref| and the timings."""
    rows = []
    for d in frames:
        t0 = time.perf_counter()
        ch = policy.predict_action_chunk(dict(d["batch"]), noise=d["noise"])
        wall = time.perf_counter() - t0
        t = dict(last_timings(policy))
        st = t.pop("status", {})
        row = {
            "frame": int(d["frame"]),
            "max_abs_diff_normalized": float((ch.float() - d["ref_chunk_raw"].float()).abs().max()),
            "seconds_per_chunk": round(wall, 3),
            "fpga_seconds": round(t["fpga"], 3),
            "host_outside_fpga_ms": round(1e3 * t["host_outside_fpga"], 2),
            "host_ms": {
                k: round(1e3 * t[k], 2) for k in ("prep", "inputs", "read", "post", "arm_gemm") if k in t
            },
            "arena": t.get("arena"),
            "list_starts": st.get("list_starts"),
            "status": st.get("code"),
            "commands_done": st.get("done_all_starts"),
        }
        rows.append(row | {"_chunk": ch})
        if verbose:
            print(json.dumps(row), flush=True)
    return rows


def check_inflight(policy, frames: list[dict], expect: list[torch.Tensor], k: int = 2) -> dict:
    """The same frames with k calls in flight at once (k threads, one frame each, all started together):
    every chunk must equal the one-at-a-time chunk bit for bit (each in-flight inference has its own arena
    and tensors), and the arenas must overlap in time."""
    import concurrent.futures as cf

    barrier = threading.Barrier(min(k, len(frames)))
    spans: list[tuple] = [None] * len(frames)

    def one(i):
        d = frames[i]
        if i < k:
            barrier.wait()
        t0 = time.perf_counter()
        ch = policy.predict_action_chunk(dict(d["batch"]), noise=d["noise"])
        t = last_timings(policy)
        spans[i] = (
            t0,
            time.perf_counter(),
            t.get("arena"),
            t.get("wait_arena", 0.0),
            t.get("wait_engine", 0.0),
        )
        return ch

    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=k) as pool:
        chunks = list(pool.map(one, range(len(frames))))
    wall = time.perf_counter() - t0
    equal = [bool(torch.equal(a, b)) for a, b in zip(chunks, expect, strict=True)]
    return {
        "inflight": k,
        "frames": [int(d["frame"]) for d in frames],
        "bit_equal_to_sequential": equal,
        "arenas": [s[2] for s in spans],
        "wait_arena_s": [round(s[3], 3) for s in spans],
        "wait_engine_s": [round(s[4], 3) for s in spans],
        "wall_s": round(wall, 2),
        "ok": all(equal),
    }


def patch_policy_server(ps, inflight: int) -> None:
    """LeRobot's PolicyServer (async_inference/policy_server.py, 0.6.2) for `inflight` inferences at once
    (design study 6): the observation queue holds `inflight` observations instead of 1, and with inflight > 1
    the "too similar to the last processed observation" filter is skipped (the second observation of a pair is
    nearly identical to the first by design); the already-predicted-timestep check stays. gRPC already runs 4
    worker threads, so `inflight` <= 4 GetActions can be in predict_action_chunk at once."""
    from queue import Queue

    if inflight <= 1:
        return
    init, reset = ps.PolicyServer.__init__, ps.PolicyServer._reset_server

    def __init__(self, config):  # noqa: N807
        init(self, config)
        self.observation_queue = Queue(maxsize=inflight)

    def _reset_server(self):
        reset(self)
        self.observation_queue = Queue(maxsize=inflight)

    def _obs_sanity_checks(self, obs, previous_obs):
        with self._predicted_timesteps_lock:
            if obs.get_timestep() in self._predicted_timesteps:
                return False
        return True

    ps.PolicyServer.__init__ = __init__
    ps.PolicyServer._reset_server = _reset_server
    ps.PolicyServer._obs_sanity_checks = _obs_sanity_checks
    ps.PolicyServer._fpga_inflight = inflight
