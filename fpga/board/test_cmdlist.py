"""fpga/board/cmdlist.py without a board: the plan's placement into physical buffers and the relocation of
every record (shared weights once, one arena per inference in flight, nothing outside its allocation, both
lists valid), the register-level list start against the golden executor's walk (END, YIELD, step, NO_LEGACY),
the probe that refuses the per-call bitstream, the FP32 ARM GEMM of the 21 FP32-stored layers (epilogue,
padding, Euler step), the LeRobot server patch for two in flight, the language trim and the dead-layer fusion
rule. Needs numpy and torch (no checkpoint).

With FPGA_CKPT=<checkpoint dir> and FPGA_BATCHES=<dir of frame_*.pt> also the whole policy on frame 0 through
the list in exact mode (no quantization: the wiring check, <= 2e-5), with the 21 layers on the engine and on
the ARM, and two calls in flight against one at a time (bit-equal).

    <venv>/python -m pytest fpga/board/test_cmdlist.py -q
    <venv>/python fpga/board/test_cmdlist.py            # without pytest
"""

import os
import sys
import threading
import types
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fpga.board import cmdlist as CL  # noqa: E402
from fpga.board.runtime import fusion_groups, trim_tokens  # noqa: E402

K, C = CL.K, CL.C


def _placed(strict: bool, arenas: int = 2):
    plan, allocs = CL.build_plan()
    allocs.append(CL.Alloc(-(-(allocs[-1].addr + allocs[-1].nbytes) // 4096) * 4096, 4096, 4096, "list"))
    mem = CL.PhysMemory()
    pl = CL.Placement(allocs, CL.FakeCma(), mem, strict)
    shared = [i for i, a in enumerate(allocs) if a.region in ("weights", "consts")]
    per = [i for i, a in enumerate(allocs) if a.region in ("arena", "list")]
    phys0 = pl.place(shared)
    lists = []
    for _ in range(arenas):
        phys = dict(phys0) | pl.place(per)
        lists.append((phys, CL.relocate(plan.cmds, pl.relocator(phys))))
    return plan, allocs, mem, pl, shared, per, lists


def test_placement_and_relocation():
    for strict in (True, False):
        plan, allocs, mem, pl, shared, per, lists = _placed(strict)
        (p0, c0), (p1, c1) = lists
        assert all(p0[i] == p1[i] for i in shared)  # weights + constants: one copy
        assert all(p0[i] != p1[i] for i in per)  # activations: one copy per arena
        for cmds in (c0, c1):
            assert not K.validate_list(cmds)
            assert len(cmds) == len(plan.cmds) == 2701
        # every relocated slot address lies in a buffer, at the same offset inside its allocation
        n_slots = 0
        for v, r in zip(plan.cmds, c0, strict=True):
            for name, sl in v.s.items():
                if sl.zero():
                    continue
                i = int(np.searchsorted([a.addr for a in allocs], sl.addr, side="right")) - 1
                assert r.s[name].addr - p0[i] == sl.addr - allocs[i].addr
                mem._index(r.s[name].addr, 1)
                n_slots += 1
        assert n_slots > 10000
        # buffers: 4 KB aligned, inside the contract's 40-bit space, non-overlapping (add_buffer asserts)
        assert all(b.paddr % 4096 == 0 and b.paddr + b.nbytes < 1 << 40 for b in pl.buffers)
        if not strict:
            assert all(b.nbytes <= CL.SLAB_BYTES for b in pl.buffers)  # no allocation is larger than a slab
    print("placement: strict and slab-packed, 2 arenas, 2,701 records each valid")


def test_relocation_refuses_unplaced():
    plan, allocs, mem, pl, shared, per, lists = _placed(True, arenas=1)
    reloc = pl.relocator({i: 0x4000_0000 for i in shared})  # arena not placed
    try:
        CL.relocate(plan.cmds, reloc)
    except ValueError:
        return
    raise AssertionError("an arena address relocated without a placement")


def _mini():
    """A tiny memory and list: QUANT_X of 3 rows x 16 -> int16 rows + scales, COPY, END."""
    mem = CL.PhysMemory()
    cma = CL.FakeCma()
    buf = cma.allocate(1 << 16)
    mem.add_buffer(buf)
    a = buf.paddr
    x = np.arange(48, dtype=np.float32).reshape(3, 16) - 20.5
    mem.write(a, x.tobytes())
    cmds = [
        K.Cmd(
            C.OP_QUANT_X,
            C.F_QOUT,
            0,
            0,
            16,
            3,
            1,
            {"A": K.Slot(a, 64), "C": K.Slot(a + 4096, 32), "F": K.Slot(a + 8192)},
        ),
        K.Cmd(C.OP_COPY, C.F_YIELD, 1, 0, 16, 3, 1, {"A": K.Slot(a, 64), "C": K.Slot(a + 12288, 64)}),
        K.Cmd(C.OP_END, 0, 2),
    ]
    assert not K.validate_list(cmds)
    lst, status = a + 32768, a + 40960
    mem.write(lst, K.encode_list(cmds))
    return mem, x, cmds, lst, status, a


def test_fake_ip_walk_and_yield():
    mem, x, cmds, lst, status, a = _mini()
    ex = CL.Executor(mem, fe="torch")
    mmio = CL.FakeListMmio(ex)
    ip = CL.ListIp(mmio, poll_sleep=0)
    r = ip.run(lst, status, 0)
    assert r == (2 << 8) | C.ST_YIELD, hex(r)  # COPY carries F_YIELD: resume at 2
    q, s = K.quant_rows(x)
    got = mem.view(a + 4096, (3, 16), (32, 2), torch.int16).numpy()
    assert np.array_equal(got, q)
    assert np.array_equal(mem.view(a + 8192, (4,), (4,)).numpy(), np.append(s, np.float32(1.0)))
    assert np.array_equal(mem.view(a + 12288, (3, 16), (64, 4)).numpy(), x)
    r = ip.run(lst, status, 2)
    assert r == (2 << 8) | C.ST_END, hex(r)
    st = K.parse_status(mem.read(status, C.STATUS_BYTES))
    assert (
        st["magic_ok"] and st["version"] == C.CONTRACT_VERSION and st["code"] == "END" and st["starts"] == 2
    )
    r = ip.run(lst, status, 0, step=True)
    assert r == (1 << 8) | C.ST_STEP, hex(r)
    mmio.write(C.REG_M, 0x1234)  # no list magic: the IP refuses and touches nothing
    mmio.write(C.REG_AP_CTRL, 1)
    assert mmio.read(C.REG_AP_RETURN) == C.ST_NO_LEGACY
    for off, v in K.start_regs(lst, status, 0).items():
        mmio.write(off, v)
    mmio.write(C.REG_W0, 0x1000)  # list mode needs every pointer register at 0 (CONTRACT.md 3.1)
    try:
        mmio.write(C.REG_AP_CTRL, 1)
    except AssertionError:
        pass
    else:
        raise AssertionError("a non-zero pointer register was accepted")
    print("fake IP: END / YIELD / STEP / NO_LEGACY, data equal to the contract quantizer")


class _PerCallMmio:
    """The shipped per-call float_top: M = 0xA5... is out of its range, ap_return 1, memory untouched."""

    def __init__(self):
        self.regs, self.done = {}, False

    def read(self, off):
        if off == C.REG_AP_CTRL:
            v, self.done = 0x4 | (0x2 if self.done else 0), False
            return v
        return self.regs.get(off, 0)

    def write(self, off, v):
        self.regs[off] = v
        if off == C.REG_AP_CTRL and v & 1:
            self.regs[C.REG_AP_RETURN] = 1 if self.regs.get(C.REG_M, 0) >> 31 else 0
            self.done = True


def test_probe():
    mem, x, cmds, lst, status, a = _mini()
    before = mem.read(lst, len(cmds) * C.REC_BYTES)
    st = CL.probe_list_ip(CL.ListIp(CL.FakeListMmio(CL.Executor(mem)), poll_sleep=0), mem, lst, status)
    assert st["tag"] == 0xE0D and st["code"] == "END"
    assert mem.read(lst, len(cmds) * C.REC_BYTES) == before  # the list buffer is restored
    try:
        CL.probe_list_ip(CL.ListIp(_PerCallMmio(), poll_sleep=0), mem, lst, status)
    except RuntimeError as e:
        assert "per-call" in str(e)
    else:
        raise AssertionError("the per-call bitstream was not refused")
    from fpga.board.check_engine import list_probe  # the board-side probe (numpy only, PYNQ's Python)

    buf = CL.FakeCma().allocate(8192)
    nop = lambda off, n: None  # noqa: E731
    r = list_probe(_PerCallMmio(), buf.array, buf.paddr, nop, nop)
    assert not r["ok"] and "per-call" in r["why"], r
    mem2 = CL.PhysMemory()
    mem2.add_buffer(buf)
    r = list_probe(CL.FakeListMmio(CL.Executor(mem2)), buf.array, buf.paddr, nop, nop)
    assert r["ok"], r
    print("probe: list-mode accepted, per-call refused (cmdlist and check_engine --list-probe)")


def test_arm_gemm():
    mem = CL.PhysMemory()
    buf = CL.FakeCma().allocate(1 << 16)
    mem.add_buffer(buf)
    a = buf.paddr
    ex = CL.Executor(mem)
    rng = np.random.default_rng(1)
    N, Kc, M, rows = 5, 16, 64, 40  # 40 real output rows padded to 64
    xq = rng.integers(-32767, 32768, (N, Kc)).astype(np.int16)
    s = rng.uniform(1e-4, 1e-3, N).astype(np.float32)
    W = torch.from_numpy(rng.standard_normal((rows, Kc)).astype(np.float32))
    b = torch.from_numpy(rng.standard_normal(rows).astype(np.float32))
    D = rng.standard_normal((1, M)).astype(np.float32)  # pitch 0: one broadcast row (the time MLP)
    C0 = rng.standard_normal((N, M)).astype(np.float32)
    X, F, Y, Dd = a, a + 4096, a + 8192, a + 16384
    mem.write(X, xq.tobytes())
    mem.write(F, np.append(s, np.ones(3, np.float32)).tobytes())
    mem.write(Dd, D.tobytes())
    x = torch.from_numpy(xq).float() * torch.from_numpy(s)[:, None]
    Wp, bp = torch.zeros(M, Kc), torch.zeros(M)
    Wp[:rows], bp[:rows] = W, b
    base = torch.nn.functional.linear(x, Wp, bp)
    sl = {
        "A": K.Slot(X, 2 * Kc),
        "B": K.Slot(a + 32768, 4096),
        "C": K.Slot(Y, 4 * M),
        "E": K.Slot(a),
        "F": K.Slot(F),
    }
    cases = [
        ("plain", 0, None, base),
        ("add_d", C.F_ADD_D, None, base + torch.from_numpy(D)),
        ("acc", C.F_ACC, C0, torch.from_numpy(C0) + base),
        ("a_out+euler", C.F_ACC, C0, torch.from_numpy(C0) + CL.DT * base),
    ]
    for name, fl, c0, want in cases:
        s2 = dict(sl) | ({"D": K.Slot(Dd, 0)} if fl & C.F_ADD_D else {})
        c = K.Cmd(C.OP_GEMM, fl, 0, M, Kc, N, 1, s2)
        mem.write(Y, (c0 if c0 is not None else np.zeros((N, M), np.float32)).tobytes())
        CL.arm_gemm(mem, ex, c, W, b, name)
        got = mem.view(Y, (N, M), (4 * M, 4)).clone()
        assert torch.equal(got, want), (name, float((got - want).abs().max()))
        assert torch.equal(got[:, rows:], want[:, rows:])
    print("ARM GEMM: plain / + D broadcast / += C / Euler, FP32 bit-equal to torch")


def test_server_patch():
    from queue import Queue

    class PolicyServer:
        def __init__(self, config):
            self.observation_queue = Queue(maxsize=1)
            self._predicted_timesteps_lock = threading.Lock()
            self._predicted_timesteps = {7}

        def _reset_server(self):
            self.observation_queue = Queue(maxsize=1)

        def _obs_sanity_checks(self, obs, previous_obs):
            return False  # stands for "too similar"

    ps = types.SimpleNamespace(PolicyServer=PolicyServer)
    CL.patch_policy_server(ps, 2)
    srv = ps.PolicyServer(None)
    assert srv.observation_queue.maxsize == 2
    srv._reset_server()
    assert srv.observation_queue.maxsize == 2

    class Obs:
        def __init__(self, t):
            self.t = t

        def get_timestep(self):
            return self.t

    assert srv._obs_sanity_checks(Obs(8), Obs(8)) is True  # the similarity filter is off
    assert srv._obs_sanity_checks(Obs(7), Obs(6)) is False  # an already predicted timestep still is dropped
    print("server patch: queue 2, similarity filter off, predicted-timestep check kept")


def test_trim_tokens_and_dead_q_fusion():
    tok = torch.tensor([[5, 6, 7, 0, 0], [5, 6, 0, 0, 0]])
    msk = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]], dtype=torch.bool)
    t2, m2 = trim_tokens(tok, msk)
    assert t2.tolist() == [[5, 6, 7], [5, 6, 0]] and m2.tolist() == [[True, True, True], [True, True, False]]
    t3, m3 = trim_tokens(t2[:1], m2[:1])
    assert t3 is t2[:1] or t3.tolist() == [[5, 6, 7]]
    from torch import nn

    from fpga.board.runtime import _ZerosModule

    attn = nn.Module()
    attn.q_proj, attn.k_proj, attn.v_proj = _ZerosModule(8, torch.float32), nn.Linear(4, 2), nn.Linear(4, 2)
    root = nn.Module()
    root.self_attn = attn
    assert fusion_groups(root) == [["self_attn.k_proj", "self_attn.v_proj"]]
    assert attn.q_proj(torch.ones(3, 4)).shape == (3, 8)
    print("language trim: pad columns no row uses are dropped; dead q: k/v fused as a pair")


def test_policy_end_to_end():
    ck, bd = os.environ.get("FPGA_CKPT"), os.environ.get("FPGA_BATCHES")
    if not ck or not bd:
        print("SKIP test_policy_end_to_end (FPGA_CKPT / FPGA_BATCHES not set)")
        return
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    d = torch.load(sorted(Path(bd).glob("frame_*.pt"))[0], weights_only=False)
    for arm in (False, True):
        p = SmolVLAPolicy.from_pretrained(ck).to("cpu").eval()
        rt = CL.ListRuntime(p, backend="mock", exact=True, fe="torch", arm_fp32=arm, verbose=False)
        CL.install(p, rt)
        rows = CL.replay(p, [d, d], verbose=False)
        e = rows[0]["max_abs_diff_normalized"]
        assert e <= 2e-5, e  # wiring check (exact mode); -12392 frame 0: 1.56e-5 engine, 1.65e-5 ARM
        assert torch.equal(rows[0]["_chunk"], rows[1]["_chunk"])  # arena 0 and arena 1 give the same bits
        assert rows[0]["list_starts"] == (58 if arm else 1)
        inf = CL.check_inflight(p, [d, d], [rows[0]["_chunk"]] * 2, 2)
        assert inf["ok"] and sorted(inf["arenas"]) == [0, 1], inf
        rt.close()
        print(
            f"policy frame 0 exact, 21 layers on the {'ARM' if arm else 'engine'}: {e:.3e};"
            " 2 in flight bit-equal"
        )


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}", flush=True)
    print("ALL PASS")
