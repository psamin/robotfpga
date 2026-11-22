"""End-to-end check of the integrated float_top C model (float_seq.h, built as build/seq_shim.dll) on SmolVLA
-12392: the contract's chunk list (fpga/kernels/contract_plan.py via sequencer/build_smolvla.Program) runs
command by command, in lockstep, through

  golden  fpga/kernels/sequencer/exec_ref.Executor in int16k mode: the contract arithmetic with every kernel's
          torch emulator (nr_emul, gs_emu NSB 10, rope_emu, smx_emu, contract.quant_rows / gemm_out)
  C model the HLS sources of float_top (sequencer walk, GEMM unit, the kernels' mode functions) compiled
          for the host, on its own copy of the DDR image, started with the hardware's register protocol (step
          mode)

After every command the C model's image is compared with the golden memory, byte for byte, over every region
the list writes (arena, status block; the cycle words of the status block excepted). The C model's x_t then
goes through predict_action_chunk (unnormalize) and is compared with ref_chunk_raw; so is the golden x_t.

    HF_HUB_OFFLINE=1 python fpga/hls/e2e_seq.py <ckpt dir> <batches dir> [--frames 0,3563,7489] [--out f.json]
"""

from __future__ import annotations

import argparse
import ctypes
import json
import pathlib
import struct
import sys
import time

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
KDIR = HERE.parent / "kernels"
sys.path.insert(0, str(KDIR / "sequencer"))
sys.path.insert(0, str(KDIR))  # contract.py (build_smolvla adds it as well)
import contract as K  # noqa: E402
from build_smolvla import Program  # noqa: E402
from contract import C  # noqa: E402


class Shim:
    """The C model: one flat image [base, base + size) holding every region of the program's memory."""

    def __init__(self, dll: pathlib.Path, mem):
        self.lib = ctypes.CDLL(str(dll))
        self.lib.seq_image.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64]
        self.lib.seq_start.argtypes = [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32]
        self.lib.seq_start.restype = ctypes.c_uint32
        self.lib.seq_cmp.argtypes = [ctypes.c_uint64, ctypes.c_void_p, ctypes.c_uint64]
        self.lib.seq_cmp.restype = ctypes.c_uint64
        self.lib.seq_decode_mismatches.restype = ctypes.c_uint64
        self.lib.seq_leftover.restype = ctypes.c_long
        self.base = min(b for b, _, _ in mem.regions)
        end = max(b + t.numel() for b, t, _ in mem.regions)
        self.img = np.zeros(end - self.base + 4096, dtype=np.uint8)
        self.lib.seq_image(self.img.ctypes.data, self.base, self.img.size)

    def load(self, mem, regions=None) -> None:
        for b, t, _ in mem.regions:
            if regions is None or b in regions:
                self.img[b - self.base : b - self.base + t.numel()] = t.numpy()

    def cmp(self, base: int, t: torch.Tensor) -> int:
        n = t.numel()
        return int(self.lib.seq_cmp(base, t.data_ptr(), n))

    def f32(self, addr: int, n: int) -> np.ndarray:
        o = addr - self.base
        return self.img[o : o + 4 * n].view(np.float32).copy()


def first_diff_info(shim: Shim, base: int, t: torch.Tensor, off: int) -> dict:
    """Count differing bytes of a region and the largest FP32 difference in a 4 KB window at the first one."""
    o = base - shim.base
    a = shim.img[o : o + t.numel()]
    g = t.numpy()
    nd = int((a != g).sum())
    w0 = max(0, (off // 4) * 4 - 2048)
    w1 = min(t.numel() // 4 * 4, w0 + 4096)
    af = a[w0:w1].view(np.float32).astype(np.float64)
    gf = g[w0:w1].view(np.float32).astype(np.float64)
    with np.errstate(invalid="ignore"):
        dmax = float(np.nanmax(np.abs(af - gf))) if w1 > w0 else 0.0
    return {"bytes_differ": nd, "first_off": off, "fp32_maxdiff_near_first": dmax}


class LockstepProgram(Program):
    def __init__(self, policy, shim_dll: pathlib.Path, **kw):
        super().__init__(policy, **kw)
        self.shim = Shim(shim_dll, self.mem)
        self.shim.load(self.mem)  # weights, constants, list, arena (the weights never change)
        lay = self.P.layout
        self.weights_base = lay["regions"]["weights"][0] if "weights" in lay["regions"] else None
        self.cmp_regions = [b for b, _, _ in self.mem.regions if b != self.weights_base]
        self.log: list[dict] = []
        self.golden_xt = None

    def run(self, images, img_masks, lang_tokens, lang_masks, state, noise) -> torch.Tensor:
        # the ARM's part exactly as Program.run, then both memories walk the list in step mode
        assert len(images) == 2 and all(bool(mk.all()) for mk in img_masks)
        CP = sys.modules["contract_plan"]
        with torch.no_grad():
            pt = torch.stack(
                [
                    im[0].float().reshape(3, 16, 16, 16, 16).permute(1, 3, 0, 2, 4).reshape(256, 768)
                    for im in images
                ]
            )
            self._put_qin(self.find("vision", "patch", C.OP_GEMM), "A", "F", pt)
            self._put_qin(
                self.find("connector", "state", C.OP_GEMM), "A", "F", state.float().reshape(1, 1, -1)
            )
            lr = self.lang_rows(lang_tokens, lang_masks)
            self.mem.write(
                self.find("connector", "lang", C.OP_COPY).slot("A").addr,
                np.ascontiguousarray(lr.numpy(), dtype=np.float32).tobytes(),
            )
            xa = self.find("steps", "x_t", C.OP_QUANT_X).slot("A")
            self.mem.write(
                xa.addr, np.ascontiguousarray(noise[0].float().numpy(), dtype=np.float32).tobytes()
            )
            self.shim.load(
                self.mem, set(self.cmp_regions)
            )  # this chunk's inputs (and the arena as the golden has it)
            lst, stat = self.list_addr, self.status
            idx, ncmd, nbad, first_bad = 0, 0, 0, None
            t_g = t_c = 0.0
            regs = {b: t for b, t, _ in self.mem.regions if b in self.cmp_regions}
            while True:
                M = K.start_regs(lst, stat, idx)[C.REG_M] | (1 << C.MODE_STEP_BIT)
                t0 = time.perf_counter()
                rg = self.ex.start(M, lst >> 4, stat >> 4)
                t1 = time.perf_counter()
                rc = self.shim.lib.seq_start(M, lst >> 4, stat >> 4)
                t2 = time.perf_counter()
                t_g += t1 - t0
                t_c += t2 - t1
                # the status block's cycle words: the C model counts host-model clocks, the golden writes 0
                so = stat - self.shim.base
                self.shim.img[so + 40 : so + 48] = 0
                rec = struct.unpack("<64I", self.mem.read(lst + idx * C.REC_BYTES, C.REC_BYTES))
                op = rec[C.FLD_OP] & 0xFF
                bad = []
                for b, t in regs.items():
                    off = self.shim.cmp(b, t)
                    if off != t.numel():
                        bad.append((b, off))
                ok = rg == rc and not bad
                ncmd += 1
                if not ok:
                    nbad += 1
                    if first_bad is None or nbad <= 20:
                        info = {
                            "index": idx,
                            "op": C.opnames.get(op, hex(op)) if hasattr(C, "opnames") else op,
                            "tag": rec[C.FLD_TAG],
                            "ret_golden": hex(rg),
                            "ret_c": hex(rc),
                            "regions": [
                                {"region": hex(b), **first_diff_info(self.shim, b, regs[b], off)}
                                for b, off in bad
                            ],
                        }
                        self.log.append(info)
                        if first_bad is None:
                            first_bad = info
                            print("  first mismatch:", json.dumps(info), flush=True)
                code, idx = rg & 0xFF, rg >> 8
                if code != C.ST_STEP:
                    break
        assert code == C.ST_END and idx == len(self.cmds) - 1, (hex(rg), idx)
        st = K.parse_status(self.mem.read(stat, C.STATUS_BYTES))
        self.last_status = st | {"list_starts": ncmd}
        shape = (1, CP.TQ_E, CP.A_D)
        self.golden_xt = self.mem.view(xa.addr, shape, (CP.TQ_E * 4 * CP.A_D, 4 * CP.A_D, 4)).clone()
        cxt = torch.from_numpy(self.shim.f32(xa.addr, CP.TQ_E * CP.A_D)).reshape(shape)
        self.stats = {
            "commands": ncmd,
            "commands_differing": nbad,
            "first_mismatch": first_bad,
            "golden_s": round(t_g, 1),
            "c_model_s": round(t_c, 1),
            "decode_mismatches": int(self.shim.lib.seq_decode_mismatches()),
            "nr_leftover": int(self.shim.lib.seq_leftover()),
            "xt_c_vs_golden_maxabs": float((cxt - self.golden_xt).abs().max()),
            "xt_bit_equal": bool(
                torch.equal(cxt.view(torch.int32), self.golden_xt.contiguous().view(torch.int32))
            ),
        }
        return cxt


def make_sampler(prog: LockstepProgram):
    """VLAFlowMatching.sample_actions for one frame: the first call runs the lockstep and returns the C
    model's x_t, the second the golden x_t (both through the policy's own post-processing)."""
    cache: dict = {}

    def sample_actions(images, img_masks, lang_tokens, lang_masks, state, noise=None, **kw):
        if "golden" in cache:
            return cache["golden"]
        xt = prog.run(images, img_masks, lang_tokens, lang_masks, state, noise)
        cache["golden"] = prog.golden_xt
        return xt

    return sample_actions


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("bdir")
    ap.add_argument("--dll", default=str(HERE / "build" / "seq_shim.dll"))
    ap.add_argument("--frames", default="")
    ap.add_argument("--base", type=lambda s: int(s, 0), default=0x8_0000_0000)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--nsb", type=int, default=10)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    p = SmolVLAPolicy.from_pretrained(a.ckpt).to("cpu").eval().float()
    frames = [torch.load(f, weights_only=False) for f in sorted(pathlib.Path(a.bdir).glob("frame_*.pt"))]
    if a.frames:
        keep = {int(x) for x in a.frames.split(",")}
        frames = [d for d in frames if int(d["frame"]) in keep]
    t0 = time.perf_counter()
    prog = LockstepProgram(p, pathlib.Path(a.dll), base=a.base, exact=False, fe="kernels", nsb=a.nsb)
    mb = prog.shim.img.size / 2**20
    print(f"{len(prog.cmds)} records, image {mb:.0f} MB ({time.perf_counter() - t0:.1f} s)")
    orig = p.model.sample_actions
    res = {"base": hex(a.base), "nsb": a.nsb, "records": len(prog.cmds), "frames": []}
    try:
        for d in frames:
            p.model.sample_actions = make_sampler(prog)
            p.reset()
            t1 = time.perf_counter()
            with torch.no_grad():
                ch_c = p.predict_action_chunk(dict(d["batch"]), noise=d["noise"])
                p.reset()
                ch_g = p.predict_action_chunk(dict(d["batch"]), noise=d["noise"])
            ref = d["ref_chunk_raw"].float()
            r = {
                "frame": int(d["frame"]),
                "c_model_vs_ref_maxabs": float((ch_c.float() - ref).abs().max()),
                "golden_vs_ref_maxabs": float((ch_g.float() - ref).abs().max()),
                "c_model_vs_golden_chunk_maxabs": float((ch_c.float() - ch_g.float()).abs().max()),
                **prog.stats,
                "wall_s": round(time.perf_counter() - t1, 1),
            }
            res["frames"].append(r)
            print(json.dumps(r), flush=True)
    finally:
        p.model.sample_actions = orig
    res["mismatch_log"] = prog.log[:40]
    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
