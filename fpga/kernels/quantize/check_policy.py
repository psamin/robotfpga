"""Full-policy check of the int16 activation quantizer on checkpoint -12392's replay frames.

Every nn.Linear of SmolVLA gets its input quantized per token by the kernel's exact arithmetic
(qz_emul.quant_kernel, the integer datapath of quant_int16.h), and each call is compared with
precision_emulation.py's quant_sym (qz_emul.quant_ref) element by element. Modes per frame:
  fp32       untouched policy (sanity: should reproduce ref_chunk_raw)
  actq       x -> dequant(quant_kernel(x)) -> FP32 linear: the quantizer's own error
  int16      quant_kernel x int16 per-output-row weights, exact accumulate, y = acc * s_tok * s_row
             (the int16 engine scheme, precision_emulation V4_int16)
  int16_ref  the same with quant_ref: must give the identical chunk (bit for bit) when the kernel is exact
Errors are max |chunk - ref_chunk_raw| in normalized action units (gate 0.05) and degrees.

    python check_policy.py [--dump captured.qz] [--json out.json] [--modes fp32,actq,int16,int16_ref]
--dump writes 16 rows of every linear's first input (frame 0) with torch's q and s for tb_quant.
Also counts mismatches against fpga/hls/pack_float.py quantize_int16_tokens (the int16 engine's host packer).
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import struct
import sys
import time
import types
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qz_emul as qe  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
CKPT = ROOT / "build/ckpt/smolvla-12392/smolvla-so101_medicine_bottle_pickplace-12392-native"
FRAMES = ROOT / "build/ckpt/smolvla-12392/batches-12392"

MODE = {"m": "fp32", "check": False}
STATS = dict(calls=0, elems=0, q_mismatch=0, s_mismatch=0, packfloat_mismatch=0, rows=0, zero_rows=0, sat=0,
             cols=set())
DUMP = {"f": None, "seen": set(), "rows": 16, "records": 0}
try:  # read-only use of the engine's packer, to prove the kernel and the host agree
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "hls"))
    import pack_float as PF  # noqa: E402
except Exception as e:  # pragma: no cover
    print("pack_float not importable, skipping that comparison:", e)
    PF = None


class Patch:
    def __init__(self, name: str, mod: nn.Linear):
        self.name, self.w, self.b = name, mod.weight, mod.bias
        self.wq = None

    def wint16(self):
        if self.wq is None:  # precision_emulation.py int16 weights: per output row absmax
            w = self.w.detach().float()
            s = w.abs().amax(1, keepdim=True).clamp_min(qe.TINY) / qe.QMAX
            q = torch.round(w / s).clamp_(-qe.QMAX, qe.QMAX)
            self.wq = (q.to(torch.int16), s.squeeze(1).double())
        return self.wq

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        m = MODE["m"]
        if m == "fp32":
            return F.linear(x, self.w, self.b)
        shape = x.shape
        x2 = x.reshape(-1, shape[-1]).float().contiguous()
        if m == "int16_ref":
            q, s = qe.quant_ref(x2)
        else:
            q, s = qe.quant_kernel(x2)
            if MODE["check"]:
                qr, sr = qe.quant_ref(x2)
                zr = x2.abs().amax(1) == 0
                STATS["calls"] += 1
                STATS["elems"] += x2.numel()
                STATS["rows"] += x2.shape[0]
                STATS["q_mismatch"] += int((q != qr).sum())
                # quant_sym's scale on every row with a nonzero element; all-zero rows: 1.0 (pack_float)
                STATS["s_mismatch"] += int((qe.f32_bits(s[~zr]) != qe.f32_bits(sr[~zr])).sum())
                STATS["s_mismatch"] += int((s[zr] != 1.0).sum())
                STATS["zero_rows"] += int(zr.sum())
                if PF is not None:  # the int16 engine's host packer (fpga/hls/pack_float.py), bit for bit
                    qp, sp = PF.quantize_int16_tokens(x2.numpy())
                    STATS["packfloat_mismatch"] += int((torch.from_numpy(qp) != q).sum())
                    STATS["packfloat_mismatch"] += int((torch.from_numpy(sp.view("int32")) != s.view(torch.int32)).sum())
                STATS["sat"] += int((q.abs() == qe.QMAX).sum())
                STATS["cols"].add(x2.shape[1])
                if DUMP["f"] is not None and self.name not in DUMP["seen"]:
                    DUMP["seen"].add(self.name)
                    n = min(DUMP["rows"], x2.shape[0])
                    nm = self.name.encode()
                    f = DUMP["f"]
                    f.write(b"QZ01" + struct.pack("<iii", n, x2.shape[1], len(nm)) + nm)
                    f.write(x2[:n].numpy().tobytes())
                    f.write(qr[:n].numpy().tobytes())
                    f.write(sr[:n].contiguous().numpy().tobytes())
                    DUMP["records"] += 1
        if m == "actq":
            y = F.linear(qe.dequant(q, s), self.w, self.b)
        else:
            wq, sw = self.wint16()
            acc = q.double() @ wq.double().t()  # exact: |partial sums| < 2^44 < 2^53
            y = (acc * s.double()[:, None] * sw[None, :]).float()
            if self.b is not None:
                y = y + self.b
        return y.reshape(*shape[:-1], y.shape[-1])


def install(policy) -> int:
    n = 0
    for name, mod in policy.named_modules():
        if isinstance(mod, nn.Linear):
            p = Patch(name, mod)
            mod.forward = types.MethodType(lambda self, x, _p=p: _p(x), mod)
            n += 1
    return n


def action_std(ckpt: Path):
    from safetensors import safe_open

    f = ckpt / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
    if not f.exists():
        return None
    with safe_open(str(f), "pt") as sf:
        return sf.get_tensor("action.std").float()


def run(policy, frame: dict) -> torch.Tensor:
    policy.reset()
    with torch.no_grad():
        return policy.predict_action_chunk(copy.deepcopy(frame["batch"]), noise=frame["noise"].clone()).float()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=CKPT)
    ap.add_argument("--frames", type=Path, default=FRAMES)
    ap.add_argument("--modes", default="fp32,actq,int16,int16_ref")
    ap.add_argument("--dump", type=Path, default=None)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()
    torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))

    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    t0 = time.perf_counter()
    policy = SmolVLAPolicy.from_pretrained(str(args.ckpt)).to("cpu").eval().float()
    n_lin = install(policy)
    std = action_std(args.ckpt)
    print(f"loaded {args.ckpt.name} in {time.perf_counter() - t0:.1f} s, {n_lin} nn.Linear patched", flush=True)
    frames = [torch.load(f, weights_only=False) for f in sorted(args.frames.glob("frame_*.pt"))]
    modes = args.modes.split(",")
    if args.dump:
        DUMP["f"] = open(args.dump, "wb")
    out = dict(ckpt=str(args.ckpt), gate=0.05, frames=[])
    for fi, fr in enumerate(frames):
        ref = fr["ref_chunk_raw"].float()
        res = dict(frame=int(fr["frame"]))
        chunks = {}
        for m in modes:
            MODE["m"] = m
            MODE["check"] = m == "int16"
            if fi > 0:
                DUMP_f, DUMP["f"] = DUMP["f"], None
            t1 = time.perf_counter()
            c = run(policy, fr)
            if fi > 0:
                DUMP["f"] = DUMP_f
            chunks[m] = c
            err = (c - ref).abs()
            res[m] = dict(
                max_abs_err=float(err.max()),
                mean_abs_err=float(err.mean()),
                max_abs_err_deg=float((err * std).max()) if std is not None else None,
                seconds=round(time.perf_counter() - t1, 1),
            )
            if "fp32" in chunks and m != "fp32":
                res[m]["max_abs_vs_laptop_fp32"] = float((c - chunks["fp32"]).abs().max())
            print(f"frame {res['frame']:5d} {m:9s} max|err| {res[m]['max_abs_err']:.6f} "
                  f"({res[m]['max_abs_err_deg'] or 0:.4f} deg) mean {res[m]['mean_abs_err']:.2e} "
                  f"[{res[m]['seconds']} s]", flush=True)
            if fi == 0 and DUMP["f"] is not None and m == "int16":
                DUMP["f"].close()
                DUMP["f"] = None
        if "int16" in chunks and "int16_ref" in chunks:
            res["int16_vs_int16_ref_max_abs"] = float((chunks["int16"] - chunks["int16_ref"]).abs().max())
            res["int16_bitwise_equal_int16_ref"] = bool(torch.equal(chunks["int16"], chunks["int16_ref"]))
            print(f"frame {res['frame']:5d} kernel vs quant_sym chunk: max diff {res['int16_vs_int16_ref_max_abs']}, "
                  f"bitwise equal {res['int16_bitwise_equal_int16_ref']}", flush=True)
        out["frames"].append(res)
    st = dict(STATS)
    st["cols"] = sorted(st["cols"])
    out["activation_check"] = st
    out["dump_records"] = DUMP["records"]
    print(f"activations seen in int16 mode: {st['calls']} linear calls, {st['rows']} token rows, {st['elems']} elements, "
          f"{st['q_mismatch']} q / {st['s_mismatch']} s mismatches kernel vs quant_sym, {st['packfloat_mismatch']} vs pack_float; zero rows {st['zero_rows']}, "
          f"|q| = 32767 {st['sat']}; row lengths {st['cols']}", flush=True)
    if args.json:
        args.json.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
