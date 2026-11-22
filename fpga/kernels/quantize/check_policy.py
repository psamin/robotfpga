"""Full-policy check of the row unit's quantizer (QUANT_X / QUANT_W / QUANT_WT, quant_int16.h) under the
integration contract (fpga/kernels/CONTRACT.md 2), on checkpoint -12392's 3 replay frames.

The kernel's integer datapath (qz_emul.quant_kernel, bit-exact with quant_int16.h by tb_quant) quantizes
every int16 operand the contract's row unit produces, and every call is compared element by element with the
contract reference (contract.quant_rows / attn_int16.quant_c). Modes per frame:
  fp32        untouched policy (sanity: reproduces ref_chunk_raw)
  actq        every nn.Linear input -> dequant(kernel(x)) -> FP32 linear: the activation quantizer's own error
  c_lin       every nn.Linear through the contract GEMM: X = kernel(x) per token (QUANT_X, or the shared
              quantizer of NORM / GELU / SILU), W = the contract quantizer per output row (pack_float),
              epilogue y = fadd(fmul(fmul(fl(acc), s_row), s_tok), bias) with sign-keeping FTZ
              (contract.gemm_out);
              attention FP32
  c_lin_ref   c_lin with the contract reference quantizer: must give the identical chunk
  c_full      c_lin + every attention through the contract (attn_int16.attention_contract): Q per (head,
              query) with p4 = -3 (QUANT_X with p1 = 64 at the vision site; ROPE F_QOUT elsewhere),
              K per (KV head, key) (QUANT_W / ROPE F_KW), V^T per (KV head, channel) over each part's keys
              (QUANT_WT), all by the kernel's datapath; P from the softmax unit's emulation (smx_emu), the
              rank masks, the GEMM epilogue
  c_full_ref  c_full with the contract reference quantizer everywhere: must give the identical chunk
Errors: max |chunk - ref_chunk_raw| in normalized action units (gate 0.05) and degrees.
Not modelled here (other units' front ends, CONTRACT.md 2.2): the NORM / GELU / SILU / ROPE FP32 emulators,
the packer's folded constants (2.4) and the residual add inside the epilogue; torch FP32 computes those.

    HF_HUB_OFFLINE=1 python check_policy.py [--modes ...] [--json out.json] [--dump captured.qz]
--dump writes 16 rows of every linear's first input (frame 0) with the reference's q and s for tb_quant.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
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

HERE = Path(__file__).resolve().parent
KERNELS = HERE.parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
import qz_emul as qe  # noqa: E402

CKPT = ROOT / "build/ckpt/smolvla-12392/smolvla-so101_medicine_bottle_pickplace-12392-native"
FRAMES = ROOT / "build/ckpt/smolvla-12392/batches-12392"

# The attention site patches and the contract attention model (fpga/kernels/attention, used read-only). Loaded
# under another module name: that directory's check_policy.py would shadow this one.
_spec = importlib.util.spec_from_file_location("attn_check_policy", KERNELS / "attention" / "check_policy.py")
ACP = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ACP)
A = ACP.A  # attn_int16
QUANT_C_REF = A.quant_c  # the contract reference quantizer (torch), = contract.quant_rows
try:  # read-only use of the engine's packer, to prove the kernel and the host agree
    sys.path.insert(0, str(ROOT / "fpga" / "hls"))
    import pack_float as PF  # noqa: E402
except Exception as e:  # pragma: no cover
    print("pack_float not importable, skipping that comparison:", e)
    PF = None

MODE = {"m": "fp32", "kernel": True, "check": False}
STATS: dict = {}
DUMP = {"f": None, "seen": set(), "rows": 16, "records": 0}


def stat(kind: str) -> dict:
    return STATS.setdefault(
        kind, dict(calls=0, rows=0, elems=0, q_mismatch=0, s_mismatch=0, zero_rows=0, sat=0, cols=set())
    )


def kernel_quant(x: torch.Tensor, dim: int = -1, shift: int = 0):
    """The row unit's quantizer along `dim` (the kernel's integer datapath), attn_int16.quant_c's interface:
    (q integer-valued FP32, stored scale FP32 keepdim). Stored scale as quant_int16.h stored_scale: the
    exponent of s plus p4 (an integer add on the bits), 1.0 for an all-zero row."""
    x = x.to(torch.float32)
    xm = x.movedim(dim, -1)
    shp = xm.shape
    x2 = xm.reshape(-1, shp[-1]).contiguous()
    q, s = qe.quant_kernel(x2)
    zero = (qe.f32_bits(x2) & 0x7FFFFFFF).amax(1) == 0
    sb = qe.f32_bits(s) + (shift << 23)
    stored = torch.where(zero, torch.ones_like(s), qe.bits_f32(sb))
    return q, stored, x2, zero, shp


def quant_c_kernel(x: torch.Tensor, dim: int = -1, shift: int = 0):
    """attn_int16.quant_c replaced by the kernel (c_full): Q (shift -3), K and V^T of every attention."""
    q, stored, x2, zero, shp = kernel_quant(x, dim, shift)
    if MODE["check"]:  # against the contract reference, bit for bit
        qr, sr = QUANT_C_REF(x2, -1, shift)
        st = stat("attn Q (p4 = -3)" if shift == -3 else "attn K / V^T")
        st["calls"] += 1
        st["rows"] += x2.shape[0]
        st["elems"] += x2.numel()
        st["q_mismatch"] += int((q.float() != qr).sum())
        st["s_mismatch"] += int((qe.f32_bits(stored) != qe.f32_bits(sr[:, 0])).sum())
        st["zero_rows"] += int(zero.sum())
        st["sat"] += int((q.abs() == qe.QMAX).sum())
        st["cols"].add(x2.shape[1])
    qf = q.float().reshape(shp).movedim(-1, dim)
    sf = stored.reshape(*shp[:-1], 1).movedim(-1, dim)
    return qf, sf


class Patch:
    def __init__(self, name: str, mod: nn.Linear):
        self.name, self.w, self.b = name, mod.weight, mod.bias
        self.wq = None

    def wint16(self):
        if self.wq is None:  # the contract quantizer per output row (pack_float._quantize_int16)
            q, s = QUANT_C_REF(self.w.detach().float(), -1)
            self.wq = (q, s[:, 0].contiguous())
        return self.wq

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        m = MODE["m"]
        if m == "fp32":
            return F.linear(x, self.w, self.b)
        shape = x.shape
        x2 = x.reshape(-1, shape[-1]).float().contiguous()
        if MODE["kernel"]:
            qi, s, _, zero, _ = kernel_quant(x2)
            q = qi.float()
            if MODE["check"]:
                qr, sr = QUANT_C_REF(x2, -1)
                sr = sr[:, 0]
                st = stat("linear inputs (QUANT_X / q-out)")
                st["calls"] += 1
                st["elems"] += x2.numel()
                st["rows"] += x2.shape[0]
                st["q_mismatch"] += int((q != qr).sum())
                st["s_mismatch"] += int((qe.f32_bits(s) != qe.f32_bits(sr)).sum())
                st["zero_rows"] += int(zero.sum())
                st["sat"] += int((qi.abs() == qe.QMAX).sum())
                st["cols"].add(x2.shape[1])
                if PF is not None:  # the int16 engine's host packer (fpga/hls/pack_float.py), bit for bit
                    qp, sp = PF.quantize_int16_tokens(x2.numpy())
                    st["packfloat_mismatch"] = (
                        st.get("packfloat_mismatch", 0)
                        + int((torch.from_numpy(qp) != qi).sum())
                        + int((torch.from_numpy(sp.view("int32")) != s.view(torch.int32)).sum())
                    )
                if DUMP["f"] is not None and self.name not in DUMP["seen"]:
                    DUMP["seen"].add(self.name)
                    n = min(DUMP["rows"], x2.shape[0])
                    nm = self.name.encode()
                    f = DUMP["f"]
                    f.write(b"QZ01" + struct.pack("<iii", n, x2.shape[1], len(nm)) + nm)
                    f.write(x2[:n].numpy().tobytes())
                    f.write(qr[:n].to(torch.int16).numpy().tobytes())
                    f.write(sr[:n].contiguous().numpy().tobytes())
                    DUMP["records"] += 1
        else:
            q, s = QUANT_C_REF(x2, -1)
            s = s[:, 0]
        if m == "actq":
            y = F.linear(q * s[:, None], self.w, self.b)
        else:  # the contract GEMM: exact integer accumulate, then the FP32 epilogue
            wq, sw = self.wint16()
            acc = q.double() @ wq.double().t()  # exact: |acc| <= K 2^30 < 2^53
            y = A.fmul_hw(A.fmul_hw(acc.float(), sw[None, :]), s[:, None])
            if self.b is not None:
                y = A.fadd_hw(y, self.b.detach().float()[None, :])
        return y.reshape(*shape[:-1], y.shape[-1])


def install(policy) -> tuple[int, int]:
    n = 0
    for name, mod in policy.named_modules():
        if isinstance(mod, nn.Linear):
            p = Patch(name, mod)
            mod.forward = types.MethodType(lambda self, x, _p=p: _p(x), mod)
            n += 1
    return n, ACP.install(policy)


def local_vlm(name: str) -> str | None:
    """The HF cache snapshot of the VLM backbone that holds its weights (model.safetensors), or None.
    Offline, transformers resolves the name through refs/main, which can point at a snapshot without the
    weights (an interrupted online download); this is the snapshot every earlier result used
    (gelu_silu/policy_check.py does the same; the fp32 mode's ~5e-6 against ref_chunk_raw confirms it)."""
    if os.path.isdir(name):
        return None
    hub = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME", os.path.join(Path.home(), ".cache", "huggingface")), "hub"
    )
    snaps = sorted(Path(hub, "models--" + name.replace("/", "--"), "snapshots").glob("*/model.safetensors"))
    return str(snaps[0].parent) if snaps else None


def action_std(ckpt: Path):
    from safetensors import safe_open

    f = ckpt / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
    if not f.exists():
        return None
    with safe_open(str(f), "pt") as sf:
        return sf.get_tensor("action.std").float()


def set_mode(m: str, check: bool) -> None:
    MODE["m"] = {"c_lin_ref": "c_lin", "c_full": "c_lin", "c_full_ref": "c_lin"}.get(m, m)
    MODE["kernel"] = not m.endswith("_ref")
    MODE["check"] = check
    full = m in ("c_full", "c_full_ref")
    ACP.STATE["variant"] = ACP.VARIANTS["contract"] if full else None
    ACP.STATE["record"] = False
    A.quant_c = quant_c_kernel if m == "c_full" else QUANT_C_REF


def run(policy, frame: dict) -> torch.Tensor:
    policy.reset()
    ACP.STATE["prefix_len"] = None
    ACP.CALLS.clear()
    with torch.no_grad():
        return policy.predict_action_chunk(
            copy.deepcopy(frame["batch"]), noise=frame["noise"].clone()
        ).float()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=CKPT)
    ap.add_argument("--frames", type=Path, default=FRAMES)
    ap.add_argument("--modes", default="fp32,actq,c_lin,c_lin_ref,c_full,c_full_ref")
    ap.add_argument("--dump", type=Path, default=None)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()
    torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    t0 = time.perf_counter()
    cfg = PreTrainedConfig.from_pretrained(str(args.ckpt))
    vlm = local_vlm(cfg.vlm_model_name)
    if vlm:
        print(f"VLM backbone from the local snapshot {vlm}", flush=True)
        cfg.vlm_model_name = vlm
    policy = SmolVLAPolicy.from_pretrained(str(args.ckpt), config=cfg).to("cpu").eval().float()
    n_lin, n_att = install(policy)
    std = action_std(args.ckpt)
    print(
        f"loaded {args.ckpt.name} in {time.perf_counter() - t0:.1f} s, {n_lin} nn.Linear + {n_att} vision "
        "attentions + the LM / expert attention patched",
        flush=True,
    )
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
            set_mode(m, check=m in ("c_lin", "c_full"))
            dump_f = DUMP["f"]
            if fi > 0 or m != "c_lin":
                DUMP["f"] = None
            t1 = time.perf_counter()
            c = run(policy, fr)
            DUMP["f"] = dump_f
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
            print(
                f"frame {res['frame']:5d} {m:10s} max|err| {res[m]['max_abs_err']:.6f} "
                f"({res[m]['max_abs_err_deg'] or 0:.4f} deg) mean {res[m]['mean_abs_err']:.2e} "
                f"[{res[m]['seconds']} s]",
                flush=True,
            )
            if fi == 0 and m == "c_lin" and DUMP["f"] is not None:
                DUMP["f"].close()
                DUMP["f"] = None
        for k in ("c_lin", "c_full"):
            if k in chunks and k + "_ref" in chunks:
                eq = bool(torch.equal(chunks[k], chunks[k + "_ref"]))
                res[f"{k}_bitwise_equal_ref"] = eq
                print(f"frame {res['frame']:5d} {k} (kernel) vs {k}_ref (reference): bitwise equal {eq}")
        out["frames"].append(res)
    st = {k: dict(v, cols=sorted(v["cols"])) for k, v in STATS.items()}
    out["quantizer_check"] = st
    out["dump_records"] = DUMP["records"]
    for k, v in st.items():
        print(
            f"{k}: {v['calls']} calls, {v['rows']} rows, {v['elems']} values: {v['q_mismatch']} q / "
            f"{v['s_mismatch']} stored-scale mismatches kernel vs contract reference"
            + (f", {v['packfloat_mismatch']} vs pack_float" if "packfloat_mismatch" in v else "")
            + f"; zero rows {v['zero_rows']}, |q| = 32767 {v['sat']}; row lengths {v['cols']}",
            flush=True,
        )
    if args.json:
        args.json.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
