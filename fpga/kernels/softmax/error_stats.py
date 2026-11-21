"""Error of the softmax kernel's arithmetic (smx_emu.py, bit-exact to softmax_core.h) against FP64 torch.softmax on
the real -12392 score rows (build/real_scores.pt from policy_check.py --dump-scores: 2 heads of every site, frame 0).

    python error_stats.py      -> table + build/error_stats.json
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

import smx_emu as E

HERE = Path(__file__).resolve().parent


def stats(p_hat: torch.Tensor, ref: torch.Tensor) -> dict:
    keep = ref.sum(-1) > 0.5  # rows with a visible key (fully masked rows are exactly zero on both sides)
    p_hat, ref = p_hat[keep], ref[keep]
    err = (p_hat - ref).abs()
    big = ref > 1e-3
    return dict(
        max_abs=float(err.max()),
        mean_abs=float(err.mean()),
        max_abs_over_rowmax=float((err / ref.amax(-1, keepdim=True)).max()),
        max_rel_p_gt_1e3=float((err[big] / ref[big]).max()),
        mean_rel_p_gt_1e3=float((err[big] / ref[big]).mean()),
        max_rowsum_err=float((p_hat.sum(-1) - 1).abs().max()),
    )


def main():
    real = torch.load(HERE / "build/real_scores.pt", weights_only=False)
    c = E.c_from_mult(0.125)
    out = {}
    for k, v in real.items():
        x, valid = v["scores"], v.get("valid")
        vv = torch.ones_like(x, dtype=torch.bool) if valid is None else valid
        xs = torch.where(vv, x.double() * (c / E.LOG2E), torch.full_like(x.double(), -1e300))
        ref = torch.softmax(xs, -1)
        ref = torch.where(vv.any(-1, keepdim=True), ref, torch.zeros_like(ref))
        q, s = E.kernel(x, c, valid, empty="zero")
        u16 = stats(q.double() * s.double(), ref)
        f32 = stats(E.kernel(x, c, valid, fp32_out=True, empty="zero").double(), ref)
        out[k] = dict(rows=x.shape[0], L=x.shape[1], u16=u16, f32=f32)
        for m, st in (("u16", u16), ("f32", f32)):
            print(f"{k:13s} {m}: rows {x.shape[0]:4d} L {x.shape[1]:3d}  max|p-p64| {st['max_abs']:.2e}  mean {st['mean_abs']:.1e}"
                  f"  /rowmax {st['max_abs_over_rowmax']:.2e}  rel(p>1e-3) max {st['max_rel_p_gt_1e3']:.2e} mean "
                  f"{st['mean_rel_p_gt_1e3']:.1e}  |rowsum-1| {st['max_rowsum_err']:.1e}")
    (HERE / "build/error_stats.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
