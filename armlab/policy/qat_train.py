"""QAT fine-tuning from a float checkpoint, then export to the FPGA handoff format.

Loss = masked L1 to the expert chunk + distill * L1 to the frozen float teacher, which keeps the
int8 model close to the float one [Park2024-QAIL]. Exponents are calibrated once (MSE) and frozen;
weights fine-tune through straight-through rounding [Bengio2013-STE, Jacob2018-IntOnly].

    uv run python -m armlab.policy.qat_train --data datasets/v1 --float-ckpt runs/bc/ckpt_040000.pt \
        --out runs/qat
Writes ckpt_qat.pt and export/ (weights.bin + manifest.json, loadable by ref/intref.py).
"""

import argparse
import copy
import json
import math
import time
from pathlib import Path

import torch

from armlab.policy.data import DemoSet
from armlab.policy.qat import convert, load_intref, qat_state
from armlab.policy.tiny import TinyPolicy
from armlab.policy.train import augment, chunk_l1, pick_device
from armlab.util.runmeta import write_runmeta
from armlab.util.seed import seed_everything


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, nargs="+")
    ap.add_argument("--data-device", default=None)
    ap.add_argument("--float-ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=10_000)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--distill", type=float, default=0.5)
    ap.add_argument("--calib", type=int, default=4096)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--val-episodes", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    seed_everything(args.seed)
    dev = pick_device()
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ddev = args.data_device or dev
    train = DemoSet(args.data, ddev, episodes=slice(args.val_episodes, None))
    val = DemoSet(args.data, ddev, episodes=slice(0, args.val_episodes))

    teacher = TinyPolicy().to(dev).eval()
    teacher.load_state_dict(torch.load(args.float_ckpt, map_location=dev)["model"])
    for p in teacher.parameters():
        p.requires_grad_(False)
    g = torch.Generator(device=dev).manual_seed(args.seed)
    cal = torch.randint(0, train.n, (args.calib,), device=dev, generator=g).to(ddev)
    q = convert(teacher, train.image[cal].to(dev), train.aux[cal].to(dev))
    print(f"exponents a={q.a} w={q.w} shifts={q.shifts()}", flush=True)

    def val_l1(fn) -> float:
        tot = 0.0
        with torch.no_grad():
            for i in range(0, val.n, 1024):
                vi, va, vc, vp = (
                    t.to(dev) for t in val.batch(torch.arange(i, min(i + 1024, val.n), device=ddev))
                )
                tot += chunk_l1(fn(vi, va), vc, vp).item() * len(vi)
        return tot / val.n

    f_val = val_l1(lambda i, a: teacher(*teacher.prep(i, a)).clamp(-1, 1))
    log = [{"step": 0, "val_l1_float": round(f_val, 5), "val_l1_int8": round(val_l1(q), 5)}]
    print(json.dumps(log[0]), flush=True)

    ema = copy.deepcopy(q)
    opt = torch.optim.AdamW(q.parameters(), lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps)))
    )
    t0 = time.perf_counter()
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, train.n, (args.bs,), device=dev, generator=g)
        img, aux, chunk, pad = (t.to(dev) for t in train.batch(idx.to(ddev)))
        img = augment(img)
        pred = q(img, aux)
        with torch.no_grad():
            tgt = teacher(*teacher.prep(img, aux)).clamp(-1, 1)
        loss = chunk_l1(pred, chunk, pad) + args.distill * (pred - tgt).abs().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(q.parameters(), 1.0)
        opt.step()
        sched.step()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), q.parameters(), strict=True):
                pe.lerp_(pm, 1 - args.ema)
        if step % 1000 == 0 or step == args.steps:
            rec = {
                "step": step,
                "loss": round(loss.item(), 5),
                "val_l1_int8": round(val_l1(ema), 5),
                "sec": round(time.perf_counter() - t0, 1),
            }
            log.append(rec)
            print(json.dumps(rec), flush=True)

    torch.save(qat_state(ema), out / "ckpt_qat.pt")
    intref = load_intref()
    note = f"QAT from {args.float_ckpt}, {args.steps} steps, seed {args.seed}"
    manifest = ema.cpu().double().to_intref(intref).save(out / "export", note=note)
    write_runmeta(
        out / "qat_meta.json",
        vars(args),
        args.seed,
        {"log": log, "a_exp": ema.a, "w_exp": ema.w, "shifts": ema.shifts(), "sha256": manifest["sha256"]},
    )
    print(f"exported {out / 'export'} sha256 {manifest['sha256'][:16]}", flush=True)


if __name__ == "__main__":
    main()
