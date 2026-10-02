"""Behavior cloning for TinyPolicy (build spec Phase 4). Practices and citations: plans/references.md.

- L1 loss on the 8-step action chunk, masked past the episode end [Zhao2023-ACT]
- Random-shift augmentation: replicate-pad 4 px, random crop back to 96 [Kostrikov2021-DrQ,
  Yarats2022-DrQv2]; brightness/contrast jitter (never hue: red/blue is the task label)
- AdamW [Loshchilov2019-AdamW], linear warmup + cosine decay, EMA of weights for evaluation
- Checkpoints are chosen by closed-loop success (armlab.eval.run), not validation loss
  [Mandlekar2021-robomimic]

    uv run python -m armlab.policy.train --data datasets/v1 --out runs/bc-v1
"""

import argparse
import copy
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from armlab.policy.data import DemoSet
from armlab.policy.tiny import TinyPolicy
from armlab.util.runmeta import write_runmeta
from armlab.util.seed import seed_everything


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def augment(img_u8: torch.Tensor, pad: int = 4, bright: float = 0.08, contrast: float = 0.15) -> torch.Tensor:
    """uint8 [B,H,W,3] -> uint8 [B,H,W,3]. Stays on the integer pixel grid so QAT sees real inputs."""
    b, h, w, _ = img_u8.shape
    x = img_u8.permute(0, 3, 1, 2).float()
    x = F.pad(x, (pad,) * 4, mode="replicate")
    ox = torch.randint(0, 2 * pad + 1, (b,), device=x.device)
    oy = torch.randint(0, 2 * pad + 1, (b,), device=x.device)
    rows = (oy[:, None] + torch.arange(h, device=x.device)[None])[:, None, :, None].expand(
        b, 3, h, w + 2 * pad
    )
    x = x.gather(2, rows)
    cols = (ox[:, None] + torch.arange(w, device=x.device)[None])[:, None, None, :].expand(b, 3, h, w)
    x = x.gather(3, cols)
    mean = x.mean((1, 2, 3), keepdim=True)
    c = 1 + (torch.rand(b, 1, 1, 1, device=x.device) * 2 - 1) * contrast
    br = (torch.rand(b, 1, 1, 1, device=x.device) * 2 - 1) * bright * 255
    x = (x - mean) * c + mean + br
    return x.round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1)


def chunk_l1(pred: torch.Tensor, target: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
    valid = (~pad).float()[..., None]
    return ((pred - target).abs() * valid).sum() / (valid.sum() * pred.shape[-1]).clamp(min=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, nargs="+", help="one or more shard directories")
    ap.add_argument("--data-device", default=None, help="where the dataset lives (default: the GPU)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=40_000)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--val-episodes", type=int, default=64)
    ap.add_argument("--ckpt-every", type=int, default=5000)
    ap.add_argument("--no-aug", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", default=None, help="checkpoint to start from")
    args = ap.parse_args()

    seed_everything(args.seed)
    dev = pick_device()
    torch.backends.cudnn.allow_tf32 = False  # keep float math honest for the later int8 comparison
    torch.backends.cuda.matmul.allow_tf32 = False
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ddev = args.data_device or dev
    train = DemoSet(args.data, ddev, episodes=slice(args.val_episodes, None))
    val = DemoSet(args.data, ddev, episodes=slice(0, args.val_episodes))
    print(f"device {dev}: {train.n} train frames, {val.n} val frames", flush=True)

    model = TinyPolicy().to(dev)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location=dev)["model"])
    ema = copy.deepcopy(model).eval()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: min(1.0, (s + 1) / args.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps))),
    )
    g = torch.Generator(device=dev).manual_seed(args.seed)
    log, t0 = [], time.perf_counter()
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, train.n, (args.bs,), device=dev, generator=g)
        img, aux, chunk, pad = (t.to(dev) for t in train.batch(idx.to(ddev)))
        if not args.no_aug:
            img = augment(img)
        loss = chunk_l1(model(*model.prep(img, aux)), chunk, pad)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters(), strict=True):
                pe.lerp_(pm, 1 - args.ema)

        if step % 500 == 0 or step == args.steps:
            with torch.no_grad():
                vl = []
                for i in range(0, val.n, 1024):
                    vb = val.batch(torch.arange(i, min(i + 1024, val.n), device=ddev))
                    vi, va, vc, vp = (t.to(dev) for t in vb)
                    vl.append(chunk_l1(ema(*ema.prep(vi, va)), vc, vp).item() * len(vi))
                vloss = sum(vl) / val.n
            rec = {
                "step": step,
                "loss": round(loss.item(), 5),
                "val_l1": round(vloss, 5),
                "lr": sched.get_last_lr()[0],
                "sec": round(time.perf_counter() - t0, 1),
            }
            log.append(rec)
            print(json.dumps(rec), flush=True)
        if step % args.ckpt_every == 0 or step == args.steps:
            torch.save(
                {"model": ema.state_dict(), "step": step, "args": vars(args)}, out / f"ckpt_{step:06d}.pt"
            )

    write_runmeta(out / "train_meta.json", vars(args), args.seed, {"log": log, "train_frames": train.n})


if __name__ == "__main__":
    main()
