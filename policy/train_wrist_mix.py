"""Fine-tune the demo SmolVLA with front-camera dropout and community SO-101 wrist-camera data.

Starts from a fine-tuned checkpoint (default: -9604) and writes a NEW checkpoint, so both stay usable.
Vision encoder frozen and only the action expert trained, as the checkpoint's config already sets.

Each step draws one batch from one source:
  - ours (probability --p-ours): our demos with front (camera1) and wrist (camera2); camera1 is dropped
    from the whole batch with probability --p-drop-front. SmolVLA skips missing cameras
    (empty_cameras=0), so a dropped camera is simply not seen.
  - community: wrist only (mapped to camera2), from the datasets selected by the filter job
    (usable_datasets.json), at most --community-episodes episodes each.
LeRobot's multi-dataset training is disabled in our fork (NotImplementedError), hence this loop.
Optimizer, schedule and augmentation match lerobot-train with the checkpoint's presets.

    python policy/train_wrist_mix.py --base <ckpt> --usable usable_datasets.json --out <dir> --steps 2000
"""

import argparse
import json
import math
import random
import re
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader

OURS = "flyingturtleboop/so101_medicine_bottle_pickplace"
RENAME = {
    "observation.images.front": "observation.images.camera1",
    "observation.images.wrist": "observation.images.camera2",
}
CAM1, CAM2 = "observation.images.camera1", "observation.images.camera2"
SKIP = re.compile(r"sim|aug|gen|test|pi0", re.I)  # synthetic, augmented, or scratch datasets


def tag_for(repo: str) -> str:
    """Dataset tag prefixed to the task text with --tag-prompt; ours gets a fixed robot tag."""
    return "[so101-psamin] " if repo == OURS else f"[{repo}] "


def usable_row(usable_json: str, repo: str) -> dict:
    return next(r for r in json.load(open(usable_json))["all"] if r["id"] == repo)


def pick_community(usable_json: str, n: int) -> list[dict]:
    rows = json.load(open(usable_json))["usable"]
    rows = [
        r for r in rows if r["license"] in ("apache-2.0", "mit", "cc-by-4.0") and not SKIP.search(r["id"])
    ]
    seen, out = set(), []
    for r in sorted(rows, key=lambda r: -r["episodes"]):
        key = (r["episodes"], r["frames"])  # merged copies of the same data share both counts
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out[:n]


def light_aug_config(ImageTransformsConfig, ImageTransformConfig):
    """LeRobot's default augmentation plus lighting transforms, for robustness to room lighting.

    Wider brightness/contrast jitter is the usual baseline (arXiv 2508.01442 compares against it);
    gamma models exposure; Planckian jitter re-illuminates with black-body light sources of 3000-15000 K
    (arXiv 2202.07993); shadows and Gaussian brightness patches model uneven indoor light.
    """
    cfg = ImageTransformsConfig(enable=True, max_num_transforms=4)
    tc = ImageTransformConfig
    cfg.tfs["brightness"] = tc(type="ColorJitter", kwargs={"brightness": (0.5, 1.5)})
    cfg.tfs["contrast"] = tc(type="ColorJitter", kwargs={"contrast": (0.6, 1.4)})
    cfg.tfs["gamma"] = tc(type="GammaCorrection", kwargs={"gamma": (0.6, 1.7)})
    cfg.tfs["planckian"] = tc(type="PlanckianJitter", kwargs={"temperature": (3000, 15000)})
    cfg.tfs["shadow"] = tc(weight=0.5, type="RandomShadow", kwargs={"opacity": (0.3, 0.6)})
    cfg.tfs["patches"] = tc(weight=0.5, type="GaussianPatchBrightness", kwargs={"factor_range": (0.5, 1.5)})
    return cfg


def resize_pad(img: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """CHW image to h x w, keeping aspect ratio, zero padding (so batches from mixed cameras stack)."""
    _, ih, iw = img.shape
    s = min(h / ih, w / iw)
    nh, nw = max(1, round(ih * s)), max(1, round(iw * s))
    img = F.interpolate(img[None], size=(nh, nw), mode="bilinear", align_corners=False)[0]
    return F.pad(img, (0, w - nw, 0, h - nh))


class View(torch.utils.data.Dataset):
    """Maps one LeRobotDataset's items to the policy's keys and fixed image shapes.

    remap: {key: (a, b)} applies x -> a * x + b to state/action, used to move a community dataset's
    joints into our normalization statistics (per-dataset normalization)."""

    def __init__(self, ds, cams: dict[str, str], shapes: dict[str, tuple[int, int]], remap=None, tag=""):
        self.ds, self.cams, self.shapes = ds, cams, shapes  # cams: source key -> policy key
        self.remap, self.tag = remap or {}, tag

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        for _ in range(10):  # some community videos have bad timestamps (FrameTimestampError): skip the item
            try:
                return self._get(i)
            except Exception as e:  # noqa: BLE001
                print(f"skip item {i}: {type(e).__name__}", flush=True)
                i = random.randrange(len(self.ds))
        return self._get(i)

    def _get(self, i):
        it = self.ds[i]
        out = {k: it[k] for k in ("observation.state", "action", "action_is_pad", "task") if k in it}
        out["task"] = self.tag + out["task"]
        for k, (a, b) in self.remap.items():
            out[k] = out[k] * a + b
        for src, dst in self.cams.items():
            out[dst] = resize_pad(it[src], *self.shapes[dst])
        return out


def stats_remap(src_stats, dst_stats) -> dict:
    """(x - mu_src) / sd_src * sd_dst + mu_dst for state and action, as x -> a * x + b."""
    out = {}
    for k in ("observation.state", "action"):
        mu_s, sd_s = (torch.as_tensor(src_stats[k][q], dtype=torch.float32) for q in ("mean", "std"))
        mu_d, sd_d = (torch.as_tensor(dst_stats[k][q], dtype=torch.float32) for q in ("mean", "std"))
        a = sd_d / sd_s.clamp_min(1e-6)
        out[k] = (a, mu_d - mu_s * a)
    return out


def loader(ds, batch, workers):
    return iter(
        DataLoader(
            ds,
            batch_size=batch,
            shuffle=True,
            num_workers=workers,
            drop_last=True,
            pin_memory=True,
            persistent_workers=True,
        )
    )


def forever(make):
    it = make()
    while True:
        try:
            yield next(it)
        except StopIteration:
            it = make()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--usable", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--p-ours", type=float, default=0.5)
    ap.add_argument("--p-drop-front", type=float, default=0.5)
    ap.add_argument("--n-community", type=int, default=10)
    ap.add_argument("--community-episodes", type=int, default=100)
    ap.add_argument("--exclude", type=int, nargs="*", default=[11, 14])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--aug", choices=["default", "light", "shadow20"], default="default", help="image augmentation set"
    )
    ap.add_argument("--shadow-p", type=float, default=0.2, help="shadow20: fraction of frames given a shadow")
    ap.add_argument("--save-every", type=int, default=500, help="intermediate checkpoints (~0.9 GB each)")
    ap.add_argument("--lr", type=float, default=None, help="peak learning rate (default: checkpoint preset)")
    ap.add_argument("--warmup", type=int, default=None, help="warmup steps (default: min(preset, steps/2))")
    ap.add_argument("--tag-prompt", action="store_true", help="prefix each task with its dataset tag")
    ap.add_argument("--dedup-tasks", action="store_true", help="skip community datasets repeating a task set")
    ap.add_argument("--holdout", default=None, help="regex: skip community datasets with a matching task")
    ap.add_argument(
        "--l2sp", type=float, default=0.0, help="weight of sum (w - w_base)^2 over trained params"
    )
    ap.add_argument(
        "--norm",
        choices=["ours", "per-dataset"],
        default="per-dataset",
        help="normalize community state/action with our stats, or with each dataset's own",
    )
    args = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.transforms import ImageTransformConfig, ImageTransforms, ImageTransformsConfig

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    policy = SmolVLAPolicy.from_pretrained(args.base).to("cuda").train()
    cfg = policy.config
    pre, post = make_pre_post_processors(cfg, pretrained_path=args.base)
    aug_cfg = (
        light_aug_config(ImageTransformsConfig, ImageTransformConfig)
        if args.aug == "light"
        else ImageTransformsConfig(enable=True)
    )
    aug = ImageTransforms(aug_cfg)
    if args.aug == "shadow20":
        # default set, plus a random shadow band on --shadow-p of frames (arXiv 2101.05361)
        from lerobot.transforms.transforms import RandomShadow
        from torchvision.transforms import v2

        aug = v2.Compose([aug, v2.RandomApply([RandomShadow(opacity=(0.3, 0.6))], p=args.shadow_p)])
    print(f"augmentation {args.aug}: {aug}", flush=True)

    def dataset(repo, episodes=None):
        meta = LeRobotDatasetMetadata(repo)
        dt = {"action": [i / meta.fps for i in cfg.action_delta_indices]}
        return LeRobotDataset(
            repo, episodes=episodes, delta_timestamps=dt, image_transforms=aug, video_backend="pyav"
        ), meta

    ours_meta = LeRobotDatasetMetadata(OURS)
    ours_eps = [e for e in range(ours_meta.total_episodes) if e not in args.exclude]
    ours, _ = dataset(OURS, ours_eps)
    shapes = {RENAME[k]: tuple(ours_meta.features[k]["shape"][:2]) for k in RENAME}  # (h, w) per camera
    ours_v = View(ours, dict(RENAME), shapes, tag=tag_for(OURS) if args.tag_prompt else "")

    picked, parts, seen_tasks = [], [], set()
    holdout = re.compile(args.holdout, re.I) if args.holdout else None
    for r in pick_community(args.usable, 10**6):
        if len(picked) >= args.n_community:
            break
        meta = LeRobotDatasetMetadata(r["id"])
        tasks = frozenset(map(str, meta.tasks.index))
        if holdout and any(holdout.search(t) for t in tasks):
            print(f"held out (task family) {r['id']}: {sorted(tasks)}", flush=True)
            continue
        if args.dedup_tasks and tasks in seen_tasks:
            print(f"skipped (same tasks as an earlier dataset) {r['id']}", flush=True)
            continue
        seen_tasks.add(tasks)
        picked.append(r)
        eps = list(range(min(args.community_episodes, meta.total_episodes)))
        ds, _ = dataset(r["id"], eps)
        remap = stats_remap(meta.stats, ours_meta.stats) if args.norm == "per-dataset" else None
        parts.append(
            View(ds, {r["wrist"][0]: CAM2}, shapes, remap, tag_for(r["id"]) if args.tag_prompt else "")
        )
        print(
            f"community {r['id']}: {len(eps)} episodes, {len(ds)} frames, wrist={r['wrist'][0]}", flush=True
        )
    comm = ConcatDataset(parts) if parts else []
    print(
        f"ours: {len(ours_eps)} episodes, {len(ours)} frames; "
        f"community: {len(parts)} datasets, {len(comm)} frames",
        flush=True,
    )
    sources = {"ours": forever(lambda: loader(ours_v, args.batch, args.workers))}
    if parts:
        sources["community"] = forever(lambda: loader(comm, args.batch, args.workers))
    else:
        args.p_ours = 1.0  # ours only (e.g. specializing a co-trained checkpoint)

    params = [p for p in policy.parameters() if p.requires_grad]
    anchor = [p.detach().clone() for p in params] if args.l2sp > 0 else None  # L2-SP (Li et al., 2018)
    opt = torch.optim.AdamW(
        params,
        lr=cfg.optimizer_lr,
        betas=tuple(cfg.optimizer_betas),
        eps=cfg.optimizer_eps,
        weight_decay=cfg.optimizer_weight_decay,
    )
    warm = args.warmup if args.warmup is not None else min(cfg.scheduler_warmup_steps, args.steps // 2)
    peak, floor = args.lr or cfg.optimizer_lr, cfg.scheduler_decay_lr

    def lr_at(s):
        if s < warm:
            return peak * (s + 1) / (warm + 1)
        p = (s - warm) / max(1, args.steps - warm)
        return floor + (peak - floor) * 0.5 * (1 + math.cos(math.pi * p))

    def save(ck: Path):
        policy.save_pretrained(ck)
        pre.save_pretrained(ck)
        post.save_pretrained(ck)
        json.dump(
            {"base": args.base, "args": vars(args), "community": [r["id"] for r in picked]},
            open(out / "runmeta.json", "w"),
            indent=1,
        )
        print(f"saved {ck}", flush=True)

    log = open(out / "metrics.jsonl", "w")
    t0 = time.time()
    for step in range(args.steps):
        src = "ours" if random.random() < args.p_ours else "community"
        batch = next(sources[src])
        dropped = src == "ours" and random.random() < args.p_drop_front
        if dropped:
            batch.pop(CAM1)
        batch = pre(batch)
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        loss, info = policy.forward(batch)
        l2 = sum(((p - a) ** 2).sum() for p, a in zip(params, anchor, strict=True)) if anchor else None
        if l2 is not None:
            loss = loss + args.l2sp * l2
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(params, cfg.optimizer_grad_clip_norm)
        opt.step()
        row = {
            "step": step + 1,
            "src": src,
            "front": not dropped and src == "ours",
            "loss": info["loss"],
            "l2sp_dist": float(l2) if l2 is not None else None,
            "grad_norm": float(gn),
            "lr": lr_at(step),
            "s": round(time.time() - t0, 1),
        }
        log.write(json.dumps(row) + "\n")
        if (step + 1) % 100 == 0:
            log.flush()
            print(json.dumps(row), flush=True)
        if (step + 1) % args.save_every == 0 and step + 1 < args.steps:
            save(out / "checkpoints" / f"{step + 1:06d}" / "pretrained_model")

    save(out / "pretrained_model")


if __name__ == "__main__":
    main()
