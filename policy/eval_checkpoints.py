"""Offline comparison of SmolVLA checkpoints: action error under camera loss, lighting shifts, unseen tasks.

For each checkpoint, predicts the 50-action chunk on fixed frames with fixed noise and reports the mean
absolute error against the recorded actions in normalized units (per-dataset statistics, so datasets are
comparable). Conditions on our demos:
  both        front + wrist, as recorded
  wrist_only  front camera removed (SmolVLA skips missing cameras)
  gamma_0.5 / gamma_2.0      exposure: brighter / darker (img ** gamma)
  warm_3000K / cool_10000K   color temperature, approximate black-body white balance vs 6500 K
  shadow      one random shadow band per image (RandomShadow, opacity 0.3-0.6, fixed seed)
Community: held-out episodes (index >= --community-episodes, never used in training), wrist only.
--eval-repos picks the community datasets; a dataset whose whole task family a run held out tests new tasks.

Checkpoints: name=<dir>, or name=merge@<dir A>@<dir B>@<alpha> for the weight interpolation
(1 - alpha) * A + alpha * B (RETAIN, arXiv 2512.08333; WiSE-FT). A run's dataset tags (--tag-prompt) are read
from its runmeta.json and applied to the task text, so tagged checkpoints are evaluated as trained.

Our frames come from the demos every checkpoint was trained on, so "both" measures fit, not generalization;
the camera, lighting and community conditions are inputs no checkpoint was trained on. Closed-loop trials
are still the real test.

    python policy/eval_checkpoints.py --ckpt a=<dir> --ckpt b=<dir> --usable usable.json --out eval.json
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_wrist_mix import (  # noqa: E402
    CAM1,
    CAM2,
    OURS,
    RENAME,
    View,
    stats_remap,
    tag_for,
    usable_row,
)

EVAL_REPOS = [
    "DmitryStrog/so101_pnp_merged_500",
    "nbirukov/so101_pnp_merged2",
    "tttonyalpha/so101_pick_cube_300_task1",
    "tttonyalpha/so101_pick_cube_300_task3",  # "Remember the color ..." family (held out by --holdout)
    "avanturist/so101_rgb_cube_300_task2",  # same family
]

# sRGB of a black body relative to 6500 K white, rounded (standard Planckian-locus tables)
WHITE = {"warm_3000K": (1.00, 0.69, 0.43), "cool_10000K": (0.79, 0.85, 1.00)}


def lighting(name: str):
    if name.startswith("gamma_"):
        g = float(name.split("_")[1])
        return lambda img: img.clamp(0, 1) ** g
    if name in WHITE:
        w = torch.tensor(WHITE[name]).view(3, 1, 1)
        return lambda img: (img * w.to(img.device)).clamp(0, 1)
    if name == "shadow":
        from lerobot.transforms.transforms import RandomShadow

        sh = RandomShadow(opacity=(0.3, 0.6))
        return lambda img: sh(img)
    return lambda img: img


def runmeta(path: str) -> dict:
    """The training run's runmeta.json above a pretrained_model dir (final or checkpoints/<step>/), or {}."""
    p = Path(path)
    for d in (p.parent, p.parent.parent.parent):
        if (d / "runmeta.json").exists():
            return json.loads((d / "runmeta.json").read_text())
    return {}


def load(spec: str, SmolVLAPolicy):
    """(policy, processor path, tag mode) for name=<dir> or name=merge@A@B@alpha."""
    if spec.startswith("merge@"):
        _, a, b, alpha = spec.split("@")
        alpha = float(alpha)
        pa = SmolVLAPolicy.from_pretrained(a)
        sb = SmolVLAPolicy.from_pretrained(b).state_dict()
        sa = pa.state_dict()
        merged = {k: (1 - alpha) * v + alpha * sb[k] if v.is_floating_point() else v for k, v in sa.items()}
        pa.load_state_dict(merged)
        return pa, a, runmeta(b).get("args", {}).get("tag_prompt", False)
    return SmolVLAPolicy.from_pretrained(spec), spec, runmeta(spec).get("args", {}).get("tag_prompt", False)


def frames(lo: int, hi: int, n: int, seed: int) -> list[int]:
    """n fixed random frame indices in [lo, hi)."""
    g = torch.Generator().manual_seed(seed)
    return (lo + torch.randperm(hi - lo, generator=g)[:n]).tolist()


@torch.no_grad()
def score(policy, pre, view, idx, cond: str, batch: int, seed: int):
    """Mean |pred - target| over the chunk in normalized units, overall and per joint."""
    light = lighting(cond)
    errs = []
    for s in range(0, len(idx), batch):
        items = [view[i] for i in idx[s : s + batch]]
        b = torch.utils.data.default_collate(items)
        if cond == "wrist_only":
            b.pop(CAM1, None)
        torch.manual_seed(seed + s)  # same shadows for every checkpoint
        for k in (CAM1, CAM2):
            if k in b:
                b[k] = torch.stack([light(im) for im in b[k]])
        b = pre(b)
        target = b["action"]
        pad = b.get("action_is_pad")
        g = torch.Generator(device="cpu").manual_seed(seed + s)
        noise = torch.randn(
            (target.shape[0], policy.config.chunk_size, policy.config.max_action_dim), generator=g
        )
        policy.reset()
        pred = policy.predict_action_chunk(b, noise=noise.to(target.device))[..., : target.shape[-1]]
        e = (pred - target).abs()
        if pad is not None:
            e = e[~pad]
        errs.append(e.reshape(-1, target.shape[-1]).cpu())
    e = torch.cat(errs)
    return {"mae_norm": float(e.mean()), "mae_norm_per_joint": [round(float(x), 4) for x in e.mean(0)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True, help="name=path")
    ap.add_argument("--usable", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=64)
    ap.add_argument("--community-frames", type=int, default=32)
    ap.add_argument("--eval-repos", nargs="*", default=EVAL_REPOS)
    ap.add_argument("--community-episodes", type=int, default=100, help="training used episodes below this")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    ours_meta = LeRobotDatasetMetadata(OURS)
    shapes = {RENAME[k]: tuple(ours_meta.features[k]["shape"][:2]) for k in RENAME}

    def dataset(repo, meta, episodes=None):
        dt = {"action": [i / meta.fps for i in range(50)]}
        return LeRobotDataset(repo, episodes=episodes, delta_timestamps=dt, video_backend="pyav")

    ours = View(dataset(OURS, ours_meta), dict(RENAME), shapes)
    sets = {"ours": (ours, 0, len(ours))}  # (view, first frame, end frame)
    for repo in args.eval_repos:
        r = usable_row(args.usable, repo)
        meta = LeRobotDatasetMetadata(r["id"])
        if meta.total_episodes <= args.community_episodes:
            continue
        # Loading an episode subset that starts mid-dataset found no parquet files (job 10241), so load the
        # whole dataset and sample frames of the held-out episodes by their global index range.
        lengths = list(meta.episodes["length"])
        lo = sum(lengths[: args.community_episodes])
        hi = lo + sum(lengths[args.community_episodes : args.community_episodes + 20])
        view = View(
            dataset(r["id"], meta), {r["wrist"][0]: CAM2}, shapes, stats_remap(meta.stats, ours_meta.stats)
        )
        sets[r["id"]] = (view, lo, hi)
        print(f"held-out {r['id']}: frames {lo}..{hi} (episodes from {args.community_episodes})", flush=True)

    conds = ["both", "wrist_only", "gamma_0.5", "gamma_2.0", "warm_3000K", "cool_10000K", "shadow"]
    results = {}
    for spec in args.ckpt:
        name, path = spec.split("=", 1)
        missing = [
            q for q in (path.split("@")[1:3] if path.startswith("merge@") else [path]) if not Path(q).exists()
        ]
        if missing:  # a run that failed: keep evaluating the rest
            print(f"skip {name}: missing {missing}", flush=True)
            continue
        policy, proc_path, tagged = load(path, SmolVLAPolicy)
        policy = policy.to("cuda").eval()
        pre, _ = make_pre_post_processors(policy.config, pretrained_path=proc_path)
        res = {}
        for sname, (view, lo, hi) in sets.items():
            view.tag = tag_for(OURS if sname == "ours" else sname) if tagged else ""
            n = args.frames if sname == "ours" else args.community_frames
            idx = frames(lo, hi, n, args.seed)
            for cond in conds if sname == "ours" else ["wrist_only"]:
                res[f"{sname}|{cond}"] = score(policy, pre, view, idx, cond, args.batch, args.seed)
                print(f"{name} {sname} {cond}: {res[f'{sname}|{cond}']['mae_norm']:.4f}", flush=True)
        results[name] = {"path": path, **res}
        del policy
        torch.cuda.empty_cache()
    Path(args.out).write_text(json.dumps(results, indent=1) + "\n")
    keys = list(next(iter(results.values())).keys())[1:]
    print("\n| condition | " + " | ".join(results) + " |\n|---|" + "---|" * len(results))
    for k in keys:
        print(f"| {k} | " + " | ".join(f"{results[n][k]['mae_norm']:.4f}" for n in results) + " |")


if __name__ == "__main__":
    main()
