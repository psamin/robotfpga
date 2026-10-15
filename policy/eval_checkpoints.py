"""Offline comparison of SmolVLA checkpoints: action error under camera loss, lighting shifts, unseen tasks.

For each checkpoint, predicts the 50-action chunk on fixed frames with fixed noise and reports the mean
absolute error against the recorded actions in normalized units (per-dataset statistics, so datasets are
comparable). Conditions on our demos:
  both        front + wrist, as recorded
  wrist_only  front camera removed (SmolVLA skips missing cameras)
  gamma_0.5 / gamma_2.0      exposure: brighter / darker (img ** gamma)
  warm_3000K / cool_10000K   color temperature, approximate black-body white balance vs 6500 K
Community: held-out episodes (index >= --community-episodes, never used in training), wrist only.

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
from train_wrist_mix import CAM1, CAM2, OURS, RENAME, View, pick_community, stats_remap  # noqa: E402

# sRGB of a black body relative to 6500 K white, rounded (standard Planckian-locus tables)
WHITE = {"warm_3000K": (1.00, 0.69, 0.43), "cool_10000K": (0.79, 0.85, 1.00)}


def lighting(name: str):
    if name.startswith("gamma_"):
        g = float(name.split("_")[1])
        return lambda img: img.clamp(0, 1) ** g
    if name in WHITE:
        w = torch.tensor(WHITE[name]).view(3, 1, 1)
        return lambda img: (img * w.to(img.device)).clamp(0, 1)
    return lambda img: img


def frames(view: View, n: int, seed: int) -> list[int]:
    g = torch.Generator().manual_seed(seed)
    return torch.randperm(len(view), generator=g)[:n].tolist()


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
    ap.add_argument("--n-community", type=int, default=10)
    ap.add_argument("--community-episodes", type=int, default=100, help="training used episodes below this")
    ap.add_argument("--eval-datasets", type=int, default=4)
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

    sets = {"ours": (View(dataset(OURS, ours_meta), dict(RENAME), shapes), None)}
    for r in pick_community(args.usable, args.n_community)[: args.eval_datasets]:
        meta = LeRobotDatasetMetadata(r["id"])
        held = list(range(args.community_episodes, meta.total_episodes))
        if not held:
            continue
        ds = dataset(r["id"], meta, held[:20])
        sets[r["id"]] = (View(ds, {r["wrist"][0]: CAM2}, shapes, stats_remap(meta.stats, ours_meta.stats)), r)
        print(f"held-out {r['id']}: episodes {held[0]}..{held[:20][-1]}, {len(ds)} frames", flush=True)

    conds = ["both", "wrist_only", "gamma_0.5", "gamma_2.0", "warm_3000K", "cool_10000K"]
    results = {}
    for spec in args.ckpt:
        name, path = spec.split("=", 1)
        policy = SmolVLAPolicy.from_pretrained(path).to("cuda").eval()
        pre, _ = make_pre_post_processors(policy.config, pretrained_path=path)
        res = {}
        for sname, (view, _) in sets.items():
            n = args.frames if sname == "ours" else args.community_frames
            idx = frames(view, n, args.seed)
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
