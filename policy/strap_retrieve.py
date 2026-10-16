"""STRAP-style sub-trajectory retrieval of community data similar to our demos (arXiv 2412.15182).

1. Embed wrist-camera frames (every --stride frames) of our demos and of the community pool with a frozen
   DINOv2 (CLS token, L2-normalized). The wrist view is the camera all SO-101 datasets share.
2. Split each of our demos into sub-trajectories at minima of joint speed (pauses between motions), as STRAP
   segments by changes in end-effector motion.
3. Match every segment against every pool episode with subsequence DTW on cosine distance (free start and
   end in the pool episode); keep the --top-k cheapest matches per segment.
Writes [{"repo", "episode", "start", "end", "cost", "query"}] with frame indices relative to the pool dataset
as training loads it (episodes 0..--community-episodes-1), for train_wrist_mix.py --retrieved.

    python policy/strap_retrieve.py --usable usable_datasets.json --out retrieved.json
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_wrist_mix import OURS, pick_community  # noqa: E402


def segments(speed: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    """Split at local minima of a smoothed speed curve; every piece at least min_len long."""
    k = np.ones(5) / 5
    s = np.convolve(speed, k, mode="same")
    cuts = [0]
    for i in range(1, len(s) - 1):
        if s[i] < s[i - 1] and s[i] <= s[i + 1] and i - cuts[-1] >= min_len and len(s) - i >= min_len:
            cuts.append(i)
    cuts.append(len(s))
    return list(zip(cuts[:-1], cuts[1:], strict=True))


def sdtw(q: np.ndarray, x: np.ndarray) -> tuple[float, int, int]:
    """Subsequence DTW of query q (m, d) inside x (n, d), free start and end in x.

    Slope-constrained steps from the previous query row only: x advances 0, 1 or 2 frames per query frame
    (pool motion between half and twice our speed), so each row is one vectorized update.
    Returns (mean cosine distance along the path, start, end) in x."""
    c = 1.0 - q @ x.T  # embeddings are unit norm
    m, n = c.shape
    d, start = c[0].copy(), np.arange(n)
    inf = np.full(2, np.inf)
    for i in range(1, m):
        cands = np.stack([d, np.concatenate([inf[:1], d[:-1]]), np.concatenate([inf, d[:-2]])])
        st = np.stack([start, np.concatenate([[0], start[:-1]]), np.concatenate([[0, 0], start[:-2]])])
        k = np.argmin(cands, axis=0)
        d = c[i] + cands[k, np.arange(n)]
        start = st[k, np.arange(n)]
    j = int(np.argmin(d))
    return float(d[j] / m), int(start[j]), j + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--usable", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-community", type=int, default=10)
    ap.add_argument("--community-episodes", type=int, default=100)
    ap.add_argument("--holdout", default=None, help="same regex as the training run")
    ap.add_argument("--stride", type=int, default=6, help="embed every n-th frame (30 fps -> 5 Hz)")
    ap.add_argument("--min-seg", type=int, default=8, help="minimum segment length, in strided frames")
    ap.add_argument("--top-k", type=int, default=40, help="matches kept per query segment")
    ap.add_argument("--exclude", type=int, nargs="*", default=[11, 14])
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from transformers import AutoImageProcessor, AutoModel

    proc = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
    enc = AutoModel.from_pretrained("facebook/dinov2-small").to("cuda").eval()

    @torch.no_grad()
    def embed(repo: str, cam: str, episodes: list[int]):
        """Per episode: (strided frame indices relative to the loaded dataset, unit embeddings, states)."""
        ds = LeRobotDataset(repo, episodes=episodes, video_backend="pyav")
        ep_index = np.asarray(ds.hf_dataset["episode_index"])
        idx = np.arange(0, len(ds), args.stride)

        class Strided(torch.utils.data.Dataset):
            def __len__(self):
                return len(idx)

            def __getitem__(self, i):
                try:
                    it = ds[int(idx[i])]
                    return int(idx[i]), it[cam], it["observation.state"]
                except Exception:  # noqa: BLE001  (bad video timestamps in some community data)
                    return None

        def keep(items):
            items = [it for it in items if it is not None]
            return [it[0] for it in items], [it[1] for it in items], [it[2] for it in items]

        loader = torch.utils.data.DataLoader(
            Strided(), batch_size=64, num_workers=args.workers, collate_fn=keep
        )
        frame_ids, feats, states = [], [], []
        for fid, img, st in loader:
            if not fid:
                continue
            px = proc(images=img, return_tensors="pt", do_rescale=False)["pixel_values"].to("cuda")
            f = enc(pixel_values=px).last_hidden_state[:, 0]
            feats.append(torch.nn.functional.normalize(f, dim=-1).cpu().numpy())
            states.append(torch.stack(st).numpy())
            frame_ids += fid
        frame_ids, feats, states = np.asarray(frame_ids), np.concatenate(feats), np.concatenate(states)
        out = {}
        for e in np.unique(ep_index[frame_ids]):
            sel = ep_index[frame_ids] == e
            if sel.sum() >= args.min_seg:
                out[int(e)] = (frame_ids[sel], feats[sel], states[sel])
        print(f"embedded {repo}: {len(idx)} frames, {len(out)} episodes", flush=True)
        return out

    ours_meta = LeRobotDatasetMetadata(OURS)
    ours = embed(
        OURS,
        "observation.images.wrist",
        [e for e in range(ours_meta.total_episodes) if e not in args.exclude],
    )
    queries = []
    for e, (_fi, f, st) in ours.items():
        speed = np.linalg.norm(
            np.diff(st[:, :5], axis=0, prepend=st[:1, :5]), axis=1
        )  # body joints, deg/step
        for a, b in segments(speed, args.min_seg):
            queries.append((e, a, b, f[a:b]))
    print(f"{len(queries)} query segments from {len(ours)} demos", flush=True)

    holdout = re.compile(args.holdout, re.I) if args.holdout else None
    pool, seen = {}, set()
    for r in pick_community(args.usable, 10**6):
        if len(pool) >= args.n_community:
            break
        meta = LeRobotDatasetMetadata(r["id"])
        tasks = frozenset(map(str, meta.tasks.index))
        if (holdout and any(holdout.search(t) for t in tasks)) or tasks in seen:
            continue
        seen.add(tasks)
        eps = list(range(min(args.community_episodes, meta.total_episodes)))
        pool[r["id"]] = embed(r["id"], r["wrist"][0], eps)

    hits = []
    for qi, (qe, a, b, qf) in enumerate(queries):
        cand = []
        for repo, eps in pool.items():
            for e, (fi, f, _) in eps.items():
                cost, s, t = sdtw(qf, f)
                cand.append(
                    {
                        "repo": repo,
                        "episode": e,
                        "start": int(fi[s]),
                        "end": int(fi[t - 1]) + 1,
                        "cost": cost,
                        "query": [qe, int(a), int(b)],
                    }
                )
        cand.sort(key=lambda h: h["cost"])
        hits += cand[: args.top_k]
        print(
            f"query {qi + 1}/{len(queries)} (demo {qe}, {a}-{b}): best cost {cand[0]['cost']:.3f} "
            f"{cand[0]['repo']} ep {cand[0]['episode']}",
            flush=True,
        )
    frames = sum(h["end"] - h["start"] for h in hits)
    by_repo = {}
    for h in hits:
        by_repo[h["repo"]] = by_repo.get(h["repo"], 0) + 1
    Path(args.out).write_text(json.dumps({"segments": hits, "args": vars(args)}, indent=1) + "\n")
    print(f"retrieved {len(hits)} segments, {frames} frames; by dataset {by_repo}", flush=True)


if __name__ == "__main__":
    main()
