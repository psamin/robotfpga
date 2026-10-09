"""Random-action rollout rendered to an MP4 (build spec Phase 1 acceptance).

uv run python scripts/rollout_video.py --out runs/random.mp4 --seed 0 --steps 150
"""

import argparse
import os
import time

import imageio.v2 as imageio
import numpy as np

from armlab.sim.env import BlockSortEnv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/random_rollout.mp4")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=150)
    args = ap.parse_args()

    env = BlockSortEnv()
    obs = env.reset(seed=args.seed)
    rng = np.random.default_rng(args.seed)
    target = obs["state"].copy()
    frames, t0 = [obs["image_full"]], time.perf_counter()
    for _ in range(args.steps):
        target = np.clip(target + rng.normal(0, 0.03, 6), -1, 1)  # smooth random walk
        obs, done, _ = env.step(target)
        frames.append(obs["image_full"])
        if done:
            break
    dt = time.perf_counter() - t0
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    imageio.mimsave(args.out, frames, fps=30)
    ms = 1e3 * dt / len(frames)
    print(f"wrote {args.out}: {len(frames)} frames, {ms:.1f} ms per control step (sim + render)")


if __name__ == "__main__":
    main()
