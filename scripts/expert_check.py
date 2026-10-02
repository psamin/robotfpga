"""Expert quality on the cluster: clean success, success under DART noise, recovery from random states.

uv run python scripts/expert_check.py --markov 1 --workers 32
"""

import argparse
import collections
import json
import multiprocessing as mp
import os

import numpy as np


def _episode(job):
    kind, i, markov, sigma = job
    from armlab.sim.env import BlockSortEnv
    from armlab.sim.expert import Expert, Phase

    env = BlockSortEnv(render_size=96)
    rng = np.random.default_rng(i)
    obs = env.reset(seed=50_000 + i, instr=i % 4)
    if kind == "recovery":  # drive the arm randomly first, then hand over
        tgt = obs["state"].copy()
        for _ in range(rng.integers(15, 45)):
            tgt = np.clip(tgt + rng.normal(0, 0.04, 6), -1, 1)
            env.step(tgt)
    ex = Expert(env, markov=markov)
    noise = np.zeros(6)
    for _ in range(450):
        label = env.normalize(ex.act())
        noise = 0.8 * noise + 0.6 * rng.normal(0, sigma, 6)
        noise[5] = 0
        env.step(np.clip(label + noise, -1, 1))
        if env.success() and ex.phase == Phase.DONE:
            return kind, sigma, True, env.t
    return kind, sigma, False, env.t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--markov", type=int, default=0)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"
    m = bool(args.markov)
    jobs = [("clean", i, m, 0.0) for i in range(args.n)]
    jobs += [("noise", i, m, s) for s in (0.005, 0.01, 0.02) for i in range(args.n)]
    jobs += [("recovery", i, m, 0.0) for i in range(args.n)]
    with mp.get_context("spawn").Pool(args.workers) as pool:
        res = pool.map(_episode, jobs, chunksize=4)
    out = collections.defaultdict(list)
    steps = collections.defaultdict(list)
    for kind, sigma, ok, t in res:
        key = f"{kind}" + (f" sigma={sigma}" if kind == "noise" else "")
        out[key].append(ok)
        steps[key].append(t)
    summary = {k: f"{sum(v)}/{len(v)}, mean steps {np.mean(steps[k]):.0f}" for k, v in out.items()}
    print(f"markov={m}", json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
