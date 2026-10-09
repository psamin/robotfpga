"""Closed-loop evaluation (build spec Phase 6): success with Wilson 95% CI, failure reasons, latency.

    uv run python -m armlab.eval.run --backend torch --ckpt runs/bc/ckpt_040000.pt --stage B \
        --episodes 50 --latency-ms 0 --workers 8 --out runs/eval/bc_B.json

Held-out seeds start at 1,000,000 (training data uses seeds below 100,000).
"""

import argparse
import collections
import multiprocessing as mp
import os
import time

import numpy as np

from armlab.control.chunking import ChunkExecutor
from armlab.eval.stats import summarize
from armlab.sim.env import TASKS, BlockSortEnv
from armlab.sim.perturb import Perturber

EVAL_SEED0 = 1_000_000


def make_backend(name: str, ckpt: str | None):
    if name == "torch":
        from armlab.backends.torch_backend import TorchBackend

        return TorchBackend(ckpt)
    if name in ("intref", "mock-fpga"):
        from armlab.backends.intref_backend import IntrefBackend

        b = IntrefBackend(ckpt)
        b.name = name
        return b
    raise ValueError(f"unknown backend {name}")


def run_one(env, ex, seed: int, instr: int, stage: str, max_steps: int) -> dict:
    obs = env.reset(seed=seed, instr=instr)
    rng = np.random.default_rng(seed + 7)
    pert = Perturber(str(rng.choice(["cube_move", "instr_swap"])) if stage == "C" else "none", rng)
    ex.reset()
    for t in range(max_steps):
        if pert.maybe_apply(env, None, t):
            obs["instr"] = env.instr
        obs, done, info = env.step(ex.step(obs, t))
        if info["success"]:
            break
    return {
        "success": env.success(),
        "reason": env.failure_reason(),
        "steps": env.t,
        "event": pert.event,
        "infer_ms": ex.infer_ms,
    }


def _worker(job):
    backend, ckpt, stage, latency_ms, items, max_steps = job
    env = BlockSortEnv(max_steps=max_steps)
    ex = ChunkExecutor(make_backend(backend, ckpt), latency_ms=latency_ms)
    return [(instr, run_one(env, ex, seed, instr, stage, max_steps)) for seed, instr in items]


def evaluate(
    backend: str,
    ckpt: str | None,
    stage: str,
    episodes: int,
    latency_ms: float = 0.0,
    workers: int = 4,
    max_steps: int = 600,
) -> dict:
    items = [(EVAL_SEED0 + 10_000 * instr + i, instr) for instr in range(4) for i in range(episodes)]
    if stage == "A":
        items = [(EVAL_SEED0 + i, 0) for i in range(episodes)]
    jobs = [(backend, ckpt, stage, latency_ms, items[w::workers], max_steps) for w in range(workers)]
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = "1"
    t0 = time.perf_counter()
    with mp.get_context("spawn").Pool(workers) as pool:
        res = [r for rs in pool.map(_worker, jobs) for r in rs]
    out = {"backend": backend, "ckpt": ckpt, "stage": stage, "latency_ms": latency_ms, "per_instr": {}}
    for instr in sorted({i for i, _ in res}):
        rs = [r for i, r in res if i == instr]
        s = summarize([r["success"] for r in rs])
        s["failures"] = dict(collections.Counter(r["reason"] for r in rs))
        out["per_instr"][TASKS[instr]] = s
    out["all"] = summarize([r["success"] for _, r in res])
    out["all"]["failures"] = dict(collections.Counter(r["reason"] for _, r in res))
    ms = np.array([m for _, r in res for m in r["infer_ms"]])
    out["infer_ms"] = {"p50": float(np.percentile(ms, 50)), "p95": float(np.percentile(ms, 95))}
    out["seconds"] = round(time.perf_counter() - t0, 1)
    return out


def main() -> None:
    from armlab.util.runmeta import write_runmeta

    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="torch")
    ap.add_argument("--ckpt", nargs="+", default=[None])
    ap.add_argument("--stage", default="B", choices=["A", "B", "C"])
    ap.add_argument("--episodes", type=int, default=50, help="per instruction")
    ap.add_argument("--latency-ms", type=float, nargs="+", default=[0.0])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    results = []
    for ckpt in args.ckpt:
        for lat in args.latency_ms:
            r = evaluate(args.backend, ckpt, args.stage, args.episodes, lat, args.workers)
            a = r["all"]
            print(
                f"{args.backend} {ckpt} stage {args.stage} latency {lat} ms: {a['k']}/{a['n']} "
                f"= {a['rate']:.3f} CI95 [{a['ci95'][0]:.3f}, {a['ci95'][1]:.3f}] {a['failures']} "
                f"infer p50 {r['infer_ms']['p50']:.2f} ms",
                flush=True,
            )
            results.append(r)
    write_runmeta(args.out, vars(args), seed=EVAL_SEED0, results={"evals": results})


if __name__ == "__main__":
    main()
