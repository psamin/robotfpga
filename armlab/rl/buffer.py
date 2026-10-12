"""SGS reset buffer: arXiv:2610.12465v1 App. A.2 pre-collects 32,768 collision-checked states,
balanced across reset strategies, and every training episode starts from one of them.

Rows are grouped by family in FAMILIES order; the row index is the SGS configuration ID. Each
fixed-size chunk has its own SeedSequence, so the buffer depends on the seed, not on --workers.

    uv run python -m armlab.rl.buffer --out artifacts/resets/v1.npz --size 32768 --seed 0 --workers 8
"""

import argparse
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np

from armlab.rl.resets import FAMILIES, ResetState, sample
from armlab.sim.env import BlockSortEnv
from armlab.sim.ik import TopDownIK

CHUNK = 256
MAX_TRIES = 20  # per accepted state, before a chunk gives up
FIELDS = ("qpos", "qvel", "ctrl")
_worker: tuple[BlockSortEnv, TopDownIK] | None = None


def _chunk(job: tuple[int, int, int, int]) -> tuple[list[ResetState], int]:
    global _worker
    if _worker is None:
        env = BlockSortEnv(jitter=None)
        _worker = env, TopDownIK(env.m)
    seed, family, index, count = job
    rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(family, index)))
    out, tries = [], 0
    while len(out) < count:
        tries += 1
        if tries > MAX_TRIES * count:
            raise RuntimeError(f"{FAMILIES[family]}: acceptance below 1/{MAX_TRIES}")
        s = sample(*_worker, FAMILIES[family], rng)
        if s is not None:
            out.append(s)
    return out, tries


def build(size: int, seed: int, workers: int = 1, chunk: int = CHUNK) -> tuple[dict, dict]:
    """Returns (buffer arrays, per-family acceptance rates)."""
    if size < len(FAMILIES) or size % len(FAMILIES):
        raise ValueError(f"size must be a positive multiple of {len(FAMILIES)}")
    per = size // len(FAMILIES)
    jobs = [
        (seed, f, k, min(chunk, per - k * chunk))
        for f in range(len(FAMILIES))
        for k in range(-(-per // chunk))
    ]
    if workers > 1:
        with mp.get_context("spawn").Pool(workers) as pool:
            results = pool.map(_chunk, jobs)
    else:
        results = [_chunk(j) for j in jobs]
    states = [s for chunk_states, _ in results for s in chunk_states]
    tries = np.zeros(len(FAMILIES))
    for (_, f, _, _), (_, t) in zip(jobs, results, strict=True):
        tries[f] += t
    buf = {k: np.stack([getattr(s, k) for s in states]) for k in FIELDS}
    buf["instr"] = np.array([s.instr for s in states], np.int8)
    buf["family"] = np.repeat(np.arange(len(FAMILIES), dtype=np.int8), per)
    return buf, {name: per / tries[f] for f, name in enumerate(FAMILIES)}


def save(path: str | Path, buf: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, families=np.array(FAMILIES), **buf)


def load(path: str | Path) -> dict:
    with np.load(path) as f:
        buf = {k: f[k] for k in f.files}
    if tuple(buf.pop("families")) != FAMILIES:
        raise ValueError("buffer was built with different reset families")
    return buf


def state(buf: dict, i: int) -> ResetState:
    return ResetState(FAMILIES[buf["family"][i]], int(buf["instr"][i]), *(buf[k][i] for k in FIELDS))


def main() -> None:
    from armlab.util.runmeta import write_runmeta

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help=".npz path; run metadata goes next to it as .json")
    ap.add_argument("--size", type=int, default=32768)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    args = ap.parse_args()
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"  # tiny IK solves per worker; threaded BLAS only contends
    t0 = time.perf_counter()
    buf, acceptance = build(args.size, args.seed, args.workers)
    save(args.out, buf)
    seconds = round(time.perf_counter() - t0, 1)
    write_runmeta(
        Path(args.out).with_suffix(".json"),
        vars(args),
        args.seed,
        {"acceptance": acceptance, "seconds": seconds, "families": FAMILIES},
    )
    print(f"{args.size} states in {seconds} s -> {args.out}")
    print("acceptance: " + ", ".join(f"{k} {v:.2f}" for k, v in acceptance.items()))


if __name__ == "__main__":
    main()
