"""SGS reset buffer for pick and place (arXiv:2610.12465v1 App. A.2: a pre-collected buffer of reset
states balanced across reset strategies; every training episode starts from one row).

Rows are grouped by size variant, then by family in FAMILIES order; the row index is the SGS
configuration ID, and SGS samples within an env's variant. Each fixed-size chunk has its own
SeedSequence, so the buffer depends on the seed, not on --workers.

    uv run --group sim python -m policy.sim.buffer --out runs/resets/v1.npz \\
        --variants 16 --per-variant 2048 --seed 0 --workers 8
"""

import argparse
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np

from policy.sim.resets import FAMILIES, ResetState, Sampler
from policy.sim.scene import reachable_grid, variants

CHUNK = 64
MAX_TRIES = 20
FIELDS = ("qpos", "qvel", "ctrl")
_sampler: tuple[tuple[int, int], Sampler] | None = None


def _chunk(job) -> tuple[list[ResetState], int]:
    global _sampler
    seed, k, variant_seed, v, family, index, count = job
    if _sampler is None or _sampler[0] != (k, variant_seed):
        _sampler = (k, variant_seed), Sampler(variants(k, variant_seed), reachable_grid())
    rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(v, family, index)))
    out, tries = [], 0
    while len(out) < count:
        tries += 1
        if tries > MAX_TRIES * count:
            raise RuntimeError(f"variant {v} {FAMILIES[family]}: acceptance below 1/{MAX_TRIES}")
        s = _sampler[1].sample(v, FAMILIES[family], rng)
        if s is not None:
            out.append(s)
    return out, tries


def build(k: int, per_variant: int, seed: int, variant_seed: int = 0, workers: int = 1, chunk: int = CHUNK):
    """Returns (buffer arrays, per-family acceptance). per_variant must split evenly across FAMILIES."""
    if per_variant < len(FAMILIES) or per_variant % len(FAMILIES):
        raise ValueError(f"per_variant must be a positive multiple of {len(FAMILIES)}")
    per = per_variant // len(FAMILIES)
    jobs = [
        (seed, k, variant_seed, v, f, c, min(chunk, per - c * chunk))
        for v in range(k)
        for f in range(len(FAMILIES))
        for c in range(-(-per // chunk))
    ]
    if workers > 1:
        with mp.get_context("spawn").Pool(workers) as pool:
            results = pool.map(_chunk, jobs)
    else:
        results = [_chunk(j) for j in jobs]
    states = [s for chunk_states, _ in results for s in chunk_states]
    tries = np.zeros(len(FAMILIES))
    for job, (_, t) in zip(jobs, results, strict=True):
        tries[job[4]] += t
    buf = {f: np.stack([getattr(s, f) for s in states]) for f in FIELDS}
    buf["variant"] = np.array([s.variant for s in states], np.int16)
    buf["family"] = np.array([FAMILIES.index(s.family) for s in states], np.int8)
    buf["bottle_xy"] = np.array([s.bottle_xy for s in states], np.float32)
    buf["plate_xy"] = np.array([s.plate_xy for s in states], np.float32)
    buf["meta"] = np.array([k, variant_seed])
    return buf, {name: k * per / tries[f] for f, name in enumerate(FAMILIES)}


def save(path, buf: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, families=np.array(FAMILIES), **buf)


def load(path) -> dict:
    with np.load(path) as f:
        buf = {key: f[key] for key in f.files}
    if tuple(buf.pop("families")) != FAMILIES:
        raise ValueError("buffer was built with different reset families")
    return buf


def state(buf: dict, i: int) -> ResetState:
    return ResetState(
        FAMILIES[buf["family"][i]],
        int(buf["variant"][i]),
        tuple(map(float, buf["bottle_xy"][i])),
        tuple(map(float, buf["plate_xy"][i])),
        *(buf[f][i] for f in FIELDS),
    )


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", type=int, default=16)
    ap.add_argument("--per-variant", type=int, default=2048)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--variant-seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    args = ap.parse_args(argv)
    t0 = time.perf_counter()
    buf, acceptance = build(args.variants, args.per_variant, args.seed, args.variant_seed, args.workers)
    save(args.out, buf)
    print(f"{len(buf['family'])} states in {time.perf_counter() - t0:.0f} s -> {args.out}")
    print("acceptance: " + ", ".join(f"{k} {v:.2f}" for k, v in acceptance.items()))


if __name__ == "__main__":
    main()
