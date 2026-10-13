"""Train the SGS teacher: reset buffer -> MJX/Warp teacher env -> SGS (or uniform) resets -> PPO.

Writes to --out: metrics.jsonl (one line per iteration: PPO stats, success by reset family among
episodes that finished, SGS coverage and mean success rate of seen configurations),
checkpoints/iter_<n>.npz (policy, critic and observation normalizer), and runmeta.json.

    uv run --extra rl python -m armlab.rl.train_teacher --buffer artifacts/resets/v1.npz \\
        --strategy sgs --envs 4096 --iters 2000 --seed 0 --out runs/teacher-sgs-s0
"""

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from armlab.rl import ppo, sgs_jax
from armlab.rl.buffer import load
from armlab.rl.resets import FAMILIES
from armlab.rl.teacher_env import MAX_STEPS, OBS_DIM
from armlab.rl.teacher_mjx import TeacherMJX
from armlab.rl.vec_env import STRATEGIES, SGSVecEnv


def family_metrics(family: jax.Array):
    """metrics_fn for ppo.make_train: success per reset family, SGS coverage and success rates."""

    def fn(dones, info, carry):
        fam, ok = family[info["config"]], info["success"] * dones
        out = {}
        for k, name in enumerate(FAMILIES):
            ended = (dones * (fam == k)).sum()
            out[f"success_{name}"] = (ok * (fam == k)).sum() / jnp.maximum(ended, 1)
        seen = carry.sgs.counts > 0
        rates = sgs_jax.success_rates(carry.sgs)
        out["sgs_coverage"] = seen.mean()
        out["sgs_seen_success"] = (rates * seen).sum() / jnp.maximum(seen.sum(), 1)
        return out

    return fn


def save_checkpoint(path: Path, learner: ppo.Learner) -> None:
    leaves = jax.tree_util.tree_leaves_with_path({"params": learner.params, "norm": learner.norm._asdict()})
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{jax.tree_util.keystr(p): np.asarray(x) for p, x in leaves})


def load_checkpoint(path: str | Path, like: ppo.Learner) -> tuple[dict, ppo.Normalizer]:
    """Restore params and normalizer into the structure of `like` (e.g. a fresh learner)."""
    tree = {"params": like.params, "norm": like.norm._asdict()}
    with np.load(path) as f:
        leaves = [
            jnp.asarray(f[jax.tree_util.keystr(p)]) for p, _ in jax.tree_util.tree_leaves_with_path(tree)
        ]
    out = jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(tree), leaves)
    return out["params"], ppo.Normalizer(**out["norm"])


def main(argv=None) -> None:
    from armlab.util.runmeta import write_runmeta

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--buffer", required=True, help="reset buffer .npz from armlab.rl.buffer")
    ap.add_argument("--strategy", choices=STRATEGIES, default="sgs")
    ap.add_argument("--envs", type=int, default=ppo.PPOConfig.envs)
    ap.add_argument("--steps", type=int, default=ppo.PPOConfig.steps, help="rollout steps per iteration")
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=ppo.PPOConfig.lr)
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS, help="episode step limit")
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    buf = load(args.buffer)
    cfg = replace(ppo.PPOConfig(), envs=args.envs, steps=args.steps, lr=args.lr)
    env = SGSVecEnv(TeacherMJX(batch=args.envs, max_steps=args.max_steps), buf, strategy=args.strategy)
    init, iteration = ppo.make_train(env, OBS_DIM, 6, cfg, family_metrics(jnp.asarray(buf["family"])))
    learner, iteration = init(jax.random.key(args.seed)), jax.jit(iteration)
    meta = {**vars(args), "ppo": asdict(cfg), "devices": [str(d) for d in jax.devices()]}
    write_runmeta(out / "runmeta.json", meta, args.seed)

    t0 = time.perf_counter()
    with open(out / "metrics.jsonl", "w") as log:
        for i in range(1, args.iters + 1):
            learner, m = iteration(learner)
            row = {"iter": i, "seconds": round(time.perf_counter() - t0, 1)}
            row |= {k: round(float(v), 5) for k, v in m.items()}
            log.write(json.dumps(row) + "\n")
            log.flush()
            if i == 1 or i % 10 == 0:
                fams = " ".join(f"{n}={row[f'success_{n}']:.2f}" for n in FAMILIES)
                print(
                    f"iter {i} {row['seconds']}s success={row['success']:.3f} {fams} "
                    f"coverage={row['sgs_coverage']:.4f}",
                    flush=True,
                )
            if i % args.save_every == 0 or i == args.iters:
                save_checkpoint(out / "checkpoints" / f"iter_{i}.npz", learner)
    steps = args.iters * args.envs * args.steps
    print(f"{steps} env steps in {time.perf_counter() - t0:.0f} s -> {out}")


if __name__ == "__main__":
    main()
