"""Train the SGS teacher for pick and place: reset buffer -> VariantVecEnv (SGS or uniform) -> PPO.

Writes to --out: metrics.jsonl (one line per iteration: PPO stats, success and finished episodes per
reset family, SGS coverage and mean success of seen rows, env steps per second),
checkpoints/iter_<n>.npz (policy, critic, observation normalizer) and runmeta.json.

    uv run --group rl --group sim --group gpu python -m policy.rl.train_teacher \\
        --buffer runs/resets/v1.npz --strategy sgs --envs-per-variant 256 --iters 2000 --seed 0 --out runs/t
"""

import argparse
import json
import platform
import subprocess
import time
from dataclasses import asdict, replace
from importlib import metadata
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from policy.rl import ppo, sgs_jax
from policy.sim.buffer import load
from policy.sim.families import FAMILIES
from policy.sim.scene import variants
from policy.sim.teacher_env import MAX_STEPS, OBS_DIM
from policy.sim.vec_env import STRATEGIES, VariantVecEnv


def family_metrics(family: jax.Array):
    """metrics_fn for ppo.make_train: success per reset family, SGS coverage and success rates."""

    def fn(dones, info, carry):
        fam, ok = family[info["config"]], info["success"] * dones
        out = {}
        for k, name in enumerate(FAMILIES):
            ended = (dones * (fam == k)).sum()
            out[f"episodes_{name}"] = ended
            out[f"success_{name}"] = (ok * (fam == k)).sum() / jnp.maximum(ended, 1)
        seen = carry.sgs.counts > 0
        out["sgs_coverage"] = seen.mean()
        out["sgs_seen_success"] = (sgs_jax.success_rates(carry.sgs) * seen).sum() / jnp.maximum(seen.sum(), 1)
        return out

    return fn


def save_checkpoint(path: Path, learner: ppo.Learner) -> None:
    leaves = jax.tree_util.tree_leaves_with_path({"params": learner.params, "norm": learner.norm._asdict()})
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{jax.tree_util.keystr(p): np.asarray(x) for p, x in leaves})


def load_checkpoint(path, like: ppo.Learner) -> tuple[dict, ppo.Normalizer]:
    """Restore params and normalizer into the structure of `like` (e.g. a fresh learner)."""
    tree = {"params": like.params, "norm": like.norm._asdict()}
    with np.load(path) as f:
        leaves = [
            jnp.asarray(f[jax.tree_util.keystr(p)]) for p, _ in jax.tree_util.tree_leaves_with_path(tree)
        ]
    out = jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(tree), leaves)
    return out["params"], ppo.Normalizer(**out["norm"])


def runmeta(config: dict) -> dict:
    git = (
        subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        or "unknown"
    )
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
    versions = {"python": platform.python_version()}
    for pkg in ("jax", "mujoco", "mujoco-warp", "numpy"):
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            pass
    return {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git": git + ("-dirty" if dirty else ""),
        "config": config,
        "versions": versions,
        "devices": [str(d) for d in jax.devices()],
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--buffer", required=True, help="reset buffer .npz from policy.sim.buffer")
    ap.add_argument("--strategy", choices=STRATEGIES, default="sgs")
    ap.add_argument("--envs-per-variant", type=int, default=256)
    ap.add_argument("--steps", type=int, default=ppo.PPOConfig.steps, help="rollout steps per iteration")
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=ppo.PPOConfig.lr)
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS, help="episode step limit")
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--impl", default="warp")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    buf = load(args.buffer)
    k, variant_seed = (int(x) for x in buf["meta"])
    env = VariantVecEnv(buf, variants(k, variant_seed), args.envs_per_variant, args.strategy, impl=args.impl)
    for sim in env.sims:
        sim.max_steps = args.max_steps
    cfg = replace(ppo.PPOConfig(), envs=env.batch, steps=args.steps, lr=args.lr)
    init, iteration = ppo.make_train(env, OBS_DIM, 6, cfg, family_metrics(jnp.asarray(buf["family"])))
    learner, iteration = init(jax.random.key(args.seed)), jax.jit(iteration)
    meta = runmeta({**vars(args), "variants": k, "variant_seed": variant_seed, "ppo": asdict(cfg)})
    (out / "runmeta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")

    t0 = last = time.perf_counter()
    with open(out / "metrics.jsonl", "w") as log:
        for i in range(1, args.iters + 1):
            learner, m = iteration(learner)
            now = time.perf_counter()
            row = {
                "iter": i,
                "seconds": round(now - t0, 1),
                "env_steps_per_s": round(cfg.envs * cfg.steps / (now - last)),
            }
            last = now
            row |= {key: round(float(v), 5) for key, v in m.items()}
            log.write(json.dumps(row) + "\n")
            log.flush()
            if i == 1 or i % 10 == 0:
                fams = " ".join(f"{n}={row[f'success_{n}']:.2f}" for n in FAMILIES)
                print(
                    f"iter {i} {row['seconds']}s {row['env_steps_per_s']} steps/s "
                    f"success={row['success']:.3f} {fams} coverage={row['sgs_coverage']:.4f}",
                    flush=True,
                )
            if i % args.save_every == 0 or i == args.iters:
                save_checkpoint(out / "checkpoints" / f"iter_{i}.npz", learner)
    print(f"{args.iters * cfg.envs * cfg.steps} env steps in {time.perf_counter() - t0:.0f} s -> {out}")


if __name__ == "__main__":
    main()
