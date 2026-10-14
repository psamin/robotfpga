"""Held-out evaluation of a pick-and-place teacher checkpoint on the CPU teacher env.

Following arXiv:2610.12465v1 §4.3, evaluation resets are freshly sampled from the training
distribution with a different seed, never training rows, using the run's size variants. The policy
acts deterministically (the Gaussian mean). --expert runs the scripted expert from rest states
instead, as a reference.

    uv run --group rl --group sim python -m policy.rl.eval_teacher \\
        --checkpoint runs/t/checkpoints/iter_2000.npz --out runs/t/eval.json
"""

import argparse
import json
import os
from pathlib import Path

import jax
import numpy as np

from policy.rl import ppo
from policy.rl.train_teacher import load_checkpoint, runmeta
from policy.sim.buffer import build, state
from policy.sim.expert import Expert
from policy.sim.families import FAMILIES
from policy.sim.resets import Sampler
from policy.sim.scene import reachable_grid, variants
from policy.sim.teacher_env import ACTION_SCALE, MAX_STEPS, OBS_DIM, TeacherEnv

EVAL_SEED = 1000  # training buffers use small seeds; keep evaluation resets disjoint


def checkpoint_policy(path):
    """Deterministic obs -> action from a checkpoint; hidden sizes and variants from runmeta.json."""
    meta = json.loads((Path(path).parent.parent / "runmeta.json").read_text())["config"]
    cfg = ppo.PPOConfig(hidden=tuple(meta["ppo"]["hidden"]))
    like = ppo.Learner(
        ppo.init_params(jax.random.key(0), OBS_DIM, 6, cfg), None, ppo.norm_init(OBS_DIM), *[None] * 3
    )
    params, norm = load_checkpoint(path, like)
    act = jax.jit(lambda obs: ppo.mlp(params["actor"], ppo.normalize(norm, obs)))
    return (lambda env, obs: np.asarray(act(obs))), (meta["variants"], meta["variant_seed"])


def expert_policy():
    experts = {}

    def act(env: TeacherEnv, obs):
        if env.t == 0 or id(env) not in experts:
            experts[id(env)] = Expert(env.m, env.d, env.p)
        return np.clip((experts[id(env)].act() - env.target) / ACTION_SCALE, -1, 1)

    return act


def evaluate(
    policy,
    k: int,
    variant_seed: int,
    per_variant: int,
    seed: int = EVAL_SEED,
    max_steps: int = MAX_STEPS,
    families=FAMILIES,
) -> dict:
    buf, _ = build(k, per_variant, seed, variant_seed, workers=min(os.cpu_count() or 1, 8))
    env, rows = TeacherEnv(Sampler(variants(k, variant_seed), reachable_grid()), max_steps), []
    for i in range(len(buf["family"])):
        s = state(buf, i)
        if s.family not in families:
            continue
        obs, done = env.reset(s), False
        while not done:
            obs, _, done, info = env.step(policy(env, obs))
        rows.append((s.family, info["success"], info["failure"], info["t"]))
    out = {}
    for f in families:
        r = [x for x in rows if x[0] == f]
        out[f] = {
            "episodes": len(r),
            "success": float(np.mean([x[1] for x in r])),
            "failure": float(np.mean([x[2] for x in r])),
            "mean_steps": float(np.mean([x[3] for x in r])),
        }
    out["overall_success"] = float(np.mean([x[1] for x in rows]))
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    who = ap.add_mutually_exclusive_group(required=True)
    who.add_argument("--checkpoint", help="checkpoints/iter_<n>.npz inside a train_teacher run")
    who.add_argument("--expert", action="store_true", help="scripted expert from rest states")
    ap.add_argument("--variants", type=int, default=8, help="with --expert; checkpoints use their run's")
    ap.add_argument(
        "--per-variant", type=int, default=40, help="evaluation states per variant (all families)"
    )
    ap.add_argument("--seed", type=int, default=EVAL_SEED)
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if args.expert:
        policy, (k, vseed), families = expert_policy(), (args.variants, 0), ("rest",)
    else:
        (policy, (k, vseed)), families = checkpoint_policy(args.checkpoint), FAMILIES
    results = evaluate(policy, k, vseed, args.per_variant, args.seed, args.max_steps, families)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({**runmeta(vars(args)), "results": results}, indent=2) + "\n")
    print(f"overall success {results['overall_success']:.3f}")
    for f in families:
        r = results[f]
        print(
            f"  {f:13s} success {r['success']:.3f}  failure {r['failure']:.3f}  steps {r['mean_steps']:.0f}"
        )


if __name__ == "__main__":
    main()
