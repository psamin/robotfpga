"""Held-out evaluation of an SGS teacher checkpoint on the CPU teacher env.

Following arXiv:2610.12465v1 §4.3, evaluation resets are freshly sampled from the training
distribution (a different buffer seed), never training rows. The policy acts deterministically
(the Gaussian mean). --expert runs the scripted expert through the same delta-action interface
instead, as a reference.

    uv run --extra rl python -m armlab.rl.eval_teacher \\
        --checkpoint runs/teacher-sgs-s0/checkpoints/iter_2000.npz --out runs/teacher-sgs-s0/eval.json
"""

import argparse
import json
import os
from pathlib import Path

import jax
import numpy as np

from armlab.rl import ppo
from armlab.rl.buffer import build, state
from armlab.rl.resets import FAMILIES
from armlab.rl.teacher_env import ACTION_SCALE, MAX_STEPS, OBS_DIM, TeacherEnv
from armlab.rl.train_teacher import load_checkpoint
from armlab.sim.expert import Expert

EVAL_SEED = 1000  # training buffers use small seeds; keep evaluation resets disjoint


def checkpoint_policy(path: str | Path):
    """Deterministic policy obs -> action from a checkpoint; hidden sizes come from runmeta.json."""
    meta = json.loads((Path(path).parent.parent / "runmeta.json").read_text())
    cfg = ppo.PPOConfig(hidden=tuple(meta["config"]["ppo"]["hidden"]))
    like = ppo.Learner(
        ppo.init_params(jax.random.key(0), OBS_DIM, 6, cfg), None, ppo.norm_init(OBS_DIM), *[None] * 3
    )
    params, norm = load_checkpoint(path, like)
    act = jax.jit(lambda obs: ppo.mlp(params["actor"], ppo.normalize(norm, obs)))
    return lambda env, obs: np.asarray(act(obs))


def expert_policy():
    experts = {}

    def act(env: TeacherEnv, obs):
        if env.env.t == 0 or id(env) not in experts:
            experts[id(env)] = Expert(env.env)
        return np.clip((experts[id(env)].act() - env.target) / ACTION_SCALE, -1, 1)

    return act


def evaluate(policy, per_family: int, seed: int = EVAL_SEED, max_steps: int = MAX_STEPS) -> dict:
    buf, _ = build(per_family * len(FAMILIES), seed, workers=min(os.cpu_count() or 1, 8))
    env, rows = TeacherEnv(max_steps=max_steps), []
    for i in range(len(buf["family"])):
        s = state(buf, i)
        obs, done = env.reset(s), False
        while not done:
            obs, _, done, info = env.step(policy(env, obs))
        rows.append((s.family, info["success"], info["failure"], info["t"]))
    out = {}
    for f in FAMILIES:
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
    from armlab.util.runmeta import write_runmeta

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    who = ap.add_mutually_exclusive_group(required=True)
    who.add_argument("--checkpoint", help="checkpoints/iter_<n>.npz inside a train_teacher run")
    who.add_argument("--expert", action="store_true", help="scripted expert reference")
    ap.add_argument("--episodes-per-family", type=int, default=100)
    ap.add_argument("--seed", type=int, default=EVAL_SEED, help="evaluation buffer seed")
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    policy = expert_policy() if args.expert else checkpoint_policy(args.checkpoint)
    results = evaluate(policy, args.episodes_per_family, args.seed, args.max_steps)
    write_runmeta(args.out, vars(args), args.seed, results)
    print(f"overall success {results['overall_success']:.3f}")
    for f in FAMILIES:
        r = results[f]
        print(
            f"  {f:13s} success {r['success']:.3f}  failure {r['failure']:.3f}  steps {r['mean_steps']:.0f}"
        )


if __name__ == "__main__":
    main()
