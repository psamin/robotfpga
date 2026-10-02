"""Collect expert demonstrations (build spec Phase 3).

DART-style noise injection [Laskey2017-DART]: the arm executes the expert's action plus
time-correlated noise, but the label is the expert's clean action, so the data covers the
states a slightly-wrong learner drifts into and shows how to recover. A share of episodes also
gets a Stage C disturbance (cube moved / instruction swapped) that the reactive expert handles.

Each episode is saved with everything needed to replay it exactly (seed, instruction, executed
actions, perturbation), so 288x288 frames for the LeRobot/SmolVLA dataset can be regenerated
without storing them.

    uv run python -m armlab.data.collect --out datasets/v1 --episodes-per-instr 300 --workers 8
"""

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

from armlab.sim.env import BlockSortEnv
from armlab.sim.expert import Expert, Phase
from armlab.sim.perturb import Perturber

NOISE_LEVELS = (0.0, 0.01, 0.02, 0.03)  # per-episode sigma, normalized units (1 unit ~ 1.7 rad); 0.05 broke grasps
NOISE_RHO = 0.8  # AR(1) correlation of the injected noise
KIND_P = {"none": 0.6, "cube_move": 0.2, "instr_swap": 0.2}


def episode_plan(seed: int, instr: int) -> dict:
    rng = np.random.default_rng(seed + 1_000_003)
    return {
        "seed": seed,
        "instr": instr,
        "sigma": float(rng.choice(NOISE_LEVELS)),
        "kind": str(rng.choice(list(KIND_P), p=list(KIND_P.values()))),
    }


def collect_episode(env: BlockSortEnv, plan: dict, max_steps: int = 450) -> dict:
    rng = np.random.default_rng(plan["seed"] + 2_000_003)
    obs = env.reset(seed=plan["seed"], instr=plan["instr"])
    expert = Expert(env)
    pert = Perturber(plan["kind"], rng)
    frames, states, instrs, labels, executed = [], [], [], [], []
    noise = np.zeros(6)
    ok = False
    for t in range(max_steps):
        pert.maybe_apply(env, expert, t)
        obs["instr"] = env.instr
        label = env.normalize(expert.act())
        noise = NOISE_RHO * noise + np.sqrt(1 - NOISE_RHO**2) * rng.normal(0, plan["sigma"], 6)
        noise[5] = 0.0  # never jitter the gripper command
        act = np.clip(label + noise, -1, 1).astype(np.float32)
        frames.append(obs["image"])
        states.append(obs["state"])
        instrs.append(env.instr)
        labels.append(label)
        executed.append(act)
        obs, _, info = env.step(act)
        if info["success"] and expert.phase == Phase.DONE:
            ok = True
            break
    return {
        "image": np.stack(frames),
        "state": np.stack(states).astype(np.float32),
        "instr": np.array(instrs, np.int8),
        "action": np.stack(labels).astype(np.float32),
        "executed": np.stack(executed),
        "meta": {**plan, "success": ok, "steps": len(frames), "event": pert.event},
    }


def _worker(args):
    plans, out_dir, wid = args
    env = BlockSortEnv()
    shard, metas = [], []
    for p in plans:
        ep = collect_episode(env, p)
        metas.append(ep["meta"])
        if ep["meta"]["success"]:
            shard.append(ep)
    path = Path(out_dir) / f"shard_{wid:03d}.npz"
    lens = np.array([len(e["action"]) for e in shard], np.int32)
    np.savez_compressed(
        path,
        lengths=lens,
        **{
            k: np.concatenate([e[k] for e in shard])
            for k in ("image", "state", "instr", "action", "executed")
        },
        meta=np.array(json.dumps([e["meta"] for e in shard])),
    )
    return metas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes-per-instr", type=int, default=300)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    plans = [
        episode_plan(args.seed0 + 4 * i + instr, instr)
        for i in range(args.episodes_per_instr)
        for instr in range(4)
    ]
    chunks = [(plans[w :: args.workers], str(out), w) for w in range(args.workers)]
    t0 = time.perf_counter()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        metas = [m for ms in pool.map(_worker, chunks) for m in ms]
    ok = [m for m in metas if m["success"]]
    summary = {
        "episodes": len(metas),
        "kept": len(ok),
        "frames": int(sum(m["steps"] for m in ok)),
        "by_kind": {k: sum(m["kind"] == k for m in ok) for k in ("none", "cube_move", "instr_swap")},
        "failed_by_kind": {k: sum(m["kind"] == k and not m["success"] for m in metas) for k in KIND_P},
        "seconds": round(time.perf_counter() - t0, 1),
        "args": vars(args),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
