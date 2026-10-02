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
import os
import time
from pathlib import Path

import numpy as np

from armlab.sim.env import BlockSortEnv
from armlab.sim.expert import Expert, Phase, expert_chunk
from armlab.sim.perturb import Perturber

# per-episode sigma, normalized units (1 unit ~ 1.7 rad). 0.05 broke grasps (4/19); with the
# stateless expert 0.03 did too (1/14). DAgger covers larger drift directly.
NOISE_LEVELS = (0.0, 0.005, 0.01, 0.02)
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


def collect_episode(
    env: BlockSortEnv, plan: dict, max_steps: int = 450, policy=None, beta: float = 0.0
) -> dict:
    """One episode. The driver is the expert plus DART noise, or (policy given) the learner via a
    chunk executor, with the expert's action swapped in at each tick with probability beta
    (DAgger's mixture policy [Ross2011-DAgger]). Either way every frame is labelled with the
    expert's next-8-step plan from that exact state (expert_chunk)."""
    rng = np.random.default_rng(plan["seed"] + 2_000_003)
    obs = env.reset(seed=plan["seed"], instr=plan["instr"])
    expert = Expert(env)
    pert = Perturber(plan["kind"], rng)
    executor = None
    if policy is not None:
        from armlab.control.chunking import ChunkExecutor

        executor = ChunkExecutor(policy)
    rec = {k: [] for k in ("image", "state", "instr", "chunk", "executed")}
    noise = np.zeros(6)
    ok = False
    for t in range(max_steps):
        pert.maybe_apply(env, expert, t)
        obs["instr"] = env.instr
        chunk = expert_chunk(env, expert)
        label = env.normalize(expert.act())
        if executor is None:
            noise = NOISE_RHO * noise + np.sqrt(1 - NOISE_RHO**2) * rng.normal(0, plan["sigma"], 6)
            noise[5] = 0.0  # never jitter the gripper command
            act = np.clip(label + noise, -1, 1).astype(np.float32)
        else:
            act = executor.step(obs, t).astype(np.float32)
            if rng.random() < beta:
                act = label.astype(np.float32)
        for k, v in (
            ("image", obs["image"]),
            ("state", obs["state"]),
            ("instr", env.instr),
            ("chunk", chunk),
            ("executed", act),
        ):
            rec[k].append(v)
        obs, _, info = env.step(act)
        if info["success"] and (executor is not None or expert.phase == Phase.DONE):
            ok = True
            break
    out = {k: np.stack(v) if k != "instr" else np.array(v, np.int8) for k, v in rec.items()}
    out["state"] = out["state"].astype(np.float32)
    out["action"] = out["chunk"][:, 0]
    out["meta"] = {
        **plan,
        "success": ok,
        "steps": len(rec["image"]),
        "event": pert.event,
        "driver": "expert" if executor is None else "policy",
        "beta": beta,
    }
    return out


def _worker(args):
    plans, out_dir, wid, policy_ckpt, beta, max_steps = args
    env = BlockSortEnv()
    policy = None
    if policy_ckpt:
        from armlab.backends.torch_backend import TorchBackend

        policy = TorchBackend(policy_ckpt)
    shard, metas = [], []
    for p in plans:
        ep = collect_episode(env, p, max_steps=max_steps, policy=policy, beta=beta)
        metas.append(ep["meta"])
        if ep["meta"]["success"] or policy is not None:  # DAgger keeps failures: they are the point
            shard.append(ep)
    path = Path(out_dir) / f"shard_{wid:03d}.npz"
    np.savez_compressed(
        path,
        lengths=np.array([len(e["action"]) for e in shard], np.int32),
        **{
            k: np.concatenate([e[k] for e in shard])
            for k in ("image", "state", "instr", "action", "chunk", "executed")
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
    ap.add_argument("--policy", default=None, help="float checkpoint: DAgger round driven by this learner")
    ap.add_argument("--beta", type=float, default=0.0, help="per-tick probability of executing the expert")
    ap.add_argument("--max-steps", type=int, default=450)
    args = ap.parse_args()

    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = "1"  # workers each solve tiny IK problems; threaded BLAS only contends
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    plans = [
        episode_plan(args.seed0 + 4 * i + instr, instr)
        for i in range(args.episodes_per_instr)
        for instr in range(4)
    ]
    chunks = [
        (plans[w :: args.workers], str(out), w, args.policy, args.beta, args.max_steps)
        for w in range(args.workers)
    ]
    t0 = time.perf_counter()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        metas = [m for ms in pool.map(_worker, chunks) for m in ms]
    ok = [m for m in metas if m["success"]]
    kept = metas if args.policy else ok
    summary = {
        "episodes": len(metas),
        "success": len(ok),
        "kept": len(kept),
        "frames": int(sum(m["steps"] for m in kept)),
        "by_kind": {k: sum(m["kind"] == k for m in ok) for k in ("none", "cube_move", "instr_swap")},
        "failed_by_kind": {k: sum(m["kind"] == k and not m["success"] for m in metas) for k in KIND_P},
        "seconds": round(time.perf_counter() - t0, 1),
        "args": vars(args),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
