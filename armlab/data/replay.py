"""Replay a collected episode in sim from its seed, executed actions and recorded perturbation.

Used to check the data (Phase 3 acceptance) and to regenerate full-resolution frames for the
LeRobot / SmolVLA dataset without having stored them.

    uv run python -m armlab.data.replay --data datasets/v1 --episodes 20
"""

import argparse
import json
from pathlib import Path

import numpy as np

from armlab.sim.env import BlockSortEnv


def apply_event(env: BlockSortEnv, event: dict | None, t: int) -> None:
    if not event or event["t"] != t:
        return
    if event["kind"] == "instr_swap":
        env.instr = event["to"]
    else:
        from armlab.sim.env import INSTRUCTIONS

        color = INSTRUCTIONS[env.instr][0]
        jid = env.m.joint(f"cube_{color}").id
        adr = env.m.jnt_qposadr[jid]
        env.d.qpos[adr : adr + 3] = [event["to"][0], event["to"][1], 0.0125]
        env.d.qvel[env.m.jnt_dofadr[jid] :][:6] = 0


def replay(env: BlockSortEnv, meta: dict, executed: np.ndarray, keep_frames: bool = False):
    """Returns (success at the end, list of observations if keep_frames)."""
    obs = env.reset(seed=meta["seed"], instr=meta["instr"])
    frames = []
    for t, a in enumerate(executed):
        apply_event(env, meta.get("event"), t)
        obs["instr"] = env.instr
        if keep_frames:
            frames.append(obs)
        obs, _, _ = env.step(a)
    return env.success(), frames


def load_episodes(root: str | Path):
    """Yields (meta, arrays) per episode across all shards."""
    for p in sorted(Path(root).glob("shard_*.npz")):
        z = np.load(p)
        metas = json.loads(str(z["meta"]))
        starts = np.concatenate([[0], np.cumsum(z["lengths"])[:-1]])
        for m, s, n in zip(metas, starts, z["lengths"], strict=True):
            yield m, {k: z[k][s : s + n] for k in ("image", "state", "instr", "action", "executed")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--episodes", type=int, default=20)
    args = ap.parse_args()
    env = BlockSortEnv()
    ok = same = n = 0
    for meta, ep in load_episodes(args.data):
        if n >= args.episodes:
            break
        n += 1
        success, frames = replay(env, meta, ep["executed"], keep_frames=True)
        ok += success
        same += all(np.array_equal(f["image"], img) for f, img in zip(frames, ep["image"], strict=True))
    print(f"replayed {n} episodes: {ok} succeed, {same} with byte-identical frames")


if __name__ == "__main__":
    main()
