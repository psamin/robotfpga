"""SmolVLA in the pick-and-place sim: the baseline of the unchanged fine-tuned checkpoint (step 8).

The observation matches training (SETUP.md §3): observation.images.camera1 = front (480x640) and
camera2 = wrist (640x480), CHW float in [0, 1]; observation.state = the 6 joints in LeRobot units
(degrees, gripper 0-100); the task text. Actions come back as LeRobot units and are converted to sim
joint targets, one per 30 Hz tick. The policy is any callable obs -> (n, 6) action chunk; the CLI
loads SmolVLA and its processors the way policy/smolvla_fpga.py does.

    python -m policy.sim.smolvla_sim --model ~/models/smolvla-so101_medicine_bottle_pickplace-9604 \\
        --cameras robot/calibration/cameras.json --episodes 40 --out runs/smolvla-sim-baseline.json
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np

from policy.sim.cameras import Renderer, load
from policy.sim.resets import Sampler, apply
from policy.sim.scene import SUBSTEPS, build, reachable_grid, success, variants
from policy.sim.so101 import gripper_to_sim, sim_to_lerobot

TASK = "Pick up the medicine bottle and place it on the plate"
MAX_STEPS = 600  # 20 s, the demos' episode limit (SETUP.md §1)


def observation(data: mujoco.MjData, renderer: Renderer) -> dict:
    frames = {
        f"observation.images.{k}": renderer.render(data, cam)
        for k, cam in (("camera1", "front"), ("camera2", "wrist"))
    }
    obs = {k: np.ascontiguousarray(v.transpose(2, 0, 1), dtype=np.float32) / 255.0 for k, v in frames.items()}
    obs["observation.state"] = sim_to_lerobot(data.qpos[:6]).astype(np.float32)
    obs["task"] = TASK
    return obs


def to_ctrl(action) -> np.ndarray:
    """One action in LeRobot units -> sim joint targets (rad)."""
    a = np.asarray(action, dtype=np.float64)
    return np.r_[np.radians(a[:5]), gripper_to_sim(a[5])]


def run_episode(policy, model, data, p, renderer: Renderer, max_steps: int = MAX_STEPS) -> dict:
    """Execute policy chunks at 30 Hz from the current state until success or max_steps."""
    t = 0
    while t < max_steps:
        for action in np.asarray(policy(observation(data, renderer)))[:, :6]:
            data.ctrl[:6] = np.clip(
                to_ctrl(action), model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
            )
            for _ in range(SUBSTEPS):
                mujoco.mj_step(model, data)
            t += 1
            if success(model, data, p):
                return {"success": True, "steps": t}
            if t >= max_steps:
                break
    return {"success": False, "steps": t}


def smolvla_policy(model_dir: str):
    """SmolVLA + processors as in policy/smolvla_fpga.py; returns obs -> (chunk, 6) in LeRobot units."""
    import torch
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = SmolVLAPolicy.from_pretrained(model_dir).float().to(device).eval()
    pre, post = make_pre_post_processors(
        policy.config,
        pretrained_path=model_dir,
        preprocessor_overrides={"device_processor": {"device": device}},
        postprocessor_overrides={"device_processor": {"device": device}},
    )

    def act(obs):
        batch = {k: (torch.from_numpy(v) if isinstance(v, np.ndarray) else v) for k, v in obs.items()}
        policy.reset()
        with torch.no_grad():
            chunk = policy.predict_action_chunk(pre(batch))
        return (
            post(chunk).float().cpu().numpy().reshape(-1, chunk.shape[-1])[: policy.config.n_action_steps, :6]
        )

    return act


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="fine-tuned SmolVLA checkpoint directory (never modified)")
    ap.add_argument("--cameras", required=True, help="calibrated cameras JSON (policy.sim.cameras)")
    ap.add_argument("--variants", type=int, default=8)
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--seed", type=int, default=2000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    cams, vs, rng = load(args.cameras), variants(args.variants), np.random.default_rng(args.seed)
    sampler, policy = Sampler(vs, reachable_grid()), smolvla_policy(args.model)
    models, rows = {}, []
    while len(rows) < args.episodes:
        v = int(rng.integers(args.variants))
        s = sampler.sample(v, "rest", rng)
        if s is None:
            continue
        if v not in models:
            m = build(vs[v], cams)
            models[v] = (m, Renderer(m))
        m, renderer = models[v]
        d = mujoco.MjData(m)
        apply(m, d, s)
        r = run_episode(policy, m, d, replace(vs[v], bottle_xy=s.bottle_xy, plate_xy=s.plate_xy), renderer)
        rows.append({"variant": v, **r})
        print(f"episode {len(rows)}: variant {v} success {r['success']} steps {r['steps']}", flush=True)
    result = {"model": args.model, "episodes": rows, "success": float(np.mean([r["success"] for r in rows]))}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(f"SmolVLA in sim, rest starts: success {result['success']:.3f} over {len(rows)} episodes")


if __name__ == "__main__":
    main()
