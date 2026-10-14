"""Estimate the pick-and-place layout from the demos' joint data (Q15 has no measurements yet).

For every episode of a local LeRobot v3 copy of the dataset, find when the gripper closes on the
bottle and when it opens again, and run the joint state through the SO-101 model (policy/sim/so101.py)
to get the gripper site position at those moments: where the gripper closes approximates the bottle,
where it opens approximates the plate. Results are estimates under the joint-conversion assumption in
so101.py, not measurements.

    uv run --group sim python -m policy.sim.demo_layout --dataset <dir with data/ and meta/> \\
        --exclude 11 14 --out runs/demo_layout.json
"""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import pyarrow.parquet as pq

from policy.sim.so101 import lerobot_to_sim, load_model

OPEN = 30.0  # gripper reading above which the gripper counts as open (0-100 scale)
CLOSING = 15.0  # a grasp starts once the reading is this far below its peak opening ...
STILL = 0.4  # ... and stops changing by more than this per frame for HOLD frames
HOLD = 10
RELEASE = 10.0  # the release starts when the reading rises this far above the held value


def grip_events(grip: np.ndarray) -> tuple[int, int, float] | None:
    """(grasp frame, release frame, held reading) from a gripper trace, or None if not found."""
    if not (grip > OPEN).any():
        return None
    i_open = int(np.argmax(grip > OPEN))
    i_peak = i_open + int(np.argmax(grip[i_open : i_open + 150]))
    for j in range(i_peak, len(grip) - HOLD):
        if grip[j] < grip[i_peak] - CLOSING and np.abs(np.diff(grip[j : j + HOLD])).max() < STILL:
            held = float(np.median(grip[j : j + 3 * HOLD]))
            after = np.flatnonzero(grip[j + 3 * HOLD :] > held + RELEASE)
            return (j, j + 3 * HOLD + int(after[0]), held) if after.size else None
    return None


def gripper_positions(model: mujoco.MjModel, states: np.ndarray) -> np.ndarray:
    data, site = mujoco.MjData(model), model.site("gripperframe").id
    out = np.empty((len(states), 3))
    for i, q in enumerate(lerobot_to_sim(states)):
        data.qpos[:5] = q
        mujoco.mj_kinematics(model, data)
        out[i] = data.site_xpos[site]
    return out


def episodes(dataset: Path) -> dict[int, np.ndarray]:
    files = sorted((dataset / "data").glob("chunk-*/file-*.parquet"))
    t = pq.ParquetDataset(files).read(columns=["episode_index", "observation.state"])
    ep = t.column("episode_index").to_numpy()
    state = np.stack(t.column("observation.state").to_pylist()).astype(np.float64)
    return {int(e): state[ep == e] for e in np.unique(ep)}


def estimate(dataset: Path, exclude=()) -> dict:
    model, rows = load_model(), {}
    for e, s in episodes(dataset).items():
        ev = grip_events(s[:, 5])
        if ev is None:
            rows[e] = {"events": None}
            continue
        g, r, held = ev
        xyz = gripper_positions(model, s)
        rows[e] = {
            "grasp_frame": g,
            "release_frame": r,
            "held_grip": held,
            "grasp_xyz": xyz[g].round(4).tolist(),
            "release_xyz": xyz[r].round(4).tolist(),
            "carry_max_z": float(xyz[g:r, 2].max().round(4)),
            "start_xyz": xyz[0].round(4).tolist(),
        }
    used = [v for e, v in rows.items() if e not in exclude and v.get("events", True) is not None]
    summary = {}
    for key in ("grasp_xyz", "release_xyz", "carry_max_z", "held_grip"):
        a = np.array([v[key] for v in used])
        summary[key] = {f: getattr(a, f)(axis=0).round(4).tolist() for f in ("mean", "std", "min", "max")}
    return {"episodes": rows, "used": len(used), "excluded": list(exclude), "summary": summary}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--exclude", type=int, nargs="*", default=[11, 14], help="SETUP.md §3: bad starts")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    res = estimate(args.dataset, tuple(args.exclude))
    for k, v in res["summary"].items():
        print(f"{k:12s} mean {v['mean']}  std {v['std']}  min {v['min']}  max {v['max']}")
    print(f"{res['used']} episodes used, excluded {res['excluded']}; estimates, not measurements")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(res, indent=2) + "\n")


if __name__ == "__main__":
    main()
