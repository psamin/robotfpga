"""Pack real sim frames into intref input packets for the FPGA golden vectors.

    uv run python scripts/pack_frames.py --data datasets/v2 --out packets.npy --n 80
    python ref/vectors.py --model runs/qat/export --packets packets.npy     # FPGA-side tool (PR #6)
    make -C hls csim ART=$PWD/runs/qat/export                              # HLS C model (PR #8)

Frames are spread across episodes, instructions and episode phases (evenly spaced in time).
"""

import argparse

import numpy as np

from armlab.data.replay import load_episodes
from armlab.policy.qat import load_intref


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--per-episode", type=int, default=4)
    args = ap.parse_args()
    ref = load_intref()
    packets = []
    for _meta, ep in load_episodes(args.data):
        for t in np.linspace(0, len(ep["image"]) - 1, args.per_episode).astype(int):
            obs = {"image": ep["image"][t], "state": ep["state"][t], "instr": int(ep["instr"][t])}
            packets.append(np.frombuffer(ref.pack_obs(obs), np.uint8))
        if len(packets) >= args.n:
            break
    arr = np.stack(packets[: args.n])
    np.save(args.out, arr)
    print(f"wrote {args.out}: {arr.shape} uint8")


if __name__ == "__main__":
    main()
