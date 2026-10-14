"""Check the demo layout against the front camera, which looks straight down at the table.

Uses the output of policy.sim.demo_layout and a local copy of the dataset's front video:
- finds the bottle's orange cap in each used episode's first frame (bottle on its start spot) and
  shortly after the release (bottle on the plate);
- fits a 2D similarity transform (rotation, scale, shift) from the joint-data positions to the cap
  pixels in a top view, once with the joint conversion as assumed and once with shoulder_pan's sign
  flipped. The right sign fits clearly better and puts the robot base on the base in the image;
- fits a circle to the plate's patterned rim and measures the cap's area, in metres via the fit.

The scale is fitted at cap height, a few cm above the table, so table-level sizes read slightly
small. Estimates, not measurements.

    uv run --group sim python -m policy.sim.demo_video --dataset <dir> --layout runs/demo_layout.json
"""

import argparse
import json
import subprocess
from io import BytesIO
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

VIDEO = "videos/observation.images.front/chunk-000/file-000.mp4"
AFTER_RELEASE = 20  # frames: let the bottle settle on the plate
PLATE_X_MAX = 260  # px: the plate is in the left part of the front image


def read_frame(dataset: Path, t: float) -> np.ndarray:
    """HSV frame (0-255 per channel) at time t in the front video."""
    import imageio_ffmpeg
    from PIL import Image

    cmd = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-loglevel",
        "error",
        "-ss",
        f"{t:.3f}",
        "-i",
        str(dataset / VIDEO),
        "-frames:v",
        "1",
        "-f",
        "image2pipe",
        "-vcodec",
        "png",
        "-",
    ]
    png = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.asarray(Image.open(BytesIO(png)).convert("HSV")).astype(float)


def hsv_parts(hsv):
    return hsv[..., 0] * 360 / 255, hsv[..., 1] / 255, hsv[..., 2] / 255


def orange_mask(hsv):
    h, s, v = hsv_parts(hsv)
    return (h > 10) & (h < 40) & (s > 0.55) & (v > 0.45)


def cap_pixel(hsv, min_px: int = 30):
    ys, xs = np.nonzero(orange_mask(hsv))
    return (xs.mean(), ys.mean()) if xs.size >= min_px else None


def fit_similarity(world, pixels, flip_pan: bool = False):
    """Least-squares top-view similarity world (x, y) -> pixel (u, v). Returns params, residuals (px)."""
    w = np.asarray(world, float) * ([1, -1] if flip_pan else [1, 1])
    p = np.asarray(pixels, float) * [1, -1]  # image rows grow downward; a top view has y up
    a = np.zeros((2 * len(w), 4))
    a[0::2] = np.column_stack([w[:, 0], -w[:, 1], np.ones(len(w)), np.zeros(len(w))])
    a[1::2] = np.column_stack([w[:, 1], w[:, 0], np.zeros(len(w)), np.ones(len(w))])
    params, *_ = np.linalg.lstsq(a, p.ravel(), rcond=None)
    return params, np.linalg.norm((a @ params - p.ravel()).reshape(-1, 2), axis=1)


def to_world(params, u, v):
    a, b, tx, ty = params
    return np.linalg.solve([[a, -b], [b, a]], [u - tx, -v - ty])


def plate_circle(hsv):
    """Center and outer / inner radius (px) of the plate's saturated, non-orange rim pattern."""
    h, s, v = hsv_parts(hsv)
    rim = (s > 0.35) & (v > 0.25) & ~orange_mask(hsv)
    rim[:150] = False
    rim[:, PLATE_X_MAX:] = False
    ys, xs = np.nonzero(rim)
    c, *_ = np.linalg.lstsq(np.column_stack([xs, ys, np.ones_like(xs)]), xs**2 + ys**2, rcond=None)
    cx, cy = c[0] / 2, c[1] / 2
    d = np.hypot(xs - cx, ys - cy)
    return cx, cy, np.percentile(d, 97), np.percentile(d, 3)


def check(dataset: Path, layout: dict) -> dict:
    t = pq.read_table(dataset / "meta/episodes/chunk-000/file-000.parquet").to_pydict()
    t0 = dict(zip(t["episode_index"], t["videos/observation.images.front/from_timestamp"], strict=True))
    excluded = set(layout["excluded"])
    world, pixels, starts = [], [], []
    for ep, ev in layout["episodes"].items():
        if int(ep) in excluded or ev.get("grasp_frame") is None:
            continue
        for key, frame in (("grasp_xyz", 0), ("release_xyz", ev["release_frame"] + AFTER_RELEASE)):
            hsv = read_frame(dataset, t0[int(ep)] + frame / 30)
            c = cap_pixel(hsv)
            if c is not None:
                world.append(ev[key][:2])
                pixels.append(c)
                if frame == 0:
                    starts.append(hsv)
    fits = {}
    for name, flip in (("as_assumed", False), ("pan_flipped", True)):
        params, res = fit_similarity(world, pixels, flip)
        fits[name] = {
            "scale_px_per_m": float(np.hypot(*params[:2])),
            "residual_mean_px": float(res.mean()),
            "residual_max_px": float(res.max()),
            "base_px": [float(params[2]), float(-params[3])],
        }
    params, _ = fit_similarity(world, pixels)
    k = fits["as_assumed"]["scale_px_per_m"]
    plates = np.array([plate_circle(h) for h in starts])
    cx, cy, r_out, r_in = plates.mean(0)
    cap_d = np.mean([2 * np.sqrt(orange_mask(h).sum() / np.pi) for h in starts])
    return {
        "points": len(world),
        "fits": fits,
        "plate_center_m": to_world(params, cx, cy).round(3).tolist(),
        "plate_outer_diameter_m": round(2 * r_out / k, 3),
        "plate_rim_inner_diameter_m": round(2 * r_in / k, 3),
        "cap_diameter_m": round(cap_d / k, 3),
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--layout", type=Path, required=True, help="JSON from policy.sim.demo_layout")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    res = check(args.dataset, json.loads(args.layout.read_text()))
    for name, f in res["fits"].items():
        print(
            f"{name:12s} scale {f['scale_px_per_m']:.0f} px/m, residual mean {f['residual_mean_px']:.1f} px, "
            f"max {f['residual_max_px']:.1f} px, base at pixel {np.round(f['base_px']).tolist()}"
        )
    print(
        f"plate center {res['plate_center_m']} m, outer diameter {res['plate_outer_diameter_m']} m, "
        f"rim inner diameter {res['plate_rim_inner_diameter_m']} m; cap diameter {res['cap_diameter_m']} m"
    )
    print(f"{res['points']} cap detections; estimates, not measurements")
    if args.out:
        args.out.write_text(json.dumps(res, indent=2) + "\n")


if __name__ == "__main__":
    main()
