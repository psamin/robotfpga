"""Camera calibration: checkerboard intrinsics, and an AprilTag on the table -> a MuJoCo camera.

intrinsics  Photos of a printed checkerboard from many angles -> K, distortion, RMS error (px).
pose        One photo of an AprilTag (36h11) lying flat on the table at a measured position and
            yaw -> the camera's pose in the sim world frame (robot base at the origin, table top
            z = 0), converted to MuJoCo's convention, printed as a <camera> MJCF line. Print the
            tag as large as fits (about 15 cm on A4); --tag-size is the black square's edge.

OpenCV cameras look along +z with +y down; MuJoCo cameras look along -z with +y up, so the two
frames differ by a 180 deg rotation about x. MuJoCo models a centered pinhole with one vertical
fovy: the printed line ignores distortion, fx != fy and the principal-point offset, which are
reported so they can be judged.

    uv run --extra calib python -m armlab.calib.camera intrinsics --images calib/front/*.png \\
        --board 9x6 --square 0.025 --out calib/front_intrinsics.json
    uv run --extra calib python -m armlab.calib.camera pose --image calib/front_tag.png \\
        --intrinsics calib/front_intrinsics.json --tag-size 0.08 --tag-pos 0.15 0.0 0.0 --name front
"""

import argparse
import json
from pathlib import Path

import cv2
import mujoco
import numpy as np

CV_TO_MJ = np.diag([1.0, -1.0, -1.0])  # MuJoCo camera axes expressed in OpenCV camera axes
TAG_DICT = cv2.aruco.DICT_APRILTAG_36h11


def calibrate_intrinsics(images: list[np.ndarray], board: tuple[int, int], square: float) -> dict:
    """board: inner corners (cols, rows); square: edge length in m."""
    obj = np.zeros((board[0] * board[1], 3), np.float32)
    obj[:, :2] = np.mgrid[: board[0], : board[1]].T.reshape(-1, 2) * square
    obj_pts, img_pts, size = [], [], None
    for img in images:
        gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]
        found, corners = cv2.findChessboardCorners(gray, board)
        if found:
            crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
            img_pts.append(cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), crit))
            obj_pts.append(obj)
    if len(img_pts) < 5:
        raise ValueError(f"checkerboard found in {len(img_pts)} images; need at least 5")
    rms, k, dist, _, _ = cv2.calibrateCamera(obj_pts, img_pts, size, None, None)
    return {
        "K": k.tolist(),
        "dist": dist.ravel().tolist(),
        "rms_px": rms,
        "size": list(size),
        "images": len(img_pts),
    }


def tag_frame(pos, yaw: float) -> np.ndarray:
    """4x4 world pose of a tag lying face-up: its x axis at `yaw` (rad) from world x, z up."""
    c, s = np.cos(yaw), np.sin(yaw)
    t = np.eye(4)
    t[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    t[:3, 3] = pos
    return t


def camera_pose_from_tag(image, k, dist, tag_size: float, tag_pos, tag_yaw: float = 0.0, tag_id=None) -> dict:
    """World pose of an OpenCV camera that sees the tag: rotation (camera -> world) and position."""
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    params = cv2.aruco.DetectorParameters()
    # Without refinement, synthetic tests put an 8 cm tag's camera 9-17 mm off; AprilTag
    # refinement brings it to ~1 mm (16 cm tag: ~0.3 mm).
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(TAG_DICT), params)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        raise ValueError("no AprilTag 36h11 found")
    i = 0 if tag_id is None else int(np.flatnonzero(ids.ravel() == tag_id)[0])
    h = tag_size / 2
    obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], np.float32)
    ok, rvec, tvec = cv2.solvePnP(
        obj, corners[i], np.asarray(k), np.asarray(dist), flags=cv2.SOLVEPNP_IPPE_SQUARE
    )
    if not ok:
        raise ValueError("solvePnP failed")
    r_cam_tag = cv2.Rodrigues(rvec)[0]
    world_tag = tag_frame(tag_pos, tag_yaw)
    rot = world_tag[:3, :3] @ r_cam_tag.T  # camera axes in world (OpenCV convention)
    pos = world_tag[:3, :3] @ (-r_cam_tag.T @ tvec.ravel()) + world_tag[:3, 3]
    return {"rot_cv": rot, "pos": pos, "tag_id": int(ids.ravel()[i])}


def mujoco_camera(rot_cv, pos, k, size, name: str = "camera") -> dict:
    """MuJoCo <camera> parameters (pos, quat w-x-y-z, fovy deg) for an OpenCV camera pose."""
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, (np.asarray(rot_cv) @ CV_TO_MJ).ravel())
    k, (w, h) = np.asarray(k), size
    fovy = float(np.degrees(2 * np.arctan(h / (2 * k[1, 1]))))
    out = {
        "name": name,
        "pos": np.round(pos, 5).tolist(),
        "quat": np.round(quat, 6).tolist(),
        "fovy": round(fovy, 3),
    }
    out["ignored"] = {
        "fx_over_fy": float(k[0, 0] / k[1, 1]),
        "principal_offset_px": [float(k[0, 2] - w / 2), float(k[1, 2] - h / 2)],
    }
    out["mjcf"] = (
        f'<camera name="{name}" pos="{" ".join(map(str, out["pos"]))}" '
        f'quat="{" ".join(map(str, out["quat"]))}" fovy="{out["fovy"]}"/>'
    )
    return out


def project_mujoco(point, pos, quat, fovy: float, size) -> np.ndarray:
    """Pixel (u, v) of a world point seen by a MuJoCo camera (centered pinhole)."""
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, np.asarray(quat, float))
    p = rot.reshape(3, 3).T @ (np.asarray(point) - np.asarray(pos))  # x right, y up, looks along -z
    w, h = size
    f = h / (2 * np.tan(np.radians(fovy) / 2))
    return np.array([w / 2 + f * p[0] / -p[2], h / 2 - f * p[1] / -p[2]])


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("intrinsics")
    a.add_argument("--images", nargs="+", required=True)
    a.add_argument("--board", required=True, help="inner corners, e.g. 9x6")
    a.add_argument("--square", type=float, required=True, help="square edge (m)")
    a.add_argument("--out", required=True)
    b = sub.add_parser("pose")
    b.add_argument("--image", required=True)
    b.add_argument("--intrinsics", required=True, help="JSON from the intrinsics command")
    b.add_argument("--tag-size", type=float, required=True, help="black-square edge (m)")
    b.add_argument("--tag-pos", type=float, nargs=3, required=True, help="tag center in the sim world (m)")
    b.add_argument("--tag-yaw", type=float, default=0.0, help="tag x axis from world x (deg)")
    b.add_argument("--tag-id", type=int)
    b.add_argument("--name", default="front")
    b.add_argument("--out")
    args = ap.parse_args(argv)
    if args.cmd == "intrinsics":
        cols, rows = map(int, args.board.lower().split("x"))
        res = calibrate_intrinsics([cv2.imread(p) for p in args.images], (cols, rows), args.square)
        Path(args.out).write_text(json.dumps(res, indent=2) + "\n")
        print(f"{res['images']} images, RMS {res['rms_px']:.3f} px -> {args.out}")
        return
    intr = json.loads(Path(args.intrinsics).read_text())
    pose = camera_pose_from_tag(
        cv2.imread(args.image),
        intr["K"],
        intr["dist"],
        args.tag_size,
        args.tag_pos,
        np.radians(args.tag_yaw),
        args.tag_id,
    )
    cam = mujoco_camera(pose["rot_cv"], pose["pos"], intr["K"], intr["size"], args.name)
    print(cam["mjcf"])
    print(
        f"ignored by MuJoCo: fx/fy {cam['ignored']['fx_over_fy']:.4f}, "
        f"principal point offset {np.round(cam['ignored']['principal_offset_px'], 1).tolist()} px"
    )
    if args.out:
        Path(args.out).write_text(json.dumps({**cam, "tag_id": pose["tag_id"]}, indent=2) + "\n")


if __name__ == "__main__":
    main()
