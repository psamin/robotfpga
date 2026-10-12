import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from armlab.calib.camera import (  # noqa: E402
    TAG_DICT,
    calibrate_intrinsics,
    camera_pose_from_tag,
    mujoco_camera,
    project_mujoco,
    tag_frame,
)

W, H = 640, 480
K = np.array([[600.0, 0, 320], [0, 600.0, 240], [0, 0, 1]])


def look_at(cam_pos, target):
    """OpenCV world->camera rotation for a camera at cam_pos looking at target, world z roughly up."""
    z = np.asarray(target, float) - cam_pos
    z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    return np.stack([x, np.cross(z, x), z])  # rows: camera axes in world


def render_plane(texture, tex_to_world, r_cw, cam_pos, background=160):
    """Image of a planar texture (pixels -> world z=0 via 3x3 tex_to_world) seen by a pinhole camera."""
    t = -r_cw @ cam_pos
    world_to_img = K @ np.column_stack([r_cw[:, 0], r_cw[:, 1], t])
    return cv2.warpPerspective(
        texture, world_to_img @ tex_to_world, (W, H), flags=cv2.INTER_LINEAR, borderValue=background
    )


def checkerboard(cols, rows, px):
    sq = (np.indices(((rows + 1) * px, (cols + 1) * px)) // px).sum(0) % 2
    board = np.pad((sq * 255).astype(np.uint8), px, constant_values=255)
    return board


def test_intrinsics_recovered_from_synthetic_checkerboards():
    cols, rows, square, px = 9, 6, 0.025, 40
    tex = checkerboard(cols, rows, px)
    scale = square / px  # m per texture pixel; texture origin = board's outer white corner
    rng, images = np.random.default_rng(0), []
    for _ in range(14):
        target = np.array([0.125, 0.0875, 0]) + rng.uniform(-0.02, 0.02, 3) * [1, 1, 0]
        cam = target + np.array([*rng.uniform(-0.15, 0.15, 2), rng.uniform(0.3, 0.45)])
        r = look_at(cam, target)
        tex_to_world = np.array([[scale, 0, -scale * px], [0, scale, -scale * px], [0, 0, 1]])
        images.append(render_plane(tex, tex_to_world, r, cam))
    res = calibrate_intrinsics(images, (cols, rows), square)
    k = np.array(res["K"])
    assert res["images"] >= 10 and res["rms_px"] < 0.5
    assert abs(k[0, 0] - 600) < 6 and abs(k[1, 1] - 600) < 6 and np.abs(k[:2, 2] - [320, 240]).max() < 4


def tag_image(tag_pos, yaw, size, cam, target):
    """A 36h11 tag (with a white quiet zone) lying face-up on a gray table."""
    marker = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(TAG_DICT), 3, 200)
    tex = np.pad(marker, 50, constant_values=255)
    s = size / 200  # m per texture pixel; the 200 px marker is `size` wide
    half = tex.shape[0] * s / 2
    # texture (col, row) -> tag frame (x right, y toward texture row 0) -> world
    tex_to_tag = np.array([[s, 0, -half + s / 2], [0, -s, half - s / 2], [0, 0, 1]])
    world_tag = tag_frame(tag_pos, yaw)
    tag_to_world = np.array(
        [[*world_tag[0, :2], world_tag[0, 3]], [*world_tag[1, :2], world_tag[1, 3]], [0, 0, 1]]
    )
    return render_plane(tex, tag_to_world @ tex_to_tag, look_at(cam, target), cam)


@pytest.mark.parametrize("yaw_deg", [0, 35])
def test_tag_pose_recovers_camera_in_world(yaw_deg):
    cam, target, tag_pos = np.array([0.5, 0.05, 0.55]), np.array([0.15, 0, 0]), [0.18, -0.03, 0.0]
    img = tag_image(tag_pos, np.radians(yaw_deg), 0.08, cam, target)
    pose = camera_pose_from_tag(img, K, np.zeros(5), 0.08, tag_pos, np.radians(yaw_deg))
    assert pose["tag_id"] == 3
    assert np.linalg.norm(pose["pos"] - cam) < 0.003  # 3 mm for an 8 cm tag at ~0.65 m
    r_true = look_at(cam, target).T  # camera -> world
    angle = np.degrees(np.arccos(np.clip((np.trace(r_true.T @ pose["rot_cv"]) - 1) / 2, -1, 1)))
    assert angle < 0.5


def test_mujoco_camera_projects_like_opencv():
    cam, target = np.array([0.5, 0.05, 0.55]), np.array([0.15, 0, 0])
    r_cw = look_at(cam, target)
    mj = mujoco_camera(r_cw.T, cam, K, (W, H), "front")
    rng = np.random.default_rng(2)
    for p in rng.uniform([0.0, -0.2, 0.0], [0.3, 0.2, 0.1], (20, 3)):
        cv_px = cv2.projectPoints(p[None], cv2.Rodrigues(r_cw)[0], -r_cw @ cam, K, None)[0].ravel()
        assert np.allclose(project_mujoco(p, mj["pos"], mj["quat"], mj["fovy"], (W, H)), cv_px, atol=0.05)
    assert mj["mjcf"].startswith('<camera name="front" pos="0.5 0.05 0.55"')


def test_cli_intrinsics_then_pose(tmp_path, capsys):
    import json

    from armlab.calib.camera import main

    cols, rows, square, px = 9, 6, 0.025, 40
    tex, scale, rng = checkerboard(cols, rows, px), square / px, np.random.default_rng(3)
    for i in range(12):
        target = np.array([0.125, 0.0875, 0])
        cam = target + np.array([*rng.uniform(-0.15, 0.15, 2), rng.uniform(0.3, 0.45)])
        tex_to_world = np.array([[scale, 0, -scale * px], [0, scale, -scale * px], [0, 0, 1]])
        cv2.imwrite(
            str(tmp_path / f"board{i}.png"), render_plane(tex, tex_to_world, look_at(cam, target), cam)
        )
    boards = sorted(map(str, tmp_path.glob("board*.png")))
    main(
        [
            "intrinsics",
            "--images",
            *boards,
            "--board",
            "9x6",
            "--square",
            "0.025",
            "--out",
            str(tmp_path / "k.json"),
        ]
    )
    cam, target = np.array([0.5, 0.05, 0.55]), np.array([0.15, 0, 0])
    cv2.imwrite(str(tmp_path / "tag.png"), tag_image([0.18, -0.03, 0.0], 0.0, 0.08, cam, target))
    main(
        [
            "pose",
            "--image",
            str(tmp_path / "tag.png"),
            "--intrinsics",
            str(tmp_path / "k.json"),
            "--tag-size",
            "0.08",
            "--tag-pos",
            "0.18",
            "-0.03",
            "0",
            "--out",
            str(tmp_path / "front.json"),
        ]
    )
    out = json.loads((tmp_path / "front.json").read_text())
    assert np.linalg.norm(np.array(out["pos"]) - cam) < 0.01 and out["mjcf"] in capsys.readouterr().out
