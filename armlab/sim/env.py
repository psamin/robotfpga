"""BlockSortEnv: SO-101 sorts a red or blue cube into the left or right bin (build spec §2, §5.1)."""

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

SCENE = Path(__file__).resolve().parent / "assets" / "block_sort.xml"
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
INSTRUCTIONS = (("red", "left"), ("red", "right"), ("blue", "left"), ("blue", "right"))
TASKS = tuple(f"put the {c} cube in the {b} bin" for c, b in INSTRUCTIONS)
SUBSTEPS = 10  # 10 x 1/300 s = one 30 Hz control tick
REST_Q = np.array([0.0, -1.74, 1.69, 0.6, 0.0, 0.5])
CUBE_X = (0.17, 0.28)  # cube spawn region (m), well clear of the bins
CUBE_Y = (-0.09, 0.09)
CUBE_MIN_GAP = 0.06
BIN_HALF = 0.045  # inner half-width
CUBE_HALF = 0.0125


@dataclass
class Jitter:
    camera_pos: float = 0.01  # m
    camera_fovy: float = 1.5  # deg
    light: float = 0.15  # diffuse intensity
    color: float = 0.08  # rgba per channel


DEFAULT_JITTER = Jitter()


class BlockSortEnv:
    def __init__(
        self,
        render_size: int = 288,
        obs_size: int = 96,
        jitter: Jitter | None = DEFAULT_JITTER,
        max_steps: int = 600,
    ):
        assert render_size % obs_size == 0, "render size must be an integer multiple of obs size"
        self.m = mujoco.MjModel.from_xml_path(str(SCENE))
        self.d = mujoco.MjData(self.m)
        self.render_size, self.obs_size, self.jitter, self.max_steps = (
            render_size,
            obs_size,
            jitter,
            max_steps,
        )
        self._renderer: mujoco.Renderer | None = None
        self.lo = self.m.actuator_ctrlrange[:, 0].copy()
        self.hi = self.m.actuator_ctrlrange[:, 1].copy()
        self._cam = self.m.camera("overhead").id
        self._light = self.m.light("key").id
        self._defaults = {
            "cam_pos": self.m.cam_pos[self._cam].copy(),
            "cam_fovy": float(self.m.cam_fovy[self._cam]),
            "light": self.m.light_diffuse[self._light].copy(),
            "mat_rgba": self.m.mat_rgba.copy(),
        }
        self._robot_bodies = self._subtree(self.m.body("base").id)
        self.instr = 0
        self.t = 0

    # ---- normalization (spec §5.1: joints in [-1, 1] from the actuator ranges) ----
    def normalize(self, q: np.ndarray) -> np.ndarray:
        return (2 * (q - self.lo) / (self.hi - self.lo) - 1).astype(np.float32)

    def denormalize(self, a: np.ndarray) -> np.ndarray:
        return self.lo + (np.clip(a, -1, 1) + 1) * 0.5 * (self.hi - self.lo)

    # ---- episode ----
    def reset(self, seed: int, instr: int | None = None) -> dict:
        rng = np.random.default_rng(seed)
        self.instr = int(rng.integers(4)) if instr is None else int(instr)
        mujoco.mj_resetData(self.m, self.d)
        self._apply_jitter(rng)
        self.d.qpos[:6] = REST_Q
        self.d.ctrl[:] = REST_Q
        (rx, ry), (bx, by) = self._sample_cubes(rng)
        for name, (x, y) in (("cube_red", (rx, ry)), ("cube_blue", (bx, by))):
            adr = self.m.jnt_qposadr[self.m.joint(name).id]
            yaw = rng.uniform(-np.pi / 4, np.pi / 4)
            self.d.qpos[adr : adr + 7] = [x, y, CUBE_HALF, np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        mujoco.mj_forward(self.m, self.d)
        for _ in range(30):  # let contacts settle
            mujoco.mj_step(self.m, self.d)
        self.t = 0
        return self.obs()

    def step(self, target: np.ndarray) -> tuple[dict, bool, dict]:
        """target: normalized joint targets [6] in [-1, 1]. Advances one 30 Hz tick."""
        self.d.ctrl[:] = self.denormalize(np.asarray(target, dtype=np.float64))
        for _ in range(SUBSTEPS):
            mujoco.mj_step(self.m, self.d)
        self.t += 1
        ok = self.success()
        done = ok or self.t >= self.max_steps
        return self.obs(), done, {"success": ok, "t": self.t}

    def step_q(self, q: np.ndarray) -> tuple[dict, bool, dict]:
        """Same as step, with raw joint angles (radians)."""
        return self.step(self.normalize(np.asarray(q)))

    # ---- observation ----
    def obs(self, render: bool = True) -> dict:
        o = {"state": self.normalize(self.d.qpos[:6]), "instr": self.instr}
        if render:
            full = self.render()
            k = self.render_size // self.obs_size
            o["image_full"] = full
            o["image"] = (
                full.reshape(self.obs_size, k, self.obs_size, k, 3).mean((1, 3)).round().astype(np.uint8)
            )
        return o

    def render(self) -> np.ndarray:
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.m, self.render_size, self.render_size)
        self._renderer.update_scene(self.d, camera="overhead")
        return self._renderer.render()

    # ---- task ----
    def cube_pos(self, color: str) -> np.ndarray:
        return self.d.body(f"cube_{color}").xpos.copy()

    def bin_pos(self, side: str) -> np.ndarray:
        return self.d.body(f"bin_{side}").xpos.copy()

    def in_bin(self, color: str, side: str) -> bool:
        c, b = self.cube_pos(color), self.bin_pos(side)
        return bool(np.all(np.abs(c[:2] - b[:2]) < BIN_HALF) and c[2] < 0.04)

    def touching_robot(self, color: str) -> bool:
        gid = self.m.geom(f"cube_{color}").id
        for i in range(self.d.ncon):
            g1, g2 = self.d.contact[i].geom1, self.d.contact[i].geom2
            other = g2 if g1 == gid else g1 if g2 == gid else None
            if other is not None and self.m.geom_bodyid[other] in self._robot_bodies:
                return True
        return False

    def success(self) -> bool:
        color, side = INSTRUCTIONS[self.instr]
        return self.in_bin(color, side) and not self.touching_robot(color)

    def failure_reason(self) -> str:
        color, side = INSTRUCTIONS[self.instr]
        other = "right" if side == "left" else "left"
        if self.success():
            return "success"
        if self.in_bin(color, other):
            return "wrong_bin"
        if self.touching_robot(color):
            return "still_holding"
        if self.cube_pos(color)[2] > 0.03:
            return "cube_stuck"
        return "not_in_bin"

    # ---- internals ----
    def _subtree(self, root: int) -> set[int]:
        out = {root}
        for b in range(self.m.nbody):
            p = b
            while p > 0:
                if p == root:
                    out.add(b)
                    break
                p = self.m.body_parentid[p]
        return out

    def _sample_cubes(self, rng: np.random.Generator):
        while True:
            a = rng.uniform([CUBE_X[0], CUBE_Y[0]], [CUBE_X[1], CUBE_Y[1]])
            b = rng.uniform([CUBE_X[0], CUBE_Y[0]], [CUBE_X[1], CUBE_Y[1]])
            if np.linalg.norm(a - b) > CUBE_MIN_GAP:
                return a, b

    def _apply_jitter(self, rng: np.random.Generator):
        m, dflt, j = self.m, self._defaults, self.jitter
        m.cam_pos[self._cam] = dflt["cam_pos"]
        m.cam_fovy[self._cam] = dflt["cam_fovy"]
        m.light_diffuse[self._light] = dflt["light"]
        m.mat_rgba[:] = dflt["mat_rgba"]
        if j is None:
            return
        m.cam_pos[self._cam] += rng.uniform(-j.camera_pos, j.camera_pos, 3)
        m.cam_fovy[self._cam] += rng.uniform(-j.camera_fovy, j.camera_fovy)
        m.light_diffuse[self._light] = np.clip(dflt["light"] + rng.uniform(-j.light, j.light), 0, 1)
        for name in ("table", "bin", "red", "blue"):
            mid = m.material(name).id
            m.mat_rgba[mid, :3] = np.clip(dflt["mat_rgba"][mid, :3] + rng.uniform(-j.color, j.color, 3), 0, 1)
