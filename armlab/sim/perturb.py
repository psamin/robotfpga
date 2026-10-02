"""Stage C disturbances: move the target cube, or swap the instruction, mid-episode."""

import numpy as np

from armlab.sim.env import CUBE_X, CUBE_Y, INSTRUCTIONS, BlockSortEnv
from armlab.sim.expert import Expert, Phase

KINDS = ("none", "cube_move", "instr_swap")


class Perturber:
    def __init__(self, kind: str, rng: np.random.Generator, window: tuple[int, int] = (10, 45)):
        assert kind in KINDS, kind
        self.kind, self.rng = kind, rng
        self.at = int(rng.integers(*window))
        self.done = kind == "none"
        self.event: dict | None = None

    def maybe_apply(self, env: BlockSortEnv, expert: Expert | None, t: int) -> bool:
        """Applies the disturbance once, at the first tick >= self.at that allows it."""
        if self.done or t < self.at:
            return False
        if self.kind == "cube_move":
            # only while reaching, so the cube is never yanked out of the gripper
            color = INSTRUCTIONS[env.instr][0]
            if expert is not None and expert.phase not in (Phase.APPROACH, Phase.DESCEND):
                return False
            if expert is None and (env.touching_robot(color) or env.cube_pos(color)[2] > 0.02):
                return False
            other = env.cube_pos("blue" if color == "red" else "red")
            while True:
                xy = self.rng.uniform([CUBE_X[0], CUBE_Y[0]], [CUBE_X[1], CUBE_Y[1]])
                if (
                    np.linalg.norm(xy - other[:2]) > 0.06
                    and np.linalg.norm(xy - env.cube_pos(color)[:2]) > 0.05
                ):
                    break
            adr = env.m.jnt_qposadr[env.m.joint(f"cube_{color}").id]
            env.d.qpos[adr : adr + 3] = [xy[0], xy[1], 0.0125]
            env.d.qvel[env.m.jnt_dofadr[env.m.joint(f"cube_{color}").id] :][:6] = 0
            self.event = {"t": t, "kind": "cube_move", "to": xy.tolist()}
        else:
            new = int(self.rng.choice([i for i in range(4) if i != env.instr]))
            self.event = {"t": t, "kind": "instr_swap", "from": env.instr, "to": new}
            env.instr = new
        self.done = True
        return True
