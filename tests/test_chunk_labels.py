import numpy as np

from armlab.sim.env import BlockSortEnv
from armlab.sim.expert import Expert, expert_chunk


def test_chunk_label_is_the_experts_future_and_has_no_side_effects():
    def run(label: bool):
        env = BlockSortEnv(render_size=96)
        env.reset(seed=5, instr=1)
        ex = Expert(env)
        chunks, acts, traj = [], [], []
        for _ in range(40):
            if label:
                chunks.append(expert_chunk(env, ex))
            q = ex.act()
            acts.append(env.normalize(q))
            env.step_q(q)
            traj.append(env.d.qpos.copy())
        return chunks, np.array(acts), np.array(traj)

    chunks, acts, traj = run(True)
    _, _, traj_plain = run(False)
    np.testing.assert_array_equal(traj, traj_plain)
    for t in range(32):
        np.testing.assert_allclose(chunks[t], acts[t : t + 8], atol=1e-5)
