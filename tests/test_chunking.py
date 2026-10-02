import numpy as np

from armlab.control.chunking import ChunkExecutor, latency_ticks


class Counter:
    """Backend whose chunk k is filled with k, so we can see which chunk/index plays."""

    name = "counter"

    def __init__(self):
        self.k = 0

    def reset(self):
        self.k = 0

    def infer(self, obs):
        self.k += 1
        return np.tile(np.arange(8, dtype=np.float32)[:, None], (1, 6)) + 100 * self.k

    def stats(self):
        return {"latency_ms": 0.0}


def obs():
    return {"state": np.zeros(6, np.float32), "image": None, "instr": 0}


def test_replans_every_four_without_latency():
    ex = ChunkExecutor(Counter(), replan=4)
    got = [ex.step(obs(), t)[0] for t in range(9)]
    assert got == [100, 101, 102, 103, 200, 201, 202, 203, 300]


def test_latency_starts_new_chunk_at_offset():
    ex = ChunkExecutor(Counter(), replan=4, latency_ms=70)  # 3 ticks
    assert latency_ticks(70) == 3
    got = [ex.step(obs(), t)[0] for t in range(8)]
    # holds for 3 ticks, then chunk 1 from index 3; replans after 4 more ticks
    assert got[:3] == [0, 0, 0] and got[3:7] == [103, 104, 105, 106]
