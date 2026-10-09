"""Action-chunk execution: replan every `replan` ticks; optionally model inference latency.

With latency d ticks, the chunk computed from the observation at tick t only arrives at tick
t + d. Until then the previous chunk keeps playing (or the arm holds still at the very start),
and the new chunk starts at index d, skipping the steps that are already in the past
(the simple form of real-time chunking in [Black2025-RTC]).
"""

import math

import numpy as np

CONTROL_DT_MS = 1000 / 30


def latency_ticks(latency_ms: float) -> int:
    return math.ceil(latency_ms / CONTROL_DT_MS - 1e-9) if latency_ms > 0 else 0


class ChunkExecutor:
    def __init__(self, backend, replan: int = 4, latency_ms: float = 0.0):
        self.backend, self.replan, self.delay = backend, replan, latency_ticks(latency_ms)
        self.reset()

    def reset(self) -> None:
        self.backend.reset()
        self.queue: np.ndarray | None = None
        self.idx = 0
        self.since = 0
        self.pending: tuple[int, np.ndarray] | None = None
        self.infer_ms: list[float] = []

    def step(self, obs: dict, t: int) -> np.ndarray:
        if self.pending is not None and t >= self.pending[0]:
            self.queue, self.idx, self.since, self.pending = self.pending[1], self.delay, 0, None
        if self.pending is None and (self.queue is None or self.since >= self.replan):
            chunk = self.backend.infer(obs)
            self.infer_ms.append(self.backend.stats()["latency_ms"])
            if self.delay == 0:
                self.queue, self.idx, self.since = chunk, 0, 0
            else:
                self.pending = (t + self.delay, chunk)
        if self.queue is None:
            return obs["state"].copy()  # first chunk still in flight: hold position
        a = self.queue[min(self.idx, len(self.queue) - 1)]
        self.idx += 1
        self.since += 1
        return a
