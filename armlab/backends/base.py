"""Every policy sits behind this interface, so one eval scores them all (build spec §5.2)."""

from typing import Protocol

import numpy as np


class PolicyBackend(Protocol):
    name: str

    def reset(self) -> None: ...

    def infer(self, obs: dict) -> np.ndarray:  # [8, 6] float32, normalized joint targets
        ...

    def stats(self) -> dict:  # {"latency_ms": last call, ...}
        ...
