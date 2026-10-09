import time

import numpy as np
import torch

from armlab.policy.tiny import TinyPolicy


class TorchBackend:
    """Float TinyPolicy on CPU (or any torch device)."""

    name = "torch"

    def __init__(self, ckpt: str, device: str = "cpu", threads: int = 1):
        torch.set_num_threads(threads)
        self.model = TinyPolicy().to(device).eval()
        self.model.load_state_dict(torch.load(ckpt, map_location=device)["model"])
        self._ms = 0.0

    def reset(self) -> None:
        pass

    def infer(self, obs: dict) -> np.ndarray:
        t0 = time.perf_counter()
        out = self.model.act(obs["image"], obs["state"], obs["instr"])
        self._ms = 1e3 * (time.perf_counter() - t0)
        return out

    def stats(self) -> dict:
        return {"latency_ms": self._ms}
