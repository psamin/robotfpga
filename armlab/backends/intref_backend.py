import time

import numpy as np

from armlab.policy.qat import load_intref


class IntrefBackend:
    """Bit-exact int8 reference (ref/intref.py) on an exported weights.bin + manifest.json.

    This is what the FPGA must reproduce byte for byte, so its closed-loop success is the
    success the FPGA will get (mock-fpga uses it too, spec §5.6)."""

    name = "intref"

    def __init__(self, export_dir: str):
        self.ref = load_intref()
        self.model = self.ref.Model.load(export_dir)
        self._ms = 0.0

    def reset(self) -> None:
        pass

    def infer(self, obs: dict) -> np.ndarray:
        t0 = time.perf_counter()
        y, _ = self.ref.forward(self.ref.pack_obs(obs), self.model)
        out = self.ref.unpack_actions(y)
        self._ms = 1e3 * (time.perf_counter() - t0)
        return out

    def stats(self) -> dict:
        return {"latency_ms": self._ms}
