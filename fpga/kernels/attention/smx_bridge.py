"""The softmax kernel's arithmetic (fpga/kernels/softmax/smx_emu.py, read-only use) as the P producer of the
attention path: scores are already scaled by 1/8 (Q's scale), so the kernel's c = log2(e)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_SMX = Path(__file__).resolve().parents[1] / "softmax"
if str(_SMX) not in sys.path:
    sys.path.insert(0, str(_SMX))
import smx_emu  # noqa: E402

C_LOG2E = float(torch.tensor(smx_emu.LOG2E, dtype=torch.float32))


def softmax_q(scores: torch.Tensor, valid: torch.Tensor | None):
    """scores (..., L) FP32, valid (..., L) bool or None -> (q FP32 integer-valued in [0, 32767], scale (..., 1))."""
    q, s = smx_emu.kernel(scores, C_LOG2E, valid)
    return q.to(torch.float32), s.to(torch.float32)
