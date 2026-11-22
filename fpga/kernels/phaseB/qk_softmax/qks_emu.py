"""Torch emulation of the QK_SOFTMAX command (phase B qk_softmax, contract_qks.h) for one item, in the order the
hardware works: per tile of 64 query rows, the scores of the tile = the GEMM epilogue of Q.K^T
(fmul(fmul(fl(acc), s_row[key]), s_tok[row]), FP32 cores with sign-keeping FTZ), visibility from the int32 rank
vectors (0 <= rank_k[key] <= rank_q[row]), then the SOFTMAX unit's arithmetic (smx_emu.kernel) on the tile's rows.
Rows [N, NP) are zero with scale 1.0 (the store writes them). Rows are independent, so the tile walk gives the same
bits as the (GEMM, SOFTMAX) pair on the whole item; policy_check.py asserts that on every attention call.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_K = Path(__file__).resolve().parents[2]
for _p in (_K, _K / "softmax", _K / "attention"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import attn_int16 as A  # noqa: E402  (gemm_out: the contract epilogue in torch)
import smx_emu  # noqa: E402

TILE = 64  # engine tile: TB x NBX = 32 x 2 query rows


def qk_softmax(xq, s_tok, kq, s_row, rank_q=None, rank_k=None, p0: float = 1.0, np_: int | None = None):
    """One item. xq (N, K) int-valued FP32 / int16, s_tok (N,) FP32, kq (L, K), s_row (L,), rank_q (N,) /
    rank_k (L,) int32 or None (no F_MASK), p0 the score multiplier. Returns (q int64 (NP, L), scale FP32 (NP,))."""
    N, L = xq.shape[0], kq.shape[0]
    NP = np_ or N
    c = smx_emu.c_from_mult(p0)
    q = torch.zeros(NP, L, dtype=torch.int64)
    s = torch.ones(NP, dtype=torch.float32)
    w = kq.to(torch.float32)
    for t0 in range(0, N, TILE):
        t1 = min(N, t0 + TILE)
        sc = A.gemm_out(xq[t0:t1].to(torch.float32), w, s_row.reshape(1, L).float(), s_tok[t0:t1].reshape(-1, 1).float())
        valid = None if rank_q is None else smx_emu.rank_valid(rank_k, rank_q[t0:t1])
        qt, st = smx_emu.kernel(sc, c, valid)
        q[t0:t1] = qt
        s[t0:t1] = st.reshape(-1)
    return q, s
