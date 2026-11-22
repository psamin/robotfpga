"""SmolVLA attention under the float_top integration contract (fpga/kernels/CONTRACT.md 2, 5): an exact torch
emulation of the arithmetic the hardware runs, used by the full-policy check (check_policy.py) and checked bit
for bit against the record-level chain (make_vectors.py, then tb_attn.cpp on the units' HLS mode code).

Per attention site, per KV head j (one GEMM item; G = H / Hkv query heads per KV head, head dim 64):

  Q      row unit (QUANT_X, or ROPE with F_QOUT): per (head, query) the contract quantizer with p4 = -3,
         so the stored scale is s / 8 (exact) and the Q.K^T GEMM output already is the scaled score.
         Head-major Q[h][TqP][64], TqP = pad4(Tq); rows [Tq, TqP) zero with scale 1.0. The G heads of KV
         head j are rows t = g * TqP + i of item j.
  K      row unit (QUANT_W; ROPE F_KW writes the same tiling): per (KV head, key) the contract quantizer,
         the KW weight tiling, Mk = pad32(keys) rows; expert self-attention: prefix keys in rows
         [0, pad32(45)), suffix keys from row pad32(45) on. Pad rows zero with scale 1.0.
  Q.K^T  GEMM: acc exact, y = fmul(fmul(fl(acc), s_row = s_k), s_tok = s_q / 8), FP32 cores, denormals
         flushed with the sign kept (contract.gemm_out).
  P      SOFTMAX unit (fpga/kernels/softmax/smx_emu.kernel, c = fl32(p0 log2 e), p0 = 1): int16 q in
         [0, 32767] with scale RN(1 / sum q), a fully masked row q = 0 with scale 1.0. Masks: int32 rank
         vectors, key k visible to row n iff 0 <= E[k] <= D[n] (CONTRACT.md 5, step 4 table; checked
         against SmolVLA's own mask on every call).
  V      row unit (QUANT_WT): V^T per (KV head, channel) over the part's keys, the VW tiling, keys
         [N, pad32(N)) zero. The expert self-attention has two parts (prefix V^T written by the LM, suffix
         V^T every step), each with its own channel scales.
  P.V    GEMM, K = pad32(keys of the part), X = P's columns of the part; the expert's suffix part adds onto
         the prefix part's output (F_ACC: y = fadd(y_suffix, O)). Output rows t = g * TqP + i go to
         O[i][(G j + g) 64] through the C row groups (R = TqP, P1 = 256 B): O is [Tq][H * 64], the o_proj
         input order.

The prefix is compacted to its valid tokens (the contract's list has 45 prefix tokens: 2 x 16 image tokens,
12 language tokens, the state); padding queries get a zero output (no valid query or key depends on them).

The pre-contract design (`attention`, the attn_pack era: one V^T scale over prefix + suffix, P from a torch
FP32 softmax or the old softmax arithmetic) stays for comparison; its ftz keeps the sign as the FP32 cores do.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

_K = Path(__file__).resolve().parents[1]
for _p in (_K, _K / "softmax"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import contract as CT  # noqa: E402
import smx_emu  # noqa: E402

QMAX = 32767
QMAX_F = torch.tensor(float(QMAX), dtype=torch.float32)
TINY = float(np.float32(1e-30))
HEAD_DIM = 64
Q_SHIFT = -3  # p4 of Q: the attention's 1 / sqrt(64)
C_SMX = smx_emu.c_from_mult(1.0)  # the SOFTMAX record's p0 = 1.0: the scores already carry the 1/8


def pad_to(n: int, m: int) -> int:
    return -(-n // m) * m


# ----------------------------------------------------------------------------------------------------
# contract arithmetic in torch (= contract.quant_rows / contract.gemm_out, bit for bit)
def ftz(x: torch.Tensor) -> torch.Tensor:
    """The Xilinx FP32 cores: a denormal input or result is flushed to zero, keeping its sign."""
    e = (x.view(torch.int32) >> 23) & 0xFF
    return torch.where(e == 0, torch.copysign(torch.zeros_like(x), x), x)


def fmul_hw(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return ftz(ftz(a) * ftz(b))


def fadd_hw(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return ftz(ftz(a) + ftz(b))


def quant_c(x: torch.Tensor, dim: int = -1, shift: int = 0):
    """The contract quantizer along `dim`: (q integer-valued FP32, stored scale FP32 keepdim).
    s = fl32(max(amax, 1e-30) / 32767), q = clamp(rne(fl32(x / s)), +-32767);
    stored = s * 2^shift, 1.0 for an all-zero row."""
    x = x.to(torch.float32)
    amax = x.abs().amax(dim, keepdim=True)
    zero = amax == 0
    s = torch.where(zero, torch.ones_like(amax), amax.clamp_min(TINY) / QMAX_F)
    q = torch.round(x / s).clamp_(-QMAX, QMAX)
    stored = torch.where(zero, torch.ones_like(s), torch.ldexp(s, torch.tensor(shift, dtype=torch.int32)))
    return q, stored


def gemm_out(xq: torch.Tensor, wq: torch.Tensor, s_row: torch.Tensor, s_tok: torch.Tensor, addend=None):
    """GEMM epilogue: xq (..., N, K), wq (..., M, K) integer-valued; s_row (..., 1, M), s_tok (..., N, 1)."""
    acc = torch.matmul(xq.double(), wq.double().transpose(-1, -2))  # exact: |acc| <= K 2^30 < 2^53
    y = fmul_hw(fmul_hw(acc.float(), s_row), s_tok)
    return y if addend is None else fadd_hw(y, addend)


# ----------------------------------------------------------------------------------------------------
# rank masks (CONTRACT.md 5, step 4)
def contract_ranks(site: str, tq: int, tqp: int, nkeys, mk: int, key_rows):
    """(D (tqp,) query-row ranks, E (mk,) key ranks) int32 of the contract table. nkeys: valid keys per part,
    key_rows: blob row of each part's first key. Pad rows / keys: -1."""
    D = torch.full((tqp,), -1, dtype=torch.int32)
    E = torch.full((mk,), -1, dtype=torch.int32)
    if site == "prefix":  # image / language 0, the state (last prefix token) 1
        D[:tq] = 0
        D[tq - 1] = 1
        E[: nkeys[0]] = 0
        E[nkeys[0] - 1] = 1
    elif site == "expert_self":  # query i: 2 + i; prefix keys 0 (state 1); suffix key j: 2 + j
        D[:tq] = 2 + torch.arange(tq, dtype=torch.int32)
        E[: nkeys[0]] = 0
        E[nkeys[0] - 1] = 1
        E[key_rows[1] : key_rows[1] + nkeys[1]] = 2 + torch.arange(nkeys[1], dtype=torch.int32)
    elif site == "expert_cross":  # every valid query sees every valid prefix key
        D[:tq] = 0
        E[: nkeys[0]] = 0
    else:
        raise ValueError(site)
    return D, E


def site_geometry(site: str, tk: int, n_suffix: int = 0):
    """Key parts of a site: [(first key, keys, blob row)], Mk. Expert self: prefix keys + n_suffix suffix
    keys."""
    if site == "expert_self":
        npre = tk - n_suffix
        lp = pad_to(npre, 32)
        return [(0, npre, 0), (npre, n_suffix, lp)], lp + pad_to(n_suffix, 32)
    return [(0, tk, 0)], pad_to(tk, 32)


def attention_contract(q, k, v, mask, site: str, n_suffix: int = 0, record: dict | None = None):
    """q (B, H, Tq, D), k / v (B, Hkv, Tk, D) FP32 (k post-RoPE), mask None (vision) or (B, Tq, Tk) bool.
    Returns (B, H, Tq, D) FP32: the contract's attention output for the valid queries, 0 for padding queries.
    record: if given, filled with the intermediates of batch 0 (make_vectors.py)."""
    B, H, Tq, D = q.shape
    out = torch.zeros(B, H, Tq, D, dtype=torch.float32)
    for b in range(B):
        if mask is None:
            qv = torch.ones(Tq, dtype=torch.bool)
            kv = torch.ones(k.shape[2], dtype=torch.bool)
        else:
            m = mask[b].bool()
            qv, kv = m.any(1), m.any(0)
        qc, kc, vc = q[b][:, qv].float(), k[b][:, kv].float(), v[b][:, kv].float()
        o = _attention_one(
            qc,
            kc,
            vc,
            None if mask is None else mask[b].bool()[qv][:, kv],
            site,
            n_suffix,
            record if b == 0 else None,
        )
        out[b][:, qv] = o
    return out


def _attention_one(q, k, v, mc, site, n_suffix, rec):
    H, tq, D = q.shape
    hkv, tk = k.shape[0], k.shape[1]
    G = H // hkv
    tqp = pad_to(tq, 4)
    parts, mk = site_geometry(site, tk, n_suffix)
    # ---- Q: head-major, rows [tq, tqp) zero (stored scale 1.0)
    qp = torch.zeros(H, tqp, D)
    qp[:, :tq] = q
    qq, sq = quant_c(qp, -1, Q_SHIFT)  # (H, tqp, D), (H, tqp, 1)
    x = qq.reshape(hkv, G * tqp, D)  # item j, row t = g * tqp + i
    sx = sq.reshape(hkv, G * tqp, 1)
    # ---- K: blob rows, pad rows zero (scale 1.0)
    kb = torch.zeros(hkv, mk, D)
    for k0, nk, r0 in parts:
        kb[:, r0 : r0 + nk] = k[:, k0 : k0 + nk]
    kq, sk = quant_c(kb, -1)  # (hkv, mk, D), (hkv, mk, 1)
    scores = gemm_out(x, kq, sk.transpose(-1, -2), sx)  # (hkv, G tqp, mk)
    # ---- softmax unit with rank masks
    if site == "vision":
        valid = None
        assert mc is None and mk == tk
    else:
        Dq, Ek = contract_ranks(site, tq, tqp, [p[1] for p in parts], mk, [p[2] for p in parts])
        rv = smx_emu.rank_valid(Ek, Dq)  # (tqp, mk)
        # the contract's ranks reproduce SmolVLA's mask on every valid (query, key)
        cols = torch.cat([torch.arange(r0, r0 + nk) for _, nk, r0 in parts])
        assert torch.equal(rv[:tq][:, cols], mc), f"{site}: contract ranks != SmolVLA mask"
        valid = rv.repeat(G, 1)  # row t = g * tqp + i has query i's rank
    pq, sp = smx_emu.kernel(scores, C_SMX, valid)  # q int64 (hkv, G tqp, mk), scale (hkv, G tqp, 1)
    pq = pq.to(torch.float32)
    # ---- P.V per key part (V^T per channel over the part's keys), the suffix part accumulates
    y = None
    vts = []
    for k0, nk, r0 in parts:
        npv = pad_to(nk, 32)
        vt = v[:, k0 : k0 + nk].transpose(-1, -2)  # (hkv, D, nk)
        vq, sv = quant_c(vt, -1)  # (hkv, D, nk), (hkv, D, 1)
        vqp = torch.zeros(hkv, D, npv)
        vqp[:, :, :nk] = vq
        vts.append((vqp, sv))
        y = gemm_out(pq[:, :, r0 : r0 + npv], vqp, sv.transpose(-1, -2), sp, y)
    o = y.reshape(hkv, G, tqp, D)[:, :, :tq].reshape(H, tq, D)
    if rec is not None:
        rec.update(
            dict(
                tq=tq,
                tqp=tqp,
                mk=mk,
                parts=parts,
                G=G,
                hkv=hkv,
                qq=qq,
                sq=sq,
                kq=kq,
                sk=sk,
                scores=scores,
                pq=pq,
                sp=sp,
                vts=vts,
                y=y,
                out=o,
                ranks=None if site == "vision" else (Dq, Ek),
            )
        )
    return o


# ----------------------------------------------------------------------------------------------------
# the pre-contract design (attn_pack era), kept for comparison
def quant(x: torch.Tensor, dim: int, scale_mul: float = 1.0, pow2: bool = False):
    """FP32 x -> (q, s), the shared quantizer (or power-of-two scales); s * scale_mul (a power of two)."""
    x = x.to(torch.float32)
    amax = x.abs().amax(dim, keepdim=True)
    if pow2:
        r = amax.clamp_min(TINY) / QMAX_F
        mnt, e = torch.frexp(r)
        e = torch.where(mnt == 0.5, e - 1, e)
        s = torch.ldexp(torch.ones_like(amax), e)
        q = torch.round(torch.ldexp(x, -e)).clamp_(-QMAX, QMAX)
    else:
        s = amax.clamp_min(TINY) / QMAX_F
        s = torch.where(amax == 0, torch.ones_like(s), s)
        q = torch.round(x / s).clamp_(-QMAX, QMAX)
    if scale_mul != 1.0:
        assert math.frexp(scale_mul)[0] == 0.5, "scale_mul must be a power of two (exact)"
        s = s * scale_mul
    return q, s


def engine(xq, s_tok, wq, s_row):
    """The int16 engine: xq (..., N, K), s_tok (..., N, 1), wq (..., M, K), s_row (..., 1, M)
    -> (..., N, M)."""
    acc = torch.matmul(xq.double(), wq.double().transpose(-1, -2))
    return fmul_hw(fmul_hw(acc.float(), s_row), s_tok)


def bf16(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.bfloat16).to(torch.float32)


def attention(
    q,
    k,
    v,
    scale: float,
    mask=None,
    qk: str = "int16",
    pv: str = "int16",
    pow2: bool = False,
    p_mode: str = "quant",
):
    """Pre-contract design: q (B, H, Tq, D), k / v (B, Hkv, Tk, D) FP32, mask None or (B, Tq, Tk) bool.
    qk / pv: 'int16', 'bf16' or 'fp32'; p_mode 'quant' (FP32 softmax + quantizer) or 'smx' (softmax unit)."""
    B, H, Tq, D = q.shape
    Hkv, Tk = k.shape[1], k.shape[2]
    G = H // Hkv
    q, k, v = q.float(), k.float(), v.float()
    if mask is not None:
        kval = mask.any(1)[:, None, :, None].float()
        k = k * kval
        v = v * kval
    if qk == "int16":
        qq, sq = quant(q, -1, scale, pow2)
        kq, sk = quant(k, -1, 1.0, pow2)
        scores = engine(
            qq.reshape(B, Hkv, G * Tq, D), sq.reshape(B, Hkv, G * Tq, 1), kq, sk.transpose(-1, -2)
        ).reshape(B, H, Tq, Tk)
    else:
        kk = k.repeat_interleave(G, dim=1)
        scores = (
            (bf16(q) @ bf16(kk).transpose(-1, -2)) if qk == "bf16" else (q @ kk.transpose(-1, -2))
        ) * scale
    if p_mode == "smx" and pv == "int16":
        valid = None if mask is None else mask[:, None].expand(B, H, Tq, Tk)
        pq, sp = smx_emu.kernel(scores, C_SMX, valid)
        pq = pq.to(torch.float32)
    else:
        if mask is not None:
            scores = torch.where(mask[:, None, :, :], scores, torch.finfo(torch.float32).min)
        p = torch.softmax(scores, dim=-1)
        if pv == "int16":
            pq, sp = quant(p, -1, 1.0, pow2)
    if pv == "int16":
        vq, sv = quant(v, -2, 1.0, pow2)
        out = engine(
            pq.reshape(B, Hkv, G * Tq, Tk), sp.reshape(B, Hkv, G * Tq, 1), vq.transpose(-1, -2), sv
        ).reshape(B, H, Tq, D)
    else:
        vv = v.repeat_interleave(G, dim=1)
        out = (bf16(p) @ bf16(vv)) if pv == "bf16" else (p @ vv)
    return out


__all__ = [
    "attention_contract",
    "attention",
    "quant_c",
    "gemm_out",
    "ftz",
    "contract_ranks",
    "site_geometry",
    "CT",
]
