"""SmolVLA attention through the int16 linear engine: exact torch emulation of the arithmetic, the engine model,
the DDR layouts of the engine's DRIVER CONTRACT (fpga/hls/int16_linear.h) and the attn_pack descriptors
(fpga/kernels/attention, README.md).

Arithmetic (bit for bit what attn_pack.h + the int16 engine do):

  quantize one row x (FP32) to int16 with one FP32 scale: the project's shared quantizer
  (fpga/kernels/quantize/quant_int16.h == build/brief/precision_emulation.py quant_sym):
      s = fl32(max(max|x|, 1e-30f) / 32767)        one correctly rounded FP32 division per row (1.0 if max|x| == 0)
      q = clamp(rne(fl32(x / s)), -32767, 32767)   correctly rounded FP32 division, round half to even
  Q is quantized with s_q = s * 2^-3 (the 1/sqrt(64) of the attention, exact: a power of two), so the
  engine's Q K^T output already is the scaled score.
  pow2 variant: s = 2^e, e = ceil(log2(max|x| / 32767)): q = rne(x * 2^-e) (what the RoPE kernel's
  ROPE_QM = 0 mode emits for Q and K).

  engine call (int16_linear.h): acc = sum_k Xq[n][k] Wq[m][k] exact (int48 in the DSP; here FP64 of integers
  below 2^53), y = ftz(ftz(fl32(acc) * s_row[m]) * s_tok[n]) (two FP32 multiplies, denormals flushed).

Attention of one layer (B = 1, H query heads, Hkv KV heads, G = H / Hkv, head dim D = 64), per KV head j
(one head iteration of a head-batched engine call):
  Q K^T : W = K_j (Tk x 64, keys are the engine's output channels, s_row = per-key scale, M = pad32(Tk)),
          X = the G query heads of KV head j stacked along the token axis (row t = g * Tq + i, N = G * Tq),
          s_tok = per (head, query) scale * 2^-3  ->  scores Y[M/32][Np][32] FP32
  mask  : in the softmax kernel (n_valid / causal / bitmap, README.md "Masks"); keys no query may attend are
          also zeroed in K and V by attn_pack (krank == 0x7FFF), so nothing of them can leak.
  softmax: FP32 (variant 'smx': the softmax kernel's own arithmetic, fpga/kernels/softmax/smx_emu.py), then
          P quantized per row with the shared quantizer (int16 >= 0, s_tok per row).
  P V   : W = V_j^T (64 x Tk: channels are the output channels, s_row = per-channel scale over the keys),
          X = P (N x Tk, K = Tk exactly)  ->  O_j Y[2][Np][32] FP32, row t = g * Tq + i
  o_proj: attn_pack mode OX gathers row i of every head h (O_{h / G}, row (h % G) Tq + i) into o_proj's X
          (Tq x H * 64) with one scale per query over all H * 64 values.
"""

from __future__ import annotations

import math

import numpy as np
import torch

QMAX = 32767
QMAX_F = torch.tensor(float(QMAX), dtype=torch.float32)
TINY = float(np.float32(1e-30))
HEAD_DIM = 64
LANES = 32  # engine output channels per group (4 hp ports x 8 int16)
PORTS = 4
FL = 8  # int16 per 128-bit word
TB = 32  # engine tokens per block: N is padded to a multiple of TB
KRANK_NEVER = 0x7FFF  # key rank of a key no query may attend
FLT_MIN = float(np.finfo(np.float32).tiny)
NONE = 0xFFFFFFFF


def pad_to(n: int, m: int) -> int:
    return -(-n // m) * m


# ----------------------------------------------------------------------------------------------------
# quantizers
def quant(x: torch.Tensor, dim: int, scale_mul: float = 1.0, pow2: bool = False):
    """FP32 x -> (q, s): integer-valued FP32 q in [-32767, 32767] and FP32 scale s (keepdim along `dim`),
    x ~= q * s / scale_mul, exactly as attn_pack.h (quant_int16.h) computes them (module doc)."""
    x = x.to(torch.float32)
    amax = x.abs().amax(dim, keepdim=True)
    if pow2:
        r = amax.clamp_min(TINY) / QMAX_F
        m, e = torch.frexp(r)  # r = m * 2^e, m in [0.5, 1)
        e = torch.where(m == 0.5, e - 1, e)
        s = torch.ldexp(torch.ones_like(amax), e)
        q = torch.round(torch.ldexp(x, -e)).clamp_(-QMAX, QMAX)
    else:
        s = amax.clamp_min(TINY) / QMAX_F
        s = torch.where(amax == 0, torch.ones_like(s), s)  # all-zero row: s = 1.0 (pack_float / quant_int16.h)
        q = torch.round(x / s).clamp_(-QMAX, QMAX)
    if scale_mul != 1.0:
        assert math.frexp(scale_mul)[0] == 0.5, "scale_mul must be a power of two (exact)"
        s = s * scale_mul
    return q, s


def ftz(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x.abs() < FLT_MIN, torch.zeros_like(x), x)


def engine(xq: torch.Tensor, s_tok: torch.Tensor, wq: torch.Tensor, s_row: torch.Tensor) -> torch.Tensor:
    """The int16 engine: xq (..., N, K), s_tok (..., N, 1), wq (..., M, K), s_row (..., 1, M) -> (..., N, M) FP32."""
    acc = torch.matmul(xq.double(), wq.double().transpose(-1, -2))  # exact: |acc| <= K 2^30 < 2^53
    return ftz(ftz(acc.float() * s_row) * s_tok)


def bf16(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.bfloat16).to(torch.float32)


# ----------------------------------------------------------------------------------------------------
# masks
def rank_descriptor(mask: torch.Tensor):
    """mask (B, Tq, Tk) bool (True = may attend) -> (krank (B, Tk), qrank (B, Tq), ok): mask ==
    (krank[None, :] <= qrank[:, None]) iff the rows' allowed key sets are nested, which every SmolVLA mask is.
    krank == KRANK_NEVER marks keys no query may attend (attn_pack zeroes them)."""
    m = mask.bool()
    qrank = m.sum(-1)
    cand = torch.where(m, qrank[:, :, None].expand(m.shape), torch.full_like(qrank[:, :, None], KRANK_NEVER))
    krank = cand.amin(1)
    ok = bool(torch.equal(krank[:, None, :] <= qrank[:, :, None], m))
    return krank, qrank, ok


# ----------------------------------------------------------------------------------------------------
# attention
def attention(q, k, v, scale: float, mask=None, qk: str = "int16", pv: str = "int16", pow2: bool = False,
              qk_pow2: bool = False, p_mode: str = "quant"):
    """q (B, H, Tq, D), k / v (B, Hkv, Tk, D) FP32 (k post-RoPE, not repeated over the G query heads);
    mask None or (B, Tq, Tk) bool. qk / pv: 'int16' (engine arithmetic), 'bf16' (BF16 operands, FP32 sums),
    'fp32'. pow2: power-of-two scales for every operand; qk_pow2: for Q and K only (the RoPE kernel's output).
    p_mode: 'quant' (FP32 softmax, shared quantizer) or 'smx' (the softmax kernel's q and 1/sum scale).
    Returns (B, H, Tq, D) FP32."""
    B, H, Tq, D = q.shape
    Hkv, Tk = k.shape[1], k.shape[2]
    G = H // Hkv
    q, k, v = q.float(), k.float(), v.float()
    if mask is not None:  # attn_pack zeroes the keys of rank KRANK_NEVER
        kval = mask.any(1)[:, None, :, None].float()
        k = k * kval
        v = v * kval
    if qk == "int16":
        qq, sq = quant(q, -1, scale, pow2 or qk_pow2)  # per (head, query)
        kq, sk = quant(k, -1, 1.0, pow2 or qk_pow2)  # per (kv head, key)
        x = qq.reshape(B, Hkv, G * Tq, D)  # row t = g * Tq + i
        sx = sq.reshape(B, Hkv, G * Tq, 1)
        scores = engine(x, sx, kq, sk.transpose(-1, -2)).reshape(B, H, Tq, Tk)
    else:
        kk = k.repeat_interleave(G, dim=1)
        if qk == "bf16":
            scores = (bf16(q) @ bf16(kk).transpose(-1, -2)) * scale
        else:
            scores = (q @ kk.transpose(-1, -2)) * scale
    if p_mode == "smx" and pv == "int16":
        import smx_bridge

        pq, sp = smx_bridge.softmax_q(scores, None if mask is None else mask[:, None].expand(B, H, Tq, Tk))
    else:
        if mask is not None:
            scores = torch.where(mask[:, None, :, :], scores, torch.finfo(torch.float32).min)
        p = torch.softmax(scores, dim=-1)
        if pv == "int16":
            pq, sp = quant(p, -1, 1.0, pow2)  # per (head, query) row over the keys
    if pv == "int16":
        vq, sv = quant(v, -2, 1.0, pow2)  # per (kv head, channel) over the keys: (B, Hkv, 1, D)
        x = pq.reshape(B, Hkv, G * Tq, Tk)
        sx = sp.reshape(B, Hkv, G * Tq, 1)
        out = engine(x, sx, vq.transpose(-1, -2), sv).reshape(B, H, Tq, D)
    else:
        vv = v.repeat_interleave(G, dim=1)
        out = (bf16(p) @ bf16(vv)) if pv == "bf16" else (p @ vv)
    return out


# ----------------------------------------------------------------------------------------------------
# DDR layouts of the int16 engine (int16_linear.h DRIVER CONTRACT), numpy little-endian, as uint32 arrays
def xmem_image(s_row: np.ndarray, s_tok: np.ndarray, xq: np.ndarray) -> np.ndarray:
    """One head iteration's xmem: s_row (M FP32), s_tok (padded to Np = pad32(N), pad value given by the caller
    in s_tok's length or 1.0), Xq as [Np/32 blocks][K][32 tokens] int16. uint32 words."""
    m = len(s_row)
    n, kk = xq.shape
    npad = pad_to(n, TB)
    assert m % LANES == 0
    st = np.ones(npad, np.float32)
    st[: len(s_tok)] = s_tok
    xi = np.zeros((npad, kk), np.int16)
    xi[:n] = xq
    xi = np.ascontiguousarray(xi.reshape(npad // TB, TB, kk).transpose(0, 2, 1))  # [b][k][a]
    return np.concatenate([np.asarray(s_row, np.float32).view(np.uint32), st.view(np.uint32),
                           xi.reshape(-1).view(np.uint32)])


def xmem_words(m: int, n: int, kk: int) -> int:
    npad = pad_to(n, TB)
    return m // 4 + npad // 4 + npad * kk // 8


def w_ports(w: np.ndarray) -> np.ndarray:
    """int16 W (M, K), M a multiple of 32 -> the four port arrays [p][M/32][K] words of 8 lanes
    W[32 g + 8 p + l][k] (pack_float.pack of the int16 bits), as one uint32 array [p][g][k][4]."""
    m, kk = w.shape
    assert m % LANES == 0
    g = np.ascontiguousarray(w.astype(np.int16)).reshape(m // LANES, PORTS, FL, kk)  # [g][p][l][k]
    return np.ascontiguousarray(g.transpose(1, 0, 3, 2)).view(np.uint32).reshape(-1)  # [p][g][k][l]


def y_tiled(y: np.ndarray) -> np.ndarray:
    """FP32 Y (Np, M) -> the engine's output Y[M/32][Np][32] as uint32 (rows must already be padded)."""
    n, m = y.shape
    assert m % LANES == 0 and n % TB == 0
    return np.ascontiguousarray(np.asarray(y, np.float32).reshape(n, m // LANES, LANES).transpose(1, 0, 2)
                                ).reshape(-1).view(np.uint32)


# ----------------------------------------------------------------------------------------------------
# attn_pack descriptors (README.md "attn_pack descriptor"): 96 uint32 = 24 io words
MODE = dict(QX=0, KW=1, VW=2, OX=3)


def f32bits(x: float) -> int:
    return int(np.float32(x).view(np.uint32))


def region(off: int, row_stride: int, half_stride: int, segtab):
    """A source of FP32 rows: segment h's row i, FP32 word w (0..15) at
    off + segtab[h] + i * row_stride + (w // 8) * half_stride + w % 8 (words)."""
    return dict(off=off, row_stride=row_stride, half_stride=half_stride, segtab=list(segtab))


def region_rowmajor(off: int, nseg: int):
    """Row-major FP32 rows of nseg * 64 values (the KV cache, a host buffer)."""
    return region(off, 16 * nseg, 8, [16 * s for s in range(nseg)])


def region_ytiled(off: int, npad: int, nseg: int):
    """An engine output Y[2 nseg][Np][32]: segment s = 64 channels = groups 2s, 2s + 1."""
    return region(off, 8, npad * 8, [2 * s * npad * 8 for s in range(nseg)])


def desc(mode: str, n_img: int, n_rows: int, n_out: int, rows_per_seg: int, segs_per_img: int, nseg_row: int,
         scale_mul: float, krank_off: int, n_split: int, regions, out_off: int, out_img: int, out_q: int,
         sc_off: int, sc_img: int) -> np.ndarray:
    u = np.zeros(96, np.uint32)
    u[0:10] = [MODE[mode], n_img, n_rows, n_out, rows_per_seg, segs_per_img, nseg_row, f32bits(scale_mul),
               krank_off, n_split]
    regs = list(regions) + [regions[-1]] * (2 - len(regions))
    for r, reg in enumerate(regs):
        u[10 + 3 * r: 13 + 3 * r] = [reg["off"], reg["row_stride"], reg["half_stride"]]
        u[32 + 32 * r: 32 + 32 * r + len(reg["segtab"])] = reg["segtab"]
    u[16:21] = [out_off, out_img, out_q, sc_off, sc_img]
    return u
