"""SmolVLA's vision attention (the SmolVLM/SigLIP tower) on the FPGA linear engine, with exact FP32 semantics.

Per camera image SigLIP runs 12 layers x 12 heads of attention over 1,024 patch tokens (head_dim 64, no
mask, scale 1/8): 7.9 s of a chunk on the KR260's ARM cores (torch's CPU flash kernel, 24 calls). Here the
two matrix products of every head run on the engine (fpga/hls/float_linear.h: Y = X @ W^T, X FP32, W BF16,
IEEE FP32 products, FP32 sums) and the softmax stays on the ARM as one replaceable function
(FpgaAttention.softmax; a later mode moves it to the FPGA as well).

Exactness. The engine's weight side is BF16, so the FP32 operand that plays W (the keys, then V^T) is split
into three BF16 parts that add up to it exactly:

    W = hi + mid + lo,   hi = bf16(W),  mid = bf16(W - hi),  lo = bf16(W - hi - mid)

(round to nearest even each time, as torch's .to(torch.bfloat16)). W - hi is a multiple of ulp(W) with at
most 16 significant bits, so mid takes its top 8 bits and lo the remaining <= 8 bits exactly, and
(hi + mid) + lo evaluated in FP32 gives W bit for bit; this holds for every normal FP32 value with
|W| >= 2^-103 (below that a part would be an FP32 denormal, which the engine flushes to zero anyway).
test_fpga_attention.py proves the identity on random FP32 and on the model's own K/V. The engine then forms
every FP32 x part product with the IEEE FP32 multiply (bit for bit the product of the FP32 multiplier) and
accumulates in FP32, so Q K^T = Q hi^T + Q mid^T + Q lo^T and P V = P (V^T)_hi^T + ... are FP32 sums of
exact-operand FP32 products: the same rounding class as torch's own FP32 matmul, which rounds the same
products and sums them in another order. Nothing is quantized.

Layouts (how the three parts of a head go through the engine; all give the same FP32 error class, they
differ in calls and data movement). T = 1,024 tokens, D = 64, s = scale:
  Q K^T (X = scaled queries, W = keys):
    "k"  one call, parts along K: X' = [sQ | sQ | sQ] (T x 3D), W' = [K_hi | K_mid | K_lo] (T x 3D). The
         engine adds the parts in its FP32 accumulator and writes the T x T scores directly (4 MB out, no
         ARM adds). Default.
    "m"  one call, parts along M: W = [K_hi; K_mid; K_lo] (3T x D); 12 MB out, the ARM adds three slices.
    "separate"  three calls (W = one part each, M = T, K = D), summed on the ARM.
  P V (X = probabilities T x T, W = V^T parts, D x T):
    "m"  one call, W = [V^T_hi; V^T_mid; V^T_lo] (3D x T): 768 KB out, the ARM adds three T x D slices.
         Default (the "k" form would triple the 4 MB of probabilities going in).
    "k"  one call, X' = [P | P | P] (T x 3T), W' = [V^T_hi | V^T_mid | V^T_lo] (D x 3T).
    "separate"  three calls (M = D, K = T), summed on the ARM.
The engine computes every output lane on its own, so a stacked call gives bit for bit the numbers of the
separate calls (with "k" the parts are added inside the accumulator instead of on the ARM). The 1/8 scale
is a power of two, so it is applied to Q before the product (exact: it only changes the exponent); a scale
that is not a power of two is applied to the scores, where torch applies it.

    from fpga.board.fpga_attention import attention_on_engine
    fa = attention_on_engine(policy, eng)   # after fpga_linears(policy, eng) and policy.float()
    chunk = policy.predict_action_chunk(batch)
    fa.calls, fa.macs, fa.t                 # engine calls and MACs added, time split

Per head the keys (then the values) are split, packed with pack_float.pack and written into a per-thread
scratch CMA buffer (384 KB each; the mock engines take the packed array itself): Engine.upload would keep
every per-call weight for the life of the engine, so it is not used. Only the vision tower is touched;
the LM and the expert keep their attention.
"""

from __future__ import annotations

import importlib.util
import math
import threading
import time
import types
from pathlib import Path

import numpy as np

try:
    from fpga.board.runtime import SlabWeights, chunk_layout
except ImportError:  # imported by file path, without the repo root on sys.path
    _here = Path(__file__).resolve().parent
    _spec = importlib.util.spec_from_file_location("fpga_runtime", _here / "runtime.py")
    _rt = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_rt)
    SlabWeights, chunk_layout = _rt.SlabWeights, _rt.chunk_layout

PARTS = 3  # BF16 parts per FP32 operand (exact for |x| >= 2^-103, see the module doc)
SCRATCH_MIN = 1 << 20  # smallest per-thread weight scratch buffer on the board (bytes)
MODES = ("qk_pv", "qk", "pv", "torch")  # which products run on the engine ("torch": none, same code path)
LAYOUTS = ("k", "m", "separate")
# the attention modules of the vision tower (transformers' SmolVLM/Idefics3/SigLIP towers share this shape)
VISION_ATTENTION_CLASSES = ("SmolVLMVisionAttention", "Idefics3VisionAttention", "SiglipAttention")


def split_bf16_parts(x: np.ndarray, pf, parts: int = PARTS) -> list[np.ndarray]:
    """FP32 array -> `parts` BF16 bit arrays (hi, mid, lo) whose values add up to x exactly (module doc).

    Every subtraction is exact in FP32 (x - bf16(x) is a rounding error, Sterbenz), so each part holds the
    top bits of what the previous ones left."""
    r = np.ascontiguousarray(x, np.float32)
    out = []
    for _ in range(parts):
        b = pf.to_bf16_bits(r)
        out.append(b)
        r = r - pf.bf16_bits_to_f32(b)
    return out


def split_bf16_parts_torch(x, parts: int = PARTS) -> list[np.ndarray]:
    """split_bf16_parts with torch's own conversions (its .to(torch.bfloat16) is the same round to nearest
    even, bit for bit, test_fpga_attention.py `identity`; multithreaded, ~10x numpy on this path).
    x: FP32 tensor -> BF16 bit arrays as numpy uint16 views of int16 tensors."""
    import torch

    out, r = [], x.detach()
    for _ in range(parts):
        b = r.to(torch.bfloat16)
        out.append(b.view(torch.int16).numpy().view(np.uint16))
        r = r - b.to(torch.float32)
    return out


def parts_sum(parts: list[np.ndarray], pf) -> np.ndarray:
    """(hi + mid) + lo in FP32, the order the ARM-side sums use."""
    s = pf.bf16_bits_to_f32(parts[0])
    for b in parts[1:]:
        s = s + pf.bf16_bits_to_f32(b)
    return s


def torch_softmax(scores):
    """The ARM softmax: FP32 over the keys (last dim) of one head's T x Tk scores. Replace
    FpgaAttention.softmax with any callable of the same shape to move it elsewhere."""
    import torch

    return torch.softmax(scores, dim=-1)


class FpgaAttention:
    """softmax(scale Q K^T + mask) V for every head, with the products on the engine. Counters: calls,
    macs (N * K * Mp per call, what the engine computes, padding included), attn_calls, t (seconds by
    stage), exact_total / exact_fail (K/V values whose three-part split was checked / was not exact,
    with check_exact)."""

    def __init__(self, eng, mode: str = "qk_pv", softmax=None, qk_layout: str = "k", pv_layout: str = "m"):
        if mode not in MODES:
            raise ValueError(f"mode {mode!r}: one of {MODES}")
        if qk_layout not in LAYOUTS or pv_layout not in LAYOUTS:
            raise ValueError(f"layouts {qk_layout!r}/{pv_layout!r}: one of {LAYOUTS}")
        self.eng, self.mode = eng, mode
        self.qk_layout, self.pv_layout = qk_layout, pv_layout
        self.softmax = softmax or torch_softmax
        self.pf = eng.pf
        # the host-library mocks and the socket engine take numpy; the board and its register-level
        # emulation (Engine(mock="ip")) take torch tensors and weights in CMA buffers
        self.numpy_io = getattr(eng, "mock", None) in ("shim", "layout", "remote")
        self.calls, self.macs, self.attn_calls = 0, 0, 0
        self.t = dict.fromkeys(("split_pack", "engine", "softmax", "other"), 0.0)
        self.check_exact, self.exact_total, self.exact_fail = False, 0, 0
        self._local = threading.local()  # per-thread scratch weights and time bookkeeping
        self._lock = threading.Lock()  # the counters above, shared by the image threads
        self.patched = []  # modules whose forward was replaced, for restore()
        self.attn_implementation = None  # what the tower used before (config._attn_implementation)

    # ---- engine plumbing -------------------------------------------------------------------------
    def _weights(self, packed: np.ndarray):
        """Packed per-call weights where Engine.gemm can read them: the array itself for a mock, else this
        thread's scratch CMA buffer (allocated through the engine, grown on demand, never uploaded)."""
        eng = self.eng
        if eng.mock in ("shim", "layout"):
            return packed
        if eng.mock == "remote" or not hasattr(eng, "_alloc"):
            raise NotImplementedError("per-call weights need an in-process Engine (xrt, pynq or mock='ip')")
        nbytes = packed.size * 2
        buf = getattr(self._local, "wbuf", None)
        if buf is None or buf.array.size < nbytes:
            if buf is not None:
                buf.free()
            buf = self._local.wbuf = eng._alloc(max(nbytes, SCRATCH_MIN))
        buf.array[:nbytes].view(np.uint16)[:] = packed
        buf.flush(0, nbytes)
        return SlabWeights(buf, 0, nbytes)

    def _gemm(self, w_bits: np.ndarray, x):
        """y[N][Mp] = x @ W^T for one engine call: W as BF16 bits (M x K), x FP32 (N x K) torch tensor."""
        import torch

        t0 = time.perf_counter()
        packed, mp = self.pf.pack(w_bits)
        wbuf = self._weights(packed)
        n, k = x.shape
        t1 = time.perf_counter()
        layout = chunk_layout(mp, k)
        if self.numpy_io:
            y = torch.from_numpy(np.ascontiguousarray(self.eng.gemm(wbuf, layout, mp, x.numpy())))
        else:
            y = self.eng.gemm(wbuf, layout, mp, x)
        dt_pack, dt_eng = t1 - t0, time.perf_counter() - t1
        self._local.inner = getattr(self._local, "inner", 0.0) + dt_pack + dt_eng
        with self._lock:
            self.t["split_pack"] += dt_pack
            self.t["engine"] += dt_eng
            self.calls += 1
            self.macs += n * k * mp
        return y

    # ---- the two products ------------------------------------------------------------------------
    def _scores(self, sq, kparts, tk: int):
        """sq: (T, D) FP32 queries (already scaled when the scale is a power of two); kparts: the three
        (Tk, D) BF16-bit parts of this head's keys. Returns the (T, Tk) FP32 scores."""
        import torch

        if self.qk_layout == "k":
            x = torch.cat([sq] * len(kparts), dim=1)  # (T, 3D)
            return self._gemm(np.concatenate(kparts, axis=1), x)[:, :tk]  # Mp = Tk padded to 32
        if self.qk_layout == "m":
            y = self._gemm(np.concatenate(kparts, axis=0), sq)  # (T, 3Tk padded)
            s = y[:, :tk] + y[:, tk : 2 * tk]
            return s + y[:, 2 * tk : 3 * tk]
        ys = [self._gemm(p, sq)[:, :tk] for p in kparts]
        return (ys[0] + ys[1]) + ys[2]

    def _pv(self, p, vtparts, d: int):
        """p: (T, Tk) FP32 probabilities; vtparts: the three (D, Tk) BF16-bit parts of this head's V^T.
        Returns the (T, D) FP32 output."""
        import torch

        if self.pv_layout == "m":
            y = self._gemm(np.concatenate(vtparts, axis=0), p)  # (T, 3D padded)
            o = y[:, :d] + y[:, d : 2 * d]
            return o + y[:, 2 * d : 3 * d]
        if self.pv_layout == "k":
            x = torch.cat([p] * len(vtparts), dim=1)  # (T, 3Tk)
            return self._gemm(np.concatenate(vtparts, axis=1), x)[:, :d]
        ys = [self._gemm(w, p)[:, :d] for w in vtparts]
        return (ys[0] + ys[1]) + ys[2]

    def _check(self, x: np.ndarray, parts) -> None:
        bad = int(np.count_nonzero(x != parts_sum(parts, self.pf)))
        with self._lock:
            self.exact_total += x.size
            self.exact_fail += bad

    # ---- one attention call (all heads of a layer) -----------------------------------------------
    def attend(self, q, k, v, attention_mask=None, scaling: float = 1.0):
        """q, k, v: (B, T, H, D) FP32 -> (B, T, H, D): softmax(scaling Q K^T + mask) V per head.

        attention_mask: None, or (B, 1 | H, T, Tk) bool (True attends) or additive float, as torch's sdpa."""
        import torch

        b_, t_, h_, d_ = q.shape
        tk = k.shape[1]
        on_qk = self.mode in ("qk_pv", "qk")
        on_pv = self.mode in ("qk_pv", "pv")
        pow2 = scaling > 0 and math.frexp(scaling)[0] == 0.5  # exact to apply on Q
        out = torch.empty_like(q)
        with self._lock:
            self.attn_calls += 1
        for b in range(b_):
            t0 = time.perf_counter()
            kparts = vtparts = None
            if on_qk:  # (H, Tk, D): this head's keys are rows of W
                kb = k[b].permute(1, 0, 2).contiguous()
                kparts = split_bf16_parts_torch(kb)
                if self.check_exact:
                    self._check(kb.numpy(), kparts)
            if on_pv:  # (H, D, Tk): this head's V^T is W
                vb = v[b].permute(1, 2, 0).contiguous()
                vtparts = split_bf16_parts_torch(vb)
                if self.check_exact:
                    self._check(vb.numpy(), vtparts)
            dt_split = time.perf_counter() - t0
            with self._lock:
                self.t["split_pack"] += dt_split
            qb = q[b] * scaling if pow2 else q[b]  # (T, H, D)
            for h in range(h_):
                t0 = time.perf_counter()
                self._local.inner = 0.0  # what this thread's _gemm calls book inside this head
                sq = qb[:, h, :].contiguous()
                if on_qk:
                    s = self._scores(sq, [p[h] for p in kparts], tk)
                else:
                    s = torch.matmul(sq, k[b, :, h, :].transpose(0, 1))
                if not pow2:
                    s = s * scaling
                if attention_mask is not None:
                    m = attention_mask[b, h if attention_mask.shape[1] == h_ else 0]
                    s = s.masked_fill(~m, float("-inf")) if m.dtype == torch.bool else s + m
                t1 = time.perf_counter()
                p = self.softmax(s)
                t2 = time.perf_counter()
                if on_pv:
                    o = self._pv(p, [w[h] for w in vtparts], d_)
                else:
                    o = torch.matmul(p, v[b, :, h, :])
                out[b, :, h, :] = o
                t3 = time.perf_counter()
                with self._lock:
                    self.t["softmax"] += t2 - t1
                    self.t["other"] += (t1 - t0) + (t3 - t2) - self._local.inner  # copies, cats, adds, mask
        return out

    def restore(self) -> None:
        """Put the original forward back on every patched attention module."""
        for mod in self.patched:
            mod.__dict__.pop("forward", None)
        self.patched = []


def _vision_forward(fa: FpgaAttention):
    """SmolVLMVisionAttention.forward with the attention interface replaced by `fa` (same projections,
    the head split kept as (B, T, H, D) so no transposes are needed). Returns (output, None) like sdpa."""

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        if self.training and getattr(self, "dropout", 0.0):
            raise RuntimeError("fpga_attention runs eval attention only (dropout would be skipped)")
        b, t = hidden_states.shape[:2]
        h, d = self.num_heads, self.head_dim
        q = self.q_proj(hidden_states).view(b, t, h, d)
        k = self.k_proj(hidden_states).view(b, t, h, d)
        v = self.v_proj(hidden_states).view(b, t, h, d)
        o = fa.attend(q.detach(), k.detach(), v.detach(), attention_mask, self.scale)
        return self.out_proj(o.reshape(b, t, h * d)), None

    return forward


def vision_attention_modules(policy, vision_model=None) -> list:
    """The attention modules of the policy's vision tower (SmolVLA: vlm.model.vision_model, 12 layers)."""
    vm = vision_model
    if vm is None:
        vm = policy.model.vlm_with_expert.get_vlm_model().vision_model
    return [m for _, m in vm.named_modules() if type(m).__name__ in VISION_ATTENTION_CLASSES]


def attention_on_engine(
    policy,
    eng,
    mode: str = "qk_pv",
    softmax=None,
    qk_layout: str = "k",
    pv_layout: str = "m",
    vision_model=None,
) -> FpgaAttention:
    """Route the vision tower's attention through `eng` (the LM and the expert keep theirs); returns the
    FpgaAttention with the counters and the replaceable `softmax`. Call after fpga_linears(policy, eng)
    and policy.float(); fa.restore() undoes it."""
    fa = FpgaAttention(eng, mode, softmax, qk_layout, pv_layout)
    mods = vision_attention_modules(policy, vision_model)
    if not mods:
        raise RuntimeError(f"no vision attention module of {VISION_ATTENTION_CLASSES} found")
    fa.attn_implementation = getattr(getattr(mods[0], "config", None), "_attn_implementation", None)
    fwd = _vision_forward(fa)
    for mod in mods:
        mod.forward = types.MethodType(fwd, mod)
        fa.patched.append(mod)
    return fa
