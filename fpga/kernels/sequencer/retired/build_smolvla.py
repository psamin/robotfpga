"""Build float_top command lists for one SmolVLA chunk (checkpoint -12392: 2 x 256-px images).

The whole policy forward after host preprocessing runs as one list:
  vision (both images batched, 512 tokens) -> connector -> language gather + state -> LM prefix
  (16 layers, KV cache) -> cross-attention K/V hoisted once per chunk -> 10 Euler steps x (suffix
  embedding, 16 expert layers, action head, x_t += dt v_t) -> END.
The host writes per-inference inputs into the arena (im2col patches, token ids, state, noise,
positions and the attention block ids), starts float_top once (or once per YIELD), and reads x_t.

    from build_smolvla import Shared, Program
    sh = Shared(policy, plan, qbytes=2)       # weight blobs + constant tables (shared, read-only)
    prog = Program(sh, plan.arena(0))         # one arena: buffers + command list
    prog.cmds                                 # list[seq_format.Cmd]
    prog.host_inputs(images, img_masks, lang_tokens, lang_masks, state, noise)  # {addr: bytes}

Layout conventions (the contract the mode kernels are written against; see README.md):
  activations   FP32 row-major, row pitch in bytes (Slot.pitch), widths padded to 32 where a GEMM
                writes them (720 -> 736)
  quantized X   int16 row-major (qbytes = 2) + one FP32 scale per row in a separate vector
  weight blob   int16 [Mp = ceil32(M)][K] (logical; the engine tiles it per port) then FP32 row
                scales [Mp] then optional FP32 bias [Mp]; Slot b = data (pitch = per-port stride),
                Slot e = scales (bias at e + 4 Mp)
qbytes = 4 builds the same program with FP32 "quantized" tensors (scale 1): exec_ref then computes
the policy in plain FP32, which checks the list itself (addresses, aliasing, order) exactly.
"""

from __future__ import annotations

import math
import struct

import torch
import torch.nn.functional as F

from arena import Buf, CmaPlan, Region, align_up
from seq_format import C, Cmd, Slot, f32bits

NS = 50          # action tokens
NSTEPS = 10
T_IMG = 256      # vision tokens per image (16 x 16 patches of 16 px)
D_V = 768
H_V = 12
HD = 64          # head dim everywhere
D_L = 960
H_L = 15
KV_L = 5
D_E = 720
D_EP = 736       # expert width padded to 32 (GEMM outputs)
I_L = 2560
I_E = 2048
KMAX = 4096      # longest reduction per GEMM command (engine limit; unknown for int16, 4096 assumed)

ST_VISION, ST_PREFIX, ST_LM, ST_XKV, ST_EXPERT, ST_HEAD = 1, 2, 3, 4, 5, 6


def c32(n: int) -> int:
    return (n + 31) // 32 * 32


def a16(n: int) -> int:
    return align_up(n, 16)


def blob_bytes(rows: int, k: int, qbytes: int, bias: bool = False) -> int:
    mp = c32(rows)
    return align_up(mp * k * qbytes + mp * 4 + (mp * 4 if bias else 0), 64)


class Blob:
    """A GEMM weight operand: data (int16 [Mp][K]), FP32 row scales [Mp], optional bias [Mp]."""

    def __init__(self, addr: int, rows: int, k: int, qbytes: int, bias: bool, name: str = ""):
        self.addr, self.M, self.Mp, self.K, self.bias, self.name = addr, rows, c32(rows), k, bias, name
        self.data_bytes = self.Mp * k * qbytes
        self.scales = addr + self.data_bytes
        self.port_stride = self.data_bytes // 4  # bytes per hp port array in the engine's tiling
        self.nbytes = blob_bytes(rows, k, qbytes, bias)


class Shared:
    """Weights, constant tables and model-derived constants; shared by both arenas."""

    def __init__(self, policy, plan: CmaPlan, qbytes: int = 2, kmax: int = KMAX):
        self.p, self.plan, self.qbytes, self.kmax = policy, plan, qbytes, kmax
        self.blobs: dict[int, dict] = {}    # addr -> {"W", "bias", "blob"} (exec_ref quantizes)
        self.tables: dict[int, torch.Tensor] = {}  # addr -> FP32 tensor (gather tables)
        self.const_bytes: list[tuple[int, bytes]] = []
        m = policy.model
        vwe = m.vlm_with_expert
        self.vlm = vwe.get_vlm_model()
        self.vis = self.vlm.vision_model
        self.text = self.vlm.text_model
        self.exp = vwe.lm_expert
        self.cfg = policy.config
        self.n_layers = vwe.num_vlm_layers
        self.every = vwe.self_attn_every_n_layers
        self.gelu = self.vis.encoder.layers[0].mlp.activation_fn
        self.eps_v = self.vis.encoder.layers[0].layer_norm1.eps
        self.eps_l = self.text.layers[0].input_layernorm.variance_epsilon
        self.eps_e = self.exp.layers[0].input_layernorm.variance_epsilon
        assert self.vlm.connector.scale_factor == 4 and self.n_layers == 16 and self.every == 2
        self._weights()
        self._consts()

    # ---- weight region
    def blob(self, name: str, W: torch.Tensor, bias: torch.Tensor | None = None) -> Blob:
        W = W.detach().float()
        M, K = W.shape
        assert K % 8 == 0, (name, K)
        b = Blob(0, M, K, self.qbytes, bias is not None, name)
        buf = self.plan.weights.alloc(name, b.nbytes)
        b = Blob(buf.addr, M, K, self.qbytes, bias is not None, name)
        self.blobs[b.addr] = dict(W=W.contiguous(), bias=None if bias is None else bias.detach().float().contiguous(),
                                  blob=b)
        return b

    def blob_pair(self, name: str, Ws, Bs) -> tuple[Blob, int]:
        """Same-shape blobs back to back: one batched GEMM walks them with a uniform stride."""
        bl = [self.blob(f"{name}.{i}", W, b) for i, (W, b) in enumerate(zip(Ws, Bs))]
        st = bl[1].addr - bl[0].addr
        assert all(bl[i + 1].addr - bl[i].addr == st for i in range(len(bl) - 1))
        return bl[0], st

    def table(self, name: str, t: torch.Tensor) -> int:
        buf = self.plan.weights.alloc(name, t.numel() * 2)  # BF16 in DDR
        self.tables[buf.addr] = t.detach().float()
        return buf.addr

    def const(self, name: str, t: torch.Tensor) -> int:
        t = t.detach().contiguous()
        raw = t.numpy().tobytes()
        buf = self.plan.consts.alloc(name, len(raw))
        self.const_bytes.append((buf.addr, raw))
        return buf.addr

    def _weights(self):
        cat = lambda *ts: torch.cat([t.detach() for t in ts], 0)
        self.v = []
        for i, L in enumerate(self.vis.encoder.layers):
            a = L.self_attn
            self.v.append(dict(
                qkv=self.blob(f"v{i}.qkv", cat(a.q_proj.weight, a.k_proj.weight, a.v_proj.weight),
                              cat(a.q_proj.bias, a.k_proj.bias, a.v_proj.bias)),
                o=self.blob(f"v{i}.o", a.out_proj.weight, a.out_proj.bias),
                fc1=self.blob(f"v{i}.fc1", L.mlp.fc1.weight, L.mlp.fc1.bias),
                fc2=self.blob(f"v{i}.fc2", L.mlp.fc2.weight, L.mlp.fc2.bias)))
        pe = self.vis.embeddings.patch_embedding
        self.patch = self.blob("v.patch", pe.weight.reshape(pe.weight.shape[0], -1))
        Wc = self.vlm.connector.modality_projection.proj.weight
        self.conn = [self.blob(f"conn.k{k0}", Wc[:, k0:k0 + self.kmax]) for k0 in range(0, Wc.shape[1], self.kmax)]
        self.embed_tab = self.table("embed_tokens", self.text.get_input_embeddings().weight)
        m = self.p.model
        self.state_proj = self.blob("state_proj", m.state_proj.weight, m.state_proj.bias)
        self.l = []
        for i, L in enumerate(self.text.layers):
            a, f = L.self_attn, L.mlp
            self.l.append(dict(
                q=self.blob(f"l{i}.q", a.q_proj.weight), kv=self.blob(f"l{i}.kv", cat(a.k_proj.weight, a.v_proj.weight)),
                o=self.blob(f"l{i}.o", a.o_proj.weight),
                gu=self.blob(f"l{i}.gu", cat(f.gate_proj.weight, f.up_proj.weight)),
                down=self.blob(f"l{i}.down", f.down_proj.weight)))
        self.e = []
        for i, L in enumerate(self.exp.layers):
            a, f = L.self_attn, L.mlp
            d = dict(q=self.blob(f"e{i}.q", a.q_proj.weight), o=self.blob(f"e{i}.o", a.o_proj.weight),
                     gu=self.blob(f"e{i}.gu", cat(f.gate_proj.weight, f.up_proj.weight)),
                     down=self.blob(f"e{i}.down", f.down_proj.weight))
            if i % self.every == 0:
                d["kv"] = self.blob(f"e{i}.kv", cat(a.k_proj.weight, a.v_proj.weight))
            else:
                d["xkv"], d["xkv_stride"] = self.blob_pair(f"e{i}.xkv", (a.k_proj.weight, a.v_proj.weight), (None, None))
            self.e.append(d)
        self.a_in = self.blob("action_in_proj", m.action_in_proj.weight, m.action_in_proj.bias)
        Wt = m.action_time_mlp_in.weight
        self.t_in = self.blob("action_time_mlp_in.a", Wt[:, :D_E])  # the time half folds into a per-step bias
        self.t_out = self.blob("action_time_mlp_out", m.action_time_mlp_out.weight, m.action_time_mlp_out.bias)
        self.a_out = self.blob("action_out_proj", m.action_out_proj.weight, m.action_out_proj.bias)

    def _consts(self):
        from lerobot.policies.smolvla.modeling_smolvla import create_sinusoidal_pos_embedding

        m = self.p.model
        with torch.no_grad():
            # patch-embedding bias + position embedding for a full 256-px image (bucketized ids)
            z = torch.zeros(1, 3, 256, 256)
            pm = torch.ones(1, 16, 16, dtype=torch.bool)
            self.pos_table = self.const("v.pos+bias", self.vis.embeddings(pixel_values=z, patch_attention_mask=pm)[0])
            # RoPE: rows = position, [cos(32) | sin(32)], exactly apply_rope's FP32 arithmetic
            d_half = HD // 2
            fe = (2.0 / HD) * torch.arange(d_half, dtype=torch.float32)
            ts = 10_000 ** fe
            rad = torch.arange(256, dtype=torch.float32)[:, None] / ts[None, :]
            self.rope_table = self.const("rope", torch.cat([torch.cos(rad), torch.sin(rad)], 1))
            # per-step bias of action_time_mlp_in: b + W[:, 720:] @ time_emb(t_s)
            Wt, bt = m.action_time_mlp_in.weight, m.action_time_mlp_in.bias
            dt = -1.0 / NSTEPS
            tb = torch.zeros(NSTEPS, D_EP)
            for s in range(NSTEPS):
                time = torch.tensor(1.0 + s * dt, dtype=torch.float32).expand(1)
                temb = create_sinusoidal_pos_embedding(time, D_E, self.cfg.min_period, self.cfg.max_period,
                                                       device=torch.device("cpu")).float()
                tb[s, :D_E] = (bt.double() + Wt[:, D_E:].double() @ temb[0].double()).float()
            self.time_bias = self.const("time_bias", tb)
            self.norms = {}
            for i, L in enumerate(self.vis.encoder.layers):
                self.norms[("v1", i)] = self.const(f"v{i}.ln1", torch.cat([L.layer_norm1.weight, L.layer_norm1.bias]))
                self.norms[("v2", i)] = self.const(f"v{i}.ln2", torch.cat([L.layer_norm2.weight, L.layer_norm2.bias]))
            pl = self.vis.post_layernorm
            self.norms[("vpost", 0)] = self.const("v.post", torch.cat([pl.weight, pl.bias]))
            for i, L in enumerate(self.text.layers):
                self.norms[("l1", i)] = self.const(f"l{i}.in", L.input_layernorm.weight)
                self.norms[("l2", i)] = self.const(f"l{i}.post", L.post_attention_layernorm.weight)
            for i, L in enumerate(self.exp.layers):
                self.norms[("e1", i)] = self.const(f"e{i}.in", L.input_layernorm.weight)
                self.norms[("e2", i)] = self.const(f"e{i}.post", L.post_attention_layernorm.weight)
            self.norms[("efinal", 0)] = self.const("e.norm", self.exp.norm.weight)
            # pixel shuffle as row segments: out row o = 4 segments of 4 consecutive vision tokens
            idx = torch.arange(T_IMG * D_V, dtype=torch.float64).view(1, T_IMG, D_V)
            sh = self.vlm.connector.pixel_shuffle(idx, 4)[0]  # [16][12288]
            seg = 4 * D_V
            self.shuffle = []  # (out row, segment, first source token)
            for o in range(sh.shape[0]):
                for j in range(sh.shape[1] // seg):
                    v = sh[o, j * seg:(j + 1) * seg]
                    s0 = int(v[0])
                    assert s0 % D_V == 0 and torch.equal(v, s0 + torch.arange(seg, dtype=torch.float64))
                    self.shuffle.append((o, j, s0 // D_V))


class Program:
    """One arena's buffers and the chunk's command list."""

    def __init__(self, sh: Shared, arena: Region, n_prefix_lang: int = 48, compact: bool = False,
                 prune_last: bool = True, yield_every_step: bool = False):
        self.sh, self.ar, self.qb = sh, arena, sh.qbytes
        self.NL = n_prefix_lang          # language rows in the prefix (48, or the valid count if compact)
        self.compact = compact
        self.NP = 2 * 16 + self.NL + 1   # prefix rows
        self.NPb = c32(self.NP)
        self.NKV = self.NP + NS          # self-attention keys of the expert (prefix + suffix)
        self.NKb = c32(self.NKV)
        self.prune_last = prune_last
        self.yield_every_step = yield_every_step
        self._persistent()
        self.cmds: list[Cmd] = []
        self._emit()                     # sizing pass (role buffers at address 0)
        self.ar.freeze_roles(dict(self.ar.roles))
        self.cmds = []
        self._emit()
        self.list_buf = None

    # ---- buffers
    def _persistent(self):
        A, qb, NP = self.ar.alloc, self.qb, self.NP
        self.status = A("status", 32)
        self.in_patches = A("in.patches", 2 * T_IMG * D_V * 4)
        self.in_ids = A("in.lang_ids", a16(self.NL * 4))
        self.in_state = A("in.state", 32 * 4)
        self.x_t = A("x_t", NS * 32 * 4)
        self.v_t = A("v_t", NS * 32 * 4)
        self.pos_prefix = A("in.pos_prefix", a16(NP * 4))
        self.qinfo_prefix = A("in.qinfo_prefix", a16(NP * 4))
        self.kinfo_prefix = A("in.kinfo_prefix", self.NPb * 4)
        self.pos_self = A("in.pos_self", NS * 4)
        self.qinfo_self = A("in.qinfo_self", NS * 4)
        self.kinfo_self = A("in.kinfo_self", self.NKb * 4)
        self.pos_cross = A("in.pos_cross", NS * 4)
        self.qinfo_cross = A("in.qinfo_cross", NS * 4)
        self.kinfo_cross = A("in.kinfo_cross", self.NPb * 4)
        self.h_vis = A("h_vis", 2 * T_IMG * D_V * 4)
        self.vis_out = A("vis_out", 2 * T_IMG * D_V * 4)
        self.shuf = A("shuf", 32 * 16 * D_V * 4)
        self.h_p = A("h_prefix", NP * D_L * 4)
        self.kv = [A(f"kv{l}", self.NKV * 2 * KV_L * HD * 4) for l in range(self.sh.n_layers)]
        self.ckb, self.cvb = {}, {}
        for l in range(self.sh.n_layers):
            if l % self.sh.every:
                self.ckb[l] = A(f"xk{l}", KV_L * blob_bytes(self.NP, HD, qb))
                self.cvb[l] = A(f"xv{l}", KV_L * blob_bytes(HD, self.NPb, qb))
        self.h_s = A("h_suffix", NS * D_EP * 4)

    def role(self, name: str, nbytes: int) -> Buf:
        return self.ar.role(name, nbytes)

    # ---- command helpers
    def _tag(self, stage, layer=0, step=0):
        t = (stage << 24) | (layer << 16) | (step << 8) | (self._k & 0xFF)
        self._k += 1
        return t

    def add(self, op, note, stage, layer=0, step=0, **kw) -> Cmd:
        p = kw.pop("p", None)
        c = Cmd(op, tag=self._tag(stage, layer, step), note=note, **kw)
        if p is not None:
            c.p = list(p) + [0] * (C.NPARAMS - len(p))
        self.cmds.append(c)
        return c

    def gemm(self, note, st, x: Slot, xs: Slot, w: Blob, y: Slot, N: int, *, B=1, G=1, w_stride=0, p0=1.0,
             acc=False, addd: Slot | None = None, K: int | None = None, layer=0, step=0):
        K = w.K if K is None else K
        fl = (C.F_BIAS if w.bias else 0) | (C.F_ACC if acc else 0) | (C.F_ADD_D if addd is not None else 0)
        return self.add(C.OP_GEMM, note, st, layer, step, flags=fl, M=w.Mp, K=K, N=N, B=B, G=G, a=x, f=xs,
                        b=Slot(w.addr, w.port_stride, w_stride), e=Slot(w.scales, 0, w_stride), c=y,
                        d=addd if addd is not None else Slot(), p=[f32bits(p0), w.M])

    def quant_x(self, note, st, x: Slot, q: Slot, s: Slot, N, K, B=1, layer=0, step=0):
        return self.add(C.OP_QUANT_X, note, st, layer, step, K=K, N=N, B=B, a=x, c=q, f=s)

    def norm(self, note, st, x: Slot, gamma: int, out: Slot, N, K, eps, *, ln=False, resid: Slot | None = None,
             sum_out: Slot | None = None, scales: Slot | None = None, B=1, layer=0, step=0):
        fl = (C.F_LN if ln else 0) | (C.F_RESID if resid is not None else 0) | (C.F_SUM if sum_out is not None else 0)
        fl |= C.F_QOUT if scales is not None else 0
        return self.add(C.OP_NORM, note, st, layer, step, flags=fl, K=K, N=N, B=B, a=x,
                        b=resid if resid is not None else Slot(), c=out, d=sum_out if sum_out is not None else Slot(),
                        e=Slot(gamma), f=scales if scales is not None else Slot(), p=[f32bits(eps)])

    # ---- the chunk
    def _emit(self):
        self._k = 0
        self.vision()
        self.prefix()
        self.lm()
        self.cross_kv()
        for s in range(NSTEPS):
            self.expert_step(s)
        self.add(C.OP_END, "end", ST_HEAD)

    def vision(self):
        sh, qb, ST = self.sh, self.qb, ST_VISION
        NV = 2 * T_IMG
        xq, xs = self.role("xq", NV * 4 * D_V * qb), self.role("xs", a16(NV * 4))
        a = self.role("a", NV * D_V * 4)
        self.quant_x("patches", ST, Slot(self.in_patches.addr, D_V * 4), Slot(xq.addr, D_V * qb), Slot(xs.addr), NV, D_V)
        self.gemm("patch_embedding", ST, Slot(xq.addr, D_V * qb), Slot(xs.addr), sh.patch, Slot(a.addr, D_V * 4), NV)
        self.norm("embed + ln1[0]", ST, Slot(a.addr, D_V * 4, T_IMG * D_V * 4), sh.norms[("v1", 0)],
                  Slot(xq.addr, D_V * qb, T_IMG * D_V * qb), T_IMG, D_V, sh.eps_v, ln=True,
                  resid=Slot(sh.pos_table, D_V * 4, 0), sum_out=Slot(self.h_vis.addr, D_V * 4, T_IMG * D_V * 4),
                  scales=Slot(xs.addr, 0, T_IMG * 4), B=2)
        for l, W in enumerate(sh.v):
            self.vision_layer(l, W)

    def vision_layer(self, l, W):
        sh, qb, ST = self.sh, self.qb, ST_VISION
        NV, T, D, H = 2 * T_IMG, T_IMG, D_V, H_V
        xq, xs = self.role("xq", NV * 4 * D * qb), self.role("xs", a16(NV * 4))
        qkv = self.role("qkv", NV * 3 * D * 4)
        qh, qs = self.role("qh", H * NV * HD * qb), self.role("qs", H * NV * 4)
        kbs, vbs = blob_bytes(T, HD, qb), blob_bytes(HD, T, qb)
        kb, vb = self.role("kb", 2 * H * kbs), self.role("vb", 2 * H * vbs)
        sc, P, ps = self.role("sc", 2 * H * T * T * 4), self.role("P", 2 * H * T * T * qb), self.role("ps", 2 * H * T * 4)
        attn, a, t = self.role("attn", NV * D * 4), self.role("a", NV * D * 4), self.role("t", NV * 4 * D * 4)
        P3 = 3 * D * 4
        k = dict(layer=l)
        self.gemm("qkv", ST, Slot(xq.addr, D * qb), Slot(xs.addr), W["qkv"], Slot(qkv.addr, P3), NV, **k)
        # Q per (head, token) row of 64, laid out [head][512 tokens][64]
        self.quant_x("q", ST, Slot(qkv.addr, P3, HD * 4), Slot(qh.addr, HD * qb, NV * HD * qb),
                     Slot(qs.addr, 0, NV * 4), NV, HD, B=H, **k)
        for img in range(2):  # K / V^T blobs of item (head, image) at index 2 head + image
            src = qkv.addr + img * T * P3
            self.add(C.OP_QUANT_W, f"k img{img}", ST, l, K=HD, N=T, B=H, a=Slot(src + D * 4, P3, HD * 4),
                     c=Slot(kb.addr + img * kbs, 0, 2 * kbs), f=Slot(kb.addr + img * kbs + c32(T) * HD * qb, 0, 2 * kbs))
            self.add(C.OP_QUANT_WT, f"v^T img{img}", ST, l, K=HD, N=T, B=H, a=Slot(src + 2 * D * 4, P3, HD * 4),
                     c=Slot(vb.addr + img * vbs, 0, 2 * vbs), f=Slot(vb.addr + img * vbs + HD * T * qb, 0, 2 * vbs),
                     p=[T])
        kblob = Blob(kb.addr, T, HD, qb, False)
        self.gemm("q k^T", ST, Slot(qh.addr, HD * qb, T * HD * qb), Slot(qs.addr, 0, T * 4), kblob,
                  Slot(sc.addr, T * 4, T * T * 4), T, B=2 * H, w_stride=kbs, p0=HD ** -0.5, **k)
        self.add(C.OP_SOFTMAX, "softmax", ST, l, flags=C.F_QOUT, K=T, N=2 * H * T, a=Slot(sc.addr, T * 4),
                 c=Slot(P.addr, T * qb), f=Slot(ps.addr), p=[f32bits(1.0)])
        for img in range(2):
            vblob = Blob(vb.addr + img * vbs, HD, T, qb, False)
            self.gemm(f"p v img{img}", ST, Slot(P.addr + img * T * T * qb, T * qb, 2 * T * T * qb),
                      Slot(ps.addr + img * T * 4, 0, 2 * T * 4), vblob,
                      Slot(attn.addr + img * T * D * 4, D * 4, HD * 4), T, B=H, w_stride=2 * vbs, **k)
        self.quant_x("attn", ST, Slot(attn.addr, D * 4), Slot(xq.addr, D * qb), Slot(xs.addr), NV, D, **k)
        self.gemm("o_proj", ST, Slot(xq.addr, D * qb), Slot(xs.addr), W["o"], Slot(a.addr, D * 4), NV, **k)
        self.norm("resid + ln2", ST, Slot(a.addr, D * 4), sh.norms[("v2", l)], Slot(xq.addr, D * qb), NV, D, sh.eps_v,
                  ln=True, resid=Slot(self.h_vis.addr, D * 4), sum_out=Slot(self.h_vis.addr, D * 4),
                  scales=Slot(xs.addr), **k)
        self.gemm("fc1", ST, Slot(xq.addr, D * qb), Slot(xs.addr), W["fc1"], Slot(t.addr, 4 * D * 4), NV, **k)
        self.add(C.OP_GELU, "gelu", ST, l, flags=C.F_QOUT, K=4 * D, N=NV, a=Slot(t.addr, 4 * D * 4),
                 c=Slot(xq.addr, 4 * D * qb), f=Slot(xs.addr))
        self.gemm("fc2", ST, Slot(xq.addr, 4 * D * qb), Slot(xs.addr), W["fc2"], Slot(a.addr, D * 4), NV, **k)
        last = l == len(sh.v) - 1
        nxt = sh.norms[("vpost", 0)] if last else sh.norms[("v1", l + 1)]
        if last:  # post_layernorm, FP32 out for the pixel shuffle
            self.norm("resid + post_layernorm", ST, Slot(a.addr, D * 4), nxt, Slot(self.vis_out.addr, D * 4), NV, D,
                      sh.eps_v, ln=True, resid=Slot(self.h_vis.addr, D * 4), **k)
        else:
            self.norm("resid + ln1", ST, Slot(a.addr, D * 4), nxt, Slot(xq.addr, D * qb), NV, D, sh.eps_v, ln=True,
                      resid=Slot(self.h_vis.addr, D * 4), sum_out=Slot(self.h_vis.addr, D * 4), scales=Slot(xs.addr), **k)

    def prefix(self):
        sh, qb, ST = self.sh, self.qb, ST_PREFIX
        D = D_V
        row = 16 * D * 4  # one shuffled row: 12288 FP32
        # pixel shuffle: for each segment j, 16 output rows per image; try one command for both images
        by_seg = {}
        for o, j, s0 in sh.shuffle:
            by_seg.setdefault(j, []).append((o, s0))
        for j, lst in sorted(by_seg.items()):
            lst.sort()
            src = [s0 for _, s0 in lst]
            # 16 rows = 4 blocks x 4 rows with uniform strides (checked), both images as 8 blocks
            r_st, b_st = src[1] - src[0], src[4] - src[0]
            ok = all(src[o] == src[0] + (o // 4) * b_st + (o % 4) * r_st for o in range(16))
            assert ok and T_IMG == 4 * b_st, "pixel shuffle is not a 2-level strided copy"
            self.add(C.OP_COPY, f"pixel shuffle seg{j}", ST, K=4 * D * 4, N=4, B=8,
                     a=Slot(self.vis_out.addr + src[0] * D * 4, r_st * D * 4, b_st * D * 4),
                     c=Slot(self.shuf.addr + j * 4 * D * 4, row, 4 * row))
        xq, xs = self.role("xq", 32 * 16 * D * qb), self.role("xs", a16(32 * 4))
        self.quant_x("connector in", ST, Slot(self.shuf.addr, row), Slot(xq.addr, 16 * D * qb), Slot(xs.addr), 32,
                     16 * D)
        for i, w in enumerate(sh.conn):
            k0 = i * sh.kmax
            self.gemm(f"connector k{k0}", ST, Slot(xq.addr + k0 * qb, 16 * D * qb), Slot(xs.addr), w,
                      Slot(self.h_p.addr, D_L * 4), 32, p0=math.sqrt(D_L), acc=i > 0)
        self.add(C.OP_GATHER, "embed_tokens", ST, flags=C.F_BF16, K=D_L, N=self.NL, a=Slot(sh.embed_tab, D_L * 2),
                 c=Slot(self.h_p.addr + 32 * D_L * 4, D_L * 4), d=Slot(self.in_ids.addr), p=[f32bits(math.sqrt(D_L))])
        self.quant_x("state", ST, Slot(self.in_state.addr, 32 * 4), Slot(xq.addr, 32 * qb), Slot(xs.addr), 1, 32)
        self.gemm("state_proj", ST, Slot(xq.addr, 32 * qb), Slot(xs.addr), sh.state_proj,
                  Slot(self.h_p.addr + (32 + self.NL) * D_L * 4, D_L * 4), 1)

    def attention(self, ST, l, step, *, q_src: Slot, n_q: int, pos: Buf, kblob: Blob, kb_stride: int,
                  vblob: Blob, vb_stride: int, n_keys_b: int, qinfo: Buf, kinfo: Buf, out: Slot):
        """RoPE(Q) -> Q K^T -> masked softmax -> P V for 15 query heads over 5 KV heads (GQA)."""
        qb = self.qb
        k = dict(layer=l, step=step)
        qh, qs = self.role("qh", H_L * n_q * HD * qb), self.role("qs", H_L * a16(n_q * 4))
        sc = self.role("sc", H_L * n_q * n_keys_b * 4)
        P, ps = self.role("P", H_L * n_q * n_keys_b * qb), self.role("ps", H_L * a16(n_q * 4))
        self.add(C.OP_ROPE, "rope q", ST, l, step, flags=C.F_QOUT, K=HD, N=n_q, B=H_L, a=q_src,
                 c=Slot(qh.addr, HD * qb, n_q * HD * qb), d=Slot(pos.addr), e=Slot(self.sh.rope_table, HD * 4),
                 f=Slot(qs.addr, 0, a16(n_q * 4)))
        self.gemm("q k^T", ST, Slot(qh.addr, HD * qb, n_q * HD * qb), Slot(qs.addr, 0, a16(n_q * 4)), kblob,
                  Slot(sc.addr, n_keys_b * 4, n_q * n_keys_b * 4), n_q, B=H_L, G=H_L // KV_L, w_stride=kb_stride,
                  p0=HD ** -0.5, **k)
        self.add(C.OP_SOFTMAX, "masked softmax", ST, l, step, flags=C.F_QOUT | C.F_MASK, K=n_keys_b, N=n_q, B=H_L,
                 a=Slot(sc.addr, n_keys_b * 4, n_q * n_keys_b * 4), c=Slot(P.addr, n_keys_b * qb, n_q * n_keys_b * qb),
                 d=Slot(qinfo.addr), e=Slot(kinfo.addr), f=Slot(ps.addr, 0, a16(n_q * 4)), p=[f32bits(1.0)])
        self.gemm("p v", ST, Slot(P.addr, n_keys_b * qb, n_q * n_keys_b * qb), Slot(ps.addr, 0, a16(n_q * 4)), vblob,
                  out, n_q, B=H_L, G=H_L // KV_L, w_stride=vb_stride, **k)

    def kv_blobs(self, ST, l, step, src: int, pitch: int, n: int, nb: int, kb: int, vb: int):
        """K blob (rows = keys) and V^T blob (rows = head dims, reduction = nb keys) per KV head."""
        qb, k = self.qb, dict(layer=l, step=step)
        kbs, vbs = blob_bytes(n, HD, qb), blob_bytes(HD, nb, qb)
        self.add(C.OP_QUANT_W, "k blob", ST, l, step, K=HD, N=n, B=KV_L, a=Slot(src, pitch, HD * 4),
                 c=Slot(kb, 0, kbs), f=Slot(kb + c32(n) * HD * qb, 0, kbs))
        self.add(C.OP_QUANT_WT, "v^T blob", ST, l, step, K=HD, N=n, B=KV_L, a=Slot(src + KV_L * HD * 4, pitch, HD * 4),
                 c=Slot(vb, 0, vbs), f=Slot(vb + HD * nb * qb, 0, vbs), p=[nb])
        return Blob(kb, n, HD, qb, False), kbs, Blob(vb, HD, nb, qb, False), vbs

    def lm(self):
        sh, qb, ST, NP = self.sh, self.qb, ST_LM, self.NP
        D, KVW = D_L, 2 * KV_L * HD
        for l, W in enumerate(sh.l):
            k = dict(layer=l)
            xq, xs = self.role("xq", NP * I_L * qb), self.role("xs", a16(NP * 4))
            a, qbuf = self.role("a", NP * D * 4), self.role("qkv", NP * D * 4)
            t, attn = self.role("t", NP * 2 * I_L * 4), self.role("attn", NP * D * 4)
            g1 = sh.norms[("l1", l)]
            if l == 0:
                self.norm("input_layernorm", ST, Slot(self.h_p.addr, D * 4), g1, Slot(xq.addr, D * qb), NP, D, sh.eps_l,
                          scales=Slot(xs.addr), **k)
            else:
                self.norm("resid + input_layernorm", ST, Slot(a.addr, D * 4), g1, Slot(xq.addr, D * qb), NP, D, sh.eps_l,
                          resid=Slot(self.h_p.addr, D * 4), sum_out=Slot(self.h_p.addr, D * 4), scales=Slot(xs.addr), **k)
            kv = self.kv[l].addr
            self.gemm("k|v", ST, Slot(xq.addr, D * qb), Slot(xs.addr), W["kv"], Slot(kv, KVW * 4), NP, **k)
            self.add(C.OP_ROPE, "rope k (in place)", ST, l, K=HD, N=NP, B=KV_L, a=Slot(kv, KVW * 4, HD * 4),
                     c=Slot(kv, KVW * 4, HD * 4), d=Slot(self.pos_prefix.addr), e=Slot(sh.rope_table, HD * 4))
            if l == len(sh.l) - 1 and self.prune_last:
                continue  # only the cache of the last layer is used
            self.gemm("q", ST, Slot(xq.addr, D * qb), Slot(xs.addr), W["q"], Slot(qbuf.addr, D * 4), NP, **k)
            kb, vb = self.role("kb", KV_L * blob_bytes(NP, HD, qb)), self.role("vb", KV_L * blob_bytes(HD, self.NPb, qb))
            kblob, kbs, vblob, vbs = self.kv_blobs(ST, l, 0, kv, KVW * 4, NP, self.NPb, kb.addr, vb.addr)
            self.attention(ST, l, 0, q_src=Slot(qbuf.addr, D * 4, HD * 4), n_q=NP, pos=self.pos_prefix, kblob=kblob,
                           kb_stride=kbs, vblob=vblob, vb_stride=vbs, n_keys_b=self.NPb, qinfo=self.qinfo_prefix,
                           kinfo=self.kinfo_prefix, out=Slot(attn.addr, D * 4, HD * 4))
            self.mlp_block(ST, l, 0, attn, D, W, sh.norms[("l2", l)], sh.eps_l, NP, I_L, self.h_p, D * 4, D * 4)

    def mlp_block(self, ST, l, step, attn: Buf, d_attn: int, W, g2: int, eps: float, N: int, I: int, h: Buf,
                  h_pitch: int, a_pitch: int):
        """o_proj -> resid + post norm -> gate|up -> SiLU * up -> down (output in role 'a')."""
        qb, k = self.qb, dict(layer=l, step=step)
        xq, xs = self.role("xq", N * max(I, d_attn) * qb), self.role("xs", a16(N * 4))
        a, t = self.role("a", N * a_pitch), self.role("t", N * 2 * I * 4)
        dh = W["o"].M
        self.quant_x("attn", ST, Slot(attn.addr, d_attn * 4), Slot(xq.addr, d_attn * qb), Slot(xs.addr), N, d_attn, **k)
        self.gemm("o_proj", ST, Slot(xq.addr, d_attn * qb), Slot(xs.addr), W["o"], Slot(a.addr, a_pitch), N, **k)
        self.norm("resid + post_attention_layernorm", ST, Slot(a.addr, a_pitch), g2, Slot(xq.addr, dh * qb), N, dh, eps,
                  resid=Slot(h.addr, h_pitch), sum_out=Slot(h.addr, h_pitch), scales=Slot(xs.addr), **k)
        self.gemm("gate|up", ST, Slot(xq.addr, dh * qb), Slot(xs.addr), W["gu"], Slot(t.addr, 2 * I * 4), N, **k)
        self.add(C.OP_SILU_MUL, "silu(gate) * up", ST, l, step, flags=C.F_QOUT, K=I, N=N, a=Slot(t.addr, 2 * I * 4),
                 b=Slot(t.addr + I * 4, 2 * I * 4), c=Slot(xq.addr, I * qb), f=Slot(xs.addr))
        self.gemm("down", ST, Slot(xq.addr, I * qb), Slot(xs.addr), W["down"], Slot(a.addr, a_pitch), N, **k)

    def cross_kv(self):
        """Expert K/V projections of the cached prefix K/V: step-invariant, so once per chunk."""
        sh, qb, ST, NP = self.sh, self.qb, ST_XKV, self.NP
        KVW = 2 * KV_L * HD
        for l, W in enumerate(sh.e):
            if l % sh.every == 0:
                continue
            k = dict(layer=l)
            xq, xs = self.role("xq", 2 * NP * KV_L * HD * qb), self.role("xs", 2 * a16(NP * 4))
            t = self.role("t", 2 * NP * KV_L * HD * 4)
            W5 = KV_L * HD
            self.quant_x("cache k|v", ST, Slot(self.kv[l].addr, KVW * 4, W5 * 4), Slot(xq.addr, W5 * qb, NP * W5 * qb),
                         Slot(xs.addr, 0, a16(NP * 4)), NP, W5, B=2, **k)
            self.gemm("expert k_proj|v_proj", ST, Slot(xq.addr, W5 * qb, NP * W5 * qb), Slot(xs.addr, 0, a16(NP * 4)),
                      W["xkv"], Slot(t.addr, W5 * 4, NP * W5 * 4), NP, B=2, w_stride=W["xkv_stride"], **k)
            # t = [k'(NP x 320) ; v'(NP x 320)]: present it as the usual [k | v] row layout via the slots
            kbs, vbs = blob_bytes(NP, HD, qb), blob_bytes(HD, self.NPb, qb)
            self.add(C.OP_QUANT_W, "k' blob", ST, l, K=HD, N=NP, B=KV_L, a=Slot(t.addr, W5 * 4, HD * 4),
                     c=Slot(self.ckb[l].addr, 0, kbs), f=Slot(self.ckb[l].addr + c32(NP) * HD * qb, 0, kbs))
            self.add(C.OP_QUANT_WT, "v'^T blob", ST, l, K=HD, N=NP, B=KV_L, a=Slot(t.addr + NP * W5 * 4, W5 * 4, HD * 4),
                     c=Slot(self.cvb[l].addr, 0, vbs), f=Slot(self.cvb[l].addr + HD * self.NPb * qb, 0, vbs), p=[self.NPb])

    def expert_step(self, s):
        sh, qb, ST = self.sh, self.qb, ST_EXPERT
        k = dict(step=s)
        xq, xs = self.role("xq", NS * D_E * qb), self.role("xs", a16(NS * 4))
        a, t = self.role("a", NS * D_EP * 4), self.role("t", NS * D_EP * 4)
        EP = D_EP * 4
        # suffix embedding
        self.quant_x("x_t", ST, Slot(self.x_t.addr, 32 * 4), Slot(xq.addr, 32 * qb), Slot(xs.addr), NS, 32, **k)
        self.gemm("action_in_proj", ST, Slot(xq.addr, 32 * qb), Slot(xs.addr), sh.a_in, Slot(a.addr, EP), NS, **k)
        self.quant_x("action emb", ST, Slot(a.addr, EP), Slot(xq.addr, D_E * qb), Slot(xs.addr), NS, D_E, **k)
        self.gemm("action_time_mlp_in (+ time bias)", ST, Slot(xq.addr, D_E * qb), Slot(xs.addr), sh.t_in,
                  Slot(t.addr, EP), NS, addd=Slot(sh.time_bias + s * EP, 0), **k)
        self.add(C.OP_SILU_MUL, "silu", ST, 0, s, flags=C.F_QOUT | C.F_NOMUL, K=D_E, N=NS, a=Slot(t.addr, EP),
                 c=Slot(xq.addr, D_E * qb), f=Slot(xs.addr))
        self.gemm("action_time_mlp_out", ST, Slot(xq.addr, D_E * qb), Slot(xs.addr), sh.t_out, Slot(self.h_s.addr, EP),
                  NS, **k)
        KVW = 2 * KV_L * HD
        for l, W in enumerate(sh.e):
            kk = dict(layer=l, step=s)
            xq, xs = self.role("xq", NS * I_E * qb), self.role("xs", a16(NS * 4))
            a, qbuf, attn = self.role("a", NS * EP), self.role("qkv", NS * D_L * 4), self.role("attn", NS * D_L * 4)
            g1 = sh.norms[("e1", l)]
            if l == 0:
                self.norm("input_layernorm", ST, Slot(self.h_s.addr, EP), g1, Slot(xq.addr, D_E * qb), NS, D_E, sh.eps_e,
                          scales=Slot(xs.addr), **kk)
            else:
                self.norm("resid + input_layernorm", ST, Slot(a.addr, EP), g1, Slot(xq.addr, D_E * qb), NS, D_E,
                          sh.eps_e, resid=Slot(self.h_s.addr, EP), sum_out=Slot(self.h_s.addr, EP),
                          scales=Slot(xs.addr), **kk)
            self.gemm("q", ST, Slot(xq.addr, D_E * qb), Slot(xs.addr), W["q"], Slot(qbuf.addr, D_L * 4), NS, **kk)
            if l % sh.every == 0:  # self-attention over [prefix cache ; suffix]
                kv = self.kv[l].addr
                sfx = kv + self.NP * KVW * 4
                self.gemm("k|v -> cache rows NP..", ST, Slot(xq.addr, D_E * qb), Slot(xs.addr), W["kv"],
                          Slot(sfx, KVW * 4), NS, **kk)
                self.add(C.OP_ROPE, "rope k (in place)", ST, l, s, K=HD, N=NS, B=KV_L, a=Slot(sfx, KVW * 4, HD * 4),
                         c=Slot(sfx, KVW * 4, HD * 4), d=Slot(self.pos_self.addr), e=Slot(sh.rope_table, HD * 4))
                kb = self.role("kb", KV_L * blob_bytes(self.NKV, HD, qb))
                vb = self.role("vb", KV_L * blob_bytes(HD, self.NKb, qb))
                kblob, kbs, vblob, vbs = self.kv_blobs(ST, l, s, kv, KVW * 4, self.NKV, self.NKb, kb.addr, vb.addr)
                self.attention(ST, l, s, q_src=Slot(qbuf.addr, D_L * 4, HD * 4), n_q=NS, pos=self.pos_self,
                               kblob=kblob, kb_stride=kbs, vblob=vblob, vb_stride=vbs, n_keys_b=self.NKb,
                               qinfo=self.qinfo_self, kinfo=self.kinfo_self, out=Slot(attn.addr, D_L * 4, HD * 4))
            else:  # cross-attention to the hoisted prefix K/V
                kbs, vbs = blob_bytes(self.NP, HD, qb), blob_bytes(HD, self.NPb, qb)
                self.attention(ST, l, s, q_src=Slot(qbuf.addr, D_L * 4, HD * 4), n_q=NS, pos=self.pos_cross,
                               kblob=Blob(self.ckb[l].addr, self.NP, HD, qb, False), kb_stride=kbs,
                               vblob=Blob(self.cvb[l].addr, HD, self.NPb, qb, False), vb_stride=vbs,
                               n_keys_b=self.NPb, qinfo=self.qinfo_cross, kinfo=self.kinfo_cross,
                               out=Slot(attn.addr, D_L * 4, HD * 4))
            self.mlp_block(ST, l, s, attn, D_L, W, sh.norms[("e2", l)], sh.eps_e, NS, I_E, self.h_s, EP, EP)
        xq, xs = self.role("xq", NS * D_E * qb), self.role("xs", a16(NS * 4))
        a = self.role("a", NS * EP)
        self.norm("resid + expert norm", ST_HEAD, Slot(a.addr, EP), sh.norms[("efinal", 0)], Slot(xq.addr, D_E * qb),
                  NS, D_E, sh.eps_e, resid=Slot(self.h_s.addr, EP), scales=Slot(xs.addr), **k)
        self.gemm("action_out_proj", ST_HEAD, Slot(xq.addr, D_E * qb), Slot(xs.addr), sh.a_out,
                  Slot(self.v_t.addr, 32 * 4), NS, **k)
        self.add(C.OP_AXPY, "x_t += dt v_t", ST_HEAD, 0, s, flags=C.F_YIELD if self.yield_every_step else 0, K=32,
                 N=NS, a=Slot(self.x_t.addr, 32 * 4), b=Slot(self.v_t.addr, 32 * 4), c=Slot(self.x_t.addr, 32 * 4),
                 p=[f32bits(-1.0 / NSTEPS)])

    # ---- host side, per inference
    def host_inputs(self, images, img_masks, lang_tokens, lang_masks, state, noise) -> dict[int, bytes]:
        """Everything the ARM writes into the arena before a start (and nothing else)."""
        assert len(images) == 2 and all(i.shape[-1] == 256 for i in images)
        pt = []
        for img in images:  # im2col in the conv weight's (c, ky, kx) order
            x = img[0].float().reshape(3, 16, 16, 16, 16).permute(1, 3, 0, 2, 4).reshape(T_IMG, D_V)
            pt.append(x)
        pad = torch.cat([img_masks[0].expand(16), img_masks[1].expand(16), lang_masks[0].bool(),
                         torch.ones(1, dtype=torch.bool)])
        blk = torch.zeros(pad.shape[0], dtype=torch.int32)
        blk[-1] = 1  # state token: its own block (image/language rows do not attend to it)
        pos = torch.cumsum(pad.int(), 0) - 1
        ids = lang_tokens[0].int()
        if self.compact:  # drop padded rows: masked keys contribute exactly 0, pad queries are never read
            assert bool(img_masks[0].all() and img_masks[1].all())
            keep = pad.nonzero()[:, 0]
            ids = ids[lang_masks[0].bool()]
            pad, blk, pos = pad[keep], blk[keep], pos[keep]
        assert pad.shape[0] == self.NP and ids.shape[0] == self.NL, (pad.shape, self.NP)
        n_valid = int(pad.sum())
        kpre = torch.full((self.NPb,), -1, dtype=torch.int32)
        kpre[: self.NP] = torch.where(pad, blk, torch.tensor(-1, dtype=torch.int32))
        kself = torch.full((self.NKb,), -1, dtype=torch.int32)
        kself[: self.NP] = torch.where(pad, 0, -1)
        kself[self.NP: self.NKV] = torch.arange(1, NS + 1, dtype=torch.int32)
        kcross = torch.full((self.NPb,), -1, dtype=torch.int32)
        kcross[: self.NP] = torch.where(pad, 0, -1)
        b = lambda t: t.contiguous().numpy().tobytes()
        i32 = lambda t: b(torch.as_tensor(t).to(torch.int32))
        return {
            self.in_patches.addr: b(torch.cat(pt, 0)),
            self.in_ids.addr: i32(ids),
            self.in_state.addr: b(state[0].float()),
            self.x_t.addr: b(noise[0].float()),
            self.pos_prefix.addr: i32(pos),
            self.qinfo_prefix.addr: i32(blk),
            self.kinfo_prefix.addr: i32(kpre),
            self.pos_self.addr: i32(n_valid + torch.arange(NS)),
            self.qinfo_self.addr: i32(torch.arange(1, NS + 1)),
            self.kinfo_self.addr: i32(kself),
            self.pos_cross.addr: i32(torch.arange(NS)),
            self.qinfo_cross.addr: i32(torch.zeros(NS)),
            self.kinfo_cross.addr: i32(kcross),
        }

    def place_list(self, lists: Region) -> Buf:
        self.list_buf = lists.alloc(f"list:{self.ar.name}", len(self.cmds) * C.REC_BYTES)
        return self.list_buf

    def counts(self) -> dict:
        by_op, by_stage = {}, {}
        for c in self.cmds:
            by_op[C.opnames[c.op]] = by_op.get(C.opnames[c.op], 0) + 1
            st = c.tag >> 24
            by_stage[st] = by_stage.get(st, 0) + 1
        names = {ST_VISION: "vision", ST_PREFIX: "connector+prefix", ST_LM: "lm", ST_XKV: "cross_kv_hoist",
                 ST_EXPERT: "expert", ST_HEAD: "head+euler+end"}
        return dict(total=len(self.cmds), by_op=by_op, by_stage={names[k]: v for k, v in sorted(by_stage.items())})
