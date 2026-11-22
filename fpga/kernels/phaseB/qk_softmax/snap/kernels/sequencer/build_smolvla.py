"""SmolVLA -12392 on the contract's chunk list: fills every buffer that fpga/kernels/contract_plan.py build()
addresses with the policy's weights, the load-time constants and the per-chunk inputs, so exec_ref.py can run
the list as the IP would (CONTRACT.md 1.3, 2.4, 6, 10). Nothing here changes a record: the list is
contract_plan's, byte for byte.

Records are identified by their note (phase:name), opcode and order: vision layer i = the i-th vision `ln1`,
LM layer i = the i-th lm `n1`, cross layer 2c+1 = the c-th `k.in`, step s = the s-th `x_t`, expert layer i =
the i-th `n1` of that step. Every slot the loader fills is read from the record itself (no address is
recomputed).

Weights (the offline packer's job, CONTRACT.md 2.4): int16 per output row by contract.quant_rows, in the port
tiling at slot B, s_row at E, bias at G; pad rows 0 with s_row 1.0 and bias 0. Folds:
  connector (3 K-chunks of 4096)  s_row' = fl32(s_row * fl32(sqrt(960)))     (the image embedding x sqrt(960))
  action_out_proj (Euler)         W' = -Wq, s_row' = fl32(fl32(0.1) s_row), b' = fl32(fl32(-0.1) b),
                                  D = C = x_t
  action_time_mlp_in              the a_emb half as the blob; W_t t_emb(t_s) + b per step as the D row (FP32)
Constants: the vision position embedding (D of the patch GEMM), the 12 language rows (embedded x sqrt(960),
FP32, from the frame's tokens: the task string is fixed), the 10 time vectors, norm gamma (/ beta), the 3 RoPE
tables (rope_emu.table_rows: torch's FP32 cos / sin -> Q1.17), the 6 rank vectors. Inputs per chunk: the
im2col patches (int16 + scales), the state (int16 + scale), x_t = noise. exact=True writes FP32 shadows with
scale 1 instead.
"""

from __future__ import annotations

import math
import struct
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rope"))
import contract as K  # noqa: E402
import contract_plan as CP  # noqa: E402
import rope_emu  # noqa: E402
from contract import C  # noqa: E402
from exec_ref import Executor, Memory  # noqa: E402

F32 = np.float32


def f32(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


QKV_S_BYTES = CP.TQ_P * 1600 * 4  # contract_plan.build(): QKV_s = m.alloc(TQ_P * 1600 * 4), 45 rows of q|k|v


def chunk_plan(base: int, fix: bool = True):
    """contract_plan.build(base). fix=True works around a sizing bug there (reported to the contract owner):
    the expert's q|k|v buffer QKV_s is allocated for TQ_P = 45 rows but every Euler step writes TQ_E = 50 rows
    into it, so the step qkv GEMM's rows 45..49 (32,000 bytes) land on the next allocation, expert layer 0's
    self-attention K blob (its prefix keys, written once per chunk by the LM). Every 45 x 1600 FP32 buffer of
    the arena is allocated for max(TQ_P, TQ_E) rows instead; no record changes but for addresses. A no-op once
    contract_plan is fixed."""
    if not fix:
        return CP.build(base)
    orig = CP.Mem.alloc

    def alloc(self, nbytes: int, align: int = 64) -> int:
        if self.cur == "arena" and nbytes == QKV_S_BYTES:
            nbytes = max(CP.TQ_P, CP.TQ_E) * 1600 * 4
        return orig(self, nbytes, align)

    CP.Mem.alloc = alloc
    try:
        return CP.build(base)
    finally:
        CP.Mem.alloc = orig


class Program:
    """One arena: contract_plan's list, its memory, and the -12392 data in it."""

    def __init__(
        self,
        policy,
        *,
        base: int = 0x8_0000_0000,
        exact: bool = False,
        fe: str = "torch",
        plan_fix: bool = True,
        nsb: int = 10,
    ):
        self.p, self.exact = policy, exact
        self.P = chunk_plan(base, plan_fix)
        self.cmds = self.P.cmds
        bad = K.validate_list(self.cmds)
        assert not bad, bad[:5]
        lay = self.P.layout
        self.mem = Memory(shadow=exact)
        for a0, a1 in lay["regions"].values():
            self.mem.add_region(a0, a1 - a0)
        self.list_addr = (
            lay["arena_end"] + 0xFFFFF
        ) & ~0xFFFFF  # the list after the arena (any 16-B address)
        blob = K.encode_list(self.cmds)
        self.mem.add_region(self.list_addr, len(blob))
        self.mem.write(self.list_addr, blob)
        self.status = lay["status"]
        self.ex = Executor(self.mem, exact=exact, fe=fe, nsb=nsb)
        m = policy.model
        vwe = m.vlm_with_expert
        self.vlm = vwe.get_vlm_model()
        self.vis, self.text, self.exp = self.vlm.vision_model, self.vlm.text_model, vwe.lm_expert
        assert (
            self.vlm.connector.scale_factor == 4
            and vwe.num_vlm_layers == 16
            and vwe.self_attn_every_n_layers == 2
        )
        self._identify()
        with torch.no_grad():
            self._weights()
            self._consts()

    # ------------------------------------------------------------ records -> roles
    def _identify(self) -> None:
        self.roles: list[tuple] = []  # per record: (phase, name, layer, step)
        layer = defaultdict(lambda: -1)
        step = -1
        for c in self.cmds:
            ph, name = c.note.split(":", 1)
            if ph == "vision" and name == "ln1":
                layer["vision"] += 1
            if ph == "lm" and name == "n1":
                layer["lm"] += 1
            if ph == "cross_kv" and name == "k.in":
                layer["cross_kv"] += 1
            if ph == "steps" and name == "x_t":
                step += 1
                layer["steps"] = -1
            if ph == "steps" and name == "n1":
                layer["steps"] += 1
            li = layer[ph] if ph != "cross_kv" else 2 * layer[ph] + 1
            self.roles.append((ph, name, li, step))
        self.first = {}  # (phase, name, opcode) -> first record index
        for i, (c, (ph, name, _, _)) in enumerate(zip(self.cmds, self.roles, strict=True)):
            self.first.setdefault((ph, name, c.op), i)

    def find(self, ph: str, name: str, op: int) -> K.Cmd:
        return self.cmds[self.first[(ph, name, op)]]

    # ------------------------------------------------------------ weights
    def _module_weight(self, ph: str, name: str, i: int):
        """(W [rows][k] FP32, bias or None) of a weight GEMM record."""
        cat = lambda *ts: torch.cat([t.detach().float() for t in ts], 0)  # noqa: E731
        m = self.p.model
        if ph == "vision":
            if name == "patch":
                pe = self.vis.embeddings.patch_embedding
                return pe.weight.detach().float().reshape(pe.weight.shape[0], -1), pe.bias.detach().float()
            L = self.vis.encoder.layers[i]
            a = L.self_attn
            if name == "qkv":
                return cat(a.q_proj.weight, a.k_proj.weight, a.v_proj.weight), cat(
                    a.q_proj.bias, a.k_proj.bias, a.v_proj.bias
                )
            mod = {"o": a.out_proj, "fc1": L.mlp.fc1, "fc2": L.mlp.fc2}[name]
            return mod.weight.detach().float(), mod.bias.detach().float()
        if ph == "connector":
            if name == "state":
                return m.state_proj.weight.detach().float(), m.state_proj.bias.detach().float()
            j = int(name[-1])
            w = self.vlm.connector.modality_projection.proj.weight.detach().float()
            return w[:, 4096 * j : 4096 * (j + 1)], None
        if ph == "lm":
            L = self.text.layers[i]
            a, f = L.self_attn, L.mlp
            if name == "qkv":
                return cat(a.q_proj.weight, a.k_proj.weight, a.v_proj.weight), None
            if name == "kv":
                return cat(a.k_proj.weight, a.v_proj.weight), None
            return {
                "o": a.o_proj.weight,
                "gu": cat(f.gate_proj.weight, f.up_proj.weight),
                "down": f.down_proj.weight,
            }[name].detach().float(), None
        if ph == "cross_kv":
            a = self.exp.layers[i].self_attn
            return {"kproj": a.k_proj, "vproj": a.v_proj}[name].weight.detach().float(), None
        if ph == "steps":
            if name == "a_in":
                return m.action_in_proj.weight.detach().float(), m.action_in_proj.bias.detach().float()
            if name == "t_in":
                return m.action_time_mlp_in.weight.detach().float()[:, : self.exp.config.hidden_size], None
            if name == "t_out":
                return (
                    m.action_time_mlp_out.weight.detach().float(),
                    m.action_time_mlp_out.bias.detach().float(),
                )
            if name == "a_out+euler":
                return m.action_out_proj.weight.detach().float(), m.action_out_proj.bias.detach().float()
            L = self.exp.layers[i]
            a, f = L.self_attn, L.mlp
            if name == "qkv":
                return cat(a.q_proj.weight, a.k_proj.weight, a.v_proj.weight), None
            if name == "q":
                return a.q_proj.weight.detach().float(), None
            return {
                "o": a.o_proj.weight,
                "gu": cat(f.gate_proj.weight, f.up_proj.weight),
                "down": f.down_proj.weight,
            }[name].detach().float(), None
        raise KeyError((ph, name))

    def _weights(self) -> None:
        done = {}
        self.n_blobs, self.blob_bytes = 0, 0
        attn = {"qk", "pv", "pv0", "pv1"}
        for c, (ph, name, i, _) in zip(self.cmds, self.roles, strict=True):
            if c.op != C.OP_GEMM or name in attn:
                continue
            Bs, Es, Gs = c.slot("B"), c.slot("E"), c.slot("G")
            key = (ph, name, i)
            if Bs.addr in done:
                assert done[Bs.addr] == key, (done[Bs.addr], key)
                continue
            done[Bs.addr] = key
            W, bias = self._module_weight(ph, name, i)
            rows, k = W.shape
            one = c.B == 1 or (Bs.stride == 0 and Es.stride == 0 and Gs.stride == 0)  # items share the blob
            assert rows <= c.M and k == c.K and one, (key, W.shape, c.M, c.K, c.B)
            Wp = torch.zeros(c.M, c.K)
            Wp[:rows] = W
            bp = np.zeros(c.M, F32)
            if bias is not None:
                bp[:rows] = bias.numpy()
            if self.exact:
                q, srow, x = None, np.ones(c.M, F32), Wp
            else:
                q, srow = K.quant_rows(Wp.numpy())
                x = None
            if ph == "connector" and name.startswith("conn"):
                srow = f32(srow * F32(math.sqrt(960)))
            if name == "a_out+euler":
                if q is not None:
                    q = -q  # exact: |q| <= 32767
                else:
                    x = -x
                srow = f32(F32(0.1) * srow)
                bp = f32(F32(-0.1) * bp)
            self.ex.put_tiled(Bs, 0, 0, q, x)
            self.mem.write(Es.addr, f32(srow).tobytes())
            if c.flags & C.F_BIAS:
                self.mem.write(Gs.addr, f32(bp).tobytes())
            else:
                assert bias is None, key
            self.n_blobs += 1
            self.blob_bytes += 4 * Bs.pitch

    # ------------------------------------------------------------ constants
    def _consts(self) -> None:
        from lerobot.policies.smolvla.modeling_smolvla import create_sinusoidal_pos_embedding

        m, cfg = self.p.model, self.p.config
        # vision position embedding: the patch embedding forward with the conv replaced by zeros (= pos)
        emb = self.vis.embeddings
        pe = emb.patch_embedding

        class _Zero(torch.nn.Module):
            def forward(self, x):
                return torch.zeros(x.shape[0], pe.out_channels, x.shape[2] // 16, x.shape[3] // 16)

        emb.patch_embedding = _Zero()
        try:
            pos = emb(
                pixel_values=torch.zeros(1, 3, 256, 256),
                patch_attention_mask=torch.ones(1, 16, 16, dtype=torch.bool),
            )[0]
        finally:
            emb.patch_embedding = pe
        d = self.find("vision", "patch", C.OP_GEMM).slot("D")
        assert d.pitch == 4 * 768 and pos.shape == (256, 768)
        self.mem.write(d.addr, f32(pos.float().numpy()).tobytes())
        # time vectors: W_t t_emb(t_s) + b, t_s = 1 + s dt (euler_integrate), padded to 736
        Wt, bt = m.action_time_mlp_in.weight.detach().float(), m.action_time_mlp_in.bias.detach().float()
        de = self.exp.config.hidden_size
        for c, (ph, name, _, s) in zip(self.cmds, self.roles, strict=True):
            if ph == "steps" and name == "t_in":
                t = torch.tensor(1.0 + s * (-1.0 / cfg.num_steps), dtype=torch.float32).expand(1)
                te = create_sinusoidal_pos_embedding(
                    t, de, cfg.min_period, cfg.max_period, device=torch.device("cpu")
                )
                v = torch.zeros(c.M)
                v[:de] = torch.nn.functional.linear(te.float(), Wt[:, de:], bt)[0]
                self.mem.write(c.slot("D").addr, f32(v.numpy()).tobytes())
        # norms
        for c, (ph, name, i, _) in zip(self.cmds, self.roles, strict=True):
            if c.op != C.OP_NORM:
                continue
            if ph == "vision":
                mod = (
                    self.vis.post_layernorm
                    if name == "post_ln"
                    else getattr(
                        self.vis.encoder.layers[i], {"ln1": "layer_norm1", "ln2": "layer_norm2"}[name]
                    )
                )
                eps = mod.eps
            elif ph == "lm":
                mod = getattr(
                    self.text.layers[i], {"n1": "input_layernorm", "n2": "post_attention_layernorm"}[name]
                )
                eps = mod.variance_epsilon
            else:
                mod = (
                    self.exp.norm
                    if name == "final"
                    else getattr(
                        self.exp.layers[i], {"n1": "input_layernorm", "n2": "post_attention_layernorm"}[name]
                    )
                )
                eps = mod.variance_epsilon
            assert K.bits_f32(c.par(C.P_EPS)) == F32(eps), (ph, name, eps)
            assert bool(c.flags & C.F_LN) == (ph == "vision")
            v = mod.weight.detach().float()
            if c.flags & C.F_LN:
                v = torch.cat([v, mod.bias.detach().float()])
            assert v.numel() == (2 if c.flags & C.F_LN else 1) * c.K
            self.mem.write(c.slot("E").addr, f32(v.numpy()).tobytes())
        # RoPE tables: prefix positions 0..44 (cumsum of the valid prefix), expert self 45..94, cross 0..49
        n_pre = CP.TQ_P
        for c, (ph, _name, i, _) in zip(self.cmds, self.roles, strict=True):
            if c.op != C.OP_ROPE:
                continue
            if ph == "lm":
                pos_t = torch.arange(c.N)
            elif i % 2 == 0:
                pos_t = n_pre + torch.arange(c.N)
            else:
                pos_t = torch.arange(c.N)
            tab = rope_emu.table_rows(pos_t).numpy().astype(np.int32)
            e = c.slot("E")
            assert e.pitch == 256 and tab.shape == (c.N, 64)
            self.mem.write(e.addr, tab.tobytes())
        # rank vectors (CONTRACT.md 5): query row t = g TqP + i; pad -1
        for c, (ph, _name, i, _) in zip(self.cmds, self.roles, strict=True):
            if c.op != C.OP_SOFTMAX or not c.flags & C.F_MASK:
                continue
            nq, L = c.N, c.K
            tq = nq // 3
            ii = torch.arange(nq) % tq
            kk = torch.arange(L)
            if ph == "lm":  # image / language 0, state 1, pad -1
                rq = torch.where(ii < n_pre - 1, 0, torch.where(ii == n_pre - 1, 1, -1))
                rk = torch.where(kk < n_pre - 1, 0, torch.where(kk == n_pre - 1, 1, -1))
            elif (
                i % 2 == 0
            ):  # expert self: suffix row i has rank 2 + i; prefix keys 0 / 1, suffix key j at 64 + j
                rq = torch.where(ii < CP.TQ_E, 2 + ii, -1)
                lp = K.pad32(n_pre)
                rk = torch.where(kk < n_pre - 1, 0, torch.where(kk == n_pre - 1, 1, -1))
                rk = torch.where((kk >= lp) & (kk < lp + CP.TQ_E), 2 + kk - lp, rk)
            else:  # expert cross: every valid prefix key, every valid row
                rq = torch.where(ii < CP.TQ_E, 0, -1)
                rk = torch.where(kk < n_pre, 0, -1)
            for sl, v in (("D", rq), ("E", rk)):
                self.mem.write(c.slot(sl).addr, v.to(torch.int32).numpy().tobytes())

    # ------------------------------------------------------------ per chunk
    def lang_rows(self, lang_tokens: torch.Tensor, lang_masks: torch.Tensor) -> torch.Tensor:
        ids = lang_tokens[0][lang_masks[0].bool()]
        assert ids.numel() == CP.TQ_P - 2 * 16 - 1, ids.numel()
        emb = self.text.get_input_embeddings()(ids[None])
        return (emb * math.sqrt(emb.shape[-1]))[0].float()

    def _put_qin(self, c: K.Cmd, a_slot: str, f_slot: str, x: torch.Tensor) -> None:
        """Host input rows [B][N][K] -> int16 + scales by the contract quantizer (exact: FP32 shadow,
        scale 1)."""
        A, Fs = c.slot(a_slot), c.slot(f_slot)
        for b in range(x.shape[0]):
            a = A.addr + b * A.stride
            if self.exact:
                self.mem.wsview(a, x.shape[1:], (A.pitch, 2)).copy_(x[b])
                s = np.ones(x.shape[1], F32)
            else:
                q, s = K.quant_rows(x[b].numpy())
                self.mem.view(a, x.shape[1:], (A.pitch, 2), torch.int16).copy_(torch.from_numpy(q))
            self.mem.write(
                Fs.addr + b * Fs.stride, np.concatenate([s, np.ones(K.pad4(len(s)) - len(s), F32)]).tobytes()
            )

    def run(self, images, img_masks, lang_tokens, lang_masks, state, noise) -> torch.Tensor:
        """The ARM's part of one chunk + one list start (restarted after any YIELD); returns x_t
        [1][50][32]."""
        assert len(images) == 2 and all(bool(mk.all()) for mk in img_masks)
        with torch.no_grad():
            pt = torch.stack(
                [
                    im[0].float().reshape(3, 16, 16, 16, 16).permute(1, 3, 0, 2, 4).reshape(256, 768)
                    for im in images
                ]
            )
            self._put_qin(self.find("vision", "patch", C.OP_GEMM), "A", "F", pt)
            self._put_qin(
                self.find("connector", "state", C.OP_GEMM), "A", "F", state.float().reshape(1, 1, -1)
            )
            lr = self.lang_rows(lang_tokens, lang_masks)
            self.mem.write(
                self.find("connector", "lang", C.OP_COPY).slot("A").addr, f32(lr.numpy()).tobytes()
            )
            xa = self.find("steps", "x_t", C.OP_QUANT_X).slot("A")
            self.mem.write(xa.addr, f32(noise[0].float().numpy()).tobytes())
            code, nxt, starts = self.ex.run_list(self.list_addr, self.status)
        st = K.parse_status(self.mem.read(self.status, C.STATUS_BYTES))
        assert code == C.ST_END and nxt == len(self.cmds) - 1 and st["magic_ok"], (code, nxt, st)
        self.last_status = st | {"list_starts": starts}
        return self.mem.view(xa.addr, (1, CP.TQ_E, CP.A_D), (CP.TQ_E * 4 * CP.A_D, 4 * CP.A_D, 4)).clone()


def status_words(raw: bytes) -> list[int]:
    return list(struct.unpack("<12I", raw[: C.STATUS_BYTES]))
