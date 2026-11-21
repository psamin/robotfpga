"""Contract command lists for tb_attn.cpp from real SmolVLA -12392 attention calls (check_policy.py --dump).

For every site (vision layer 0, LM prefix layer 0, the expert's self- and cross-attention of step 0; prefix
compacted to its 45 valid tokens) this writes vectors/<site>/:

  mem.bin, meta.txt  a DDR image at a 40-bit base (0x8_0000_0000, 0xFF_0000_0000, 0x4000_0000, 0xFF_8000_0000) holding
                     the FP32 Q / K / V rows as the projections leave them, the rank vectors and a list of contract
                     records (CONTRACT.md 5): QUANT_X Q (p4 = -3, head-major) -> QUANT_W K (KW blob) -> QUANT_WT V
                     (VW blob; expert self: prefix and suffix parts) -> GEMM Q.K^T -> SOFTMAX -> GEMM P.V (C row
                     groups R = TqP, P1 = 256 B; expert self: the suffix part with F_ACC) -> QUANT_X O (o_proj input)
                     -> END. Every record passes contract.validate.
  script.txt         check lines (expected bytes after a record) and load lines (the SOFTMAX record's P and scales,
                     computed by the softmax unit's golden executor smx_emu.run_cmd on the expected scores)
  exp_*.bin          the expected bytes, from contract.py's reference arithmetic (numpy: quant_rows, pack_w,
                     gemm_out) independently of attn_int16; asserted equal, bit for bit, to
                     attn_int16.attention_contract (the function the full-policy check runs).

    C:\\kp\\lerobot-venv\\Scripts\\python.exe fpga\\kernels\\attention\\make_vectors.py [vectors/calls.pt]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import attn_int16 as A  # noqa: E402

CT = A.CT
C = CT.C
import smx_emu  # noqa: E402  (on sys.path through attn_int16)

D = 64
BASES = {
    "vision": 0x8_0000_0000,
    "prefix": 0xFF_0000_0000,
    "expert_self": 0x4000_0000,
    "expert_cross": 0xFF_8000_0000,
}


class Img:
    """Bump allocator over a byte image at a physical base."""

    def __init__(self, base: int):
        self.base, self.top, self.data = base, base, bytearray()

    def alloc(self, nbytes: int, align: int = 64) -> int:
        a = (self.top + align - 1) // align * align
        self.top = a + nbytes
        if len(self.data) < self.top - self.base:
            self.data.extend(bytes(self.top - self.base - len(self.data)))
        return a

    def put(self, addr: int, arr) -> None:
        b = np.ascontiguousarray(arr).tobytes()
        o = addr - self.base
        self.data[o : o + len(b)] = b

    def array(self) -> np.ndarray:
        return np.frombuffer(bytes(self.data), np.uint8).copy()


def build(site: str, rec: dict, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*"):
        f.unlink()
    q, k, v = rec["q"].float(), rec["k"].float(), rec["v"].float()
    mask = rec["mask"]
    assert q.shape[0] == 1 and float(rec["scale"]) == 0.125
    n_suf = q.shape[2] if site == "expert_self" else 0
    trec: dict = {}
    t_out = A.attention_contract(q, k, v, mask, site, n_suf, record=trec)  # the policy-check emulation
    if mask is None:
        qc, kc, vc = q[0], k[0], v[0]
    else:
        m = mask[0].bool()
        qv, kv = m.any(1), m.any(0)
        qc, kc, vc = q[0][:, qv], k[0][:, kv], v[0][:, kv]
    qc, kc, vc = qc.numpy(), kc.numpy(), vc.numpy()
    H, tq, _ = qc.shape
    hkv, tk = kc.shape[0], kc.shape[1]
    G = H // hkv
    tqp = CT.pad4(tq)
    NQ = G * tqp
    parts, mk = A.site_geometry(site, tk, n_suf)
    img = Img(BASES[site])
    lst = img.alloc(256 * 16, 4096)
    # ---------------- inputs: FP32 rows as the projections leave them
    rowq = (
        H * D + 2 * hkv * D
    )  # fused q | k | v row (prefix, vision); expert: q rows, and k | v rows per part
    if site in ("vision", "prefix"):
        qkv = np.concatenate(
            [
                qc.transpose(1, 0, 2).reshape(tq, H * D),
                kc.transpose(1, 0, 2).reshape(tk, hkv * D),
                vc.transpose(1, 0, 2).reshape(tk, hkv * D),
            ],
            1,
        ).astype(np.float32)
        a_qkv = img.alloc(qkv.nbytes)
        img.put(a_qkv, qkv)
        q_src = (a_qkv, 4 * rowq)
        kv_src = [(a_qkv + 4 * H * D, a_qkv + 4 * (H + hkv) * D, 4 * rowq)]
    else:
        qr = qc.transpose(1, 0, 2).reshape(tq, H * D).astype(np.float32)
        a_q = img.alloc(qr.nbytes)
        img.put(a_q, qr)
        q_src = (a_q, 4 * H * D)
        kv_src = []
        for k0, nk, _ in parts:
            kvr = np.concatenate(
                [
                    kc[:, k0 : k0 + nk].transpose(1, 0, 2).reshape(nk, hkv * D),
                    vc[:, k0 : k0 + nk].transpose(1, 0, 2).reshape(nk, hkv * D),
                ],
                1,
            ).astype(np.float32)
            a = img.alloc(kvr.nbytes)
            img.put(a, kvr)
            kv_src.append((a, a + 4 * hkv * D, 4 * 2 * hkv * D))
    # ---------------- buffers
    Qh, sQ = img.alloc(H * tqp * 2 * D), img.alloc(4 * H * tqp)
    kitem = (mk // 32) * D * 16
    kport = hkv * kitem
    KWw, KWs = img.alloc(4 * kport, 4096), img.alloc(4 * hkv * mk)
    VW = []
    for _, nk, _ in parts:
        npv = CT.pad32(nk)
        vitem = (D // 32) * npv * 16
        VW.append((img.alloc(4 * hkv * vitem, 4096), img.alloc(4 * hkv * D), vitem, hkv * vitem, npv))
    S = img.alloc(4 * hkv * NQ * mk)
    Pm, sP = img.alloc(2 * hkv * NQ * mk), img.alloc(4 * hkv * NQ)
    O = img.alloc(4 * tqp * H * D)
    Xo, sXo = img.alloc(2 * tq * H * D), img.alloc(4 * CT.pad4(tq))
    rq = rk = None
    if site != "vision":
        Dq, Ek = trec["ranks"]
        rq, rk = img.alloc(4 * NQ), img.alloc(4 * mk)
        img.put(rq, np.tile(Dq.numpy().astype(np.int32), G))
        img.put(rk, Ek.numpy().astype(np.int32))
    # ---------------- the list
    cmds: list[CT.Cmd] = []
    Sl = CT.Slot
    cmds.append(
        CT.Cmd(
            C.OP_QUANT_X,
            C.F_QOUT,
            0,
            0,
            H * D,
            tq,
            1,
            {"A": Sl(q_src[0], q_src[1]), "C": Sl(Qh, 2 * D), "F": Sl(sQ)},
            {
                C.P_SEG: 64,
                C.P_SEGC: tqp * 2 * D,
                C.P_SEGF: 4 * tqp,
                C.P_SHIFT: (-3) & 0xFFFFFFFF,
                C.P_NP: tqp,
            },
            "q",
        )
    )
    for (k0, nk, r0), (ka, va, pitch) in zip(parts, kv_src):
        cmds.append(
            CT.Cmd(
                C.OP_QUANT_W,
                0,
                0,
                0,
                hkv * D,
                nk,
                1,
                {"A": Sl(ka, pitch), "C": Sl(KWw + (r0 // 32) * D * 16, kport), "F": Sl(KWs + 4 * r0)},
                {C.P_SEG: 64, C.P_SEGC: kitem, C.P_SEGF: 4 * mk, C.P_NP: CT.pad32(nk)},
                f"k{r0}",
            )
        )
    for (k0, nk, r0), (ka, va, pitch), (vw, vs, vitem, vport, npv) in zip(parts, kv_src, VW):
        cmds.append(
            CT.Cmd(
                C.OP_QUANT_WT,
                0,
                0,
                0,
                hkv * D,
                nk,
                1,
                {"A": Sl(va, pitch), "C": Sl(vw, vport), "F": Sl(vs)},
                {C.P_SEG: 64, C.P_SEGC: vitem, C.P_SEGF: 4 * D, C.P_NP: npv},
                f"v{r0}",
            )
        )
    i_qk = len(cmds)
    cmds.append(
        CT.Cmd(
            C.OP_GEMM,
            0,
            0,
            mk,
            D,
            NQ,
            hkv,
            {
                "A": Sl(Qh, 2 * D, NQ * 2 * D),
                "B": Sl(KWw, kport, kitem),
                "C": Sl(S, 4 * mk, NQ * 4 * mk),
                "E": Sl(KWs, 0, 4 * mk),
                "F": Sl(sQ, 0, 4 * NQ),
            },
            {},
            "qk",
        )
    )
    i_smx = len(cmds)
    s_smx = {"A": Sl(S, 4 * mk, NQ * 4 * mk), "C": Sl(Pm, 2 * mk, NQ * 2 * mk), "F": Sl(sP, 0, 4 * NQ)}
    fl = C.F_QOUT
    if rq is not None:
        fl |= C.F_MASK
        s_smx["D"], s_smx["E"] = Sl(rq), Sl(rk)
    cmds.append(CT.Cmd(C.OP_SOFTMAX, fl, 0, 0, mk, NQ, hkv, s_smx, {C.P_EPS: CT.f32bits(1.0)}, "softmax"))
    for pi, ((k0, nk, r0), (vw, vs, vitem, vport, npv)) in enumerate(zip(parts, VW)):
        cmds.append(
            CT.Cmd(
                C.OP_GEMM,
                C.F_ACC if pi else 0,
                0,
                D,
                npv,
                NQ,
                hkv,
                {
                    "A": Sl(Pm + 2 * r0, 2 * mk, NQ * 2 * mk),
                    "B": Sl(vw, vport, vitem),
                    "C": Sl(O, 4 * H * D, G * 4 * D),
                    "E": Sl(vs, 0, 4 * D),
                    "F": Sl(sP, 0, 4 * NQ),
                },
                {C.P_SEGC: tqp, C.P_SEGF: 4 * D},
                f"pv{pi}",
            )
        )
    i_pv = len(cmds) - 1
    cmds.append(
        CT.Cmd(
            C.OP_QUANT_X,
            C.F_QOUT,
            0,
            0,
            H * D,
            tq,
            1,
            {"A": Sl(O, 4 * H * D), "C": Sl(Xo, 2 * H * D), "F": Sl(sXo)},
            {},
            "o.in",
        )
    )
    cmds.append(CT.Cmd(C.OP_END))
    for i, c in enumerate(cmds):
        c.tag = i
    bad = CT.validate_list(cmds)
    assert not bad, bad
    assert len(cmds) <= 16
    img.put(lst, np.frombuffer(CT.encode_list(cmds), np.uint8))
    # ---------------- expected bytes (contract.py, numpy)
    script = []

    def expect(after: int, name: str, addr: int, arr) -> None:
        b = np.ascontiguousarray(arr).tobytes()
        fn = f"exp_{name}.bin"
        (out / fn).write_bytes(b)
        script.append(f"check {after} {name} {addr:x} {len(b)} {fn}")

    # Q: head-major, pad rows zero / 1.0
    qp = np.zeros((H, tqp, D), np.float32)
    qp[:, :tq] = qc
    qq, sq = CT.quant_rows(qp, shift=-3)  # (H, tqp, D) int16, (H, tqp)
    expect(0, "Qh", Qh, qq)
    expect(0, "sQ.f32", sQ, sq.astype(np.float32))
    # K: KW blob
    kb = np.zeros((hkv, mk, D), np.float32)
    for k0, nk, r0 in parts:
        kb[:, r0 : r0 + nk] = kc[:, k0 : k0 + nk]
    kq, sk = CT.quant_rows(kb)
    blob = np.zeros((4, hkv, kitem // 2), np.uint16)
    for j in range(hkv):
        blob[:, j] = CT.pack_w(kq[j]).reshape(4, -1)
    i_kw = len(parts)  # the last QUANT_W record
    expect(i_kw, "KW", KWw, blob)
    expect(i_kw, "KWs.f32", KWs, sk.astype(np.float32))
    # V: VW blobs per part
    vparts = []
    for pi, ((k0, nk, r0), (vw, vs, vitem, vport, npv)) in enumerate(zip(parts, VW)):
        vt = vc[:, k0 : k0 + nk].transpose(0, 2, 1)  # (hkv, 64, nk)
        vq, sv = CT.quant_rows(vt)
        vqp = np.zeros((hkv, D, npv), np.int16)
        vqp[:, :, :nk] = vq
        vblob = np.zeros((4, hkv, vitem // 2), np.uint16)
        for j in range(hkv):
            vblob[:, j] = CT.pack_w(vqp[j]).reshape(4, -1)
        expect(len(parts) + 1 + pi, f"VW{pi}", vw, vblob)
        expect(len(parts) + 1 + pi, f"VWs{pi}.f32", vs, sv.astype(np.float32))
        vparts.append((vqp, sv))
    # Q.K^T
    xq = qq.reshape(hkv, NQ, D).astype(np.int64)
    sx = sq.reshape(hkv, NQ)
    scores = np.stack([CT.gemm_out(xq[j] @ kq[j].astype(np.int64).T, sk[j], sx[j]) for j in range(hkv)])
    expect(i_qk, "scores.f32", S, scores)
    # SOFTMAX: the softmax unit's golden executor on the expected scores
    mem = img.array()
    o = S - img.base
    mem[o : o + scores.nbytes] = np.frombuffer(scores.astype(np.float32).tobytes(), np.uint8)
    smx_emu.run_cmd(cmds[i_smx], mem, img.base)
    pbytes = mem[Pm - img.base : Pm - img.base + 2 * hkv * NQ * mk].copy()
    spbytes = mem[sP - img.base : sP - img.base + 4 * hkv * NQ].copy()
    (out / "smx_P.bin").write_bytes(pbytes.tobytes())
    (out / "smx_sP.bin").write_bytes(spbytes.tobytes())
    script.append(f"load {i_smx} {Pm:x} smx_P.bin")
    script.append(f"load {i_smx} {sP:x} smx_sP.bin")
    P = pbytes.view(np.int16).reshape(hkv, NQ, mk).astype(np.int64)
    sp = spbytes.view(np.float32).reshape(hkv, NQ)
    # P.V (suffix part: F_ACC)
    y = None
    for pi, ((k0, nk, r0), (vqp, sv)) in enumerate(zip(parts, vparts)):
        npv = vqp.shape[2]
        yp = np.stack(
            [
                CT.gemm_out(
                    P[j][:, r0 : r0 + npv] @ vqp[j].astype(np.int64).T,
                    sv[j],
                    sp[j],
                    addend=None if y is None else y[j],
                )
                for j in range(hkv)
            ]
        )
        y = yp
    Oexp = (
        y.reshape(hkv, G, tqp, D).transpose(2, 0, 1, 3).reshape(tqp, H * D)
    )  # row i, column (G j + g) 64 + c
    expect(i_pv, "O.f32", O, Oexp)
    xo, so = CT.quant_rows(Oexp[:tq])
    sop = np.ones(CT.pad4(tq), np.float32)
    sop[:tq] = so
    expect(i_pv + 1, "Xo", Xo, xo)
    expect(i_pv + 1, "sXo.f32", sXo, sop)
    # ---------------- the two emulations agree bit for bit
    t_o = trec["out"].numpy()  # (H, tq, D)
    same = dict(
        scores=bool(
            np.array_equal(trec["scores"].numpy().view(np.uint32), scores.astype(np.float32).view(np.uint32))
        ),
        P=bool(np.array_equal(trec["pq"].numpy().astype(np.int64), P))
        and bool(np.array_equal(trec["sp"].numpy()[..., 0].view(np.uint32), sp.view(np.uint32))),
        O=bool(
            np.array_equal(
                t_o.transpose(1, 0, 2).reshape(tq, H * D).view(np.uint32),
                Oexp[:tq].astype(np.float32).view(np.uint32),
            )
        ),
    )
    full = t_out[0].numpy() if mask is None else t_out[0][:, mask[0].bool().any(1)].numpy()
    same["policy_output"] = bool(np.array_equal(full.view(np.uint32), t_o.view(np.uint32)))
    print(f"{site}: contract.py chain == attn_int16.attention_contract (policy check) bit for bit: {same}")
    assert all(same.values()), same
    (out / "mem.bin").write_bytes(img.array().tobytes())
    (out / "meta.txt").write_text(f"{img.base:x} {lst:x} {img.top - img.base} {len(cmds)}\n")
    (out / "script.txt").write_text("\n".join(script) + "\n")
    info = dict(
        site=site,
        H=H,
        Hkv=hkv,
        G=G,
        Tq=tq,
        TqP=tqp,
        keys=[p[1] for p in parts],
        Mk=mk,
        base=hex(img.base),
        bytes=img.top - img.base,
        records=len(cmds),
        checks=sum(s.startswith("check") for s in script),
        ops=[C.opnames[c.op] + (f"/{c.note}" if c.note else "") for c in cmds],
    )
    print(f"  {out}: {json.dumps({k: v for k, v in info.items() if k != 'ops'})}")
    return info


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "vectors" / "calls.pt"
    calls = torch.load(src, weights_only=False)
    infos = [
        build(site, calls[site], HERE / "vectors" / site)
        for site in ("vision", "prefix", "expert_self", "expert_cross")
    ]
    (HERE / "vectors" / "cases.json").write_text(json.dumps(infos, indent=1))


if __name__ == "__main__":
    main()
