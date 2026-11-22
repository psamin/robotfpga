"""Full-policy check of the qk_softmax fusion on -12392's 3 replay frames (the attention agent's method,
fpga/kernels/attention/check_policy.py, whose harness this imports).

Every attention of SmolVLA runs the contract's int16 arithmetic (attn_int16.attention_contract: Q with the 1/8 in its
scale, K per key, the GEMM epilogue, rank masks, V^T per channel, P.V int16); everything else stays torch FP32.
  contract  the contract 1.0 pair: scores = GEMM epilogue of Q.K^T for the whole item, then smx_emu.kernel
  qks       the fused command: qks_emu.qk_softmax (tile by tile of 64 rows, the scores never leave the tile)
            on the operands exactly as the QK_SOFTMAX record carries them (int16 Q rows + s_tok, KW rows + s_row,
            int32 rank_q per stacked row, rank_k), then the same P.V
On every call of the qks variant the fused P and scales are compared with the pair's, bit for bit, and the chunk of
each frame with the contract variant's. --dump-vec writes real items for tb_qks --vec.

    set HF_HUB_OFFLINE=1  (and HF_HUB_CACHE at a cache whose SmolVLM2 refs/main names 7b375e1...)
    C:\\kp\\lerobot-venv\\Scripts\\python.exe fpga\\kernels\\phaseB\\qk_softmax\\policy_check.py [--dump-vec results/vec]
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
KD = HERE.parents[1]
sys.path.insert(0, str(KD / "attention"))
sys.path.insert(0, str(HERE))
import check_policy as CPOL  # noqa: E402  (the attention agent's harness: patches, frames, ref)
import qks_emu  # noqa: E402

A = CPOL.A  # attn_int16
STATS = dict(calls=0, items=0, rows=0, p_mismatch=0, s_mismatch=0, dumped=0)
DUMP = dict(dir=None, per_site={}, limit=3)


def dump_item(site, j, xq, sx, kq, sk, rq, rk, q, s):
    d = DUMP["dir"]
    n = DUMP["per_site"].get(site, 0)
    if d is None or n >= DUMP["limit"]:
        return
    DUMP["per_site"][site] = n + 1
    N, K = xq.shape
    L = kq.shape[0]
    mask = rq is not None
    b = bytearray(struct.pack("<6If", 0x514B5331, L, K, N, int(mask), N, 1.0))
    b += xq.round().to(torch.int16).numpy().tobytes()
    b += sx.to(torch.float32).numpy().tobytes()
    b += kq.round().to(torch.int16).numpy().tobytes()
    b += sk.to(torch.float32).numpy().tobytes()
    if mask:
        b += rq.to(torch.int32).numpy().tobytes() + rk.to(torch.int32).numpy().tobytes()
    b += q[:N].to(torch.int16).numpy().tobytes()
    b += s[:N].numpy().astype(np.float32).tobytes()
    Path(d).mkdir(parents=True, exist_ok=True)
    (Path(d) / f"{site}_{n}_item{j}.bin").write_bytes(bytes(b))
    STATS["dumped"] += 1


def _attention_one_qks(q, k, v, mc, site, n_suffix, rec):
    """attn_int16._attention_one with the (Q.K^T GEMM, SOFTMAX) pair replaced by one QK_SOFTMAX per KV head."""
    H, tq, D = q.shape
    hkv, tk = k.shape[0], k.shape[1]
    G = H // hkv
    tqp = A.pad_to(tq, 4)
    parts, mk = A.site_geometry(site, tk, n_suffix)
    qp = torch.zeros(H, tqp, D)
    qp[:, :tq] = q
    qq, sq = A.quant_c(qp, -1, A.Q_SHIFT)
    x = qq.reshape(hkv, G * tqp, D)
    sx = sq.reshape(hkv, G * tqp, 1)
    kb = torch.zeros(hkv, mk, D)
    for k0, nk, r0 in parts:
        kb[:, r0 : r0 + nk] = k[:, k0 : k0 + nk]
    kq, sk = A.quant_c(kb, -1)
    if site == "vision":
        Dq = Ek = None
        valid = None
    else:
        Dq, Ek = A.contract_ranks(site, tq, tqp, [p[1] for p in parts], mk, [p[2] for p in parts])
        rv = A.smx_emu.rank_valid(Ek, Dq)
        cols = torch.cat([torch.arange(r0, r0 + nk) for _, nk, r0 in parts])
        assert torch.equal(rv[:tq][:, cols], mc), f"{site}: contract ranks != SmolVLA mask"
        valid = rv.repeat(G, 1)
    # the contract 1.0 pair, for the bitwise check
    scores = A.gemm_out(x, kq, sk.transpose(-1, -2), sx)
    pq_ref, sp_ref = A.smx_emu.kernel(scores, A.C_SMX, valid)
    # the fused command, one item per KV head, with the record's operands: rank_q of stacked row g * tqp + i = Dq[i]
    pq = torch.zeros(hkv, G * tqp, mk, dtype=torch.int64)
    sp = torch.zeros(hkv, G * tqp, 1)
    rq_rows = None if Dq is None else Dq.repeat(G)
    for j in range(hkv):
        qj, sj = qks_emu.qk_softmax(x[j], sx[j, :, 0], kq[j], sk[j, :, 0], rq_rows, Ek, 1.0)
        pq[j], sp[j, :, 0] = qj, sj
        dump_item(site, j, x[j], sx[j, :, 0], kq[j], sk[j, :, 0], rq_rows, Ek, qj, sj)
    STATS["calls"] += 1
    STATS["items"] += hkv
    STATS["rows"] += hkv * G * tqp
    STATS["p_mismatch"] += int((pq != pq_ref).sum())
    STATS["s_mismatch"] += int((sp.view(torch.int32) != sp_ref.view(torch.int32)).sum())
    pq = pq.to(torch.float32)
    y = None
    for k0, nk, r0 in parts:
        npv = A.pad_to(nk, 32)
        vt = v[:, k0 : k0 + nk].transpose(-1, -2)
        vq, sv = A.quant_c(vt, -1)
        vqp = torch.zeros(hkv, D, npv)
        vqp[:, :, :nk] = vq
        y = A.gemm_out(pq[:, :, r0 : r0 + npv], vqp, sv.transpose(-1, -2), sp, y)
    return y.reshape(hkv, G, tqp, D)[:, :, :tq].reshape(H, tq, D)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="0,1,2")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--dump-vec", default=None, help="write up to 3 real items per site (frame 0) for tb_qks --vec")
    ap.add_argument("--out", default=str(HERE / "results" / "policy_check.json"))
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    policy = SmolVLAPolicy.from_pretrained(str(CPOL.CKPT)).to("cpu").eval().float()
    CPOL.install(policy)
    files = sorted(CPOL.BATCHES.glob("frame_*.pt"))
    frames = [torch.load(files[int(i)], weights_only=False) for i in a.frames.split(",")]
    orig_one = A._attention_one
    res = dict(ckpt=CPOL.CKPT.name, gate=CPOL.GATE, frames=[])
    for fi, d in enumerate(frames):
        ref = d["ref_chunk_raw"].float()
        row = dict(frame=int(d["frame"]))
        CPOL.STATE["variant"] = CPOL.VARIANTS["contract"]
        CPOL.STATE["record"] = False
        A._attention_one = orig_one
        t0 = time.perf_counter()
        ch_c = CPOL.run_frame(policy, d)
        row["contract_s"] = round(time.perf_counter() - t0, 1)
        A._attention_one = _attention_one_qks
        DUMP["dir"] = a.dump_vec if fi == 0 else None
        before = dict(STATS)
        t0 = time.perf_counter()
        ch_q = CPOL.run_frame(policy, d)
        row["qks_s"] = round(time.perf_counter() - t0, 1)
        A._attention_one = orig_one
        row["contract_max_abs_err"] = float((ch_c - ref).abs().max())
        row["qks_max_abs_err"] = float((ch_q - ref).abs().max())
        row["qks_vs_contract_chunk_bit_equal"] = bool(torch.equal(ch_q.view(torch.int32), ch_c.view(torch.int32)))
        row["gate_pass"] = row["qks_max_abs_err"] <= CPOL.GATE
        row |= {k: STATS[k] - before[k] for k in STATS}
        res["frames"].append(row)
        print(json.dumps(row), flush=True)
    res["total"] = dict(STATS)
    res["worst_qks_max_abs_err"] = max(r["qks_max_abs_err"] for r in res["frames"])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res["total"]), f"wrote {a.out}", flush=True)
    return 0 if STATS["p_mismatch"] == 0 and STATS["s_mismatch"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
