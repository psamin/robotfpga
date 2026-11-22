"""Merge emit_act onto emit_amax's patched files (both phase-B patches start from the same snapshot of
fpga/hls/gemm_contract.h, sha256 1bf960e7...). Writes build/merge/{hls,kernels}/ (repo-shaped, with emit_act.h,
emit_amax.h, gelu_silu and quantize copied in) for tb_combo.cpp and the merged csynth, and prints the sha256 of the
emit_amax inputs it used (that agent's work is in progress: re-run this after it changes).

The edits are exactly the emit_act diff (diffs/gemm_contract.diff, diffs/contract_h.diff) except where emit_amax
rewrote the same lines:
  - store (emit_amax: one io word a clock, w = 0..YU-1 per (token, group)): the group's word count is it.yu and
    its column offset g * it.yu * 16 (a select of two shifts, no multiplier);
  - the GEMM parameter whitelist in validate gains P_ACT next to emit_amax's P_SHIFT / P_AMX_SG / P_XQ.

    C:/kp/lerobot-venv/Scripts/python.exe fpga/kernels/phaseB/emit_act/merge_emit_amax.py
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
KER = HERE.parents[1]
AMX = HERE.parent / "emit_amax"
OUT = HERE / "build" / "merge"


def rd(p: Path) -> tuple[str, str]:
    b = p.read_bytes()
    s = b.decode()
    return s.replace("\r\n", "\n"), ("\r\n" if b"\r\n" in b else "\n")


def rep(s: str, old: str, new: str, what: str) -> str:
    n = s.count(old)
    assert n == 1, f"{what}: anchor found {n} times"
    return s.replace(old, new)


def section(s: str, start: str, end: str) -> str:
    a = s.index(start)
    return s[a : s.index(end, a)]


def main() -> None:
    srcs = [AMX / "patched/hls/gemm_contract.h", AMX / "patched/kernels/contract.h", AMX / "emit_amax.h"]
    for p in srcs:
        print(hashlib.sha256(p.read_bytes()).hexdigest(), p.relative_to(KER.parents[1]))
    mine, _ = rd(HERE / "patched/hls/gemm_contract.h")
    g, gnl = rd(srcs[0])
    g = rep(
        g,
        '#include "../kernels/phaseB/emit_amax/emit_amax.h"\n',
        '#include "../kernels/phaseB/emit_amax/emit_amax.h"\n#include "../kernels/phaseB/emit_act/emit_act.h"\n',
        "include",
    )
    g = rep(
        g,
        "//   store    ys -> io                   row-major Y, one 128-byte burst per (token, group) (E-3)\n",
        "//   store    ys -> io                   row-major Y, one 128-byte burst per (token, group) (E-3); EPI_SILU_MUL:\n"
        "//                                       64 bytes (16 output columns per group)\n"
        "// Phase B emit_act (fpga/kernels/phaseB/emit_act/emit_act.h): p6 = P_ACT, GELU / SILU / SILU_MUL applied\n"
        "//   in dequant_ep (2 values a clock); with P_AMX_SG the absmax is taken over the activated values.\n",
        "header",
    )
    g = rep(
        g,
        "  bool bias, addd;\n",
        "  bool bias, addd;\n  int act;  // p6 = P_ACT: ctr::EPI_* (0 = none)\n"
        "  int yu;   // io words of Y per (token, group): YU, or YU / 2 for EPI_SILU_MUL\n",
        "Item",
    )
    g = rep(
        g,
        "  it.bias = (c.flags & ctr::F_BIAS) != 0;\n",
        "  it.bias = (c.flags & ctr::F_BIAS) != 0;\n  it.act = (int)c.p[ctr::P_ACT];\n"
        "  it.yu = it.act == ea::ACT_SILU_MUL ? YU / 2 : YU;\n",
        "plan_item",
    )
    p6 = "// ---------------------------------------------------------------- process 6"
    p7 = "// ---------------------------------------------------------------- process 7"
    g = rep(g, section(g, p6, p7), section(mine, p6, p7), "dequant_ep")
    g = rep(
        g,
        "        if (w == 0) base = (it.ya + rw.off() + (uint64_t)g * (LANES * 4)) >> 4;\n",
        "        if (w == 0) base = (it.ya + rw.off() + ((uint64_t)g << (it.yu == YU ? 7 : 6))) >> 4;  // g * yu * 16\n",
        "store offset",
    )
    g = rep(g, "        if (w == YU - 1) {\n", "        if (w == it.yu - 1) {\n", "store words")
    c, cnl = rd(srcs[1])
    c = rep(
        c,
        "                                  // p6: GEMM P_ACT (phase B emit_act)\n",
        "constexpr uint32_t P_ACT = 6;     // GEMM (phase B emit_act): activation of the epilogue output, EPI_* (0 = none)\n",
        "P_ACT",
    )
    ch, _ = rd(HERE / "patched/kernels/contract.h")
    epi = section(ch, "// GEMM p6 = P_ACT (phase B emit_act", "\n// ---------------------------------------------------------------- decoded record")
    c = rep(
        c,
        "\n// ---------------------------------------------------------------- decoded record",
        "\n" + epi.rstrip("\n") + "\n\n// ---------------------------------------------------------------- decoded record",
        "EPI",
    )
    c = rep(
        c,
        "    if (R > N || (R == 0 && P1 != 0) || P1 % ALIGN) return ME_PARAM;\n",
        "    if (R > N || (R == 0 && P1 != 0) || P1 % ALIGN) return ME_PARAM;\n"
        "    const uint32_t act = p[P_ACT];\n"
        "    if (act > EPI_MAX || (act && R)) return ME_PARAM;  // no C row groups with an activation\n"
        "    if (act && (acc || (act == EPI_SILU_MUL && addd))) return ME_FLAGS;\n"
        "    const uint64_t MO = act == EPI_SILU_MUL ? M / 2 : M;  // output columns\n",
        "validate act",
    )
    c = rep(c, "i != P_SHIFT && i != P_AMX_SG && i != P_XQ && p[i]", "i != P_SHIFT && i != P_AMX_SG && i != P_XQ && i != P_ACT && p[i]", "whitelist")
    c = rep(c, "(N > 1 && R != 1 && y.pitch < 4 * M)) return ME_ALIGN;", "(N > 1 && R != 1 && y.pitch < 4 * MO)) return ME_ALIGN;", "C pitch")
    c = rep(
        c,
        "(in_grp - 1) * (uint64_t)y.pitch + 4 * M > ADDR_LIMIT))",
        "(in_grp - 1) * (uint64_t)y.pitch + 4 * MO > ADDR_LIMIT))",
        "C extent",
    )
    if OUT.exists():
        shutil.rmtree(OUT)
    for d in ("hls", "kernels/phaseB/emit_act", "kernels/phaseB/emit_amax", "kernels/gelu_silu", "kernels/quantize"):
        (OUT / d).mkdir(parents=True)
    for f in (HERE / "snap/hls").glob("*.h"):
        shutil.copy(f, OUT / "hls")
    shutil.copy(HERE / "snap/hls/tb_gemm.cpp", OUT / "hls")
    (OUT / "hls/gemm_contract.h").write_bytes(g.replace("\n", gnl).encode())
    (OUT / "kernels/contract.h").write_bytes(c.replace("\n", cnl).encode())
    shutil.copy(HERE / "emit_act.h", OUT / "kernels/phaseB/emit_act")
    shutil.copy(AMX / "emit_amax.h", OUT / "kernels/phaseB/emit_amax")
    for k in ("gelu_silu", "quantize"):
        for f in (KER / k).glob("*.h"):
            shutil.copy(f, OUT / "kernels" / k)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
