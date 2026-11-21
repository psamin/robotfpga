"""Element-level error of the RoPE kernel (rope_emu.py, bit-exact with rope_core.h) against lerobot's FP32
apply_rope, over every apply_rope call of one -12392 replay frame (272 calls, 165,920 (token, head) rows).

Reported per output mode, relative to each row's max |torch result| (the scale the int16 formats use):
  f32    MODE_F32                      vs torch FP32
  int16  MODE_INT16 q * 2^e            vs torch FP32
  absmax torch FP32 -> int16 with s = amax/32767, q = round(x/s) (the format reference, no kernel)
plus the fraction of MODE_F32 outputs bit-equal to torch.

    set HF_HUB_OFFLINE=1
    C:\\kp\\lerobot-venv\\Scripts\\python.exe fpga\\kernels\\rope\\kernel_error.py [frame_index]
"""

from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rope_emu as R  # noqa: E402

ROOT = HERE.parents[2] / "build/ckpt/smolvla-12392"


def main():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import lerobot.policies.smolvla.smolvlm_with_expert as swe
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    fi = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    policy = SmolVLAPolicy.from_pretrained(str(ROOT / "smolvla-so101_medicine_bottle_pickplace-12392-native")).eval().float()
    fr = torch.load(sorted((ROOT / "batches-12392").glob("frame_*.pt"))[fi], weights_only=False)
    calls, orig = [], swe.apply_rope

    def hook(x, positions, max_wavelength=10_000):
        y = orig(x, positions, max_wavelength)
        calls.append((x.detach().clone(), positions.detach().clone(), y.detach().clone()))
        return y

    swe.apply_rope = hook
    try:
        with torch.no_grad():
            policy.predict_action_chunk(copy.deepcopy(fr["batch"]), noise=fr["noise"].clone())
    finally:
        swe.apply_rope = orig
    st = {k: dict(max=0.0, sq=0.0, n=0) for k in ("f32", "int16", "absmax")}
    rows, exact = 0, 0
    for x, pos, ref in calls:
        amax = ref.abs().amax(-1, keepdim=True).clamp_min(1e-30)
        outs = {"f32": R.kernel_from_natural(x, pos, "f32"), "int16": R.kernel_from_natural(x, pos, "int16", 0),
                "absmax": R.absmax_int16(ref)}
        for k, y in outs.items():
            e = ((y - ref).abs() / amax).double()
            st[k]["max"] = max(st[k]["max"], float(e.max()))
            st[k]["sq"] += float((e**2).sum())
            st[k]["n"] += e.numel()
        f = outs["f32"]
        exact += int((f == ref).sum())
        rows += x.shape[0] * x.shape[1] * x.shape[2]
    res = dict(frame=int(fr["frame"]), calls=len(calls), rows=rows, f32_bit_equal_to_torch=exact / (rows * 64))
    for k, v in st.items():
        res[f"{k}_max_err_over_row_amax"] = v["max"]
        res[f"{k}_rms_err_over_row_amax"] = (v["sq"] / v["n"]) ** 0.5
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
