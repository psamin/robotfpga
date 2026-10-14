"""SmolVLA with its linear layers computed the way the FPGA linear mode would (issue #148).

Every nn.Linear runs as: int8 weights (symmetric, one scale per output channel), int8 activations
(symmetric, one scale per token), exact int32 accumulation (hls/stream linear_stream), then the
float rescale + bias on the CPU. Nonlinear ops (softmax, norms, activations, RoPE) stay in float,
as they would on the KR260's ARM cores. Compares action chunks against the float model and the
recorded actions on real frames, records every executed linear layer for the latency model, and
dumps a few real layers for C simulation of the engine.

    python scripts/smolvla_int8.py --model ~/models/smolvla-... --out runs/smolvla-int8 --frames 40
"""

import argparse
import importlib.util
import json
import math
import time
import types
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("pack", ROOT / "hls" / "stream" / "pack.py")
pack = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pack)

RENAME = {
    "observation.images.front": "observation.images.camera1",
    "observation.images.wrist": "observation.images.camera2",
}


class Recorder:
    """Per-linear stats during a forward pass, and the int8 emulation switch."""

    def __init__(self):
        self.int8 = False
        self.calls = {}  # name -> {"M", "K", "calls", "tokens"}
        self.dump_dir = None
        self.dumped = set()
        self.max_dumps = 4


def quant_rows(t: torch.Tensor):
    """Symmetric int8 per row: returns int-valued float64 tensor and per-row scale."""
    scale = t.abs().amax(1).clamp(min=1e-12) / 127.0
    q = torch.round(t / scale[:, None]).clamp(-127, 127)
    return q.double(), scale


def make_forward(name, rec: Recorder):
    def fwd(self, x):
        k = self.in_features
        m = self.out_features
        n = x.numel() // k
        st = rec.calls.setdefault(name, {"M": m, "K": k, "calls": 0, "tokens": 0})
        st["calls"] += 1
        st["tokens"] += n
        if not rec.int8:
            return nn.functional.linear(x, self.weight, self.bias)
        if not hasattr(self, "_wq"):
            self._wq, self._sw = quant_rows(self.weight.detach().float())
        xq, sx = quant_rows(x.detach().reshape(n, k).float())
        acc = xq @ self._wq.T  # exact: |acc| <= K * 127 * 127 < 2^31 << 2^53
        y = acc.float() * sx[:, None] * self._sw[None, :]
        if self.bias is not None:
            y = y + self.bias.float()
        if (
            rec.dump_dir
            and name not in rec.dumped
            and len(rec.dumped) < rec.max_dumps
            and m % 16 == 0
            and k <= 4096
        ):
            shape = (m, k)
            if shape not in {tuple(v) for v in getattr(rec, "dumped_shapes", [])}:
                rec.dumped_shapes = getattr(rec, "dumped_shapes", []) + [shape]
                rec.dumped.add(name)
                d = rec.dump_dir / name.replace(".", "_")
                pack.write_linear_case(
                    d, self._wq.cpu().numpy().astype(np.int8), xq[: min(n, 128)].cpu().numpy().astype(np.int8)
                )
        return y.to(x.dtype).reshape(*x.shape[:-1], m)

    return fwd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="flyingturtleboop/so101_medicine_bottle_pickplace")
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--exclude", type=int, nargs="*", default=[11, 14])
    args = ap.parse_args()
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    policy = SmolVLAPolicy.from_pretrained(str(Path(args.model).expanduser())).to(dev).eval()
    pre, post = make_pre_post_processors(policy.config, pretrained_path=str(Path(args.model).expanduser()))
    rec = Recorder()
    for name, mod in policy.named_modules():
        if isinstance(mod, nn.Linear):
            mod.forward = types.MethodType(make_forward(name, rec), mod)
    n_params = sum(p.numel() for p in policy.parameters())

    ds = LeRobotDataset(args.dataset)
    ep = np.asarray(ds.hf_dataset["episode_index"])
    idx = [i for i in range(len(ds)) if int(ep[i]) not in args.exclude]
    picks = [idx[j] for j in np.linspace(0, len(idx) - 1, args.frames).astype(int)]

    def run(item, noise):
        batch = {RENAME.get(k, k): v for k, v in item.items() if k.startswith("observation.")}
        batch["task"] = item["task"]
        batch = pre(batch)
        policy.reset()
        with torch.no_grad():
            chunk = policy.predict_action_chunk(batch, noise=noise)
        return post(chunk).float().cpu().numpy().reshape(-1, chunk.shape[-1])[:, :6]

    errs_q, errs_f_gt, errs_q_gt, t_float, t_int8 = [], [], [], [], []
    g = torch.Generator(device=dev).manual_seed(0)
    for fi, i in enumerate(picks):
        item = ds[i]
        noise = torch.randn(
            (1, policy.config.chunk_size, policy.config.max_action_dim), generator=g, device=dev
        )
        rec.int8, rec.calls = False, rec.calls if fi else {}
        t0 = time.perf_counter()
        a_f = run(item, noise)
        t_float.append(time.perf_counter() - t0)
        if fi == 0:
            float_calls = json.loads(json.dumps(rec.calls))  # executed linears for one chunk
        rec.int8 = True
        rec.dump_dir = out / "layers" if fi == 0 else None
        t0 = time.perf_counter()
        a_q = run(item, noise)
        t_int8.append(time.perf_counter() - t0)
        gt = item["action"].float().numpy()[:6]
        errs_q.append(np.abs(a_q - a_f).mean(0))
        errs_f_gt.append(np.abs(a_f[0] - gt))
        errs_q_gt.append(np.abs(a_q[0] - gt))
        print(f"frame {fi + 1}/{len(picks)}: |int8 - float| mean {errs_q[-1].mean():.3f}", flush=True)

    # latency model inputs: executed linears for one action chunk (prefix + 10 flow steps)
    layers = [{"name": k, **v} for k, v in float_calls.items()]
    w_bytes = sum(L["M"] * L["K"] for L in layers)
    moved = sum(L["calls"] * math.ceil(L["tokens"] / L["calls"] / 64) * L["M"] * L["K"] for L in layers)
    macs = sum(L["tokens"] * L["M"] * L["K"] for L in layers)
    summary = {
        "model": args.model,
        "params_total": n_params,
        "frames": len(picks),
        "linear_layers_executed": len(layers),
        "linear_weight_bytes_int8": w_bytes,
        "weight_bytes_streamed_per_chunk_ntile64": moved,
        "linear_macs_per_chunk": macs,
        "action_mae_int8_vs_float_deg": np.mean(errs_q, 0).round(4).tolist(),
        "action_mae_int8_vs_float_deg_mean": float(np.mean(errs_q)),
        "first_action_mae_vs_recorded_float_deg": float(np.mean(errs_f_gt)),
        "first_action_mae_vs_recorded_int8_deg": float(np.mean(errs_q_gt)),
        "gpu_seconds_per_chunk_float_p50": float(np.median(t_float)),
        "dumped_layers": sorted(rec.dumped),
        "layers": layers,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "layers"}, indent=1))


if __name__ == "__main__":
    main()
