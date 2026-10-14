"""Run the fine-tuned SmolVLA with every linear layer computed by the FPGA engine code (issue #148).

Full precision: weights in BF16 (as stored in the checkpoint), activations and arithmetic in FP32.
Every nn.Linear is replaced by a call into hls/stream/float_linear.h (built as a shared library,
the same core Vitis HLS synthesizes); bias and every nonlinear op stay in PyTorch FP32, as they
would on the KR260's ARM. Compares the action chunks with the plain FP32 PyTorch model on the
same frames and noise, and with the recorded actions. Also records every executed linear layer
(shape, calls, tokens) for the latency model and dumps real layers for tb_float.

    python scripts/smolvla_fpga.py --model ~/models/smolvla-<run> --out runs/smolvla-fpga --frames 5
"""

import argparse
import importlib.util
import json
import time
import types
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("pack_float", ROOT / "hls" / "stream" / "pack_float.py")
pf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pf)

RENAME = {
    "observation.images.front": "observation.images.camera1",
    "observation.images.wrist": "observation.images.camera2",
}


class Engine:
    def __init__(self, dump_dir: Path, max_dumps: int = 4):
        self.lib = pf.Lib()
        self.fpga = False
        self.stats = {}
        self.dump_dir, self.max_dumps, self.dumped = dump_dir, max_dumps, []


def patch(name: str, mod: nn.Linear, eng: Engine):
    w_bits = pf.to_bf16_bits(mod.weight.detach().float().cpu().numpy())
    mod._fpga_packed, mod._fpga_mp = pf.pack(w_bits)
    mod._fpga_bits = w_bits

    def fwd(self, x):
        k, m = self.in_features, self.out_features
        n = x.numel() // k
        st = eng.stats.setdefault(name, {"M": m, "K": k, "calls": 0, "tokens": 0})
        st["calls"] += 1
        st["tokens"] += n
        if not eng.fpga:
            return nn.functional.linear(x, self.weight, self.bias)
        x2 = x.detach().reshape(n, k).float().cpu().numpy()
        y = eng.lib(self._fpga_packed, self._fpga_mp, x2)[:, :m]
        if (
            len(eng.dumped) < eng.max_dumps
            and name not in eng.dumped
            and (m, k) not in [tuple(d[1]) for d in eng.dumped]
        ):
            eng.dumped.append((name, (m, k)))
            pf.write_case(eng.dump_dir / name.replace(".", "_"), self._fpga_bits, x2[:64])
        out = torch.from_numpy(y).to(x.device)
        if self.bias is not None:
            out = out + self.bias.float()
        return out.to(x.dtype).reshape(*x.shape[:-1], m)

    mod.forward = types.MethodType(fwd, mod)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="flyingturtleboop/so101_medicine_bottle_pickplace")
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=5)
    ap.add_argument("--exclude", type=int, nargs="*", default=[11, 14])
    args = ap.parse_args()
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    model_dir = str(Path(args.model).expanduser())

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    torch.set_num_threads(8)
    policy = SmolVLAPolicy.from_pretrained(model_dir)
    stored = sorted({str(p.dtype) for p in policy.parameters()})
    policy = policy.float().to("cpu").eval()
    pre, post = make_pre_post_processors(policy.config, pretrained_path=model_dir)
    eng = Engine(out / "layers")
    n_linear = 0
    for name, mod in policy.named_modules():
        if isinstance(mod, nn.Linear):
            patch(name, mod, eng)
            n_linear += 1
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"loaded: {n_params:,} params (stored {stored}), {n_linear} linear layers", flush=True)

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

    rows, chunk_stats = [], None
    g = torch.Generator().manual_seed(0)
    for fi, i in enumerate(picks):
        item = ds[i]
        noise = torch.randn((1, policy.config.chunk_size, policy.config.max_action_dim), generator=g)
        eng.fpga, eng.stats = False, {}
        t0 = time.perf_counter()
        a_ref = run(item, noise)
        t_ref = time.perf_counter() - t0
        if chunk_stats is None:
            chunk_stats = json.loads(json.dumps(eng.stats))
        eng.fpga = True
        t0 = time.perf_counter()
        a_fpga = run(item, noise)
        t_fpga = time.perf_counter() - t0
        gt = item["action"].float().numpy()[:6]
        rows.append(
            {
                "frame": int(i),
                "max_abs_diff_deg": float(np.abs(a_fpga - a_ref).max()),
                "mean_abs_diff_deg": float(np.abs(a_fpga - a_ref).mean()),
                "first_action_err_vs_recorded_ref_deg": float(np.abs(a_ref[0] - gt).mean()),
                "first_action_err_vs_recorded_fpga_deg": float(np.abs(a_fpga[0] - gt).mean()),
                "cpu_seconds_ref": round(t_ref, 2),
                "cpu_seconds_fpga_code": round(t_fpga, 2),
            }
        )
        print(json.dumps(rows[-1]), flush=True)

    layers = [{"name": k, **v} for k, v in chunk_stats.items()]
    summary = {
        "model": model_dir,
        "params_total": n_params,
        "stored_dtypes": stored,
        "linear_layers": n_linear,
        "frames": rows,
        "max_abs_diff_deg_all": max(r["max_abs_diff_deg"] for r in rows),
        "linear_weight_bytes_bf16": 2 * sum(L["M"] * L["K"] for L in layers),
        "linear_macs_per_chunk": sum(L["tokens"] * L["M"] * L["K"] for L in layers),
        "dumped_layers": [d[0] for d in eng.dumped],
        "layers": layers,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "layers"}, indent=1))


if __name__ == "__main__":
    main()
