"""Quantization-aware training with power-of-two scales, bit-exact with ref/intref.py (Phase 5).

Integer semantics (spec §5.4, ref/intref.py): int8 weights in [-127, 127], int32 biases in the
accumulator scale, one right shift per layer with round-half-up, clamp to [0, 127] (ReLU) or
[-127, 127] (last layer). Every tensor has a power-of-two scale 2^-e:

    layer k: input scale 2^-a[k-1], weight scale 2^-w[k], output scale 2^-a[k]
             shift s[k] = a[k-1] + w[k] - a[k]   (must be in [1, 30])

The fake-quant forward pass below computes exactly those integers (times their scales), with a
straight-through estimator for rounding [Bengio2013-STE, Jacob2018-IntOnly]. Exponents are
picked per tensor by minimizing quantization MSE on calibration data [Nagel2021-WP], then frozen
while weights fine-tune. Two places the contract is not a pure power of two, handled exactly:

  - FC6 input concatenates conv5's output (scale 2^-a[5]) with aux bytes. The hardware treats the
    1,162 bytes as one vector, so aux bytes are read at scale 2^-a[5] too. The float model saw
    aux/128, so FC6's 10 aux weight columns are rescaled by 2^(a[5]-7) once, at conversion.
  - Actions are int8/127 (spec), not int8 * 2^-a[8]. a[8] is fixed at 7 and FC8 is pre-scaled
    by 127/128 so the float model's output maps onto the int8 range.
"""

import importlib.util
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from armlab.policy.tiny import TinyPolicy

REF = Path(__file__).resolve().parents[2] / "ref" / "intref.py"
IN_EXP = 7  # image int = pixel - 128 -> float (pixel - 128) / 128; aux bytes / 128 in the float model
OUT_EXP = 7


def load_intref():
    spec = importlib.util.spec_from_file_location("intref", REF)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def round_half_up(t: torch.Tensor) -> torch.Tensor:
    return torch.floor(t + 0.5)


def ste_round(t: torch.Tensor) -> torch.Tensor:
    return t + (round_half_up(t) - t).detach()


def quant_w(w: torch.Tensor, e: int) -> torch.Tensor:
    return torch.clamp(ste_round(w * 2.0**e), -127, 127) * 2.0**-e


def quant_act(t: torch.Tensor, e: int, relu: bool) -> torch.Tensor:
    return torch.clamp(ste_round(t * 2.0**e), 0 if relu else -127, 127) * 2.0**-e


def best_exp(t: torch.Tensor, lo: int, hi: int, relu: bool | None, candidates=range(-4, 20)) -> int:
    """Power-of-two exponent minimizing quantization MSE of t (weights: relu=None, symmetric)."""
    t = t.detach().cpu().double().flatten()
    best = None
    for e in candidates:
        q = torch.clamp(round_half_up(t * 2.0**e), lo, hi) * 2.0**-e
        err = torch.mean((q - t) ** 2).item()
        if best is None or err < best[0]:
            best = (err, e)
    return best[1]


class QTiny(nn.Module):
    """Fake-quant TinyPolicy. Parameters are float; forward reproduces intref's integers."""

    def __init__(self, float_model: TinyPolicy, a_exp: list[int], w_exp: list[int]):
        super().__init__()
        self.net = float_model
        self.a = list(a_exp)  # a[0] = input exponent (7), a[1..8] layer outputs
        self.w = list(w_exp)  # w[1..8]; w[0] unused
        for k in range(1, 9):
            s = self.a[k - 1] + self.w[k] - self.a[k]
            if not 1 <= s <= 30:
                raise ValueError(f"layer {k}: shift {s} out of [1, 30] (a={self.a}, w={self.w})")

    def layers(self):
        return [*self.net.convs, self.net.fc6, self.net.fc7, self.net.fc8]

    def shifts(self) -> list[int]:
        return [self.a[k - 1] + self.w[k] - self.a[k] for k in range(1, 9)]

    def forward(self, image_u8: torch.Tensor, aux_i8: torch.Tensor) -> torch.Tensor:
        """Returns the action chunk as int8_out / 127, shape [B, 8, 6]."""
        x = (image_u8.to(self.net.fc6.weight.dtype) - 128.0).permute(0, 3, 1, 2) * 2.0**-IN_EXP
        for k, conv in enumerate(self.net.convs, start=1):
            x = self._layer(x, conv, k, relu=True)
        x = x.permute(0, 2, 3, 1).flatten(1)
        x = torch.cat([x, aux_i8.to(x.dtype) * 2.0 ** -self.a[5]], 1)
        x = self._layer(x, self.net.fc6, 6, relu=True)
        x = self._layer(x, self.net.fc7, 7, relu=True)
        x = self._layer(x, self.net.fc8, 8, relu=False)
        return (x * 2.0**OUT_EXP / 127.0).view(-1, 8, 6)

    def _layer(self, x, mod, k, relu):
        e_in, e_w, e_out = self.a[k - 1], self.w[k], self.a[k]
        wq = quant_w(mod.weight, e_w)
        bq = ste_round(mod.bias * 2.0 ** (e_in + e_w)) * 2.0 ** -(e_in + e_w)
        acc = F.conv2d(x, wq, bq, stride=2, padding=1) if isinstance(mod, nn.Conv2d) else F.linear(x, wq, bq)
        return quant_act(acc, e_out, relu)

    # ---- export to the handoff format ----
    def to_intref(self, intref):
        layers, params = [], []
        for k, (spec, mod) in enumerate(zip(intref.LAYERS, self.layers(), strict=True), start=1):
            e_in, e_w = self.a[k - 1], self.w[k]
            w = torch.clamp(round_half_up(mod.weight.detach().double() * 2.0**e_w), -127, 127)
            if isinstance(mod, nn.Conv2d):
                w = w.permute(0, 2, 3, 1)  # torch [out][in][ky][kx] -> contract [out][ky][kx][in]
            b = round_half_up(mod.bias.detach().double() * 2.0 ** (e_in + e_w))
            if b.abs().max() >= 2**31:
                raise OverflowError(f"layer {k} bias does not fit int32")
            params.append((w.cpu().numpy().astype(np.int8), b.cpu().numpy().astype(np.int32)))
            layers.append(dict(spec, shift=self.shifts()[k - 1]))
        return intref.Model(layers, params)


@torch.no_grad()
def convert(float_model: TinyPolicy, image_u8: torch.Tensor, aux_i8: torch.Tensor) -> QTiny:
    """Calibrate exponents layer by layer on a batch, with the quantized prefix feeding each layer."""
    net = TinyPolicy().to(image_u8.device)
    net.load_state_dict(float_model.state_dict())
    net.fc8.weight.mul_(127 / 128)
    net.fc8.bias.mul_(127 / 128)
    a, w = [IN_EXP], [0]
    x = (image_u8.float() - 128.0).permute(0, 3, 1, 2) * 2.0**-IN_EXP
    mods = [*net.convs, net.fc6, net.fc7, net.fc8]
    for k, mod in enumerate(mods, start=1):
        if k == 6:
            x = x.permute(0, 2, 3, 1).flatten(1)
            mod.weight[:, -10:].mul_(2.0 ** (a[5] - IN_EXP))  # aux bytes now read at conv5's scale
            x = torch.cat([x, aux_i8.float() * 2.0 ** -a[5]], 1)
        w.append(best_exp(mod.weight, -127, 127, None))
        wq = quant_w(mod.weight, w[k])
        acc = F.conv2d(x, wq, mod.bias, stride=2, padding=1) if k <= 5 else F.linear(x, wq, mod.bias)
        relu = k < 8
        if k == 8:
            e_out = OUT_EXP
        else:
            # shift >= 1 means a[k] <= a[k-1] + w[k] - 1
            hi = a[k - 1] + w[k] - 1
            e_out = best_exp(torch.relu(acc), 0, 127, True, candidates=range(-4, hi + 1))
        a.append(e_out)
        x = quant_act(acc, e_out, relu)
    return QTiny(net, a, w)


def qat_state(q: QTiny) -> dict:
    return {"model": q.net.state_dict(), "a_exp": q.a, "w_exp": q.w}


def load_qat(path: str, device: str = "cpu") -> QTiny:
    ck = torch.load(path, map_location=device)
    net = TinyPolicy().to(device)
    net.load_state_dict(ck["model"])
    return QTiny(net, ck["a_exp"], ck["w_exp"])
