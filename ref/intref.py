"""Bit-exact int8 reference for the tiny policy (numpy only).

STAND-IN: the canonical copy is the software side's `ref/intref.py`. This one is
written from the spec so hardware work can start now. When theirs lands, run
both on the same vectors; any difference is a spec question, not a hardware bug.

Spec summary (build spec section 5.4, FPGA brief section 5):
  input packet  27,658 bytes: image uint8 HWC 96x96x3, then 10 aux int8
  image prep    x = pixel - 128 (int8), padding value 0
  conv          3x3, stride 2, pad 1, weights [out][ky][kx][in], ReLU
  fc            weights [out][in]
  requantize    y = (acc + (1 << (s - 1))) >> s, arithmetic shift
  clamp         ReLU layers [0, 127], last layer [-127, 127]
  flatten       HWC order, then the 10 aux bytes appended
  output        48 int8, index = step * 6 + joint
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

IMG_H, IMG_W, IMG_C = 96, 96, 3
IMG_BYTES = IMG_H * IMG_W * IMG_C  # 27,648
AUX_LEN = 10
PACKET_BYTES = IMG_BYTES + AUX_LEN  # 27,658
OUT_LEN = 48

# The architecture contract. in/out shapes are HWC for convs, flat for FCs.
LAYERS = [
    {"name": "conv1", "type": "conv", "in_shape": [96, 96, 3], "out_shape": [48, 48, 16], "relu": True},
    {"name": "conv2", "type": "conv", "in_shape": [48, 48, 16], "out_shape": [24, 24, 32], "relu": True},
    {"name": "conv3", "type": "conv", "in_shape": [24, 24, 32], "out_shape": [12, 12, 64], "relu": True},
    {"name": "conv4", "type": "conv", "in_shape": [12, 12, 64], "out_shape": [6, 6, 96], "relu": True},
    {"name": "conv5", "type": "conv", "in_shape": [6, 6, 96], "out_shape": [3, 3, 128], "relu": True},
    {"name": "fc6", "type": "fc", "in_shape": [1162], "out_shape": [256], "relu": True},
    {"name": "fc7", "type": "fc", "in_shape": [256], "out_shape": [256], "relu": True},
    {"name": "fc8", "type": "fc", "in_shape": [256], "out_shape": [48], "relu": False},
]


def weight_shape(layer: dict) -> tuple[int, ...]:
    if layer["type"] == "conv":
        return (layer["out_shape"][2], 3, 3, layer["in_shape"][2])
    return (layer["out_shape"][0], layer["in_shape"][0])


def n_out_channels(layer: dict) -> int:
    return layer["out_shape"][-1]


def requant(acc: np.ndarray, s: int, lo: int, hi: int) -> np.ndarray:
    """Round-half-up shift then saturate. acc must be a signed integer array."""
    if not 1 <= s <= 30:
        raise ValueError(f"shift {s} out of range; the rounding term needs s >= 1")
    acc = acc.astype(np.int64)
    y = (acc + (1 << (s - 1))) >> s  # numpy >> on signed ints is arithmetic
    return np.clip(y, lo, hi).astype(np.int8)


def _check_int32(acc: np.ndarray, where: str) -> None:
    lo, hi = int(acc.min()), int(acc.max())
    if lo < -(2**31) or hi + (1 << 30) >= 2**31:  # leave room for the rounding term
        raise OverflowError(f"{where}: accumulator range [{lo}, {hi}] does not fit int32")


def conv_acc(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    """x int8 [H, W, Ci], w int8 [Co, 3, 3, Ci], b int32 [Co] -> accumulators [H/2, W/2, Co]."""
    h, wd, _ = x.shape
    ho, wo = h // 2, wd // 2
    xp = np.zeros((h + 2, wd + 2, x.shape[2]), dtype=np.int64)
    xp[1:-1, 1:-1] = x
    acc = np.broadcast_to(b.astype(np.int64), (ho, wo, w.shape[0])).copy()
    for ky in range(3):
        for kx in range(3):
            # output (oy, ox) reads input (2*oy + ky - 1, 2*ox + kx - 1); +1 for the pad
            patch = xp[ky : ky + 2 * ho : 2, kx : kx + 2 * wo : 2, :]
            acc += patch @ w[:, ky, kx, :].astype(np.int64).T
    _check_int32(acc, "conv")
    return acc


def fc_acc(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    """x int8 [In], w int8 [Out, In], b int32 [Out] -> accumulators [Out]."""
    acc = w.astype(np.int64) @ x.astype(np.int64) + b.astype(np.int64)
    _check_int32(acc, "fc")
    return acc


def conv3x3s2(x: np.ndarray, w: np.ndarray, b: np.ndarray, s: int, relu: bool = True) -> np.ndarray:
    return requant(conv_acc(x, w, b), s, 0 if relu else -127, 127)


def fc(x: np.ndarray, w: np.ndarray, b: np.ndarray, s: int, relu: bool) -> np.ndarray:
    return requant(fc_acc(x, w, b), s, 0 if relu else -127, 127)


def unpack_input(packet: bytes | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = np.frombuffer(bytes(packet), dtype=np.uint8)
    if p.size != PACKET_BYTES:
        raise ValueError(f"packet is {p.size} bytes, expected {PACKET_BYTES}")
    img = (p[:IMG_BYTES].astype(np.int16) - 128).astype(np.int8).reshape(IMG_H, IMG_W, IMG_C)
    aux = p[IMG_BYTES:].view(np.int8).copy()
    return img, aux


def forward(packet: bytes | np.ndarray, model: "Model") -> tuple[np.ndarray, list[np.ndarray]]:
    """Returns (48 int8 outputs, list of the 8 per-layer int8 outputs)."""
    x, aux = unpack_input(packet)
    outs = []
    for layer, (w, b) in zip(model.layers, model.params):
        if layer["type"] == "conv":
            x = conv3x3s2(x, w, b, layer["shift"], layer["relu"])
        else:
            if layer["name"] == "fc6":
                x = np.concatenate([x.reshape(-1), aux])  # HWC flatten + aux
            x = fc(x, w, b, layer["shift"], layer["relu"])
        outs.append(x)
    return x.reshape(-1), outs


class Model:
    """Layer specs (with shifts) plus int8 weights and int32 biases."""

    def __init__(self, layers: list[dict], params: list[tuple[np.ndarray, np.ndarray]]):
        self.layers = layers
        self.params = params

    @classmethod
    def load(cls, directory: str | Path) -> "Model":
        d = Path(directory)
        manifest = json.loads((d / "manifest.json").read_text())
        blob = (d / manifest["weights_file"]).read_bytes()
        digest = hashlib.sha256(blob).hexdigest()
        if digest != manifest["sha256"]:
            raise ValueError(f"weights.bin sha256 {digest} != manifest {manifest['sha256']}")
        params = []
        for layer in manifest["layers"]:
            shape = tuple(layer["weight_shape"])
            w = np.frombuffer(blob, np.int8, int(np.prod(shape)), layer["weight_offset"]).reshape(shape)
            b = np.frombuffer(blob, "<i4", n_out_channels(layer), layer["bias_offset"]).astype(np.int32)
            params.append((w, b))
        return cls(manifest["layers"], params)

    def save(self, directory: str | Path, note: str = "") -> dict:
        """Writes weights.bin and manifest.json. Returns the manifest."""
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        blob = bytearray()
        layers = []
        for layer, (w, b) in zip(self.layers, self.params):
            assert w.dtype == np.int8 and w.shape == weight_shape(layer), layer["name"]
            assert int(w.min()) >= -127, f"{layer['name']}: weights must be in [-127, 127]"
            entry = dict(layer)
            entry["weight_shape"] = list(w.shape)
            entry["weight_offset"] = len(blob)
            entry["weight_bytes"] = w.size
            blob += w.tobytes()
            entry["bias_offset"] = len(blob)
            entry["bias_bytes"] = b.size * 4
            blob += b.astype("<i4").tobytes()
            layers.append(entry)
        (d / "weights.bin").write_bytes(bytes(blob))
        manifest = {
            "format_version": 1,
            "note": note,
            "weights_file": "weights.bin",
            "total_bytes": len(blob),
            "sha256": hashlib.sha256(blob).hexdigest(),
            "input": {"packet_bytes": PACKET_BYTES, "image_shape": [IMG_H, IMG_W, IMG_C], "aux_len": AUX_LEN},
            "output": {"len": OUT_LEN, "shape": [8, 6]},
            "layers": layers,
        }
        (d / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return manifest
