"""TinyPolicy: the exact network in build spec §5.3 (the FPGA contract).

5 x conv3x3/s2/p1 + ReLU (16, 32, 64, 96, 128 ch) on a 96x96 RGB image, HWC flatten, concat
10 aux values, FC 256 + ReLU, FC 256 + ReLU, FC 48 -> [8, 6] action chunk. 565,552 parameters.

Inputs are the integer encodings from ref/intref.py, rescaled by 2^-7 so float training sees
what QAT and the hardware will see:
  image  (pixel - 128) / 128
  aux    round_half_up(state * 127) / 128, one-hot entries 127 / 128
"""

import numpy as np
import torch
from torch import nn

CHANNELS = (3, 16, 32, 64, 96, 128)
AUX = 10
CHUNK, JOINTS = 8, 6


def encode_aux(state: np.ndarray, instr: np.ndarray) -> np.ndarray:
    """int8 aux exactly as spec §5.4: state round-half-up(x*127) clamped, one-hot 127."""
    s = np.clip(np.floor(np.asarray(state, np.float64) * 127 + 0.5), -127, 127)
    instr = np.asarray(instr)
    onehot = np.zeros((*instr.shape, 4))
    np.put_along_axis(onehot, instr[..., None].astype(np.int64), 127, axis=-1)
    return np.concatenate([s, onehot], -1).astype(np.int8)


class TinyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleList(
            nn.Conv2d(cin, cout, 3, stride=2, padding=1)
            for cin, cout in zip(CHANNELS, CHANNELS[1:], strict=False)
        )
        self.fc6 = nn.Linear(3 * 3 * 128 + AUX, 256)
        self.fc7 = nn.Linear(256, 256)
        self.fc8 = nn.Linear(256, CHUNK * JOINTS)
        for m in self.modules():
            if isinstance(m, nn.Conv2d | nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.fc8.weight, std=1e-3)

    @staticmethod
    def prep(image_u8: torch.Tensor, aux_i8: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """uint8 [B,96,96,3] HWC + int8 aux [B,10] -> float network inputs."""
        x = (image_u8.float() - 128.0) / 128.0
        return x.permute(0, 3, 1, 2).contiguous(), aux_i8.float() / 128.0

    def features(self, x: torch.Tensor) -> torch.Tensor:
        for conv in self.convs:
            x = torch.relu(conv(x))
        return x.permute(0, 2, 3, 1).flatten(1)  # HWC flatten, as the hardware does

    def forward(self, x: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        h = torch.cat([self.features(x), aux], 1)
        h = torch.relu(self.fc6(h))
        h = torch.relu(self.fc7(h))
        return self.fc8(h).view(-1, CHUNK, JOINTS)

    def act(self, image_u8: np.ndarray, state: np.ndarray, instr: int) -> np.ndarray:
        """One observation -> [8, 6] normalized action chunk (float32)."""
        dev = next(self.parameters()).device
        img = torch.from_numpy(np.ascontiguousarray(image_u8))[None].to(dev)
        aux = torch.from_numpy(encode_aux(state, np.array(instr)))[None].to(dev)
        with torch.no_grad():
            return self(*self.prep(img, aux))[0].clamp(-1, 1).cpu().numpy()
