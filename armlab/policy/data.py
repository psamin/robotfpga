"""Demonstration shards -> (image, aux, 8-step action chunk, pad mask) tensors.

The whole dataset is held as uint8/int8 tensors (on the GPU when there is one) and batches are
indexed directly, so training is not bound by decoding or a DataLoader.
"""

import json
from pathlib import Path

import numpy as np
import torch

from armlab.policy.tiny import CHUNK, encode_aux


class DemoSet:
    def __init__(self, root: str | Path, device: str = "cpu", episodes: slice | None = None):
        shards = sorted(Path(root).glob("shard_*.npz"))
        if not shards:
            raise FileNotFoundError(f"no shard_*.npz in {root}")
        imgs, auxs, acts, lens, metas = [], [], [], [], []
        for p in shards:
            z = np.load(p)
            imgs.append(z["image"])
            auxs.append(encode_aux(z["state"], z["instr"]))
            acts.append(z["action"])
            lens.append(z["lengths"])
            metas += json.loads(str(z["meta"]))
        lengths = np.concatenate(lens)
        starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
        keep = np.arange(len(lengths))[episodes] if episodes is not None else np.arange(len(lengths))
        image, aux, action = np.concatenate(imgs), np.concatenate(auxs), np.concatenate(acts)

        # chunk index table: frame t -> frames t..t+7 of the same episode, padded with the last one
        rows, idx, pad = [], [], []
        for e in keep:
            s, n = starts[e], lengths[e]
            t = np.arange(n)
            k = t[:, None] + np.arange(CHUNK)[None]
            pad.append(k >= n)
            idx.append(s + np.minimum(k, n - 1))
            rows.append(s + t)
        rows = np.concatenate(rows)
        self.meta = [metas[e] for e in keep]
        self.n = len(rows)
        self.device = device
        self.image = torch.from_numpy(image[rows]).to(device)
        self.aux = torch.from_numpy(aux[rows]).to(device)
        self.chunk = torch.from_numpy(action[np.concatenate(idx)]).to(device)  # [N, 8, 6]
        self.pad = torch.from_numpy(np.concatenate(pad)).to(device)  # [N, 8] True = past episode end

    def batch(self, idx: torch.Tensor):
        return self.image[idx], self.aux[idx], self.chunk[idx], self.pad[idx]
