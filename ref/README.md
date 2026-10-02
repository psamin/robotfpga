# ref: the shared int8 contract

Everything in this folder is the agreement between software and FPGA. Changing a format or the math
needs an issue and sign-off from both sides ([CONTRIBUTING.md](../CONTRIBUTING.md)).
If the hardware and `intref.py` disagree, `intref.py` is right.

| File | What it is |
|---|---|
| `intref.py` | Bit-exact numpy forward pass, `pack_obs` / `unpack_actions`, `Model.save` / `Model.load` |
| `vectors.py` | Builds the 100 golden vectors from a model and real frames |
| `make_standin.py` | Random stand-in weights + synthetic frames, until trained weights exist |
| `test_*.py` | `python -m pytest ref` |

## Quick start

```
pip install numpy pytest
python ref/make_standin.py --out artifacts/standin    # ~1 s, deterministic for a given --seed
python -m pytest ref
```

`artifacts/` is not committed. Regenerate it; same seed, same bytes.

## Formats

### Input packet: 27,658 bytes

| Bytes | Content |
|---|---|
| 0 .. 27,647 | image, uint8, HWC 96×96×3, row-major. The hardware computes `pixel - 128`. |
| 27,648 .. 27,653 | 6 joint states, int8: `floor(state * 127 + 0.5)` clamped to [-127, 127] |
| 27,654 .. 27,657 | instruction one-hot, int8, 127 for the hot entry |

`intref.pack_obs(obs)` builds it from `{"image", "state", "instr"}`.

### Output: 48 bytes

int8, index `step * 6 + joint`. `intref.unpack_actions(y)` gives float32 `[8, 6]` = `y / 127`.

### `weights.bin`

Layers in order conv1..conv5, fc6..fc8. Per layer: int8 weights in [-127, 127], then int32 little-endian
biases. Conv weights are `[out_ch][ky][kx][in_ch]`, FC weights `[out][in]`. 568,240 bytes in total.

### `manifest.json`

```json
{
  "format_version": 1,
  "note": "free text",
  "weights_file": "weights.bin",
  "total_bytes": 568240,
  "sha256": "<hex of weights.bin>",
  "input": {"packet_bytes": 27658, "image_shape": [96, 96, 3], "aux_len": 10},
  "output": {"len": 48, "shape": [8, 6]},
  "layers": [
    {"name": "conv1", "type": "conv", "in_shape": [96, 96, 3], "out_shape": [48, 48, 16], "relu": true,
     "shift": 8, "weight_shape": [16, 3, 3, 3], "weight_offset": 0, "weight_bytes": 432,
     "bias_offset": 432, "bias_bytes": 64}
  ]
}
```

`shift` must be 1..30. The hardware reads the 8 shifts from here into its registers.

### `vectors/`

| File | Content |
|---|---|
| `NNN_in.bin` | 27,658-byte input packet, NNN = 000..099 |
| `NNN_out.bin` | expected 48-byte output |
| `NNN_layerK.bin` | int8 output of layer K = 1..8 (HWC for convs), for NNN = 000..004 only |

000 all-zero, 001 all-max, 002 biases only, 003 a frame, 004 noise, 005–013 edge patterns, then frames and
noise interleaved. Details in `vectors.py`.

## From a trained model (software side)

```python
from intref import LAYERS, Model, pack_obs
from vectors import make_vector_set, write_vectors

layers = [dict(layer, shift=s) for layer, s in zip(LAYERS, shifts)]   # shifts from QAT
model = Model(layers, [(w_int8, b_int32), ...])               # conv w: [out][ky][kx][in]
model.save("artifacts/v1")                                    # weights.bin + manifest.json
frames = [pack_obs(obs) for obs in held_out_observations]     # at least 58
write_vectors(Model.load("artifacts/v1"), make_vector_set(frames, rng), "artifacts/v1")
```

Or from the command line, with frames saved as a uint8 `[N, 27658]` array:
`python ref/vectors.py --model artifacts/v1 --packets frames.npy`.
