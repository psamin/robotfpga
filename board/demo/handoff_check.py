"""Read-only small-policy artifact preflight; never accesses FPGA or motors."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import time


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_reference(path):
    spec = importlib.util.spec_from_file_location("handoff_intref", path)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    return reference


def check_handoff(directory, reference, verify=False):
    directory = Path(directory)
    raw_manifest = (directory / "manifest.json").read_bytes()
    manifest = json.loads(raw_manifest)
    require(type(manifest["format_version"]) is int and manifest["format_version"] == 1, "Unsupported manifest version")
    require(manifest["weights_file"] == "weights.bin", "Expected weights.bin")
    require(manifest["input"] == {"packet_bytes": reference.PACKET_BYTES,
            "image_shape": [reference.IMG_H, reference.IMG_W, reference.IMG_C],
            "aux_len": reference.AUX_LEN}, "Input interface mismatch")
    require(manifest["output"] == {"len": reference.OUT_LEN, "shape": [8, 6]}, "Output interface mismatch")
    weights = (directory / "weights.bin").read_bytes()
    digest = hashlib.sha256(weights).hexdigest()
    require(digest == manifest["sha256"], "Weights SHA256 mismatch")
    require(len(manifest["layers"]) == len(reference.LAYERS), "Layer count mismatch")
    offset = 0
    for layer, expected in zip(manifest["layers"], reference.LAYERS):
        require(all(type(layer[key]) is type(value) and layer[key] == value
                    for key, value in expected.items()), "Layer architecture mismatch")
        require(type(layer["shift"]) is int and 1 <= layer["shift"] <= 30, "Invalid layer shift")
        shape = list(reference.weight_shape(expected))
        size = math.prod(shape)
        bias_size = 4 * reference.n_out_channels(expected)
        require(layer["weight_shape"] == shape, "Weight shape mismatch")
        for key, value in {"weight_offset": offset, "weight_bytes": size,
                           "bias_offset": offset + size, "bias_bytes": bias_size}.items():
            require(type(layer[key]) is int and layer[key] == value, "Invalid layout: " + key)
        require(128 not in weights[offset:offset + size], "Weight -128 violates export range")
        offset += size + bias_size
    require(type(manifest["total_bytes"]) is int and len(weights) == offset == manifest["total_bytes"], "Weights length mismatch")
    sizes = {}
    for index in range(100):
        sizes["{:03d}_in.bin".format(index)] = reference.PACKET_BYTES
        sizes["{:03d}_out.bin".format(index)] = reference.OUT_LEN
        if index < 5:
            for k, layer in enumerate(reference.LAYERS, 1):
                sizes["{:03d}_layer{}.bin".format(index, k)] = math.prod(layer["out_shape"])
    vector_dir = directory / "vectors"
    require({p.name for p in vector_dir.glob("*.bin")} == set(sizes), "Incomplete or extra vector set")
    vectors = {name: (vector_dir / name).read_bytes() for name in sizes}
    require(all(len(vectors[name]) == size for name, size in sizes.items()), "Vector size mismatch")
    times = []
    if verify:
        model = reference.Model.load(directory)
        for index in range(100):
            prefix = "{:03d}".format(index)
            started = time.perf_counter()
            output, layers = reference.forward(vectors[prefix + "_in.bin"], model)
            times.append(1000 * (time.perf_counter() - started))
            require(output.tobytes() == vectors[prefix + "_out.bin"], "Reference output mismatch: " + prefix)
            if index < 5:
                require(len(layers) == 8, "Reference layer count mismatch")
                for k, layer_output in enumerate(layers, 1):
                    require(layer_output.tobytes() == vectors[prefix + "_layer{}.bin".format(k)],
                            "Reference layer mismatch: {} layer {}".format(prefix, k))
    return {"preflight": "PASS", "reference_verified": verify, "golden_inputs": 100,
            "layer_dumps": 40, "weights_sha256": digest,
            "manifest_sha256": hashlib.sha256(raw_manifest).hexdigest(),
            "reference_sha256": hashlib.sha256(Path(reference.__file__).read_bytes()).hexdigest(),
            "reference_calls": len(times), "reference_mean_ms": sum(times) / len(times) if times else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--verify-reference", action="store_true")
    args = parser.parse_args()
    try:
        result = check_handoff(args.artifacts, load_reference(args.reference), args.verify_reference)
    except (ValueError, KeyError, TypeError, OSError, ImportError, OverflowError) as exc:
        print(json.dumps({"preflight": "FAIL", "error": str(exc)}))
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
