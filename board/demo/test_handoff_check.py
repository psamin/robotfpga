"""Run: python test_handoff_check.py /path/to/the/existing/ref/intref.py"""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

from handoff_check import check_handoff, load_reference


class HandoffTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.base = Path(cls.temp.name) / "zero-model"
        layers = [dict(layer, shift=8) for layer in REF.LAYERS]
        params = [(REF.np.zeros(REF.weight_shape(layer), REF.np.int8),
                   REF.np.zeros(REF.n_out_channels(layer), REF.np.int32)) for layer in layers]
        REF.Model(layers, params).save(cls.base, note="TEST ONLY: zero synthetic model")
        vectors = cls.base / "vectors"
        vectors.mkdir()
        for index in range(100):
            (vectors / "{:03d}_in.bin".format(index)).write_bytes(bytes(REF.PACKET_BYTES))
            (vectors / "{:03d}_out.bin".format(index)).write_bytes(bytes(REF.OUT_LEN))
            if index < 5:
                for k, layer in enumerate(layers, 1):
                    (vectors / "{:03d}_layer{}.bin".format(index, k)).write_bytes(
                        bytes(int(REF.np.prod(layer["out_shape"]))))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        self.work = Path(self.temp.name) / self.id().split(".")[-1]
        shutil.copytree(self.base, self.work)

    def test_full_reference(self):
        result = check_handoff(self.work, REF, verify=True)
        self.assertEqual(result["reference_calls"], 100)
        self.assertTrue(result["reference_verified"])

    def test_bad_files(self):
        for name, data in [("weights.bin", b"bad"), ("vectors/099_in.bin", b"short"),
                           ("vectors/000_layer1.bin", b"short")]:
            path = self.work / name
            original = path.read_bytes()
            path.write_bytes(data)
            with self.assertRaises(ValueError): check_handoff(self.work, REF)
            path.write_bytes(original)
        path = self.work / "vectors/099_out.bin"
        path.unlink()
        with self.assertRaises(ValueError): check_handoff(self.work, REF)

    def test_bad_manifest(self):
        path = self.work / "manifest.json"
        original = path.read_text()
        for key, value in [("shift", 0), ("shift", True), ("weight_offset", 1),
                           ("weight_shape", [1]), ("out_shape", [1])]:
            manifest = json.loads(original)
            manifest["layers"][0][key] = value
            path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError): check_handoff(self.work, REF)

    def test_extra_and_wrong_output(self):
        extra = self.work / "vectors/100_out.bin"
        extra.write_bytes(bytes(48))
        with self.assertRaises(ValueError): check_handoff(self.work, REF)
        extra.unlink()
        (self.work / "vectors/000_out.bin").write_bytes(bytes([1]) + bytes(47))
        with self.assertRaisesRegex(ValueError, "Reference output mismatch"):
            check_handoff(self.work, REF, verify=True)
        (self.work / "vectors/000_out.bin").write_bytes(bytes(48))
        layer = self.work / "vectors/000_layer1.bin"
        layer.write_bytes(bytes([1]) + layer.read_bytes()[1:])
        with self.assertRaisesRegex(ValueError, "Reference layer mismatch"):
            check_handoff(self.work, REF, verify=True)

    def test_bad_weight_range(self):
        path = self.work / "weights.bin"
        blob = bytes([128]) + path.read_bytes()[1:]
        path.write_bytes(blob)
        manifest_path = self.work / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["sha256"] = hashlib.sha256(blob).hexdigest()
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "Weight -128"):
            check_handoff(self.work, REF)


if __name__ == "__main__":
    REF = load_reference(Path(sys.argv.pop(1)))
    unittest.main()
