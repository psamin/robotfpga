"""Integrity checks must reject damaged transfers and unsafe provenance."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from verify_bundle import verify


class BundleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name) / "bundle"
        self.base.mkdir()
        self.payload = b"\x00\xff\x0a\x80"
        (self.base / "tensor.bin").write_bytes(self.payload)
        self.digest = hashlib.sha256(self.payload).hexdigest()
        self.manifest({"tensor.bin": self.digest})

    def manifest(self, files):
        (self.base / "bundle.json").write_text(json.dumps({"sha256": files}))

    def test_complete_binary_transfer(self):
        self.assertEqual(verify(self.base), {"bundle": "PASS", "checked_files": 1})

    def test_changed_byte_rejected(self):
        (self.base / "tensor.bin").write_bytes(self.payload[:-1] + b"\x81")
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            verify(self.base)

    def test_missing_file_rejected(self):
        (self.base / "tensor.bin").unlink()
        with self.assertRaisesRegex(ValueError, "Missing file"):
            verify(self.base)

    def test_unsafe_paths_rejected(self):
        (self.base.parent / "outside.bin").write_bytes(self.payload)
        for name in ("../outside.bin", "/outside.bin", "C:/outside.bin", "a\\outside.bin"):
            with self.subTest(name=name):
                self.manifest({name: self.digest})
                with self.assertRaisesRegex(ValueError, "Unsafe bundle path"):
                    verify(self.base)

    def test_bad_provenance_rejected(self):
        for files in ({}, [], {"tensor.bin": "not-a-hash"}):
            with self.subTest(files=files):
                self.manifest(files)
                with self.assertRaises(ValueError):
                    verify(self.base)


if __name__ == "__main__":
    unittest.main()
