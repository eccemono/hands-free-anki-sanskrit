import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class AddonManifestTests(unittest.TestCase):
    def test_manifest_has_required_fields_and_importable_package(self):
        manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))

        self.assertTrue(manifest["name"])
        self.assertRegex(manifest["package"], re.compile(r"^[A-Za-z_]\w*$"))
        self.assertGreaterEqual(manifest["min_point_version"], 54)

    def test_meta_targets_same_package_homepage_and_minimum_version(self):
        manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
        meta = json.loads((ROOT / "meta.json").read_text(encoding="utf-8"))

        self.assertEqual(meta["homepage"], manifest["homepage"])
        self.assertEqual(meta["min_point_version"], manifest["min_point_version"])


if __name__ == "__main__":
    unittest.main()
