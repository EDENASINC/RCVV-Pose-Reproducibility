import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_builder():
    path = ROOT / "tools" / "build_model_archives.py"
    spec = importlib.util.spec_from_file_location("build_model_archives", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReleaseAssetTests(unittest.TestCase):
    def test_release_contract_has_nine_unique_assets(self):
        builder = load_builder()
        names = builder.EXPECTED_RELEASE_ASSET_NAMES
        self.assertEqual(len(names), 9)
        self.assertEqual(len(set(names)), 9)
        self.assertIn("learned_calibration_artifacts.zip", names)


if __name__ == "__main__":
    unittest.main()
