import csv
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ReleaseContractTests(unittest.TestCase):
    def test_checkpoint_manifest_has_locked_matrix(self):
        with (ROOT / "models" / "checkpoint_manifest.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 72)
        self.assertEqual({row["split"] for row in rows}, {"split_a", "split_b", "split_c"})
        self.assertEqual({int(row["seed"]) for row in rows}, {42, 123, 2026})
        self.assertEqual({row["detector"] for row in rows}, {"rtmpose_performance", "yolo11l_pose"})
        self.assertEqual({row["arm_short"] for row in rows}, {"O", "OV", "OR", "OVR"})

    def test_official_protocol_is_locked(self):
        path = ROOT / "evidence" / "13_locked_official_ts1_ts6_evaluation_protocol.json"
        protocol = json.loads(path.read_text(encoding="utf-8"))
        matrix = protocol["locked_model_matrix"]
        self.assertEqual(matrix["checkpoint_count"], 72)
        self.assertFalse(matrix["official_test_used_for_checkpoint_or_arm_selection"])


if __name__ == "__main__":
    unittest.main()
