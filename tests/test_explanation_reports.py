import csv
import json
import tempfile
from pathlib import Path
import unittest

import numpy as np

from src.utils.explanations import export_instance_report_csv, save_bag_explanation


class ExplanationReportTest(unittest.TestCase):
    def test_export_instance_report_csv_writes_expected_columns(self):
        explanation = {
            "bag_prediction": np.array([0.8], dtype=np.float32),
            "bag_label": np.array([1], dtype=np.int64),
            "bag_index": np.array([3], dtype=np.int64),
            "attention_weights": np.array([[[0.2], [0.8]]], dtype=np.float32),
            "instance_keyness_scores": np.array([[0.1, 0.9]], dtype=np.float32),
            "normalized_rule_membership": np.array(
                [[[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]]], dtype=np.float32
            ),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            explanation_path = base / "sample_explanation.npz"
            manifest_path = base / "manifest.json"
            csv_path = base / "instance_report.csv"

            save_bag_explanation(explanation, explanation_path)
            manifest_path.write_text(
                json.dumps(
                    [
                        {
                            "fold": 1,
                            "sample_position": 0,
                            "bag_index": 3,
                            "bag_label": 1,
                            "file": explanation_path.name,
                        }
                    ]
                ),
                encoding="utf-8",
            )

            export_instance_report_csv(manifest_path, base, csv_path)

            self.assertTrue(csv_path.exists())
            with csv_path.open("r", encoding="utf-8", newline="") as file:
                rows = list(csv.DictReader(file))

            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["bag_index"], "3")
            self.assertEqual(rows[0]["instance_index"], "0")
            self.assertEqual(rows[0]["top_rule"], "0")
            self.assertEqual(rows[1]["top_rule"], "2")
            self.assertIn("top_rule_prob", rows[0])
            self.assertIn("attention_weight", rows[0])


if __name__ == "__main__":
    unittest.main()
