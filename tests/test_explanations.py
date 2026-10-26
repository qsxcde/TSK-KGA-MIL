import tempfile
from pathlib import Path
import unittest

import numpy as np

from src.utils.explanations import save_bag_explanation, save_explanation_manifest


class ExplanationPersistenceTest(unittest.TestCase):
    def test_save_bag_explanation_persists_arrays_and_metadata(self):
        explanation = {
            "bag_prediction": np.array([0.8], dtype=np.float32),
            "attention_weights": np.array([[[0.2], [0.8]]], dtype=np.float32),
            "instance_keyness_scores": np.array([[0.1, 0.9]], dtype=np.float32),
            "bag_index": np.array([3], dtype=np.int64),
            "bag_label": np.array([1], dtype=np.int64),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "fold_1_sample.npz"
            save_bag_explanation(explanation, output_path)

            self.assertTrue(output_path.exists())
            loaded = np.load(output_path)
            self.assertAlmostEqual(float(loaded["bag_prediction"][0]), 0.8, places=6)
            self.assertEqual(int(loaded["bag_index"][0]), 3)
            self.assertEqual(int(loaded["bag_label"][0]), 1)
            self.assertEqual(tuple(loaded["attention_weights"].shape), (1, 2, 1))

    def test_save_explanation_manifest_persists_fold_records(self):
        records = [
            {
                "fold": 1,
                "bag_index": 3,
                "bag_label": 1,
                "file": "fold_1_sample_0.npz",
            },
            {
                "fold": 1,
                "bag_index": 8,
                "bag_label": 0,
                "file": "fold_1_sample_1.npz",
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "manifest.json"
            save_explanation_manifest(records, output_path)

            self.assertTrue(output_path.exists())
            content = output_path.read_text(encoding="utf-8")
            self.assertIn("\"fold\": 1", content)
            self.assertIn("\"bag_index\": 8", content)
            self.assertIn("fold_1_sample_1.npz", content)


if __name__ == "__main__":
    unittest.main()
