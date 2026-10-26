import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from main import run_kfold_experiment


class KFoldResultContractTests(unittest.TestCase):
    @patch("main.save_explanation_manifest")
    @patch("main.save_bag_explanation")
    @patch("main.torch.save")
    @patch("main.get_experiment_logger")
    @patch("main.load_data")
    @patch("main.run_single_fold_training")
    def test_run_kfold_experiment_returns_structured_metrics(
        self,
        mock_train,
        mock_load_data,
        mock_get_logger,
        _mock_torch_save,
        _mock_save_bag_explanation,
        _mock_save_manifest,
    ):
        mock_get_logger.return_value = MagicMock()
        bags = [np.zeros((2, 3), dtype=np.float32) for _ in range(6)]
        labels = np.array([0, 1, 0, 1, 0, 1], dtype=np.int64)
        mock_load_data.return_value = (bags, labels)
        mock_train.side_effect = [
            {
                "best_acc": 0.7,
                "model": MagicMock(state_dict=MagicMock(return_value={})),
                "history": {
                    "best_epoch": 2,
                    "val_f1": [0.5, 0.6],
                    "val_auc": [0.55, 0.65],
                    "best_val_acc": 0.7,
                    "training_time": 1.0,
                },
                "explanation_samples": [],
            },
            {
                "best_acc": 0.8,
                "model": MagicMock(state_dict=MagicMock(return_value={})),
                "history": {
                    "best_epoch": 3,
                    "val_f1": [0.55, 0.65],
                    "val_auc": [0.6, 0.7],
                    "best_val_acc": 0.8,
                    "training_time": 1.2,
                },
                "explanation_samples": [],
            },
            {
                "best_acc": 0.9,
                "model": MagicMock(state_dict=MagicMock(return_value={})),
                "history": {
                    "best_epoch": 4,
                    "val_f1": [0.6, 0.7],
                    "val_auc": [0.65, 0.75],
                    "best_val_acc": 0.9,
                    "training_time": 1.4,
                },
                "explanation_samples": [],
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_kfold_experiment(
                {
                    "data_path": "data/newsgroups/alt_atheism+.mat",
                    "dataset_name": "alt_atheism+",
                    "n_splits": 3,
                    "random_state": 42,
                    "output_root_dir": tmpdir,
                    "device": "cpu",
                    "num_rules": 4,
                    "rule_output_dim": 1,
                    "aggregation_method": "attention",
                    "lr": 0.001,
                    "max_epoch": 2,
                    "fcm_m": 2.0,
                    "fcm_max_iter": 5,
                    "patience": 2,
                    "hidden_dim": 8,
                    "rule_temperature": 0.5,
                    "use_competitive_scoring": True,
                    "explanation_samples_per_fold": 0,
                }
            )

        self.assertEqual(result["dataset_name"], "alt_atheism+")
        self.assertEqual(result["dataset"], "alt_atheism+")
        self.assertEqual(result["fold_scores"], [0.7, 0.8, 0.9])
        self.assertEqual(result["best_epochs"], [3, 4, 5])
        self.assertAlmostEqual(result["mean_acc"], result["mean_metrics"]["acc"])
        self.assertAlmostEqual(result["std_acc"], result["std_metrics"]["acc"])
        self.assertAlmostEqual(result["mean_metrics"]["acc"], 0.8)
        self.assertAlmostEqual(result["mean_metrics"]["f1"], 0.65)
        self.assertAlmostEqual(result["mean_metrics"]["auc"], 0.7)
        self.assertAlmostEqual(result["std_metrics"]["acc"], 0.1)
        self.assertAlmostEqual(result["std_metrics"]["f1"], 0.05)
        self.assertAlmostEqual(result["std_metrics"]["auc"], 0.05)
        self.assertEqual(result["artifacts_path"], str(Path(tmpdir) / "artifacts"))
        self.assertEqual(result["log_dir"], str(Path(tmpdir) / "logs"))
        self.assertEqual(result["explanation_dir"], str(Path(tmpdir) / "artifacts" / "explanations" / "alt_atheism+"))
        self.assertEqual(len(result["fold_metrics"]), 3)
        self.assertEqual(
            result["fold_metrics"][0],
            {
                "fold": 1,
                "acc": 0.7,
                "f1": 0.6,
                "auc": 0.65,
                "best_epoch": 3,
                "training_time_sec": 1.0,
            },
        )
        self.assertIn("training_time_sec", result)
        self.assertIn("dataset_name", result)
        self.assertIn("fold_metrics", result)
        self.assertIn("mean_metrics", result)
        self.assertIn("std_metrics", result)
        self.assertIn("best_epochs", result)


if __name__ == "__main__":
    unittest.main()
