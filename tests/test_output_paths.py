from pathlib import Path
import tempfile
import unittest

from main import run_kfold_experiment


class OutputPathTest(unittest.TestCase):
    def test_custom_output_root_and_run_name_redirect_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir)
            config = {
                "data_path": "data/bechmark/musk1+.mat",
                "dataset_name": "musk1+",
                "num_rules": 4,
                "rule_output_dim": 1,
                "aggregation_method": "attention",
                "batch_size": 1,
                "lr": 0.001,
                "max_epoch": 1,
                "fcm_m": 2.0,
                "fcm_max_iter": 5,
                "patience": 1,
                "device": "cpu",
                "hidden_dim": 16,
                "rule_temperature": 0.5,
                "use_competitive_scoring": True,
                "explanation_samples_per_fold": 1,
                "n_splits": 2,
                "random_state": 42,
                "output_root_dir": str(output_root),
                "run_name": "smoke_run",
            }

            run_kfold_experiment(config)

            run_dir = output_root / "smoke_run"
            self.assertTrue((run_dir / "logs").exists())
            self.assertTrue((run_dir / "artifacts" / "models" / "musk1+").exists())
            self.assertTrue((run_dir / "artifacts" / "explanations" / "musk1+").exists())

    def test_sensitivity_output_directories_are_documented(self):
        readme = Path("README.md").read_text(encoding="utf-8")
        self.assertIn("run_sensitivity_analysis.py", readme)
        self.assertIn("build_sensitivity_report.py", readme)


if __name__ == "__main__":
    unittest.main()
