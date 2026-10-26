import tempfile
import unittest
from pathlib import Path

import yaml

from src.analysis.sensitivity.config import load_sensitivity_config
from src.analysis.sensitivity.grid import expand_sensitivity_runs, write_resolved_config


class SensitivityGridTests(unittest.TestCase):
    def test_expand_runs_uses_one_factor_behavior(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            base_config_path = root / "base.yaml"
            base_config_path.write_text(
                yaml.safe_dump(
                    {
                        "model": {"num_rules": 8, "hidden_dim": 32},
                        "training": {"epochs": 10},
                        "datasets": [{"config": "demo.yaml"}],
                    }
                ),
                encoding="utf-8",
            )

            study = {
                "base_experiment_config": str(base_config_path),
                "study": {"name": "demo", "output_root_dir": "artifacts/sensitivity"},
                "sweeps": [
                    {"name": "num_rules", "param": "model.num_rules", "values": [2, 4]},
                    {"name": "hidden_dim", "param": "model.hidden_dim", "values": [16]},
                ],
                "report": {"metrics": ["auc"], "primary_metric": "auc"},
            }

            runs = expand_sensitivity_runs(study)

            self.assertEqual(3, len(runs))
            self.assertEqual(2, runs[0]["resolved_config"]["model"]["num_rules"])
            self.assertEqual(32, runs[0]["resolved_config"]["model"]["hidden_dim"])
            self.assertEqual(4, runs[1]["resolved_config"]["model"]["num_rules"])
            self.assertEqual(32, runs[1]["resolved_config"]["model"]["hidden_dim"])
            self.assertEqual(8, runs[2]["resolved_config"]["model"]["num_rules"])
            self.assertEqual(16, runs[2]["resolved_config"]["model"]["hidden_dim"])

    def test_expand_runs_slugifies_unsafe_sweep_names_into_safe_run_ids(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            base_config_path = root / "base.yaml"
            base_config_path.write_text(
                yaml.safe_dump(
                    {
                        "model": {"num_rules": 8},
                        "training": {"epochs": 10},
                        "datasets": [{"config": "demo.yaml"}],
                    }
                ),
                encoding="utf-8",
            )

            study = {
                "base_experiment_config": str(base_config_path),
                "study": {"name": "demo", "output_root_dir": "artifacts/sensitivity"},
                "sweeps": [
                    {"name": "model/num rules", "param": "model.num_rules", "values": [2, 4]},
                ],
                "report": {"metrics": ["auc"], "primary_metric": "auc"},
            }

            runs = expand_sensitivity_runs(study)

            self.assertEqual(["model_num_rules__2", "model_num_rules__4"], [run["run_id"] for run in runs])
            self.assertTrue(all("/" not in run["run_id"] and "\\" not in run["run_id"] for run in runs))

    def test_expand_runs_rejects_colliding_run_ids_from_slugified_sweep_names(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            base_config_path = root / "base.yaml"
            base_config_path.write_text(
                yaml.safe_dump(
                    {
                        "model": {"num_rules": 8, "hidden_dim": 32},
                        "training": {"epochs": 10},
                        "datasets": [{"config": "demo.yaml"}],
                    }
                ),
                encoding="utf-8",
            )

            study = {
                "base_experiment_config": str(base_config_path),
                "study": {"name": "demo", "output_root_dir": "artifacts/sensitivity"},
                "sweeps": [
                    {"name": "model/num rules", "param": "model.num_rules", "values": [2]},
                    {"name": "model_num_rules", "param": "model.hidden_dim", "values": [2]},
                ],
                "report": {"metrics": ["auc"], "primary_metric": "auc"},
            }

            with self.assertRaisesRegex(ValueError, "Duplicate generated run_id"):
                expand_sensitivity_runs(study)

    def test_expand_runs_rejects_colliding_run_ids_from_slugified_values(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            base_config_path = root / "base.yaml"
            base_config_path.write_text(
                yaml.safe_dump(
                    {
                        "model": {"num_rules": 8},
                        "training": {"epochs": 10},
                        "datasets": [{"config": "demo.yaml"}],
                    }
                ),
                encoding="utf-8",
            )

            study = {
                "base_experiment_config": str(base_config_path),
                "study": {"name": "demo", "output_root_dir": "artifacts/sensitivity"},
                "sweeps": [
                    {
                        "name": "num_rules",
                        "param": "model.num_rules",
                        "values": ["alpha/beta", "alpha_beta"],
                    },
                ],
                "report": {"metrics": ["auc"], "primary_metric": "auc"},
            }

            with self.assertRaisesRegex(ValueError, "Duplicate generated run_id"):
                expand_sensitivity_runs(study)

    def test_write_resolved_config_writes_yaml_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "runs" / "demo"
            output_path = write_resolved_config(
                run_dir,
                {"model": {"num_rules": 2}, "training": {"epochs": 10}},
            )

            self.assertEqual(run_dir / "experiment_config_resolved.yaml", output_path)
            self.assertTrue(output_path.exists())
            self.assertEqual(
                {"model": {"num_rules": 2}, "training": {"epochs": 10}},
                yaml.safe_load(output_path.read_text(encoding="utf-8")),
            )

    def test_expand_runs_loads_relative_base_config_through_loader(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sensitivity_dir = root / "configs" / "sensitivity"
            experiments_dir = root / "configs" / "experiments"
            sensitivity_dir.mkdir(parents=True)
            experiments_dir.mkdir(parents=True)

            base_config_path = experiments_dir / "base.yaml"
            base_config_path.write_text(
                yaml.safe_dump(
                    {
                        "model": {"num_rules": 8, "hidden_dim": 32},
                        "training": {"epochs": 10},
                        "datasets": [{"config": "demo.yaml"}],
                    }
                ),
                encoding="utf-8",
            )

            sensitivity_config_path = sensitivity_dir / "demo.yaml"
            sensitivity_config_path.write_text(
                yaml.safe_dump(
                    {
                        "base_experiment_config": "../experiments/base.yaml",
                        "study": {"name": "demo", "output_root_dir": "artifacts/sensitivity"},
                        "sweeps": [
                            {"name": "num_rules", "param": "model.num_rules", "values": [2, 4]},
                        ],
                        "report": {"metrics": ["auc"], "primary_metric": "auc"},
                    }
                ),
                encoding="utf-8",
            )

            study = load_sensitivity_config(sensitivity_config_path)
            runs = expand_sensitivity_runs(study)

            self.assertEqual(2, len(runs))
            self.assertEqual(2, runs[0]["resolved_config"]["model"]["num_rules"])
            self.assertEqual(32, runs[0]["resolved_config"]["model"]["hidden_dim"])
            self.assertEqual(4, runs[1]["resolved_config"]["model"]["num_rules"])
            self.assertEqual(32, runs[1]["resolved_config"]["model"]["hidden_dim"])


if __name__ == "__main__":
    unittest.main()
