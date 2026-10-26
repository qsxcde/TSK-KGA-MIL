import tempfile
import unittest
from pathlib import Path

import yaml

from src.analysis.sensitivity.config import load_sensitivity_config, validate_sensitivity_config
from src.analysis.sensitivity.grid import expand_sensitivity_runs


class SensitivityConfigTests(unittest.TestCase):
    def test_validate_rejects_empty_values(self):
        config = {
            "base_experiment_config": "configs/experiments/ds_tsk_mil/ds_tsk_mil_smoke.yaml",
            "study": {"name": "demo", "output_root_dir": "artifacts/sensitivity"},
            "sweeps": [{"name": "num_rules", "param": "model.num_rules", "values": []}],
            "report": {"metrics": ["auc"], "primary_metric": "auc"},
        }

        with self.assertRaises(ValueError):
            validate_sensitivity_config(config)

    def test_validate_rejects_unsupported_report_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            base_config_path = root / "base.yaml"
            base_config_path.write_text(
                yaml.safe_dump({"model": {"num_rules": 4}, "datasets": [{"config": "demo.yaml"}]}),
                encoding="utf-8",
            )

            config = {
                "base_experiment_config": str(base_config_path),
                "study": {"name": "demo", "output_root_dir": "artifacts/sensitivity"},
                "sweeps": [{"name": "num_rules", "param": "model.num_rules", "values": [2, 4]}],
                "report": {"metrics": ["auc", "precision"], "primary_metric": "auc"},
            }

            with self.assertRaisesRegex(ValueError, "Unsupported report metric"):
                validate_sensitivity_config(config)

    def test_validate_rejects_unsupported_primary_report_metric(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            base_config_path = root / "base.yaml"
            base_config_path.write_text(
                yaml.safe_dump({"model": {"num_rules": 4}, "datasets": [{"config": "demo.yaml"}]}),
                encoding="utf-8",
            )

            config = {
                "base_experiment_config": str(base_config_path),
                "study": {"name": "demo", "output_root_dir": "artifacts/sensitivity"},
                "sweeps": [{"name": "num_rules", "param": "model.num_rules", "values": [2, 4]}],
                "report": {"metrics": ["auc"], "primary_metric": "precision"},
            }

            with self.assertRaisesRegex(ValueError, "Unsupported primary report metric"):
                validate_sensitivity_config(config)

    def test_validate_rejects_missing_or_blank_sweep_name(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            base_config_path = root / "base.yaml"
            base_config_path.write_text(
                yaml.safe_dump({"model": {"num_rules": 4}, "datasets": [{"config": "demo.yaml"}]}),
                encoding="utf-8",
            )

            missing_name = {
                "base_experiment_config": str(base_config_path),
                "study": {"name": "demo", "output_root_dir": "artifacts/sensitivity"},
                "sweeps": [{"param": "model.num_rules", "values": [2, 4]}],
                "report": {"metrics": ["auc"], "primary_metric": "auc"},
            }
            blank_name = {
                "base_experiment_config": str(base_config_path),
                "study": {"name": "demo", "output_root_dir": "artifacts/sensitivity"},
                "sweeps": [{"name": "   ", "param": "model.num_rules", "values": [2, 4]}],
                "report": {"metrics": ["auc"], "primary_metric": "auc"},
            }

            with self.assertRaises(ValueError):
                validate_sensitivity_config(missing_name)
            with self.assertRaises(ValueError):
                validate_sensitivity_config(blank_name)

    def test_validate_requires_explicit_config_path_for_relative_base_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sensitivity_dir = root / "configs" / "sensitivity"
            experiments_dir = root / "configs" / "experiments"
            sensitivity_dir.mkdir(parents=True)
            experiments_dir.mkdir(parents=True)

            base_config_path = experiments_dir / "base.yaml"
            base_config_path.write_text(
                yaml.safe_dump({"model": {"num_rules": 4}, "datasets": [{"config": "demo.yaml"}]}),
                encoding="utf-8",
            )

            config = {
                "base_experiment_config": "../experiments/base.yaml",
                "study": {"name": "demo", "output_root_dir": "artifacts/sensitivity"},
                "sweeps": [{"name": "num_rules", "param": "model.num_rules", "values": [2, 4]}],
                "report": {"metrics": ["auc"], "primary_metric": "auc"},
            }

            with self.assertRaises(ValueError):
                validate_sensitivity_config(config)

            validate_sensitivity_config(config, config_path=sensitivity_dir / "demo.yaml")

    def test_load_resolves_base_experiment_relative_to_config_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sensitivity_dir = root / "configs" / "sensitivity"
            experiments_dir = root / "configs" / "experiments"
            sensitivity_dir.mkdir(parents=True)
            experiments_dir.mkdir(parents=True)

            base_config_path = experiments_dir / "base.yaml"
            base_config_path.write_text(
                yaml.safe_dump({"model": {"num_rules": 4}, "datasets": [{"config": "demo.yaml"}]}),
                encoding="utf-8",
            )

            sensitivity_config_path = sensitivity_dir / "demo.yaml"
            sensitivity_config_path.write_text(
                yaml.safe_dump(
                    {
                        "base_experiment_config": "../experiments/base.yaml",
                        "study": {"name": "demo", "output_root_dir": "artifacts/sensitivity"},
                        "sweeps": [{"name": "num_rules", "param": "model.num_rules", "values": [2, 4]}],
                        "report": {"metrics": ["auc"], "primary_metric": "auc"},
                    }
                ),
                encoding="utf-8",
            )

            config = load_sensitivity_config(sensitivity_config_path)

            self.assertEqual(config["base_experiment_config"], "../experiments/base.yaml")
            self.assertNotIn("config_path", config)

    def test_validate_rejects_multi_dataset_base_experiment_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            base_config_path = root / "base.yaml"
            base_config_path.write_text(
                yaml.safe_dump(
                    {
                        "model": {"num_rules": 4},
                        "datasets": [{"config": "one.yaml"}, {"config": "two.yaml"}],
                    }
                ),
                encoding="utf-8",
            )

            config = {
                "base_experiment_config": str(base_config_path),
                "study": {"name": "demo", "output_root_dir": "artifacts/sensitivity"},
                "sweeps": [{"name": "num_rules", "param": "model.num_rules", "values": [2, 4]}],
                "report": {"metrics": ["auc"], "primary_metric": "auc"},
            }

            with self.assertRaisesRegex(ValueError, "exactly one dataset"):
                validate_sensitivity_config(config)

    def test_validate_requires_study_name_and_output_root_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            base_config_path = root / "base.yaml"
            base_config_path.write_text(
                yaml.safe_dump({"model": {"num_rules": 4}, "datasets": [{"config": "demo.yaml"}]}),
                encoding="utf-8",
            )

            missing_name = {
                "base_experiment_config": str(base_config_path),
                "study": {"output_root_dir": "artifacts/sensitivity"},
                "sweeps": [{"name": "num_rules", "param": "model.num_rules", "values": [2, 4]}],
                "report": {"metrics": ["auc"], "primary_metric": "auc"},
            }
            blank_output_root = {
                "base_experiment_config": str(base_config_path),
                "study": {"name": "demo", "output_root_dir": "  "},
                "sweeps": [{"name": "num_rules", "param": "model.num_rules", "values": [2, 4]}],
                "report": {"metrics": ["auc"], "primary_metric": "auc"},
            }

            with self.assertRaisesRegex(ValueError, "study"):
                validate_sensitivity_config(missing_name)
            with self.assertRaisesRegex(ValueError, "study"):
                validate_sensitivity_config(blank_output_root)

    def test_validate_rejects_multi_parameter_and_unknown_sweep_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            base_config_path = root / "base.yaml"
            base_config_path.write_text(
                yaml.safe_dump({"model": {"num_rules": 4}, "datasets": [{"config": "demo.yaml"}]}),
                encoding="utf-8",
            )

            multi_param_config = {
                "base_experiment_config": str(base_config_path),
                "study": {"name": "demo", "output_root_dir": "artifacts/sensitivity"},
                "sweeps": [
                    {
                        "name": "num_rules",
                        "param": "model.num_rules",
                        "params": ["model.num_rules", "training.lr"],
                        "values": [2, 4],
                    }
                ],
                "report": {"metrics": ["auc"], "primary_metric": "auc"},
            }
            unknown_field_config = {
                "base_experiment_config": str(base_config_path),
                "study": {"name": "demo", "output_root_dir": "artifacts/sensitivity"},
                "sweeps": [
                    {
                        "name": "num_rules",
                        "param": "model.num_rules",
                        "values": [2, 4],
                        "unexpected": True,
                    }
                ],
                "report": {"metrics": ["auc"], "primary_metric": "auc"},
            }

            with self.assertRaisesRegex(ValueError, "multiple parameters|unsupported"):
                validate_sensitivity_config(multi_param_config)
            with self.assertRaisesRegex(ValueError, "unsupported"):
                validate_sensitivity_config(unknown_field_config)

    def test_committed_smoke_sensitivity_config_expands_with_relative_base_path(self):
        project_root = Path(__file__).resolve().parents[1]
        config_path = (
            project_root / "configs" / "sensitivity" / "ds_tsk_mil_smoke_sensitivity.yaml"
        )

        config = load_sensitivity_config(config_path)
        runs = expand_sensitivity_runs(config)

        self.assertEqual(16, len(runs))
        self.assertEqual(2, runs[0]["resolved_config"]["model"]["num_rules"])


if __name__ == "__main__":
    unittest.main()
