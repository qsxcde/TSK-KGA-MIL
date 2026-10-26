import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from scripts.run_sensitivity_analysis import build_argument_parser
from src.analysis.sensitivity.config import load_sensitivity_config
from src.analysis.sensitivity.grid import expand_sensitivity_runs
from src.analysis.sensitivity.runner import run_single_sensitivity_item, run_sensitivity_study


def _write_study_files(
    root: Path,
    *,
    continue_on_error: bool,
    resume: bool = False,
) -> Path:
    dataset_config_path = root / "dataset.yaml"
    dataset_config_path.write_text(
        yaml.safe_dump({"name": "demo-dataset", "path": "data/demo.mat"}, sort_keys=False),
        encoding="utf-8",
    )
    dataset_ref = "dataset.yaml"

    base_config_path = root / "base.yaml"
    base_config_path.write_text(
        yaml.safe_dump(
            {
                "algorithm": {"name": "ds_tsk_mil"},
                "model": {
                    "num_rules": 8,
                    "rule_output_dim": 1,
                    "aggregation_method": "attention",
                    "hidden_dim": 32,
                    "fcm_m": 2.0,
                    "fcm_max_iter": 10,
                    "rule_temperature": 0.5,
                    "use_competitive_scoring": True,
                },
                "training": {
                    "batch_size": 1,
                    "lr": 0.001,
                    "max_epoch": 2,
                    "patience": 2,
                    "device": "cpu",
                },
                "cross_validation": {"n_splits": 2, "random_state": 42},
                "output": {
                    "root_dir": ".",
                    "run_name": None,
                    "explanation_samples_per_fold": 2,
                },
                "datasets": [{"config": dataset_ref}],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    study_config_path = root / "study.yaml"
    study_config_path.write_text(
        yaml.safe_dump(
            {
                "base_experiment_config": "base.yaml",
                "study": {
                    "name": "demo-study",
                    "output_root_dir": str(root / "artifacts" / "sensitivity"),
                    "continue_on_error": continue_on_error,
                    "resume": resume,
                },
                "sweeps": [
                    {"name": "num_rules", "param": "model.num_rules", "values": [2, 4]},
                ],
                "report": {
                    "metrics": ["auc", "f1", "acc"],
                    "primary_metric": "auc",
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return study_config_path


class SensitivityRunnerTests(unittest.TestCase):
    @patch("src.analysis.sensitivity.runner.run_kfold_experiment")
    def test_runner_writes_run_result_json_for_expand_runs_contract(self, mock_run_kfold):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            study_config_path = _write_study_files(root, continue_on_error=True)
            study_config = load_sensitivity_config(study_config_path)
            run_spec = expand_sensitivity_runs(study_config)[1]
            study_output_dir = root / "artifacts" / "sensitivity" / "demo-study"
            mock_run_kfold.return_value = {
                "dataset_name": "demo-dataset",
                "mean_metrics": {"acc": 0.8, "f1": 0.7, "auc": 0.9},
                "std_metrics": {"acc": 0.1, "f1": 0.2, "auc": 0.05},
                "best_epochs": [2, 3],
                "training_time_sec": 1.5,
                "artifacts_path": "artifacts/demo",
            }

            result = run_single_sensitivity_item(
                run_spec,
                {
                    "study_name": "demo-study",
                    "study_dir": study_output_dir,
                },
            )

            self.assertEqual("success", result["status"])
            runtime_config = mock_run_kfold.call_args[0][0]
            self.assertEqual(4, runtime_config["num_rules"])
            self.assertEqual("demo-dataset", runtime_config["dataset_name"])
            run_result = study_output_dir / "runs" / "num_rules__4" / "run_result.json"
            self.assertTrue(run_result.exists())
            payload = json.loads(run_result.read_text(encoding="utf-8"))
            self.assertEqual(4, payload["param_value"])
            self.assertEqual(2.5, payload["best_epoch_mean"])
            self.assertEqual(2, payload["n_splits"])
            self.assertEqual(42, payload["random_state"])

    @patch("src.analysis.sensitivity.runner.run_kfold_experiment")
    def test_study_continues_on_error_and_persists_setup_failures(self, mock_run_kfold):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            study_config_path = _write_study_files(root, continue_on_error=True)
            base_config_path = root / "base.yaml"
            base_config = yaml.safe_load(base_config_path.read_text(encoding="utf-8"))
            base_config["datasets"] = [{}]
            base_config_path.write_text(yaml.safe_dump(base_config, sort_keys=False), encoding="utf-8")

            def side_effect(config):
                return {
                    "dataset_name": config["dataset_name"],
                    "mean_metrics": {"acc": 0.8, "f1": 0.7, "auc": 0.9},
                    "std_metrics": {"acc": 0.1, "f1": 0.2, "auc": 0.05},
                    "best_epochs": [2, 3],
                    "training_time_sec": 1.5,
                    "artifacts_path": str(Path(config["output_root_dir"]) / "artifacts"),
                }

            mock_run_kfold.side_effect = side_effect

            result = run_sensitivity_study(study_config_path)

            self.assertEqual("completed_with_failures", result["status"])
            self.assertEqual(2, result["total_runs"])
            self.assertEqual(0, result["success_count"])
            self.assertEqual(2, result["failure_count"])
            self.assertTrue(Path(result["manifest_path"]).exists())

            study_dir = Path(result["study_dir"])
            failed_payload = json.loads(
                (study_dir / "runs" / "num_rules__2" / "run_result.json").read_text(encoding="utf-8")
            )
            self.assertEqual("failed", failed_payload["status"])
            self.assertEqual("ValueError", failed_payload["error_type"])
            self.assertIn("config", failed_payload["error_message"])
            self.assertEqual(2, failed_payload["n_splits"])
            self.assertEqual(42, failed_payload["random_state"])
            self.assertEqual(0, mock_run_kfold.call_count)

            manifest_payload = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
            self.assertEqual("demo-study", manifest_payload["study_name"])
            self.assertEqual(2, manifest_payload["total_runs"])
            self.assertEqual("completed_with_failures", manifest_payload["status"])

    @patch("src.analysis.sensitivity.runner.run_kfold_experiment")
    def test_study_raises_when_continue_on_error_is_disabled(self, mock_run_kfold):
        with tempfile.TemporaryDirectory() as tmpdir:
            study_config_path = _write_study_files(Path(tmpdir), continue_on_error=False)
            mock_run_kfold.side_effect = RuntimeError("boom")

            with self.assertRaises(RuntimeError):
                run_sensitivity_study(study_config_path)

    @patch("src.analysis.sensitivity.runner.run_kfold_experiment")
    def test_study_resume_counts_existing_success_in_manifest(self, mock_run_kfold):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            study_config_path = _write_study_files(root, continue_on_error=True, resume=True)
            study_dir = root / "artifacts" / "sensitivity" / "demo-study"
            resumed_run_dir = study_dir / "runs" / "num_rules__2"
            resumed_run_dir.mkdir(parents=True)
            (resumed_run_dir / "run_result.json").write_text(
                json.dumps(
                    {
                        "study_name": "demo-study",
                        "sweep_name": "num_rules",
                        "param_path": "model.num_rules",
                        "param_value": 2,
                        "run_id": "num_rules__2",
                        "dataset_name": "demo-dataset",
                        "status": "success",
                        "mean_acc": 0.81,
                        "mean_f1": 0.71,
                        "mean_auc": 0.91,
                        "std_acc": 0.01,
                        "std_f1": 0.02,
                        "std_auc": 0.03,
                        "best_epoch_mean": 2.0,
                        "training_time_sec": 1.0,
                        "artifacts_path": "artifacts/existing",
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            mock_run_kfold.return_value = {
                "dataset_name": "demo-dataset",
                "mean_metrics": {"acc": 0.8, "f1": 0.7, "auc": 0.9},
                "std_metrics": {"acc": 0.1, "f1": 0.2, "auc": 0.05},
                "best_epochs": [2, 3],
                "training_time_sec": 1.5,
                "artifacts_path": "artifacts/demo",
            }

            result = run_sensitivity_study(study_config_path)

            self.assertEqual("completed", result["status"])
            self.assertEqual(2, result["total_runs"])
            self.assertEqual(2, result["success_count"])
            self.assertEqual(0, result["failure_count"])
            self.assertEqual(1, result["skipped_count"])
            self.assertEqual(1, mock_run_kfold.call_count)

    @patch("src.analysis.sensitivity.runner.run_kfold_experiment")
    @patch("src.analysis.sensitivity.runner.load_experiment_config")
    def test_study_loads_resolved_experiment_config_through_existing_loader(
        self, mock_load_experiment_config, mock_run_kfold
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            study_config_path = _write_study_files(root, continue_on_error=True)
            mock_load_experiment_config.return_value = {
                "model": {
                    "num_rules": 2,
                    "rule_output_dim": 1,
                    "aggregation_method": "attention",
                    "hidden_dim": 32,
                    "fcm_m": 2.0,
                    "fcm_max_iter": 10,
                    "rule_temperature": 0.5,
                    "use_competitive_scoring": True,
                },
                "training": {
                    "batch_size": 1,
                    "lr": 0.001,
                    "max_epoch": 2,
                    "patience": 2,
                    "device": "cpu",
                },
                "cross_validation": {"n_splits": 2, "random_state": 42},
                "output": {
                    "root_dir": ".",
                    "run_name": None,
                    "explanation_samples_per_fold": 2,
                },
                "dataset_entries": [
                    {
                        "dataset": {"name": "demo-dataset", "path": "data/demo.mat"},
                        "training": {"max_epoch": 3},
                        "cross_validation": {},
                        "output": {"run_name": "entry-override"},
                    }
                ],
            }
            mock_run_kfold.return_value = {
                "dataset_name": "demo-dataset",
                "mean_metrics": {"acc": 0.8, "f1": 0.7, "auc": 0.9},
                "std_metrics": {"acc": 0.1, "f1": 0.2, "auc": 0.05},
                "best_epochs": [2, 3],
                "training_time_sec": 1.5,
                "artifacts_path": "artifacts/demo",
            }

            run_sensitivity_study(study_config_path)

            loaded_path = Path(mock_load_experiment_config.call_args[0][0]).resolve()
            self.assertEqual(root.resolve(), loaded_path.parent)
            runtime_config = mock_run_kfold.call_args[0][0]
            self.assertEqual(3, runtime_config["max_epoch"])
            self.assertIsNone(runtime_config["run_name"])

    @patch("src.analysis.sensitivity.runner.build_sensitivity_html")
    @patch("src.analysis.sensitivity.runner.build_sensitivity_plots")
    @patch("src.analysis.sensitivity.runner.build_sensitivity_tables")
    @patch("src.analysis.sensitivity.runner.run_kfold_experiment")
    def test_study_builds_tables_plots_and_html_after_runs(
        self,
        mock_run_kfold,
        mock_build_tables,
        mock_build_plots,
        mock_build_html,
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            study_config_path = _write_study_files(root, continue_on_error=True)
            study_dir = root / "artifacts" / "sensitivity" / "demo-study"
            mock_run_kfold.return_value = {
                "dataset_name": "demo-dataset",
                "mean_metrics": {"acc": 0.8, "f1": 0.7, "auc": 0.9},
                "std_metrics": {"acc": 0.1, "f1": 0.2, "auc": 0.05},
                "best_epochs": [2, 3],
                "training_time_sec": 1.5,
                "artifacts_path": "artifacts/demo",
            }
            mock_build_tables.return_value = {"summary_by_value": str(study_dir / "tables" / "summary_by_value.csv")}
            mock_build_plots.return_value = [str(study_dir / "plots" / "num_rules_auc.png")]
            mock_build_html.return_value = study_dir / "report" / "index.html"

            result = run_sensitivity_study(study_config_path)

            self.assertEqual(study_dir.resolve(), mock_build_tables.call_args.args[0])
            self.assertEqual(study_dir.resolve(), mock_build_plots.call_args.args[0])
            self.assertEqual("auc", mock_build_plots.call_args.args[1])
            self.assertEqual(["auc", "f1", "acc"], mock_build_plots.call_args.kwargs["metrics"])
            self.assertEqual(study_dir.resolve(), mock_build_html.call_args.args[0])
            self.assertEqual(mock_build_tables.return_value, result["table_paths"])
            self.assertEqual(mock_build_plots.return_value, result["plot_paths"])
            self.assertEqual(str(mock_build_html.return_value), result["report_path"])

    @patch("src.analysis.sensitivity.runner.build_sensitivity_html")
    @patch("src.analysis.sensitivity.runner.build_sensitivity_plots")
    @patch("src.analysis.sensitivity.runner.build_sensitivity_tables")
    @patch("src.analysis.sensitivity.runner.run_kfold_experiment")
    def test_study_skips_optional_report_outputs_when_disabled(
        self,
        mock_run_kfold,
        mock_build_tables,
        mock_build_plots,
        mock_build_html,
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            study_config_path = _write_study_files(root, continue_on_error=True)
            study_config = yaml.safe_load(study_config_path.read_text(encoding="utf-8"))
            study_config["report"]["generate_plots"] = False
            study_config["report"]["generate_html"] = False
            study_config_path.write_text(yaml.safe_dump(study_config, sort_keys=False), encoding="utf-8")
            study_dir = root / "artifacts" / "sensitivity" / "demo-study"
            mock_run_kfold.return_value = {
                "dataset_name": "demo-dataset",
                "mean_metrics": {"acc": 0.8, "f1": 0.7, "auc": 0.9},
                "std_metrics": {"acc": 0.1, "f1": 0.2, "auc": 0.05},
                "best_epochs": [2, 3],
                "training_time_sec": 1.5,
                "artifacts_path": "artifacts/demo",
            }
            mock_build_tables.return_value = {"summary_by_value": str(study_dir / "tables" / "summary_by_value.csv")}

            result = run_sensitivity_study(study_config_path)

            self.assertEqual(study_dir.resolve(), mock_build_tables.call_args.args[0])
            mock_build_plots.assert_not_called()
            mock_build_html.assert_not_called()
            self.assertEqual([], result["plot_paths"])
            self.assertIsNone(result["report_path"])

    def test_cli_requires_config_argument(self):
        parser = build_argument_parser()
        args = parser.parse_args(["--config", "configs/sensitivity/demo.yaml"])
        self.assertEqual("configs/sensitivity/demo.yaml", args.config)


if __name__ == "__main__":
    unittest.main()
