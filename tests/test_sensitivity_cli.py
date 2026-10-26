import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from scripts.build_sensitivity_report import build_argument_parser as build_report_argument_parser
from scripts.build_sensitivity_report import main as build_report_main
from scripts.run_sensitivity_analysis import build_argument_parser as build_analysis_argument_parser
from scripts.run_sensitivity_analysis import main as run_analysis_main


class SensitivityCliTests(unittest.TestCase):
    def test_run_analysis_cli_requires_config_argument(self):
        parser = build_analysis_argument_parser()
        args = parser.parse_args(["--config", "configs/sensitivity/demo.yaml"])
        self.assertEqual("configs/sensitivity/demo.yaml", args.config)

    def test_build_report_cli_accepts_study_dir_without_primary_metric(self):
        parser = build_report_argument_parser()
        args = parser.parse_args(["--study-dir", "artifacts/sensitivity/demo"])
        self.assertEqual("artifacts/sensitivity/demo", args.study_dir)
        self.assertIsNone(args.primary_metric)

    def test_run_analysis_main_prints_json_payload(self):
        expected = {
            "study_name": "demo-study",
            "status": "completed",
            "total_runs": 1,
            "success_count": 1,
            "failure_count": 0,
            "skipped_count": 0,
            "manifest_path": "/tmp/demo-study/manifest.json",
        }

        with patch("scripts.run_sensitivity_analysis.run_sensitivity_study", return_value=expected) as mock_run:
            with patch.object(sys, "argv", ["run_sensitivity_analysis.py", "--config", "configs/sensitivity/demo.yaml"]):
                with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                    run_analysis_main()

        self.assertEqual("configs/sensitivity/demo.yaml", mock_run.call_args.args[0])
        self.assertEqual(expected, json.loads(stdout.getvalue()))

    def test_build_report_main_prints_json_payload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            study_dir = root / "artifacts" / "sensitivity" / "demo"
            study_dir.mkdir(parents=True)
            study_config_path = root / "study.yaml"
            study_config_path.write_text(
                yaml.safe_dump(
                    {
                        "base_experiment_config": "base.yaml",
                        "study": {"name": "demo", "output_root_dir": str(root / "artifacts" / "sensitivity")},
                        "sweeps": [{"name": "num_rules", "param": "model.num_rules", "values": [2, 4]}],
                        "report": {"metrics": ["auc", "f1"], "primary_metric": "auc"},
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            (study_dir / "manifest.json").write_text(
                json.dumps({"config_path": str(study_config_path)}, indent=2),
                encoding="utf-8",
            )

            table_paths = {
                "raw_runs": str(study_dir / "tables" / "raw_runs.csv"),
                "summary_by_value": str(study_dir / "tables" / "summary_by_value.csv"),
                "summary_by_parameter": str(study_dir / "tables" / "summary_by_parameter.csv"),
                "failure_report": str(study_dir / "tables" / "failure_report.csv"),
            }
            plot_paths = [str(study_dir / "plots" / "num_rules_auc.png")]
            report_path = study_dir / "report" / "index.html"

            with patch("scripts.build_sensitivity_report.build_sensitivity_tables", return_value=table_paths) as mock_tables:
                with patch("scripts.build_sensitivity_report.build_sensitivity_plots", return_value=plot_paths) as mock_plots:
                    with patch("scripts.build_sensitivity_report.build_sensitivity_html", return_value=report_path) as mock_html:
                        with patch.object(
                            sys,
                            "argv",
                            [
                                "build_sensitivity_report.py",
                                "--study-dir",
                                str(study_dir),
                            ],
                        ):
                            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                                build_report_main()

            self.assertEqual(study_dir.resolve(), mock_tables.call_args.args[0])
            self.assertEqual("auc", mock_tables.call_args.kwargs["primary_metric"])
            self.assertEqual(study_dir.resolve(), mock_plots.call_args.args[0])
            self.assertEqual("auc", mock_plots.call_args.args[1])
            self.assertEqual(["auc", "f1"], mock_plots.call_args.kwargs["metrics"])
            self.assertEqual(study_dir.resolve(), mock_html.call_args.args[0])
            self.assertEqual(table_paths, mock_html.call_args.args[1])
            self.assertEqual(plot_paths, mock_html.call_args.args[2])
            self.assertEqual(
                {
                    "study_dir": str(study_dir.resolve()),
                    "table_paths": table_paths,
                    "plot_paths": plot_paths,
                    "report_path": str(report_path),
                },
                json.loads(stdout.getvalue()),
            )

    def test_build_report_main_rejects_unsupported_primary_metric(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            study_dir = Path(tmpdir) / "study"
            study_dir.mkdir(parents=True)

            with patch.object(
                sys,
                "argv",
                [
                    "build_sensitivity_report.py",
                    "--study-dir",
                    str(study_dir),
                    "--primary-metric",
                    "precision",
                ],
            ):
                with self.assertRaisesRegex(ValueError, "Unsupported primary report metric"):
                    build_report_main()


if __name__ == "__main__":
    unittest.main()
