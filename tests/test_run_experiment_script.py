import json
import subprocess
import sys
import unittest
from pathlib import Path


class RunExperimentScriptTest(unittest.TestCase):
    def test_smoke_config_runs_and_returns_json(self):
        project_root = Path(__file__).resolve().parents[1]
        command = [
            sys.executable,
            "scripts/run_experiment.py",
            "--config",
            "configs/experiments/ds_tsk_mil/ds_tsk_mil_smoke.yaml",
        ]

        completed = subprocess.run(
            command,
            cwd=project_root,
            capture_output=True,
            text=True,
            check=True,
        )

        report = json.loads(completed.stdout)
        self.assertIsInstance(report, list)
        self.assertEqual(report[0]["dataset"], "musk1+")
        self.assertIn("dataset_name", report[0])
        self.assertIn("fold_metrics", report[0])
        self.assertIn("mean_metrics", report[0])
        self.assertIn("std_metrics", report[0])
        self.assertIn("fold_scores", report[0])
        self.assertIn("mean_acc", report[0])
        self.assertIn("std_acc", report[0])


if __name__ == "__main__":
    unittest.main()
