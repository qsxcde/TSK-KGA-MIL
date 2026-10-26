from pathlib import Path
import unittest

from src.utils import load_experiment_config


class ExperimentYamlInventoryTest(unittest.TestCase):
    def test_benchmark_all_config_covers_all_benchmark_datasets(self):
        project_root = Path(__file__).resolve().parents[1]
        config = load_experiment_config(
            project_root / "configs" / "experiments" / "ds_tsk_mil" / "ds_tsk_mil_benchmark_all.yaml"
        )
        expected = sorted((project_root / "configs" / "datasets" / "benchmark").glob("*.yaml"))
        resolved = sorted(Path(entry["dataset_config_path"]) for entry in config["dataset_entries"])
        self.assertEqual(expected, resolved)

    def test_newsgroups_all_config_covers_all_newsgroups_datasets(self):
        project_root = Path(__file__).resolve().parents[1]
        config = load_experiment_config(
            project_root / "configs" / "experiments" / "ds_tsk_mil" / "ds_tsk_mil_newsgroups_all.yaml"
        )
        expected = sorted((project_root / "configs" / "datasets" / "newsgroups").glob("*.yaml"))
        resolved = sorted(Path(entry["dataset_config_path"]) for entry in config["dataset_entries"])
        self.assertEqual(expected, resolved)

    def test_smoke_config_targets_the_smoke_dataset_inventory(self):
        project_root = Path(__file__).resolve().parents[1]
        config = load_experiment_config(
            project_root / "configs" / "experiments" / "ds_tsk_mil" / "ds_tsk_mil_smoke.yaml"
        )
        expected = [
            project_root / "configs" / "datasets" / "benchmark" / "musk1.yaml",
        ]
        resolved = sorted(Path(entry["dataset_config_path"]) for entry in config["dataset_entries"])
        self.assertEqual(expected, resolved)


if __name__ == "__main__":
    unittest.main()
