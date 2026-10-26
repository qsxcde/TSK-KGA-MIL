import tempfile
from pathlib import Path
import unittest


class YamlConfigTest(unittest.TestCase):
    def test_load_experiment_config_resolves_dataset_entries(self):
        from src.utils.config import load_experiment_config

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_dir = root / "datasets"
            experiment_dir = root / "experiments"
            dataset_dir.mkdir()
            experiment_dir.mkdir()

            (dataset_dir / "musk1.yaml").write_text(
                "name: musk1\npath: data/bechmark/musk1+.mat\n",
                encoding="utf-8",
            )
            (experiment_dir / "demo.yaml").write_text(
                "\n".join(
                    [
                        "algorithm:",
                        "  name: ds_tsk_mil",
                        "training:",
                        "  max_epoch: 2",
                        "cross_validation:",
                        "  n_splits: 2",
                        "datasets:",
                        "  - config: ../datasets/musk1.yaml",
                    ]
                ),
                encoding="utf-8",
            )

            config = load_experiment_config(experiment_dir / "demo.yaml")

            self.assertEqual(config["algorithm"]["name"], "ds_tsk_mil")
            self.assertEqual(len(config["dataset_entries"]), 1)
            self.assertEqual(config["dataset_entries"][0]["dataset"]["name"], "musk1")
            self.assertTrue(config["dataset_entries"][0]["dataset_config_path"].endswith("musk1.yaml"))


if __name__ == "__main__":
    unittest.main()
