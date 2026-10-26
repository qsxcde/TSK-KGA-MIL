from pathlib import Path
import unittest


class DatasetYamlInventoryTest(unittest.TestCase):
    def test_dataset_yaml_inventory_covers_available_mat_files(self):
        project_root = Path(__file__).resolve().parents[1]
        benchmark_mat = sorted((project_root / "data" / "bechmark").glob("*.mat"))
        newsgroups_mat = sorted((project_root / "data" / "newsgroups").glob("*.mat"))

        benchmark_yaml = sorted((project_root / "configs" / "datasets" / "benchmark").glob("*.yaml"))
        newsgroups_yaml = sorted((project_root / "configs" / "datasets" / "newsgroups").glob("*.yaml"))

        self.assertEqual(len(benchmark_mat), len(benchmark_yaml))
        self.assertEqual(len(newsgroups_mat), len(newsgroups_yaml))


if __name__ == "__main__":
    unittest.main()
