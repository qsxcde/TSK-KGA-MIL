import csv
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from src.analysis.sensitivity.combined_plots import _compute_legend_box_size
from src.analysis.sensitivity.combined_plots import _draw_legend
from src.analysis.sensitivity.combined_plots import _load_font
from src.analysis.sensitivity.combined_plots import build_combined_sensitivity_plots


class SensitivityCombinedPlotsTests(unittest.TestCase):
    def test_compute_legend_box_size_keeps_legend_on_single_row(self):
        image = Image.new("RGB", (200, 100), "#ffffff")
        draw = ImageDraw.Draw(image)
        width, height = _compute_legend_box_size(
            draw,
            [("musk1", "#1f77b4"), ("fox", "#d62728"), ("tiger", "#2ca02c"), ("musk2", "#ff7f0e")],
            _load_font(18, bold=True),
        )

        self.assertGreater(width, 200)
        self.assertLess(height, 80)

    def test_draw_legend_does_not_paint_background_box(self):
        image = Image.new("RGB", (500, 120), "#d0d0d0")
        draw = ImageDraw.Draw(image)
        font = _load_font(18, bold=True)

        _draw_legend(
            draw,
            [("musk1", "#1f77b4"), ("fox", "#d62728"), ("tiger", "#2ca02c"), ("musk2", "#ff7f0e")],
            x=20,
            y=20,
            font=font,
        )

        self.assertEqual((208, 208, 208), image.getpixel((22, 22)))

    def test_build_combined_plots_creates_three_acc_images(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            studies_root = root / "artifacts" / "sensitivity"
            dataset_names = ["704", "704_1", "704_2", "704_3"]

            for offset, dataset_name in enumerate(dataset_names):
                table_dir = studies_root / dataset_name / "ds_tsk_mil_smoke_sensitivity" / "tables"
                table_dir.mkdir(parents=True)
                csv_path = table_dir / "summary_by_value.csv"
                with csv_path.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(
                        handle,
                        fieldnames=[
                            "sweep_name",
                            "param_path",
                            "param_value",
                            "dataset_name",
                            "mean_acc",
                            "mean_f1",
                            "mean_auc",
                            "std_acc",
                            "std_f1",
                            "std_auc",
                            "best_epoch_mean",
                            "training_time_sec",
                        ],
                    )
                    writer.writeheader()
                    writer.writerows(
                        [
                            {
                                "sweep_name": "hidden_dim",
                                "param_path": "model.hidden_dim",
                                "param_value": value,
                                "dataset_name": dataset_name,
                                "mean_acc": 0.80 + offset * 0.01 + index * 0.001,
                                "mean_f1": 0.0,
                                "mean_auc": 0.0,
                                "std_acc": 0.0,
                                "std_f1": 0.0,
                                "std_auc": 0.0,
                                "best_epoch_mean": 1.0,
                                "training_time_sec": 1.0,
                            }
                            for index, value in enumerate([32, 64])
                        ]
                        + [
                            {
                                "sweep_name": "num_rules",
                                "param_path": "model.num_rules",
                                "param_value": value,
                                "dataset_name": dataset_name,
                                "mean_acc": 0.75 + offset * 0.01 + index * 0.001,
                                "mean_f1": 0.0,
                                "mean_auc": 0.0,
                                "std_acc": 0.0,
                                "std_f1": 0.0,
                                "std_auc": 0.0,
                                "best_epoch_mean": 1.0,
                                "training_time_sec": 1.0,
                            }
                            for index, value in enumerate([2, 4])
                        ]
                        + [
                            {
                                "sweep_name": "lr",
                                "param_path": "training.lr",
                                "param_value": value,
                                "dataset_name": dataset_name,
                                "mean_acc": 0.70 + offset * 0.01 + index * 0.001,
                                "mean_f1": 0.0,
                                "mean_auc": 0.0,
                                "std_acc": 0.0,
                                "std_f1": 0.0,
                                "std_auc": 0.0,
                                "best_epoch_mean": 1.0,
                                "training_time_sec": 1.0,
                            }
                            for index, value in enumerate([0.0001, 0.001])
                        ]
                    )

            output_paths = build_combined_sensitivity_plots(studies_root, output_dir=studies_root / "combined_plots")

            self.assertEqual(
                [
                    str(studies_root / "combined_plots" / "hidden_dim_acc_combined.png"),
                    str(studies_root / "combined_plots" / "lr_acc_combined.png"),
                    str(studies_root / "combined_plots" / "num_rules_acc_combined.png"),
                ],
                sorted(output_paths),
            )
            for path in output_paths:
                image_path = Path(path)
                self.assertTrue(image_path.exists())
                self.assertGreater(image_path.stat().st_size, 0)

    def test_build_combined_plots_uses_dataset_aliases(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            studies_root = root / "artifacts" / "sensitivity"
            dataset_names = ["704", "704_1", "704_2", "704_3"]

            for dataset_name in dataset_names:
                table_dir = studies_root / dataset_name / "ds_tsk_mil_smoke_sensitivity" / "tables"
                table_dir.mkdir(parents=True)
                csv_path = table_dir / "summary_by_value.csv"
                with csv_path.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(
                        handle,
                        fieldnames=[
                            "sweep_name",
                            "param_path",
                            "param_value",
                            "dataset_name",
                            "mean_acc",
                            "mean_f1",
                            "mean_auc",
                            "std_acc",
                            "std_f1",
                            "std_auc",
                            "best_epoch_mean",
                            "training_time_sec",
                        ],
                    )
                    writer.writeheader()
                    writer.writerow(
                        {
                            "sweep_name": "hidden_dim",
                            "param_path": "model.hidden_dim",
                            "param_value": 32,
                            "dataset_name": dataset_name,
                            "mean_acc": 0.80,
                            "mean_f1": 0.0,
                            "mean_auc": 0.0,
                            "std_acc": 0.0,
                            "std_f1": 0.0,
                            "std_auc": 0.0,
                            "best_epoch_mean": 1.0,
                            "training_time_sec": 1.0,
                        }
                    )

            output_paths = build_combined_sensitivity_plots(studies_root, output_dir=studies_root / "combined_plots")

            self.assertTrue(any(path.endswith("hidden_dim_acc_combined.png") for path in output_paths))


if __name__ == "__main__":
    unittest.main()
