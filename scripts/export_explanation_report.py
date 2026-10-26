from pathlib import Path
import argparse
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import export_instance_report_csv


def main():
    parser = argparse.ArgumentParser(description="Export instance-level explanation report to CSV.")
    parser.add_argument(
        "--explanation-dir",
        required=True,
        help="Directory containing explanation .npz files and manifest JSON.",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Path to manifest JSON. Defaults to the only *_explanations_manifest.json in explanation dir.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path. Defaults to <explanation-dir>/instance_report.csv",
    )
    args = parser.parse_args()

    explanation_dir = Path(args.explanation_dir)
    if args.manifest is None:
        manifest_candidates = list(explanation_dir.glob("*_explanations_manifest.json"))
        if len(manifest_candidates) != 1:
            raise ValueError("Expected exactly one manifest file in explanation dir.")
        manifest_path = manifest_candidates[0]
    else:
        manifest_path = Path(args.manifest)

    output_path = Path(args.output) if args.output is not None else explanation_dir / "instance_report.csv"
    export_instance_report_csv(manifest_path, explanation_dir, output_path)
    print(output_path)


if __name__ == "__main__":
    main()
