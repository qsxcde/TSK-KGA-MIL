from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.sensitivity.combined_plots import build_combined_sensitivity_plots


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build combined ACC sensitivity plots across multiple study directories."
    )
    parser.add_argument(
        "--studies-root",
        default="artifacts/sensitivity",
        help="Directory containing multiple sensitivity study outputs such as 704, 704_1, 704_2, 704_3.",
    )
    parser.add_argument(
        "--output-dir",
        help="Optional output directory for combined plots. Defaults to <studies-root>/combined_plots.",
    )
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    studies_root = Path(args.studies_root).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else None
    plot_paths = build_combined_sensitivity_plots(studies_root, output_dir=output_dir)
    print(json.dumps({"plot_paths": plot_paths}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
