from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.sensitivity.runner import run_sensitivity_study


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one-factor sensitivity analysis from a YAML config.")
    parser.add_argument("--config", required=True, help="Path to sensitivity study YAML.")
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    result = run_sensitivity_study(args.config)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
