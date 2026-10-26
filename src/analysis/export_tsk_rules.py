"""Export TSK rule parameters from DS-TSK-MIL checkpoint files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")

    if not isinstance(checkpoint, dict):
        raise ValueError(f"{path} is not a PyTorch state dictionary.")

    for key in ("state_dict", "model_state_dict"):
        nested = checkpoint.get(key)
        if isinstance(nested, dict):
            return nested

    return checkpoint


def extract_tsk_rules(state_dict: dict[str, Any]) -> dict[str, np.ndarray]:
    """Extract TSK centers, widths, and consequent parameters from a state dict."""
    missing = [
        key
        for key in ("tsk_layer.means", "tsk_layer.vars")
        if key not in state_dict
    ]
    if missing:
        raise ValueError(f"Missing required TSK parameter(s): {', '.join(missing)}")

    centers = _as_numpy(state_dict["tsk_layer.means"]).astype(np.float32, copy=False)
    vars_ = _as_numpy(state_dict["tsk_layer.vars"]).astype(np.float32, copy=False)
    widths = np.sqrt(np.maximum(vars_, 0.0)).astype(np.float32, copy=False)

    if centers.ndim != 2:
        raise ValueError(f"Expected tsk_layer.means to be 2-D, got shape {centers.shape}.")
    if vars_.shape != centers.shape:
        raise ValueError(
            "Expected tsk_layer.vars to have the same shape as tsk_layer.means, "
            f"got {vars_.shape} and {centers.shape}."
        )

    consequent_weights = []
    consequent_biases = []
    for rule_idx in range(centers.shape[0]):
        weight_key = f"tsk_layer.consequent.{rule_idx}.weight"
        bias_key = f"tsk_layer.consequent.{rule_idx}.bias"
        if weight_key not in state_dict or bias_key not in state_dict:
            raise ValueError(
                f"Missing consequent parameter(s) for rule {rule_idx}: "
                f"{weight_key} / {bias_key}"
            )
        consequent_weights.append(_as_numpy(state_dict[weight_key]).astype(np.float32, copy=False))
        consequent_biases.append(_as_numpy(state_dict[bias_key]).astype(np.float32, copy=False))

    return {
        "centers": centers,
        "widths": widths,
        "consequent_weights": np.stack(consequent_weights),
        "consequent_biases": np.stack(consequent_biases),
    }


def _find_checkpoints(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(input_path.rglob("*.pth"))
    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def _output_path(checkpoint_path: Path, output_dir: Path | None) -> Path:
    target_dir = output_dir if output_dir is not None else checkpoint_path.parent
    return target_dir / f"{checkpoint_path.stem}_tsk_rules"


def _save_csv(path: Path, array: np.ndarray) -> None:
    np.savetxt(path, array, delimiter=",", fmt="%.10g")


def save_tsk_rule_values(rules: dict[str, np.ndarray], output_dir: Path) -> Path:
    """Save TSK rule values as separate readable CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    centers = rules["centers"]
    widths = rules["widths"]
    consequent_weights = rules["consequent_weights"]
    consequent_biases = rules["consequent_biases"]

    _save_csv(output_dir / "centers.csv", centers)
    _save_csv(output_dir / "widths.csv", widths)
    _save_csv(
        output_dir / "consequent_weights.csv",
        consequent_weights.reshape(consequent_weights.shape[0], -1),
    )
    _save_csv(
        output_dir / "consequent_biases.csv",
        consequent_biases.reshape(consequent_biases.shape[0], -1),
    )

    metadata = {
        "files": {
            "centers": "centers.csv",
            "widths": "widths.csv",
            "consequent_weights": "consequent_weights.csv",
            "consequent_biases": "consequent_biases.csv",
        },
        "shapes": {
            key: list(value.shape)
            for key, value in rules.items()
        },
        "notes": {
            "widths": "Computed as sqrt(tsk_layer.vars).",
            "consequent_weights": "Flattened to one row per rule for CSV export.",
            "consequent_biases": "One row per rule.",
        },
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    return output_dir


def export_checkpoint(checkpoint_path: Path, output_dir: Path | None = None) -> Path:
    state_dict = _load_checkpoint(checkpoint_path)
    rules = extract_tsk_rules(state_dict)

    output_path = _output_path(checkpoint_path, output_dir)
    return save_tsk_rule_values(rules, output_path)


def write_summary(exported_paths: list[Path], output_dir: Path) -> Path:
    summary_path = output_dir / "tsk_rules_export_manifest.json"
    records = [{"file": str(path)} for path in exported_paths]
    summary_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    return summary_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export TSK rule centers, widths, and consequents from .pth files."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="A .pth checkpoint file or a directory containing .pth files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for exported parameter folders. Defaults to each checkpoint directory.",
    )
    parser.add_argument(
        "--manifest",
        action="store_true",
        help="Write tsk_rules_export_manifest.json when --output-dir is provided.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    checkpoints = _find_checkpoints(args.input)
    if not checkpoints:
        raise ValueError(f"No .pth files found under {args.input}")

    exported_paths = [export_checkpoint(path, args.output_dir) for path in checkpoints]
    for path in exported_paths:
        print(path)

    if args.manifest:
        manifest_dir = args.output_dir or checkpoints[0].parent
        manifest_path = write_summary(exported_paths, manifest_dir)
        print(manifest_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
