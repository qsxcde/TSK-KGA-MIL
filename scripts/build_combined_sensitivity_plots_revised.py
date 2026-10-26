# -*- coding: utf-8 -*-
"""Revised combined sensitivity plot builder (for the Information Sciences revision).

Variant of scripts/build_combined_sensitivity_plots.py with two changes made to
address reviewer comment R4-C7 (abbreviations in figures vs. captions):

1. Plot titles now use the same wording as the manuscript captions instead of
   the raw sweep names:
       lr        -> "Learning rate"
       num_rules -> "Number of fuzzy rules"
       hidden_dim-> "Hidden dimension"
2. The right margin is widened so that the series end-tags (musk1/musk2/tiger/fox)
   are fully drawn inside the canvas. In the original 1280x760 render the tags
   ran past the right edge and were visibly truncated.

Everything else (colors, fonts, grid, legend, tick formatting) is kept identical
to the original so the regenerated figures match the manuscript style.

Outputs (into --output-dir):
    {sweep}_acc_combined.png       1280x760 master render
    {sweep}_acc_combined_doc.png   658x390 downscaled copy, ready for the Word manuscript

Usage:
    python scripts/build_combined_sensitivity_plots_revised.py
    python scripts/build_combined_sensitivity_plots_revised.py --studies-root artifacts/sensitivity --output-dir artifacts/sensitivity/combined_plots_revised

Only requires Pillow.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PLOT_THEME = {
    "bg": "#ffffff",
    "title": "#111111",
    "text": "#222222",
    "muted": "#5f5f5f",
    "axis": "#555555",
    "grid": "#e5e5e5",
    "frame": "#bdbdbd",
}

SERIES_COLORS = ("#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#17becf")
TARGET_SWEEPS = ("hidden_dim", "lr", "num_rules")
METRIC_FIELD = "mean_acc"
STUDY_LABELS = {
    "704": "musk1",
    "704_1": "fox",
    "704_2": "tiger",
    "704_3": "musk2",
}
# R4-C7: titles now mirror the wording used in the manuscript captions.
TITLE_BY_SWEEP = {
    "lr": "Learning rate",
    "num_rules": "Number of fuzzy rules",
    "hidden_dim": "Hidden dimension",
}
# The published figures used Arial Bold (resolved from Windows fonts).
BOLD_CANDIDATES = ("arialbd.ttf", "C:/Windows/Fonts/arialbd.ttf", "Arial Bold.ttf", "DejaVuSans-Bold.ttf")
REGULAR_CANDIDATES = ("arial.ttf", "C:/Windows/Fonts/arial.ttf", "Arial.ttf", "DejaVuSans.ttf")
DOC_SIZE = (658, 390)  # size of the figures embedded in the manuscript


def _load_font(size: int, *, bold: bool = True) -> ImageFont.ImageFont:
    candidates = BOLD_CANDIDATES if bold else REGULAR_CANDIDATES
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    return right - left, bottom - top


def _load_summary_rows(study_dir: Path) -> list[dict]:
    csv_path = study_dir / "ds_tsk_mil_smoke_sensitivity" / "tables" / "summary_by_value.csv"
    if not csv_path.exists():
        return []
    with csv_path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _parse_param_value(raw_value: str) -> float:
    return float(raw_value)


def _format_param_value(value: float) -> str:
    if value >= 1 and float(int(value)) == value:
        return str(int(value))
    return f"{value:.4g}"


def _collect_series(studies_root: Path) -> dict[str, dict[str, list[dict[str, float | str]]]]:
    grouped: dict[str, dict[str, list[dict[str, float | str]]]] = {sweep: defaultdict(list) for sweep in TARGET_SWEEPS}
    for study_dir in sorted(path for path in studies_root.iterdir() if path.is_dir() and path.name != "combined_plots" and path.name != "combined_plots_revised"):
        study_name = STUDY_LABELS.get(study_dir.name, study_dir.name)
        for row in _load_summary_rows(study_dir):
            sweep_name = str(row.get("sweep_name", ""))
            if sweep_name not in grouped:
                continue
            grouped[sweep_name][study_name].append(
                {
                    "param_value": _parse_param_value(str(row["param_value"])),
                    "mean_acc": float(row[METRIC_FIELD]),
                }
            )
    for sweep_name in grouped:
        for study_name in grouped[sweep_name]:
            grouped[sweep_name][study_name].sort(key=lambda item: float(item["param_value"]))
    return grouped


def _draw_legend(
    draw: ImageDraw.ImageDraw,
    entries: list[tuple[str, str]],
    *,
    x: int,
    y: int,
    font: ImageFont.ImageFont,
) -> None:
    padding = 14
    swatch_size = 16
    gap = 22
    entry_widths = [swatch_size + 10 + _text_size(draw, label, font)[0] for label, _ in entries]
    cursor_x = x + padding
    for index, (label, color) in enumerate(entries):
        row_y = y + padding
        draw.line((cursor_x, row_y + swatch_size / 2, cursor_x + swatch_size, row_y + swatch_size / 2), fill=color, width=4)
        draw.ellipse(
            (cursor_x + 4, row_y + 4, cursor_x + swatch_size - 4, row_y + swatch_size - 4),
            fill=color,
            outline=color,
        )
        draw.text((cursor_x + swatch_size + 10, row_y - 2), label, fill=PLOT_THEME["text"], font=font)
        cursor_x += entry_widths[index] + gap


def _compute_legend_box_size(
    draw: ImageDraw.ImageDraw,
    entries: list[tuple[str, str]],
    font: ImageFont.ImageFont,
) -> tuple[int, int]:
    padding = 14
    row_height = 28
    swatch_size = 16
    gap = 22
    entry_widths = [swatch_size + 10 + _text_size(draw, label, font)[0] for label, _ in entries]
    box_width = padding * 2 + sum(entry_widths) + gap * max(len(entries) - 1, 0)
    box_height = padding * 2 + row_height
    return box_width, box_height


def _draw_series_label(
    draw: ImageDraw.ImageDraw,
    text: str,
    *,
    x: float,
    box_top: float,
    color: str,
    font: ImageFont.ImageFont,
) -> None:
    text_width, text_height = _text_size(draw, text, font)
    label_x = x + 8
    draw.rounded_rectangle(
        (label_x - 4, box_top, label_x + text_width + 4, box_top + text_height + 4),
        radius=6,
        fill="#ffffff",
        outline=color,
        width=1,
    )
    draw.text((label_x, box_top + 2), text, fill=color, font=font)


def _resolve_tag_box_tops(
    draw: ImageDraw.ImageDraw,
    tag_specs: list[tuple[str, float, float, str]],
    font: ImageFont.ImageFont,
    gap: int = 2,
) -> list[tuple[str, float, float, str]]:
    """R4-C7: keep every series end-tag fully legible by pushing colliding
    tags downwards (tags are processed top-to-bottom)."""
    resolved = []
    last_bottom: float | None = None
    for name, x, y, color in sorted(tag_specs, key=lambda spec: spec[2]):
        _, text_height = _text_size(draw, name, font)
        box_top = y - text_height - 6
        if last_bottom is not None and box_top < last_bottom + gap:
            box_top = last_bottom + gap
        resolved.append((name, x, box_top, color))
        last_bottom = box_top + text_height + 4
    return resolved


def _draw_combined_plot(image_path: Path, sweep_name: str, series_map: dict[str, list[dict[str, float | str]]]) -> None:
    width = 1280
    height = 760
    margin_left = 110
    margin_top = 70
    margin_bottom = 145

    title_font = _load_font(34, bold=True)
    tick_font = _load_font(18, bold=True)
    legend_font = _load_font(20, bold=True)
    annotation_font = _load_font(18, bold=True)

    # R4-C7 fix: reserve enough room on the right so that the series end-tag
    # boxes are not clipped by the canvas edge.
    probe = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    max_tag_width = max((_text_size(probe, name, annotation_font)[0] for name in series_map), default=0)
    margin_right = max(40, max_tag_width + 18)

    image = Image.new("RGB", (width, height), PLOT_THEME["bg"])
    draw = ImageDraw.Draw(image)

    left = margin_left
    right = width - margin_right
    top = margin_top
    bottom = height - margin_bottom
    plot_width = right - left
    plot_height = bottom - top

    all_param_values = sorted({float(point["param_value"]) for series in series_map.values() for point in series})
    all_scores = [float(point["mean_acc"]) for series in series_map.values() for point in series]
    if not all_param_values or not all_scores:
        return

    min_score = min(all_scores)
    max_score = max(all_scores)
    if min_score == max_score:
        min_score -= 0.01
        max_score += 0.01
    else:
        padding = max((max_score - min_score) * 0.12, 0.01)
        min_score -= padding
        max_score += padding

    tick_count = 5
    for index in range(tick_count):
        ratio = index / (tick_count - 1)
        y = bottom - plot_height * ratio
        tick_value = min_score + (max_score - min_score) * ratio
        draw.line((left, y, right, y), fill=PLOT_THEME["grid"], width=1)
        tick_text = f"{tick_value:.3f}"
        tick_width, tick_height = _text_size(draw, tick_text, tick_font)
        draw.text((left - 14 - tick_width, y - tick_height / 2), tick_text, fill=PLOT_THEME["text"], font=tick_font)

    draw.rectangle((left, top, right, bottom), outline=PLOT_THEME["frame"], width=2)
    draw.line((left, bottom, right, bottom), fill=PLOT_THEME["axis"], width=2)
    draw.line((left, top, left, bottom), fill=PLOT_THEME["axis"], width=2)

    x_positions = {
        value: left if len(all_param_values) == 1 else left + plot_width * index / (len(all_param_values) - 1)
        for index, value in enumerate(all_param_values)
    }
    for value, x in x_positions.items():
        draw.line((x, bottom, x, bottom + 6), fill=PLOT_THEME["axis"], width=2)
        label = _format_param_value(value)
        label_width, _ = _text_size(draw, label, tick_font)
        draw.text((x - label_width / 2, bottom + 16), label, fill=PLOT_THEME["text"], font=tick_font)

    legend_entries: list[tuple[str, str]] = []
    tag_specs: list[tuple[str, float, float, str]] = []
    for color_index, study_name in enumerate(sorted(series_map)):
        color = SERIES_COLORS[color_index % len(SERIES_COLORS)]
        legend_entries.append((study_name, color))
        points = []
        for point in series_map[study_name]:
            x = x_positions[float(point["param_value"])]
            y = bottom - ((float(point["mean_acc"]) - min_score) / (max_score - min_score) * plot_height)
            points.append((x, y))
        if len(points) > 1:
            draw.line(points, fill=color, width=4)
        for point_index, (x, y) in enumerate(points):
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color, outline=color)
            if point_index == len(points) - 1:
                tag_specs.append((study_name, x, y, color))

    # R4-C7: draw the series end-tags after all series so that overlapping
    # tags (near-identical end values) are de-collided and fully legible.
    for name, x, box_top, color in _resolve_tag_box_tops(draw, tag_specs, annotation_font):
        _draw_series_label(draw, name, x=x, box_top=box_top, color=color, font=annotation_font)

    title = TITLE_BY_SWEEP.get(sweep_name, f"{sweep_name} Combined Comparison")
    title_width, _ = _text_size(draw, title, title_font)
    draw.text(((width - title_width) / 2, 20), title, fill=PLOT_THEME["title"], font=title_font)

    legend_width, _ = _compute_legend_box_size(draw, legend_entries, legend_font)
    legend_x = left + int((plot_width - legend_width) / 2)
    legend_y = bottom + 34
    _draw_legend(draw, legend_entries, x=legend_x, y=legend_y, font=legend_font)
    image.save(image_path, format="PNG")


def build_combined_sensitivity_plots(studies_root: Path, *, output_dir: Path | None = None) -> list[str]:
    studies_root = Path(studies_root)
    target_dir = Path(output_dir) if output_dir is not None else studies_root / "combined_plots_revised"
    target_dir.mkdir(parents=True, exist_ok=True)

    grouped_series = _collect_series(studies_root)
    output_paths: list[str] = []
    for sweep_name in TARGET_SWEEPS:
        series_map = grouped_series.get(sweep_name, {})
        if not series_map:
            continue
        image_path = target_dir / f"{sweep_name}_acc_combined.png"
        _draw_combined_plot(image_path, sweep_name, series_map)
        output_paths.append(str(image_path))
        # manuscript-ready downscaled copy (same size as the figures in the Word file)
        doc_image = Image.open(image_path).convert("RGB").resize(DOC_SIZE, Image.LANCZOS)
        doc_path = target_dir / f"{sweep_name}_acc_combined_doc.png"
        doc_image.save(doc_path, format="PNG")
        output_paths.append(str(doc_path))
    return output_paths


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build revised combined ACC sensitivity plots (R4-C7) across multiple study directories."
    )
    parser.add_argument(
        "--studies-root",
        default=str(PROJECT_ROOT / "artifacts" / "sensitivity"),
        help="Directory containing multiple sensitivity study outputs such as 704, 704_1, 704_2, 704_3.",
    )
    parser.add_argument(
        "--output-dir",
        help="Optional output directory for revised plots. Defaults to <studies-root>/combined_plots_revised.",
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
