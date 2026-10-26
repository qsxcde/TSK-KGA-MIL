from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

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


def _load_font(size: int, *, bold: bool = True) -> ImageFont.ImageFont:
    candidates = ["DejaVuSans-Bold.ttf", "Arial Bold.ttf"] if bold else ["DejaVuSans.ttf", "Arial.ttf"]
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
    for study_dir in sorted(path for path in studies_root.iterdir() if path.is_dir() and path.name != "combined_plots"):
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
    y: float,
    color: str,
    font: ImageFont.ImageFont,
) -> None:
    text_width, text_height = _text_size(draw, text, font)
    label_x = x + 8
    label_y = y - text_height - 4
    draw.rounded_rectangle(
        (label_x - 4, label_y - 2, label_x + text_width + 4, label_y + text_height + 2),
        radius=6,
        fill="#ffffff",
        outline=color,
        width=1,
    )
    draw.text((label_x, label_y), text, fill=color, font=font)


def _draw_combined_plot(image_path: Path, sweep_name: str, series_map: dict[str, list[dict[str, float | str]]]) -> None:
    width = 1280
    height = 760
    margin_left = 110
    margin_right = 40
    margin_top = 70
    margin_bottom = 145

    title_font = _load_font(34, bold=True)
    tick_font = _load_font(18, bold=True)
    legend_font = _load_font(20, bold=True)
    annotation_font = _load_font(18, bold=True)

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
                _draw_series_label(draw, study_name, x=x, y=y, color=color, font=annotation_font)

    title = f"{sweep_name} Combined Comparison"
    title_width, _ = _text_size(draw, title, title_font)
    draw.text(((width - title_width) / 2, 20), title, fill=PLOT_THEME["title"], font=title_font)

    legend_width, _ = _compute_legend_box_size(draw, legend_entries, legend_font)
    legend_x = left + int((plot_width - legend_width) / 2)
    legend_y = bottom + 34
    _draw_legend(draw, legend_entries, x=legend_x, y=legend_y, font=legend_font)
    image.save(image_path, format="PNG")


def build_combined_sensitivity_plots(studies_root: Path, *, output_dir: Path | None = None) -> list[str]:
    studies_root = Path(studies_root)
    target_dir = Path(output_dir) if output_dir is not None else studies_root / "combined_plots"
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
    return output_paths
