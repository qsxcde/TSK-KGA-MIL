from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PERFORMANCE_METRIC_FIELDS = frozenset({"mean_acc", "mean_f1", "mean_auc"})
HEATMAP_METRIC_FIELDS = ("mean_acc", "mean_f1", "mean_auc", "training_time_sec", "best_epoch_mean")
LOWER_IS_BETTER_FIELDS = frozenset({"training_time_sec", "best_epoch_mean"})

PLOT_THEME = {
    "bg": "#ffffff",
    "title": "#111111",
    "text": "#222222",
    "muted": "#6e6e6e",
    "axis": "#666666",
    "grid": "#e6e6e6",
    "error": "#8e8e8e",
    "frame": "#bdbdbd",
}

SWEEP_COLOR_PALETTE = (
    {"line": "#1f77b4", "marker": "#124c73"},
    {"line": "#d62728", "marker": "#8b191a"},
    {"line": "#2ca02c", "marker": "#1b6d1b"},
    {"line": "#9467bd", "marker": "#5f4184"},
    {"line": "#ff7f0e", "marker": "#a55208"},
    {"line": "#17becf", "marker": "#0f7a85"},
    {"line": "#8c564b", "marker": "#5b3831"},
    {"line": "#e377c2", "marker": "#974e81"},
)


def _load_success_rows(study_dir: Path) -> list[dict]:
    rows = []
    for path in sorted((study_dir / "runs").glob("*/run_result.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") == "success":
            rows.append(payload)
    return rows


def _sort_key(row: dict) -> tuple[int, float | str]:
    value = row.get("param_value")
    if isinstance(value, bool):
        return (0, int(value))
    if isinstance(value, (int, float)):
        return (0, float(value))
    try:
        return (0, float(value))
    except (TypeError, ValueError):
        return (1, str(value))


def _metric_error_field(metric_name: str) -> str | None:
    if metric_name in PERFORMANCE_METRIC_FIELDS:
        return f"std_{metric_name[5:]}"
    return None


def _load_font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
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


def _centered_text(
    draw: ImageDraw.ImageDraw,
    center_x: float,
    y: float,
    text: str,
    *,
    font: ImageFont.ImageFont,
    fill: str,
) -> None:
    text_width, _ = _text_size(draw, text, font)
    draw.text((center_x - text_width / 2, y), text, fill=fill, font=font)


def _blend_hex(start_hex: str, end_hex: str, ratio: float) -> tuple[int, int, int]:
    ratio = max(0.0, min(1.0, ratio))
    start = tuple(int(start_hex[index : index + 2], 16) for index in (1, 3, 5))
    end = tuple(int(end_hex[index : index + 2], 16) for index in (1, 3, 5))
    return tuple(int(start_value + (end_value - start_value) * ratio) for start_value, end_value in zip(start, end))


def _metric_title(metric_name: str) -> str:
    titles = {
        "mean_acc": "Accuracy",
        "mean_f1": "F1 Score",
        "mean_auc": "AUC",
        "training_time_sec": "Training Time",
        "best_epoch_mean": "Best Epoch",
    }
    return titles.get(metric_name, metric_name.replace("_", " ").title())


def _format_metric_value(metric_name: str, value: float) -> str:
    if metric_name == "training_time_sec":
        return f"{value:.2f}s"
    if metric_name == "best_epoch_mean":
        return f"{value:.2f}"
    return f"{value:.3f}"


def _label_text(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _sweep_colors(sweep_name: str) -> dict[str, str]:
    palette_index = sum(ord(char) for char in sweep_name) % len(SWEEP_COLOR_PALETTE)
    return SWEEP_COLOR_PALETTE[palette_index]


def _draw_plot(
    image_path: Path,
    sweep_name: str,
    metric_name: str,
    rows: list[dict],
    *,
    error_values: list[float] | None = None,
) -> None:
    width = 900
    height = 600
    margin_left = 96
    margin_right = 42
    margin_top = 58
    margin_bottom = 88

    title_font = _load_font(22, bold=True)
    label_font = _load_font(16)
    tick_font = _load_font(13)

    image = Image.new("RGB", (width, height), PLOT_THEME["bg"])
    draw = ImageDraw.Draw(image)
    sweep_colors = _sweep_colors(sweep_name)

    left = margin_left
    right = width - margin_right
    top = margin_top
    bottom = height - margin_bottom
    plot_width = right - left
    plot_height = bottom - top

    values = [float(row[metric_name]) for row in rows]
    labels = [_label_text(row.get("param_value")) for row in rows]
    if error_values is None:
        lower_bounds = values
        upper_bounds = values
    else:
        lower_bounds = [value - error for value, error in zip(values, error_values)]
        upper_bounds = [value + error for value, error in zip(values, error_values)]

    min_value = min(lower_bounds)
    max_value = max(upper_bounds)
    if min_value == max_value:
        min_value -= 0.05
        max_value += 0.05
    else:
        padding = max((max_value - min_value) * 0.08, 0.01)
        min_value -= padding
        max_value += padding

    tick_count = 5
    for index in range(tick_count):
        ratio = index / (tick_count - 1)
        y = bottom - (plot_height * ratio)
        tick_value = min_value + (max_value - min_value) * ratio
        draw.line((left, y, right, y), fill=PLOT_THEME["grid"], width=1)
        tick_text = _format_metric_value(metric_name, tick_value)
        tick_width, tick_height = _text_size(draw, tick_text, tick_font)
        draw.text((left - 14 - tick_width, y - tick_height / 2), tick_text, fill=PLOT_THEME["text"], font=tick_font)

    draw.rectangle((left, top, right, bottom), outline=PLOT_THEME["frame"], width=1)
    draw.line((left, bottom, right, bottom), fill=PLOT_THEME["axis"], width=1)
    draw.line((left, top, left, bottom), fill=PLOT_THEME["axis"], width=1)

    points = []
    count = len(values)
    for index, value in enumerate(values):
        x = left + plot_width / 2 if count == 1 else left + (plot_width * index / (count - 1))
        y = bottom - ((value - min_value) / (max_value - min_value) * plot_height)
        points.append((x, y))

        draw.line((x, bottom, x, bottom + 5), fill=PLOT_THEME["axis"], width=1)
        label_y = bottom + 14
        _centered_text(draw, x, label_y, labels[index], font=tick_font, fill=PLOT_THEME["text"])

    if len(points) > 1:
        draw.line(points, fill=sweep_colors["line"], width=3)

    for index, (x, y) in enumerate(points):
        if error_values is not None:
            error = float(error_values[index])
            y_low = bottom - ((values[index] - error - min_value) / (max_value - min_value) * plot_height)
            y_high = bottom - ((values[index] + error - min_value) / (max_value - min_value) * plot_height)
            draw.line((x, y_low, x, y_high), fill=PLOT_THEME["error"], width=1)
            draw.line((x - 5, y_low, x + 5, y_low), fill=PLOT_THEME["error"], width=1)
            draw.line((x - 5, y_high, x + 5, y_high), fill=PLOT_THEME["error"], width=1)

        marker_radius = 4
        draw.ellipse(
            (x - marker_radius, y - marker_radius, x + marker_radius, y + marker_radius),
            fill=sweep_colors["marker"],
            outline=sweep_colors["marker"],
            width=1,
        )

    x_label = "Parameter Value"
    y_label = _metric_title(metric_name)
    title = f"{sweep_name} ({y_label})"
    title_width, _ = _text_size(draw, title, title_font)
    draw.text(((width - title_width) / 2, 18), title, fill=PLOT_THEME["title"], font=title_font)
    x_width, _ = _text_size(draw, x_label, label_font)
    draw.text((left + (plot_width - x_width) / 2, height - 42), x_label, fill=PLOT_THEME["text"], font=label_font)
    draw.text((18, top - 2), y_label, fill=PLOT_THEME["text"], font=label_font)
    image.save(image_path, format="PNG")


def _normalize_scores(values: list[float], *, invert: bool = False) -> list[float]:
    if not values:
        return []
    min_value = min(values)
    max_value = max(values)
    if min_value == max_value:
        base = [0.5 for _ in values]
    else:
        base = [(value - min_value) / (max_value - min_value) for value in values]
    if invert:
        return [1.0 - value for value in base]
    return base


def _draw_overview_heatmap(image_path: Path, grouped_rows: dict[str, list[dict]]) -> None:
    sweep_names = sorted(grouped_rows)
    metric_labels = ["Best ACC", "Best F1", "Best AUC", "Fastest Time", "Earliest Epoch"]
    cell_width = 164
    cell_height = 76
    table_left = 244
    table_top = 168
    width = table_left + cell_width * len(metric_labels) + 72
    height = table_top + cell_height * max(len(sweep_names), 1) + 116

    image = Image.new("RGB", (width, height), PLOT_THEME["bg"])
    draw = ImageDraw.Draw(image)
    title_font = _load_font(30, bold=True)
    subtitle_font = _load_font(16)
    header_font = _load_font(16, bold=True)
    row_font = _load_font(18, bold=True)
    cell_font = _load_font(17, bold=True)
    legend_font = _load_font(14)

    draw.rounded_rectangle((28, 28, width - 28, height - 28), radius=28, fill="#ffffff", outline=PLOT_THEME["frame"], width=2)
    draw.text((48, 50), "Sensitivity Overview Heatmap", fill=PLOT_THEME["title"], font=title_font)
    draw.text((48, 90), "Higher intensity indicates a stronger overall setting after metric normalization.", fill=PLOT_THEME["muted"], font=subtitle_font)

    aggregated_rows = []
    for sweep_name in sweep_names:
        rows = grouped_rows[sweep_name]
        aggregated_rows.append(
            {
                "sweep_name": sweep_name,
                "mean_acc": max(float(row["mean_acc"]) for row in rows),
                "mean_f1": max(float(row["mean_f1"]) for row in rows),
                "mean_auc": max(float(row["mean_auc"]) for row in rows),
                "training_time_sec": min(float(row["training_time_sec"]) for row in rows),
                "best_epoch_mean": min(float(row["best_epoch_mean"]) for row in rows),
            }
        )

    normalized_columns = {}
    for metric_name in HEATMAP_METRIC_FIELDS:
        values = [row[metric_name] for row in aggregated_rows]
        normalized_columns[metric_name] = _normalize_scores(values, invert=metric_name in LOWER_IS_BETTER_FIELDS)

    for column_index, label in enumerate(metric_labels):
        x0 = table_left + column_index * cell_width
        draw.rounded_rectangle((x0, table_top - 62, x0 + cell_width - 14, table_top - 14), radius=14, fill="#eef3f8")
        label_width, label_height = _text_size(draw, label, header_font)
        draw.text((x0 + (cell_width - 14 - label_width) / 2, table_top - 47 + (20 - label_height) / 2), label, fill=PLOT_THEME["title"], font=header_font)

    for row_index, row in enumerate(aggregated_rows):
        y0 = table_top + row_index * cell_height
        label_bg = "#f7f9fc" if row_index % 2 == 0 else "#f2f5fa"
        draw.rounded_rectangle((52, y0, table_left - 24, y0 + cell_height - 12), radius=16, fill=label_bg, outline="#e3e8f0")
        draw.text((74, y0 + 21), row["sweep_name"], fill=PLOT_THEME["title"], font=row_font)

        for column_index, metric_name in enumerate(HEATMAP_METRIC_FIELDS):
            score = normalized_columns[metric_name][row_index]
            fill_color = _blend_hex("#edf3fb", "#28527a", score)
            text_color = "#f8fbff" if score > 0.58 else "#14324a"
            x0 = table_left + column_index * cell_width
            x1 = x0 + cell_width - 14
            y1 = y0 + cell_height - 12
            draw.rounded_rectangle((x0, y0, x1, y1), radius=16, fill=fill_color, outline="#d5deea")
            value_text = _format_metric_value(metric_name, float(row[metric_name]))
            value_width, value_height = _text_size(draw, value_text, cell_font)
            draw.text((x0 + (x1 - x0 - value_width) / 2, y0 + (y1 - y0 - value_height) / 2), value_text, fill=text_color, font=cell_font)

    legend_x0 = width - 262
    legend_y0 = height - 78
    for step in range(8):
        ratio = step / 7
        color = _blend_hex("#edf3fb", "#28527a", ratio)
        draw.rounded_rectangle((legend_x0 + step * 22, legend_y0, legend_x0 + step * 22 + 20, legend_y0 + 16), radius=6, fill=color)
    draw.text((legend_x0, legend_y0 - 22), "Normalized overall quality", fill=PLOT_THEME["muted"], font=legend_font)
    draw.text((legend_x0, legend_y0 + 22), "lower", fill=PLOT_THEME["muted"], font=legend_font)
    draw.text((legend_x0 + 122, legend_y0 + 22), "higher", fill=PLOT_THEME["muted"], font=legend_font)

    image.save(image_path, format="PNG")


def _metric_field(metric: str) -> str:
    if metric.startswith("mean_") or metric in {"training_time_sec", "best_epoch_mean"}:
        return metric
    return f"mean_{metric}"


def _metric_slug(metric: str) -> str:
    return metric[5:] if metric.startswith("mean_") else metric


def build_sensitivity_plots(
    study_dir: Path,
    primary_metric: str,
    *,
    metrics: list[str] | None = None,
) -> list[str]:
    study_dir = Path(study_dir)
    plot_dir = study_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_success_rows(study_dir)
    requested_metrics = list(metrics or [primary_metric])
    if primary_metric not in requested_metrics:
        requested_metrics.insert(0, primary_metric)

    metric_fields = []
    for metric in requested_metrics + ["training_time_sec", "best_epoch_mean"]:
        field_name = _metric_field(metric)
        if field_name not in metric_fields:
            metric_fields.append(field_name)

    if rows:
        for metric_name in metric_fields:
            if any(metric_name not in row for row in rows):
                raise ValueError(f"Missing metric '{metric_name}' in sensitivity results.")

    grouped_rows: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped_rows[str(row["sweep_name"])].append(row)

    plot_paths: list[str] = []
    for sweep_name in sorted(grouped_rows):
        sweep_rows = sorted(grouped_rows[sweep_name], key=_sort_key)
        for metric_name in metric_fields:
            metric_slug = _metric_slug(metric_name)
            image_path = plot_dir / f"{sweep_name}_{metric_slug}.png"
            error_field = _metric_error_field(metric_name)
            error_values = None
            if error_field is not None:
                error_values = [float(row.get(error_field, 0.0)) for row in sweep_rows]
            _draw_plot(
                image_path,
                sweep_name,
                metric_name,
                sweep_rows,
                error_values=error_values,
            )
            plot_paths.append(str(image_path))

    if grouped_rows:
        heatmap_path = plot_dir / "overview_heatmap.png"
        _draw_overview_heatmap(heatmap_path, grouped_rows)
        plot_paths.append(str(heatmap_path))
    return plot_paths
