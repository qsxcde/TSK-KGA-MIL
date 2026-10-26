from __future__ import annotations

import csv
import json
from html import escape
from pathlib import Path


def _relative_href(base_dir: Path, target: str | Path) -> str:
    target_path = Path(target)
    return escape(target_path.relative_to(base_dir).as_posix())


def _read_csv_rows(path: str | Path) -> list[dict]:
    if not path:
        return []
    csv_path = Path(path)
    if not csv_path.exists() or csv_path.is_dir() or not csv_path.read_text(encoding="utf-8").strip():
        return []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_manifest(study_dir: Path) -> dict:
    manifest_path = study_dir / "manifest.json"
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _metric_from_parameter_summary(rows: list[dict]) -> str:
    if not rows:
        return "auc"
    for field_name in rows[0].keys():
        if field_name.startswith("best_mean_"):
            return field_name.removeprefix("best_mean_")
    return "auc"


def _format_label(label: str) -> str:
    return label.replace("_", " ").title()


def _format_value(value: object) -> str:
    if value in (None, ""):
        return "n/a"
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(numeric_value) >= 100 or numeric_value.is_integer():
        return f"{numeric_value:.0f}"
    return f"{numeric_value:.3f}"


def _render_definition_grid(items: list[tuple[str, str]]) -> str:
    cards = "".join(
        f"""
        <div class="summary-item">
          <span class="summary-label">{escape(label)}</span>
          <strong class="summary-value">{escape(value)}</strong>
        </div>
        """
        for label, value in items
    )
    return f'<div class="summary-grid">{cards}</div>'


def _render_table(rows: list[dict], *, columns: list[str]) -> str:
    if not rows:
        return '<p class="empty-state">No data available.</p>'
    header = "".join(f"<th>{escape(_format_label(column))}</th>" for column in columns)
    body_rows = []
    for row in rows:
        cells = []
        for column in columns:
            value = row.get(column, "")
            class_name = "numeric" if isinstance(value, (int, float)) or str(value).replace(".", "", 1).isdigit() else ""
            cells.append(f'<td class="{class_name}">{escape(str(value))}</td>')
        body_rows.append("<tr>" + "".join(cells) + "</tr>")
    return f'<div class="table-wrap"><table><thead><tr>{header}</tr></thead><tbody>{"".join(body_rows)}</tbody></table></div>'


def _render_plot_gallery(study_dir: Path, plot_paths: list[str]) -> str:
    if not plot_paths:
        return '<p class="empty-state">No plots available.</p>'
    figures = []
    for path in plot_paths:
        file_name = Path(path).name
        figures.append(
            f"""
            <figure class="plot-card">
              <img src="../{_relative_href(study_dir, path)}" alt="{escape(file_name)}">
              <figcaption>{escape(file_name)}</figcaption>
            </figure>
            """
        )
    return f'<div class="plot-gallery">{"".join(figures)}</div>'


def _render_key_findings(
    summary_by_value_rows: list[dict],
    summary_by_parameter_rows: list[dict],
    failure_rows: list[dict],
    primary_metric: str,
) -> str:
    findings: list[str] = []
    grouped_rows: dict[str, list[dict]] = {}
    for row in summary_by_value_rows:
        grouped_rows.setdefault(str(row.get("sweep_name", "")), []).append(row)

    metric_field = f"mean_{primary_metric}"
    std_field = f"std_{primary_metric}"
    for sweep_name in sorted(grouped_rows):
        rows = grouped_rows[sweep_name]
        best_row = max(rows, key=lambda row: float(row.get(metric_field, 0.0)))
        findings.append(
            f"{sweep_name} peaks at {best_row.get('param_value')} with {primary_metric.upper()}={float(best_row.get(metric_field, 0.0)):.3f}."
        )
        if std_field in best_row and best_row.get(std_field) not in (None, ""):
            findings.append(
                f"{sweep_name} shows fold spread {float(best_row.get(std_field, 0.0)):.3f} at its best-performing setting."
            )
        costliest_row = max(rows, key=lambda row: float(row.get("training_time_sec", 0.0)))
        findings.append(
            f"{sweep_name} reaches its highest reported training time at {costliest_row.get('param_value')} ({float(costliest_row.get('training_time_sec', 0.0)):.3f}s)."
        )

    if failure_rows:
        findings.append(f"{len(failure_rows)} run(s) failed and are listed in Failures and Warnings.")

    findings = findings[:5]
    if not findings and summary_by_parameter_rows:
        findings.append("Successful runs were aggregated into the parameter summary table.")
    if not findings:
        return '<p class="empty-state">No findings available.</p>'
    items = "".join(f'<li class="finding-item">{escape(item)}</li>' for item in findings)
    return f'<ul class="finding-list">{items}</ul>'


def _render_best_summary(summary_row: dict) -> str:
    if not summary_row:
        return '<p class="empty-state">No best-summary row available.</p>'
    items = []
    for key, value in summary_row.items():
        items.append(
            f"""
            <div class="fact-item">
              <span class="fact-label">{escape(_format_label(key))}</span>
              <strong class="fact-value">{escape(_format_value(value))}</strong>
            </div>
            """
        )
    return f'<div class="facts-grid">{"".join(items)}</div>'


def build_sensitivity_html(study_dir: Path, table_paths: dict, plot_paths: list[str]) -> Path:
    study_dir = Path(study_dir)
    report_dir = study_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    manifest = _load_manifest(study_dir)
    raw_rows = _read_csv_rows(table_paths.get("raw_runs", ""))
    summary_by_value_rows = _read_csv_rows(table_paths.get("summary_by_value", ""))
    summary_by_parameter_rows = _read_csv_rows(table_paths.get("summary_by_parameter", ""))
    failure_rows = _read_csv_rows(table_paths.get("failure_report", ""))
    primary_metric = _metric_from_parameter_summary(summary_by_parameter_rows)

    datasets = sorted({str(row.get("dataset_name")) for row in raw_rows if row.get("dataset_name")})
    total_time_sec = sum(float(row.get("training_time_sec", 0.0) or 0.0) for row in summary_by_value_rows)
    total_runs = manifest.get("total_runs", len(raw_rows))
    success_count = manifest.get("success_count", sum(1 for row in raw_rows if row.get("status") == "success"))
    failure_count = manifest.get("failure_count", sum(1 for row in raw_rows if row.get("status") == "failed"))

    table_items = "".join(
        (
            f'<li><a href="../{_relative_href(study_dir, path)}">'
            f"{escape(name.replace('_', ' ').title())}</a></li>"
        )
        for name, path in table_paths.items()
    )
    overview_plots = [path for path in plot_paths if Path(path).name == "overview_heatmap.png"]
    grouped_rows: dict[str, list[dict]] = {}
    for row in summary_by_value_rows:
        grouped_rows.setdefault(str(row.get("sweep_name", "")), []).append(row)
    grouped_plots = {
        sweep_name: [
            path
            for path in plot_paths
            if Path(path).name != "overview_heatmap.png" and Path(path).name.startswith(f"{sweep_name}_")
        ]
        for sweep_name in grouped_rows
    }

    parameter_sections = []
    for sweep_name in sorted(grouped_rows):
        sweep_rows = sorted(grouped_rows[sweep_name], key=lambda row: str(row.get("param_value", "")))
        summary_row = next((row for row in summary_by_parameter_rows if row.get("sweep_name") == sweep_name), {})
        parameter_sections.append(
            f"""
            <section class="parameter-card">
              <div class="parameter-header">
                <div>
                  <h3>{escape(sweep_name)}</h3>
                  <p class="parameter-path">Parameter path: <code>{escape(str(sweep_rows[0].get("param_path", "")))}</code></p>
                </div>
                <span class="parameter-badge">{escape(str(len(sweep_rows)))} setting(s)</span>
              </div>
              {_render_plot_gallery(study_dir, grouped_plots.get(sweep_name, []))}
              <div class="card-split">
                <div>
                  <h4>Key Summary Table</h4>
                  {_render_table(
                      sweep_rows,
                      columns=[
                          "param_value",
                          f"mean_{primary_metric}",
                          f"std_{primary_metric}",
                          "training_time_sec",
                          "best_epoch_mean",
                      ],
                  )}
                </div>
                <div>
                  <h4>Best Summary Facts</h4>
                  {_render_best_summary(summary_row)}
                </div>
              </div>
            </section>
            """
        )
    parameter_sections_html = "".join(parameter_sections) or '<p class="empty-state">No parameter-level summaries available.</p>'

    resolved_config_paths = sorted((study_dir / "runs").glob("*/experiment_config_resolved.yaml"))
    reproducibility_links = "".join(
        f'<li><a href="../{_relative_href(study_dir, path)}">{escape(path.parent.name)} / experiment_config_resolved.yaml</a></li>'
        for path in resolved_config_paths
    )
    overview_items = _render_definition_grid(
        [
            ("Study name", str(manifest.get("study_name", study_dir.name))),
            ("Base experiment config", str(manifest.get("base_experiment_config", "n/a"))),
            ("Sensitivity config", str(manifest.get("config_path", "n/a"))),
            ("Datasets", ", ".join(datasets) if datasets else "n/a"),
            ("Total runs", str(total_runs)),
            ("Successful runs", str(success_count)),
            ("Failed runs", str(failure_count)),
            ("Total reported training time (s)", f"{total_time_sec:.3f}"),
        ]
    )
    failure_html = _render_table(
        failure_rows,
        columns=["run_id", "sweep_name", "param_value", "error_type", "error_message"],
    )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Sensitivity Analysis Report</title>
  <style>
    :root {{
      --bg: #f4f6fb;
      --paper: #ffffff;
      --paper-soft: #f8fafc;
      --line: #d9e2ec;
      --line-strong: #c5d0dd;
      --text: #16324f;
      --muted: #5b6b7f;
      --accent: #1d4e89;
      --accent-soft: #eef3f8;
      --shadow: 0 18px 48px rgba(18, 50, 79, 0.08);
      --radius: 18px;
      --content-width: 1180px;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at top left, rgba(29, 78, 137, 0.06), transparent 28%),
        linear-gradient(180deg, #f7f9fc 0%, var(--bg) 100%);
    }}
    .page {{
      max-width: var(--content-width);
      margin: 0 auto;
      padding: 40px 24px 72px;
    }}
    .hero {{
      background: linear-gradient(180deg, #f9fbfe 0%, #f1f5fa 100%);
      border: 1px solid var(--line);
      border-radius: 24px;
      padding: 30px 32px;
      box-shadow: var(--shadow);
      margin-bottom: 28px;
    }}
    .eyebrow {{
      display: inline-block;
      padding: 6px 10px;
      border-radius: 999px;
      background: var(--accent-soft);
      color: var(--accent);
      font-size: 13px;
      font-weight: 700;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }}
    h1, h2, h3, h4 {{ margin: 0 0 0.6rem; }}
    h1 {{ font-size: 2.1rem; margin-top: 12px; }}
    h2 {{ font-size: 1.4rem; }}
    h3 {{ font-size: 1.15rem; }}
    p {{ margin: 0; line-height: 1.6; color: var(--muted); }}
    section {{
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      padding: 24px;
      margin-bottom: 24px;
    }}
    .summary-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 14px;
      margin-top: 18px;
    }}
    .summary-item, .fact-item {{
      background: var(--paper-soft);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 14px 16px;
    }}
    .summary-label, .fact-label {{
      display: block;
      font-size: 0.85rem;
      color: var(--muted);
      margin-bottom: 6px;
    }}
    .summary-value, .fact-value {{
      font-size: 1rem;
      color: var(--text);
    }}
    .plot-gallery {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 18px;
      margin-top: 18px;
    }}
    .plot-card {{
      margin: 0;
      background: var(--paper-soft);
      border: 1px solid var(--line);
      border-radius: 16px;
      overflow: hidden;
    }}
    .plot-card img {{
      display: block;
      width: 100%;
      height: auto;
      background: white;
    }}
    .plot-card figcaption {{
      padding: 12px 14px 14px;
      font-size: 0.9rem;
      color: var(--muted);
      border-top: 1px solid var(--line);
    }}
    .finding-list {{
      list-style: none;
      padding: 0;
      margin: 0;
      display: grid;
      gap: 12px;
    }}
    .finding-item {{
      padding: 14px 16px;
      border-left: 4px solid var(--accent);
      background: var(--paper-soft);
      border-radius: 12px;
      color: var(--text);
    }}
    .parameter-card {{
      padding: 24px;
      background: linear-gradient(180deg, #ffffff 0%, #fbfcfe 100%);
    }}
    .parameter-header {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: flex-start;
      margin-bottom: 12px;
    }}
    .parameter-path {{
      margin-top: 4px;
    }}
    .parameter-badge {{
      white-space: nowrap;
      display: inline-flex;
      align-items: center;
      padding: 8px 12px;
      border-radius: 999px;
      background: var(--accent-soft);
      color: var(--accent);
      font-size: 0.9rem;
      font-weight: 700;
    }}
    .card-split {{
      display: grid;
      grid-template-columns: 1.4fr 1fr;
      gap: 20px;
      margin-top: 22px;
    }}
    .facts-grid {{
      display: grid;
      gap: 12px;
    }}
    .table-wrap {{
      overflow-x: auto;
      margin-top: 12px;
      border: 1px solid var(--line);
      border-radius: 14px;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      background: white;
    }}
    th, td {{
      padding: 12px 14px;
      text-align: left;
      border-bottom: 1px solid var(--line);
      vertical-align: top;
    }}
    th {{
      background: #eef3f8;
      color: var(--text);
      font-weight: 700;
    }}
    tbody tr:nth-child(even) td {{
      background: #fbfcfe;
    }}
    td.numeric {{
      font-variant-numeric: tabular-nums;
    }}
    code {{
      background: #eef2f7;
      padding: 0.15rem 0.4rem;
      border-radius: 6px;
      color: var(--text);
    }}
    a {{
      color: var(--accent);
      text-decoration: none;
    }}
    a:hover {{
      text-decoration: underline;
    }}
    ul {{
      margin: 12px 0 0;
      padding-left: 20px;
      color: var(--muted);
    }}
    .empty-state {{
      color: var(--muted);
      font-style: italic;
      margin-top: 10px;
    }}
    .appendix-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 18px;
    }}
    @media (max-width: 900px) {{
      .card-split,
      .appendix-grid {{
        grid-template-columns: 1fr;
      }}
      .parameter-header {{
        flex-direction: column;
      }}
    }}
  </style>
</head>
<body>
  <main class="page">
    <header class="hero">
      <span class="eyebrow">Sensitivity Analysis</span>
      <h1>Sensitivity Analysis Report</h1>
      <p>Compact experiment summary with static plots, sweep-level findings, and reproducibility artifacts.</p>
    </header>
    <section>
      <h2>Study Overview</h2>
      <p>High-level metadata and the combined view of the evaluated sweep settings.</p>
      {overview_items}
      {_render_plot_gallery(study_dir, overview_plots)}
    </section>
    <section>
      <h2>Key Findings</h2>
      <p>Best-performing settings and cost observations derived from the aggregated sweep results.</p>
      {_render_key_findings(summary_by_value_rows, summary_by_parameter_rows, failure_rows, primary_metric)}
    </section>
    <section>
      <h2>Parameter Sections</h2>
      <p>Each parameter card combines plots, compact tables, and the best-setting summary for direct use in reports.</p>
      {parameter_sections_html}
    </section>
    <section>
      <h2>Failures and Warnings</h2>
      <p>Any failed sensitivity runs are listed here for diagnosis and reruns.</p>
      {failure_html}
    </section>
    <section>
      <h2>Reproducibility</h2>
      <div class="appendix-grid">
        <div>
          <p>Base experiment config: <code>{escape(str(manifest.get("base_experiment_config", "n/a")))}</code></p>
          <p style="margin-top: 10px;">Sensitivity config: <code>{escape(str(manifest.get("config_path", "n/a")))}</code></p>
          <p style="margin-top: 10px;">Rebuild command: <code>python scripts/build_sensitivity_report.py --study-dir {escape(str(study_dir))}</code></p>
          <p style="margin-top: 10px;">Run command: <code>python scripts/run_sensitivity_analysis.py --config {escape(str(manifest.get("config_path", "path/to/study.yaml")))}</code></p>
        </div>
        <div>
          <h3>Resolved Configs</h3>
          <ul>{reproducibility_links or '<li>No resolved configs found.</li>'}</ul>
          <h3 style="margin-top: 18px;">Tables</h3>
          <ul>{table_items}</ul>
        </div>
      </div>
    </section>
  </main>
</body>
</html>"""
    output = report_dir / "index.html"
    output.write_text(html, encoding="utf-8")
    return output
