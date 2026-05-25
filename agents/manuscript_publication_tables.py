"""Publication-style LaTeX tables (reference-corpus aligned booktabs/tabularx)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from agents.manuscript_submission_style import humanize_ablation_variant, short_method_name
from agents.paper_orchestra_pipeline import _latex_escape

_METHOD_ROW_ORDER = (
    "vanilla",
    "chain-of-thought",
    "always-reason",
    "cot",
    "cpg",
    "contrastive perceptual",
    "ours",
)

_METRIC_DISPLAY = {
    "primary_score": r"Primary score $\uparrow$",
    "accuracy": r"Accuracy $\uparrow$",
    "f1": r"F1 $\uparrow$",
}


_NUM_TOL = 5e-5


def _smart_decimals(value: float | None) -> int:
    """Choose decimals so all scores share a stable column width.

    Per-dataset / ablation values are usually 0..1 (3 dp) or percentages (1 dp).
    """
    if value is None:
        return 3
    abs_v = abs(value)
    if abs_v >= 10:
        return 2
    if abs_v >= 1:
        return 3
    return 3


def _best_in_column(values: list[float | None]) -> float | None:
    numeric = [v for v in values if v is not None]
    if not numeric:
        return None
    return max(numeric)


def _is_best(value: float | None, best: float | None) -> bool:
    if value is None or best is None:
        return False
    return abs(value - best) <= _NUM_TOL


def _metric_header(metric_name: str) -> str:
    key = str(metric_name or "primary_score").strip().lower()
    return _METRIC_DISPLAY.get(key, _latex_escape(metric_name.replace("_", r"\_")))


def _method_sort_key(name: str) -> tuple[int, str]:
    lower = str(name or "").lower()
    for idx, token in enumerate(_METHOD_ROW_ORDER):
        if token in lower:
            return (idx, lower)
    return (len(_METHOD_ROW_ORDER), lower)


def _format_score(value: float | None, *, bold: bool = False, decimals: int | None = None) -> str:
    if value is None:
        return r"\textendash"
    dec = decimals if decimals is not None else _smart_decimals(value)
    text = f"{value:.{dec}f}"
    return rf"\textbf{{{text}}}" if bold else text


def _parse_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _rows_from_benchmark(state: dict) -> list[tuple[str, float]]:
    summary = state.get("benchmark_summary") if isinstance(state.get("benchmark_summary"), dict) else {}
    if not summary:
        packet = state.get("result_packet") if isinstance(state.get("result_packet"), dict) else {}
        summary = packet.get("benchmark_summary") if isinstance(packet.get("benchmark_summary"), dict) else {}
    per_method = summary.get("per_method") if isinstance(summary, dict) else {}
    metric_name = str(
        summary.get("metric_name") or summary.get("primary_metric") or state.get("baseline_metric_name") or "primary_score"
    )
    rows: list[tuple[str, float]] = []
    if isinstance(per_method, dict):
        for name, row in sorted(per_method.items(), key=lambda x: str(x[0])):
            val = None
            if isinstance(row, dict):
                for key in (metric_name, "metric_value", "primary_score"):
                    if key in row:
                        val = row.get(key)
                        break
                if val is None and len(row) == 1:
                    val = next(iter(row.values()))
            else:
                val = row
            parsed = _parse_float(val)
            if parsed is not None:
                rows.append((str(name), parsed))
    if rows:
        return rows
    raw = state.get("main_results_table")
    if isinstance(raw, dict):
        for name, row in sorted(raw.items(), key=lambda x: str(x[0])):
            if isinstance(row, dict):
                for val in row.values():
                    parsed = _parse_float(val)
                    if parsed is not None:
                        rows.append((str(name), parsed))
                        break
            else:
                parsed = _parse_float(row)
                if parsed is not None:
                    rows.append((str(name), parsed))
    return rows


def _ablation_rows(state: dict) -> list[tuple[str, float | None, str]]:
    """Return (variant, score, delta_vs_full) rows for ablation table."""
    summary = state.get("benchmark_summary") if isinstance(state.get("benchmark_summary"), dict) else {}
    ablations = {}
    if isinstance(summary.get("ablations"), dict):
        ablations = summary["ablations"]
    elif isinstance(summary.get("ablation_results"), dict):
        ablations = summary["ablation_results"]
    metric = str(
        summary.get("metric_name") or summary.get("primary_metric") or state.get("baseline_metric_name") or "primary_score"
    )
    rows: list[tuple[str, float | None, str]] = []
    if ablations:
        full_score = _parse_float(state.get("best_metric_value") or state.get("baseline_metric_value"))
        for name, row in sorted(ablations.items(), key=lambda x: str(x[0])):
            if isinstance(row, dict) and row.get("executed") is False:
                rows.append((str(name), None, "n/a"))
                continue
            val = row.get(metric) if isinstance(row, dict) else row
            score = _parse_float(val)
            if isinstance(row, dict) and row.get("delta_vs_full") is not None:
                try:
                    delta = f"{float(row['delta_vs_full']):+.4f}"
                except (TypeError, ValueError):
                    delta = "—"
            elif score is not None and full_score is not None:
                delta = f"{score - full_score:+.4f}"
            else:
                delta = "—"
            rows.append((str(name), score, delta))
        return rows
    contract = state.get("publication_evidence_contract") if isinstance(state.get("publication_evidence_contract"), dict) else {}
    names = contract.get("required_ablations") or []
    for name in names:
        if not name:
            continue
        rows.append((str(name), None, "not run"))
    return rows


def build_publication_main_results_table_tex(state: dict) -> str:
    """Conference-style main results table (booktabs + bold best).

    When per-dataset and seed-variance data are also available, the panel is
    emitted as a *single* table* spanning both columns (main + per-dataset +
    seed std), eliminating the previous "3 short tables stacked vertically"
    look that the reviewers flagged as ugly.
    """
    rows = _rows_from_benchmark(state)
    if not rows:
        return ""
    metric_name = str(state.get("baseline_metric_name") or "primary_score")
    metric_header = _metric_header(metric_name)
    best_val = _best_in_column([v for _, v in rows])

    summary = state.get("benchmark_summary") if isinstance(state.get("benchmark_summary"), dict) else {}
    per_dataset = summary.get("per_dataset") if isinstance(summary.get("per_dataset"), dict) else {}
    seed_var = summary.get("seed_variance") if isinstance(summary.get("seed_variance"), dict) else {}
    if not per_dataset:
        per_dataset = state.get("per_dataset_results") if isinstance(state.get("per_dataset_results"), dict) else {}
    if not seed_var:
        seed_var = state.get("seed_variance_table") if isinstance(state.get("seed_variance_table"), dict) else {}

    if per_dataset and len(per_dataset) >= 2:
        return _build_combined_main_panel(
            rows,
            per_dataset=per_dataset,
            seed_var=seed_var,
            metric_header=metric_header,
            method_for_paper=str(state.get("method_name") or "Ours"),
        )

    lines = _table_open_lines(
        r"Main results on reasoning benchmarks. \textbf{Bold} marks the best score per column; $\uparrow$ indicates higher is better.",
        "tab:main_results",
        r"@{}l r@{}",
    )
    lines.append("Method & " + metric_header + r" \\")
    lines.append(r"\midrule")
    for method_name, value in sorted(rows, key=lambda r: _method_sort_key(r[0])):
        short = short_method_name(method_name)
        name_tex = _latex_escape(short)
        if "ours" in short.lower() or "cpg" in short.lower():
            name_tex = r"\textbf{" + name_tex + "}"
        val_tex = _format_score(value, bold=_is_best(value, best_val))
        lines.append(f"{name_tex} & {val_tex} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines).replace(r"\begin{tabularx}{\linewidth}", r"\begin{tabular}")


def _build_combined_main_panel(
    rows: list[tuple[str, float]],
    *,
    per_dataset: dict[str, dict[str, Any]],
    seed_var: dict[str, dict[str, Any]],
    metric_header: str,
    method_for_paper: str,
) -> str:
    """Single full-width table*: aggregate + per-dataset + seed std side by side."""

    datasets: list[str] = []
    for row in per_dataset.values():
        if isinstance(row, dict):
            for ds in row:
                if ds not in datasets:
                    datasets.append(str(ds))
    datasets = sorted(datasets)

    method_keys: list[str] = []
    for name, _ in sorted(rows, key=lambda r: _method_sort_key(r[0])):
        method_keys.append(name)
    for key in per_dataset:
        if key not in method_keys:
            method_keys.append(key)

    agg_score = {name: val for name, val in rows}
    best_agg = _best_in_column([val for _, val in rows])
    best_per_ds: dict[str, float | None] = {}
    for ds in datasets:
        vals = [
            _parse_float(per_dataset.get(m, {}).get(ds))
            for m in method_keys
            if isinstance(per_dataset.get(m), dict)
        ]
        best_per_ds[ds] = _best_in_column(vals)

    header_cells = ["Method", metric_header]
    header_cells.extend(_latex_escape(ds) for ds in datasets)
    if seed_var:
        header_cells.append(r"Seed std")

    colspec_parts = ["@{}l", "r"]
    colspec_parts.extend(["r"] * len(datasets))
    if seed_var:
        colspec_parts.append("r")
    colspec_parts.append("@{}")
    colspec = "".join(colspec_parts)

    n_cols = len(header_cells)
    groups: list[tuple[str, int]] = [("Aggregate", 1)]
    if datasets:
        groups.append(("Per-dataset", len(datasets)))
    if seed_var:
        groups.append(("Stability", 1))

    lines: list[str] = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.15}",
        rf"\caption{{Main results: aggregate {metric_header.split('$')[0].strip().lower()}, per-dataset breakdown, and seed stability. \textbf{{Bold}} marks the best per column.}}",
        r"\label{tab:main_results}",
        rf"\begin{{tabular}}{{{colspec}}}",
        r"\toprule",
    ]

    group_header_cells: list[str] = ["", ""]
    col_idx = 2
    for name, span in groups[1:]:
        if span == 1:
            group_header_cells.append(rf"\textsc{{{name}}}")
        else:
            group_header_cells.append(rf"\multicolumn{{{span}}}{{c}}{{\textsc{{{name}}}}}")
            for _ in range(span - 1):
                pass
        col_idx += span
    if any(cell for cell in group_header_cells if cell.strip()):
        gh = " & ".join(group_header_cells[:n_cols]) if False else (
            " & ".join([""] * 2 + ([rf"\multicolumn{{{len(datasets)}}}{{c}}{{\textsc{{Per-dataset}}}}"] if datasets else []) + ([r"\textsc{Stability}"] if seed_var else []))
        )
        cmid: list[str] = []
        start = 3
        if datasets:
            end = start + len(datasets) - 1
            cmid.append(rf"\cmidrule(lr){{{start}-{end}}}")
            start = end + 1
        if seed_var:
            cmid.append(rf"\cmidrule(lr){{{start}-{start}}}")
        lines.append(gh + r" \\")
        if cmid:
            lines.append(" ".join(cmid))

    lines.append(" & ".join(header_cells) + r" \\")
    lines.append(r"\midrule")

    for method_key in method_keys:
        short = short_method_name(method_key)
        name_tex = _latex_escape(short)
        if "ours" in short.lower() or method_for_paper.lower() in short.lower():
            name_tex = r"\textbf{" + name_tex + "}"
        cells: list[str] = [name_tex]
        agg = agg_score.get(method_key)
        cells.append(_format_score(agg, bold=_is_best(agg, best_agg)))
        for ds in datasets:
            row = per_dataset.get(method_key, {}) if isinstance(per_dataset.get(method_key), dict) else {}
            val = _parse_float(row.get(ds))
            cells.append(_format_score(val, bold=_is_best(val, best_per_ds.get(ds))))
        if seed_var:
            sv = seed_var.get(method_key, {}) if isinstance(seed_var.get(method_key), dict) else {}
            std_v = _parse_float(sv.get("std"))
            cells.append(_format_score(std_v))
        lines.append(" & ".join(cells) + r" \\")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    return "\n".join(lines)


def build_per_dataset_results_table_tex(state: dict) -> str:
    summary = state.get("benchmark_summary") if isinstance(state.get("benchmark_summary"), dict) else {}
    per_dataset = summary.get("per_dataset") if isinstance(summary.get("per_dataset"), dict) else {}
    if not per_dataset:
        per_dataset = state.get("per_dataset_results") if isinstance(state.get("per_dataset_results"), dict) else {}
    if not per_dataset:
        return ""
    datasets: list[str] = []
    for row in per_dataset.values():
        if isinstance(row, dict):
            for ds in row:
                if ds not in datasets:
                    datasets.append(str(ds))
    datasets = sorted(datasets)
    if not datasets:
        return ""
    best_by_ds: dict[str, float] = {}
    for ds in datasets:
        vals = [
            _parse_float(row.get(ds))
            for row in per_dataset.values()
            if isinstance(row, dict)
        ]
        vals = [v for v in vals if v is not None]
        if vals:
            best_by_ds[ds] = max(vals)
    col_spec = "@{}l" + "r" * len(datasets) + "@{}"
    lines = _table_open_lines(
        r"Per-dataset \texttt{primary\_score} ($\uparrow$). \textbf{Bold} is best per column.",
        "tab:per_dataset",
        col_spec,
    )
    lines.append("Method & " + " & ".join(_latex_escape(d) for d in datasets) + r" \\")
    lines.append(r"\midrule")
    for method_name, row in sorted(per_dataset.items(), key=lambda x: _method_sort_key(x[0])):
        if not isinstance(row, dict):
            continue
        short = short_method_name(method_name)
        name_tex = _latex_escape(short)
        if "ours" in short.lower() or "cpg" in short.lower():
            name_tex = r"\textbf{" + name_tex + "}"
        cells = []
        for ds in datasets:
            val = _parse_float(row.get(ds))
            best = best_by_ds.get(ds)
            bold = val is not None and best is not None and abs(val - best) < 1e-9
            cells.append(_format_score(val, bold=bold, decimals=3))
        lines.append(f"{name_tex} & " + " & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabularx}", r"\end{table}"])
    return "\n".join(lines)


def build_seed_variance_table_tex(state: dict) -> str:
    summary = state.get("benchmark_summary") if isinstance(state.get("benchmark_summary"), dict) else {}
    seed_var = summary.get("seed_variance") if isinstance(summary.get("seed_variance"), dict) else {}
    if not seed_var:
        seed_var = state.get("seed_variance_table") if isinstance(state.get("seed_variance_table"), dict) else {}
    if not seed_var:
        return ""
    metric_name = str(state.get("baseline_metric_name") or "primary_score")
    lines = _table_open_lines(
        r"Seed stability (mean $\pm$ std over seeds).",
        "tab:seed_variance",
        r"@{}Y r r r@{}",
    )
    lines.append(r"Method & Mean & Std & Seeds \\")
    lines.append(r"\midrule")
    for method_name, row in sorted(seed_var.items(), key=lambda x: str(x[0])):
        if not isinstance(row, dict):
            continue
        mean_v = _parse_float(row.get("mean"))
        std_v = _parse_float(row.get("std"))
        n_seeds = int(row.get("n_seeds") or 0)
        mean_tex = f"{mean_v:.4f}" if mean_v is not None else "---"
        std_tex = f"{std_v:.4f}" if std_v is not None else "---"
        lines.append(f"{_latex_escape(short_method_name(method_name))} & {mean_tex} & {std_tex} & {n_seeds} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabularx}", r"\end{table}"])
    return "\n".join(lines)


def build_publication_ablation_table_tex(state: dict) -> str:
    """Component ablation table (executed rows only, or status table if none ran)."""
    rows = _ablation_rows(state)
    if not rows:
        return ""
    executed = [r for r in rows if r[1] is not None and r[2] not in ("n/a", "not run")]
    pending = [r for r in rows if r[1] is None or r[2] in ("n/a", "not run")]
    metric_name = str(state.get("baseline_metric_name") or "primary_score")
    metric_header = _metric_header(metric_name)

    if not executed and pending:
        lines = _table_open_compact_lines(
            r"Planned component ablations (pending execution in this run).",
            "tab:ablations",
            r"@{}>{\raggedright\arraybackslash}p{0.78\linewidth}>{\centering\arraybackslash}p{1.35cm}@{}",
        )
        lines.append(r"Component variant & Status \\")
        lines.append(r"\midrule")
        for name, _, _status in sorted(pending, key=lambda r: _ablation_sort_key(r[0])):
            variant = humanize_ablation_variant(name)
            lines.append(f"{variant} & \\textsc{{Pending}} \\\\")
        lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
        return "\n".join(lines)

    lines = _table_open_lines(
        r"Ablation study isolating CPG components. $\Delta$ is change vs.\ full model.",
        "tab:ablations",
        r"@{}Y r c@{}",
    )
    lines.append("Variant & " + metric_header + r" & $\Delta$ \\")
    lines.append(r"\midrule")
    for name, score, delta in executed + pending:
        score_tex = f"{score:.4f}" if score is not None else "---"
        delta_tex = delta if isinstance(delta, str) else f"{delta:+.4f}"
        variant_tex = _table_cell(humanize_ablation_variant(name))
        lines.append(f"{variant_tex} & {score_tex} & {delta_tex} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabularx}", r"\end{table}"])
    return "\n".join(lines)


def _ablation_sort_key(name: str) -> tuple[int, str]:
    label = humanize_ablation_variant(name).lower()
    if "compute-matched" in label:
        return (0, label)
    if "full cpg" in label:
        return (9, label)
    return (5, label)


def _table_open_compact_lines(caption: str, label: str, colspec: str) -> list[str]:
    """Fixed-width tabular (no tabularx stretch) for variant + status tables."""
    return [
        r"\begin{table}[t]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.12}",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{colspec}}}",
        r"\toprule",
    ]


def _table_open_lines(caption: str, label: str, colspec: str, *, table_star: bool = False) -> list[str]:
    env = "table*" if table_star else "table"
    return [
        rf"\begin{{{env}}}[t]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{5pt}",
        r"\renewcommand{\arraystretch}{1.12}",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabularx}}{{\linewidth}}{{{colspec}}}",
        r"\toprule",
    ]


def ensure_publication_table_packages(main_tex: str) -> str:
    """Ensure tabularx/small tables compile under ICLR preamble."""
    if "\\begin{document}" not in main_tex:
        return main_tex
    preamble, marker, body = main_tex.partition(r"\begin{document}")
    for pkg in ("tabularx", "array"):
        if pkg not in preamble:
            preamble = preamble.rstrip() + "\n" + rf"\usepackage{{{pkg}}}" + "\n"
    if r"\newcolumntype{Y}" not in preamble:
        preamble = preamble.rstrip() + (
            "\n\\newcolumntype{Y}{>{\\raggedright\\arraybackslash}X}\n"
        )
    return preamble + marker + body


def _replace_labeled_table(tex: str, label: str, block: str) -> str:
    if not block:
        return tex
    pat = rf"\\begin\{{(?:table\*?|table)\}}.*?\\label\{{{re.escape(label)}\}}.*?\\end\{{(?:table\*?|table)\}}"
    m = re.search(pat, tex, flags=re.DOTALL)
    if m:
        return tex[: m.start()] + block + tex[m.end() :]
    return tex


def _table_cell(text: str) -> str:
    s = str(text or "")
    return s.replace("\\", r"\textbackslash{}").replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")


def _inject_table_after_label(tex: str, after_label: str, block: str, *, new_label: str | None = None) -> str:
    if not block:
        return tex
    if new_label and rf"\label{{{new_label}}}" in tex:
        return tex
    m = re.search(
        rf"(\\label\{{{re.escape(after_label)}\}}.*?\\end\{{(?:table\*?|table)\}})",
        tex,
        flags=re.DOTALL,
    )
    if m:
        return tex[: m.end()] + "\n\n" + block + tex[m.end() :]
    return tex + "\n\n" + block


def ensure_experiments_section(tex: str) -> str:
    if re.search(r"\\section\*?\{Experiments\}", tex, flags=re.IGNORECASE):
        return tex
    m = re.search(
        r"(\\subsection\*?\{Experimental Setup\}[^\n]*\n|We evaluate CPG on \d reasoning)",
        tex,
        flags=re.IGNORECASE,
    )
    if m:
        return tex[: m.start()] + r"\section{Experiments}" + "\n\n" + tex[m.start() :]
    m2 = re.search(r"(\\section\*?\{Discussion\})", tex, flags=re.IGNORECASE)
    if m2:
        return tex[: m2.start()] + r"\section{Experiments}" + "\n\n" + tex[m2.start() :]
    return tex


def replace_tables_in_tex(main_tex: str, state: dict) -> tuple[str, dict[str, Any]]:
    """Replace naive tables and inject ablation table in Experiments."""
    meta: dict[str, Any] = {
        "replaced_main_table": False,
        "injected_ablation_table": False,
        "injected_per_dataset_table": False,
        "injected_seed_variance_table": False,
        "combined_main_panel": False,
    }
    main_table = build_publication_main_results_table_tex(state)
    combined = bool(main_table and r"\begin{table*}" in main_table)
    meta["combined_main_panel"] = combined
    per_ds_table = "" if combined else build_per_dataset_results_table_tex(state)
    seed_table = "" if combined else build_seed_variance_table_tex(state)
    ablation_table = build_publication_ablation_table_tex(state)
    out = ensure_publication_table_packages(ensure_experiments_section(main_tex))
    if main_table:
        old = re.search(
            r"\\begin\{table\*?\}.*?\\label\{tab:main_results\}.*?\\end\{table\*?\}",
            out,
            flags=re.DOTALL,
        )
        if old:
            out = out[: old.start()] + main_table + out[old.end() :]
            meta["replaced_main_table"] = True
        elif r"\label{tab:main_results}" not in out:
            out = _inject_after_experiments(out, main_table)
            meta["replaced_main_table"] = True
    if combined:
        # Remove any pre-existing per-dataset / seed-variance tables so we do
        # not double-print them next to the combined main panel. We must find
        # the OPENING that BELONGS to this label specifically: scan backward
        # from the label past any prior closed table/table* blocks, then take
        # the next `\\begin{table*}`/`\\begin{table}` that opens the unclosed
        # block.
        for stale_label in ("tab:per_dataset", "tab:seed_variance"):
            label_token = rf"\label{{{stale_label}}}"
            idx = out.find(label_token)
            if idx < 0:
                continue
            prev_end_star = out.rfind(r"\end{table*}", 0, idx)
            prev_end_plain = out.rfind(r"\end{table}", 0, idx)
            prev_end = max(prev_end_star, prev_end_plain)
            scan_from = 0 if prev_end < 0 else prev_end
            next_begin_star = out.find(r"\begin{table*}", scan_from, idx)
            next_begin_plain = out.find(r"\begin{table}", scan_from, idx)
            candidates = [b for b in (next_begin_star, next_begin_plain) if b >= 0]
            if not candidates:
                continue
            begin_idx = min(candidates)
            search_from = idx + len(label_token)
            end_star = out.find(r"\end{table*}", search_from)
            end_plain = out.find(r"\end{table}", search_from)
            ends = [e for e in (end_star, end_plain) if e >= 0]
            if not ends:
                continue
            end_idx = min(ends)
            close_len = len(r"\end{table*}") if end_idx == end_star else len(r"\end{table}")
            out = out[:begin_idx] + out[end_idx + close_len :]
    if per_ds_table:
        if rf"\label{{tab:per_dataset}}" in out:
            out = _replace_labeled_table(out, "tab:per_dataset", per_ds_table)
        elif r"\label{tab:main_results}" in out:
            out = _inject_table_after_label(out, "tab:main_results", per_ds_table, new_label="tab:per_dataset")
        else:
            out = _inject_after_experiments(out, per_ds_table)
        meta["injected_per_dataset_table"] = True
    if seed_table:
        if rf"\label{{tab:seed_variance}}" in out:
            out = _replace_labeled_table(out, "tab:seed_variance", seed_table)
        else:
            out = _inject_table_after_label(out, "tab:per_dataset", seed_table, new_label="tab:seed_variance")
        meta["injected_seed_variance_table"] = True
    if ablation_table:
        if r"\label{tab:ablations}" in out:
            out = _replace_labeled_table(out, "tab:ablations", ablation_table)
            meta["injected_ablation_table"] = True
        else:
            ab_m = re.search(r"\\subsection\*?\{Ablation[^}]*\}", out, flags=re.IGNORECASE)
            if ab_m:
                out = out[: ab_m.end()] + "\n" + ablation_table + "\n" + out[ab_m.end() :]
                meta["injected_ablation_table"] = True
            else:
                block = r"\subsection{Ablation Study}" + "\n" + ablation_table + "\n"
                m = re.search(r"(\\section\*?\{Experiments\}[^\n]*\n)", out, flags=re.IGNORECASE)
                if m:
                    out = out[: m.end()] + block + out[m.end() :]
                    meta["injected_ablation_table"] = True
    return out, meta


def _inject_after_experiments(tex: str, block: str) -> str:
    for pat in (
        r"(\\section\*?\{Experiments\}[^\n]*\n)",
        r"(\\subsection\*?\{Experimental Setup\}[^\n]*\n)",
        r"(\\subsection\*?\{Main Results\}[^\n]*\n)",
    ):
        m = re.search(pat, tex, flags=re.IGNORECASE)
        if m:
            return tex[: m.end()] + block + "\n\n" + tex[m.end() :]
    return tex + "\n\n" + block


def attach_benchmark_artifacts_to_state(state: dict, *, run_id: int | None = None) -> dict:
    """Load/materialize benchmark_summary + all paper tables into manuscript state."""
    from agents.benchmark_artifacts import materialize_deep_benchmark_artifacts

    out = dict(state)
    results_dir: Path | None = None
    if run_id is not None and out.get("deep_insight_id"):
        results_dir = Path(
            f"/root/deepgraph_ideas/idea_{out.get('deep_insight_id')}/experiments/main/runs/run_{run_id}/results"
        )
    workdir = out.get("experiment_workdir") or out.get("workdir")
    if results_dir is None and workdir:
        results_dir = Path(str(workdir)) / "results"
    if results_dir and (results_dir / "raw_predictions.jsonl").is_file():
        contract = out.get("publication_evidence_contract") if isinstance(out.get("publication_evidence_contract"), dict) else {}
        materialize_deep_benchmark_artifacts(
            results_dir,
            publication_contract=contract,
            metric_name=str(out.get("baseline_metric_name") or "primary_score"),
        )
    if not results_dir or not results_dir.is_dir():
        return out

    def _load_json(name: str, key: str) -> None:
        path = results_dir / name
        if not path.is_file():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                out[key] = payload
        except (OSError, json.JSONDecodeError):
            pass

    _load_json("benchmark_summary.json", "benchmark_summary")
    _load_json("main_results_table.json", "main_results_table")
    _load_json("per_dataset_results.json", "per_dataset_results")
    _load_json("seed_variance_table.json", "seed_variance_table")
    _load_json("ablation_table.json", "ablation_table")
    summary = out.get("benchmark_summary") if isinstance(out.get("benchmark_summary"), dict) else {}
    if summary:
        out["benchmark_summary"] = summary
        if out.get("ablation_table"):
            summary = dict(summary)
            summary["ablations"] = out["ablation_table"]
            out["benchmark_summary"] = summary
    return out


def attach_ablation_artifacts_to_state(state: dict, *, run_id: int | None = None) -> dict:
    return attach_benchmark_artifacts_to_state(state, run_id=run_id)
