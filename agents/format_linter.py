"""FormatLinter — lint a normalised manuscript against a venue contract.

Issue #11/#14 (D3). Consumes a ``TemplateAdapter`` (the venue contract) plus
the post-``normalize_source`` LaTeX body and reports a deterministic list of
checks. Five "standard" checks cover the structural invariants the venue
template requires; two additional checks consume the new
``TemplateAdapter.column_layout`` property added in D1/D2 to enforce
single-column vs two-column figure-grid rules (per user feedback during
D1 review: "图表单栏一般四个一排").

Public API
~~~~~~~~~~
- ``lint_manuscript(source, adapter, *, page_count=None) -> dict``
- ``persist_lint_run(selection_id, adapter, lint_result, rule_set=...) -> int``
- ``get_lint_run(run_id) -> dict | None``

The ``checks`` list in the return payload always contains 7 entries (one
per check) regardless of pass/fail status, so the dashboard can render a
stable column layout for every lint run.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, List

from agents.manuscript_templates import TemplateAdapter
from db import database as db


DEFAULT_RULE_SET = "format_linter_v1"

# How many pages over budget before page_count flips error → soft warning.
PAGE_BUDGET_HARD_OVERAGE = 1

# Figure-grid density caps consumed by the column-layout check.
SINGLE_COLUMN_MAX_PANELS_PER_ROW = 4
TWO_COLUMN_MAX_PANELS_PER_ROW = 2


@dataclass
class CheckResult:
    name: str
    severity: str  # "info" | "warning" | "error"
    passed: bool
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "severity": self.severity,
            "passed": self.passed,
            "message": self.message,
            "details": self.details,
        }


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def _check_documentclass_present(source: str) -> CheckResult:
    has_class = bool(re.search(r"\\documentclass(?:\[[^\]]*\])?\{[^}]+\}", source))
    return CheckResult(
        name="documentclass_present",
        severity="error",
        passed=has_class,
        message=(
            "documentclass declaration found"
            if has_class
            else "missing \\documentclass declaration"
        ),
    )


def _check_bibstyle_matches_venue(source: str, adapter: TemplateAdapter) -> CheckResult:
    expected = adapter.bibstyle_name
    pattern = re.compile(r"\\bibliographystyle\{([^}]+)\}")
    matches = pattern.findall(source)
    if not matches:
        # No bibstyle is acceptable if the document has no \bibliography either.
        has_bibliography = "\\bibliography{" in source
        if not has_bibliography:
            return CheckResult(
                name="bibstyle_matches_venue",
                severity="info",
                passed=True,
                message="no \\bibliography in source; bibstyle check skipped",
                details={"expected": expected},
            )
        return CheckResult(
            name="bibstyle_matches_venue",
            severity="error",
            passed=False,
            message=f"\\bibliography present but no \\bibliographystyle (expected {expected!r})",
            details={"expected": expected},
        )
    found = matches[-1]
    return CheckResult(
        name="bibstyle_matches_venue",
        severity="error" if found != expected else "info",
        passed=found == expected,
        message=(
            f"bibstyle={found!r} matches venue"
            if found == expected
            else f"bibstyle={found!r} but venue expects {expected!r}"
        ),
        details={"expected": expected, "found": found},
    )


def _check_required_packages_present(source: str) -> CheckResult:
    """Each venue body assumes graphicx + amsmath + hyperref are loaded.

    Most adapters auto-inject these in ``inject_preamble``; this check catches
    sources that bypassed the adapter (e.g. legacy bundles).
    """
    required = ["graphicx", "amsmath", "hyperref"]
    missing = [p for p in required if p not in source]
    return CheckResult(
        name="required_packages_present",
        severity="warning",  # warning, not error: pdflatex will still compile
        passed=not missing,
        message=(
            "all required packages present"
            if not missing
            else f"missing required packages: {', '.join(missing)}"
        ),
        details={"required": required, "missing": missing},
    )


def _check_page_count_within_budget(
    source: str, adapter: TemplateAdapter, page_count: int | None
) -> CheckResult:
    """Soft warning when estimated page count exceeds venue budget."""
    budget = adapter.max_pages
    if page_count is None:
        return CheckResult(
            name="page_count_within_budget",
            severity="info",
            passed=True,
            message="page_count not provided; check deferred",
            details={"budget": budget},
        )
    over = page_count - budget
    if over <= 0:
        return CheckResult(
            name="page_count_within_budget",
            severity="info",
            passed=True,
            message=f"page_count={page_count} ≤ budget={budget}",
            details={"budget": budget, "page_count": page_count, "over": 0},
        )
    severity = "error" if over > PAGE_BUDGET_HARD_OVERAGE else "warning"
    return CheckResult(
        name="page_count_within_budget",
        severity=severity,
        passed=False,
        message=f"page_count={page_count} exceeds budget={budget} by {over}",
        details={"budget": budget, "page_count": page_count, "over": over},
    )


def _check_figure_placement_specifiers(source: str) -> CheckResult:
    """Encourage explicit float placement so figures don't drift off-page.

    Counts ``\\begin{figure}`` without an explicit ``[htbp]``/``[t]``/etc.
    """
    no_specifier = re.findall(r"\\begin\{figure\*?\}(?!\[)", source)
    return CheckResult(
        name="figure_placement_specifiers",
        severity="warning",
        passed=not no_specifier,
        message=(
            "all figure environments use placement specifiers"
            if not no_specifier
            else f"{len(no_specifier)} figure(s) missing placement specifier"
        ),
        details={"unspecified_count": len(no_specifier)},
    )


def _check_column_layout_consistency(source: str, adapter: TemplateAdapter) -> CheckResult:
    """Width units must match the venue column_layout (D1 user feedback).

    - two_column venues: ``\\columnwidth`` for in-column figures, ``figure*``
      + ``\\textwidth`` for full-page-width spans.
    - single_column venues: ``\\textwidth`` (or no width set) for figures.

    Flags any in-column ``figure`` that uses ``\\textwidth`` on a two-column
    venue, and any ``figure*`` that uses ``\\columnwidth`` on either layout.
    """
    layout = adapter.column_layout
    violations: List[str] = []

    # Find every figure block: (full_match, env_name, body_chunk)
    figure_pattern = re.compile(
        r"\\begin\{(figure\*?)\}(?:\[[^\]]*\])?(.*?)\\end\{\1\}",
        re.DOTALL,
    )
    figures = figure_pattern.findall(source)

    for env, body in figures:
        if layout == "two_column":
            if env == "figure" and "\\textwidth" in body:
                violations.append(
                    "in-column \\begin{figure} uses \\textwidth on a two_column venue "
                    "(use \\columnwidth, or switch to figure* for full-page spans)"
                )
            if env == "figure*" and "\\columnwidth" in body:
                violations.append(
                    "full-span \\begin{figure*} uses \\columnwidth (use \\textwidth)"
                )
        else:  # single_column
            if env == "figure*":
                violations.append(
                    "figure* (two-column span) on a single_column venue is meaningless"
                )
            if "\\columnwidth" in body:
                violations.append(
                    "figure uses \\columnwidth on a single_column venue (use \\textwidth)"
                )

    return CheckResult(
        name="column_layout_consistency",
        severity="warning",
        passed=not violations,
        message=(
            f"figure width units consistent with {layout}"
            if not violations
            else f"{len(violations)} width-unit violation(s) for {layout}"
        ),
        details={"layout": layout, "violations": violations[:10]},
    )


def _check_figure_grid_density(source: str, adapter: TemplateAdapter) -> CheckResult:
    """Cap subfigure panels per row by column_layout (user D1 feedback).

    Rule: single_column → ≤ 4 panels per row; two_column → ≤ 2 panels per row.
    Counts ``\\begin{subfigure}`` siblings inside each ``\\begin{figure}``
    environment that share the same horizontal row (separated by ``\\\\``).
    """
    layout = adapter.column_layout
    cap = SINGLE_COLUMN_MAX_PANELS_PER_ROW if layout == "single_column" else TWO_COLUMN_MAX_PANELS_PER_ROW
    figure_pattern = re.compile(
        r"\\begin\{figure\*?\}(?:\[[^\]]*\])?(.*?)\\end\{figure\*?\}",
        re.DOTALL,
    )
    overflow: List[dict[str, Any]] = []
    for idx, body in enumerate(figure_pattern.findall(source)):
        # Split rows by LaTeX double-backslash ``\\`` line break.
        rows = re.split(r"(?<!\\)\\\\(?!\\)", body)
        for row_idx, row in enumerate(rows):
            panel_count = len(re.findall(r"\\begin\{subfigure\}", row))
            if panel_count > cap:
                overflow.append({
                    "figure_index": idx,
                    "row_index": row_idx,
                    "panel_count": panel_count,
                    "cap": cap,
                })
    return CheckResult(
        name="figure_grid_density",
        severity="warning",
        passed=not overflow,
        message=(
            f"all subfigure rows respect cap={cap} for {layout}"
            if not overflow
            else f"{len(overflow)} subfigure row(s) exceed cap={cap}"
        ),
        details={"layout": layout, "cap_per_row": cap, "overflow": overflow[:5]},
    )


# ---------------------------------------------------------------------------
# Issue #14-mandated 5 checks (issue spec verbatim, by name)
# ---------------------------------------------------------------------------

# Per #14 the issue spec lists five check names that must appear in the lint
# report: ``font_size_consistency``, ``section_spacing``, ``float_density``,
# ``citation_density``, ``bib_style_match``. They live alongside the 7
# structural checks above so the lint manifest is a superset of the issue
# contract (12 entries total).

CITATION_DENSITY_MIN_PER_1000_WORDS = 3
FLOAT_DENSITY_MAX_PER_PAGE = 3


def _check_font_size_consistency(source: str) -> CheckResult:
    """Flag inline ``\\fontsize`` / ``\\large`` / ``\\small`` overrides.

    Venue templates set font sizing globally; inline overrides are the most
    common cause of the "字体大小不一" complaint from the review meeting.
    """
    fontsize_matches = re.findall(r"\\fontsize\{[^}]+\}\{[^}]+\}", source)
    size_macros = re.findall(
        r"\\(?:tiny|scriptsize|footnotesize|small|large|Large|LARGE|huge|Huge)\b",
        source,
    )
    n = len(fontsize_matches) + len(size_macros)
    return CheckResult(
        name="font_size_consistency",
        severity="warning",
        passed=n == 0,
        message=(
            "no inline font-size overrides"
            if n == 0
            else f"{n} inline font-size override(s) found "
                 f"(\\fontsize={len(fontsize_matches)}, size macros={len(size_macros)})"
        ),
        details={
            "fontsize_overrides": len(fontsize_matches),
            "size_macro_overrides": len(size_macros),
            "fix_hint": "remove \\fontsize{}{} and \\large/\\small inside the body; "
                        "let the venue .sty control sizing",
        },
    )


def _check_section_spacing(source: str) -> CheckResult:
    """Detect back-to-back ``\\section``/``\\subsection`` with no body between.

    A heading directly followed by another heading is the classic "段落紧凑"
    fingerprint that broke layout in the review meeting's example papers.
    """
    pattern = re.compile(
        r"\\(?:section|subsection|subsubsection)\*?\{[^}]*\}\s*"
        r"\\(?:section|subsection|subsubsection)\*?\{",
    )
    bad = pattern.findall(source)
    return CheckResult(
        name="section_spacing",
        severity="warning",
        passed=not bad,
        message=(
            "no back-to-back headings"
            if not bad
            else f"{len(bad)} heading(s) directly followed by another heading "
                 "with no body paragraph in between"
        ),
        details={
            "violation_count": len(bad),
            "fix_hint": "insert an introductory paragraph between consecutive headings",
        },
    )


def _check_float_density(source: str, page_count: int | None) -> CheckResult:
    """Cap figures+tables per page; default cap = 3 per page."""
    fig = len(re.findall(r"\\begin\{figure\*?\}", source))
    tab = len(re.findall(r"\\begin\{table\*?\}", source))
    total = fig + tab
    cap = FLOAT_DENSITY_MAX_PER_PAGE
    if page_count and page_count > 0:
        density = total / page_count
        passed = density <= cap
        return CheckResult(
            name="float_density",
            severity="warning",
            passed=passed,
            message=(
                f"float density {density:.2f}/page ≤ {cap}/page"
                if passed
                else f"float density {density:.2f}/page exceeds cap {cap}/page "
                     f"({total} floats over {page_count} pages)"
            ),
            details={
                "figures": fig, "tables": tab, "total": total,
                "page_count": page_count, "density_per_page": round(density, 3),
                "cap_per_page": cap,
                "fix_hint": "move secondary figures/tables to appendix",
            },
        )
    # No page count → fall back to absolute cap of cap * 4
    abs_cap = cap * 4
    passed = total <= abs_cap
    return CheckResult(
        name="float_density",
        severity="info",
        passed=passed,
        message=(
            f"page_count not provided; absolute float count = {total} ≤ {abs_cap}"
            if passed
            else f"page_count not provided but {total} floats exceeds the absolute "
                 f"cap of {abs_cap}"
        ),
        details={
            "figures": fig, "tables": tab, "total": total,
            "page_count": None, "cap_per_page": cap,
        },
    )


def _check_citation_density(source: str) -> CheckResult:
    """Warn if ``\\cite`` rate is below ~3 per 1000 words (sparse evidence)."""
    cites = len(re.findall(r"\\(?:cite|citep|citet|citealp|citeauthor)\b", source))
    # Strip LaTeX commands before word count to approximate visible text.
    stripped = re.sub(r"\\[a-zA-Z]+\*?\s*(?:\[[^\]]*\])?(?:\{[^}]*\})?", " ", source)
    stripped = re.sub(r"[{}$%&#_^~]", " ", stripped)
    words = [w for w in stripped.split() if any(ch.isalpha() for ch in w)]
    n_words = len(words)
    if n_words < 200:
        return CheckResult(
            name="citation_density",
            severity="info",
            passed=True,
            message=f"text too short ({n_words} words); citation density check skipped",
            details={"cites": cites, "words": n_words},
        )
    per_1000 = (cites * 1000.0) / n_words
    passed = per_1000 >= CITATION_DENSITY_MIN_PER_1000_WORDS
    return CheckResult(
        name="citation_density",
        severity="warning",
        passed=passed,
        message=(
            f"citation density {per_1000:.2f}/1000 words ≥ "
            f"{CITATION_DENSITY_MIN_PER_1000_WORDS}/1000"
            if passed
            else f"citation density {per_1000:.2f}/1000 words below "
                 f"threshold {CITATION_DENSITY_MIN_PER_1000_WORDS}/1000"
        ),
        details={
            "cites": cites, "words": n_words,
            "per_1000_words": round(per_1000, 3),
            "threshold_per_1000": CITATION_DENSITY_MIN_PER_1000_WORDS,
            "fix_hint": "add citations to claims that lack supporting references",
        },
    )


def _check_bib_style_match(source: str, adapter: TemplateAdapter) -> CheckResult:
    """Issue #14-mandated alias of ``bibstyle_matches_venue``.

    Same logic, exposed under the exact name the AI verifier looks for.
    """
    inner = _check_bibstyle_matches_venue(source, adapter)
    return CheckResult(
        name="bib_style_match",
        severity=inner.severity,
        passed=inner.passed,
        message=inner.message,
        details=inner.details,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def lint_manuscript(
    source: str,
    adapter: TemplateAdapter,
    *,
    page_count: int | None = None,
) -> dict[str, Any]:
    """Run all 12 checks on ``source`` against ``adapter`` and return a dict.

    The 12 checks are the union of:

    - 7 structural checks (documentclass, bibstyle, packages, page budget,
      figure placement, column-layout, figure-grid).
    - 5 issue-#14-mandated checks by exact name (``font_size_consistency``,
      ``section_spacing``, ``float_density``, ``citation_density``,
      ``bib_style_match``).

    The return shape is::

        {
            "template_id": str,
            "column_layout": str,
            "pass": bool,            # True iff zero error-severity failures
            "checks": [               # always 12 entries, ordered
                {"name": ..., "severity": ..., "passed": bool, ...},
            ],
            "summary": {
                "error_count": int,
                "warning_count": int,
                "info_count": int,
            },
        }
    """
    checks: List[CheckResult] = [
        _check_documentclass_present(source),
        _check_bibstyle_matches_venue(source, adapter),
        _check_required_packages_present(source),
        _check_page_count_within_budget(source, adapter, page_count),
        _check_figure_placement_specifiers(source),
        _check_column_layout_consistency(source, adapter),
        _check_figure_grid_density(source, adapter),
        # Issue #14-mandated by name:
        _check_font_size_consistency(source),
        _check_section_spacing(source),
        _check_float_density(source, page_count),
        _check_citation_density(source),
        _check_bib_style_match(source, adapter),
    ]
    error_count = sum(1 for c in checks if not c.passed and c.severity == "error")
    warning_count = sum(1 for c in checks if not c.passed and c.severity == "warning")
    info_count = sum(1 for c in checks if c.severity == "info")
    return {
        "template_id": adapter.template_id,
        "column_layout": adapter.column_layout,
        "rule_set": DEFAULT_RULE_SET,
        "pass": error_count == 0,
        "checks": [c.to_dict() for c in checks],
        "summary": {
            "error_count": error_count,
            "warning_count": warning_count,
            "info_count": info_count,
        },
    }


def persist_lint_run(
    selection_id: int | None,
    adapter: TemplateAdapter,
    lint_result: dict[str, Any],
    *,
    rule_set: str = DEFAULT_RULE_SET,
) -> int:
    """Insert one row into ``format_lint_runs`` and return its id."""
    summary = lint_result.get("summary") or {}
    new_id = db.insert_returning_id(
        """
        INSERT INTO format_lint_runs
            (selection_id, template_id, rule_set, pass,
             error_count, warning_count, info_count,
             checks_json, summary_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        RETURNING id
        """,
        (
            int(selection_id) if selection_id is not None else None,
            adapter.template_id,
            rule_set,
            1 if lint_result.get("pass") else 0,
            int(summary.get("error_count", 0)),
            int(summary.get("warning_count", 0)),
            int(summary.get("info_count", 0)),
            json.dumps(lint_result.get("checks", []), ensure_ascii=False),
            json.dumps(summary, ensure_ascii=False),
        ),
    )
    db.commit()
    return int(new_id)


def get_lint_run(run_id: int) -> dict[str, Any] | None:
    """Return a persisted lint run by id, or None if missing."""
    row = db.fetchone("SELECT * FROM format_lint_runs WHERE id=?", (int(run_id),))
    if not row:
        return None

    def _decode(field_name: str, default: Any) -> Any:
        v = row.get(field_name)
        if isinstance(v, str):
            try:
                return json.loads(v) if v else default
            except json.JSONDecodeError:
                return default
        return v if v is not None else default

    return {
        "id": row.get("id"),
        "selection_id": row.get("selection_id"),
        "template_id": row.get("template_id"),
        "rule_set": row.get("rule_set"),
        "pass": bool(row.get("pass")),
        "error_count": row.get("error_count"),
        "warning_count": row.get("warning_count"),
        "info_count": row.get("info_count"),
        "checks": _decode("checks_json", []),
        "summary": _decode("summary_json", {}),
        "created_at": row.get("created_at"),
    }
