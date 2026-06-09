"""TeX Code Agent for manuscript structural and compile repairs.

This agent owns mechanical LaTeX/code issues only: duplicate document
structures, missing optional packages, and compile-log driven fallbacks. It does
not repair weak science, missing evidence, or unsupported claims.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


TEX_CODE_AGENT_VERSION = "tex_code_agent_v1"


SECTION_RE = re.compile(r"\\section\*?\{([^}]+)\}")
ABSTRACT_RE = re.compile(r"\\begin\{abstract\}([\s\S]*?)\\end\{abstract\}", re.IGNORECASE)
PACKAGE_RE_TEMPLATE = r"\\usepackage(?:\[[^\]]*\])?\{%s\}\s*"


_ALGORITHM_FALLBACK = r"""
% TeX Code Agent fallback for environments normally provided by algorithm/algpseudocode.
\usepackage{float}
\newenvironment{algorithm}[1][]{\begin{figure}[t]\centering}{\end{figure}}
\newenvironment{algorithmic}[1][]{\begin{list}{}{\leftmargin=1.5em\itemsep=0.12em\parsep=0pt}}{\end{list}}
\newcommand{\State}{\item }
\newcommand{\Require}{\item[\textbf{Input:}] }
\newcommand{\Ensure}{\item[\textbf{Output:}] }
\newcommand{\Return}{\item \textbf{return} }
\newcommand{\For}[1]{\item \textbf{for} #1 \textbf{do}}
\newcommand{\EndFor}{\item \textbf{end for}}
\newcommand{\If}[1]{\item \textbf{if} #1 \textbf{then}}
\newcommand{\Else}{\item \textbf{else}}
\newcommand{\EndIf}{\item \textbf{end if}}
\newcommand{\While}[1]{\item \textbf{while} #1 \textbf{do}}
\newcommand{\EndWhile}{\item \textbf{end while}}
""".strip()


_MISC_FALLBACKS = {
    "multirow": r"\newcommand{\multirow}[3]{#3}",
    "cleveref": r"\newcommand{\cref}[1]{\ref{#1}}\newcommand{\Cref}[1]{\ref{#1}}",
}


def _insert_before_document(source: str, block: str) -> str:
    if not block.strip() or block.strip() in source:
        return source
    marker = r"\begin{document}"
    if marker not in source:
        return source + "\n" + block + "\n"
    preamble, sep, body = source.partition(marker)
    return preamble.rstrip() + "\n" + block.strip() + "\n" + sep + body


def _remove_package(source: str, package: str) -> tuple[str, bool]:
    pattern = re.compile(PACKAGE_RE_TEMPLATE % re.escape(package))
    new_source, count = pattern.subn("", source)
    return new_source, bool(count)


def _deduplicate_abstracts(source: str, actions: list[dict[str, Any]]) -> str:
    matches = list(ABSTRACT_RE.finditer(source))
    if len(matches) <= 1:
        return source
    keep = matches[0]
    pieces = [source[: keep.end()]]
    cursor = keep.end()
    for match in matches[1:]:
        pieces.append(source[cursor: match.start()])
        cursor = match.end()
    pieces.append(source[cursor:])
    actions.append(
        {
            "kind": "deduplicate_abstract",
            "removed": len(matches) - 1,
            "reason": "Only one abstract environment is allowed.",
        }
    )
    return "".join(pieces)


def _convert_abstract_sections(source: str, actions: list[dict[str, Any]]) -> str:
    if r"\section{Abstract}" not in source and r"\section*{Abstract}" not in source:
        return source
    if ABSTRACT_RE.search(source):
        new_source, count = re.subn(r"\\section\*?\{Abstract\}", r"\\paragraph{Abstract material.}", source)
        if count:
            actions.append({"kind": "abstract_section_demoted", "count": count})
        return new_source
    match = re.search(r"\\section\*?\{Abstract\}([\s\S]*?)(?=\\section\*?\{[^}]+\}|\\begin\{document\}|\\end\{document\})", source)
    if not match:
        return source
    body = match.group(1).strip()
    replacement = "\\begin{abstract}\n" + body + "\n\\end{abstract}\n"
    actions.append({"kind": "abstract_section_converted", "count": 1})
    return source[: match.start()] + replacement + source[match.end():]


def _deduplicate_top_sections(source: str, actions: list[dict[str, Any]]) -> str:
    seen: dict[str, int] = {}

    def replace(match: re.Match[str]) -> str:
        title = re.sub(r"\s+", " ", match.group(1).strip())
        key = title.lower()
        seen[key] = seen.get(key, 0) + 1
        if seen[key] == 1:
            return match.group(0)
        actions.append({"kind": "duplicate_section_demoted", "section": title, "occurrence": seen[key]})
        return r"\subsection{" + f"Additional {title} Details" + "}"

    return SECTION_RE.sub(replace, source)


def _remove_forced_question_spine(source: str, actions: list[dict[str, Any]]) -> str:
    pattern = re.compile(r"\\paragraph\{(Question|Motivation|Answer|Result)\.\}\s*", re.IGNORECASE)
    new_source, count = pattern.subn("", source)
    if count:
        actions.append({"kind": "forced_spine_labels_removed", "count": count})
    return new_source


def _repair_missing_packages(source: str, compile_log: str, actions: list[dict[str, Any]]) -> str:
    normalized_log = (compile_log or "").replace(chr(39), "`").replace("’", "`")
    missing = set(re.findall(r"File\s+`?([^`\s]+)\.sty`?\s+not\s+found", normalized_log, re.IGNORECASE))
    uses_algorithm = any(token in source for token in (r"\begin{algorithm}", r"\begin{algorithmic}", r"\State", r"\Require"))
    if uses_algorithm or {"algorithm", "algpseudocode", "algorithmic"}.intersection(missing):
        changed = False
        for package in ("algorithm", "algpseudocode", "algorithmic"):
            source, removed = _remove_package(source, package)
            changed = changed or removed
        if "TeX Code Agent fallback for environments normally provided by algorithm" not in source:
            source = _insert_before_document(source, _ALGORITHM_FALLBACK)
            changed = True
        if changed:
            actions.append({"kind": "algorithm_package_fallback", "missing": sorted(missing)})
    for package, fallback in _MISC_FALLBACKS.items():
        if package not in missing:
            continue
        source, removed = _remove_package(source, package)
        if fallback not in source:
            source = _insert_before_document(source, fallback)
        actions.append({"kind": "package_fallback", "package": package, "removed_package": removed})
    return source


def repair_latex_source(source: str, *, compile_log: str = "") -> tuple[str, dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    repaired = source or ""
    repaired = _convert_abstract_sections(repaired, actions)
    repaired = _deduplicate_abstracts(repaired, actions)
    repaired = _deduplicate_top_sections(repaired, actions)
    repaired = _remove_forced_question_spine(repaired, actions)
    repaired = _repair_missing_packages(repaired, compile_log, actions)
    changed = repaired != (source or "")
    return repaired, {
        "schema_version": TEX_CODE_AGENT_VERSION,
        "changed": changed,
        "actions": actions,
        "action_count": len(actions),
    }


def repair_latex_bundle(bundle_dir: Path, *, stage: str, compile_result: dict | None = None) -> dict[str, Any]:
    bundle_dir = Path(bundle_dir)
    main_tex = bundle_dir / "main.tex"
    if not main_tex.exists():
        return {
            "schema_version": TEX_CODE_AGENT_VERSION,
            "stage": stage,
            "changed": False,
            "error": "main.tex missing",
        }
    source = main_tex.read_text(encoding="utf-8", errors="replace")
    log_text = ""
    if compile_result and compile_result.get("log"):
        log_path = Path(str(compile_result.get("log")))
        if log_path.exists():
            log_text = log_path.read_text(encoding="utf-8", errors="replace")[-60000:]
    if not log_text:
        for name in ("latex_compile.log", "main.log"):
            p = bundle_dir / name
            if p.exists():
                log_text = p.read_text(encoding="utf-8", errors="replace")[-60000:]
                break
    repaired, report = repair_latex_source(source, compile_log=log_text)
    report["stage"] = stage
    report["bundle_dir"] = str(bundle_dir)
    if report.get("changed"):
        main_tex.write_text(repaired, encoding="utf-8")
    report_path = bundle_dir / "tex_code_agent_report.json"
    prior: list[dict[str, Any]] = []
    if report_path.exists():
        try:
            raw = json.loads(report_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict) and isinstance(raw.get("runs"), list):
                prior = raw["runs"]
        except Exception:
            prior = []
    prior.append(report)
    report_path.write_text(
        json.dumps({"schema_version": TEX_CODE_AGENT_VERSION, "runs": prior}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return report
