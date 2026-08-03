#!/usr/bin/env python3
"""Side-effect-free source audit for meta-harness-v1.

This command parses files as text/AST. It does not import DeepGraph modules,
open a database connection, start the application, or execute a migration.
It is safe for the restricted production-adjacent host described in the
integration runbooks.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOTS = (
    "agents",
    "contracts",
    "meta_harness",
    "orchestrator",
    "plugins/examples/cggr",
    "scripts",
    "tests",
    "web",
)
TOPIC_MARKERS = re.compile(r"(?i)\b(?:cggr|crpp|idea8|run13)\b|eval_cggr")
TOPIC_BOUNDARY_ALLOWLIST = {
    "agents/experiment_forge.py",
    "agents/paperorchestra/figure_orchestra.py",
    "agents/paperorchestra/full_pipeline.py",
    "scripts/meta_harness_static_audit.py",
}
REMOVED_INTEGRITY_SYMBOLS = {
    "_complete_known_main_results_rows",
    "_deemphasize_significance_caveats",
}
DESTRUCTIVE_SQL = re.compile(
    r"\b(?:DROP|TRUNCATE|DELETE|ALTER\s+COLUMN|RENAME\s+TO)\b",
    re.IGNORECASE,
)
SECRET_LITERAL = re.compile(
    r"(?i)^\s*(?:api[_-]?key|password|token|secret|oauth[_-]?token)"
    r"\s*=\s*['\"][^'\"]+['\"]\s*$"
)


def _relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _python_files() -> list[Path]:
    return sorted(
        {
            path
            for root in PYTHON_ROOTS
            for path in (ROOT / root).rglob("*.py")
            if "__pycache__" not in path.parts
        }
    )


def _strip_sql_comments(sql: str) -> str:
    return "\n".join(
        line for line in sql.splitlines() if not line.lstrip().startswith("--")
    )


def audit() -> dict:
    findings: list[dict[str, object]] = []
    python_files = _python_files()
    for path in python_files:
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except Exception as exc:
            findings.append(
                {
                    "check": "python_ast",
                    "path": _relative(path),
                    "detail": f"{type(exc).__name__}: {exc}",
                }
            )

    generic_roots = [
        ROOT / "agents",
        ROOT / "orchestrator",
        ROOT / "scripts",
        ROOT / "web",
    ]
    generic_files = sorted(
        path for root in generic_roots for path in root.rglob("*.py")
    )
    for path in generic_files:
        relative = _relative(path)
        text = path.read_text(encoding="utf-8")
        if TOPIC_MARKERS.search(text) and relative not in TOPIC_BOUNDARY_ALLOWLIST:
            findings.append(
                {
                    "check": "topic_plugin_boundary",
                    "path": relative,
                    "detail": "topic marker remains outside the explicit example boundary",
                }
            )
        if relative == "scripts/meta_harness_static_audit.py":
            continue
        for symbol in REMOVED_INTEGRITY_SYMBOLS:
            if symbol in text:
                findings.append(
                    {
                        "check": "removed_integrity_bypass",
                        "path": relative,
                        "detail": symbol,
                    }
                )

    migrations = sorted((ROOT / "db" / "migrations").glob("*.sql"))
    for migration in migrations:
        migration_text = migration.read_text(encoding="utf-8")
        destructive = sorted(
            {match.group(0).upper() for match in DESTRUCTIVE_SQL.finditer(
                _strip_sql_comments(migration_text)
            )}
        )
        if destructive:
            findings.append(
                {
                    "check": "additive_migration",
                    "path": _relative(migration),
                    "detail": destructive,
                }
            )

    for path in (
        ROOT / "deepgraph.toml",
        ROOT / "research_agendas" / "meta_harness_v1.example.json",
    ):
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if SECRET_LITERAL.match(line):
                findings.append(
                    {
                        "check": "literal_secret",
                        "path": _relative(path),
                        "line": line_number,
                        "detail": "credential-like literal must be an environment/secret reference",
                    }
                )

    return {
        "status": "passed" if not findings else "failed",
        "python_files_ast_parsed": len(python_files),
        "migrations_audited": len(migrations),
        "migration_sha256_checked": True,
        "database_accessed": False,
        "application_imported": False,
        "findings": findings,
    }


def main() -> int:
    report = audit()
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
