#!/usr/bin/env python3
"""Side-effect-free SQL placeholder audit.

The script parses Python source only. It never imports DeepGraph modules or
opens a database connection. Literal SQL calls are checked for ``?``
placeholder/argument arity; dynamic SQL or dynamic parameter collections are
reported separately for manual review.
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
    "scripts",
    "tests",
    "web",
)
DB_METHODS = {
    "execute",
    "executemany",
    "fetchone",
    "fetchall",
    "insert_returning_id",
}
PSYCOPG_PLACEHOLDER = re.compile(r"(?<!%)%s")


def _python_files() -> list[Path]:
    return sorted(
        {
            path
            for root in PYTHON_ROOTS
            for path in (ROOT / root).rglob("*.py")
            if "__pycache__" not in path.parts
        }
    )


def _literal_sql(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _parameter_arity(node: ast.AST | None) -> int | None:
    if node is None:
        return 0
    if isinstance(node, (ast.Tuple, ast.List)):
        if any(isinstance(item, ast.Starred) for item in node.elts):
            return None
        return len(node.elts)
    return None


def _placeholder_count(sql: str) -> int:
    return sql.count("?") + len(PSYCOPG_PLACEHOLDER.findall(sql))


def audit() -> dict:
    literal_calls = 0
    checked_calls = 0
    dynamic_calls: list[dict[str, object]] = []
    mismatches: list[dict[str, object]] = []

    for path in _python_files():
        relative = path.relative_to(ROOT).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            function = node.func
            if not isinstance(function, ast.Attribute) or function.attr not in DB_METHODS:
                continue
            if not node.args:
                dynamic_calls.append(
                    {
                        "path": relative,
                        "line": node.lineno,
                        "method": function.attr,
                        "reason": "sql_not_positional",
                    }
                )
                continue
            sql = _literal_sql(node.args[0])
            if sql is None:
                dynamic_calls.append(
                    {
                        "path": relative,
                        "line": node.lineno,
                        "method": function.attr,
                        "reason": "dynamic_sql",
                    }
                )
                continue
            literal_calls += 1
            params_node = node.args[1] if len(node.args) > 1 else None
            if params_node is None:
                params_node = next(
                    (
                        keyword.value
                        for keyword in node.keywords
                        if keyword.arg in {"params", "parameters"}
                    ),
                    None,
                )
            arity = _parameter_arity(params_node)
            if arity is None:
                dynamic_calls.append(
                    {
                        "path": relative,
                        "line": node.lineno,
                        "method": function.attr,
                        "reason": "dynamic_parameters",
                        "placeholders": _placeholder_count(sql),
                    }
                )
                continue
            checked_calls += 1
            placeholders = _placeholder_count(sql)
            if placeholders != arity:
                mismatches.append(
                    {
                        "path": relative,
                        "line": node.lineno,
                        "method": function.attr,
                        "placeholders": placeholders,
                        "parameters": arity,
                    }
                )

    return {
        "status": "passed" if not mismatches else "failed",
        "application_imported": False,
        "database_accessed": False,
        "literal_sql_calls": literal_calls,
        "literal_calls_checked": checked_calls,
        "dynamic_call_count": len(dynamic_calls),
        "dynamic_calls": dynamic_calls,
        "definite_mismatches": mismatches,
    }


def main() -> int:
    report = audit()
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
