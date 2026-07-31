#!/usr/bin/env python3
"""Static audit for unscoped mutations of agenda-owned tables.

Only Python source literals are parsed. Dynamic SQL is reported separately;
no application module or database is loaded.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOTS = ("agents", "meta_harness", "orchestrator", "web")
SCOPED_TABLES = {
    "research_problems",
    "deep_insights",
    "auto_research_jobs",
    "experiment_runs",
    "experiment_iterations",
    "experimental_claims",
    "experiment_artifacts",
    "experimental_evidence_edges",
    "gpu_jobs",
    "manuscript_runs",
    "manuscript_assets",
    "submission_bundles",
    "benchmark_harness_jobs",
    "frontier_packets",
    "idea_decision_packets",
    "resource_grants",
    "outcome_records",
    "compute_jobs_v1",
    "harness_candidates",
    "harness_patches",
    "harness_evaluation_runs",
    "harness_regression_reports",
    "failure_clusters",
    "harness_archives",
}
MUTATION = re.compile(
    r"\b(?:UPDATE\s+|DELETE\s+FROM\s+)([a-z_][a-z0-9_]*)",
    re.IGNORECASE,
)


def _literal(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        chunks = [
            value.value
            for value in node.values
            if isinstance(value, ast.Constant) and isinstance(value.value, str)
        ]
        return "".join(chunks)
    return None


def audit() -> dict:
    unscoped: list[dict[str, object]] = []
    dynamic: list[dict[str, object]] = []
    mutation_count = 0
    for root in PYTHON_ROOTS:
        for path in sorted((ROOT / root).rglob("*.py")):
            relative = path.relative_to(ROOT).as_posix()
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            parents = {
                child: parent
                for parent in ast.walk(tree)
                for child in ast.iter_child_nodes(parent)
            }
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call) or not node.args:
                    continue
                function = node.func
                if (
                    not isinstance(function, ast.Attribute)
                    or function.attr not in {"execute", "executemany"}
                ):
                    continue
                sql = _literal(node.args[0])
                if sql is None:
                    dynamic.append(
                        {
                            "path": relative,
                            "line": node.lineno,
                            "reason": "dynamic_sql",
                        }
                    )
                    continue
                matches = MUTATION.findall(sql)
                for raw_table in matches:
                    table = raw_table.lower()
                    if table not in SCOPED_TABLES:
                        continue
                    mutation_count += 1
                    if "agenda_id" not in sql.lower():
                        parent = parents.get(node)
                        while parent is not None and not isinstance(
                            parent,
                            (ast.FunctionDef, ast.AsyncFunctionDef),
                        ):
                            parent = parents.get(parent)
                        unscoped.append(
                            {
                                "path": relative,
                                "line": node.lineno,
                                "table": table,
                                "function": (
                                    parent.name
                                    if isinstance(
                                        parent,
                                        (ast.FunctionDef, ast.AsyncFunctionDef),
                                    )
                                    else "<module>"
                                ),
                            }
                        )
    return {
        "status": "passed" if not unscoped else "failed",
        "application_imported": False,
        "database_accessed": False,
        "scoped_literal_mutations": mutation_count,
        "definite_unscoped_mutations": unscoped,
        "dynamic_mutations_for_review": dynamic,
    }


def main() -> int:
    result = audit()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
