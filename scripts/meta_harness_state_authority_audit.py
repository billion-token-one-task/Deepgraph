#!/usr/bin/env python3
"""AST-only audit for the canonical scientific evidence state authority.

This script parses source text and never imports application modules. It
rejects any Python SQL mutation of ``scientific_evidence_state`` outside the
single repository authority. Initial INSERTs may set ``planned``; all later
transitions must pass through ``EvidenceRepository.advance_state``.
"""

from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path


MUTATION = re.compile(
    r"\b(?:UPDATE|INSERT\s+INTO)\s+experiment_runs\b",
    re.IGNORECASE,
)
STATE_COLUMN = re.compile(r"\bscientific_evidence_state\b", re.IGNORECASE)
UPDATE = re.compile(r"\bUPDATE\s+experiment_runs\b", re.IGNORECASE)
AUTHORITY = Path("meta_harness/repository.py")
SCAN_ROOTS = (
    Path("agents"),
    Path("contracts"),
    Path("db"),
    Path("meta_harness"),
    Path("orchestrator"),
    Path("scripts"),
    Path("web"),
)


def _string_literals(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield node.lineno, node.value


def audit(root: Path) -> tuple[int, list[str]]:
    observed = 0
    violations: list[str] = []
    for scan_root in SCAN_ROOTS:
        base = root / scan_root
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.py")):
            relative = path.relative_to(root)
            tree = ast.parse(
                path.read_text(encoding="utf-8"),
                filename=str(relative),
            )
            for line, literal in _string_literals(tree):
                if not MUTATION.search(literal) or not STATE_COLUMN.search(literal):
                    continue
                observed += 1
                if UPDATE.search(literal) and relative != AUTHORITY:
                    violations.append(
                        f"{relative}:{line}: scientific state UPDATE outside "
                        f"{AUTHORITY}"
                    )
                if not UPDATE.search(literal) and relative != Path(
                    "agents/experiment_forge.py"
                ):
                    violations.append(
                        f"{relative}:{line}: scientific state INSERT outside "
                        "the reviewed run factory"
                    )
    return observed, violations


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).parents[1])
    args = parser.parse_args()
    observed, violations = audit(args.root.resolve())
    print(f"scientific_state_sql_literals={observed}")
    print(f"authority_violations={len(violations)}")
    for violation in violations:
        print(violation)
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
