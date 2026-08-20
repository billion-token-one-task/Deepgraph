"""Find keys the code reads but nothing ever writes.

Four defects in one day shared exactly this shape -- a gate waiting on a flag
that has no writer anywhere in the tree:

    evidence_decision_passed   the ladder refused every `supported` verdict,
                               because evidence_audit never set it
    reviewer_approval          manuscript_allowed verified an approval that no
                               code path minted
    harness_materialized       and its three siblings; agenda 14 lost 15 runs
                               to a readiness flag nothing sets for a v1 plan
    empirical_posterior        four signal tables, 100% NULL, because the only
                               writer copied the value from itself

Each was found by hand, days apart, after it had already cost real runs. The
shape is mechanical, so finding it should be too.

The scan is deliberately crude and reports rather than judges: it collects every
string literal read through ``.get("x")`` or ``["x"]`` and every string literal
that appears where a value is written -- a dict literal key, a keyword argument,
an assignment to ``obj["x"]``, or a SQL column in INSERT/SET. A key read in the
first set and absent from the second is a candidate.

False positives are expected and fine: keys from provider JSON, HTTP payloads
and dataset rows are legitimately read-only. The output is a starting list for a
human, not a verdict. What matters is that a flag with no writer cannot hide in
it -- ``harness_materialized`` sits in this report today.

Usage:
    python scripts/find_unwritten_flags.py [--root .] [--min-reads 2]
"""

from __future__ import annotations

import argparse
import ast
import re
from collections import defaultdict
from pathlib import Path

# Written by the database, the ORM layer or an external payload rather than by
# our own code; reading them without a local writer is correct.
_SQL_WRITE = re.compile(
    r"(?:INSERT\s+INTO|UPDATE)\s+[\"\w.]+|SET\s+([\w\"]+)\s*=|\(([^)]*)\)\s*VALUES",
    re.IGNORECASE,
)


def _read_keys(tree: ast.AST) -> set[str]:
    keys: set[str] = set()
    for node in ast.walk(tree):
        # obj.get("x") / obj.get("x", default)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            keys.add(node.args[0].value)
        # obj["x"] in a load context only
        elif (
            isinstance(node, ast.Subscript)
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, str)
            and isinstance(node.ctx, ast.Load)
        ):
            keys.add(node.slice.value)
    return keys


def _written_keys(tree: ast.AST, source: str) -> set[str]:
    keys: set[str] = set()
    for node in ast.walk(tree):
        # {"x": ...}
        if isinstance(node, ast.Dict):
            for key in node.keys:
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    keys.add(key.value)
        # f(x=...) and dict(x=...)
        elif isinstance(node, ast.keyword) and node.arg:
            keys.add(node.arg)
        # obj["x"] = ...
        elif (
            isinstance(node, ast.Subscript)
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, str)
            and isinstance(node.ctx, ast.Store)
        ):
            keys.add(node.slice.value)
        # x = ..., def f(x=...), class attr x: T = ...
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            keys.add(node.id)
        elif isinstance(node, ast.arg):
            keys.add(node.arg)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            keys.add(node.target.id)
    # SQL columns: cheap and over-broad on purpose, so a column this code
    # writes only through raw SQL is never reported as unwritten.
    for match in _SQL_WRITE.finditer(source):
        for group in match.groups():
            if group:
                keys.update(re.findall(r"\w+", group))
    return keys


def scan(root: Path, *, min_reads: int = 2) -> list[tuple[str, int, list[str]]]:
    reads: dict[str, int] = defaultdict(int)
    read_sites: dict[str, set[str]] = defaultdict(set)
    written: set[str] = set()
    for path in sorted(root.rglob("*.py")):
        parts = set(path.parts)
        if parts & {".git", "node_modules", "__pycache__", "tests", "plugins"}:
            continue
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except (OSError, SyntaxError):
            continue
        rel = str(path.relative_to(root))
        for key in _read_keys(tree):
            reads[key] += 1
            read_sites[key].add(rel)
        written |= _written_keys(tree, source)
    out = []
    for key, count in reads.items():
        if key in written or count < min_reads:
            continue
        # Single characters and dunders are noise, not flags.
        if len(key) < 4 or key.startswith("__"):
            continue
        out.append((key, count, sorted(read_sites[key])[:3]))
    return sorted(out, key=lambda item: (-item[1], item[0]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--min-reads", type=int, default=2)
    args = parser.parse_args()
    findings = scan(Path(args.root).resolve(), min_reads=args.min_reads)
    print(f"keys read but never written: {len(findings)}")
    print("(reports; does not judge -- external payload keys belong here too)")
    for key, count, sites in findings:
        print(f"  {key:<44} read {count:>3}x  {', '.join(sites)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
