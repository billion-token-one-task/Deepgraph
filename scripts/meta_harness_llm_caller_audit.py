#!/usr/bin/env python3
"""Inventory legacy direct LLM calls without importing the application.

The release rule is conservative: any newly introduced direct ``call_llm`` or
``call_llm_json`` call fails this audit until it is role-routed, explicitly
classified as pre-agenda ingestion, or made unreachable from the default
registry/runtime. Classification is not authorization; the open ingestion
budget boundary remains visible in ``LLM_CALLER_INVENTORY.md``.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path


CLASSIFICATION = {
    "agents/abstraction_agent.py": "pre_agenda_ingestion_open",
    "agents/domain_summary_agent.py": "pre_agenda_ingestion_open",
    "agents/extraction_agent.py": "pre_agenda_ingestion_open",
    "agents/insight_agent.py": "pre_agenda_ingestion_open",
    "agents/multi_agent_extraction.py": "pre_agenda_ingestion_open",
    "agents/reasoning_agent.py": "pre_agenda_ingestion_open",
    "agents/taxonomy_expander.py": "pre_agenda_ingestion_open",
    "agents/codebase_scout.py": "legacy_not_default_registered",
    "agents/figure_agent.py": "legacy_no_generic_call_site",
    "agents/insight_ranker.py": "legacy_endpoint_and_registry_blocked",
    "agents/paperorchestra/refinement_loop.py": "legacy_not_default_registered",
    "agents/paperorchestra/tracing.py": "legacy_no_generic_call_site",
    "agents/paradigm_agent.py": "legacy_scheduler_and_registry_blocked",
}
DIRECT_NAMES = {"call_llm", "call_llm_json"}


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def audit(root: Path) -> tuple[list[dict[str, object]], list[str]]:
    calls: list[dict[str, object]] = []
    unknown: list[str] = []
    for base_name in ("agents", "orchestrator", "meta_harness", "web"):
        base = root / base_name
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.py")):
            relative = path.relative_to(root).as_posix()
            if relative == "agents/llm_client.py":
                continue
            tree = ast.parse(
                path.read_text(encoding="utf-8"),
                filename=relative,
            )
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                name = _call_name(node)
                if name not in DIRECT_NAMES:
                    continue
                classification = CLASSIFICATION.get(relative, "unclassified")
                calls.append(
                    {
                        "path": relative,
                        "line": int(node.lineno),
                        "call": name,
                        "classification": classification,
                    }
                )
                if classification == "unclassified":
                    unknown.append(f"{relative}:{node.lineno}:{name}")
    return calls, unknown


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).parents[1])
    args = parser.parse_args()
    calls, unknown = audit(args.root.resolve())
    counts: dict[str, int] = {}
    for call in calls:
        key = str(call["classification"])
        counts[key] = counts.get(key, 0) + 1
    print(f"direct_llm_calls={len(calls)}")
    for key, count in sorted(counts.items()):
        print(f"{key}={count}")
    print(f"unclassified={len(unknown)}")
    for item in unknown:
        print(item)
    return 1 if unknown else 0


if __name__ == "__main__":
    raise SystemExit(main())
