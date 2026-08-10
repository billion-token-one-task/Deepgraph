"""Every job state a writer can produce must have someone who reads it back.

Four times now the autonomy chain has stopped because code moved a job into a
``(status, stage)`` no consumer selects: ``portfolio_granted``,
``required_output=experiment_plan``, ``proposal_generation_granted``, and the
capability-preflight states. Each was found by hand, months apart, after the
chain had already stalled in production.

The defect class is structural: ``_candidate_pool`` claims jobs from a
hard-coded allowlist, so adding a stage anywhere in the codebase silently
creates a parking spot unless someone remembers to extend that allowlist. This
test closes the loop mechanically - it enumerates the states writers can
actually produce and fails when one of them has nowhere to go.

It deliberately does not assert that the ledgers below are empty. Terminal
states are legitimate, and the acknowledged-debt ledger records states that are
stranded today. What it enforces is that the debt cannot grow silently: a new
unconsumed state fails this test at the commit that introduces it.
"""

from __future__ import annotations

import ast
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules", "static",
             "plugins", "tests"}

# States that are meant to have no consumer: the job is finished, successfully
# or otherwise, and nothing should pick it up again.
TERMINAL_BY_DESIGN: dict[tuple[str, str], str] = {
    ("completed", "outcome_recorded"): "settled into an OutcomeRecord; rerunning would double-charge",
    ("completed", "closed_loop_complete"): "full loop finished",
    ("completed", "manuscript_stale"): "watchdog retired a superseded manuscript",
    ("bundle_ready", "manuscript_stale"): "same, from the bundle-ready branch",
    ("completed", "full_benchmark_complete"): "remote full-benchmark monitor's terminal state",
    ("blocked", "novelty_not_novel"): "novelty gate rejected the idea; not a resource strand",
    ("blocked", "prior_work_exists"): "novelty gate found prior work; idea retired",
    ("blocked", "novelty_partially_exists"): "novelty gate rejection",
    ("failed", "capability_preflight_blocked"): "candidate is outside declared runner capabilities",
}

# States whose consumer exists but is not a stage predicate, so no SQL scan can
# find it. Each entry names the consumer so the claim stays checkable by hand.
CONSUMED_OUTSIDE_SQL: dict[tuple[str, str], str] = {
    ("deferred", "proposal_generation_granted"):
        "MetaHarnessRepository.complete_proposal_generation, reached from "
        "discovery_scheduler.run_tier2_discovery via the grant id on the "
        "realized insight; it returns the job to awaiting_portfolio_decision",
}

# States that are stranded today and known to be so. Shrink this list; do not
# extend it. Census 2026-08-10, recorded with the V1 chain-unblock work.
ACKNOWLEDGED_STRANDED: dict[tuple[str, str], str] = {
    ("blocked", "deep_research_input_missing"): "census 2026-08-10",
    ("blocked", "evosci_binary_missing"): "census 2026-08-10",
    ("blocked", "evosci_report_required_before_compute"): "census 2026-08-10",
    ("blocked", "gpu_unavailable"): "census 2026-08-10",
    ("failed", "experiment_failed"): "census 2026-08-10; 35 live rows",
    ("failed", "missing_run"): "census 2026-08-10",
    ("failed", "verification_failed"): "census 2026-08-10",
    ("failed", "verification_stale"): "census 2026-08-10",
    ("review_pending", "experiment_review"): "census 2026-08-10; in-flight, no stale recovery",
    ("running_cpu", "validation_loop"): "census 2026-08-10; in-flight, no stale recovery",
    ("smoke_only", "experiment_review_smoke_only"): "census 2026-08-10; 1 live row",
    ("verifying", "novelty_verification"): "census 2026-08-10; in-flight, no stale recovery",
    ("blocked", "experiment_automation_failed_final"): "census 2026-08-10",
    ("blocked", "novelty_verification_required"): "census 2026-08-10",
    ("blocked", "review_scaffold_stale_repair_exhausted"): "census 2026-08-10",
    ("deferred", "capability_preflight_deferred"): "census 2026-08-10; nothing re-runs preflight",
    ("failed", "deep_research_launch_failed"): "census 2026-08-10",
    ("failed", "experiment_review_failed"): "census 2026-08-10",
    ("failed", "gpu_blocked"): "census 2026-08-10",
    ("failed", "research_stale"): "census 2026-08-10",
    ("failed", "verification_launch_failed"): "census 2026-08-10",
    ("researching", "evosci_deep_research_running"): "census 2026-08-10; in-flight, no stale recovery",
    ("review_pending", "scientific_decision_required"): "census 2026-08-10",
    ("running_cpu", "manuscript_revision"): "census 2026-08-10; in-flight, no stale recovery",
}


def _source_files() -> list[Path]:
    out = []
    for path in ROOT.rglob("*.py"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        out.append(path)
    return out


class _UpsertVisitor(ast.NodeVisitor):
    """Collect status/stage kwargs from every _upsert_job call site."""

    def __init__(self, rel: str):
        self.rel = rel
        self.found: list[tuple[str, str, str]] = []

    def visit_Call(self, node: ast.Call) -> None:
        name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        if name in {"_upsert_job", "upsert_job"}:
            kw: dict[str, str] = {}
            for keyword in node.keywords:
                if keyword.arg not in {"status", "stage"}:
                    continue
                if isinstance(keyword.value, ast.Constant) and isinstance(
                    keyword.value.value, str
                ):
                    kw[keyword.arg] = keyword.value.value
                else:
                    kw[keyword.arg] = "<dynamic>"
            self.found.append(
                (kw.get("status", "<unset>"), kw.get("stage", "<unset>"),
                 f"{self.rel}:{node.lineno}")
            )
        self.generic_visit(node)


_SQL_UPDATE = re.compile(
    r"UPDATE\s+auto_research_jobs\s+SET(.{0,400}?)(?:WHERE|\"\"\")", re.S | re.I
)


def writable_pairs() -> dict[tuple[str, str], list[str]]:
    pairs: dict[tuple[str, str], list[str]] = {}
    for path in _source_files():
        rel = str(path.relative_to(ROOT))
        text = path.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        visitor = _UpsertVisitor(rel)
        visitor.visit(tree)
        for status, stage, site in visitor.found:
            pairs.setdefault((status, stage), []).append(site)
        for match in _SQL_UPDATE.finditer(text):
            body = match.group(1)
            status = re.search(r"status\s*=\s*'([a-z0-9_]+)'", body, re.I)
            stage = re.search(r"stage\s*=\s*'([a-z0-9_]+)'", body, re.I)
            if not (status or stage):
                continue
            line = text.count("\n", 0, match.start()) + 1
            pairs.setdefault(
                (status.group(1) if status else "<unset>",
                 stage.group(1) if stage else "<unset>"),
                [],
            ).append(f"{rel}:{line}")
    return pairs


def _candidate_pool_sql() -> str:
    text = (ROOT / "orchestrator" / "auto_research.py").read_text(encoding="utf-8")
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_candidate_pool":
            return ast.get_source_segment(text, node) or ""
    raise AssertionError("_candidate_pool not found; the claim predicate moved")


def pool_predicate() -> tuple[set[str], dict[str, set[str]]]:
    """Read the claimable statuses and per-status stage allowlists out of the
    live candidate-pool SQL, so this test tracks the real predicate."""

    sql = _candidate_pool_sql()
    where = sql.split("WHERE di.agenda_id", 1)[-1].split("ORDER BY", 1)[0]

    statuses = set()
    match = re.search(r"arj\.status\s+IN\s*\(([^)]*)\)", where, re.I)
    if match:
        statuses = set(re.findall(r"'([a-z0-9_]+)'", match.group(1)))

    # Each `arj.status='X'` opens a branch that runs until the next one; take
    # every stage literal inside that span. Nesting the equality and IN forms
    # under an inner AND ( ... ) makes bracket matching unreliable, and this
    # span rule tolerates both.
    stage_allowlists: dict[str, set[str]] = {}
    marks = [
        (m.group(1), m.start())
        for m in re.finditer(r"arj\.status\s*=\s*'([a-z0-9_]+)'", where, re.I)
    ]
    for index, (status, start) in enumerate(marks):
        end = marks[index + 1][1] if index + 1 < len(marks) else len(where)
        block = where[start:end]
        stages = set(re.findall(r"arj\.stage\s*=\s*'([a-z0-9_]+)'", block, re.I))
        for group in re.findall(r"arj\.stage\s+IN\s*\(([^)]*)\)", block, re.S | re.I):
            stages |= set(re.findall(r"'([a-z0-9_]+)'", group))
        if stages:
            stage_allowlists.setdefault(status, set()).update(stages)
    return statuses, stage_allowlists


_SELECT_JOBS = re.compile(
    r"(?:SELECT|UPDATE)\b(?:(?!\bSELECT\b|\bUPDATE\b).)*?auto_research_jobs"
    r"(?:(?!\bSELECT\b|\bUPDATE\b).)*",
    re.S | re.I,
)


def selected_stages() -> set[str]:
    """Stages named in the WHERE clause of any query over auto_research_jobs.

    Recovery and requeue paths - the advancer's withdrawn-grant requeue, the
    legacy-job recyclers - are genuine consumers even though they sit outside
    the candidate pool. Deriving them from the queries themselves keeps this
    test honest as those queries change.
    """

    stages: set[str] = set()
    for path in _source_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        if "auto_research_jobs" not in text:
            continue
        for statement in _SELECT_JOBS.findall(text):
            head, _, where = statement.partition("WHERE")
            if not where:
                continue
            stages |= set(
                re.findall(r"(?:arj\.)?stage\s*=\s*'([a-z0-9_]+)'", where, re.I)
            )
            for group in re.findall(
                r"(?:arj\.)?stage\s+IN\s*\(([^)]*)\)", where, re.S | re.I
            ):
                stages |= set(re.findall(r"'([a-z0-9_]+)'", group))
    return stages


def dead_end_recycled() -> set[tuple[str, str]]:
    """The advancer's bounded recycler is a real, if rationed, consumer."""

    text = (ROOT / "scripts" / "auto_advance.py").read_text(encoding="utf-8")
    block = re.search(r"DEAD_END\s*=\s*\{(.*?)\n\}", text, re.S)
    if not block:
        raise AssertionError("DEAD_END set not found in auto_advance.py")
    return {
        (status, stage)
        for status, stage in re.findall(
            r"\(\s*\"([a-z0-9_]+)\"\s*,\s*\"([a-z0-9_]+)\"\s*\)", block.group(1)
        )
    }


class JobStateConsumerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.pairs = writable_pairs()
        self.statuses, self.stage_allowlists = pool_predicate()
        self.recycled = dead_end_recycled()
        self.selected = selected_stages()

    def _has_consumer(self, status: str, stage: str) -> bool:
        pair = (status, stage)
        return (
            status in self.statuses
            or stage in self.stage_allowlists.get(status, set())
            or stage in self.selected
            or pair in self.recycled
            or pair in CONSUMED_OUTSIDE_SQL
        )

    def test_claim_predicate_parses(self) -> None:
        """Guard the guard: a restructured SQL must not silently pass everything."""

        self.assertIn("queued", self.statuses)
        self.assertTrue(self.stage_allowlists, "no per-status stage allowlist parsed")
        self.assertTrue(self.recycled, "no DEAD_END recycle pairs parsed")

    def test_every_writable_state_has_a_consumer(self) -> None:
        orphans: list[str] = []
        for (status, stage), sites in sorted(self.pairs.items()):
            if "<" in status or "<" in stage:
                # Dynamic or absent literal: the pair cannot be resolved
                # statically, and the column keeps its previous value.
                continue
            pair = (status, stage)
            if self._has_consumer(status, stage):
                continue
            if pair in TERMINAL_BY_DESIGN or pair in ACKNOWLEDGED_STRANDED:
                continue
            orphans.append(
                f"  status={status!r} stage={stage!r} written at {sites[0]}"
            )
        self.assertEqual(
            orphans,
            [],
            "These job states have a writer and no reader. A job that lands "
            "here stops the autonomy chain silently.\n" + "\n".join(orphans)
            + "\n\nFix the wiring, or - if the state really is terminal - add "
            "it to TERMINAL_BY_DESIGN with a reason.",
        )

    def test_acknowledged_debt_does_not_grow(self) -> None:
        """Entries that got a consumer must leave the debt ledger."""

        resolved = [
            pair
            for pair in sorted(ACKNOWLEDGED_STRANDED)
            if self._has_consumer(*pair)
        ]
        self.assertEqual(
            resolved,
            [],
            "These states now have a consumer; remove them from "
            f"ACKNOWLEDGED_STRANDED: {resolved}",
        )


if __name__ == "__main__":
    unittest.main()
