"""One declaration of what every auto_research_jobs state means.

Four times the autonomy chain stopped because code moved a job into a
``(status, stage)`` that no consumer selects: ``portfolio_granted``,
``required_output=experiment_plan``, ``proposal_generation_granted``, and the
capability-preflight states. Each was found by hand, months apart, after the
chain had already stalled in production.

The cause was never the individual state. It was that "which states get picked
up" lived as a hand-written SQL allowlist inside the claim query, "which states
get recycled" lived as a second hand-written set in the advancer, and "which
states are finished" lived nowhere at all. Adding a stage anywhere in the
codebase silently created a parking spot unless three separate places were
remembered.

This module is the one place. The claim query and the recycler are generated
from it, and the regression test in tests/test_job_state_consumers.py checks
every state a writer can produce against it. Adding a stage without declaring
its disposition fails that test at the commit that introduces it.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# A status that is claimable whatever the stage is. The stage on these rows is
# progress information, not authorization.
CLAIMABLE_STATUSES: tuple[str, ...] = (
    "queued",
    "eligible",
    "queued_cpu",
    "queued_gpu",
)


@dataclass(frozen=True)
class ClaimRule:
    """A status that is claimable only for specific stages.

    ``extra_sql`` carries a condition that is part of the authorization rather
    than of the state - the tier-1 rule below is only safe while the insight
    has no run yet.
    """

    status: str
    stages: tuple[str, ...]
    extra_sql: str = ""


CLAIM_RULES: tuple[ClaimRule, ...] = (
    ClaimRule(
        "failed",
        (
            "manual_reforge_unfinished",
            "manual_requeue_unfinished",
            "retry_failed_run",
            "manual_rerun_completed",
            "reset_completed_experiments",
        ),
    ),
    ClaimRule(
        "completed",
        ("tier1_research_complete",),
        extra_sql=(
            "NOT EXISTS (SELECT 1 FROM experiment_runs er "
            "WHERE er.deep_insight_id = {insight}.id)"
        ),
    ),
    ClaimRule(
        "blocked",
        (
            "cpu_ineligible",
            "verification_input_missing",
            "research_input_missing",
            "experiment_review_blocked",
            "experiment_review_repair_failed",
            "experiment_review_blocked_final",
        ),
    ),
)


# States a bounded recycler may rescue: work that spent resources and stopped
# somewhere no consumer claims. Rationed per idea by the advancer, because a
# candidate that fails the same way forever must not be retried forever.
RECYCLABLE: frozenset[tuple[str, str]] = frozenset(
    {
        ("failed", "forge_failed"),
        ("failed", "gpu_failed"),
        ("failed", "experiment_failed_repair_failed"),
        ("failed", "exception"),
        ("review_pending", "benchmark_harness_design_repair"),
        ("blocked", "experiment_review_blocked_final"),
    }
)


def _quote(values) -> str:
    return ", ".join("'" + str(value) + "'" for value in values)


def claim_predicate_sql(*, job: str = "arj", insight: str = "di") -> str:
    """Render the authorization half of the candidate-pool WHERE clause.

    Generated rather than hand-written so that the claim query, the recycler,
    and the regression test cannot drift apart. The shape is deliberately the
    same one the query used before: a NULL status, a claimable status, then one
    OR-group per stage-scoped rule.
    """

    parts = [
        f"{job}.status IS NULL",
        f"{job}.status IN ({_quote(CLAIMABLE_STATUSES)})",
    ]
    for rule in CLAIM_RULES:
        clause = (
            f"{job}.status='{rule.status}'\n"
            f"                    AND {job}.stage IN ({_quote(rule.stages)})"
        )
        if rule.extra_sql:
            clause += "\n                    AND " + rule.extra_sql.format(
                insight=insight, job=job
            )
        parts.append("(\n                    " + clause + "\n                  )")
    return "(\n               " + "\n               OR ".join(parts) + "\n             )"


def is_claimable(status: str | None, stage: str | None) -> bool:
    """Same decision as the SQL, for callers that already hold the row."""

    if status is None or status == "":
        return True
    if status in CLAIMABLE_STATUSES:
        return True
    for rule in CLAIM_RULES:
        if status == rule.status and stage in rule.stages:
            # extra_sql conditions are row-level and cannot be judged here; the
            # SQL remains the authority for those.
            return True
    return False


def declared_stages() -> frozenset[str]:
    """Every stage this module gives a disposition to."""

    stages: set[str] = set()
    for rule in CLAIM_RULES:
        stages.update(rule.stages)
    stages.update(stage for _status, stage in RECYCLABLE)
    return frozenset(stages)
