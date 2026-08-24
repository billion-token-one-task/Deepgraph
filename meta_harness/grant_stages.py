"""Canonical ResourceGrant stage classification.

``ResourceGrant.stage`` is an authority boundary, not a free-form progress
label.  In particular, a scoped-ingestion grant must not be mistaken for the
initial scientific execution grant that advances ``auto_research_jobs``.
Keep the classification anchored and closed so spelling variants cannot
silently select a different lifecycle lane.
"""

from __future__ import annotations

import re


class ResourceGrantStageError(ValueError):
    """A grant stage does not name one supported authority lane."""


PROPOSAL_GRANT_LANE = "proposal"
INITIAL_RESEARCH_GRANT_LANE = "initial_research"
LATER_RESEARCH_GRANT_LANE = "later_research"
INGESTION_GRANT_LANE = "ingestion"

LATER_RESEARCH_GRANT_STAGES = frozenset(
    {
        "validation",
        "full_benchmark",
        "evidence_audit",
        "manuscript",
    }
)

# The status API needs the same closed namespace as grant admission.  Export a
# value set instead of teaching its SQL a second, fuzzy ``ingestion%`` rule:
# unknown legacy spellings are not research authority, and exact ingestion
# variants remain outside this set by construction.
RESEARCH_GRANT_STAGES = frozenset(
    {"proposal", "pilot", *LATER_RESEARCH_GRANT_STAGES}
)

# ``ingestion`` is the stable lane name.  Bounded operator variants such as
# ``ingestion_backfill_canary`` remain explicit members of that namespace.
# The full match is intentional: ``ingestionish`` and punctuation/case/space
# variants must not acquire ingestion semantics by a fuzzy prefix check.
_INGESTION_STAGE = re.compile(r"ingestion(?:_[a-z0-9]+)*\Z")


def classify_resource_grant_stage(stage: str) -> str:
    """Return the one lifecycle lane authorized by an exact stage string."""

    value = str(stage or "")
    if value == "proposal":
        return PROPOSAL_GRANT_LANE
    if value == "pilot":
        return INITIAL_RESEARCH_GRANT_LANE
    if value in LATER_RESEARCH_GRANT_STAGES:
        return LATER_RESEARCH_GRANT_LANE
    if _INGESTION_STAGE.fullmatch(value):
        return INGESTION_GRANT_LANE
    raise ResourceGrantStageError(f"unsupported ResourceGrant stage: {value!r}")


def require_ingestion_grant_stage(stage: str) -> None:
    """Require an exact member of the scoped-ingestion stage namespace."""

    if classify_resource_grant_stage(stage) != INGESTION_GRANT_LANE:
        raise ResourceGrantStageError(
            f"scoped ingestion requires an ingestion ResourceGrant stage: {stage!r}"
        )
