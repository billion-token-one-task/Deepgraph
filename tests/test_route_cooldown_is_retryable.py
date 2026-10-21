"""A rate limit must cost time, never a paper.

When every route for a role is in transient cooldown the router raises
LLMRouteUnavailableError, whose message the pipeline classifies to decide
whether the paper is retried or retired. Classifying that as permanent is not
a cosmetic mistake: the paper leaves the work queue and no operator path
brings it back, so a three-minute gateway throttle silently costs the corpus.
"""

from meta_harness.llm_routing import LLMRouteUnavailableError
from orchestrator.pipeline import _is_retryable_pipeline_error


def test_router_cooldown_message_is_retryable():
    error = LLMRouteUnavailableError(
        "all_explicit_routes_failed;manual_review_required:"
    )
    assert _is_retryable_pipeline_error(error)


def test_cooldown_wrapped_by_multi_agent_extraction_is_retryable():
    """The pipeline sees the router's message wrapped per role, not bare."""
    wrapped = (
        "multi-agent extraction failed closed: "
        "claims_methods=all_explicit_routes_failed;manual_review_required:; "
        "empirical_results=all_explicit_routes_failed;manual_review_required:"
    )
    assert _is_retryable_pipeline_error(wrapped)


def test_a_paper_with_no_text_stays_permanent():
    """The guard must not turn every failure into an infinite retry."""
    assert not _is_retryable_pipeline_error("No text available")
