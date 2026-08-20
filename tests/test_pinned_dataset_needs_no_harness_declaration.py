"""A pinned, config-scoped dataset is already the materialization it asks for.

`resolve_benchmark_protocol` blocks any dataset registered as
`requires_harness` unless the row carries one of four readiness flags:
harness_materialized, materialized, dataset_cache_verified,
benchmark_harness_ready.

**None of those four names is written anywhere in the tree for a v1 plan.** The
flag the check waits for has no writer, so for a capability-bound experiment the
condition can never become true -- the same shape as the manuscript gate, which
verified an approval nobody minted.

Measured 2026-08-20: agenda 14 lost 14 of 14 runs here. Every M3 candidate, none
reaching compute, while the identical experiment ran by hand from the same
pinned revision and produced a positive control at p=0.000999.

BIG-Bench is registered requires_harness because its tasks need per-task
extraction. `config` IS that extraction -- declared, revision-pinned, with the
field mapping carried in execution_requirements and checked by capability
preflight before the plan is granted. A row with all three is not "awaiting
materialization"; it is materialized more precisely than the flag would prove.

The exemption is deliberately narrow: capability-bound contract AND a pinned
revision AND a config. Drop any one and the blocker returns, which the tests
below assert one at a time.
"""

import unittest

from agents.benchmark_protocol import resolve_benchmark_protocol

_ROW = {
    "name": "tasksource/bigbench:object_counting@210c156",
    "hf_dataset": "tasksource/bigbench",
    "config": "object_counting",
    "split": "train",
    "revision": "210c156767d2f4f05d2f4fd0bb275017a67040fd",
}
_V1 = {"schema_version": "experiment_requirements_v1"}


def _harness_blockers(plan):
    result = resolve_benchmark_protocol(plan)
    return [b for b in (result.get("blockers") or []) if "harness" in str(b)]


def _plan(row=None, requirements=_V1):
    plan = {"datasets": [dict(row or _ROW)], "metrics": ["numeric_accuracy"]}
    if requirements is not None:
        plan["execution_requirements"] = dict(requirements)
    return plan


class PinnedDatasetTests(unittest.TestCase):
    def test_a_fully_pinned_capability_bound_row_is_not_blocked(self):
        self.assertEqual(_harness_blockers(_plan()), [])


class TheExemptionStaysNarrowTests(unittest.TestCase):
    """Each condition is load-bearing. Remove one, the blocker comes back."""

    def test_no_contract_is_still_blocked(self):
        self.assertTrue(_harness_blockers(_plan(requirements=None)))

    def test_an_unrecognised_schema_version_is_still_blocked(self):
        self.assertTrue(
            _harness_blockers(_plan(requirements={"schema_version": "v99"}))
        )

    def test_an_unpinned_revision_is_still_blocked(self):
        row = {k: v for k, v in _ROW.items() if k != "revision"}
        self.assertTrue(_harness_blockers(_plan(row)))

    def test_a_blank_revision_is_still_blocked(self):
        self.assertTrue(_harness_blockers(_plan(dict(_ROW, revision="   "))))

    def test_a_missing_config_is_still_blocked(self):
        # Without the sub-task there is no per-task extraction to point at,
        # which is the whole reason BIG-Bench is registered requires_harness.
        row = {k: v for k, v in _ROW.items() if k != "config"}
        self.assertTrue(_harness_blockers(_plan(row)))

    def test_a_non_mapping_contract_is_still_blocked(self):
        plan = _plan(requirements=None)
        plan["execution_requirements"] = "experiment_requirements_v1"
        self.assertTrue(_harness_blockers(plan))


if __name__ == "__main__":
    unittest.main()
