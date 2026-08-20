"""A capability-bound plan must not be refused for arms its runner cannot run.

Two evidence standards live in this system, and the older one blocked the newer
one from ever executing.

The review gate is paper-grade benchmark design: literature-sourced evidence per
dataset, a rationale for the benchmark set, and `minimum_baselines`, which has a
hard floor of 2. A meta-harness v1 experiment cannot satisfy that last one even
in principle -- GenericTransformersRunner declares a single
`BASELINE_METHOD = "unmodified_input_baseline"`, so a v1 plan naming two
baselines would be describing an arm that never runs. The requirement is not
strict-but-achievable; it is contradictory.

Measured on 2026-08-20: 38 of 143 failed runs died at this gate, the single
largest failure category, and agenda 14 lost 13 of 13 without reaching compute
once -- while the same experiments, run by hand outside the system, produced
results.

v1 does not lower the bar, it moves it: both arms are recomputed from
raw_predictions.jsonl and refused on mismatch, a cross-vendor evaluator may
dissent, the holdout is disjoint, and the verdict is significance-gated. What
this gate protects -- whether the topic is literature-grounded -- belongs at
topic selection, not in front of execution.

So these become warnings for a v1 contract and stay blockers for everything
else. The tests below pin BOTH halves, because a gate that stops refusing
everyone is not a fix, it is a hole.
"""

import unittest

from agents.experiment_review import review_experiment_candidate

_V1_REQUIREMENTS = {
    "schema_version": "experiment_requirements_v1",
    "task_protocol": "generative_qa",
}


def _insight(*, capability_bound, baselines=("unmodified_input_baseline",)):
    plan = {
        "baselines": list(baselines),
        "datasets": ["openai/gsm8k"],
        "model_targets": ["Qwen/Qwen2.5-1.5B"],
        "metrics": ["numeric_accuracy"],
        "seeds": [0],
        "minimum_seeds": 1,
        "benchmark_design_status": "literature_review_required",
        "benchmark_design_blockers": ["domain literature review pending"],
        "compute_budget": {"total_gpu_hours": 1.0},
        "procedure": "paired baseline/candidate evaluation at n=200",
    }
    if capability_bound:
        plan["execution_requirements"] = dict(_V1_REQUIREMENTS)
    return {
        "id": 1,
        "title": "t",
        "proposed_method": {"name": "m", "one_line": "o"},
        "experimental_plan": plan,
        "resource_class": "gpu",
    }


def _blockers(judgement):
    return [str(b) for b in (judgement.blockers or [])]


def _warnings(judgement):
    return [str(w) for w in (judgement.warnings or [])]


class CapabilityBoundExemptionTests(unittest.TestCase):
    def test_a_single_baseline_no_longer_blocks_a_v1_plan(self):
        judgement = review_experiment_candidate(_insight(capability_bound=True))
        self.assertFalse(
            [b for b in _blockers(judgement) if "baseline" in b.lower()],
            f"v1 plan still blocked on baselines: {_blockers(judgement)}",
        )

    def test_the_shortfall_is_still_recorded(self):
        # Exempt is not silent. The record must still say what was not met.
        judgement = review_experiment_candidate(_insight(capability_bound=True))
        self.assertTrue(
            [w for w in _warnings(judgement) if "baseline" in w.lower()],
            "the baseline shortfall vanished instead of being recorded",
        )

    def test_the_benchmark_design_gate_is_recorded_not_enforced(self):
        judgement = review_experiment_candidate(_insight(capability_bound=True))
        self.assertFalse(
            [b for b in _blockers(judgement) if "Benchmark design gate" in b],
            f"v1 plan still blocked by the design gate: {_blockers(judgement)}",
        )


class TheGateStillRefusesEveryoneElseTests(unittest.TestCase):
    """The half that matters: this must not have become a hole."""

    def test_a_plan_without_the_contract_is_still_blocked_on_baselines(self):
        judgement = review_experiment_candidate(_insight(capability_bound=False))
        self.assertTrue(
            [b for b in _blockers(judgement) if "baseline" in b.lower()],
            "a non-v1 plan with one baseline was allowed through",
        )

    def test_a_plan_without_the_contract_is_still_blocked_by_the_design_gate(self):
        judgement = review_experiment_candidate(_insight(capability_bound=False))
        self.assertTrue(
            [b for b in _blockers(judgement) if "Benchmark design gate" in b],
            "a non-v1 plan skipped the benchmark design gate",
        )

    def test_a_forged_schema_version_does_not_qualify(self):
        insight = _insight(capability_bound=True)
        insight["experimental_plan"]["execution_requirements"] = {
            "schema_version": "experiment_requirements_v99"
        }
        judgement = review_experiment_candidate(insight)
        self.assertTrue(
            [b for b in _blockers(judgement) if "baseline" in b.lower()],
            "an unrecognised schema version was treated as capability-bound",
        )

    def test_a_non_dict_contract_does_not_qualify(self):
        insight = _insight(capability_bound=True)
        insight["experimental_plan"]["execution_requirements"] = "yes"
        judgement = review_experiment_candidate(insight)
        self.assertTrue(
            [b for b in _blockers(judgement) if "baseline" in b.lower()],
            "a string was accepted as an execution requirements contract",
        )


if __name__ == "__main__":
    unittest.main()
