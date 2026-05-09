import unittest

from contracts import (
    DeepInsightSpec,
    ManuscriptInputState,
    StructuredPaperRecord,
    normalize_deep_insight_storage,
)
from contracts.base import ContractValidationError


class StructuredPaperRecordTests(unittest.TestCase):
    def test_checkpoint_payload_omits_full_text_but_keeps_length(self):
        record = StructuredPaperRecord.from_processing_state(
            paper={"id": "p1", "title": "Test Paper"},
            full_text="abcdef",
            extraction={"claims": [{"text": "c1"}], "results": [{"metric": 1.0}]},
            processing_stage="extracted",
        )

        payload = record.checkpoint_payload()["structured_paper_record"]

        self.assertNotIn("full_text", payload)
        self.assertEqual(payload["full_text_length"], 6)
        self.assertEqual(len(payload["claims"]), 1)


class DeepInsightSpecTests(unittest.TestCase):
    def test_normalize_deep_insight_storage_dedupes_signal_and_source_lists(self):
        spec = DeepInsightSpec.from_raw(
            {
                "tier": 2,
                "title": "Insight",
                "signal_mix": ["plateau", "plateau", "protocol_artifact"],
                "supporting_papers": ["2401.1", "2401.1", "2401.2"],
                "source_node_ids": ["ml.a", "ml.a", "ml.b"],
            }
        )

        payload = normalize_deep_insight_storage(spec)

        self.assertEqual(payload["signal_mix"], ["plateau", "protocol_artifact"])
        self.assertEqual(payload["supporting_papers"], ["2401.1", "2401.2"])
        self.assertEqual(payload["source_node_ids"], ["ml.a", "ml.b"])


class ManuscriptInputStateTests(unittest.TestCase):
    def test_submission_ready_blocks_non_formal_state(self):
        state = ManuscriptInputState(
            run_id=1,
            deep_insight_id=2,
            formal_experiment=False,
            smoke_test_only=True,
            title="Smoke",
            method_name="Method",
            claims=[{"claim_text": "x"}],
            citation_seed_paper_ids=["2401.1"],
            result_packet={"formal_experiment": False},
        )

        with self.assertRaises(ContractValidationError):
            state.require_submission_ready()

    def test_submission_ready_blocks_reproduction_only_state(self):
        state = ManuscriptInputState(
            run_id=1,
            deep_insight_id=2,
            formal_experiment=True,
            smoke_test_only=False,
            title="Pilot",
            method_name="Method",
            claims=[{"claim_text": "x"}],
            citation_seed_paper_ids=["2401.1"],
            result_packet={
                "formal_experiment": True,
                "verdict": "reproduced",
                "hypothesis_iterations": [],
            },
        )

        with self.assertRaises(ContractValidationError):
            state.require_submission_ready()

    def test_submission_ready_blocks_bootstrap_probe_evidence(self):
        state = ManuscriptInputState(
            run_id=1,
            deep_insight_id=2,
            formal_experiment=True,
            smoke_test_only=False,
            title="Pilot",
            method_name="Method",
            claims=[{"claim_text": "x"}],
            citation_seed_paper_ids=["2401.1"],
            result_packet={
                "formal_experiment": True,
                "verdict": "confirmed",
                "hypothesis_iterations": [{"iteration_number": 1}],
                "evidence_tier": "bootstrap_probe",
                "blocks_manuscript": True,
            },
        )

        with self.assertRaises(ContractValidationError):
            state.require_submission_ready()

    def test_submission_ready_blocks_sanity_real_benchmark_evidence(self):
        state = ManuscriptInputState(
            run_id=1,
            deep_insight_id=2,
            formal_experiment=True,
            smoke_test_only=False,
            title="Pilot",
            method_name="Method",
            claims=[{"claim_text": "x"}],
            citation_seed_paper_ids=["2401.1"],
            result_packet={
                "formal_experiment": True,
                "verdict": "confirmed",
                "hypothesis_iterations": [{"iteration_number": 1}],
                "evidence_tier": "sanity_real_benchmark",
                "blocks_manuscript": True,
            },
        )

        with self.assertRaises(ContractValidationError):
            state.require_submission_ready()

    def test_submission_ready_requires_full_benchmark_package_when_gate_is_set(self):
        state = ManuscriptInputState(
            run_id=1,
            deep_insight_id=2,
            formal_experiment=True,
            smoke_test_only=False,
            title="Full Gate",
            method_name="Method",
            claims=[{"claim_text": "x"}],
            citation_seed_paper_ids=["2401.1"],
            result_packet={
                "formal_experiment": True,
                "verdict": "confirmed",
                "hypothesis_iterations": [{"iteration_number": 1}],
                "evidence_tier": "benchmark_plan",
                "quality_gates": {"requires_full_benchmark_package": True},
            },
        )

        with self.assertRaises(ContractValidationError):
            state.require_submission_ready()

    def test_submission_ready_blocks_non_full_paper_claim_route(self):
        state = ManuscriptInputState(
            run_id=1,
            deep_insight_id=2,
            formal_experiment=True,
            smoke_test_only=False,
            title="Workshop Route",
            method_name="Method",
            claims=[{"claim_text": "x"}],
            citation_seed_paper_ids=["2401.1"],
            result_packet={
                "formal_experiment": True,
                "verdict": "confirmed",
                "hypothesis_iterations": [{"iteration_number": 1}],
                "evidence_tier": "benchmark_plan",
                "claim_route": {
                    "route": "workshop",
                    "paper_allowed": False,
                },
            },
        )

        with self.assertRaises(ContractValidationError):
            state.require_submission_ready()

    def test_submission_ready_blocks_benchmark_semantic_warnings(self):
        state = ManuscriptInputState(
            run_id=1,
            deep_insight_id=2,
            formal_experiment=True,
            smoke_test_only=False,
            title="Upper Bound Warning",
            method_name="Method",
            claims=[{"claim_text": "x"}],
            citation_seed_paper_ids=["2401.1"],
            result_packet={
                "formal_experiment": True,
                "verdict": "confirmed",
                "hypothesis_iterations": [{"iteration_number": 1}],
                "evidence_tier": "benchmark_plan",
                "benchmark_semantic_warnings": ["candidate exceeds upper_bound comparator"],
            },
        )

        with self.assertRaises(ContractValidationError):
            state.require_submission_ready()

    def test_submission_ready_blocks_reproduction_only_even_with_benchmark_summary(self):
        state = ManuscriptInputState(
            run_id=1,
            deep_insight_id=2,
            formal_experiment=True,
            smoke_test_only=False,
            title="Pilot",
            method_name="Method",
            claims=[{"claim_text": "x"}],
            citation_seed_paper_ids=["2401.1"],
            result_packet={
                "formal_experiment": True,
                "verdict": "reproduced",
                "hypothesis_iterations": [],
                "benchmark_summary": {
                    "primary_metric": "utility",
                    "per_method": {
                        "baseline": {"utility": 0.7},
                        "cggr": {"utility": 0.8},
                    },
                },
            },
        )

        with self.assertRaises(ContractValidationError):
            state.require_submission_ready()

    def test_submission_ready_allows_confirmed_full_benchmark_package(self):
        state = ManuscriptInputState(
            run_id=1,
            deep_insight_id=2,
            formal_experiment=True,
            smoke_test_only=False,
            title="Pilot",
            method_name="Method",
            claims=[{"claim_text": "x"}],
            citation_seed_paper_ids=["2401.1"],
            result_packet={
                "formal_experiment": True,
                "verdict": "confirmed",
                "hypothesis_iterations": [{"iteration_number": 1}],
                "evidence_tier": "benchmark_plan",
                "full_benchmark_completed": True,
                "artifact_paths": {"artifact_manifest": "/tmp/manifest.json"},
                "quality_gates": {
                    "requires_full_benchmark_package": True,
                    "minimum_seeds": 3,
                },
                "benchmark_summary": {
                    "primary_metric": "utility",
                    "num_seeds": 3,
                    "seed_results": [{"seed": 0}, {"seed": 1}, {"seed": 2}],
                    "per_method": {
                        "baseline": {"utility": 0.7},
                        "cggr": {"utility": 0.8},
                    },
                },
            },
        )

        state.require_submission_ready()


if __name__ == "__main__":
    unittest.main()
