"""A bare dataset name must be refused before it can spend a grant.

Idea 131 declared its dataset as "gsm8k" rather than "openai/gsm8k". Nothing
checked the shape, so the plan passed preflight, took a pilot grant, shipped
code to an A10G and died eleven seconds later on

    Invalid HF URI 'hf://datasets/gsm8k@<rev>/...'.
    Repository id must be 'namespace/name', got 'gsm8k'.

Three times, across three grants (requests 18, 56, 61), each one a run in the
M2 denominator. Worse, the remote reports it as reason_code
"authentication_required" -- the HF client's unauthenticated-request warning
sits directly above the real error -- so the failure does not even look like
the plan defect it is.

This is the cheapest possible check and it belongs before the spend.
"""

import unittest

from meta_harness.runner_capability import (
    CapabilityContractError,
    DatasetRequirement,
    ExperimentRequirements,
    MetricRequirement,
    ModelRequirement,
    _is_hf_repository_id,
)


def _requirements(dataset_id="openai/gsm8k", model_id="Qwen/Qwen2.5-1.5B-Instruct"):
    return ExperimentRequirements(
        task_protocol="generative_qa",
        dataset=DatasetRequirement(
            repository_id=dataset_id,
            revision="740312add88f781978c0658806c59bc2815b9866",
            config="main",
            split="test",
            field_mapping={"input": "question", "target": "answer"},
        ),
        model=ModelRequirement(
            repository_id=model_id,
            revision="989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        ),
        metric=MetricRequirement(name="numeric_accuracy"),
        candidate_hook="candidate_prompt",
    )


class HfRepositoryIdTests(unittest.TestCase):
    def test_namespaced_ids_are_accepted(self):
        for value in (
            "openai/gsm8k",
            "Qwen/Qwen2.5-1.5B-Instruct",
            "allenai/ai2_arc",
            "EleutherAI/gpt-neo-1.3B",
            "org-name/data.set_v2",
        ):
            self.assertTrue(_is_hf_repository_id(value), value)

    def test_bare_names_are_rejected(self):
        for value in ("gsm8k", "squad", "", "   ", "/gsm8k", "openai/", "a/b/c"):
            self.assertFalse(_is_hf_repository_id(value), repr(value))

    def test_the_exact_plan_that_burned_three_grants_is_refused(self):
        with self.assertRaises(CapabilityContractError) as caught:
            _requirements(dataset_id="gsm8k").validate()
        self.assertEqual(str(caught.exception), "dataset_repository_id_malformed")

    def test_a_bare_model_id_is_refused_too(self):
        with self.assertRaises(CapabilityContractError) as caught:
            _requirements(model_id="Qwen2.5-1.5B-Instruct").validate()
        self.assertEqual(str(caught.exception), "model_repository_id_malformed")

    def test_a_correct_plan_still_validates(self):
        _requirements().validate()


if __name__ == "__main__":
    unittest.main()


class MalformedRepositoryIdClassificationTests(unittest.TestCase):
    """The remote error must not be read as a credential problem.

    The HF client prints "You are sending unauthenticated requests to the HF
    Hub" directly above the real error, and the classifier's authentication
    branch matched on the bare word. authentication_required routes to
    'defer' -- exactly wrong for a defect no credential can fix.
    """

    OBSERVED = (
        "Warning: You are sending unauthenticated requests to the HF Hub. "
        "Please set a HF_TOKEN to enable higher rate limits and faster downloads. "
        'RUNNER_ERROR: {"reason_code": "authentication_required", "detail": '
        "\"authentication_required:Invalid HF URI "
        "'hf://datasets/gsm8k@740312add88f781978c0658806c59bc2815b9866/"
        ".huggingface.yaml'. Repository id must be 'namespace/name', "
        "got 'gsm8k'.\"}"
    )

    def test_request_61s_exact_output_classifies_as_malformed_not_auth(self):
        from meta_harness.failure_policy import classify_failure

        self.assertEqual(
            classify_failure(message=self.OBSERVED, returncode=2),
            "repository_id_malformed",
        )

    def test_a_real_credential_failure_still_classifies_as_auth(self):
        from meta_harness.failure_policy import classify_failure

        self.assertEqual(
            classify_failure(
                message="401 Client Error: Unauthorized for url: https://huggingface.co/api/models/x",
                returncode=1,
            ),
            "authentication_required",
        )

    def test_recovery_repairs_the_plan_instead_of_deferring(self):
        from meta_harness.failure_policy import FailureContext, decide_recovery

        decision = decide_recovery(
            FailureContext(
                reason_code="repository_id_malformed",
                detail="Repository id must be 'namespace/name', got 'gsm8k'.",
                code_hash="c" * 64,
                environment_hash="e" * 64,
                remaining_gpu_seconds=3000.0,
            ),
            fingerprint_seen=False,
        )
        self.assertEqual(decision.action, "repair_plan")
        self.assertFalse(decision.retryable)
