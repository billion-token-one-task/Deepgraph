"""The contract review a proposal must pass before it is allowed to be stored.

The behaviour under test is the one the 2026-08-25..27 batch needed and did not
have: report *every* reason a generated plan cannot run, in one pass, in words
the generator can act on, using only facts read from the live runner registry.
"""

from __future__ import annotations

import copy
import unittest

from agents.candidate_contract import (
    RepositoryResolver,
    render_violations,
    review_candidate_plan,
)
from meta_harness.runner_capability import (
    ExperimentRequirements,
    PreflightEnvironment,
    RepositoryMetadata,
    RunnerCapability,
    RunnerRegistry,
)


ENVIRONMENT = PreflightEnvironment(
    enabled_backends=("cpu", "ssh_gpu", "colab_gpu"),
    backend_vram_gb={"cpu": 0.0, "ssh_gpu": 22.0, "colab_gpu": 14.56},
    network_available=True,
    disk_free_gb=100.0,
)


def executable_plan() -> dict:
    """A plan that satisfies every judgement preflight makes."""
    return {
        "datasets": [{"name": "openai/gsm8k", "split": "test", "why": "numeric answers"}],
        "baselines": [{"name": "unmodified input", "model": "Qwen/Qwen3-8B"}],
        "metrics": {"primary": "numeric_accuracy on the held-out split"},
        "execution_requirements": {
            "schema_version": "experiment_requirements_v1",
            "task_protocol": "generative_qa",
            "candidate_hook": "candidate_prompt",
            "dataset": {
                "repository_id": "openai/gsm8k",
                "revision": "main",
                "config": "main",
                "split": "test",
                "field_mapping": {"prompt": "question", "target": "answer"},
            },
            "model": {
                "repository_id": "Qwen/Qwen3-8B",
                "revision": "main",
                "framework": "transformers",
                "task": "causal_lm",
                "min_vram_gb": 16,
            },
            "metric": {"name": "numeric_accuracy", "direction": "higher"},
            "seeds": [0],
            "sample_cap": 200,
            "artifact_contract": [
                "final_results",
                "raw_predictions",
                "environment_manifest",
            ],
            "preferred_backends": ["ssh_gpu", "colab_gpu"],
        },
    }


class StubProbe:
    """A hub that answers, without a network."""

    def __init__(self, *, datasets=None, models=None, canonical=None):
        self.datasets = datasets or {}
        self.models = models or {}
        self.canonical = canonical or {}
        self.calls: list[tuple] = []

    def dataset(self, repository_id, revision, config):
        self.calls.append(("dataset", repository_id, revision, config))
        return self.datasets.get(
            repository_id,
            RepositoryMetadata(
                True,
                resolved_revision="sha",
                fields=("question", "answer", "text", "label"),
            ),
        )

    def model(self, repository_id, revision):
        self.calls.append(("model", repository_id, revision))
        return self.models.get(
            repository_id,
            RepositoryMetadata(
                True, resolved_revision="sha", task="causal_lm", size_gb=1.0
            ),
        )

    def dependency_available(self, name):
        return True


class StubResolver(RepositoryResolver):
    def __init__(self, probe: StubProbe):
        super().__init__(probe=probe)
        self._stub = probe

    def _ask_canonical(self, kind, repository_id):
        return self._stub.canonical.get((kind, repository_id), repository_id)


def stub(**kwargs) -> StubResolver:
    return StubResolver(StubProbe(**kwargs))


class ParseAndValidateAreSeparable(unittest.TestCase):
    """The refactor that makes multi-error reporting possible changes nothing."""

    def test_parsing_then_validating_equals_from_dict(self):
        block = executable_plan()["execution_requirements"]
        parsed = ExperimentRequirements.parse_unvalidated(block)
        parsed.validate()
        self.assertEqual(parsed, ExperimentRequirements.from_dict(block))

    def test_parsing_alone_refuses_nothing(self):
        block = dict(executable_plan()["execution_requirements"])
        block["metric"] = {"name": "", "direction": "sideways"}
        # No exception: judging is the caller's job now.
        ExperimentRequirements.parse_unvalidated(block)
        with self.assertRaises(Exception):
            ExperimentRequirements.from_dict(block)


class AnExecutablePlanPasses(unittest.TestCase):
    def test_no_violations_offline(self):
        review = review_candidate_plan(executable_plan(), check_remote=False)
        self.assertEqual(review.violations, ())
        self.assertTrue(review.ok)

    def test_no_violations_with_the_hub_consulted(self):
        review = review_candidate_plan(
            executable_plan(),
            environment=ENVIRONMENT,
            resolver=stub(),
            check_remote=True,
        )
        self.assertEqual(review.codes, ())
        self.assertTrue(review.remote_checked)

    def test_offline_review_asks_the_hub_nothing(self):
        probe = StubProbe()
        review_candidate_plan(
            executable_plan(), resolver=StubResolver(probe), check_remote=False
        )
        self.assertEqual(probe.calls, [])


class EveryDefectIsReportedAtOnce(unittest.TestCase):
    """One defect per attempt cannot repair a plan that carries four."""

    def setUp(self):
        plan = executable_plan()
        requirements = plan["execution_requirements"]
        requirements["metric"]["name"] = "spectral_wasserstein_distance"
        requirements["dataset"]["field_mapping"] = {"question": "question"}
        requirements["preferred_backends"] = ["cpu"]
        plan["datasets"] = [{"name": "GSM8K (main config)", "split": "test"}]
        self.review = review_candidate_plan(plan, check_remote=False)

    def test_reports_all_four(self):
        self.assertEqual(
            set(self.review.codes),
            {
                "metric_contract_unsupported",
                "dataset_schema_role_mismatch",
                "backend_contract_mismatch",
                "execution_dataset_identity_unbound",
            },
        )

    def test_each_message_says_what_to_write_instead(self):
        detail = {item.code: item.detail for item in self.review.violations}
        self.assertIn("numeric_accuracy", detail["metric_contract_unsupported"])
        self.assertIn("prompt", detail["dataset_schema_role_mismatch"])
        self.assertIn("target", detail["dataset_schema_role_mismatch"])
        self.assertIn("colab_gpu", detail["backend_contract_mismatch"])
        self.assertIn("openai/gsm8k", detail["execution_dataset_identity_unbound"])

    def test_the_prompt_block_carries_every_item(self):
        rendered = render_violations(self.review)
        for code in self.review.codes:
            self.assertIn(code, rendered)


class AdviceComesFromTheRegistryNotFromThisFile(unittest.TestCase):
    """A runner added tomorrow changes the advice with no edit here."""

    def test_metric_advice_names_the_registered_vocabulary(self):
        capability = RunnerCapability(
            adapter_id="stub_v1",
            version="1",
            task_protocols=("generative_qa",),
            candidate_hooks=("candidate_prompt",),
            dataset_roles=("prompt", "target"),
            model_frameworks=("transformers",),
            model_tasks=("causal_lm",),
            metric_names=("brier_score",),
            dependencies=(),
            can_install_dependencies=False,
            network_required=True,
            min_disk_gb=1.0,
            min_vram_gb=0.0,
            supports_seed=True,
            supports_sample_cap=True,
            output_artifacts=(
                "final_results",
                "raw_predictions",
                "environment_manifest",
            ),
            backends=("cpu",),
        )
        review = review_candidate_plan(
            executable_plan(),
            registry=RunnerRegistry([capability]),
            check_remote=False,
        )
        detail = {item.code: item.detail for item in review.violations}
        self.assertIn("metric_contract_unsupported", detail)
        self.assertIn("brier_score", detail["metric_contract_unsupported"])


class LegacyRepositoryIdsAreResolvedNotRefused(unittest.TestCase):
    """``climate_fever`` is the dataset's old real name, not a hallucination."""

    def setUp(self):
        self.plan = executable_plan()
        self.plan["execution_requirements"]["dataset"]["repository_id"] = "gsm8k"
        self.plan["datasets"] = [{"name": "gsm8k", "split": "test"}]
        self.resolver = stub(canonical={("dataset", "gsm8k"): "openai/gsm8k"})

    def test_the_plan_passes_after_resolution(self):
        review = review_candidate_plan(
            self.plan, resolver=self.resolver, check_remote=False
        )
        self.assertEqual(review.codes, ())

    def test_the_prose_moves_with_the_contract(self):
        review = review_candidate_plan(
            self.plan, resolver=self.resolver, check_remote=False
        )
        self.assertEqual(
            review.plan["execution_requirements"]["dataset"]["repository_id"],
            "openai/gsm8k",
        )
        self.assertEqual(review.plan["datasets"][0]["name"], "openai/gsm8k")

    def test_the_rewrite_is_reported_to_the_generator(self):
        review = review_candidate_plan(
            self.plan, resolver=self.resolver, check_remote=False
        )
        self.assertTrue(any("openai/gsm8k" in note for note in review.normalizations))

    def test_a_name_the_hub_does_not_know_is_still_refused(self):
        plan = copy.deepcopy(self.plan)
        plan["execution_requirements"]["dataset"]["repository_id"] = "not_a_dataset"
        plan["datasets"] = [{"name": "not_a_dataset"}]
        review = review_candidate_plan(plan, resolver=stub(), check_remote=False)
        self.assertIn("dataset_repository_id_malformed", review.codes)


class TheHubBeingDownIsNotTheProposalsFault(unittest.TestCase):
    def test_resolution_failure_is_swallowed_and_the_plan_survives(self):
        resolver = RepositoryResolver(probe=StubProbe())
        # The real _ask_canonical catches everything; prove the contract by
        # making the transport itself fail.
        import urllib.request

        original = urllib.request.urlopen

        def explode(*args, **kwargs):
            raise OSError("network unreachable")

        urllib.request.urlopen = explode
        try:
            self.assertEqual(resolver.canonical_id("dataset", "gsm8k"), "gsm8k")
        finally:
            urllib.request.urlopen = original

    def test_a_plan_needing_no_resolution_is_unaffected(self):
        resolver = RepositoryResolver(probe=StubProbe())
        review = review_candidate_plan(
            executable_plan(), resolver=resolver, check_remote=False
        )
        self.assertEqual(review.codes, ())


class TheHubsVerdictIsPreflightsVerdict(unittest.TestCase):
    """The remote stage is PreflightEngine, not a second copy of it."""

    def test_a_missing_dataset_is_reported(self):
        probe = StubProbe(datasets={"openai/gsm8k": RepositoryMetadata(False)})
        review = review_candidate_plan(
            executable_plan(),
            environment=ENVIRONMENT,
            resolver=StubResolver(probe),
            check_remote=True,
        )
        self.assertIn("dataset_unavailable", review.codes)

    def test_a_checkpoint_without_the_declared_head_is_reported(self):
        probe = StubProbe(
            models={
                "Qwen/Qwen3-8B": RepositoryMetadata(
                    True, resolved_revision="sha", task="fill_mask", size_gb=1.0
                )
            }
        )
        review = review_candidate_plan(
            executable_plan(),
            environment=ENVIRONMENT,
            resolver=StubResolver(probe),
            check_remote=True,
        )
        self.assertIn("model_task_mismatch", review.codes)


class TheAgendaKeywordRuleIsAskedBeforeTheRowExists(unittest.TestCase):
    """Five candidates were re-refused 642 times in a day without being told why."""

    class Agenda:
        reject = {"keywords": ["fine-tuning", "weight update"]}

    def test_the_matched_phrase_is_named(self):
        review = review_candidate_plan(
            executable_plan(),
            agenda=self.Agenda(),
            claim_text="A prompt-only intervention, with no fine-tuning.",
            check_remote=False,
        )
        self.assertIn("topic_gate_agenda_reject_keyword", review.codes)
        detail = next(
            item.detail
            for item in review.violations
            if item.code == "topic_gate_agenda_reject_keyword"
        )
        self.assertIn("fine-tuning", detail)

    def test_a_clean_claim_is_not_flagged(self):
        review = review_candidate_plan(
            executable_plan(),
            agenda=self.Agenda(),
            claim_text="A prompt-only intervention with frozen weights.",
            check_remote=False,
        )
        self.assertEqual(review.codes, ())

    def test_the_plan_prose_is_not_searched(self):
        """The pre-check must look exactly where the gate looks, or it refuses
        plans the gate would admit -- a risk register that disclaims weight
        updates is not a claim to perform them."""
        plan = executable_plan()
        plan["risks"] = [{"risk": "may need fine-tuning to converge"}]
        review = review_candidate_plan(
            plan,
            agenda=self.Agenda(),
            claim_text="A prompt-only intervention with frozen weights.",
            check_remote=False,
        )
        self.assertEqual(review.codes, ())

    def test_no_agenda_means_no_opinion(self):
        review = review_candidate_plan(
            executable_plan(),
            claim_text="We do no fine-tuning.",
            check_remote=False,
        )
        self.assertEqual(review.codes, ())


class DeploymentWeatherIsNotAPlanDefect(unittest.TestCase):
    """Regenerating cannot fix a backend that is down; do not spend an attempt."""

    def test_an_environment_condition_is_reported_but_not_actionable(self):
        offline = PreflightEnvironment(
            enabled_backends=("cpu",),
            backend_vram_gb={"cpu": 0.0},
            network_available=True,
            disk_free_gb=100.0,
        )
        review = review_candidate_plan(
            executable_plan(),
            environment=offline,
            resolver=stub(),
            check_remote=True,
        )
        self.assertIn("backend_unavailable", review.codes)
        self.assertEqual(review.actionable, ())

    def test_a_plan_defect_stays_actionable(self):
        plan = executable_plan()
        plan["execution_requirements"]["metric"]["name"] = "brier_score"
        plan["metrics"]["primary"] = "brier_score on the held-out split"
        review = review_candidate_plan(plan, check_remote=False)
        self.assertEqual(
            [item.code for item in review.actionable],
            ["metric_contract_unsupported"],
        )


class AContractIsNotOptional(unittest.TestCase):
    def test_a_plan_without_requirements_is_refused(self):
        review = review_candidate_plan({"datasets": []}, check_remote=False)
        self.assertEqual(review.codes, ("candidate_execution_requirements_missing",))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()


class AnAbandonedCandidateGivesBackItsSlot(unittest.TestCase):
    """A refused candidate must not pin the agenda's only concurrency slot.

    Nothing settles a proposal grant except a stored insight, so the contract
    loop's own refusal path has to hand the grant back or it recreates the
    2026-08-17 stall it was written to prevent.
    """

    def test_the_grant_is_expired_with_a_reason(self):
        from unittest import mock

        from agents import paper_idea_agent

        repository = mock.Mock()
        repository.expire_grant_now.return_value = True
        with mock.patch(
            "meta_harness.repository.MetaHarnessRepository", return_value=repository
        ):
            released = paper_idea_agent._release_abandoned_proposal_grant(
                {"id": 91}, 16
            )
        self.assertTrue(released)
        repository.expire_grant_now.assert_called_once_with(
            91, agenda_id=16, reason="proposal_contract_unsatisfied"
        )

    def test_a_failure_to_release_does_not_break_the_pass(self):
        from unittest import mock

        from agents import paper_idea_agent

        with mock.patch(
            "meta_harness.repository.MetaHarnessRepository",
            side_effect=RuntimeError("db down"),
        ):
            self.assertFalse(
                paper_idea_agent._release_abandoned_proposal_grant({"id": 91}, 16)
            )

    def test_the_refusal_path_calls_it(self):
        import inspect

        from agents import paper_idea_agent

        source = inspect.getsource(paper_idea_agent.discover_paper_ideas)
        self.assertIn("_release_abandoned_proposal_grant(proposal_grant", source)
