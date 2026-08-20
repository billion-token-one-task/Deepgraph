"""The designer must be told what the runners can execute, and what already failed.

Between 2026-08-17 and 2026-08-20, nineteen candidates were refused at
capability preflight. Thirteen of them -- and EVERY refusal after 08-18 13:34
-- were `model_task_mismatch`, and every one of those declared a base encoder
for a classification protocol:

    idea 159   sequence_classification   FacebookAI/roberta-base
    idea 150   sequence_classification   roberta-base
    idea 144   sequence_classification   bert-base-uncased

Those declarations sit exactly inside the capability envelope the prompt
already published: the protocol is supported, the declared model task matches,
the metric is supported. Preflight refused them anyway, and it was right to --
the runners have no Trainer, no optimizer and no backward pass, so they can
only measure a checkpoint as published. A base encoder publishes a fill-mask
head; loading it for classification attaches a random head and measures noise.

So the envelope was true and still insufficient: it said which task the model
must declare, never that the model must ALREADY carry that head.

The second gap compounded it. Ideas 136, 144, 150 and 159 each declared the
same unexecutable pairing twice. The reason codes were recorded on the retired
candidate and in its outcome, the problem went back to the pool, and the
redesign was never told what killed the last attempt.
"""

import json
import unittest
from unittest.mock import patch

from agents.paper_idea_agent import _build_experiment_prompt

_PROBLEM = {
    "id": 77,
    "title": "t",
    "formal_statement": "f",
    "current_failure_mode": "m",
    "desideratum": "d",
    "impact_scope": "i",
    "source_evidence": "s",
    "source_type": "paper",
}
_METHOD = {"name": "n", "type": "prompt", "one_line": "o"}


class _Profile:
    """The live configuration: production reports gpu_allowed=True, which is
    the branch that publishes the capability envelope."""

    gpu_allowed = True


class _Env:
    enabled_backends = ("colab_gpu",)
    backend_vram_gb = {"colab_gpu": 24.0}


def _prompt(prior_codes=None, *, fetchall=None):
    rows = [{"reason_codes_json": json.dumps(prior_codes)}] if prior_codes else []
    fetchall = fetchall or (lambda *a, **k: rows)
    with patch(
        "agents.paper_idea_agent.detect_compute_profile", return_value=_Profile()
    ), patch(
        "meta_harness.preflight_repository.runtime_preflight_environment",
        return_value=_Env(),
    ), patch(
        "agents.paper_idea_agent.db.fetchall", side_effect=fetchall
    ):
        return _build_experiment_prompt(_PROBLEM, _METHOD)


class EnvelopeTests(unittest.TestCase):
    def test_the_designer_is_told_the_runners_never_train(self):
        prompt = _prompt()
        self.assertIn("EVALUATE ONLY", prompt)

    def test_the_named_base_encoders_are_called_out(self):
        # These three are the actual repeat offenders, so name them rather
        # than describing the category and hoping it generalises.
        prompt = _prompt()
        for model in ("roberta-base", "bert-base-uncased", "distilbert-base-uncased"):
            with self.subTest(model=model):
                self.assertIn(model, prompt)

    def test_it_explains_what_goes_wrong_not_just_that_it_is_refused(self):
        prompt = _prompt()
        self.assertIn("randomly initialised head", prompt)


class RefusalFeedbackTests(unittest.TestCase):
    def test_prior_refusal_reasons_reach_the_redesign(self):
        prompt = _prompt(["model_task_mismatch"])
        self.assertIn("Earlier plans for THIS problem", prompt)
        self.assertIn("model_task_mismatch", prompt)

    def test_several_reasons_are_all_carried(self):
        prompt = _prompt(["model_task_mismatch", "dataset_schema_role_mismatch"])
        for code in ("model_task_mismatch", "dataset_schema_role_mismatch"):
            with self.subTest(code=code):
                self.assertIn(code, prompt)

    def test_a_clean_problem_gets_no_refusal_section(self):
        # A first attempt must not be told about failures that never happened.
        self.assertNotIn("Earlier plans for THIS problem", _prompt())

    def test_a_database_failure_does_not_break_the_prompt(self):
        # The feedback is an improvement, never a dependency: an unreachable
        # database must not stop an idea from being designed at all.
        def _down(*_a, **_k):
            raise RuntimeError("db down")

        prompt = _prompt(fetchall=_down)
        self.assertIn("PROPOSED RESEARCH", prompt)
        self.assertNotIn("Earlier plans for THIS problem", prompt)
        # And the envelope it does not depend on must survive.
        self.assertIn("EVALUATE ONLY", prompt)


if __name__ == "__main__":
    unittest.main()
