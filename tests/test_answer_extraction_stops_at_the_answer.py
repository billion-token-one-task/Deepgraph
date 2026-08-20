"""A generation that runs past its own answer must be read at the answer.

Few-shot prompting makes a model continue the pattern: it answers the question
correctly, then invents the next question and answers that one too. Taking the
last number in the generation therefore scores the arm on a fabricated
continuation.

Measured on 2026-08-20 with an 8-shot chain-of-thought probe, 200 GSM8K items,
Qwen2.5-1.5B-Instruct:

    continuation present   70% of candidate outputs (baseline: 0%)
    last-number scoring    0.210
    reading its own answer 0.510

Thirty points, and the loss is ONE-SIDED. The unmodified baseline never
continues -- 0% across all fourteen adjudicated runs -- so the penalty falls
only on candidates, and only on those whose prompt style invites continuation.
A metric that silently punishes one arm for its prompt shape is not measuring
the intervention.

The fourteen adjudicated verdicts were re-scored under the corrected rule
before it shipped: only two arms moved by more than two points and no verdict
changed direction, so this fixes a live hazard rather than rewriting history.
"""

import unittest

from meta_harness.runner_contract import _numeric, recompute_metric


class AnswerExtractionTests(unittest.TestCase):
    def test_the_observed_continuation_is_read_at_its_own_answer(self):
        # verbatim shape of candidate sample 1 from the probe
        text = (
            "The robe takes 2 * .5 = 1 bolt of white fiber. "
            "So it takes 2 + 1 = 3 bolts in total.\n"
            "#### 3\n\n"
            "Question: John bought a shirt for $25 and a pair of pants for $35. "
            "He also bought a tie for $10.\n"
            "#### 70"
        )
        self.assertEqual(_numeric(text), 3.0)

    def test_the_marker_wins_over_a_trailing_number(self):
        self.assertEqual(_numeric("blah #### 42 and then 999 more"), 42.0)

    def test_without_a_marker_it_stops_at_the_continuation(self):
        text = "So the answer is 12.\n\nQuestion: something else\nAnswer: 500"
        self.assertEqual(_numeric(text), 12.0)

    def test_plain_answers_are_unchanged(self):
        # the regime production actually runs in: no marker, no continuation
        self.assertEqual(_numeric("He has 5 apples and buys 7 more, so 12."), 12.0)
        self.assertEqual(_numeric("42"), 42.0)
        self.assertEqual(_numeric("no digits here"), None)
        self.assertEqual(_numeric(None), None)

    def test_negative_and_decimal_survive(self):
        self.assertEqual(_numeric("the change is -3.5"), -3.5)
        self.assertEqual(_numeric("#### -7"), -7.0)

    def test_commas_are_still_stripped(self):
        self.assertEqual(_numeric("#### 1,234"), 1234.0)

    def test_a_target_stated_gsm8k_style_reads_its_marker(self):
        target = "Natalia sold 48/2 = 24 clips in May.\n#### 72"
        self.assertEqual(_numeric(target), 72.0)

    def test_metric_scores_the_answer_not_the_continuation(self):
        rows = [
            {
                "prediction": "so it is 3.\n#### 3\n\nQuestion: another\n#### 999",
                "target": "#### 3",
            },
            {"prediction": "the total is 10.\n#### 10", "target": "#### 10"},
        ]
        self.assertEqual(recompute_metric(rows, "numeric_accuracy"), 1.0)


if __name__ == "__main__":
    unittest.main()
