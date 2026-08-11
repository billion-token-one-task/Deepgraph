"""A comparison must be fair before its result can mean anything.

Every fixture here is a real measurement taken on 2026-08-11 on yzy-A100-0001,
running idea 105's own forged train.py against Qwen/Qwen3.5-4B on openai/gsm8k.
Only the generation budget was varied between them.

As forged, the candidate arm was given max_new_tokens=160 and the baseline 48.
Both arms hit their cap on every sample, so neither ever emitted the
"#### <answer>" the prompt required, and exact_match degraded to scraping a
digit out of half-finished reasoning. Re-run with an equal 512 budget the
baseline went 0.0156 -> 0.25 and beat the candidate 8x. Left unguarded the
system would have recorded a CONFIRMED result for a method that is worse than
its own baseline; the only reason it did not is that the model weights failed
to download.
"""

from __future__ import annotations

import unittest

from agents.benchmark_audit import comparison_validity_blockers


AS_FORGED = {
    "per_method": {
        "direct_answer_baseline": {
            "metric_value": 0.015625,
            "num_examples": 64,
            "avg_new_tokens": 48.0,
        },
        "process_guided_candidate": {
            "metric_value": 0.03125,
            "num_examples": 64,
            "avg_new_tokens": 160.0,
        },
    }
}

EQUAL_BUDGET_512 = {
    "per_method": {
        "direct_answer_baseline": {
            "metric_value": 0.25,
            "num_examples": 32,
            "avg_new_tokens": 492.59375,
            "max_new_tokens": 512,
        },
        "process_guided_candidate": {
            "metric_value": 0.03125,
            "num_examples": 32,
            "avg_new_tokens": 512.0,
            "max_new_tokens": 512,
        },
    }
}


class DeclaredBudgetTests(unittest.TestCase):
    def test_unequal_declared_budgets_are_rejected(self):
        summary = {
            "per_method": {
                "baseline": {"avg_new_tokens": 40.0, "max_new_tokens": 48, "num_examples": 64},
                "candidate": {"avg_new_tokens": 120.0, "max_new_tokens": 160, "num_examples": 64},
            }
        }
        blockers = comparison_validity_blockers(summary)
        self.assertTrue(any("unequal generation budgets" in b for b in blockers), blockers)

    def test_equal_budgets_with_room_to_finish_are_accepted(self):
        summary = {
            "per_method": {
                "baseline": {"avg_new_tokens": 492.59375, "max_new_tokens": 512, "num_examples": 32},
                "candidate": {"avg_new_tokens": 301.5, "max_new_tokens": 512, "num_examples": 32},
            }
        }
        self.assertEqual(comparison_validity_blockers(summary), [])

    def test_an_arm_pinned_at_its_declared_budget_is_rejected(self):
        blockers = comparison_validity_blockers(EQUAL_BUDGET_512)
        self.assertTrue(any("cut off" in b for b in blockers), blockers)
        # The baseline finished inside its budget; only the candidate is named.
        self.assertTrue(all("direct_answer_baseline" not in b for b in blockers), blockers)


class UndeclaredBudgetTests(unittest.TestCase):
    """The forged script predates the reporting requirement."""

    def test_the_real_forged_run_is_rejected(self):
        blockers = comparison_validity_blockers(AS_FORGED)
        self.assertEqual(len(blockers), 2, blockers)
        self.assertTrue(all("stopped at the same cap" in b for b in blockers), blockers)

    def test_a_natural_length_distribution_is_accepted(self):
        summary = {
            "per_method": {
                "baseline": {"avg_new_tokens": 492.59375, "num_examples": 32},
                "candidate": {"avg_new_tokens": 301.53125, "num_examples": 32},
            }
        }
        self.assertEqual(comparison_validity_blockers(summary), [])

    def test_a_flat_average_over_too_few_samples_is_not_evidence(self):
        summary = {
            "per_method": {
                "baseline": {"avg_new_tokens": 48.0, "num_examples": 4},
                "candidate": {"avg_new_tokens": 48.0, "num_examples": 4},
            }
        }
        self.assertEqual(comparison_validity_blockers(summary), [])


class ScopeTests(unittest.TestCase):
    def test_a_single_arm_run_has_nothing_to_compare(self):
        summary = {"per_method": {"baseline": {"avg_new_tokens": 48.0, "num_examples": 64}}}
        self.assertEqual(comparison_validity_blockers(summary), [])

    def test_missing_or_empty_summaries_are_left_to_other_gates(self):
        self.assertEqual(comparison_validity_blockers(None), [])
        self.assertEqual(comparison_validity_blockers({}), [])


class VerdictWiringTests(unittest.TestCase):
    """An invalid comparison carries no direction, refutation included."""

    def _verdict(self, summary, **kwargs):
        from agents.validation_loop import _determine_final_verdict

        params = dict(
            baseline=0.015625,
            best_value=0.03125,
            direction="higher",
            criteria={},
            total_iters=5,
            total_kept=1,
            refute_min=3,
            benchmark_summary=summary,
        )
        params.update(kwargs)
        return _determine_final_verdict(**params)

    def test_the_as_forged_run_cannot_be_confirmed(self):
        self.assertEqual(self._verdict(AS_FORGED), "inconclusive")

    def test_it_is_not_downgraded_to_refuted_either(self):
        """Refutation is a claim too, and this measurement cannot support one."""

        verdict = self._verdict(AS_FORGED, best_value=0.0, baseline=0.25)
        self.assertEqual(verdict, "inconclusive")

    def test_a_clean_comparison_still_reaches_the_normal_path(self):
        clean = {
            "per_method": {
                "baseline": {"avg_new_tokens": 492.59375, "max_new_tokens": 512, "num_examples": 32},
                "candidate": {"avg_new_tokens": 301.5, "max_new_tokens": 512, "num_examples": 32},
            }
        }
        self.assertNotEqual(self._verdict(clean, total_iters=5, best_value=0.0, baseline=0.25), "inconclusive")


if __name__ == "__main__":
    unittest.main()
