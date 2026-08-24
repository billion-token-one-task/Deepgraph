"""No new gate may wait on a flag that nothing writes.

Four defects on 2026-08-20 shared exactly one shape, and each cost real runs
before anyone noticed:

    evidence_decision_passed   the ladder refused every `supported` verdict --
                               V1's success state was unreachable in code
    reviewer_approval          manuscript_allowed verified an approval that no
                               code path minted; 0 runs ever reached it
    harness_materialized       agenda 14 lost 15 consecutive runs to a
                               readiness flag nothing sets for a v1 plan
    empirical_posterior        four signal tables, 100% NULL, because the only
                               writer copied the value from itself

All four were found by hand, days apart, by following a symptom backwards. The
shape is mechanical, so this test finds it forwards: scripts/find_unwritten_flags.py
collects every key read through ``.get("x")`` or ``["x"]`` and every key written
anywhere -- dict literal, keyword argument, subscript assignment, SQL column --
and reports the difference.

The scan is crude by design and its output includes legitimate read-only keys
from provider JSON and third-party APIs. Those are listed below with the reason
each is expected. The test does not care about the count; it fails when a key
appears that nobody has accounted for, which is the only moment the distinction
between "external payload" and "gate with no writer" is cheap to make.

To resolve a failure: decide which it is. If it is an external payload key, add
it here with the source. If it is a flag your code expects someone to set, find
the writer -- or accept that there isn't one.
"""

import unittest
from pathlib import Path

from scripts.find_unwritten_flags import scan

_ROOT = Path(__file__).resolve().parent.parent

# Each entry is a key the scan reports and a human has accounted for.
_ACCOUNTED = {
    # --- third-party API responses (OpenAlex works objects) ---
    "abstract_inverted_index": "OpenAlex works field",
    "authorships": "OpenAlex works field",
    "cited_by_count": "OpenAlex works field",
    "primary_location": "OpenAlex works field",
    "publication_year": "OpenAlex works field",
    "https://api.openalex.org/works": "OpenAlex endpoint literal",
    "cardData": "HuggingFace Hub model/dataset card payload",
    # --- LLM response payloads, written by the model not by us ---
    "concur": "evaluator/reviewer JSON verdict field",
    "pseudocode": "LLM-authored plan field",
    "hyperparameters": "LLM-authored plan field",
    "falsification_hook": "LLM-authored idea field",
    "mechanism_repair": "LLM-authored idea field",
    "phenomenon": "LLM-authored paradigm field",
    "per_dataset_table": "LLM-authored audit field",
    "route_rate_sweep": "LLM-authored audit field",
    "subset_analysis": "LLM-authored audit field",
    "avg_new_tokens": "runner-emitted metrics payload",
    "publication_evidence": "LLM-authored contract field",
    "submission_target": "LLM-authored venue field",
    "dataset_selection_source": "LLM-authored dataset field",
    # --- environment and SQL aliases ---
    "TEMP": "environment variable",
    "DEEPGRAPH_DATABASE_URL": "environment variable",
    "grant_live": "SQL boolean alias (expires_at > CURRENT_TIMESTAMP)",
    "insight_status": "deep_insights.status SQL alias",
    "open_count": "aggregate SQL alias for open reservations",
    "auto_experiment_run_id": "SQL column alias",
    "patch_agenda_id": "SQL column alias",
    "max_vram_gb": "preflight environment probe payload",
    # --- the benchmark readiness flags: KNOWN to have no writer ---
    # This is the defect that cost agenda 14 fifteen runs. They stay listed
    # because benchmark_harness_loop still reads them for the legacy path;
    # a capability-bound plan no longer depends on any of them being set
    # (see tests/test_pinned_dataset_needs_no_harness_declaration.py).
    "harness_materialized": "legacy harness readiness flag; no writer, known",
    "dataset_cache_verified": "legacy harness readiness flag; no writer, known",
    "benchmark_harness_ready": "legacy harness readiness flag; no writer, known",
}


class NoNewUnwrittenFlagsTests(unittest.TestCase):
    def test_every_reported_key_is_accounted_for(self):
        reported = {key for key, _count, _sites in scan(_ROOT)}
        unexplained = sorted(reported - set(_ACCOUNTED))
        self.assertEqual(
            unexplained,
            [],
            "keys are read but nothing writes them, and no reason is recorded.\n"
            "Decide which kind each is -- an external payload field, or a gate\n"
            "waiting on a writer that does not exist -- then add it to\n"
            "_ACCOUNTED with its source, or wire the writer:\n  "
            + "\n  ".join(unexplained),
        )

    def test_the_accounted_list_does_not_rot(self):
        # An entry that stopped being reported has either gained a writer or
        # lost its reader. Either way the note is now describing something that
        # is not there, which is how a register becomes fiction.
        reported = {key for key, _count, _sites in scan(_ROOT)}
        stale = sorted(set(_ACCOUNTED) - reported)
        self.assertEqual(
            stale,
            [],
            "these are listed as expected but no longer reported; remove them:\n  "
            + "\n  ".join(stale),
        )

    def test_the_scan_still_catches_the_defect_it_was_built_for(self):
        # If the scan ever stops seeing the harness flags, it has been broken
        # or narrowed, and the next flag with no writer will pass silently.
        reported = {key for key, _count, _sites in scan(_ROOT)}
        self.assertIn("harness_materialized", reported)


if __name__ == "__main__":
    unittest.main()
