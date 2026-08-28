"""No public surface may name the hardware or the vendor that ran a flight.

The rule, decided 2026-08-28: pages served from the public site stay honest and
checkable about the SCIENCE -- the pre-registration, the frozen holdout, the
permutation test, the independent evaluator, the hashes -- and say nothing
about the infrastructure underneath, which is evidence for nothing and is
nobody's business.

This has already gone wrong twice in one afternoon, in opposite directions:

  * resource_grants.gpu_class was rendered as though it said where a run
    executed. It says what the grant REQUESTED, so three flights that ran
    elsewhere came out labelled with a card they never touched -- inaccurate
    AND a disclosure.
  * Replacing it with the real compute account names fixed the accuracy and
    made the disclosure worse.

So the gate is a scan, not a review habit. It reads the templates and the
static assets that reach a browser, and it reads the rendered HTML of every
public page, because a name can arrive from a ledger value at render time as
easily as from a literal in a template -- which is exactly how
`colab_terminal_handoff_v1` reached the page through an actor column.

What is NOT hidden, and must not be: `retrospective_review:operator` means a
human entered that rung by hand. That is the most important caveat a reader can
have about a row, it is not infrastructure, and the display layer says
"operator" in both languages.
"""

from __future__ import annotations

import pathlib
import re
import unittest

from web import judge_demo_routes as judge

REPO = pathlib.Path(__file__).resolve().parents[1]

# Vendors, clouds, accelerator families and instance shapes. Matched
# case-insensitively as whole-ish words so that ordinary English survives:
# "draws" contains "aws" and "attribute" contains "t4" only if you match
# substrings, which is how the first version of this scan produced a false
# positive on the roadmap paragraph.
FORBIDDEN = re.compile(
    r"(?<![A-Za-z0-9])("
    r"colab|aws|amazon\s+web|gcp|google\s+cloud|azure|runpod|vast\.ai|lambda\s+labs"
    r"|nvidia|a10g|a100|h100|v100|rtx|tesla\s+[tv]\d"
    r"|g5\.[a-z0-9]+|p[34]\.[a-z0-9]+|gpu_class"
    r")(?![A-Za-z0-9])",
    re.IGNORECASE,
)

# Files that reach a browser on a public page. app.js and style.css are the
# dashboard's own and are covered because the Evidence tab is where this
# drill-down lands.
PUBLIC_ASSETS = [
    "web/templates/index.html",
    "web/templates/judge_demo.html",
    "web/templates/judge_preview.html",
    "web/static/js/evidence-ladder.js",
    "web/static/js/app.js",
    "web/static/js/i18n.js",
    "web/static/css/evidence-ladder.css",
]


def _offending_lines(text: str) -> list[str]:
    hits = []
    for number, line in enumerate(text.splitlines(), start=1):
        # A comment explaining WHY a name is banned has to be able to say the
        # name. Only what a reader can see is in scope.
        stripped = line.strip()
        if stripped.startswith(("*", "//", "/*", "#", "<!--")):
            continue
        match = FORBIDDEN.search(line)
        if match:
            hits.append(f"line {number}: {match.group(0)!r} in {stripped[:120]}")
    return hits


class PublicAssetScanTests(unittest.TestCase):
    def test_no_public_asset_names_a_vendor_or_a_card(self):
        for relative in PUBLIC_ASSETS:
            path = REPO / relative
            if not path.exists():
                continue
            with self.subTest(asset=relative):
                self.assertEqual(
                    _offending_lines(path.read_text(encoding="utf-8")), [],
                    f"{relative} names infrastructure on a public surface",
                )


class ActorLabelTests(unittest.TestCase):
    def test_a_vendor_bearing_actor_is_shown_as_a_stage(self):
        zh, en = judge.actor_labels("colab_terminal_handoff_v1")
        self.assertNotIn("colab", (zh + en).lower())
        self.assertTrue(zh and en)

    def test_an_unknown_vendor_actor_is_caught_by_the_token_list(self):
        """The map is a courtesy; the token list is the guarantee.

        A new backend lands as a new actor name long before anyone remembers to
        teach ACTOR_LABELS about it, and that gap is exactly when it would
        reach the page.
        """
        for unseen in ("aws_batch_handoff_v2", "runpod_terminal_handoff_v1",
                       "NVIDIA_dgx_handoff"):
            zh, en = judge.actor_labels(unseen)
            self.assertEqual(_offending_lines(zh + " " + en), [],
                             f"{unseen} leaked through actor_labels")

    def test_an_operator_entered_rung_still_says_operator(self):
        """Not infrastructure. The single most important caveat on a row."""
        zh, en = judge.actor_labels("retrospective_review:operator")
        self.assertIn("人工", zh)
        self.assertIn("perator", en)

    def test_an_unrecognised_neutral_actor_passes_through_unchanged(self):
        # Blanking an actor nobody taught the map about would hide provenance
        # to no purpose; only vendor identity is removed.
        self.assertEqual(judge.actor_labels("evidence_audit_v2"),
                         ("evidence_audit_v2", "evidence_audit_v2"))


class VerdictPhraseTests(unittest.TestCase):
    def test_the_phrasing_does_not_hardcode_one_metric(self):
        """"Accuracy improved" is wrong the first time a run measures something else.

        The benchmark contract pins whichever metric the candidate registered
        against. What a verdict says is whether the pre-registered predicted
        effect was met, and the sentence has to survive a run whose metric is
        latency, calibration or cost.
        """
        banned = ("accuracy", "准确率", "正确率", "score", "分数")
        for verdict, (zh, en) in judge.VERDICT_PHRASE.items():
            for word in banned:
                self.assertNotIn(word, zh.lower(), f"{verdict} zh phrasing")
                self.assertNotIn(word, en.lower(), f"{verdict} en phrasing")

    def test_every_verdict_the_ledger_writes_has_a_sentence(self):
        for verdict in ("supported", "refuted", "inconclusive", "invalid"):
            zh, en = judge.VERDICT_PHRASE[verdict]
            self.assertTrue(zh.strip() and en.strip())
            self.assertNotEqual(zh, en, f"{verdict} is untranslated")


class LadderPayloadTests(unittest.TestCase):
    """The JSON the drill-down consumes carries evidence and nothing else."""

    def setUp(self):
        from web.app import app

        self.app = app
        self._exhibit = judge._exhibit
        self._one = judge._one

    def tearDown(self):
        judge._exhibit = self._exhibit
        judge._one = self._one

    def _payload(self):
        judge._exhibit = lambda run_id: {
            "run": {"id": run_id, "deep_insight_id": 229, "agenda_id": 14,
                    "baseline_metric_name": "numeric_accuracy",
                    "baseline_metric_value": 0.32, "best_metric_value": 0.625,
                    "effect_pct": 95.3125},
            "idea": {"id": 229},
            "title": "C04 complexity_cot_3shot",
            "operator_frozen": True,
            "model_version": judge.OPERATOR_FROZEN_MODEL_VERSION,
            "grants": [{"id": 410, "stage": "evidence_audit", "token_cap": 40000,
                        "max_gpu_hours": 2.0, "status": "consumed"}],
            "audit": {"holdout_ref": "tasksource/bigbench:abc:test[200:400]",
                      "holdout_hash": "7404f3", "evaluator_ref": "vendor:model",
                      "evaluator_hash": "993984", "raw_artifacts_hash": "4a90a8",
                      "claim_ledger_hash": "9f91b6",
                      "benchmark_contract_hash": "debcae"},
            "decision": {"id": 101, "verdict_hash": "e380d5", "created_at": "x"},
            "verdict": "supported",
            "p_value": 0.000999, "alpha": 0.05,
            "metric_value": 0.625, "baseline_value": 0.32,
            "blockers": [], "significant": True,
            "outcome": {"actual_tokens": 5908, "actual_gpu_hours": 0.4345,
                        "wall_seconds": 1564.25},
            "ladder": [{"state": "planned", "label_zh": "预注册", "label_en": "Pre-registration",
                        "why_zh": "x", "why_en": "y", "reached": True,
                        "actor_zh": "预注册", "actor_en": "Pre-registration",
                        "operator_entered": False, "at": "2026-08-28 12:53:18"}],
            "complete": True,
        }
        judge._one = lambda sql, params=(): {
            "id": 229, "title": "internal filing name",
            "proposed_method": '{"one_line": "Use long, distractor-rich exemplars."}',
            "evidence_summary": "", "model_version": judge.OPERATOR_FROZEN_MODEL_VERSION,
        }
        with self.app.test_client() as client:
            response = client.get("/api/v1/judge/ladder/274")
        self.assertEqual(response.status_code, 200)
        return response.get_json()

    def test_the_headline_is_the_written_sentence_not_the_filing_name(self):
        payload = self._payload()
        self.assertEqual(payload["headline"], "Use long, distractor-rich exemplars.")
        self.assertNotIn("internal filing name", str(payload))

    def test_the_verdict_travels_with_a_sentence_in_both_languages(self):
        payload = self._payload()
        self.assertEqual(payload["verdict"], "supported")
        self.assertTrue(payload["verdict_phrase"]["zh"])
        self.assertTrue(payload["verdict_phrase"]["en"])

    def test_the_evidence_a_reader_would_check_is_all_there(self):
        payload = self._payload()
        for key in ("holdout_ref", "holdout_hash", "evaluator_ref", "evaluator_hash",
                    "raw_artifacts_hash", "claim_ledger_hash", "benchmark_contract_hash"):
            self.assertTrue(payload["audit"][key], f"audit.{key} missing")
        self.assertIsNotNone(payload["statistics"]["p_value"])
        self.assertTrue(payload["decision"]["verdict_hash"])

    def test_the_payload_carries_no_hardware_and_no_raw_actor(self):
        payload = self._payload()
        text = str(payload)
        self.assertNotIn("gpu_class", text)
        # The raw actor column never leaves the process: a field that exists is
        # a field the next person renders.
        self.assertNotIn('"actor"', text)
        self.assertEqual(_offending_lines(text), [])

    def test_a_run_the_ledger_does_not_have_is_a_404_not_an_empty_shell(self):
        judge._exhibit = lambda run_id: None
        with self.app.test_client() as client:
            self.assertEqual(client.get("/api/v1/judge/ladder/999999").status_code, 404)


if __name__ == "__main__":
    unittest.main()
