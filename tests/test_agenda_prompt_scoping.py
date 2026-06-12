"""Agenda direction hard constraint: prompt injection + deterministic scope gate.

Field report that motivated this: an agenda about outlier rejection /
correspondence pruning / SfM produced five Tier 2 ideas about RAG, code
generation and graph expressivity — the signal scan was scoped (PR #41) but
the generation prompts never mentioned the agenda and nothing checked the
output. These tests pin down both fixes:

- with an agenda, every generation prompt carries the direction verbatim plus
  the scope keywords; without one, prompts are built exactly as before;
- generated insights that match none of the agenda's scope terms are dropped
  before storage and counted in dropped_out_of_scope.
"""

from __future__ import annotations

import json
import os
import unittest
from unittest import mock

os.environ["DEEPGRAPH_DATABASE_URL"] = ""  # force SQLite tmpdir; never touch a real DB from the environment

CONSTRAINT_HEADER = "# RESEARCH DIRECTION CONSTRAINT"

DIRECTION_TEXT = "特征对应图上的粗差检测（outlier rejection / correspondence pruning / SfM）"


def _make_agenda():
    from agents.agenda_loader import parse_agenda

    return parse_agenda(
        {
            "version": "v1",
            "name": "outlier-correspondence-sfm",
            "description": DIRECTION_TEXT,
            "focus": ["outlier rejection", "correspondence pruning", "sfm"],
            "prefer": {"keywords": ["robust estimation"]},
        }
    )


def _tier1_signals():
    return {
        "entity_overlaps": [
            {
                "node_a_id": "cv.sfm",
                "node_b_id": "cv.matching",
                "taxonomic_distance": 4,
                "shared_entity_count": 3,
                "overlap_score": 0.42,
                "shared_entity_ids": json.dumps(
                    [{"name": "ransac"}, {"name": "essential matrix"}]
                ),
                "shared_entity_types": json.dumps({"method": 2}),
            }
        ],
        "pattern_matches": [],
        "contradiction_clusters": [],
        "taxonomy_map": [],
    }


def _tier2_signals():
    return {
        "contradiction_clusters": [],
        "performance_plateaus": [],
        "limitation_clusters": [],
        "high_potential_insights": [
            {
                "title": "Matching pipelines disagree on inlier definitions",
                "insight_type": "limitation",
                "hypothesis": "Correspondence filtering thresholds are protocol artifacts",
                "paradigm_score": 7,
            }
        ],
        "mechanism_mismatches": [],
        "protocol_artifacts": [],
        "negative_space_gaps": [],
        "hidden_variable_bridges": [],
        "claim_method_gaps": [],
    }


class ConstraintBlockTests(unittest.TestCase):
    def test_block_carries_direction_and_keywords(self):
        from agents.agenda_relevance import agenda_constraint_block

        block = agenda_constraint_block(_make_agenda())
        self.assertIn(CONSTRAINT_HEADER, block)
        self.assertIn(DIRECTION_TEXT, block)
        self.assertIn("outlier rejection", block)
        self.assertIn("correspondence pruning", block)
        self.assertIn("robust estimation", block)

    def test_match_terms_include_phrases_and_tokens(self):
        from agents.agenda_relevance import agenda_match_terms

        terms = agenda_match_terms(_make_agenda())
        for expected in (
            "outlier rejection",  # verbatim phrase
            "outlier",            # phrase token
            "pruning",
            "sfm",
            "robust estimation",  # prefer.keywords
        ):
            self.assertIn(expected, terms)


class Tier1PromptInjectionTests(unittest.TestCase):
    def test_structure_prompt_without_agenda_is_unchanged(self):
        from agents.paradigm_agent import _build_structure_prompt

        signals = _tier1_signals()
        default_prompt = _build_structure_prompt(signals)
        self.assertEqual(default_prompt, _build_structure_prompt(signals, agenda=None))
        self.assertNotIn(CONSTRAINT_HEADER, default_prompt)

    def test_structure_prompt_with_agenda_appends_constraint(self):
        from agents.paradigm_agent import _build_structure_prompt

        signals = _tier1_signals()
        prompt = _build_structure_prompt(signals, agenda=_make_agenda())
        self.assertIn(CONSTRAINT_HEADER, prompt)
        self.assertIn(DIRECTION_TEXT, prompt)
        self.assertIn("outlier rejection", prompt)
        # Evidence section still present, before the constraint block.
        self.assertLess(prompt.index("CROSS-FIELD EVIDENCE"), prompt.index(CONSTRAINT_HEADER))

    def test_formalization_prompt_injection(self):
        from agents.paradigm_agent import _build_formalization_prompt

        candidate = {"title": "Candidate", "field_a": {}, "field_b": {}}
        signals = _tier1_signals()
        default_prompt = _build_formalization_prompt(candidate, signals)
        self.assertEqual(
            default_prompt, _build_formalization_prompt(candidate, signals, agenda=None)
        )
        self.assertNotIn(CONSTRAINT_HEADER, default_prompt)

        prompt = _build_formalization_prompt(candidate, signals, agenda=_make_agenda())
        self.assertIn(CONSTRAINT_HEADER, prompt)
        self.assertIn(DIRECTION_TEXT, prompt)


class Tier2PromptInjectionTests(unittest.TestCase):
    PROBLEM = {
        "title": "Inlier threshold drift",
        "source_type": "limitation",
        "source_evidence": "3 papers",
        "formal_statement": "Minimize false matches",
        "current_failure_mode": "Fixed thresholds",
        "desideratum": "Adaptive filtering",
        "impact_scope": "matching pipelines",
        "related_node_ids": ["cv.sfm"],
    }
    METHOD = {
        "name": "GraphGate",
        "type": "training_procedure",
        "one_line": "Filters correspondences",
        "definition": "L = ...",
        "key_properties": [],
        "limitations": "none stated",
    }

    def test_problem_prompt_injection(self):
        from agents.paper_idea_agent import _build_problem_prompt

        signals = _tier2_signals()
        default_prompt = _build_problem_prompt(signals)
        self.assertEqual(default_prompt, _build_problem_prompt(signals, agenda=None))
        self.assertNotIn(CONSTRAINT_HEADER, default_prompt)

        prompt = _build_problem_prompt(signals, agenda=_make_agenda())
        self.assertIn(CONSTRAINT_HEADER, prompt)
        self.assertIn(DIRECTION_TEXT, prompt)
        self.assertIn("outlier rejection", prompt)

    def test_method_prompt_injection(self):
        from agents.paper_idea_agent import _build_method_prompt

        default_prompt = _build_method_prompt(self.PROBLEM)
        self.assertEqual(default_prompt, _build_method_prompt(self.PROBLEM, agenda=None))
        self.assertNotIn(CONSTRAINT_HEADER, default_prompt)

        prompt = _build_method_prompt(self.PROBLEM, agenda=_make_agenda())
        self.assertIn(CONSTRAINT_HEADER, prompt)
        self.assertIn(DIRECTION_TEXT, prompt)

    def test_experiment_prompt_injection(self):
        from agents.paper_idea_agent import _build_experiment_prompt

        default_prompt = _build_experiment_prompt(self.PROBLEM, self.METHOD)
        self.assertEqual(
            default_prompt, _build_experiment_prompt(self.PROBLEM, self.METHOD, agenda=None)
        )
        self.assertNotIn(CONSTRAINT_HEADER, default_prompt)

        prompt = _build_experiment_prompt(self.PROBLEM, self.METHOD, agenda=_make_agenda())
        self.assertIn(CONSTRAINT_HEADER, prompt)
        self.assertIn(DIRECTION_TEXT, prompt)


class ScopeGateTests(unittest.TestCase):
    IN_SCOPE = {
        "title": "Adaptive outlier rejection on correspondence graphs",
        "problem_statement": "Minimize wrong matches kept by fixed RANSAC thresholds in SfM",
        "proposed_method": json.dumps({"name": "Consistency-weighted pruning"}),
    }
    OFF_SCOPE = {
        "title": "Retrieval grounding gap in code generation",
        "problem_statement": "Align retriever and generator objectives to cut hallucinated APIs",
        "proposed_method": json.dumps({"name": "Joint retriever-generator objective"}),
    }

    def test_in_scope_insight_passes(self):
        from agents.agenda_relevance import insight_in_scope

        self.assertTrue(insight_in_scope(self.IN_SCOPE, _make_agenda()))

    def test_off_scope_insight_is_rejected(self):
        from agents.agenda_relevance import insight_in_scope

        self.assertFalse(insight_in_scope(self.OFF_SCOPE, _make_agenda()))

    def test_no_agenda_passes_everything(self):
        from agents.agenda_relevance import insight_in_scope

        self.assertTrue(insight_in_scope(self.OFF_SCOPE, None))

    def test_threshold_is_configurable(self):
        from agents.agenda_relevance import insight_in_scope

        # Title-only insight has a couple of hits; a high threshold drops it.
        sparse = {"title": "A note on sfm"}
        self.assertTrue(insight_in_scope(sparse, _make_agenda(), min_hits=1))
        self.assertFalse(insight_in_scope(sparse, _make_agenda(), min_hits=3))
        # Zero disables the gate.
        self.assertTrue(insight_in_scope(self.OFF_SCOPE, _make_agenda(), min_hits=0))

    def test_non_ascii_only_agenda_disables_gate(self):
        from agents.agenda_loader import parse_agenda
        from agents.agenda_relevance import insight_in_scope

        agenda = parse_agenda(
            {
                "version": "v1",
                "name": "chinese-only",
                "description": "粗差检测",
                "focus": ["粗差检测"],
            }
        )
        # No ASCII-matchable term: dropping everything would be worse than
        # dropping nothing, so the gate stands down.
        self.assertTrue(insight_in_scope(self.OFF_SCOPE, agenda))


class Tier2DroppedCountTests(unittest.TestCase):
    def _run(self):
        import agents.paper_idea_agent as pia

        in_scope_problem = {
            "title": "Threshold-free outlier rejection for correspondence graphs",
            "source_type": "limitation",
            "source_evidence": "3 papers report fixed inlier thresholds",
            "formal_statement": "Minimize surviving wrong matches in SfM correspondence pruning",
            "current_failure_mode": "Fixed global thresholds ignore scene geometry",
            "desideratum": "Per-edge adaptive filtering",
            "impact_scope": "matching pipelines",
            "mechanism_type": "protocol_artifact",
            "related_node_ids": ["cv.sfm"],
        }
        off_scope_problem = {
            "title": "Retrieval grounding gap in code synthesis",
            "source_type": "plateau",
            "source_evidence": "5 papers within 1%",
            "formal_statement": "Minimize hallucinated API calls in retrieval-augmented code synthesis",
            "current_failure_mode": "Retriever and generator trained separately",
            "desideratum": "Joint objective",
            "impact_scope": "code assistants",
            "mechanism_type": "plateau",
            "related_node_ids": ["nlp.codegen"],
        }
        method_payload = {
            "method": {
                "name": "EdgeGate",
                "type": "training_procedure",
                "one_line": "Per-edge gating",
                "definition": "g(e) = sigma(w^T f(e))",
                "key_properties": [],
                "why_novel": "Differs from the three closest filters by gating per edge with learned geometry features",
                "limitations": "needs labels",
            }
        }
        exp_in = {"paper_title": "Adaptive correspondence pruning without global thresholds"}
        exp_off = {"paper_title": "Jointly trained retrieval for grounded code synthesis"}

        with (
            mock.patch.object(pia, "get_tier2_signals", return_value=_tier2_signals()),
            mock.patch.object(pia, "agenda_taxonomy_node_ids", return_value=[]),
            mock.patch.object(
                pia,
                "call_llm_json",
                side_effect=[
                    ({"problems": [in_scope_problem, off_scope_problem]}, 10),
                    (method_payload, 10),
                    (exp_in, 10),
                    (method_payload, 10),
                    (exp_off, 10),
                ],
            ) as llm,
            mock.patch.object(pia, "get_evosci_input_issue", return_value=None),
            mock.patch.object(pia, "enrich_deep_insight", side_effect=lambda d: d),
            mock.patch.object(pia, "build_evidence_packet", return_value={}),
        ):
            result = pia.discover_paper_ideas(agenda=_make_agenda())
        return result, llm

    def test_off_scope_idea_is_dropped_and_counted(self):
        result, llm = self._run()
        self.assertEqual(result["dropped_out_of_scope"], 1)
        self.assertEqual(len(result["insights"]), 1)
        self.assertIn("pruning", result["insights"][0]["title"].lower())
        # The generation prompt itself carried the constraint block.
        problem_prompt = llm.call_args_list[0].args[1]
        self.assertIn(CONSTRAINT_HEADER, problem_prompt)
        self.assertIn(DIRECTION_TEXT, problem_prompt)


class Tier1DroppedCountTests(unittest.TestCase):
    def _run(self):
        import agents.paradigm_agent as pa

        in_scope_candidate = {
            "title": "Outlier rejection and correspondence pruning share a consistency structure",
            "confidence": 9,
            "field_a": {},
            "field_b": {},
            "unifying_structure": "graph consistency",
            "shared_failure_mode": "threshold sensitivity",
            "evidence_from_graph": "shared ransac usage",
        }
        off_scope_candidate = {
            "title": "Dialogue state tracking unifies with knowledge base completion",
            "confidence": 5,
            "field_a": {},
            "field_b": {},
            "unifying_structure": "slot filling",
            "shared_failure_mode": "schema drift",
            "evidence_from_graph": "shared ontology entities",
        }
        formal_in = {
            "title": "Consistency-constrained outlier rejection across SfM pipelines",
            "formal_structure": "argmin over correspondence subgraphs",
            "transformation": "map pruning rules to consistency potentials",
        }
        formal_off = {
            "title": "Dialogue state tracking as schema alignment",
            "formal_structure": "argmax over slot assignments",
            "transformation": "map ontology edges to slot values",
        }
        adversarial = {
            "overall_score": 8,
            "verdict": "interesting",
            "attacks": [],
            "strongest_attack": "",
        }

        with (
            mock.patch.object(pa, "get_tier1_signals", return_value=_tier1_signals()),
            mock.patch.object(pa, "agenda_taxonomy_node_ids", return_value=[]),
            mock.patch.object(
                pa,
                "call_llm_json",
                side_effect=[
                    ({"candidates": [in_scope_candidate, off_scope_candidate]}, 10),
                    (formal_in, 10),
                    (formal_off, 10),
                ],
            ) as llm,
            mock.patch.object(pa, "_call_with_provider", return_value=(adversarial, 5)),
            mock.patch.object(pa, "get_evosci_input_issue", return_value=None),
            mock.patch.object(pa, "enrich_deep_insight", side_effect=lambda d: d),
            mock.patch.object(pa, "build_evidence_packet", return_value={}),
        ):
            result = pa.discover_paradigm_insights(agenda=_make_agenda())
        return result, llm

    def test_off_scope_insight_is_dropped_and_counted(self):
        result, llm = self._run()
        self.assertEqual(result["dropped_out_of_scope"], 1)
        self.assertEqual(len(result["insights"]), 1)
        self.assertIn("outlier rejection", result["insights"][0]["title"].lower())
        structure_prompt = llm.call_args_list[0].args[1]
        self.assertIn(CONSTRAINT_HEADER, structure_prompt)
        self.assertIn(DIRECTION_TEXT, structure_prompt)


if __name__ == "__main__":
    unittest.main()
