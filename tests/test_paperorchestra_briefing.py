import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agents.paperorchestra.briefing import (
    build_deterministic_outline,
    build_evidence_brief,
    evidence_brief_markdown,
)
from agents.paperorchestra.tracing import PaperGenerationTrace, call_json_traced

# The CRPP-specific manuscript implementation was demoted into the disabled
# plugins.examples.cggr boundary; agents/paperorchestra/full_pipeline.py is now
# a fail-closed entry point that no longer carries these helpers. Importing the
# old path aborted collection of this whole file. Gate the one test that needs
# them behind the same opt-in the runtime uses for that plugin.
from plugins.examples.cggr.full_pipeline import (
    _completed_benchmark_mode,
    _repair_completed_evidence_section,
)

NONPROD_PLUGINS = os.getenv(
    "DEEPGRAPH_ENABLE_NONPROD_EXAMPLE_PLUGINS", ""
).strip().lower() in {"1", "true", "yes"}
requires_nonprod_plugins = unittest.skipUnless(
    NONPROD_PLUGINS,
    "covers demoted plugins.examples.cggr; set "
    "DEEPGRAPH_ENABLE_NONPROD_EXAMPLE_PLUGINS=1 to run",
)


class PaperOrchestraBriefingTests(unittest.TestCase):
    def _state(self):
        return {
            "title": "Training-Free Multi-Agent Reasoning",
            "problem_statement": "How can LLMs spend test-time compute without model training?",
            "existing_weakness": "Static self-consistency wastes tokens on easy items.",
            "method_name": "Budgeted Consensus Routing",
            "method_summary": "Generate diverse candidate answers, verify disagreement, and allocate extra agents only when useful.",
            "contributions": [
                "A training-free multi-agent routing rule.",
                "A utility view that includes answer quality, tokens, and latency.",
            ],
            "baseline_metric_name": "accuracy",
            "baseline_metric_value": 0.58,
            "best_metric_value": 0.66,
            "effect_pct": 13.8,
            "verdict": "confirmed",
            "problem_awareness": {
                "central_question": "Can small multi-agent ensembles improve reasoning under a fixed budget?",
                "motivation": "Training is unavailable on local CPU-only hardware.",
                "method_answer": "Use inference-time disagreement to route budget.",
                "limitation": "Evidence is bounded to the recorded benchmark suite.",
            },
            "paper_intent": {"main_message": "Dynamic consensus preserves most gains with less cost."},
            "result_packet": {
                "benchmark_summary": {
                    "primary_metric": "accuracy",
                    "num_seeds": 5,
                    "datasets": [{"name": "GSM8K", "split": "test", "num_test": 90}],
                    "model": {"id": "Qwen2.5-7B", "hardware": "CPU"},
                    "per_method": {
                        "Direct": {"metric_value": 0.58, "avg_new_tokens": 40},
                        "Always Multi-Agent": {"metric_value": 0.67, "avg_new_tokens": 260},
                        "Budgeted Consensus Routing": {"metric_value": 0.66, "avg_new_tokens": 130},
                    },
                    "ablation_table": [{"ablation": "no_verifier", "metric_value": 0.62}],
                    "latency_tokens_table": [{"method": "Budgeted Consensus Routing", "avg_latency_seconds": 1.4}],
                }
            },
            "claims": [{"id": "c1", "claim_text": "The method improves accuracy with lower token cost.", "verdict": "confirmed"}],
            "quality_gates": {"manuscript_allowed": True},
            "reviewer_report": {"status": "pass"},
        }

    def test_evidence_brief_is_compact_and_keeps_core_tables(self):
        brief = build_evidence_brief(
            self._state(),
            "Related work positioning.",
            [{"iteration_number": 1, "metric_value": 0.66, "status": "keep"}],
            paper_ids=["2401.00001"],
            baseline=0.58,
            metric_name="accuracy",
        )
        rendered = evidence_brief_markdown(brief, max_chars=12000)

        self.assertLess(len(rendered), 12000)
        self.assertEqual(brief["experiment"]["primary_metric"], "accuracy")
        self.assertEqual(len(brief["experiment"]["per_method"]), 3)
        self.assertIn("Budgeted Consensus Routing", rendered)
        self.assertIn("training-free", rendered.lower())

    def test_deterministic_outline_has_required_paperorchestra_keys(self):
        state = self._state()
        brief = build_evidence_brief(state, "", [], paper_ids=["2401.00001"], metric_name="accuracy")
        outline = build_deterministic_outline(state, brief, metric_name="accuracy")

        self.assertIn("intro_related_work_plan", outline)
        self.assertIn("plotting_plan", outline)
        self.assertIn("section_plan", outline)
        self.assertEqual(
            [fig["figure_id"] for fig in outline["plotting_plan"]],
            ["fig_main_results", "fig_ablation_results", "fig_hyperparameter_sweep"],
        )
        self.assertTrue(all(fig["role"] == "experiment_figure_pack" for fig in outline["plotting_plan"]))
        self.assertEqual(
            [fig["chart_type"] for fig in outline["plotting_plan"]],
            ["main_results_bar", "ablation_bar", "hyperparameter_sweep"],
        )
        self.assertTrue(any("multi-agent" in q for q in outline["intro_related_work_plan"]["introduction_strategy"]["search_directions"]))

    @requires_nonprod_plugins
    def test_completed_evidence_mode_repairs_plan_language(self):
        brief = {
            "experiment": {
                "per_method": [{"method": "Certified Residual Policy Packets", "metric_value": 0.77}],
                "ablation_table": [{"ablation": "selector_family_confidence_gate", "metric_value": 0.73}],
            },
            "gate": {
                "quality_gates": {"full_benchmark_completed": True, "requires_ablation_table": True},
                "required_evidence": {"artifacts": ["main_results_table", "ablation_table"]},
            },
        }
        bad_tex = "The benchmark plan does not provide completed benchmark measurements, so results remain hypotheses."

        self.assertTrue(_completed_benchmark_mode(brief))
        self.assertEqual(
            _repair_completed_evidence_section(
                bad_tex,
                fallback="Completed artifact-backed results.",
                section_name="experiments",
                evidence_brief=brief,
            ),
            "Completed artifact-backed results.",
        )

    def test_deterministic_outline_uses_backend_figure_pack_for_backend_matrix(self):
        state = self._state()
        state["result_packet"]["benchmark_summary"]["per_method_backend"] = {
            "Direct": {
                "HF": {"accuracy": 0.60, "std": 0.02},
                "vLLM": {"accuracy": 0.58, "std": 0.03},
            },
            "DPC": {
                "HF": {"accuracy": 0.72, "std": 0.01},
                "vLLM": {"accuracy": 0.70, "std": 0.02},
            },
        }
        state["result_packet"]["benchmark_summary"]["per_dataset_backend"] = {
            "GSM8K": state["result_packet"]["benchmark_summary"]["per_method_backend"],
            "MATH": state["result_packet"]["benchmark_summary"]["per_method_backend"],
            "BBH": state["result_packet"]["benchmark_summary"]["per_method_backend"],
            "MMLU": state["result_packet"]["benchmark_summary"]["per_method_backend"],
        }
        state["result_packet"]["benchmark_summary"]["backends"] = ["HF", "vLLM"]
        state["result_packet"]["benchmark_summary"]["methods"] = ["Direct", "DPC"]
        brief = build_evidence_brief(state, "", [], paper_ids=["2401.00001"], metric_name="accuracy")
        outline = build_deterministic_outline(state, brief, metric_name="accuracy")

        self.assertEqual(
            [fig["figure_id"] for fig in outline["plotting_plan"]],
            ["fig_backend_grouped_bars", "fig_backend_heatmap_single", "fig_backend_rank_lines_1x4"],
        )
        self.assertEqual(
            [fig["chart_type"] for fig in outline["plotting_plan"]],
            ["backend_grouped_bars", "backend_heatmap_single", "backend_rank_lines_1x4"],
        )
        self.assertTrue(all(fig["role"] == "experiment_figure_pack" for fig in outline["plotting_plan"]))

    def test_call_json_traced_uses_fallback_and_writes_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.jsonl"
            trace = PaperGenerationTrace(trace_path)
            with mock.patch(
                "agents.paperorchestra.tracing.call_llm",
                side_effect=[RuntimeError("empty response"), (json.dumps({"ok": True}), 12)],
            ):
                parsed, tokens = call_json_traced(
                    "stage",
                    "system",
                    "large user prompt",
                    trace=trace,
                    fallback_user_prompts=["small user prompt"],
                )

            self.assertEqual(parsed, {"ok": True})
            self.assertEqual(tokens, 12)
            rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(rows[0]["status"], "started")
            self.assertTrue(any(row["status"] == "error" for row in rows))
            self.assertTrue(any(row["status"] == "ok" and row["fallback_tier"] == 2 for row in rows))


if __name__ == "__main__":
    unittest.main()
