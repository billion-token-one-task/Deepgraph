import json
import tempfile
import unittest
from pathlib import Path

from agents.manuscript_pipeline import build_manuscript_input_state


class ManuscriptPipelineStateTests(unittest.TestCase):
    def test_contribution_states_negative_effect_explicitly(self):
        run = {
            "id": 110,
            "deep_insight_id": 13,
            "baseline_metric_name": "utility",
            "baseline_metric_value": 0.42061197916666665,
            "best_metric_value": 0.3922689335724044,
            "effect_pct": -6.738525529020034,
            "hypothesis_verdict": "inconclusive",
            "proxy_config": json.dumps({"formal_experiment": True, "smoke_test_only": False}),
            "success_criteria": json.dumps({"metric_name": "utility", "metric_direction": "higher"}),
        }
        insight = {
            "id": 13,
            "title": "Counterfactual Gain Gated Reasoning",
            "proposed_method": json.dumps({"name": "CGGR", "one_line": "Route extra reasoning by estimated gain."}),
            "experimental_plan": json.dumps({"datasets": ["MuSiQue-Ans"], "baselines": ["direct"]}),
            "evidence_plan": "{}",
            "mechanism_type": "adaptive_compute",
        }

        state = build_manuscript_input_state(run, insight, [], [])

        self.assertTrue(
            any("remains below baseline" in contribution for contribution in state.contributions),
            state.contributions,
        )
        self.assertFalse(
            any(contribution.startswith("Validated with baseline") for contribution in state.contributions),
            state.contributions,
        )

    def test_result_packet_uses_best_kept_benchmark_summary_not_latest_discard(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            results = workdir / "results"
            packet_dir = results / "iteration_packets"
            packet_dir.mkdir(parents=True)
            latest_summary = {"per_method": {"CGGR": {"metric_value": 0.38}}}
            best_summary = {"per_method": {"CGGR": {"metric_value": 0.52}}}
            (results / "benchmark_summary.json").write_text(json.dumps(latest_summary), encoding="utf-8")
            (packet_dir / "hypothesis_testing_001.json").write_text(
                json.dumps(
                    {
                        "status": "keep",
                        "metric_value": 0.52,
                        "execution_report": {"benchmark_summary": best_summary},
                    }
                ),
                encoding="utf-8",
            )
            (packet_dir / "hypothesis_testing_002.json").write_text(
                json.dumps(
                    {
                        "status": "discard",
                        "metric_value": 0.38,
                        "execution_report": {"benchmark_summary": latest_summary},
                    }
                ),
                encoding="utf-8",
            )
            run = {
                "id": 110,
                "deep_insight_id": 13,
                "workdir": str(workdir),
                "baseline_metric_name": "utility",
                "baseline_metric_value": 0.42,
                "best_metric_value": 0.52,
                "hypothesis_verdict": "confirmed",
                "proxy_config": json.dumps({"formal_experiment": True, "smoke_test_only": False}),
                "success_criteria": json.dumps({"metric_name": "utility", "metric_direction": "higher"}),
            }
            insight = {
                "id": 13,
                "title": "Counterfactual Gain Gated Reasoning",
                "proposed_method": json.dumps({"name": "CGGR", "one_line": "Route extra reasoning by estimated gain."}),
                "experimental_plan": json.dumps({"datasets": ["MuSiQue-Ans"], "baselines": ["direct"]}),
                "evidence_plan": "{}",
                "mechanism_type": "adaptive_compute",
            }

            state = build_manuscript_input_state(run, insight, [], [])

        self.assertEqual(
            state.result_packet["benchmark_summary"]["per_method"]["CGGR"]["metric_value"],
            0.52,
        )


if __name__ == "__main__":
    unittest.main()
