import json
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

from agents.benchmark_audit import (
    benchmark_diagnostic_notes,
    benchmark_fairness_warnings_from_diff,
    benchmark_semantic_warnings,
    best_iteration_benchmark_summary,
)
from agents import validation_loop


class ValidationMetricParsingTests(unittest.TestCase):
    def test_parse_metric_from_final_results_line(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "run.log"
            metric_name = "Cost-adjusted utility"
            log_path.write_text(
                "\n".join(
                    [
                        "device: NVIDIA GeForce RTX 5070",
                        "peak_vram_mb: 10368.0",
                        "FINAL_RESULTS: "
                        + json.dumps(
                            {
                                metric_name: 1.400625,
                                "peak_vram_mb": 10368.0,
                                "reserved_vram_gb": 10.0,
                            }
                        ),
                    ]
                ),
                encoding="utf-8",
            )

            value = validation_loop._parse_metric_from_log(log_path, metric_name)

        self.assertAlmostEqual(value, 1.400625)

    def test_run_experiment_prefers_structured_final_results_over_eval_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            code_dir = workdir / "code"
            spec_dir = workdir / "spec"
            code_dir.mkdir()
            spec_dir.mkdir()
            metric_name = "gpu_score"
            (code_dir / "train.py").write_text(
                textwrap.dedent(
                    f"""
                    import json
                    print('peak_vram_mb: 10368.0')
                    print('FINAL_RESULTS: ' + json.dumps({{{metric_name!r}: 1.25, 'peak_vram_mb': 10368.0}}))
                    """
                ).strip(),
                encoding="utf-8",
            )
            (spec_dir / "evaluate.py").write_text(
                "print('metric_value: 0.0')\n",
                encoding="utf-8",
            )

            result = validation_loop._run_experiment(
                workdir,
                code_dir,
                30,
                baseline_command=f'"{sys.executable}" train.py',
                metric_name=metric_name,
            )

        self.assertEqual(result["status"], "ok")
        self.assertAlmostEqual(result["metric"], 1.25)
        self.assertAlmostEqual(result["peak_memory_mb"], 10368.0)

    def test_run_experiment_classifies_missing_final_results_as_crash(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            code_dir = workdir / "code"
            code_dir.mkdir()
            (code_dir / "train.py").write_text(
                "\n".join(
                    [
                        "print('BENCHMARK_STAGE: start')",
                        "print('BENCHMARK_STAGE: model_ready Qwen/Qwen2.5-7B-Instruct')",
                    ]
                ),
                encoding="utf-8",
            )

            result = validation_loop._run_experiment(
                workdir,
                code_dir,
                30,
                baseline_command=f'"{sys.executable}" train.py',
                metric_name="metric",
            )

        self.assertEqual(result["status"], "crash")
        self.assertEqual(result["failure_type"], "missing_final_results")
        self.assertIn("model_ready", result["last_benchmark_stage"])

    def test_validation_benchmark_env_preserves_paper_grade_contract_budget(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            spec_dir = workdir / "spec"
            spec_dir.mkdir()
            (spec_dir / "proxy_config.json").write_text(
                json.dumps(
                    {
                        "benchmark_max_examples_per_seed": 128,
                        "benchmark_seeds": 5,
                        "benchmark_manifest": {
                            "full_benchmark_stage": {"models": ["Qwen/Qwen2.5-7B-Instruct"]}
                        },
                    }
                ),
                encoding="utf-8",
            )

            env = validation_loop._benchmark_env_for_execution(workdir)

        self.assertEqual(env["DEEPGRAPH_BENCHMARK_MAX_EXAMPLES"], "128")
        self.assertEqual(env["DEEPGRAPH_BENCHMARK_SEEDS"], "5")
        self.assertIn("CGGR", env["DEEPGRAPH_BENCHMARK_METHODS"])
        self.assertNotIn("DEEPGRAPH_BENCHMARK_FULL_RUN", env)

    def test_full_benchmark_env_preserves_contract_budget(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            spec_dir = workdir / "spec"
            spec_dir.mkdir()
            (spec_dir / "proxy_config.json").write_text(
                json.dumps(
                    {
                        "benchmark_max_examples_per_seed": 128,
                        "benchmark_seeds": 5,
                        "benchmark_manifest": {
                            "full_benchmark_stage": {"models": ["Qwen/Qwen2.5-7B-Instruct"]}
                        },
                    }
                ),
                encoding="utf-8",
            )

            env = validation_loop._benchmark_env_for_execution(workdir, full_benchmark=True)

        self.assertEqual(env["DEEPGRAPH_BENCHMARK_MAX_EXAMPLES"], "128")
        self.assertEqual(env["DEEPGRAPH_BENCHMARK_SEEDS"], "5")
        self.assertEqual(env["DEEPGRAPH_BENCHMARK_FULL_RUN"], "1")

    def test_iteration_judge_labels_below_baseline_keep_as_partial_recovery(self):
        judgement = validation_loop._judge_iteration_result(
            result={"status": "ok"},
            metric=0.38936,
            best_before=0.38280,
            baseline=0.42061,
            direction="higher",
            criteria={},
            iteration_index=3,
            refute_min=30,
        )

        self.assertEqual(judgement["status"], "keep")
        self.assertEqual(judgement["anomaly_type"], "partial_recovery")
        self.assertIn("remains below baseline", judgement["summary"])
        self.assertLess(judgement["baseline_effect"], 0)
        self.assertFalse(judgement["beats_baseline"])

    def test_iteration_judge_does_not_terminate_on_threshold_without_baseline_gain(self):
        judgement = validation_loop._judge_iteration_result(
            result={"status": "ok"},
            metric=0.38936,
            best_before=0.38280,
            baseline=0.42061,
            direction="higher",
            criteria={"exciting": 0.38},
            iteration_index=3,
            refute_min=30,
        )

        self.assertFalse(judgement["terminate"])
        self.assertEqual(judgement["anomaly_type"], "partial_recovery")

    def test_upper_bound_semantic_warning_is_reported(self):
        summary = {
            "primary_metric": "utility",
            "candidate_method": "cggr",
            "per_method": {
                "cggr": {"utility": 0.52},
                "oracle_router": {"utility": 0.42, "upper_bound": True},
            },
        }

        warnings = benchmark_semantic_warnings(summary, direction="higher")

        self.assertEqual(len(warnings), 1)
        self.assertIn("upper_bound", warnings[0])

    def test_upper_bound_tie_is_diagnostic_note_not_semantic_warning(self):
        summary = {
            "primary_metric": "utility",
            "candidate_method": "cggr",
            "per_method": {
                "cggr": {"utility": 0.52},
                "oracle_router": {"utility": 0.52, "upper_bound": True},
            },
        }

        warnings = benchmark_semantic_warnings(summary, direction="higher")
        notes = benchmark_diagnostic_notes(summary, direction="higher")

        self.assertEqual(warnings, [])
        self.assertEqual(len(notes), 1)
        self.assertIn("ties", notes[0])
        self.assertIn("upper_bound", notes[0])

    def test_benchmark_artifact_manifest_records_diagnostic_notes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            (workdir / "results").mkdir()
            summary = {
                "primary_metric": "utility",
                "candidate_method": "cggr",
                "num_seeds": 3,
                "full_benchmark_completed": True,
                "per_method": {
                    "cggr": {"utility": 0.52},
                    "oracle_router": {"utility": 0.52, "upper_bound": True},
                },
                "seed_results": [{"seed": 1}, {"seed": 2}, {"seed": 3}],
                "datasets": ["real"],
            }

            path, full_completed = validation_loop._write_benchmark_artifact_manifest(
                workdir,
                run_id=1,
                metric_name="utility",
                benchmark_summary=summary,
                criteria={"metric_direction": "higher", "publication_evidence_contract": {"minimum_seeds": 3}},
                verdict="confirmed",
                validation_summary_path=workdir / "validation_summary.json",
            )

            manifest = json.loads(path.read_text(encoding="utf-8"))
            self.assertTrue(full_completed)
            self.assertEqual(len(manifest["diagnostic_notes"]), 1)
            self.assertIn("diagnostic ceiling tie", manifest["diagnostic_notes"][0])

    def test_best_iteration_benchmark_summary_uses_best_kept_packet(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            packet_dir = Path(tmpdir) / "results" / "iteration_packets"
            packet_dir.mkdir(parents=True)
            latest_summary = {"per_method": {"cggr": {"metric_value": 0.38}}}
            best_summary = {"per_method": {"cggr": {"metric_value": 0.52}}}
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

            summary = best_iteration_benchmark_summary(tmpdir, best_metric=0.52, direction="higher")

        self.assertEqual(summary, best_summary)

    def test_candidate_only_scoring_postprocess_diff_is_warned(self):
        diff = """
diff --git a/train.py b/train.py
+def _cggr_canonicalize_zero_budget_answer(example, output):
+    return _extract_final_answer(output).lower()
+output = _cggr_canonicalize_zero_budget_answer(example, direct_output)
"""

        warnings = benchmark_fairness_warnings_from_diff(diff)

        self.assertEqual(len(warnings), 1)
        self.assertIn("candidate-specific", warnings[0])

    def test_pre_benchmark_guard_blocks_zero_budget_answer_shape_diff(self):
        diff = """
diff --git a/train.py b/train.py
@@
 def _build_cggr_zero_budget_prompt(question):
+    if _is_goal_purpose_question(question):
+        return f"{question}\\nAnswer with only the goal or purpose phrase. Do not explain."
+    return question
+def _is_goal_purpose_question(question):
+    return bool(GOAL_RE.search(question))
+metadata["reasoning_budget"] = "zero-budget goal/purpose phrase-only prompt"
"""

        warnings = validation_loop._blocked_pre_benchmark_diff_warnings(diff)

        self.assertGreaterEqual(len(warnings), 1)
        self.assertIn("zero-budget", " ".join(warnings).lower())

    def test_pre_benchmark_guard_blocks_broad_context_prompt_propagation(self):
        diff = """
diff --git a/train.py b/train.py
@@
+def _context_to_text(example):
+    return " ".join(example.get("context", []))
+def _question_with_context(question, example):
+    return f"{question}\\nBenchmark-provided context: {_context_to_text(example)}"
"""

        warnings = validation_loop._blocked_pre_benchmark_diff_warnings(diff)

        self.assertEqual(len(warnings), 1)
        self.assertIn("context", warnings[0].lower())

    def test_pre_benchmark_guard_allows_benign_routing_diff(self):
        diff = """
diff --git a/train.py b/train.py
@@
+def _calibrate_route_threshold(scores):
+    if not scores:
+        return 0.0
+    return max(0.1, min(0.9, sum(scores) / len(scores)))
"""

        warnings = validation_loop._blocked_pre_benchmark_diff_warnings(diff)

        self.assertEqual(warnings, [])

    def test_candidate_only_scoring_postprocess_diff_is_discarded(self):
        judgement = {
            "role": "ResultJudge",
            "status": "keep",
            "summary": "Metric improved over best-so-far and beats the baseline.",
            "anomaly_type": "hypothesis_signal",
            "continue": True,
            "terminate": False,
            "benchmark_semantic_warnings": [],
        }
        diff = """
diff --git a/train.py b/train.py
+def _cggr_canonicalize_zero_budget_answer(example, output):
+    return _extract_final_answer(output).lower()
+output = _cggr_canonicalize_zero_budget_answer(example, direct_output)
"""

        status, warnings = validation_loop._apply_benchmark_fairness_guard(
            status="keep",
            result_judgement=judgement,
            diff=diff,
        )

        self.assertEqual(status, "discard")
        self.assertEqual(judgement["status"], "discard")
        self.assertEqual(judgement["anomaly_type"], "benchmark_fairness_risk")
        self.assertFalse(judgement["terminate"])
        self.assertTrue(judgement["paper_evidence_warning"])
        self.assertEqual(len(warnings), 1)

    def test_iteration_judge_discards_positive_result_with_upper_bound_warning(self):
        judgement = validation_loop._judge_iteration_result(
            result={
                "status": "ok",
                "benchmark_summary": {
                    "primary_metric": "utility",
                    "candidate_method": "cggr",
                    "per_method": {
                        "cggr": {"utility": 0.52},
                        "oracle_router": {"utility": 0.42, "upper_bound": True},
                    },
                },
                "benchmark_metric_name": "utility",
                "benchmark_candidate_method": "cggr",
            },
            metric=0.52,
            best_before=0.42,
            baseline=0.42,
            direction="higher",
            criteria={"metric_name": "utility"},
            iteration_index=9,
            refute_min=30,
        )

        self.assertEqual(judgement["status"], "discard")
        self.assertEqual(judgement["anomaly_type"], "benchmark_semantic_risk")
        self.assertTrue(judgement["paper_evidence_warning"])
        self.assertEqual(len(judgement["benchmark_semantic_warnings"]), 1)
        self.assertIn("Discarding", judgement["summary"])

    def test_iteration_plan_focus_flags_kept_result_below_baseline(self):
        spec = validation_loop.ExperimentSpec(
            deep_insight_id=1,
            success_criteria={"metric_direction": "higher"},
        )

        judgement = validation_loop._judge_iteration_plan(
            spec,
            iteration=6,
            history=[{"iteration": 5, "status": "keep", "metric": 0.39}],
            baseline=0.42,
            best_so_far=0.39,
        )

        self.assertIn("gap to baseline", judgement["focus"])


if __name__ == "__main__":
    unittest.main()
