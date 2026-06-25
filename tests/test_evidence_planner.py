import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agents.evidence_planner import build_evidence_plan
from agents.paperorchestra import figure_orchestra
from agents.paperorchestra.figure_orchestra import run_figure_orchestra


class EvidencePlannerTests(unittest.TestCase):
    def test_performance_claim_requires_main_table_but_can_skip_visualization(self):
        insight = {
            "tier": 2,
            "title": "Better classifier",
            "problem_statement": "Improve benchmark accuracy.",
            "proposed_method": {
                "name": "BetterNet",
                "type": "architecture",
                "one_line": "Improves classification.",
                "key_properties": ["new block"],
            },
            "experimental_plan": {
                "baselines": [{"name": "BaseNet"}],
                "datasets": [{"name": "ImageNet"}],
                "metrics": {"primary": "accuracy"},
                "ablations": [],
            },
        }
        plan = build_evidence_plan(insight)
        self.assertEqual(plan["claim_type"], "performance")
        self.assertTrue(plan["main_table"]["enabled"])
        self.assertEqual(plan["main_table"]["priority"], "required")
        self.assertFalse(plan["visualization"]["enabled"])

    def test_efficiency_claim_requires_visualization(self):
        insight = {
            "tier": 2,
            "title": "Faster model with lower latency",
            "problem_statement": "Reduce inference latency and memory.",
            "proposed_method": {
                "name": "FastNet",
                "type": "architecture",
            },
            "experimental_plan": {
                "baselines": [{"name": "BaseNet"}],
                "datasets": [{"name": "ImageNet"}],
                "metrics": {"primary": "accuracy", "secondary": ["latency", "memory"]},
            },
        }
        plan = build_evidence_plan(insight)
        self.assertEqual(plan["claim_type"], "efficiency")
        self.assertTrue(plan["visualization"]["enabled"])
        self.assertEqual(plan["visualization"]["priority"], "required")

    def test_disabled_visualization_skips_default_figure_generation(self):
        state = {
            "title": "No Figure Insight",
            "method_name": "NoFigureNet",
            "evidence_plan": {
                "visualization": {
                    "enabled": False,
                    "priority": "skip",
                    "reason": "Table-first claim.",
                }
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = run_figure_orchestra(
                outline={},
                state=state,
                iterations=[],
                figures_dir=Path(tmpdir),
                baseline=None,
                metric_name="accuracy",
                paperbanana_cmd=None,
            )
            self.assertEqual(manifest["assets"], [])

    def test_motivation_overview_diagram_uses_banana_by_default(self):
        outline = {
            "plotting_plan": [
                {
                    "figure_id": "fig_motivation_overview",
                    "plot_type": "diagram",
                    "title": "Motivation overview",
                    "objective": "Motivation and overview for selective reasoning.",
                }
            ]
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = run_figure_orchestra(
                outline=outline,
                state={"title": "Selective reasoning"},
                iterations=[],
                figures_dir=Path(tmpdir),
                baseline=None,
                metric_name="accuracy",
                paperbanana_cmd="printf x > {output}",
            )
            self.assertGreaterEqual(manifest["generated_count"], 1)
            asset = next(
                row for row in manifest["assets"]
                if row.get("figure_id") == "fig_motivation_overview"
            )
            self.assertEqual(asset["notes"], "paperbanana_ok")
            self.assertTrue(Path(asset["path"]).exists())

    def test_paperbanana_timeout_is_configurable_for_slow_image_generation(self):
        fig = {
            "figure_id": "fig_motivation_overview",
            "plot_type": "diagram",
            "title": "Motivation overview",
            "objective": "Motivation and overview for selective reasoning.",
        }

        def _fake_run(command, **kwargs):
            output_arg = command.split("--out ", 1)[1].split(" --spec", 1)[0].strip("'")
            Path(output_arg).write_bytes(b"x")
            proc = mock.Mock()
            proc.returncode = 0
            proc.stdout = ""
            proc.stderr = ""
            return proc

        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.object(
            figure_orchestra, "PAPERBANANA_EXTERNAL_TIMEOUT_SECONDS", 777
        ), mock.patch.object(figure_orchestra.subprocess, "run", side_effect=_fake_run) as run:
            asset = figure_orchestra._run_external_diagram(
                fig,
                figures_dir=Path(tmpdir),
                state={"title": "Selective reasoning"},
                paperbanana_cmd="paperbanana --out {output} --spec {spec}",
            )

        self.assertEqual(asset["notes"], "paperbanana_ok")
        self.assertEqual(run.call_args.kwargs["timeout"], 777)

    def test_motivation_overview_diagram_cannot_opt_out_to_native(self):
        outline = {
            "plotting_plan": [
                {
                    "figure_id": "fig_motivation_overview",
                    "plot_type": "diagram",
                    "title": "Motivation overview",
                    "objective": "Motivation and overview for selective reasoning.",
                }
            ]
        }
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.dict(
            "os.environ",
            {"DEEPGRAPH_PAPERBANANA_MOTIVATION_OVERVIEW": "false"},
        ):
            manifest = run_figure_orchestra(
                outline=outline,
                state={"title": "Selective reasoning"},
                iterations=[],
                figures_dir=Path(tmpdir),
                baseline=None,
                metric_name="accuracy",
                paperbanana_cmd="printf x > {output}",
            )
            asset = next(
                row for row in manifest["assets"]
                if row.get("figure_id") == "fig_motivation_overview"
            )
            self.assertEqual(asset["notes"], "paperbanana_ok")
            self.assertTrue(Path(asset["path"]).exists())

    def test_motivation_overview_diagram_blocks_without_banana(self):
        outline = {
            "plotting_plan": [
                {
                    "figure_id": "fig_motivation_overview",
                    "plot_type": "diagram",
                    "title": "Motivation overview",
                    "objective": "Motivation and overview for selective reasoning.",
                }
            ]
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = run_figure_orchestra(
                outline=outline,
                state={"title": "Selective reasoning"},
                iterations=[],
                figures_dir=Path(tmpdir),
                baseline=None,
                metric_name="accuracy",
                paperbanana_cmd=None,
            )
            asset = next(
                row for row in manifest["assets"]
                if row.get("figure_id") == "fig_motivation_overview"
            )
            self.assertEqual(asset["kind"], "blocked")
            self.assertIn("Gemini/PaperBanana", asset["blocker"])
            self.assertTrue(manifest["blockers"])

    def test_non_backend_benchmark_uses_only_main_results_plot(self):
        outline = {
            "plotting_plan": [
                {"figure_id": "fig_metric_trajectory", "plot_type": "plot"},
                {"figure_id": "fig_search_dynamics_keep_discard", "plot_type": "plot"},
                {"figure_id": "fig_benchmark_method_panel", "plot_type": "plot"},
            ]
        }
        state = {
            "title": "Training-free selector",
            "result_packet": {
                "benchmark_summary": {
                    "primary_metric": "accuracy",
                    "per_method": {
                        "Direct": {"accuracy": 0.6, "std": 0.02},
                        "DPC": {"accuracy": 0.72, "std": 0.01},
                    },
                }
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = run_figure_orchestra(
                outline=outline,
                state=state,
                iterations=[{"iteration_number": 1, "metric_value": 0.72}],
                figures_dir=Path(tmpdir),
                baseline=0.6,
                metric_name="accuracy",
                paperbanana_cmd=None,
            )

            self.assertEqual([asset["figure_id"] for asset in manifest["assets"]], ["fig_main_results"])
            asset = manifest["assets"][0]
            self.assertEqual(asset["notes"], "native_main_results_bar")
            self.assertEqual(asset.get("aspect_ratio"), "4:3")
            self.assertTrue(Path(asset["path"]).exists())
            self.assertTrue(Path(asset["pdf_path"]).exists())

    def test_main_results_stays_standard_bar_when_token_cost_exists(self):
        state = {
            "title": "Training-free selector",
            "result_packet": {
                "benchmark_summary": {
                    "primary_metric": "accuracy",
                    "per_method": {
                        "Direct": {"accuracy": 0.6, "std": 0.02, "avg_new_tokens": 40, "avg_latency_seconds": 0.5, "route_rate": 0.0},
                        "DPC": {"accuracy": 0.72, "std": 0.01, "avg_new_tokens": 100, "avg_latency_seconds": 1.2, "route_rate": 0.7},
                    },
                }
            },
            "evidence_plan": {"visualization": {"enabled": True, "priority": "required"}},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = run_figure_orchestra(
                outline={},
                state=state,
                iterations=[],
                figures_dir=Path(tmpdir),
                baseline=0.6,
                metric_name="accuracy",
                paperbanana_cmd=None,
            )

            asset = manifest["assets"][0]
            self.assertEqual(asset["figure_id"], "fig_main_results")
            self.assertEqual(asset.get("aspect_ratio"), "4:3")
            self.assertEqual(asset["notes"], "native_main_results_bar")

    def test_backend_matrix_uses_three_standard_backend_figures(self):
        matrix = {
            "Direct": {
                "HF": {"accuracy": 0.60, "std": 0.02},
                "vLLM": {"accuracy": 0.58, "std": 0.03},
            },
            "DPC": {
                "HF": {"accuracy": 0.72, "std": 0.01},
                "vLLM": {"accuracy": 0.70, "std": 0.02},
            },
        }
        state = {
            "title": "Backend-aware selector",
            "result_packet": {
                "benchmark_summary": {
                    "primary_metric": "accuracy",
                    "per_method_backend": matrix,
                    "per_dataset_backend": {
                        "GSM8K": matrix,
                        "MATH": matrix,
                        "BBH": matrix,
                        "MMLU": matrix,
                    },
                    "backends": ["HF", "vLLM"],
                    "methods": ["Direct", "DPC"],
                }
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = run_figure_orchestra(
                outline={"plotting_plan": []},
                state=state,
                iterations=[],
                figures_dir=Path(tmpdir),
                baseline=0.6,
                metric_name="accuracy",
                paperbanana_cmd=None,
            )

            self.assertEqual(
                [asset["figure_id"] for asset in manifest["assets"]],
                ["fig_backend_grouped_bars", "fig_backend_heatmap_single", "fig_backend_rank_lines_1x4"],
            )
            self.assertEqual(
                [asset["notes"] for asset in manifest["assets"]],
                ["native_backend_grouped_bars", "native_backend_heatmap_single", "native_backend_rank_lines_1x4"],
            )
            for asset in manifest["assets"]:
                self.assertTrue(Path(asset["path"]).exists())
                self.assertTrue(Path(asset["pdf_path"]).exists())


if __name__ == "__main__":
    unittest.main()
