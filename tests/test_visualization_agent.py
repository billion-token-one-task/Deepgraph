import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agents import visualization_agent


class VisualizationAgentTests(unittest.TestCase):
    def test_generate_visualization_bundle_emits_result_diagrams_and_report_references(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            (workdir / "results").mkdir(parents=True, exist_ok=True)
            final_report = workdir / "final_report.md"
            final_report.write_text("# Existing Report\n\nBody.\n", encoding="utf-8")
            summary_path = workdir / "results" / "validation_summary.json"
            summary_path.write_text("{}", encoding="utf-8")

            def fake_metric_figure(iterations, baseline, metric_name, out_svg, **kwargs):
                out_svg.parent.mkdir(parents=True, exist_ok=True)
                out_svg.write_text("<svg>trajectory</svg>", encoding="utf-8")
                out_pdf = out_svg.with_suffix(".pdf")
                out_pdf.write_text("%PDF", encoding="utf-8")
                code_path = out_svg.with_suffix(".py")
                code_path.write_text("print('plot')\n", encoding="utf-8")
                return {
                    "ok": True,
                    "score": 0.91,
                    "notes": "critic_pass",
                    "attempts": 1,
                    "svg_path": str(out_svg),
                    "pdf_path": str(out_pdf),
                    "code_path": str(code_path),
                    "used_fallback": False,
                }

            insight = {
                "id": 3,
                "title": "Adaptive Routing for Robust Validation",
                "problem_statement": "Existing methods fail under distribution shift.",
                "existing_weakness": "Baselines overfit a narrow evidence slice.",
                "proposed_method": json.dumps(
                    {
                        "name": "Confidence Gated Routing",
                        "one_line": "Route examples by uncertainty before applying the solver.",
                        "definition": "A lightweight controller sends high-uncertainty cases to specialist modules.",
                    }
                ),
                "experimental_plan": json.dumps(
                    {
                        "datasets": [{"name": "SyntheticShift"}],
                        "baselines": [{"name": "DirectSolver"}],
                        "metrics": {"primary": "accuracy"},
                    }
                ),
                "source_node_ids": json.dumps(["ml.robustness"]),
                "source_paper_ids": json.dumps(["2401.00001"]),
            }
            iterations = [
                {"iteration_number": 1, "metric_value": 0.5, "status": "ok"},
                {"iteration_number": 2, "metric_value": 0.64, "status": "keep"},
            ]
            result_rows = [
                {
                    "method_name": "DirectSolver",
                    "dataset_name": "SyntheticShift",
                    "metric_name": "accuracy",
                    "metric_value": 0.58,
                }
            ]
            relation_rows = [
                {
                    "subject_name": "Confidence routing",
                    "object_name": "Robust validation",
                    "predicate": "improves",
                    "confidence": 0.93,
                }
            ]

            with (
                mock.patch.object(
                    visualization_agent.db,
                    "fetchall",
                    side_effect=[iterations, result_rows, relation_rows],
                ),
                mock.patch.object(
                    visualization_agent.figure_agent,
                    "generate_metric_figure_with_retry",
                    side_effect=fake_metric_figure,
                ),
            ):
                bundle = visualization_agent.generate_visualization_bundle(
                    run_id=9,
                    workdir=workdir,
                    insight=insight,
                    metric_name="accuracy",
                    baseline_metric_value=0.5,
                    best_metric_value=0.64,
                    verdict="confirmed",
                    summary_path=summary_path,
                )

            figure_ids = {asset["figure_id"] for asset in bundle["assets"]}
            self.assertIn("fig_approach_overview", figure_ids)
            self.assertIn("fig_method_architecture", figure_ids)
            self.assertIn("fig_metric_trajectory", figure_ids)
            self.assertIn("fig_baseline_vs_proposed", figure_ids)
            self.assertIn("fig_literature_results", figure_ids)
            self.assertIn("fig_knowledge_subgraph", figure_ids)
            self.assertTrue(Path(bundle["manifest_path"]).exists())
            self.assertTrue((workdir / "figures" / "fig_knowledge_subgraph.dot").exists())
            self.assertIn("Generated Figures", final_report.read_text(encoding="utf-8"))
            self.assertIn("fig_method_architecture", final_report.read_text(encoding="utf-8"))

    def test_write_figure_references_creates_sidecar_when_final_report_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            asset_path = workdir / "figures" / "fig.svg"
            asset_path.parent.mkdir(parents=True, exist_ok=True)
            asset_path.write_text("<svg/>", encoding="utf-8")

            ref = visualization_agent.write_figure_references(
                workdir,
                [
                    {
                        "figure_id": "fig_test",
                        "asset_kind": "svg",
                        "path": str(asset_path),
                        "caption": "A generated figure.",
                    }
                ],
            )

            self.assertTrue(Path(ref).exists())
            self.assertIn("fig_test", Path(ref).read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
