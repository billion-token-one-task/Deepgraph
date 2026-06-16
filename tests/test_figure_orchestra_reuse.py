import tempfile
import unittest
from pathlib import Path

from agents.paperorchestra.figure_orchestra import _augment_plotting_plan, _run_external_diagram


class FigureOrchestraPlanTests(unittest.TestCase):
    def test_default_plot_pack_skips_hyperparameter_without_sweep_artifact(self):
        state = {
            "benchmark_summary": {
                "primary_metric": "accuracy",
                "per_method": {
                    "Direct": {"accuracy": 0.70},
                    "Ours": {"accuracy": 0.80},
                },
                "ablation_table": [
                    {"ablation": "Full", "accuracy": 0.80},
                    {"ablation": "No gate", "accuracy": 0.75},
                ],
            }
        }

        plan = _augment_plotting_plan([], state, [], "accuracy")

        self.assertEqual([fig.get("figure_id") for fig in plan], ["fig_main_results", "fig_ablation_results"])

    def test_default_plot_pack_includes_hyperparameter_when_sweep_exists(self):
        state = {
            "benchmark_summary": {
                "primary_metric": "accuracy",
                "per_method": {
                    "Direct": {"accuracy": 0.70},
                    "Ours": {"accuracy": 0.80},
                },
                "ablation_table": [
                    {"ablation": "Full", "accuracy": 0.80},
                    {"ablation": "No gate", "accuracy": 0.75},
                ],
                "route_rate_sweep_table": [
                    {"route_rate": 0.1, "accuracy": 0.76, "avg_new_tokens": 8},
                    {"route_rate": 0.2, "accuracy": 0.80, "avg_new_tokens": 12},
                ],
            }
        }

        plan = _augment_plotting_plan([], state, [], "accuracy")

        self.assertEqual(
            [fig.get("figure_id") for fig in plan],
            ["fig_main_results", "fig_ablation_results", "fig_hyperparameter_sweep"],
        )


class FigureOrchestraReuseTests(unittest.TestCase):
    def test_external_diagram_failure_reuses_existing_png(self):
        with tempfile.TemporaryDirectory() as tmp:
            figures_dir = Path(tmp)
            existing = figures_dir / "fig_motivation_symbolic.png"
            existing.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 5000)

            asset = _run_external_diagram(
                {
                    "figure_id": "fig_motivation_symbolic",
                    "title": "Motivation",
                    "objective": "Show the motivation schematic.",
                },
                figures_dir=figures_dir,
                state={},
                paperbanana_cmd="python3 -c 'import sys; sys.exit(4)'",
            )

            self.assertEqual(asset["path"], str(existing))
            self.assertEqual(asset["svg_path"], "")
            self.assertIn("reused_existing_png", asset["notes"])
            self.assertFalse((figures_dir / "fig_motivation_symbolic.svg").exists())


if __name__ == "__main__":
    unittest.main()
