import os
import tempfile
import unittest
from pathlib import Path

from agents.paperorchestra.figure_orchestra import _augment_plotting_plan, _run_external_diagram


# These cases describe the generative figure planner that meta-harness-v1
# demoted to plugins.examples.cggr (c25e63c): in production
# _augment_plotting_plan is a passthrough and _run_external_diagram refuses.
# The refusal itself is asserted by
# tests/test_evidence_planner.py::FigureOrchestraBoundaryTests.
GENERATIVE_FIGURES = os.getenv(
    "DEEPGRAPH_ENABLE_NONPROD_EXAMPLE_PLUGINS", ""
).strip().lower() in {"1", "true", "yes"}


def _demoted_figure_orchestra():
    if not GENERATIVE_FIGURES:
        return None
    try:
        from plugins.examples.cggr import figure_orchestra as demoted
    except Exception:
        return None
    return demoted


DEMOTED_FIGURES = _demoted_figure_orchestra()
if DEMOTED_FIGURES is not None:
    _augment_plotting_plan = DEMOTED_FIGURES._augment_plotting_plan
    _run_external_diagram = DEMOTED_FIGURES._run_external_diagram

requires_generative_figures = unittest.skipUnless(
    DEMOTED_FIGURES is not None,
    "figure generation was demoted to plugins.examples.cggr in "
    "meta-harness-v1; set DEEPGRAPH_ENABLE_NONPROD_EXAMPLE_PLUGINS=1 and "
    "install the plotting stack to run",
)


@requires_generative_figures
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


@requires_generative_figures
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
