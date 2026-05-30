import unittest

from agents.paper_orchestra_pipeline import _figure_latex_blocks


class PaperOrchestraFigureStandardTests(unittest.TestCase):
    def test_figure_latex_blocks_follow_experiment_figure_standard(self):
        orchestrated = {
            "plotting": {
                "figure_captions": [
                    {"figure_id": "fig_main_results", "caption": "Main results."},
                    {"figure_id": "fig_backend_rank_lines_1x4", "caption": "Rank stability."},
                    {"figure_id": "fig_metric_trajectory", "caption": "Internal trajectory."},
                ],
                "plotting_executor": {
                    "assets": [
                        {
                            "figure_id": "fig_main_results",
                            "path": "/tmp/fig_main_results.pdf",
                            "kind": "plot",
                        },
                        {
                            "figure_id": "fig_backend_rank_lines_1x4",
                            "path": "/tmp/fig_backend_rank_lines_1x4.pdf",
                            "kind": "plot",
                            "aspect_ratio": "4:1",
                        },
                        {
                            "figure_id": "fig_metric_trajectory",
                            "path": "/tmp/fig_metric_trajectory.pdf",
                            "kind": "plot",
                        },
                    ]
                },
            }
        }

        tex = _figure_latex_blocks(orchestrated)

        self.assertIn(r"\begin{figure}[t]", tex)
        self.assertIn(r"\includegraphics[width=\linewidth]{figures/fig_main_results.pdf}", tex)
        self.assertIn(r"\begin{figure*}[t]", tex)
        self.assertIn(r"\includegraphics[width=\textwidth]{figures/fig_backend_rank_lines_1x4.pdf}", tex)
        self.assertNotIn("fig_metric_trajectory", tex)

    def test_single_column_main_result_stays_single_column(self):
        orchestrated = {
            "plotting": {
                "plotting_executor": {
                    "assets": [
                        {
                            "figure_id": "fig_main_results",
                            "path": "/tmp/fig_main_results.pdf",
                            "kind": "plot",
                            "aspect_ratio": "4:3",
                        }
                    ]
                }
            }
        }

        tex = _figure_latex_blocks(orchestrated)

        self.assertIn(r"\begin{figure}[t]", tex)
        self.assertIn(r"\includegraphics[width=\linewidth]{figures/fig_main_results.pdf}", tex)
        self.assertNotIn(r"\begin{figure*}[t]", tex)


if __name__ == "__main__":
    unittest.main()
