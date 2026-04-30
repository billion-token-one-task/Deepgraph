import os
import json
import tempfile
import unittest
from unittest import mock
from pathlib import Path

from agents.manuscript_pipeline import generate_submission_bundle
from agents import workspace_layout
from db import database


class ManuscriptBundleTests(unittest.TestCase):
    def setUp(self):
        self._saved_pg_url = os.environ.pop("DEEPGRAPH_DATABASE_URL", None)
        self._old_database_url = database.DATABASE_URL
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmpdir.name) / "test.db"
        self.old_db_path = database.DB_PATH
        for attr in ("pg_conn", "sqlite_conn", "conn"):
            if hasattr(database._local, attr):
                try:
                    getattr(database._local, attr).close()
                except Exception:
                    pass
                setattr(database._local, attr, None)
        database.DATABASE_URL = ""
        database.DB_PATH = self.db_path
        database.init_db()
        self.workspace_root = Path(self.tmpdir.name) / "ideas"
        self.workspace_patch = mock.patch.object(workspace_layout, "IDEA_WORKSPACE_DIR", self.workspace_root)
        self.workspace_patch.start()

        database.execute(
            """
            INSERT INTO deep_insights
            (id, tier, title, mechanism_type, submission_status, supporting_papers, source_paper_ids, source_node_ids, evidence_summary)
            VALUES (1, 2, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "Auto Manuscript Insight",
                "mechanism_mismatch",
                "not_started",
                '["2401.12345"]',
                '["2401.12345"]',
                '["ml.test"]',
                "Evidence summary from the graph.",
            ),
        )
        database.execute(
            """
            INSERT INTO experiment_runs
            (id, deep_insight_id, status, baseline_metric_name, baseline_metric_value, best_metric_value, effect_pct, hypothesis_verdict, workdir, proxy_config)
            VALUES (1, 1, 'completed', 'accuracy', 0.5, 0.61, 22.0, 'confirmed', ?, ?)
            """,
            (
                str(Path(self.tmpdir.name) / "run1"),
                json.dumps({"formal_experiment": True, "smoke_test_only": False}),
            ),
        )
        database.execute(
            """
            INSERT INTO experiment_iterations
            (run_id, iteration_number, phase, metric_value, status, description)
            VALUES (1, 1, 'hypothesis_testing', 0.61, 'keep', 'best run')
            """
        )
        database.execute(
            """
            INSERT INTO experimental_claims
            (run_id, deep_insight_id, claim_text, verdict)
            VALUES (1, 1, 'The method improved accuracy.', 'confirmed')
            """
        )
        database.commit()

    def tearDown(self):
        for attr in ("pg_conn", "sqlite_conn", "conn"):
            if hasattr(database._local, attr):
                try:
                    getattr(database._local, attr).close()
                except Exception:
                    pass
                setattr(database._local, attr, None)
        database.DATABASE_URL = self._old_database_url
        database.DB_PATH = self.old_db_path
        self.workspace_patch.stop()
        if self._saved_pg_url is not None:
            os.environ["DEEPGRAPH_DATABASE_URL"] = self._saved_pg_url
        self.tmpdir.cleanup()

    def _stub_orchestra(self, state, literature_block, paper_ids, iterations, *, figures_dir, baseline, metric_name):
        figures_dir.mkdir(parents=True, exist_ok=True)
        (figures_dir / "fig_metric_trajectory.svg").write_text(
            '<svg xmlns="http://www.w3.org/2000/svg"><text>metric</text></svg>',
            encoding="utf-8",
        )
        return {
            "outline": {"plotting_plan": [{"figure_id": "fig_metric_trajectory", "plot_type": "plot"}]},
            "plotting": {
                "figure_captions": [{"figure_id": "fig_metric_trajectory", "caption": "Metric trajectory."}],
                "plotting_executor": {
                    "assets": [
                        {
                            "figure_id": "fig_metric_trajectory",
                            "path": str(figures_dir / "fig_metric_trajectory.svg"),
                            "svg_path": str(figures_dir / "fig_metric_trajectory.svg"),
                            "pdf_path": "",
                            "code_path": "",
                            "objective": "Show the metric trajectory.",
                            "kind": "plot",
                        }
                    ]
                },
            },
            "literature_discovery": {},
            "literature_text": r"\section{Introduction}Intro with \cite{cite_a}.\section{Related Work}Related work with \cite{cite_a}.",
            "sections_raw": "",
            "refined": {
                "abstract": "Abstract text.",
                "introduction": "Introduction text with \\cite{cite_a}.",
                "method": "Method text.",
                "experiments": "Experiments text.",
                "discussion": "Discussion text.",
            },
            "refinement_full_text": "",
            "agentreview_worklog": [],
            "bibtex": "@misc{cite_a,\n  title = {Verified Paper},\n  author = {Author One},\n  year = {2024}\n}\n",
            "bib_keys": ["cite_a"],
            "citation_registry": [
                {
                    "cite_key": "cite_a",
                    "title": "Verified Paper",
                    "abstract": "Paper abstract.",
                    "year": 2024,
                    "source_claim_ids": ["1"],
                    "source_node_ids": ["ml.test"],
                }
            ],
            "claim_citation_map": {
                "1": {
                    "claim_text": "The method improved accuracy.",
                    "source_paper_ids": ["2401.12345"],
                    "source_node_ids": ["ml.test"],
                    "cite_keys": ["cite_a"],
                }
            },
        }

    @mock.patch("agents.paper_orchestra_pipeline._run_full_pipeline")
    def test_generate_submission_bundle_creates_verified_bundle_files_and_db_rows(self, run_full):
        run_full.side_effect = self._stub_orchestra
        result = generate_submission_bundle(1, bundle_formats=["conference"])
        self.assertIn("manuscript_run_id", result)
        self.assertEqual(result["backend"], "paper_orchestra")
        bundle = database.fetchone("SELECT * FROM submission_bundles WHERE manuscript_run_id=?", (result["manuscript_run_id"],))
        self.assertIsNotNone(bundle)
        bundle_path = Path(bundle["bundle_path"])
        self.assertTrue((bundle_path / "main.tex").exists())
        self.assertTrue((bundle_path / "artifact_manifest.json").exists())
        self.assertTrue((bundle_path / "citation_registry.json").exists())
        self.assertTrue((bundle_path / "claim_citation_map.json").exists())
        self.assertTrue((bundle_path / "figures" / "figure_manifest.json").exists())
        self.assertIn("cite_a", (bundle_path / "references.bib").read_text(encoding="utf-8"))
        self.assertIn("fig_metric_trajectory.svg", (bundle_path / "main.tex").read_text(encoding="utf-8"))
        self.assertTrue((self.workspace_root / "idea_1" / "paper" / "current" / "main.tex").exists())

    def test_generate_submission_bundle_blocks_non_formal_run(self):
        database.execute(
            "UPDATE experiment_runs SET proxy_config=? WHERE id=1",
            (json.dumps({"formal_experiment": False, "smoke_test_only": True}),),
        )
        database.commit()

        result = generate_submission_bundle(1, bundle_formats=["conference"])

        self.assertIn("error", result)
        self.assertIn("formal", result["error"].lower())


if __name__ == "__main__":
    unittest.main()
