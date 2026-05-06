import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agents import validation_loop


class ValidationLoopGitFallbackTests(unittest.TestCase):
    def test_git_helpers_are_safe_when_git_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            code_dir = Path(tmpdir)
            (code_dir / "train.py").write_text("print('hello')", encoding="utf-8")

            with mock.patch.object(validation_loop, "_git_binary", return_value=None):
                self.assertIsNone(
                    validation_loop._git_commit(code_dir, "test commit")
                )
                self.assertEqual(validation_loop._git_diff(code_dir), "")
                validation_loop._git_reset(code_dir, "deadbeef")

    def test_find_train_file_prefers_nested_proxy_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            code_dir = Path(tmpdir)
            nested = code_dir / "src" / "qa"
            nested.mkdir(parents=True, exist_ok=True)
            target = nested / "inference.py"
            target.write_text("print('hello')", encoding="utf-8")

            resolved = validation_loop._find_train_file(
                code_dir, "src/qa/inference.py"
            )

        self.assertEqual(resolved, target)

    def test_generate_validation_figures_records_plot_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            summary_path = workdir / "results" / "validation_summary.json"
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text("{}", encoding="utf-8")

            def fake_generate(**kwargs):
                figures_dir = kwargs["workdir"] / "figures"
                figures_dir.mkdir(parents=True, exist_ok=True)
                out_svg = figures_dir / "fig_metric_trajectory.svg"
                out_svg.write_text("<svg>accuracy trajectory</svg>", encoding="utf-8")
                out_pdf = figures_dir / "fig_metric_trajectory.pdf"
                out_pdf.write_text("%PDF", encoding="utf-8")
                code_path = figures_dir / "fig_metric_trajectory.py"
                code_path.write_text("print('plot')\n", encoding="utf-8")
                manifest_path = figures_dir / "figure_manifest.json"
                manifest_path.write_text("{}", encoding="utf-8")
                return {
                    "assets": [
                        {
                            "figure_id": "fig_metric_trajectory",
                            "figure_kind": "metric_trajectory",
                            "asset_kind": "svg",
                            "path": str(out_svg),
                            "caption": "Metric trajectory.",
                            "source": "experiment_iterations",
                            "metric_name": "accuracy",
                        },
                        {
                            "figure_id": "fig_metric_trajectory",
                            "figure_kind": "metric_trajectory",
                            "asset_kind": "pdf",
                            "path": str(out_pdf),
                            "caption": "Metric trajectory.",
                            "source": "experiment_iterations",
                            "metric_name": "accuracy",
                        },
                        {
                            "figure_id": "fig_metric_trajectory",
                            "figure_kind": "metric_trajectory",
                            "asset_kind": "source",
                            "path": str(code_path),
                            "caption": "Metric trajectory.",
                            "source": "experiment_iterations",
                            "metric_name": "accuracy",
                        },
                    ],
                    "manifest_path": str(manifest_path),
                    "references_path": "",
                }

            with (
                mock.patch.object(validation_loop.db, "execute") as execute,
                mock.patch.object(
                    validation_loop.visualization_agent,
                    "generate_visualization_bundle",
                    side_effect=fake_generate,
                ) as generate,
            ):
                assets = validation_loop._generate_validation_figures(
                    7,
                    workdir,
                    insight={"id": 3, "title": "Insight"},
                    metric_name="accuracy",
                    baseline_metric_value=0.5,
                    best_metric_value=0.62,
                    verdict="confirmed",
                    summary_path=summary_path,
                )

            self.assertEqual(len(assets), 3)
            self.assertTrue((workdir / "figures" / "figure_manifest.json").exists())
            generate.assert_called_once()
            artifact_types = [call.args[0].strip() for call in execute.call_args_list]
            self.assertTrue(all("INSERT INTO experiment_artifacts" in sql for sql in artifact_types))
            params = [call.args[1] for call in execute.call_args_list]
            self.assertEqual([row[1] for row in params], ["plot", "plot", "source_data", "source_data"])

    def test_generate_validation_figures_is_non_blocking_on_render_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            with (
                mock.patch.object(validation_loop.db, "execute") as execute,
                mock.patch.object(
                    validation_loop.visualization_agent,
                    "generate_visualization_bundle",
                    side_effect=RuntimeError("renderer unavailable"),
                ),
            ):
                assets = validation_loop._generate_validation_figures(
                    8,
                    workdir,
                    insight={"id": 4, "title": "Insight"},
                    metric_name="accuracy",
                    baseline_metric_value=0.5,
                )

        self.assertEqual(assets, [])
        execute.assert_not_called()

    def test_generate_validation_figures_is_non_blocking_on_artifact_registration_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            figures_dir = workdir / "figures"
            figures_dir.mkdir(parents=True, exist_ok=True)
            out_svg = figures_dir / "fig_metric_trajectory.svg"
            out_svg.write_text("<svg/>", encoding="utf-8")
            manifest_path = figures_dir / "figure_manifest.json"
            manifest_path.write_text("{}", encoding="utf-8")

            with (
                mock.patch.object(
                    validation_loop.visualization_agent,
                    "generate_visualization_bundle",
                    return_value={
                        "assets": [
                            {
                                "figure_id": "fig_metric_trajectory",
                                "figure_kind": "metric_trajectory",
                                "asset_kind": "svg",
                                "path": str(out_svg),
                                "caption": "Metric trajectory.",
                                "source": "experiment_iterations",
                                "metric_name": "accuracy",
                            }
                        ],
                        "manifest_path": str(manifest_path),
                        "references_path": "",
                    },
                ),
                mock.patch.object(validation_loop.db, "execute", side_effect=RuntimeError("db unavailable")),
            ):
                assets = validation_loop._generate_validation_figures(
                    9,
                    workdir,
                    insight={"id": 5, "title": "Insight"},
                    metric_name="accuracy",
                    baseline_metric_value=0.5,
                )

        self.assertEqual(len(assets), 1)

    def test_run_validation_loop_blocks_non_formal_experiment(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            (workdir / "code").mkdir()
            run = {
                "id": 7,
                "deep_insight_id": 3,
                "workdir": str(workdir),
                "proxy_config": '{"formal_experiment": false, "smoke_test_only": true}',
            }
            insight = {
                "id": 3,
                "tier": 2,
                "title": "Smoke",
                "proposed_method": '{"name": "M", "definition": "f(x)"}',
                "experimental_plan": '{"baselines": [], "datasets": [], "metrics": {"primary": "acc"}}',
            }

            with (
                mock.patch.object(validation_loop.db, "fetchone", side_effect=[run, insight]),
                mock.patch.object(validation_loop.db, "execute") as execute,
                mock.patch.object(validation_loop.db, "commit"),
            ):
                result = validation_loop.run_validation_loop(7)

        self.assertEqual(result["verdict"], "blocked")
        self.assertEqual(result["reason"], "non_formal_experiment")
        execute.assert_called()

    def test_determine_final_verdict_marks_reproduction_only_runs(self):
        verdict = validation_loop._determine_final_verdict(
            baseline=1.0,
            best_value=1.0,
            direction="higher",
            criteria={"exciting": 0.8, "solid": 0.7},
            total_iters=0,
            total_kept=0,
            refute_min=30,
        )

        self.assertEqual(verdict, "reproduced")

    def test_determine_final_verdict_requires_real_improvement_for_confirmation(self):
        verdict = validation_loop._determine_final_verdict(
            baseline=1.0533,
            best_value=1.0533,
            direction="higher",
            criteria={"exciting": 0.79, "solid": 0.77},
            total_iters=1,
            total_kept=0,
            refute_min=30,
        )

        self.assertEqual(verdict, "inconclusive")

    def test_determine_final_verdict_accepts_benchmark_evidence(self):
        verdict = validation_loop._determine_final_verdict(
            baseline=0.77,
            best_value=0.80,
            direction="higher",
            criteria={"exciting": 0.79, "solid": 0.77},
            total_iters=0,
            total_kept=0,
            refute_min=30,
            benchmark_summary={
                "primary_metric": "utility",
                "candidate_method": "cggr",
                "best_method": "cggr",
                "num_seeds": 5,
                "per_method": {
                    "direct": {"utility": 0.71},
                    "adaptive_confidence": {"utility": 0.77},
                    "cggr": {"utility": 0.80},
                },
            },
        )

        self.assertEqual(verdict, "confirmed")

    def test_repo_snapshot_restore_recovers_multi_file_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            code_dir = Path(tmpdir) / "code"
            code_dir.mkdir(parents=True, exist_ok=True)
            snapshot = Path(tmpdir) / "snapshot"
            (code_dir / "train.py").write_text("print('a')\n", encoding="utf-8")
            (code_dir / "helper.py").write_text("VALUE = 1\n", encoding="utf-8")

            validation_loop._snapshot_repo_tree(code_dir, snapshot)

            (code_dir / "train.py").write_text("print('b')\n", encoding="utf-8")
            (code_dir / "helper.py").unlink()
            (code_dir / "new_file.py").write_text("X = 2\n", encoding="utf-8")

            validation_loop._restore_repo_tree(snapshot, code_dir)

            self.assertEqual((code_dir / "train.py").read_text(encoding="utf-8"), "print('a')\n")
            self.assertEqual((code_dir / "helper.py").read_text(encoding="utf-8"), "VALUE = 1\n")
            self.assertFalse((code_dir / "new_file.py").exists())

    def test_launch_coding_agent_returns_codex_summary_when_available(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            code_dir = workdir / "code"
            code_dir.mkdir(parents=True, exist_ok=True)
            (code_dir / "train.py").write_text("print('baseline')\n", encoding="utf-8")
            spec = validation_loop.ExperimentSpec(
                deep_insight_id=1,
                experimental_plan={"baselines": [], "datasets": [], "metrics": {}},
                evidence_plan={"main_table": {"enabled": True}},
            )

            with (
                mock.patch.object(validation_loop.codex_executor, "codex_available", return_value=True),
                mock.patch.object(
                    validation_loop.codex_executor,
                    "run_codex_iteration",
                    return_value={
                        "ok": True,
                        "summary": "Codex changed repo files",
                        "artifact_paths": {"codex_last_message": "/tmp/last.json"},
                    },
                ),
                mock.patch.object(validation_loop, "_read_proxy_config", return_value={}),
            ):
                result = validation_loop._launch_coding_agent(
                    workdir,
                    code_dir,
                    1,
                    "Name: Method",
                    0.8,
                    0.7,
                    [],
                    spec=spec,
                    success_criteria={"metric_name": "acc"},
                    supervisor_plan={"mode": "bootstrap"},
                )

        self.assertEqual(result["executor"], "codex")
        self.assertIn("Codex", result["description"])
        self.assertIn("codex_last_message", result["artifact_paths"])


if __name__ == "__main__":
    unittest.main()
