import os
import json
import shutil
import unittest
import uuid
from unittest import mock
from pathlib import Path

from agents.manuscript_pipeline import generate_submission_bundle
from agents.paperorchestra.figure_orchestra import run_postwriting_api_figure_stage
from agents import workspace_layout
from db import database


class ManuscriptBundleTests(unittest.TestCase):
    def setUp(self):
        self._saved_pg_url = os.environ.pop("DEEPGRAPH_DATABASE_URL", None)
        self._old_database_url = database.DATABASE_URL
        test_tmp_root = Path(os.getenv("DEEPGRAPH_TEST_TMPDIR", Path.cwd() / "workspace" / "tmp" / "unit_tests"))
        test_tmp_root.mkdir(parents=True, exist_ok=True)
        self.tmpdir_path = test_tmp_root / f"test_vnext_manuscript_{uuid.uuid4().hex}"
        self.tmpdir_path.mkdir(parents=True, exist_ok=True)
        self.db_path = self.tmpdir_path / "test.db"
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
        self.workspace_root = self.tmpdir_path / "ideas"
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
                str(self.tmpdir_path / "run1"),
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
        self._write_complete_benchmark_packet()

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
        shutil.rmtree(self.tmpdir_path, ignore_errors=True)

    def _write_complete_benchmark_packet(self):
        run_dir = self.tmpdir_path / "run1"
        results_dir = run_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        per_method = {
            "Vanilla Direct Answering": {
                "metric_value": 0.50,
                "score": 0.55,
                "avg_new_tokens": 42.0,
                "avg_latency_seconds": 0.8,
                "route_rate": 0.0,
                "count": 1200,
            },
            "Always-Reason Chain-of-Thought": {
                "metric_value": 0.57,
                "score": 0.63,
                "avg_new_tokens": 180.0,
                "avg_latency_seconds": 2.4,
                "route_rate": 1.0,
                "count": 1200,
            },
            "Random Budget-Matched Routing": {
                "metric_value": 0.54,
                "score": 0.60,
                "avg_new_tokens": 108.0,
                "avg_latency_seconds": 1.4,
                "route_rate": 0.4,
                "count": 1200,
            },
            "Self-Consistency Reasoning": {
                "metric_value": 0.565,
                "score": 0.62,
                "avg_new_tokens": 210.0,
                "avg_latency_seconds": 2.6,
                "route_rate": 1.0,
                "count": 1200,
            },
            "Best-of-N Verifier": {
                "metric_value": 0.575,
                "score": 0.63,
                "avg_new_tokens": 190.0,
                "avg_latency_seconds": 2.2,
                "route_rate": 1.0,
                "count": 1200,
            },
            "Debate Vote": {
                "metric_value": 0.568,
                "score": 0.625,
                "avg_new_tokens": 240.0,
                "avg_latency_seconds": 2.9,
                "route_rate": 1.0,
                "count": 1200,
            },
            "Confidence Routing": {
                "metric_value": 0.55,
                "score": 0.61,
                "avg_new_tokens": 96.0,
                "avg_latency_seconds": 1.2,
                "route_rate": 0.35,
                "count": 1200,
            },
            "Disagreement Routing": {
                "metric_value": 0.56,
                "score": 0.62,
                "avg_new_tokens": 104.0,
                "avg_latency_seconds": 1.35,
                "route_rate": 0.38,
                "count": 1200,
            },
            "Oracle Routing": {
                "metric_value": 0.67,
                "score": 0.72,
                "avg_new_tokens": 130.0,
                "avg_latency_seconds": 1.8,
                "route_rate": 0.48,
                "count": 1200,
            },
            "Candidate Method": {
                "metric_value": 0.61,
                "score": 0.66,
                "avg_new_tokens": 110.0,
                "avg_latency_seconds": 1.5,
                "route_rate": 0.4,
                "count": 1200,
            },
        }
        datasets = [
            {
                "name": "GSM8K",
                "split": "test",
                "num_test": 1200,
                "num_materialized_examples": 1200,
                "preprocessing": "answer normalization",
                "license_or_source": "openai/gsm8k",
            }
        ]
        model = {
            "id": "Qwen/Qwen2.5-7B-Instruct",
            "name": "Qwen/Qwen2.5-7B-Instruct",
            "hardware": "NVIDIA L40S",
            "prompt_template": "direct and deliberate QA prompts",
            "decoding": {"default": "greedy"},
            "reasoning_budget": {"direct": 48, "deliberate": 192},
            "backend": "transformers",
        }
        benchmark_summary = {
            "primary_metric": "accuracy",
            "metric_name": "accuracy",
            "candidate_method": "Candidate Method",
            "per_method": per_method,
            "seed_results": [{"seed": i, "methods": {name: {"metric_value": row["metric_value"]} for name, row in per_method.items()}} for i in range(5)],
            "num_seeds": 5,
            "datasets": datasets,
            "dataset": datasets[0],
            "model": model,
            "hardware": "NVIDIA L40S",
            "latency_tokens_table": [
                {"method": name, "avg_new_tokens": row["avg_new_tokens"], "avg_latency_seconds": row["avg_latency_seconds"]}
                for name, row in per_method.items()
            ],
            "token_cost": {"per_method": {name: row["avg_new_tokens"] for name, row in per_method.items()}},
            "latency": {"per_method": {name: row["avg_latency_seconds"] for name, row in per_method.items()}},
            "ablation_table": [{"ablation": "no_guard", "method": "Candidate/no_guard", "metric_value": 0.58}],
            "bootstrap_ci": {"paired_permutation_p": 0.01, "candidate_ci95": [0.59, 0.63]},
            "route_rate": {"Candidate Method": 0.4, "Confidence Routing": 0.35, "Disagreement Routing": 0.38},
            "cost_saving": {"Candidate Method": 0.39, "relative_to_always_reason": 0.39},
            "easy_medium_hard_breakdown": {
                "easy": {"metric_value": 0.82, "simple_degradation": 0.0},
                "medium": {"metric_value": 0.62},
                "hard": {"metric_value": 0.41},
            },
            "difficulty_breakdown_table": [
                {"difficulty": "easy", "metric_value": 0.82, "simple_degradation": 0.0},
                {"difficulty": "medium", "metric_value": 0.62},
                {"difficulty": "hard", "metric_value": 0.41},
            ],
            "simple_case_degradation": {"easy": 0.0, "simple": 0.0, "degradation": 0.0},
            "calibration_reliability": {"ece": 0.04, "calibration": "temperature scaled confidence"},
            "routing_analysis": {
                "route_rate": 0.4,
                "cost_saving": 0.39,
                "easy_medium_hard_breakdown": "easy medium hard breakdown recorded",
                "simple_case_degradation": "simple degradation is zero on easy cases",
                "calibration": "calibration reliability checked with ECE 0.04",
            },
            "subset_analysis": {"severe_disagreement": {"num_examples": 300, "metric_gain": 0.06}},
            "quality_cost_frontier": {"pareto_methods": ["Confidence Routing", "Candidate Method"]},
            "live_sanity_check": {"datasets": ["GSM8K"], "num_examples": 200},
            "full_benchmark_completed": True,
        }
        manifest = {
            "full_benchmark_completed": True,
            "datasets": datasets,
            "model": model,
            "hardware": "NVIDIA L40S",
            "artifacts": {
                "benchmark_summary": str(results_dir / "benchmark_summary.json"),
                "main_results_table": str(results_dir / "main_results_table.json"),
                "ablation_table": str(results_dir / "ablation_table.json"),
                "latency_tokens_table": str(results_dir / "latency_tokens_table.json"),
                "run_log": str(run_dir / "run.log"),
            },
            "readiness_blockers": [],
        }
        contract = {
            "evidence_tier": "benchmark_plan",
            "blocks_manuscript": False,
            "minimum_seeds": 5,
            "required_real_benchmarks": ["GSM8K"],
            "required_models": ["Qwen/Qwen2.5-7B-Instruct"],
            "required_baselines": [
                "Vanilla Direct Answering",
                "Always-Reason Chain-of-Thought",
                "Confidence Routing",
                "Disagreement Routing",
                "Oracle Routing",
            ],
            "required_ablations": ["no_guard"],
            "primary_metric": "accuracy",
            "statistical_test": "paired bootstrap confidence interval plus paired permutation test",
            "quality_gates": {
                "has_real_benchmark": True,
                "requires_full_benchmark_package": True,
                "minimum_seeds": 5,
                "manuscript_allowed": True,
            },
            "claim_route": {"route": "full_paper", "paper_allowed": True},
        }
        packet = {
            "formal_experiment": True,
            "smoke_test_only": False,
            "verdict": "confirmed",
            "metric_name": "accuracy",
            "baseline": 0.5,
            "best": 0.61,
            "effect_pct": 22.0,
            "evidence_tier": "benchmark_plan",
            "blocks_manuscript": False,
            "full_benchmark_completed": True,
            "minimum_seeds": 5,
            "p_value": 0.01,
            "benchmark_summary": benchmark_summary,
            "benchmark_artifact_manifest": manifest,
            "artifact_paths": {"artifact_manifest": str(results_dir / "benchmark_artifact_manifest.json")},
            "publication_evidence_contract": contract,
            "claim_route": contract["claim_route"],
            "quality_gates": contract["quality_gates"],
            "claim_text": "The method improved accuracy.",
        }
        (results_dir / "benchmark_summary.json").write_text(json.dumps(benchmark_summary), encoding="utf-8")
        (results_dir / "benchmark_artifact_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (results_dir / "experiment_result_packet.json").write_text(json.dumps(packet), encoding="utf-8")
        (results_dir / "main_results_table.json").write_text(json.dumps(per_method), encoding="utf-8")
        (results_dir / "ablation_table.json").write_text(json.dumps(benchmark_summary["ablation_table"]), encoding="utf-8")
        (results_dir / "latency_tokens_table.json").write_text(json.dumps(benchmark_summary["latency_tokens_table"]), encoding="utf-8")

    def _stub_orchestra(self, state, literature_block, paper_ids, iterations, *, figures_dir, baseline, metric_name):
        figures_dir.mkdir(parents=True, exist_ok=True)
        assets = []
        for figure_id, label in (
            ("fig_main_benchmark", "main benchmark"),
            ("fig_ablation", "ablation"),
            ("fig_cost_quality", "cost quality"),
        ):
            fig_path = figures_dir / f"{figure_id}.png"
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(7, 4.2), dpi=220)
            ax.bar(["A", "B", "C"], [0.50, 0.57, 0.61], color=["#64748b", "#94a3b8", "#2563eb"])
            ax.set_ylim(0.0, 0.75)
            ax.set_title(label + " evidence panel with benchmark detail")
            ax.set_ylabel("accuracy")
            fig.tight_layout()
            fig.savefig(fig_path)
            plt.close(fig)
            assets.append(
                {
                    "figure_id": figure_id,
                    "path": str(fig_path),
                    "svg_path": "",
                    "pdf_path": "",
                    "code_path": "",
                    "objective": f"Show {label} evidence.",
                    "kind": "plot",
                }
            )
        body = " ".join(
            [
                "The problem is budgeted reasoning under disagreement, where a system must decide when extra candidate traces are worth the latency and token cost.",
                "The motivation is that fixed direct answering and fixed majority aggregation either miss useful evidence or spend compute uniformly.",
                "The method uses confidence, route, cost, disagreement, and calibration signals to select reasoning paths under a compute budget.",
                "The result is reported against four benchmark baselines with seed variation, ablation evidence, latency analysis, cost analysis, difficulty breakdown, and a disagreement subset.",
                "The limitations include a single-model setting, one primary benchmark family, and sensitivity to confidence calibration.",
            ]
            * 220
        )
        full_text = rf"""\documentclass{{article}}
\usepackage{{iclr2026_conference,times}}
\input{{math_commands.tex}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{array}}
\title{{Auto Manuscript Insight}}
\author{{Anonymous authors}}
\begin{{document}}
\maketitle
\begin{{abstract}}
This paper studies the problem of budgeted reasoning under disagreement. The challenge is to preserve useful minority evidence without paying keep-all token and latency cost. The proposed method routes candidate traces with confidence, disagreement, and cost signals. On GSM8K it improves the candidate accuracy to 0.61 while the strongest deployable baseline reaches 0.57, and the analysis reports ablation, cost, latency, and disagreement subset behavior.
\end{{abstract}}
\section{{Introduction}}
{body} \cite{{cite_a}}
\section{{Related Work}}
{body} Prior work on selective prediction, test-time compute, uncertainty routing, and multi-agent aggregation motivates the benchmark design \cite{{cite_a}}.
\section{{Method}}
{body}
\begin{{figure}}[t]
\centering
\includegraphics[width=\linewidth]{{figures/fig_main_benchmark.png}}
\caption{{Main benchmark comparison across four methods.}}
\label{{fig:main-benchmark}}
\end{{figure}}
\section{{Experiments}}
{body}
\begin{{table}}[t]
\centering
\begin{{tabular}}{{lrr}}
\toprule
Method & Accuracy & Tokens \\
\midrule
Direct & 0.50 & 42 \\
CoT & 0.57 & 180 \\
Random & 0.54 & 108 \\
Candidate & 0.61 & 110 \\
\bottomrule
\end{{tabular}}
\caption{{Main benchmark results.}}
\end{{table}}
\begin{{figure}}[t]
\centering
\includegraphics[width=\linewidth]{{figures/fig_ablation.png}}
\caption{{Ablation results for the routing guard.}}
\label{{fig:ablation}}
\end{{figure}}
\section{{Results}}
{body} The ablation without the guard reaches 0.58, confirming that the routing guard matters. The latency and cost analysis shows the candidate keeps accuracy gains while using fewer tokens than always-reasoning. The disagreement subset analysis isolates examples where candidates disagree and confirms the method keeps high-confidence minority evidence.
\begin{{table}}[t]
\centering
\begin{{tabular}}{{lrr}}
\toprule
Analysis & Accuracy & Latency \\
\midrule
No guard ablation & 0.58 & 1.3 \\
Candidate & 0.61 & 1.5 \\
\bottomrule
\end{{tabular}}
\caption{{Ablation and latency/cost analysis.}}
\end{{table}}
\begin{{figure}}[t]
\centering
\includegraphics[width=\linewidth]{{figures/fig_cost_quality.png}}
\caption{{Cost-quality tradeoff and disagreement subset behavior.}}
\label{{fig:cost-quality}}
\end{{figure}}
\section{{Discussion}}
{body} These results support the method while preserving clear limitations about benchmark scope, calibration, and model coverage.
\bibliographystyle{{iclr2026_conference}}
\bibliography{{references}}
\end{{document}}
"""
        return {
            "outline": {"plotting_plan": [{"figure_id": asset["figure_id"], "plot_type": "plot"} for asset in assets]},
            "plotting": {
                "figure_captions": [{"figure_id": asset["figure_id"], "caption": asset["objective"]} for asset in assets],
                "plotting_executor": {"assets": assets},
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
            "refinement_full_text": full_text,
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
        self.assertTrue((bundle_path / "paper_intent.json").exists())
        self.assertTrue((bundle_path / "problem_awareness.json").exists())
        self.assertTrue((bundle_path / "publication_evidence_contract.json").exists())
        self.assertTrue((bundle_path / "evidence_manifest.json").exists())
        self.assertTrue((bundle_path / "claim_evidence_matrix.json").exists())
        self.assertTrue((bundle_path / "reviewer_report.json").exists())
        self.assertTrue((bundle_path / "latex_sanity_report.json").exists())
        self.assertTrue((bundle_path / "citation_audit.json").exists())
        self.assertTrue((bundle_path / "figures" / "figure_manifest.json").exists())
        self.assertTrue((bundle_path / "iclr2026_conference.sty").exists())
        self.assertTrue((bundle_path / "iclr2026_conference.bst").exists())
        self.assertTrue((bundle_path / "math_commands.tex").exists())
        self.assertIn("cite_a", (bundle_path / "references.bib").read_text(encoding="utf-8"))
        main_tex = (bundle_path / "main.tex").read_text(encoding="utf-8")
        self.assertIn("iclr2026_conference", main_tex)
        self.assertNotIn("fig_metric_trajectory", main_tex)
        self.assertNotIn("fig_search_dynamics_keep_discard", main_tex)
        self.assertNotIn("fig_benchmark_method_panel", main_tex)
        self.assertTrue((self.workspace_root / "idea_1" / "paper" / "current" / "main.tex").exists())

    def test_generate_submission_bundle_blocks_non_formal_run(self):
        database.execute(
            "UPDATE experiment_runs SET proxy_config=? WHERE id=1",
            (json.dumps({"formal_experiment": False, "smoke_test_only": True}),),
        )
        database.commit()
        packet_path = self.tmpdir_path / "run1" / "results" / "experiment_result_packet.json"
        packet = json.loads(packet_path.read_text(encoding="utf-8"))
        packet["formal_experiment"] = False
        packet["smoke_test_only"] = True
        packet_path.write_text(json.dumps(packet), encoding="utf-8")

        result = generate_submission_bundle(1, bundle_formats=["conference"])

        self.assertIn("error", result)
        self.assertIn("formal", result["error"].lower())
        layout = workspace_layout.get_idea_workspace(1, create=True)
        current_root = Path(layout["paper_current_root"])
        self.assertTrue((current_root / "MANUSCRIPT_BLOCKED.json").exists())
        self.assertTrue((current_root / "DO_NOT_SUBMIT.md").exists())

    def test_generate_submission_bundle_blocks_benchmark_plan_without_artifact_manifest(self):
        database.execute(
            "UPDATE experiment_runs SET success_criteria=? WHERE id=1",
            (
                json.dumps(
                    {
                        "metric_name": "accuracy",
                        "publication_evidence_contract": {
                            "evidence_tier": "benchmark_plan",
                            "blocks_manuscript": False,
                            "required_real_benchmarks": ["GSM8K"],
                            "minimum_seeds": 3,
                            "quality_gates": {
                                "has_real_benchmark": True,
                                "requires_full_benchmark_package": True,
                                "minimum_seeds": 3,
                            },
                        },
                    }
                ),
            ),
        )
        database.commit()
        results_dir = self.tmpdir_path / "run1" / "results"
        for name in ("benchmark_artifact_manifest.json", "experiment_result_packet.json"):
            path = results_dir / name
            if path.exists():
                path.unlink()

        result = generate_submission_bundle(1, bundle_formats=["conference"])

        self.assertIn("error", result)
        self.assertIn("full benchmark", result["error"].lower())

    @mock.patch("agents.paper_orchestra_pipeline._run_full_pipeline")
    def test_generate_submission_bundle_blocks_placeholder_figure_assets(self, run_full):
        def _stub_with_placeholder(state, literature_block, paper_ids, iterations, *, figures_dir, baseline, metric_name):
            out = self._stub_orchestra(
                state,
                literature_block,
                paper_ids,
                iterations,
                figures_dir=figures_dir,
                baseline=baseline,
                metric_name=metric_name,
            )
            (figures_dir / "fig_main_benchmark.png").write_text(
                "Diagram placeholder: failed API figure.",
                encoding="utf-8",
            )
            return out

        run_full.side_effect = _stub_with_placeholder

        result = generate_submission_bundle(1, bundle_formats=["conference"])

        self.assertIn("error", result)
        self.assertIn("placeholder", " ".join(result.get("submission_blockers") or []).lower())
        current_root = Path(workspace_layout.get_idea_workspace(1, create=True)["paper_current_root"])
        self.assertTrue((current_root / "MANUSCRIPT_BLOCKED.json").exists())
        self.assertTrue((current_root / "DO_NOT_SUBMIT.md").exists())

    def test_postwriting_api_figure_stage_blocks_without_command(self):
        figures_dir = self.tmpdir_path / "figures"
        result = run_postwriting_api_figure_stage(
            {},
            {"problem_awareness": {"central_question": "What is being tested?"}},
            "\\section{Introduction}Draft paper.",
            figures_dir,
            paperbanana_cmd="",
        )
        self.assertEqual(result["stage"], "postwriting_api_figures")
        self.assertTrue(result["required"])
        self.assertEqual(result["generated_count"], 0)
        self.assertTrue(result["blockers"])
        self.assertTrue((figures_dir / "postwriting_api_figure_manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
