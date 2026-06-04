import json
import subprocess
import sys

from agents.benchmark_artifacts import materialize_deep_benchmark_artifacts
from agents.paper_orchestra_pipeline import assemble_main_tex


FORBIDDEN_MODULES = {
    "torch",
    "transformers",
    "vllm",
    "agents.experiment_forge",
    "agents.experiment_executor",
}


def test_m5_importing_presentation_modules_does_not_load_gpu_or_execution_modules():
    script = (
        "import sys\n"
        "import agents.paper_orchestra_pipeline\n"
        "import agents.paper_completeness\n"
        f"forbidden = {sorted(FORBIDDEN_MODULES)!r}\n"
        "loaded = [name for name in forbidden if name in sys.modules]\n"
        "print('\\n'.join(loaded))\n"
        "raise SystemExit(1 if loaded else 0)\n"
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=".",
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == ""


def test_m5_assemble_main_tex_renders_offline_without_llm_or_gpu_modules():
    state = {
        "title": "CPU Boundary Paper",
        "baseline_metric_name": "exact_match",
        "baseline_metric_value": 0.5,
        "best_metric_value": 0.61,
        "effect_pct": 22.0,
        "verdict": "confirmed",
        "problem_statement": "Can a deterministic renderer run offline?",
        "method_summary": "A presentation-only method summary.",
        "problem_awareness": {
            "central_question": "Can rendering run without benchmark execution?",
            "motivation": "Presentation gates must stay CPU-only.",
            "method_answer": "Use already materialized evidence.",
            "result_claim": "The renderer emits LaTeX from snapshots.",
        },
        "contributions": ["A CPU-only rendering path."],
        "benchmark_summary": {
            "primary_metric": "exact_match",
            "per_method": {
                "Candidate": {"exact_match": 0.61, "n": 20},
                "Baseline": {"exact_match": 0.50, "n": 20},
            },
        },
    }
    orchestrated = {
        "refined": {
            "abstract": "Offline abstract.",
            "introduction": "Offline introduction.",
            "method": "Offline method.",
            "experiments": "Offline experiments.",
            "discussion": "Offline discussion.",
        },
        "plotting": {"plotting_executor": {"assets": []}},
    }

    main_tex = assemble_main_tex(state, orchestrated, "conference")

    assert "\\documentclass" in main_tex
    assert "CPU Boundary Paper" in main_tex
    assert "\\section{Experiments}" in main_tex
    assert not (FORBIDDEN_MODULES & set(sys.modules))


def test_m5_materialize_raw_predictions_to_cpu_artifacts(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    methods = ["CPG", "Baseline", "CPG/no_guard"]
    datasets = ["GSM8K", "StrategyQA"]
    with (results_dir / "raw_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for method in methods:
            for dataset in datasets:
                for seed in range(3):
                    for _ in range(7):
                        score = 0.72 if method == "CPG" else 0.61 if method == "Baseline" else 0.66
                        row = {
                            "method": method,
                            "dataset": dataset,
                            "seed": seed,
                            "exact_match": score,
                        }
                        handle.write(json.dumps(row) + "\n")

    report = materialize_deep_benchmark_artifacts(
        results_dir,
        publication_contract={"required_ablations": ["CPG/no_guard"]},
        metric_name="exact_match",
        min_lines=100,
    )

    assert report["ok"] is True
    for name in (
        "benchmark_summary.json",
        "main_results_table.json",
        "seed_variance_table.json",
        "per_dataset_results.json",
        "ablation_table.json",
    ):
        assert (results_dir / name).exists()
    summary = json.loads((results_dir / "benchmark_summary.json").read_text(encoding="utf-8"))
    assert summary["primary_metric"] == "exact_match"
    assert summary["per_method"]["CPG"]["exact_match"] == 0.72
    assert summary["per_method"]["CPG"]["n"] == 42
    assert summary["seed_variance"]["CPG"]["n_seeds"] == 3
    assert summary["seed_variance"]["CPG"]["per_seed"] == {"0": 0.72, "1": 0.72, "2": 0.72}
    assert summary["ablations"]["CPG/no_guard"]["executed"] is True
