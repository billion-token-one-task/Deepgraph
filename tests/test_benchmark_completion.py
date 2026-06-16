from orchestrator.benchmark_completion import benchmark_completion_blockers


def test_soft_full_benchmark_gaps_do_not_trigger_blocking_completion():
    bundle = {
        "error": "benchmark_summary.full_benchmark_completed is false",
        "submission_blockers": [
            "required baselines missing: Extra Baseline",
            "Only 1 seed(s) found; required minimum is 3.",
            "Required ablation table/results are missing.",
            "full benchmark policy: required model coverage missing: DeBERTa",
        ],
    }

    assert benchmark_completion_blockers(bundle) == []


def test_hard_benchmark_gaps_trigger_blocking_completion():
    bundle = {
        "submission_blockers": [
            "benchmark_artifact_manifest.json is missing or not linked.",
            "Benchmark summary must include at least two methods/baselines.",
        ]
    }

    assert benchmark_completion_blockers(bundle) == [
        "benchmark_artifact_manifest.json is missing or not linked.",
        "Benchmark summary must include at least two methods/baselines.",
    ]
