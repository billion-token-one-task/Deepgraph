from agents.paper_orchestra_pipeline import _submission_blockers_from_state


def _draft_ready_state(**packet_overrides):
    packet = {
        "formal_experiment": True,
        "smoke_test_only": False,
        "verdict": "confirmed",
        "evidence_tier": "benchmark_plan",
        "blocks_manuscript": True,
        "full_benchmark_completed": False,
        "artifact_paths": {"artifact_manifest": "/tmp/benchmark_artifact_manifest.json"},
        "benchmark_artifact_manifest": {
            "readiness_blockers": [
                "full_benchmark_completed is false.",
                "required baselines missing: Extra Baseline",
                "Only 1 seed(s) found; required minimum is 3.",
                "Required ablation table/results are missing.",
            ]
        },
        "quality_gates": {"requires_full_benchmark_package": True, "minimum_seeds": 3},
        "publication_evidence_contract": {
            "required_baselines": ["Extra Baseline"],
            "required_ablations": ["No component"],
            "minimum_seeds": 3,
        },
        "benchmark_summary": {
            "primary_metric": "accuracy",
            "num_seeds": 1,
            "per_method": {
                "baseline": {"accuracy": 0.7},
                "candidate": {"accuracy": 0.8},
            },
        },
    }
    packet.update(packet_overrides)
    return {
        "formal_experiment": True,
        "smoke_test_only": False,
        "result_packet": packet,
    }


def test_submission_blockers_allow_draft_with_soft_full_benchmark_gaps():
    assert _submission_blockers_from_state(_draft_ready_state()) == []


def test_submission_blockers_still_block_smoke_probe():
    state = _draft_ready_state(
        evidence_tier="bootstrap_probe",
        artifact_paths={},
        benchmark_summary={"per_method": {"candidate": {"accuracy": 0.8}}},
    )
    state["smoke_test_only"] = True

    blockers = _submission_blockers_from_state(state)

    assert "Run is not a formal non-smoke experiment." in blockers
    assert any("bootstrap_probe" in item for item in blockers)
    assert "benchmark_artifact_manifest.json is missing or not linked." in blockers
    assert "Benchmark summary must include at least two methods/baselines." in blockers
