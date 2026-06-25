from orchestrator.benchmark_completion import benchmark_completion_blockers


def test_full_benchmark_gaps_trigger_completion():
    bundle = {
        "error": "benchmark_summary.full_benchmark_completed is false",
        "submission_blockers": [
            "required baselines missing: Extra Baseline",
            "Only 1 seed(s) found; required minimum is 3.",
            "Required ablation table/results are missing.",
            "full benchmark policy: required model coverage missing: DeBERTa",
        ],
    }

    assert benchmark_completion_blockers(bundle) == [
        "required baselines missing: Extra Baseline",
        "Only 1 seed(s) found; required minimum is 3.",
        "Required ablation table/results are missing.",
        "full benchmark policy: required model coverage missing: DeBERTa",
        "benchmark_summary.full_benchmark_completed is false",
    ]


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

def test_quality_report_experiment_advisories_trigger_completion_when_submission_blockers_are_writing_only():
    bundle = {
        "error": "Manuscript quality gate failed",
        "submission_blockers": [
            "Writing structure: The manuscript is missing required conference-paper sections.",
            "Reference auditor / metadata: Bibliography entry lacks venue and DOI/arXiv/URL identifiers.",
        ],
        "quality_report": {
            "writing_guideline_audit": {
                "experiment_scope_advisories": [
                    {
                        "severity": "high",
                        "standard": "Evidence gate",
                        "issue": "Full benchmark evidence is not complete; manuscript cannot be bundle_ready.",
                    },
                    {
                        "severity": "high",
                        "standard": "Benchmark/baseline requirement",
                        "issue": "Benchmark comparison does not cover the required baseline set.",
                        "evidence": "per_method_count=4 required_baselines=12",
                    },
                ]
            }
        },
    }

    blockers = benchmark_completion_blockers(bundle)

    assert blockers == [
        "Evidence gate: Full benchmark evidence is not complete; manuscript cannot be bundle_ready.",
        "Benchmark/baseline requirement: Benchmark comparison does not cover the required baseline set. (per_method_count=4 required_baselines=12)",
    ]


def test_quality_report_path_triggers_completion(tmp_path):
    quality_report = tmp_path / "paper_quality_report.json"
    quality_report.write_text(
        '{"writing_guideline_audit": {"experiment_scope_advisories": '
        '[{"standard": "Evidence gate", '
        '"issue": "Quality gate requires full benchmark package, but summary is not marked complete."}]}}',
        encoding="utf-8",
    )

    assert benchmark_completion_blockers({"quality_report": str(quality_report)}) == [
        "Evidence gate: Quality gate requires full benchmark package, but summary is not marked complete."
    ]


def test_benchmark_evidence_passed_note_is_not_a_completion_blocker():
    bundle = {
        "submission_blockers": [
            "Only manuscript polish blockers remain after benchmark evidence passed."
        ]
    }

    assert benchmark_completion_blockers(bundle) == []
