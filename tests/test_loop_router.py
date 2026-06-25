import json
from unittest import mock

from agents.benchmark_harness_loop import prepare_harness_loop_task
from agents.benchmark_manager import build_harness_task
from agents.loop_router import route_blockers
from orchestrator import auto_research


def test_routes_dataset_materialization_before_generic_harness():
    report = route_blockers(
        [
            "metadata probe failed: dataset birdsql/bird does not exist or cannot access official BIRD files",
            "Generated real-benchmark runner does not support text-to-SQL; dedicated benchmark harness required.",
        ],
        context={"source": "experiment_review"},
    )

    assert report["primary_owner"] == "Dataset Fetch Agent"
    assert report["primary_stage"] == "dataset_materialization"
    assert report["owner_routes"][0]["categories"] == ["dataset_materialization"]


def test_routes_benchmark_evidence_to_completion_runner():
    report = route_blockers(
        [
            "benchmark_artifact_manifest.full_benchmark_completed is false",
            "Required ablation table/results are missing.",
        ],
        context={"source": "manuscript_quality_gate"},
    )

    assert report["primary_owner"] == "Benchmark Completion Runner"
    assert report["primary_stage"] == "benchmark_completion_required"
    assert "baselines, seeds, ablations" in report["primary_action"]


def test_harness_loop_normalizes_legacy_cifar_hf_id():
    task = prepare_harness_loop_task(
        {
            "schema_version": "benchmark_harness_task_v1",
            "dataset_refs": [
                {
                    "name": "CIFAR-10",
                    "hf_dataset": "cifar10",
                    "requires_harness": True,
                    "generated_runner_supported": False,
                }
            ],
        },
        benchmark_name="CIFAR-10",
    )

    ref = task["dataset_refs"][0]
    assert ref["name"] == "CIFAR-10"
    assert ref["hf_dataset"] == ""
    assert ref["direct_files"][0]["url"].endswith("cifar-10-python.tar.gz")
    assert task["loop_state"]["status"] == "dataset_materialization_required"
    assert task["loop_state"]["owner"] == "Dataset Fetch Agent"


def test_harness_loop_moves_materialized_harness_to_code_agent():
    task = prepare_harness_loop_task(
        {
            "schema_version": "benchmark_harness_task_v1",
            "dataset_refs": [
                {
                    "name": "CIFAR-10",
                    "direct_files": [{"url": "https://example.test/cifar.tar.gz"}],
                    "requires_harness": True,
                    "num_materialized_examples": 10000,
                    "dataset_cache_verified": True,
                }
            ],
        },
        benchmark_name="CIFAR-10",
    )

    assert task["loop_state"]["status"] == "benchmark_harness_code_required"
    assert task["loop_state"]["owner"] == "Benchmark Harness Code Agent"


def test_harness_task_records_loop_router_for_dataset_blocker():
    task = build_harness_task(
        {
            "id": 91,
            "tier": 2,
            "title": "Text-to-SQL schema repair",
            "problem_statement": "The claim needs text-to-SQL database benchmarks.",
            "proposed_method": {"name": "Schema Repair"},
            "experimental_plan": {
                "datasets": [{"name": "BIRD", "task_type": "text_to_sql"}],
                "benchmark_recipe_blockers": [
                    "metadata probe failed: dataset birdsql/bird does not exist; use official BIRD data and DB files"
                ],
            },
        },
        judgement_payload={
            "judgement": {
                "summary": "Generated real-benchmark runner does not support BIRD until the official dataset is materialized.",
                "blockers": [
                    "Dataset materialization is missing for official BIRD database files."
                ],
                "warnings": [],
            }
        },
        source="unit_test",
    )

    assert task["loop_router"]["primary_owner"] == "Dataset Fetch Agent"
    assert task["loop_router"]["primary_stage"] == "dataset_materialization"


def test_unrecovered_harness_job_writes_loop_state():
    row = {
        "id": 7,
        "deep_insight_id": 91,
        "benchmark_name": "BIRD",
        "last_error": "metadata probe failed: dataset birdsql/bird does not exist",
        "last_note": "",
        "task_plan": json.dumps(
            {
                "schema_version": "benchmark_harness_task_v1",
                "dataset_refs": [
                    {
                        "name": "BIRD Dev",
                        "task_type": "text_to_sql",
                        "hf_dataset": "birdsql/bird",
                        "requires_harness": True,
                    }
                ],
                "recipe_blockers": [
                    "metadata probe failed: dataset birdsql/bird does not exist"
                ],
                "review_judgement": {
                    "summary": "Generated real-benchmark runner does not support BIRD; official dataset materialization is missing.",
                    "blockers": ["Dataset materialization is missing for BIRD."],
                },
            }
        ),
    }

    calls = []

    def _capture_execute(sql, params=()):
        calls.append((sql, params))

    with (
        mock.patch.object(auto_research.db, "execute", side_effect=_capture_execute),
        mock.patch.object(auto_research.db, "commit"),
        mock.patch.object(auto_research, "write_plan_files", return_value={}) as write_plan,
    ):
        changed = auto_research._annotate_unrecovered_harness_job(row)

    assert changed is True
    payload = json.loads(calls[0][1][0])
    dataset_refs = json.loads(calls[0][1][1])
    assert payload["loop_router"]["primary_owner"] == "Dataset Fetch Agent"
    assert payload["loop_state"]["stage"] == "dataset_materialization"
    assert payload["loop_state"]["status"] == "dataset_materialization_required"
    assert payload["dataset_materialization_plan"]["status"] == "dataset_materialization_required"
    assert dataset_refs[0]["name"] == "BIRD"
    assert dataset_refs[0]["hf_dataset"] == ""
    assert "bird-bench" in dataset_refs[0]["official_url"]
    assert "Loop owner: Dataset Fetch Agent" in calls[0][1][3]
    write_plan.assert_called_once()
    written_files = write_plan.call_args.kwargs["files"]
    assert written_files["benchmark_harness_status.json"]["loop_state"]["stage"] == "dataset_materialization"
    assert written_files["dataset_materialization_status.json"]["status"] == "dataset_materialization_required"
