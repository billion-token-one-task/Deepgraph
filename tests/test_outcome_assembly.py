from unittest import mock

from meta_harness import repository as repository_module
from meta_harness.repository import MetaHarnessRepository


def _grant():
    return {
        "id": 5,
        "agenda_id": 2,
        "idea_id": 3,
        "decision_packet_id": 4,
        "estimates_json": (
            '{"success_probability":{"value":0.6},'
            '"expected_token_cost":{"value":100},'
            '"expected_gpu_cost":{"value":0},'
            '"expected_impact":{"value":0.2}}'
        ),
    }


def _run():
    return {
        "id": 6,
        "agenda_id": 2,
        "deep_insight_id": 3,
        "status": "completed",
        "scientific_evidence_state": "full_benchmark_complete",
        "baseline_metric_value": 0.5,
        "effect_size": 0.1,
        "hypothesis_verdict": "supported",
    }


def test_trusted_outcome_assembly_does_not_promote_operational_support():
    repo = MetaHarnessRepository()
    with (
        mock.patch.object(
            repository_module.db,
            "fetchone",
            side_effect=[
                _grant(),
                _run(),
                {"tokens_used": 80, "open_reservations": 0},
                None,
            ],
        ),
        mock.patch.object(
            repository_module.db,
            "fetchall",
            side_effect=[
                [],
                [{"id": 9, "artifact_type": "source_data", "path": "run/a.json"}],
                [{"id": 10, "role": "proposer", "status": "succeeded"}],
            ],
        ),
        mock.patch.object(repo, "record_outcome", return_value=12) as record,
    ):
        outcome_id = repo.assemble_and_record_outcome(
            resource_grant_id=5,
            experiment_run_id=6,
        )

    assert outcome_id == 12
    outcome = record.call_args.args[0]
    assert outcome.verdict == "inconclusive"
    assert outcome.actual_tokens == 80
    assert outcome.actual_gpu_hours == 0
    assert outcome.experiment_run_id == 6
    assert outcome.prediction_error["token_cost"] == -20
    assert outcome.artifact_manifest["source"] == "trusted_persistence_v1"


def test_trusted_outcome_assembly_uses_canonical_supported_decision():
    repo = MetaHarnessRepository()
    run = _run()
    run["scientific_evidence_state"] = "scientifically_decided"
    with (
        mock.patch.object(
            repository_module.db,
            "fetchone",
            side_effect=[
                _grant(),
                run,
                {"tokens_used": 80, "open_reservations": 0},
                {
                    "id": 11,
                    "verdict": "supported",
                    "evidence_decision_json": '{"reason_codes":["complete_evidence"]}',
                },
            ],
        ),
        mock.patch.object(
            repository_module.db,
            "fetchall",
            side_effect=[[], [], []],
        ),
        mock.patch.object(repo, "record_outcome", return_value=13) as record,
    ):
        repo.assemble_and_record_outcome(
            resource_grant_id=5,
            experiment_run_id=6,
        )

    outcome = record.call_args.args[0]
    assert outcome.verdict == "supported"
    assert outcome.new_information["decision_reason_codes"] == [
        "complete_evidence"
    ]
