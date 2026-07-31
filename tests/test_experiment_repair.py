from unittest import mock

from agents import experiment_forge


def test_repair_experiment_plan_merges_llm_patch_and_persists():
    parsed = {
        "id": 101,
        "agenda_id": 7,
        "title": "Repair me",
        "problem_statement": "Need a benchmark.",
        "proposed_method": {"name": "Method", "definition": "f(x)"},
        "experimental_plan": {
            "datasets": [],
            "baselines": [],
            "metrics": {},
            "compute_budget": {},
        },
    }
    persisted = []

    def enrich_plan(p, method, plan):
        plan = dict(plan)
        plan["enriched"] = True
        return plan

    with (
        mock.patch.object(
            experiment_forge.db,
            "fetchone",
            side_effect=[
                dict(parsed),
                {
                    "id": 44,
                    "agenda_id": 7,
                    "idea_id": 101,
                    "stage": "experiment_repair",
                    "status": "active",
                    "expires_at": "2099-01-01T00:00:00+00:00",
                },
            ],
        ),
        mock.patch.object(experiment_forge, "_resource_granted_proposer_json", return_value=(
            {
                "repair_summary": "Added GSM8K and Qwen.",
                "experimental_plan_patch": {
                    "datasets": [{"name": "GSM8K", "split": "test"}],
                    "model_targets": [{"name": "Qwen", "hf_model": "Qwen/Qwen2.5-3B-Instruct"}],
                    "baselines": [{"name": "Direct"}, {"name": "Self-consistency"}],
                    "metrics": {"primary": "accuracy"},
                    "compute_budget": {"total_gpu_hours": 4},
                    "minimum_seeds": 3,
                },
            },
            123,
            {
                "provider": "provider-a",
                "model": "model-a",
                "model_family": "family-a",
                "prompt_version": "repair-v1",
            },
        )),
        mock.patch.object(experiment_forge, "_enrich_proposed_method", side_effect=lambda p, plan: p["proposed_method"]),
        mock.patch.object(experiment_forge, "_finalize_repaired_experiment_plan", side_effect=enrich_plan),
        mock.patch.object(experiment_forge, "_persist_enriched_insight", side_effect=lambda insight_id, p: persisted.append((insight_id, p))),
    ):
        result = experiment_forge.repair_experiment_plan_from_review(
            101,
            judgement={"summary": "blocked", "blockers": ["missing dataset"], "warnings": []},
            attempt=1,
            resource_grant_id=44,
        )

    assert result["status"] == "repaired"
    assert result["llm_repair_used"] is True
    assert persisted[0][0] == 101
    plan = persisted[0][1]["experimental_plan"]
    assert plan["datasets"][0]["name"] == "GSM8K"
    assert plan["metrics"]["primary"] == "accuracy"
    assert plan["minimum_seeds"] == 3
    assert plan["enriched"] is True
    assert result["llm_route"]["provider"] == "provider-a"


def test_repair_experiment_plan_fails_closed_without_active_grant():
    insight = {
        "id": 101,
        "agenda_id": 7,
        "proposed_method": {"name": "Method"},
        "experimental_plan": {},
    }
    with (
        mock.patch.object(
            experiment_forge.db,
            "fetchone",
            side_effect=[insight, None],
        ),
        mock.patch.object(
            experiment_forge,
            "_persist_enriched_insight",
        ) as persist,
    ):
        result = experiment_forge.repair_experiment_plan_from_review(
            101,
            resource_grant_id=44,
        )

    assert result["route"] == "manual_review_required"
    persist.assert_not_called()
