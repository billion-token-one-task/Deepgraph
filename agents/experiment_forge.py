"""Experiment Forge: bridge from deep_insights to runnable experiments.

Three sub-components:
  2a. Code Scout — find/clone relevant codebases for a hypothesis
  2b. Scaffold Builder — generate program.md, evaluate.py, success_criteria.json
  2c. Proxy Task Builder — configure time-budgeted experiment for fast iteration

This is the hardest layer: translating a structured method description
into something an autonomous coding agent can actually run.
"""
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import textwrap
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

from agents.discovery_metadata import infer_resource_class
from agents.benchmark_design_agent import (
    DESIGN_STATUS_RESOLVED,
    apply_benchmark_design_contract,
    build_benchmark_design_contract,
)
from agents.benchmark_protocol import resolve_benchmark_protocol
from agents.dataset_resolver import resolve_plan_datasets
from agents.evidence_planner import build_evidence_plan
from agents.evosci_requirements import evosci_strict_gate_insight
from agents.experiment_review import review_experiment_candidate
from agents.idea_route import classify_idea_route
from agents.llm_client import (
    call_llm_json_for_role,
    configured_role_prompt_version,
)
from agents.stage_prompts import prompt_block
from agents.workspace_layout import (
    ensure_run_workspace,
    get_idea_workspace,
    promote_canonical_run,
    write_latest_status,
    write_plan_files,
)
from config import (
    EXPERIMENT_EARLY_STOP_THRESHOLD,
    EXPERIMENT_ALLOW_SYNTHETIC_FALLBACK,
    EXPERIMENT_FULL_BENCHMARK_MIN_BASELINES,
    EXPERIMENT_FULL_BENCHMARK_MIN_DATASETS,
    EXPERIMENT_FULL_BENCHMARK_MIN_EXAMPLES,
    EXPERIMENT_FULL_BENCHMARK_MIN_MODELS,
    EXPERIMENT_FULL_BENCHMARK_REQUIRE_SIGNIFICANCE,
    EXPERIMENT_FULL_BENCHMARK_REQUIRE_STRONGEST_WIN,
    EXPERIMENT_MAX_ITERATIONS,
    EXPERIMENT_PROXY_DATA_FRACTION,
    EXPERIMENT_PROXY_MAX_EPOCHS,
    EXPERIMENT_REFUTE_MIN_ITERS,
    EXPERIMENT_REPRODUCTION_ITERS,
    EXPERIMENT_REAL_BENCHMARK_DATASET,
    EXPERIMENT_REAL_BENCHMARK_DATASET_CONFIG,
    EXPERIMENT_REAL_BENCHMARK_MAX_EXAMPLES,
    EXPERIMENT_REAL_BENCHMARK_SEEDS,
    EXPERIMENT_REAL_BENCHMARK_TIME_BUDGET,
    EXPERIMENT_REAL_LLM_MODEL,
    EXPERIMENT_REQUIRE_REAL_BENCHMARK,
    EXPERIMENT_TIME_BUDGET,
    GPU_DEFAULT_MODEL,
    GPU_DEFAULT_VRAM_GB,
    GPU_VISIBLE_DEVICES,
)
from contracts import DeepInsightSpec, ExperimentSpec
from db import database as db
from db.insight_outcomes import apply_experiment_queued_deep


SCAFFOLD_SYSTEM = prompt_block(
    "experiment_contract_architect",
    "sanity_runner_builder",
    "full_benchmark_compiler",
) + """

You are an expert ML engineer. Given a research hypothesis with a proposed method, you produce THREE files that enable an autonomous coding agent to run experiments.

You will receive:
1. A proposed method (name, type, definition, pseudocode, properties)
2. An experimental plan (baselines, datasets, metrics, ablations)
3. A codebase description (what repo was cloned, its structure)

You must output JSON with three keys:

{
  "program_md": "Complete program.md content in Markdown (instructions for the coding agent)",
  "evaluate_py": "Complete evaluate.py Python script (metric computation)",
  "success_criteria": {
    "metric_name": "primary metric name",
    "metric_direction": "lower|higher",
    "exciting": <number>,
    "solid": <number>,
    "disappointing": <number>,
    "publication_evidence_contract": {
      "claim_to_validate": "one sentence scientific claim",
      "evidence_tier": "benchmark_plan|formal_proxy|bootstrap_probe",
      "claim_route": {"route": "full_paper|workshop|research_note|probe|blocked"},
      "blocks_manuscript": <bool>,
      "minimum_seeds": <int>,
      "required_datasets": ["..."],
      "required_baselines": ["..."],
      "required_ablations": ["..."],
      "statistical_test": "...",
      "required_artifacts": ["main_results_table", "ablation_table", "..."],
      "reviewer_objections": ["..."],
      "problem_awareness": {
        "central_question": "...",
        "motivation": "...",
        "method_answer": "...",
        "result_claim": "...",
        "falsification_result": "..."
      },
      "paper_intent": {
        "central_claim": "...",
        "reader_takeaway": "...",
        "narrative_spine": ["gap", "method", "evidence", "limitation"]
      }
    }
  }
}

## program.md Requirements
- Must follow the autoresearch format: setup, experimentation loop, output format, logging
- MUST specify which file(s) the agent can modify
- MUST describe the proposed method clearly enough for implementation
- MUST include the baseline to beat and specific success criteria
- MUST include the evaluation command
- MUST tell the agent to NEVER STOP until interrupted
- MUST use real benchmark data and real model execution. Do not replace the
  planned experiment with synthetic data, random tensors, mocked examples, or
  a pure CUDA memory probe.
- MUST include a "Publication Evidence Contract" section that names:
  datasets, baselines, ablations, seed count, statistical test, expected tables/figures,
  and the exact manuscript claim this experiment is allowed to support.
- MUST include a "Problem Awareness" section that states the research question,
  motivation, method answer, result claim, and falsification result before coding starts.
- MUST instruct the coding agent to run baseline(s), the proposed method, required
  ablations, and at least the requested number of seeds when the time budget allows.

## evaluate.py Requirements
- Self-contained Python script
- Takes a log file or results directory as input
- Outputs the primary metric value to stdout
- Handles errors gracefully (outputs 0.0 on failure)

## success_criteria Requirements
- Use the primary metric from the experimental plan
- exciting = would be a strong contribution (top-venue accept)
- solid = clear improvement over baseline
- disappointing = not worth publishing
- Include publication_evidence_contract. Synthetic smoke/probe scaffolds must set
  evidence_tier="bootstrap_probe" and blocks_manuscript=true.
- Include problem_awareness both inside success_criteria and inside
  publication_evidence_contract so the writing pipeline can preserve the paper's
  question-motivation-method-result spine.
- Do not mark a proxy, smoke, synthetic-only, or CUDA bootstrap experiment as a
  submission-ready benchmark result.

## CRITICAL: train.py (bootstrap code)
If the codebase is "scratch" or empty, you MUST also output a "train_py" key containing a COMPLETE, RUNNABLE Python script that:
- Loads the named real benchmark dataset or a documented public benchmark fallback.
- Loads a real model from the model targets (for LLM tasks, use Hugging Face Transformers/vLLM/API backends).
- Runs baseline method(s) and the proposed method on real benchmark examples.
- Emits a FINAL_RESULTS JSON line with per_method, seed_results, candidate_method,
  best_method, primary_metric, num_seeds, and dataset/model metadata.
- Never uses synthetic/simulated/random examples unless the deployment explicitly
  opts into smoke tests outside formal validation.

When Resource class is gpu_small or gpu_large:
- train_py MUST use PyTorch CUDA (torch.cuda.is_available, tensors/models on cuda)
- train_py MUST print peak_vram_mb and a FINAL_RESULTS JSON line
- train_py MUST NOT be a numpy/scipy-only toy script
- gpu_large scripts should run the actual model/benchmark path. If the full
  model cannot fit or dependencies are missing, fail with an actionable error;
  do not replace it with a VRAM probe.

When Resource class is cpu:
- train_py may use stdlib + numpy + scipy and must not require CUDA.

For framework/evaluation-type methods (not model training), train.py should:
- Generate synthetic test scenarios
- Run the baseline evaluation approach
- Print the primary metric"""


CODE_SCOUT_SYSTEM = prompt_block("code_scout") + """

You are a research engineer. Given a method description and its related taxonomy area, suggest the BEST open-source codebase to use as a starting point for implementing and testing this method.

Return JSON:
{
  "codebase": {
    "url": "GitHub URL (full https://github.com/...)",
    "name": "short name",
    "reason": "why this is the best base",
    "setup_commands": ["pip install ...", "python setup.py ..."],
    "main_train_file": "path/to/train.py (the file to modify)",
    "main_eval_command": "python evaluate.py --args",
    "expected_baseline_metric": "approximate value"
  },
  "alternatives": [
    {"url": "...", "name": "...", "reason": "..."}
  ]
}

Prefer:
- Well-maintained repos (recent commits, many stars)
- Repos with clear training scripts and evaluation
- Repos that already implement the BASELINE the hypothesis compares against
- Simple codebases over complex frameworks
- Repos or scripts that run real public benchmark datasets and real models.
- For LLM/reasoning experiments, prefer Hugging Face Transformers/vLLM evaluation
  harnesses over toy repos.

If no suitable codebase exists, set url to "scratch" and provide setup commands for
a generated real-benchmark runner. Do not recommend synthetic proxy experiments."""




EXPERIMENT_REVIEW_REPAIR_SYSTEM = """You are the experiment design repair agent for DeepGraph.

You receive a research idea, current proposed method, current experimental plan, and
structured review blockers. Repair the experimental plan so it can pass formal
benchmark review.

Rules:
- Do not downgrade to smoke/proxy/synthetic experiments.
- Use real public benchmark datasets and concrete model targets/checkpoints.
- Preserve the scientific claim when possible, but narrow the benchmark scope if the
  current scope has no runnable public benchmark path.
- Prefer official benchmark splits, official metrics, and benchmark-specific repeat
  policies. If the official protocol is unknown, mark that dataset for dataset-card
  inspection rather than inventing global thresholds.
- Include at least two explicit baselines, concrete model targets, primary metric,
  ablations, compute budget, and seed policy.
- If a generated runner cannot support the requested benchmark, choose a compatible
  public benchmark only when it still tests the same claim; otherwise explain that a
  dedicated harness is required.

Return one strict JSON object:
{
  "repair_summary": "what changed and why",
  "experimental_plan_patch": {
    "datasets": [{"name": "...", "split": "...", "why": "..."}],
    "benchmark_targets": [{"name": "...", "hf_dataset": "...", "split": "...", "task_type": "..."}],
    "model_targets": [{"name": "...", "hf_model": "...", "backend": "transformers|official_eval|custom_harness"}],
    "baselines": [{"name": "...", "why": "..."}],
    "metrics": {"primary": "...", "secondary": ["..."]},
    "ablations": [{"name": "...", "removes": "..."}],
    "compute_budget": {"total_gpu_hours": 0},
    "minimum_seeds": 1,
    "benchmark_repair_notes": ["..."]
  }
}
"""


def _merge_experiment_plan_patch(plan: dict, patch: dict) -> dict:
    merged = dict(plan or {})
    for key, value in (patch or {}).items():
        if value in (None, "", [], {}):
            continue
        if key in {
            "datasets",
            "benchmark_targets",
            "model_targets",
            "models",
            "baselines",
            "ablations",
            "benchmark_repair_notes",
        } and isinstance(value, list):
            merged[key] = value
        elif key in {"metrics", "compute_budget", "benchmark_protocol"} and isinstance(value, dict):
            base = dict(merged.get(key) or {}) if isinstance(merged.get(key), dict) else {}
            base.update(value)
            merged[key] = base
        elif key in {"minimum_seeds", "max_eval_examples", "sanity_max_eval_examples"}:
            merged[key] = value
        elif key in {"procedure", "statistical_test", "benchmark_scope"}:
            merged[key] = value
    return merged



def _finalize_repaired_experiment_plan(parsed: dict, method: dict, plan: dict) -> dict:
    """Normalize a repaired plan without network dataset probing."""
    plan = dict(plan or {})
    resource_class = _non_empty_text(parsed.get("resource_class")) or infer_resource_class(
        {**parsed, "proposed_method": method, "experimental_plan": plan}
    )

    if not isinstance(parsed.get("evidence_plan"), dict) or not parsed.get("evidence_plan"):
        parsed["evidence_plan"] = build_evidence_plan(
            {**parsed, "proposed_method": method, "experimental_plan": plan}
        )
    benchmark_design = build_benchmark_design_contract(parsed, method, plan)
    plan = apply_benchmark_design_contract(plan, benchmark_design)

    if EXPERIMENT_REQUIRE_REAL_BENCHMARK and plan.get("benchmark_design_status") != DESIGN_STATUS_RESOLVED:
        plan["real_benchmark_required"] = True
        plan["requires_real_model"] = True
        plan["requires_real_dataset"] = True
        plan["proxy_allowed"] = bool(EXPERIMENT_ALLOW_SYNTHETIC_FALLBACK)
        plan["generated_runner_supported"] = False
        blockers = plan.get("benchmark_design_blockers") if isinstance(plan.get("benchmark_design_blockers"), list) else []
        plan["benchmark_recipe_blockers"] = [
            {"name": "benchmark_literature_review", "reason": str(blocker)}
            for blocker in blockers
            if str(blocker).strip()
        ] or [
            {"name": "benchmark_literature_review", "reason": "Benchmark design requires domain literature review."}
        ]

    if EXPERIMENT_REQUIRE_REAL_BENCHMARK and plan.get("benchmark_design_status") == DESIGN_STATUS_RESOLVED:
        targets = []
        for row in plan.get("benchmark_targets") if isinstance(plan.get("benchmark_targets"), list) else []:
            if isinstance(row, dict):
                name = _non_empty_text(row.get("name") or row.get("hf_dataset") or row.get("dataset"))
                if name and not _looks_like_synthetic_dataset(name):
                    targets.append(_normalize_benchmark_target(row, parsed=parsed))
        dataset_names = _unique_non_empty(_named_values(plan.get("datasets"), keys=("name", "dataset", "hf_dataset")))
        for name in dataset_names:
            if not _looks_like_synthetic_dataset(name) and not any(
                target.get("name") == name or target.get("hf_dataset") == name for target in targets
            ):
                targets.append(_normalize_benchmark_target({"name": name}, parsed=parsed))
        if not targets:
            targets = _default_real_benchmark_targets({**parsed, "proposed_method": method})
        recipe_blockers = []
        normalized_targets = []
        for target in targets:
            target = _normalize_benchmark_target(target, parsed=parsed)
            supported, reason = _generated_runner_support_reason(target)
            target["generated_runner_supported"] = supported
            if reason:
                target["generated_runner_blocker"] = reason
                recipe_blockers.append(
                    {
                        "name": target.get("name") or target.get("hf_dataset") or "benchmark",
                        "reason": reason,
                    }
                )
            normalized_targets.append(target)
        plan["benchmark_targets"] = normalized_targets
        plan["datasets"] = [
            {
                "name": row.get("name") or row.get("hf_dataset") or row.get("dataset"),
                "split": row.get("split") or row.get("official_split") or "inspect_dataset_card",
            }
            for row in normalized_targets
        ]
        plan["real_benchmark_required"] = True
        plan["requires_real_model"] = True
        plan["requires_real_dataset"] = True
        plan["proxy_allowed"] = bool(EXPERIMENT_ALLOW_SYNTHETIC_FALLBACK)
        plan["generated_runner_supported"] = not recipe_blockers
        if recipe_blockers:
            plan["benchmark_recipe_blockers"] = recipe_blockers
            plan["deferred_benchmark_targets"] = [item["name"] for item in recipe_blockers if item.get("name")]
        else:
            plan.pop("benchmark_recipe_blockers", None)
            plan.pop("deferred_benchmark_targets", None)

    models = _model_target_names(plan)
    if not models:
        plan["model_targets"] = _default_real_model_targets(parsed, resource_class)
    baselines = _planned_baselines(plan)
    if len(baselines) < 2:
        method_name = _non_empty_text(method.get("name")) or "candidate_method"
        baselines = _unique_non_empty(baselines + [f"{method_name}_reference_baseline", f"{method_name}_ablation"])
    plan["baselines"] = [{"name": name} for name in baselines[:6]]

    metrics = dict(plan.get("metrics") or {}) if isinstance(plan.get("metrics"), dict) else {}
    if not _non_empty_text(metrics.get("primary")):
        metrics["primary"] = _fallback_metric_name(parsed, plan)
    plan["metrics"] = metrics

    compute = dict(plan.get("compute_budget") or {}) if isinstance(plan.get("compute_budget"), dict) else {}
    if not compute.get("total_gpu_hours"):
        compute["total_gpu_hours"] = 0.0 if resource_class == "cpu" else (24.0 if resource_class == "gpu_large" else 4.0)
    plan["compute_budget"] = compute

    try:
        planned_seeds = int(plan.get("minimum_seeds") or 0)
    except (TypeError, ValueError):
        planned_seeds = 0
    plan["minimum_seeds"] = max(1, planned_seeds)
    plan["ablations"] = [{"name": name} for name in _planned_ablations(method, plan)]

    benchmark_protocol = resolve_benchmark_protocol(
        plan,
        method=method,
        claim=parsed.get("hypothesis") or parsed.get("title") or parsed.get("problem_statement"),
    )
    protocol_seed_policy = benchmark_protocol.get("seed_policy") if isinstance(benchmark_protocol.get("seed_policy"), dict) else {}
    try:
        protocol_min_seeds = int(protocol_seed_policy.get("minimum_repeats") or 1)
    except (TypeError, ValueError):
        protocol_min_seeds = 1
    plan["minimum_seeds"] = max(int(plan.get("minimum_seeds") or 1), protocol_min_seeds)
    plan["benchmark_protocol"] = benchmark_protocol
    publication_contract = _publication_evidence_contract(
        {**parsed, "proposed_method": method},
        plan,
        evidence_plan=parsed.get("evidence_plan") if isinstance(parsed.get("evidence_plan"), dict) else {},
        scaffold_kind="planned",
    )
    plan["publication_evidence_contract"] = publication_contract
    plan["paper_intent"] = publication_contract.get("paper_intent", {})
    return plan


def repair_experiment_plan_from_review(
    insight_id: int,
    *,
    judgement: dict | None = None,
    attempt: int = 1,
    resource_grant_id: int | None = None,
) -> dict:
    """Repair a review-blocked experimental plan and persist it on deep_insights.

    This is the automatic feedback loop from structured experiment review back
    to experiment design. It never creates an experiment_run; the caller should
    requeue the insight so forge/review runs again on the repaired contract.
    """

    insight = db.fetchone("SELECT * FROM deep_insights WHERE id=?", (insight_id,))
    if not insight:
        return {"error": f"Deep insight {insight_id} not found"}
    agenda_id = int(insight.get("agenda_id") or 0)
    if agenda_id <= 0:
        return {
            "error": "experiment repair requires an agenda-scoped insight",
            "route": "blocked",
        }
    try:
        resource_grant_id = int(resource_grant_id or 0)
    except (TypeError, ValueError):
        resource_grant_id = 0
    grant = db.fetchone(
        """
        SELECT id, agenda_id, idea_id, stage, status, expires_at
        FROM resource_grants
        WHERE id=? AND agenda_id=? AND idea_id=?
          AND status='active' AND expires_at > CURRENT_TIMESTAMP
        """,
        (resource_grant_id, agenda_id, insight_id),
    )
    if not grant or str(grant.get("stage") or "") not in {
        "experiment_forge",
        "experiment_repair",
        "benchmark_design",
        "pilot",
    }:
        return {
            "error": "valid experiment repair ResourceGrant is required",
            "route": "manual_review_required",
        }
    llm_scope = {
        "agenda_id": agenda_id,
        "idea_id": insight_id,
        "resource_grant_id": resource_grant_id,
        "stage": str(grant["stage"]),
    }

    parsed = _parse_insight_fields(dict(insight))
    current_plan = parsed.get("experimental_plan") if isinstance(parsed.get("experimental_plan"), dict) else {}
    method = parsed.get("proposed_method") if isinstance(parsed.get("proposed_method"), dict) else {}
    method = _enrich_proposed_method(parsed, current_plan)
    parsed["proposed_method"] = method
    judgement = judgement or {}
    blockers = judgement.get("blockers") if isinstance(judgement.get("blockers"), list) else []
    warnings = judgement.get("warnings") if isinstance(judgement.get("warnings"), list) else []
    summary = _non_empty_text(judgement.get("summary")) or "Experiment review blocked formalization."

    prompt = {
        "insight_id": insight_id,
        "attempt": attempt,
        "title": parsed.get("title"),
        "problem_statement": parsed.get("problem_statement"),
        "proposed_method": method,
        "current_experimental_plan": current_plan,
        "review_summary": summary,
        "review_blockers": blockers,
        "review_warnings": warnings,
        "benchmark_protocol": current_plan.get("benchmark_protocol"),
        "instruction": "Repair only the experiment design/benchmark contract; do not produce manuscript text.",
    }

    try:
        repaired, _, repair_route = _resource_granted_proposer_json(
            EXPERIMENT_REVIEW_REPAIR_SYSTEM,
            json.dumps(prompt, ensure_ascii=False, indent=2),
            llm_scope=llm_scope,
            operation=f"experiment_forge.repair.attempt_{attempt}",
            max_tokens=6000,
        )
    except Exception as exc:
        return {
            "error": f"resource-granted experiment repair unavailable: {exc}",
            "route": "manual_review_required",
        }

    patch = {}
    repair_summary = "Resource-granted experiment repair returned no patch."
    if isinstance(repaired, dict) and not repaired.get("error"):
        patch = repaired.get("experimental_plan_patch") if isinstance(repaired.get("experimental_plan_patch"), dict) else {}
        if not patch and isinstance(repaired.get("experimental_plan"), dict):
            patch = repaired["experimental_plan"]
        repair_summary = _non_empty_text(repaired.get("repair_summary")) or repair_summary
    elif isinstance(repaired, dict) and repaired.get("error"):
        return {
            "error": f"resource-granted experiment repair rejected: {repaired.get('error')}",
            "route": "manual_review_required",
        }

    merged_plan = _merge_experiment_plan_patch(current_plan, patch)
    merged_plan.setdefault("review_repair_history", [])
    if isinstance(merged_plan["review_repair_history"], list):
        merged_plan["review_repair_history"].append(
            {
                "attempt": attempt,
                "summary": repair_summary,
                "blockers": blockers,
                "warnings": warnings,
            }
        )
    parsed["experimental_plan"] = _finalize_repaired_experiment_plan(parsed, method, merged_plan)
    _persist_enriched_insight(insight_id, parsed)
    return {
        "status": "repaired",
        "attempt": attempt,
        "repair_summary": repair_summary,
        "llm_repair_used": bool(isinstance(repaired, dict) and not repaired.get("error") and patch),
        "llm_route": repair_route,
        "blocker_count": len(blockers),
    }

def _parse_insight_fields(insight: dict) -> dict:
    """Extract and parse JSON fields from a deep_insight row."""
    parsed = dict(insight)
    for field in ("proposed_method", "experimental_plan", "related_work_positioning",
                  "evidence_plan",
                  "field_a", "field_b", "predictions", "falsification",
                  "supporting_papers", "source_node_ids", "adversarial_critique"):
        val = parsed.get(field)
        if isinstance(val, str) and val.strip():
            try:
                parsed[field] = json.loads(val)
            except (json.JSONDecodeError, TypeError):
                pass
    for field in ("proposed_method", "experimental_plan", "related_work_positioning", "evidence_plan",
                  "field_a", "field_b", "falsification", "adversarial_critique"):
        if not isinstance(parsed.get(field), dict):
            parsed[field] = {}
    for field in ("predictions", "supporting_papers", "source_node_ids", "source_paper_ids", "source_signal_ids"):
        if not isinstance(parsed.get(field), list):
            parsed[field] = []
    return parsed


def _non_empty_text(value) -> str:
    return str(value or "").strip()


def _unique_non_empty(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        text = _non_empty_text(item)
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result


def _named_values(rows, *, keys: tuple[str, ...] = ("name", "model")) -> list[str]:
    values: list[str] = []
    for row in rows or []:
        if isinstance(row, dict):
            for key in keys:
                text = _non_empty_text(row.get(key))
                if text:
                    values.append(text)
                    break
        else:
            text = _non_empty_text(row)
            if text:
                values.append(text)
    return values


def _fallback_metric_name(parsed: dict, plan: dict) -> str:
    corpus = " ".join(
        [
            _non_empty_text(parsed.get("title")),
            _non_empty_text(parsed.get("problem_statement")),
            _non_empty_text(plan.get("procedure")),
            json.dumps(plan.get("metrics", {}), ensure_ascii=False),
        ]
    ).lower()
    if "bit error" in corpus or "ber" in corpus:
        return "bit_error_rate"
    if "auc" in corpus:
        return "auc"
    if "accuracy" in corpus:
        return "accuracy"
    if "reward" in corpus:
        return "reward"
    if "utility" in corpus:
        return "utility"
    if "latency" in corpus:
        return "latency"
    if "success" in corpus:
        return "task_success_rate"
    return "primary_score"


_SYNTHETIC_DATASET_MARKERS = (
    "synthetic",
    "simulated",
    "simulation",
    "toy",
    "smoke",
    "probe",
    "dummy",
    "random",
    "minimal",
)

_GENERIC_DATASET_NAMES = {
    "dataset",
    "dataset-a",
    "dataset-b",
    "dataset-1",
    "dataset-2",
    "benchmark",
    "benchmark-a",
    "benchmark-b",
}


def _looks_like_synthetic_dataset(name: str) -> bool:
    lowered = _non_empty_text(name).lower()
    normalized = lowered.replace("_", "-").strip()
    return (
        not lowered
        or normalized in _GENERIC_DATASET_NAMES
        or any(marker in lowered for marker in _SYNTHETIC_DATASET_MARKERS)
    )


_STANDARD_REASONING_BASELINES = [
    "Vanilla Direct Answering",
    "Always-Reason Chain-of-Thought",
    "Self-Consistency Reasoning",
    "Least-to-Most Prompting",
    "Confidence Gate",
    "Disagreement Routing",
    "Random Budget-Matched Routing",
    "Oracle Routing Upper Bound",
]


_TOP_VENUE_REASONING_BASELINES = [
    "CAR-Style Certainty Adaptive Routing",
    "Self-Route-Style Mode Routing",
]


# Ablations are method contracts, not universal defaults. An explicitly
# enabled implementation plugin must declare them.
_STANDARD_REASONING_ABLATIONS: list[str] = []


_GENERIC_BASELINE_NAMES = {
    "a",
    "b",
    "baseline",
    "model a",
    "model b",
    "method a",
    "method b",
    "candidate",
    "proposed method",
}


def _canonical_name(value: str) -> str:
    return "".join(ch for ch in str(value or "").lower() if ch.isalnum())


def _reasoning_benchmark_target(name: str | None = None) -> dict | None:
    key = _canonical_name(name or "")
    registry = [
        {
            "name": "GSM8K",
            "aliases": {"gsm8k", "openaigsm8k", "gradeschoolmath"},
            "hf_dataset": EXPERIMENT_REAL_BENCHMARK_DATASET,
            "hf_candidates": [EXPERIMENT_REAL_BENCHMARK_DATASET, "openai/gsm8k"],
            "config": EXPERIMENT_REAL_BENCHMARK_DATASET_CONFIG,
            "config_candidates": [EXPERIMENT_REAL_BENCHMARK_DATASET_CONFIG, "main", ""],
            "split": "test",
            "split_candidates": ["test", "train"],
            "task_type": "math_qa",
            "question_field": "question",
            "answer_field": "answer",
        },
        {
            "name": "MuSiQue-Ans",
            "aliases": {"musique", "musiqueans", "dgslibiseymusique", "voidfulmusique"},
            "hf_dataset": "dgslibisey/MuSiQue",
            "hf_candidates": ["dgslibisey/MuSiQue", "voidful/MuSiQue", "bdsaglam/musique"],
            "direct_files": [
                {
                    "id": "dgslibisey/MuSiQue:musique_ans_v1.0_dev.jsonl",
                    "url": "https://huggingface.co/datasets/dgslibisey/MuSiQue/resolve/main/musique_ans_v1.0_dev.jsonl",
                    "format": "jsonl",
                    "split": "validation",
                },
                {
                    "id": "voidful/MuSiQue:musique_ans_v1.0_test.jsonl",
                    "url": "https://huggingface.co/datasets/voidful/MuSiQue/resolve/main/musique_ans_v1.0_test.jsonl",
                    "format": "jsonl",
                    "split": "test",
                },
            ],
            "config": "",
            "config_candidates": ["", "answerable"],
            "split": "validation",
            "split_candidates": ["validation", "test", "train"],
            "task_type": "multihop_qa",
            "question_field": "question",
            "answer_field": "answer",
        },
        {
            "name": "StrategyQA",
            "aliases": {"strategyqa", "strategyqaopen", "tasksourcestrategyqa"},
            "hf_dataset": "tasksource/strategy-qa",
            "hf_candidates": ["tasksource/strategy-qa", "ChilleD/StrategyQA", "wics/strategy-qa"],
            "direct_files": [
                {
                    "id": "ChilleD/StrategyQA:test.json",
                    "url": "https://huggingface.co/datasets/ChilleD/StrategyQA/resolve/main/test.json",
                    "format": "json",
                    "split": "test",
                },
                {
                    "id": "tasksource/strategy-qa:strategyQA_train.json",
                    "url": "https://huggingface.co/datasets/tasksource/strategy-qa/resolve/main/strategyQA_train.json",
                    "format": "json",
                    "split": "train",
                },
            ],
            "config": "",
            "config_candidates": ["", "default"],
            "split": "validation",
            "split_candidates": ["validation", "test", "train"],
            "task_type": "boolean_qa",
            "question_field": "question",
            "answer_field": "answer",
        },
        {
            "name": "2WikiMultihopQA",
            "aliases": {"2wiki", "2wikimultihopqa", "twowikimultihopqa", "xanhho2wikimultihopqa"},
            "hf_dataset": "xanhho/2WikiMultihopQA",
            "hf_candidates": ["xanhho/2WikiMultihopQA", "voidful/2WikiMultihopQA"],
            "direct_files": [
                {
                    "id": "xanhho/2WikiMultihopQA:dev.parquet",
                    "url": "https://huggingface.co/datasets/xanhho/2WikiMultihopQA/resolve/main/dev.parquet",
                    "format": "parquet",
                    "split": "validation",
                },
                {
                    "id": "voidful/2WikiMultihopQA:dev.json",
                    "url": "https://huggingface.co/datasets/voidful/2WikiMultihopQA/resolve/main/dev.json",
                    "format": "json",
                    "split": "validation",
                },
            ],
            "config": "",
            "config_candidates": ["", "default"],
            "split": "validation",
            "split_candidates": ["validation", "test", "train"],
            "task_type": "multihop_qa",
            "question_field": "question",
            "answer_field": "answer",
        },
        {
            "name": "Stress Test Split: Simple-vs-Hard Counterfactual Partition",
            "aliases": {"stresstestsplit", "simplevshard", "counterfactualpartition"},
            "hf_dataset": "",
            "hf_candidates": [],
            "config": "",
            "config_candidates": [""],
            "split": "derived",
            "split_candidates": ["derived"],
            "task_type": "derived_stress_split",
            "derive_from_loaded_benchmarks": True,
        },
    ]
    if not key:
        return None
    for row in registry:
        aliases = {_canonical_name(row["name"]), *row.get("aliases", set())}
        if key in aliases or any(alias and alias in key for alias in aliases):
            clean = dict(row)
            clean.pop("aliases", None)
            return clean
    return None


_GENERATED_RUNNER_TEXT_TASK_TYPES = {
    "",
    "qa",
    "math_qa",
    "multihop_qa",
    "boolean_qa",
    "code_generation",
    "derived_stress_split",
}

# Public data does not imply that the built-in runner knows the benchmark's
# metric and annotation protocol.  These names remain on the dedicated
# harness path even when a literature design supplies an HF dataset id.
_DEDICATED_HARNESS_BENCHMARK_ALIASES = {
    "math",
    "math500",
    "prm800k",
    "processbench",
    "spider",
    "bird",
    "harmbench",
    "advbench",
    "agentdojo",
    "longmemeval",
    "cifar10",
    "clevrer",
    "t2icompbench",
}


def _generated_runner_support_reason(target: dict) -> tuple[bool, str]:
    """Return whether the built-in runner can execute this benchmark target."""
    name = _non_empty_text(target.get("name") or target.get("hf_dataset") or "benchmark")
    if target.get("generated_runner_supported") is False:
        return False, f"{name} is explicitly marked as requiring a dedicated domain benchmark harness."
    if target.get("derive_from_loaded_benchmarks"):
        return True, ""
    if target.get("requires_harness"):
        return False, f"{name} requires a dedicated domain benchmark harness."

    benchmark_names = [
        target.get("name"),
        target.get("hf_dataset"),
        *(target.get("hf_candidates") or []),
    ]
    has_concrete_source = bool(
        target.get("direct_files")
        or _non_empty_text(target.get("hf_dataset"))
        or any(_non_empty_text(value) for value in (target.get("hf_candidates") or []))
    )
    if has_concrete_source:
        for value in benchmark_names:
            key = _canonical_name(value)
            if any(alias in key for alias in _DEDICATED_HARNESS_BENCHMARK_ALIASES):
                return False, f"{name} requires a dedicated domain benchmark harness."

    task_type = _non_empty_text(target.get("task_type")).lower()
    if task_type == "benchmark":
        task_type = ""
    if task_type not in _GENERATED_RUNNER_TEXT_TASK_TYPES:
        return (
            False,
            f"{name} requires task_type={task_type}; the built-in generated runner only supports text QA/code rows.",
        )

    candidates = [
        _non_empty_text(target.get("hf_dataset")),
        *[_non_empty_text(value) for value in (target.get("hf_candidates") or [])],
    ]
    candidates = [value for value in candidates if value]
    if target.get("direct_files"):
        return True, ""
    if any("/" in value for value in candidates):
        return True, ""
    return (
        False,
        f"{name} has no concrete Hugging Face dataset id, direct file, or registered benchmark recipe.",
    )


def _normalize_benchmark_target(row, *, parsed: dict | None = None) -> dict:
    parsed = parsed or {}
    source = dict(row) if isinstance(row, dict) else {"name": row}
    name = _non_empty_text(
        source.get("name") or source.get("dataset") or source.get("hf_dataset") or source.get("id")
    )
    template = _reasoning_benchmark_target(name) or {}
    target = {**template, **source}
    target["name"] = _non_empty_text(target.get("name") or name or target.get("hf_dataset")) or "unresolved_benchmark"
    hf_candidates = []
    for value in (
        [target.get("hf_dataset")]
        + list(target.get("hf_candidates") or [])
        + list(template.get("hf_candidates") or [])
    ):
        text = _non_empty_text(value)
        if template and text and "/" not in text and _canonical_name(text) in {
            _canonical_name(target.get("name")),
            *{_canonical_name(alias) for alias in (template.get("aliases") or [])},
        }:
            continue
        if text and text not in hf_candidates:
            hf_candidates.append(text)
    target["hf_candidates"] = hf_candidates
    target["hf_dataset"] = _non_empty_text(target.get("hf_dataset")) or (hf_candidates[0] if hf_candidates else "")
    split_candidates = []
    for value in [target.get("split")] + list(target.get("split_candidates") or []):
        text = _non_empty_text(value)
        if text and text not in split_candidates:
            split_candidates.append(text)
    target["split_candidates"] = split_candidates or ["validation", "test", "train"]
    config_candidates = []
    for value in [target.get("config")] + list(target.get("config_candidates") or []):
        text = "" if value is None else str(value).strip()
        if text not in config_candidates:
            config_candidates.append(text)
    target["config_candidates"] = config_candidates or [""]
    try:
        target["max_eval_examples"] = int(target.get("max_eval_examples"))
    except (TypeError, ValueError):
        target["max_eval_examples"] = 0
    if _non_empty_text(target.get("task_type")).lower() in {"", "benchmark"} and template.get("task_type"):
        target["task_type"] = template["task_type"]
    else:
        target.setdefault("task_type", "benchmark")
    supported, reason = _generated_runner_support_reason(target)
    target["generated_runner_supported"] = supported
    if reason:
        target["generated_runner_blocker"] = reason
    return target


def _planned_baselines(plan: dict) -> list[str]:
    raw = _unique_non_empty(_named_values(plan.get("baselines"), keys=("name", "model", "method")))
    filtered = [name for name in raw if name.lower().strip() not in _GENERIC_BASELINE_NAMES]
    corpus = json.dumps(plan, ensure_ascii=False).lower()
    if plan.get("real_benchmark_required") or any(
        token in corpus for token in ("gsm8k", "qa", "reason", "cot", "llm", "musique", "strategyqa", "2wiki")
    ):
        filtered.extend(_STANDARD_REASONING_BASELINES)
    if any(token in corpus for token in ("top venue", "top-venue", "sota", "state of the art", "car-style", "self-route")):
        filtered.extend(_TOP_VENUE_REASONING_BASELINES)
    return _unique_non_empty(filtered)


def _default_real_benchmark_targets(parsed: dict) -> list[dict]:
    corpus = " ".join(
        [
            _non_empty_text(parsed.get("title")),
            _non_empty_text(parsed.get("problem_statement")),
            _non_empty_text(parsed.get("existing_weakness")),
            _non_empty_text((parsed.get("proposed_method") or {}).get("type")),
            _non_empty_text((parsed.get("proposed_method") or {}).get("definition")),
            json.dumps(parsed.get("source_node_ids") or [], ensure_ascii=False),
        ]
    ).lower()
    if any(token in corpus for token in ("math", "reasoning", "cot", "deliberation", "qa", "question answering", "llm")):
        return [
            _normalize_benchmark_target(name, parsed=parsed)
            for name in (
                "GSM8K",
                "MuSiQue-Ans",
                "StrategyQA",
                "2WikiMultihopQA",
                "Stress Test Split: Simple-vs-Hard Counterfactual Partition",
            )
        ]
    if any(token in corpus for token in ("code", "program", "python", "humaneval", "mbpp")):
        return [
            {
                "name": "MBPP",
                "hf_dataset": "google-research-datasets/mbpp",
                "config": "",
                "split": "test",
                "task_type": "code_generation",
                "question_field": "text",
                "answer_field": "code",
                "max_eval_examples": 0,
            }
        ]
    if any(token in corpus for token in ("vision", "image", "classification", "imagenet", "cifar")):
        return [
            {
                "name": "CIFAR-10",
                "hf_dataset": "cifar10",
                "config": "",
                "split": "test",
                "task_type": "image_classification",
                "max_eval_examples": 0,
            }
        ]
    return []


def _supported_probe_benchmark_targets(parsed: dict, method: dict, deferred_targets: list[dict]) -> list[dict]:
    deferred_names = _unique_non_empty([
        _non_empty_text(row.get("name") or row.get("hf_dataset") or row.get("dataset"))
        for row in deferred_targets
        if isinstance(row, dict)
    ])
    deferred_keys = {name.lower() for name in deferred_names}
    probes: list[dict] = []
    for row in _default_real_benchmark_targets({**parsed, "proposed_method": method}):
        target = _normalize_benchmark_target(row, parsed=parsed)
        name = _non_empty_text(target.get("name") or target.get("hf_dataset") or target.get("dataset"))
        if name.lower() in deferred_keys:
            continue
        supported, reason = _generated_runner_support_reason(target)
        if not supported:
            continue
        target["generated_runner_supported"] = True
        target.pop("generated_runner_blocker", None)
        target["benchmark_role"] = "executable_probe"
        target["probe_for_deferred_benchmark_targets"] = deferred_names
        target["formal_target_deferred"] = True
        probes.append(target)
        if len(probes) >= 1:
            break
    return probes


def _default_real_model_targets(parsed: dict, resource_class: str | None = None) -> list[dict]:
    corpus = " ".join(
        [
            _non_empty_text(parsed.get("title")),
            _non_empty_text(parsed.get("problem_statement")),
            _non_empty_text((parsed.get("proposed_method") or {}).get("definition")),
        ]
    ).lower()
    if any(token in corpus for token in ("vision", "image", "classification", "imagenet", "cifar")):
        return [
            {
                "name": "ViT-B/16",
                "hf_model": "google/vit-base-patch16-224",
                "backend": "transformers",
                "role": "candidate_base_model",
            }
        ]
    if str(resource_class or "").strip().lower() == "cpu":
        return [
            {
                "name": "TinyLlama-1.1B CPU",
                "hf_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                "backend": "transformers",
                "role": "candidate_base_model",
                "load_in_4bit": False,
                "requires_cuda": False,
                "cpu_allowed": True,
            },
            {
                "name": "Qwen2.5-0.5B CPU",
                "hf_model": "Qwen/Qwen2.5-0.5B-Instruct",
                "backend": "transformers",
                "role": "secondary_model",
                "load_in_4bit": False,
                "requires_cuda": False,
                "cpu_allowed": True,
            },
        ]
    primary = EXPERIMENT_REAL_LLM_MODEL
    candidates = [
        primary,
        "Qwen/Qwen2.5-3B-Instruct",
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    ]
    out = []
    seen = set()
    for idx, model_id in enumerate(candidates):
        if not model_id or model_id.lower() in seen:
            continue
        seen.add(model_id.lower())
        out.append(
            {
                "name": model_id,
                "hf_model": model_id,
                "backend": "transformers",
                "role": "candidate_base_model" if idx == 0 else "secondary_model",
                "load_in_4bit": True,
                "requires_cuda": True,
            }
        )
        if len(out) >= EXPERIMENT_FULL_BENCHMARK_MIN_MODELS:
            break
    return out


def _real_benchmark_dataset_names(plan: dict) -> list[str]:
    targets = plan.get("benchmark_targets") if isinstance(plan.get("benchmark_targets"), list) else []
    names = _unique_non_empty(_named_values(targets, keys=("name", "hf_dataset", "dataset")))
    if names:
        return [name for name in names if not _looks_like_synthetic_dataset(name)]
    return [
        name for name in _unique_non_empty(_named_values(plan.get("datasets"), keys=("name", "dataset")))
        if not _looks_like_synthetic_dataset(name)
    ]


def _model_target_names(plan: dict) -> list[str]:
    rows = []
    for key in ("model_targets", "models"):
        value = plan.get(key)
        if isinstance(value, list):
            rows.extend(value)
    return _unique_non_empty(_named_values(rows, keys=("hf_model", "model", "name")))


def _ensure_real_benchmark_plan(
    parsed: dict,
    method: dict,
    plan: dict,
    resource_class: str | None = None,
    *,
    resolve_datasets: bool = True,
    llm_scope: dict | None = None,
) -> dict:
    plan = dict(plan or {})
    if not EXPERIMENT_REQUIRE_REAL_BENCHMARK:
        return plan
    if not plan.get("benchmark_design_status"):
        benchmark_design = build_benchmark_design_contract(
            parsed,
            method,
            plan,
            llm_scope=llm_scope,
        )
        plan = apply_benchmark_design_contract(plan, benchmark_design)
    # The harness consumer can recover an executable benchmark subset while
    # the formal/domain benchmark design remains unresolved.  This is an
    # honest bootstrap probe; manuscript evidence stays blocked.
    harness_recovery_probe = bool(plan.get("harness_recovery_fresh_forge"))
    if (
        plan.get("benchmark_design_status")
        and plan.get("benchmark_design_status") != DESIGN_STATUS_RESOLVED
        and not harness_recovery_probe
    ):
        blockers = plan.get("benchmark_design_blockers") if isinstance(plan.get("benchmark_design_blockers"), list) else []
        plan["real_benchmark_required"] = True
        plan["requires_real_model"] = True
        plan["requires_real_dataset"] = True
        plan["generated_runner_supported"] = False
        plan["benchmark_recipe_blockers"] = [
            {"name": "benchmark_literature_review", "reason": str(blocker)}
            for blocker in blockers
            if str(blocker).strip()
        ] or [
            {
                "name": "benchmark_literature_review",
                "reason": "Benchmark design must be resolved against domain literature before formal execution.",
            }
        ]
        plan["benchmark_protocol"] = resolve_benchmark_protocol(
            plan,
            method=method,
            claim=parsed.get("hypothesis") or parsed.get("title") or parsed.get("problem_statement"),
        )
        return plan
    if resolve_datasets:
        plan = resolve_plan_datasets(plan)
    existing_targets = plan.get("benchmark_targets") if isinstance(plan.get("benchmark_targets"), list) else []
    real_targets = []
    for row in existing_targets:
        if isinstance(row, dict):
            name = _non_empty_text(row.get("name") or row.get("hf_dataset") or row.get("dataset"))
            if name and not _looks_like_synthetic_dataset(name):
                real_targets.append(_normalize_benchmark_target(row, parsed=parsed))
    dataset_names = _unique_non_empty(_named_values(plan.get("datasets"), keys=("name", "dataset")))
    for name in dataset_names:
        if not _looks_like_synthetic_dataset(name) and not any((t.get("name") == name or t.get("hf_dataset") == name) for t in real_targets):
            real_targets.append(_normalize_benchmark_target({"name": name}, parsed=parsed))
    if not real_targets:
        real_targets = _default_real_benchmark_targets({**parsed, "proposed_method": method})
    if not real_targets:
        blocker = "No semantically aligned benchmark target is available; run benchmark literature review before formal execution."
        plan["benchmark_design_status"] = plan.get("benchmark_design_status") or "literature_review_required"
        plan["benchmark_design_blockers"] = list(plan.get("benchmark_design_blockers") or []) + [blocker]
        plan["real_benchmark_required"] = True
        plan["requires_real_model"] = True
        plan["requires_real_dataset"] = True
        plan["generated_runner_supported"] = False
        plan["benchmark_recipe_blockers"] = [{"name": "benchmark_literature_review", "reason": blocker}]
        plan["benchmark_protocol"] = resolve_benchmark_protocol(
            plan,
            method=method,
            claim=parsed.get("hypothesis") or parsed.get("title") or parsed.get("problem_statement"),
        )
        return plan
    real_targets = [
        target
        if isinstance(target, dict) and "generated_runner_supported" in target
        else _normalize_benchmark_target(target, parsed=parsed)
        for target in real_targets
    ]
    recipe_blockers = []
    for target in real_targets:
        supported, reason = _generated_runner_support_reason(target)
        target["generated_runner_supported"] = supported
        if reason:
            target["generated_runner_blocker"] = reason
            recipe_blockers.append(
                {
                    "name": target.get("name") or target.get("hf_dataset") or "benchmark",
                    "reason": reason,
                }
            )
    runnable_targets = [
        target for target in real_targets
        if target.get("generated_runner_supported") is not False
    ]
    deferred_targets = [
        target for target in real_targets
        if target.get("generated_runner_supported") is False
    ]
    if recipe_blockers and not runnable_targets and not plan.get("benchmark_design_contract"):
        probe_targets = _supported_probe_benchmark_targets(parsed, method, deferred_targets)
        if probe_targets:
            real_targets = probe_targets + real_targets
            runnable_targets = probe_targets
            plan["benchmark_probe_added"] = {
                "reason": "all requested benchmark targets require dedicated harness recipes",
                "active_probe_targets": [
                    target.get("name") or target.get("hf_dataset") or target.get("dataset")
                    for target in probe_targets
                ],
                "deferred_formal_targets": [
                    item.get("name") for item in recipe_blockers if item.get("name")
                ],
            }
    active_targets = runnable_targets or real_targets
    if runnable_targets and (recipe_blockers or harness_recovery_probe):
        for target in runnable_targets:
            target.setdefault("benchmark_role", "executable_probe")
            target.setdefault("formal_target_deferred", bool(recipe_blockers))
            if recipe_blockers:
                target.setdefault(
                    "probe_for_deferred_benchmark_targets",
                    [item.get("name") for item in recipe_blockers if item.get("name")],
                )
    plan["benchmark_targets"] = active_targets
    plan["datasets"] = [
        {"name": row.get("name") or row.get("hf_dataset") or row.get("dataset")}
        for row in active_targets
    ]
    plan["baselines"] = [{"name": name} for name in _planned_baselines(plan)]

    model_targets = plan.get("model_targets") if isinstance(plan.get("model_targets"), list) else []
    normalized_models = []
    for row in model_targets:
        if isinstance(row, dict):
            name = _non_empty_text(row.get("hf_model") or row.get("model") or row.get("name"))
            if name and name.lower() not in {"toy", "dummy", "mock", "synthetic"}:
                normalized_models.append(dict(row))
    default_models = _default_real_model_targets(parsed, resource_class)
    if str(resource_class or "").strip().lower() != "cpu":
        # Old plans can retain the Qwen2.5/TinyLlama bootstrap list forever.
        # GPU evidence must prefer the configured current-generation model.
        legacy_markers = ("qwen2.5", "tinyllama")
        normalized_models = [
            row
            for row in normalized_models
            if not any(
                marker in str(
                    row.get("hf_model") or row.get("model") or row.get("name") or ""
                ).lower()
                for marker in legacy_markers
            )
        ]
        current_defaults = [
            row
            for row in default_models
            if not any(
                marker in str(
                    row.get("hf_model") or row.get("model") or row.get("name") or ""
                ).lower()
                for marker in legacy_markers
            )
        ]
        if current_defaults:
            default_models = current_defaults
    if not normalized_models:
        normalized_models = default_models
    else:
        seen_models = {
            str(row.get("hf_model") or row.get("name") or row.get("model") or "").strip().lower()
            for row in normalized_models
            if isinstance(row, dict)
        }
        for row in default_models:
            model_key = str(row.get("hf_model") or row.get("name") or row.get("model") or "").strip().lower()
            if model_key and model_key not in seen_models:
                normalized_models.append(dict(row))
                seen_models.add(model_key)
            if len(normalized_models) >= EXPERIMENT_FULL_BENCHMARK_MIN_MODELS:
                break
    normalized_models.sort(
        key=lambda row: (
            0 if bool(row.get("requires_cuda")) else 1,
            0 if "/" in str(row.get("hf_model") or row.get("model") or row.get("name") or "") else 1,
            0 if "qwen" in str(row.get("hf_model") or row.get("model") or row.get("name") or "").lower() else 1,
        )
    )
    plan["model_targets"] = normalized_models
    plan["real_benchmark_required"] = True
    plan["proxy_allowed"] = bool(EXPERIMENT_ALLOW_SYNTHETIC_FALLBACK)
    try:
        planned_seed_count = int(plan.get("minimum_seeds") or 0)
    except (TypeError, ValueError):
        planned_seed_count = 0
    if planned_seed_count > 0:
        plan["minimum_seeds"] = planned_seed_count
    else:
        plan["minimum_seeds"] = 1
    try:
        planned_examples = int(plan.get("max_eval_examples"))
    except (TypeError, ValueError):
        planned_examples = 0
    plan["max_eval_examples"] = max(0, planned_examples)
    for target in real_targets:
        try:
            target_examples = int(target.get("max_eval_examples"))
        except (TypeError, ValueError):
            target_examples = 0
        target["max_eval_examples"] = max(0, target_examples)
    plan["requires_real_model"] = True
    plan["requires_real_dataset"] = True
    plan["generated_runner_supported"] = bool(runnable_targets) if recipe_blockers else True
    if recipe_blockers:
        plan["benchmark_recipe_blockers"] = recipe_blockers
        plan["deferred_benchmark_targets"] = [
            item["name"] for item in recipe_blockers if item.get("name")
        ]
        plan["deferred_benchmark_target_details"] = deferred_targets
        plan["benchmark_harness_deferred"] = bool(runnable_targets)
    else:
        plan.pop("benchmark_recipe_blockers", None)
        plan.pop("deferred_benchmark_targets", None)
        plan.pop("deferred_benchmark_target_details", None)
        plan.pop("benchmark_harness_deferred", None)
    benchmark_protocol = resolve_benchmark_protocol(
        plan,
        method=method,
        claim=parsed.get("hypothesis") or parsed.get("title") or parsed.get("problem_statement"),
    )
    protocol_seed_policy = benchmark_protocol.get("seed_policy") if isinstance(benchmark_protocol.get("seed_policy"), dict) else {}
    try:
        protocol_min_seeds = int(protocol_seed_policy.get("minimum_repeats") or 1)
    except (TypeError, ValueError):
        protocol_min_seeds = 1
    plan["minimum_seeds"] = max(int(plan.get("minimum_seeds") or 1), protocol_min_seeds)
    plan["benchmark_protocol"] = benchmark_protocol
    plan["benchmark_execution"] = {
        "mode": "real_benchmark",
        "synthetic_fallback_allowed": bool(EXPERIMENT_ALLOW_SYNTHETIC_FALLBACK),
        "generated_runner_supported": plan["generated_runner_supported"],
        "benchmark_protocol_status": benchmark_protocol.get("status"),
        "default_model": normalized_models[0].get("hf_model") or normalized_models[0].get("name"),
        "default_dataset": active_targets[0].get("hf_dataset") or active_targets[0].get("name"),
        "target_count": len(active_targets),
        "deferred_target_count": len(deferred_targets),
    }
    return plan


def _planned_ablations(method: dict, plan: dict) -> list[str]:
    names = _unique_non_empty(
        _named_values(plan.get("ablations"), keys=("name", "component", "factor"))
        + _named_values(plan.get("components"), keys=("name", "component"))
    )
    if names:
        return _unique_non_empty(names + _STANDARD_REASONING_ABLATIONS)[:8]
    method_name = _non_empty_text(method.get("name")) or "proposed_method"
    properties = _unique_non_empty([str(x) for x in (method.get("key_properties") or [])[:3]])
    ablations = [f"remove_{method_name}", "compute_matched_baseline"]
    for prop in properties:
        safe = "".join(ch if ch.isalnum() else "_" for ch in prop.lower()).strip("_")
        if safe:
            ablations.append(f"disable_{safe[:40]}")
    return _unique_non_empty(ablations + _STANDARD_REASONING_ABLATIONS)[:8]


def _benchmark_manifest(
    parsed: dict,
    plan: dict,
    *,
    codebase: dict | None = None,
    scaffold_kind: str = "planned",
) -> dict:
    """Deterministic benchmark contract used by later agents and UI stages."""
    method = parsed.get("proposed_method", {}) if isinstance(parsed, dict) else {}
    codebase = codebase or {}
    datasets = _unique_non_empty(_named_values(plan.get("datasets"), keys=("name", "dataset")))
    real_benchmarks = _real_benchmark_dataset_names(plan) or datasets
    models = _model_target_names(plan)
    baselines = _planned_baselines(plan)
    ablations = _planned_ablations(method, plan)
    metric = _fallback_metric_name(parsed, plan)
    metrics = plan.get("metrics")
    if isinstance(metrics, dict):
        metric = _non_empty_text(metrics.get("primary") or metrics.get("name")) or metric
        secondary_metrics = metrics.get("secondary") if isinstance(metrics.get("secondary"), list) else []
    else:
        secondary_metrics = []
    benchmark_protocol = resolve_benchmark_protocol(
        plan,
        method=method,
        claim=parsed.get("hypothesis") or parsed.get("title") or parsed.get("problem_statement"),
    )
    protocol_seed_policy = benchmark_protocol.get("seed_policy") if isinstance(benchmark_protocol.get("seed_policy"), dict) else {}
    try:
        protocol_minimum_seeds = max(1, int(protocol_seed_policy.get("minimum_repeats") or 1))
    except (TypeError, ValueError):
        protocol_minimum_seeds = 1
    seed_raw = plan.get("minimum_seeds") or plan.get("seeds")
    try:
        if isinstance(seed_raw, list):
            minimum_seeds = max(len(seed_raw), protocol_minimum_seeds)
        elif seed_raw not in (None, "", "unknown"):
            minimum_seeds = max(int(seed_raw), protocol_minimum_seeds)
        else:
            minimum_seeds = protocol_minimum_seeds
    except (TypeError, ValueError):
        minimum_seeds = protocol_minimum_seeds
    seed_list = list(range(max(1, minimum_seeds)))
    full_artifacts = [
        "run_config.json",
        "raw_predictions.jsonl",
        "routing_decisions.jsonl",
        "per_seed_results.json",
        "per_dataset_results.json",
        "main_results_table.json",
        "cost_utility_tradeoff_table.json",
        "quality_cost_frontier.json",
        "route_rate_sweep_table.json",
        "ablation_table.json",
        "difficulty_breakdown_table.json",
        "routing_analysis.json",
        "latency_tokens_table.json",
        "simple_case_degradation.json",
        "calibration_reliability.json",
        "bootstrap_ci.json",
        "failure_cases.jsonl",
        "artifact_manifest.json",
    ]
    sanity_only = scaffold_kind in {"real_benchmark_fallback", "bootstrap_probe", "fallback_bootstrap"}
    return {
        "schema_version": "benchmark_manifest_v1",
        "scaffold_kind": scaffold_kind,
        "sanity_only": sanity_only,
        "paper_claims_require_full_benchmark": True,
        "agent_roles": {
            "code_scout": "select repo/entrypoint only",
            "benchmark_protocol_resolver": "resolve official splits, sample policy, metrics, and benchmark-specific completeness gates",
            "experiment_contract_architect": "freeze datasets/baselines/metrics/artifact gates",
            "sanity_runner_builder": "small real-data runner for environment and signal checks",
            "full_benchmark_compiler": "expand the locked contract into a job matrix",
            "method_worker": "change method implementation without changing benchmark contract",
            "evidence_auditor": "audit artifacts before paper claims",
            "manuscript_writer": "write only audited claims",
        },
        "benchmark_protocol": benchmark_protocol,
        "locked_fields": [
            "benchmark_protocol",
            "datasets",
            "models",
            "baselines",
            "ablations",
            "metrics",
            "seeds",
            "splits",
            "paper_claims_require_full_benchmark",
        ],
        "sanity_stage": {
            "purpose": "Verify the runner, environment, real model loading, logging, and metric parsing.",
            "may_reduce_examples": True,
            "max_examples_per_seed": _optional_nonnegative_int(plan.get("sanity_max_eval_examples"), 0),
            "datasets": real_benchmarks[:1],
            "models": models[:1],
            "methods": ["baseline", "candidate"],
            "seeds": seed_list[:1],
            "allowed_claim": "pipeline_sanity_only",
        },
        "full_benchmark_stage": {
            "purpose": "Produce paper-eligible evidence.",
            "datasets": real_benchmarks,
            "models": models,
            "baselines": baselines,
            "candidate_method": _non_empty_text(method.get("name")) or "candidate_method",
            "ablations": ablations,
            "seeds": seed_list,
            "primary_metric": metric,
            "secondary_metrics": secondary_metrics,
            "examples_policy": "benchmark_specific_official_or_materialized_full_split",
            "min_examples_total": None,
            "min_datasets": len(benchmark_protocol.get("full_benchmark_requirements", {}).get("required_dataset_names", []) or real_benchmarks),
            "min_models": len(benchmark_protocol.get("model_policy", {}).get("required_models", []) or models),
            "min_deployable_baselines": len(benchmark_protocol.get("baseline_policy", {}).get("required_baselines", []) or baselines),
            "global_numeric_thresholds_allowed": False,
            "must_beat_strongest_deployable_baseline": EXPERIMENT_FULL_BENCHMARK_REQUIRE_STRONGEST_WIN,
            "must_report_significance": EXPERIMENT_FULL_BENCHMARK_REQUIRE_SIGNIFICANCE,
            "required_analyses": [
                "ablation_table",
                "route_rate_or_budget_sweep_for_routing_methods",
                "quality_cost_frontier",
                "per_dataset_breakdown",
                "difficulty_breakdown",
                "pairwise_vs_strongest_deployable_baseline",
            ],
            "statistical_tests": [
                "paired_bootstrap_ci",
                "paired_permutation_test",
            ],
            "required_artifacts": benchmark_protocol.get("full_benchmark_requirements", {}).get("required_artifacts") or full_artifacts,
            "job_matrix_dimensions": [
                "dataset",
                "model",
                "method_or_baseline",
                "ablation",
                "seed",
            ],
        },
        "codebase": {
            "url": codebase.get("url") or "scratch",
            "main_train_file": codebase.get("main_train_file") or "train.py",
            "baseline_command": codebase.get("main_eval_command") or "python train.py",
        },
    }


def _publication_evidence_contract(
    parsed: dict,
    plan: dict,
    *,
    codebase: dict | None = None,
    evidence_plan: dict | None = None,
    scaffold_kind: str = "planned",
) -> dict:
    method = parsed.get("proposed_method", {}) if isinstance(parsed, dict) else {}
    codebase = codebase or {}
    evidence_plan = evidence_plan or {}
    title = _non_empty_text(parsed.get("title")) or _non_empty_text(method.get("one_line"))
    method_name = _non_empty_text(method.get("name")) or title or "proposed_method"
    claim = (
        _non_empty_text(parsed.get("hypothesis"))
        or _non_empty_text(method.get("one_line"))
        or title
        or f"{method_name} improves the primary metric under the planned benchmark."
    )
    datasets = _unique_non_empty(_named_values(plan.get("datasets"), keys=("name", "dataset")))
    baselines = _planned_baselines(plan)
    real_datasets = _real_benchmark_dataset_names(plan)
    model_targets = _model_target_names(plan)
    metric = _fallback_metric_name(parsed, plan)
    metrics = plan.get("metrics")
    if isinstance(metrics, dict):
        metric = _non_empty_text(metrics.get("primary") or metrics.get("name")) or metric
    benchmark_protocol = resolve_benchmark_protocol(plan, method=method, claim=claim)
    protocol_seed_policy = benchmark_protocol.get("seed_policy") if isinstance(benchmark_protocol.get("seed_policy"), dict) else {}
    try:
        minimum_seeds = max(1, int(protocol_seed_policy.get("minimum_repeats") or 1))
    except (TypeError, ValueError):
        minimum_seeds = 1
    seed_raw = plan.get("minimum_seeds") or plan.get("seeds")
    try:
        if isinstance(seed_raw, list):
            minimum_seeds = max(minimum_seeds, len(seed_raw))
        elif seed_raw not in (None, "", "unknown"):
            minimum_seeds = max(minimum_seeds, int(seed_raw))
    except (TypeError, ValueError):
        pass

    claim_route = classify_idea_route(
        {**parsed, "proposed_method": method},
        plan=plan,
        method=method,
    )
    paper_allowed = bool(claim_route.get("paper_allowed"))

    if scaffold_kind in {"bootstrap_probe", "fallback_bootstrap"}:
        evidence_tier = "bootstrap_probe"
        blocks_manuscript = True
    elif scaffold_kind == "real_benchmark_fallback":
        evidence_tier = "sanity_real_benchmark"
        blocks_manuscript = True
    elif scaffold_kind == "full_benchmark_compiled":
        evidence_tier = "benchmark_plan"
        blocks_manuscript = True
    elif real_datasets and len(baselines) >= 2 and model_targets:
        evidence_tier = "benchmark_plan"
        blocks_manuscript = True
    else:
        evidence_tier = "formal_proxy"
        blocks_manuscript = bool(EXPERIMENT_REQUIRE_REAL_BENCHMARK)
    if not paper_allowed:
        blocks_manuscript = True
    recipe_blockers = plan.get("benchmark_recipe_blockers") if isinstance(plan, dict) else None
    if plan.get("generated_runner_supported") is False:
        blocks_manuscript = True

    main_table_enabled = bool((evidence_plan.get("main_table") or {}).get("enabled", True))
    visualization_enabled = bool((evidence_plan.get("visualization") or {}).get("enabled", True))
    protocol_requirements = benchmark_protocol.get("full_benchmark_requirements") if isinstance(benchmark_protocol.get("full_benchmark_requirements"), dict) else {}
    protocol_artifacts = protocol_requirements.get("required_artifacts") if isinstance(protocol_requirements.get("required_artifacts"), list) else []
    required_artifacts = ["run_config", "raw_metrics_jsonl", "seed_variance_table"]
    if main_table_enabled:
        required_artifacts.insert(0, "main_results_table")
    required_artifacts.append("ablation_table")
    required_artifacts.extend(
        [
            "route_rate_sweep_table",
            "quality_cost_frontier_figure",
            "per_dataset_breakdown_table",
            "cost_utility_tradeoff_table",
            "difficulty_breakdown_table",
            "routing_analysis",
            "latency_tokens_table",
            "simple_case_degradation",
            "calibration_reliability",
        ]
    )
    if visualization_enabled:
        required_artifacts.append("metric_trajectory_figure")
    required_artifacts = _unique_non_empty(required_artifacts + [str(item).removesuffix(".json").removesuffix(".jsonl") for item in protocol_artifacts])

    reviewer_objections = [
        "Are the datasets real benchmarks rather than synthetic probes?",
        "Are baselines current, fairly tuned, and compute matched?",
        "Do ablations isolate the claimed mechanism instead of only showing a metric delta?",
        "Are improvements stable across seeds with an explicit statistical test?",
        "Does the manuscript state proxy limitations without overclaiming?",
    ]
    if evidence_tier == "formal_proxy":
        reviewer_objections.insert(0, "Proxy evidence is not enough for a top-venue performance claim.")
    if evidence_tier == "bootstrap_probe":
        reviewer_objections.insert(0, "Bootstrap evidence only proves the pipeline can execute, not the scientific claim.")
    if evidence_tier == "benchmark_plan":
        reviewer_objections.insert(
            0,
            "A full benchmark artifact package with full_benchmark_completed=true is required before manuscript claims.",
        )
    if not paper_allowed:
        reviewer_objections.insert(
            0,
            f"Claim route is {claim_route.get('route')}; do not write a top-tier manuscript until missing route requirements are resolved.",
        )
    if recipe_blockers:
        blocked_names = [
            _non_empty_text(item.get("name") if isinstance(item, dict) else item)
            for item in recipe_blockers
        ]
        blocked_names = [name for name in blocked_names if name]
        reviewer_objections.insert(
            0,
            "Requested benchmark target(s) require dedicated executable recipes before GPU execution: "
            + (", ".join(blocked_names[:4]) or "unspecified target"),
        )

    narrative_spine = [
        f"Gap: {_non_empty_text(parsed.get('existing_weakness') or parsed.get('problem_statement'))[:220]}",
        f"Mechanism: {method_name} targets the gap by {_non_empty_text(method.get('definition') or method.get('one_line'))[:220]}",
        f"Evidence: compare {method_name} against {', '.join(baselines[:3]) or 'named baselines'} on {', '.join(datasets[:3]) or 'named datasets'} using {metric}.",
        f"Limitation: evidence tier is {evidence_tier}; synthetic or bootstrap evidence must not be sold as a full benchmark.",
    ]

    manifest = _benchmark_manifest(parsed, plan, codebase=codebase, scaffold_kind=scaffold_kind)
    manifest["claim_route"] = {
        "route": claim_route.get("route"),
        "required_evidence_level": claim_route.get("required_evidence_level"),
        "paper_allowed": paper_allowed,
    }

    return {
        "claim_to_validate": claim[:500],
        "evidence_tier": evidence_tier,
        "publication_ready": False,
        "blocks_manuscript": blocks_manuscript,
        "minimum_seeds": minimum_seeds,
        "benchmark_protocol": benchmark_protocol,
        "required_datasets": datasets,
        "required_real_benchmarks": real_datasets,
        "required_models": model_targets,
        "required_baselines": baselines,
        "required_ablations": _planned_ablations(method, plan),
        "primary_metric": metric,
        "metric_direction": (
            "higher"
            if any(token in metric.lower() for token in ("utility", "accuracy", "score", "reward"))
            else "lower" if any(token in metric.lower() for token in ("loss", "error", "latency", "cost"))
            else "higher"
        ),
        "statistical_test": "paired bootstrap confidence interval plus paired permutation test across seeds/tasks when required by the benchmark protocol",
        "full_benchmark_examples_policy": "benchmark_specific_official_or_materialized_full_split",
        "full_benchmark_min_examples": None,
        "full_benchmark_min_datasets": len(protocol_requirements.get("required_dataset_names", []) or real_datasets),
        "full_benchmark_min_models": len(protocol_requirements.get("required_model_names", []) or model_targets),
        "full_benchmark_min_baselines": len(protocol_requirements.get("required_baseline_names", []) or baselines),
        "global_numeric_thresholds_allowed": False,
        "require_statistical_significance": EXPERIMENT_FULL_BENCHMARK_REQUIRE_SIGNIFICANCE,
        "require_strongest_baseline_win": EXPERIMENT_FULL_BENCHMARK_REQUIRE_STRONGEST_WIN,
        "required_analyses": [
            "ablation_table",
            "route_rate_or_budget_sweep_for_routing_methods",
            "quality_cost_frontier",
            "per_dataset_breakdown",
            "difficulty_breakdown",
            "pairwise_vs_strongest_deployable_baseline",
        ],
        "required_artifacts": required_artifacts,
        "benchmark_manifest": manifest,
        "claim_route": claim_route,
        "claim_strength": claim_route.get("claim_strength"),
        "evidence_stage_policy": {
            "sanity_cannot_support_paper_claims": True,
            "full_benchmark_required_for_paper_claims": True,
            "benchmark_plan_blocks_until_full_artifacts": True,
            "contract_revision_required_to_change_locked_fields": True,
        },
        "quality_gates": {
            "has_real_benchmark": bool(real_datasets),
            "has_real_model": bool(model_targets),
            "generated_runner_supported": plan.get("generated_runner_supported") is not False,
            "benchmark_recipe_blockers": recipe_blockers or [],
            "baseline_count": len(baselines),
            "claim_route": claim_route.get("route"),
            "claim_strength": claim_route.get("claim_strength"),
            "route_missing": claim_route.get("missing", []),
            "requires_ablation_table": True,
            "requires_seed_variance": True,
            "requires_full_benchmark_package": bool(
                evidence_tier == "benchmark_plan" and paper_allowed
            ),
            "minimum_seeds": minimum_seeds,
            "benchmark_protocol": benchmark_protocol,
            "full_benchmark_examples_policy": "benchmark_specific_official_or_materialized_full_split",
            "full_benchmark_min_examples": None,
            "full_benchmark_min_datasets": len(protocol_requirements.get("required_dataset_names", []) or real_datasets),
            "full_benchmark_min_models": len(protocol_requirements.get("required_model_names", []) or model_targets),
            "full_benchmark_min_baselines": len(protocol_requirements.get("required_baseline_names", []) or baselines),
            "global_numeric_thresholds_allowed": False,
            "require_statistical_significance": EXPERIMENT_FULL_BENCHMARK_REQUIRE_SIGNIFICANCE,
            "require_strongest_baseline_win": EXPERIMENT_FULL_BENCHMARK_REQUIRE_STRONGEST_WIN,
            "requires_quality_cost_frontier": True,
            "requires_route_rate_sweep_for_routing_methods": True,
            "requires_per_dataset_breakdown": True,
            "requires_difficulty_breakdown": True,
            "manuscript_allowed": not blocks_manuscript,
            "synthetic_fallback_allowed": bool(EXPERIMENT_ALLOW_SYNTHETIC_FALLBACK),
        },
        "reviewer_objections": reviewer_objections,
        "paper_intent": {
            "central_claim": claim[:500],
            "claim_route": claim_route.get("route"),
            "claim_strength": claim_route.get("claim_strength"),
            "required_evidence_level": claim_route.get("required_evidence_level"),
            "target_venue": "top-tier ML venue",
            "reader_takeaway": (
                f"{method_name} should be judged by whether it improves {metric} "
                "for the stated mechanism under fair baselines and ablations."
            ),
            "narrative_spine": narrative_spine,
        },
        "codebase_url": codebase.get("url") or "scratch",
        "scaffold_kind": scaffold_kind,
    }


def _normalize_success_criteria(success: dict, plan: dict, contract: dict) -> dict:
    success = dict(success or {})
    metrics = plan.get("metrics", {}) if isinstance(plan, dict) else {}
    primary_metric = ""
    if isinstance(metrics, dict):
        primary_metric = _non_empty_text(metrics.get("primary") or metrics.get("name"))
    elif isinstance(metrics, list):
        primary_metric = _named_values(metrics, keys=("name",))[0] if _named_values(metrics, keys=("name",)) else ""
    success.setdefault("metric_name", primary_metric or contract.get("primary_metric") or "primary_score")
    success.setdefault("metric_direction", contract.get("metric_direction") or "higher")
    success.setdefault("exciting", 1.0)
    success.setdefault("solid", 0.7)
    success.setdefault("disappointing", 0.1)

    existing_contract = success.get("publication_evidence_contract") or success.get("publication_evidence")
    if not isinstance(existing_contract, dict):
        existing_contract = {}
    merged_contract = {**contract, **existing_contract}
    success["publication_evidence_contract"] = merged_contract
    for key in (
        "evidence_tier",
        "publication_ready",
        "blocks_manuscript",
        "minimum_seeds",
        "required_datasets",
        "required_models",
        "required_baselines",
        "required_ablations",
        "statistical_test",
        "required_artifacts",
        "benchmark_manifest",
        "benchmark_protocol",
        "claim_route",
        "claim_strength",
        "reviewer_objections",
        "paper_intent",
        "quality_gates",
    ):
        if key in merged_contract:
            success[key] = merged_contract[key]
    return success


def _default_dataset_name(parsed: dict) -> str:
    corpus = " ".join(
        [
            _non_empty_text(parsed.get("title")),
            _non_empty_text((parsed.get("proposed_method") or {}).get("type")),
        ]
    ).lower()
    if any(token in corpus for token in ("gpu", "cuda", "systems_validation", "smoke")):
        return "synthetic_remote_gpu_probe"
    return "synthetic_stress_test"


def _enrich_proposed_method(parsed: dict, plan: dict) -> dict:
    method = dict(parsed.get("proposed_method") or {})
    title = _non_empty_text(parsed.get("title")) or f"insight_{parsed.get('id', 'unknown')}"
    if not _non_empty_text(method.get("name")):
        method["name"] = title.split(" as ", 1)[0][:120]
    if not _non_empty_text(method.get("type")):
        method["type"] = _non_empty_text(parsed.get("mechanism_type")) or "hypothesis"
    if not _non_empty_text(method.get("one_line")):
        method["one_line"] = title[:200]
    if not _non_empty_text(method.get("definition")):
        definition_bits = [f"Hypothesis: {title}."]
        problem = _non_empty_text(parsed.get("problem_statement") or parsed.get("existing_weakness"))
        if problem:
            definition_bits.append(problem[:280])
        procedure = _non_empty_text(plan.get("procedure"))
        if procedure:
            definition_bits.append(f"Operationalization: {procedure[:420]}")
        method["definition"] = " ".join(bit for bit in definition_bits if bit).strip()
    return method


def _enrich_experimental_plan(
    parsed: dict,
    method: dict,
    *,
    llm_scope: dict | None = None,
) -> dict:
    plan = dict(parsed.get("experimental_plan") or {})

    baseline_names = _unique_non_empty(
        _named_values(plan.get("baselines"), keys=("name", "model"))
        + _named_values(plan.get("models"), keys=("name", "model"))
        + _named_values(parsed.get("supporting_papers"), keys=("name",))
    )
    if len(baseline_names) < 2:
        method_name = _non_empty_text(method.get("name")) or "candidate_method"
        baseline_names.extend(
            [
                f"{method_name}_reference_baseline",
                f"{method_name}_ablation",
            ]
        )
        baseline_names = _unique_non_empty(baseline_names)
    plan["baselines"] = [{"name": name} for name in baseline_names[:4]]

    if not isinstance(parsed.get("evidence_plan"), dict) or not parsed.get("evidence_plan"):
        parsed["evidence_plan"] = build_evidence_plan(
            {**parsed, "proposed_method": method, "experimental_plan": plan}
        )

    # Re-forging must reuse the locked scientific design. Re-running the
    # benchmark designer spends the replacement grant and can move the target
    # after observing an operational failure.
    existing_design = (
        plan.get("benchmark_design_contract")
        if isinstance(plan.get("benchmark_design_contract"), dict)
        else {}
    )
    if (
        plan.get("benchmark_design_status") == DESIGN_STATUS_RESOLVED
        and existing_design.get("status") == DESIGN_STATUS_RESOLVED
    ):
        benchmark_design = existing_design
    else:
        benchmark_design = build_benchmark_design_contract(
            parsed,
            method,
            plan,
            llm_scope=llm_scope,
        )
    plan = apply_benchmark_design_contract(plan, benchmark_design)
    design_resolved = plan.get("benchmark_design_status") == DESIGN_STATUS_RESOLVED

    dataset_names = _unique_non_empty(_named_values(plan.get("datasets"), keys=("name", "dataset", "hf_dataset")))
    if design_resolved and (not dataset_names or (EXPERIMENT_REQUIRE_REAL_BENCHMARK and all(_looks_like_synthetic_dataset(name) for name in dataset_names))):
        real_targets = _default_real_benchmark_targets({**parsed, "proposed_method": method})
        dataset_names = [str(row.get("name") or row.get("hf_dataset")) for row in real_targets if row.get("name") or row.get("hf_dataset")]
        if real_targets:
            plan["benchmark_targets"] = real_targets
    if dataset_names:
        plan["datasets"] = [{"name": name} for name in dataset_names[:4]]

    metrics = plan.get("metrics")
    if isinstance(metrics, dict):
        primary_metric = _non_empty_text(metrics.get("primary") or metrics.get("name"))
        normalized_metrics = dict(metrics)
    elif isinstance(metrics, list):
        metric_names = _unique_non_empty(_named_values(metrics, keys=("name",)))
        primary_metric = metric_names[0] if metric_names else ""
        normalized_metrics = {"primary": primary_metric}
        if len(metric_names) > 1:
            normalized_metrics["secondary"] = metric_names[1:]
    else:
        primary_metric = _non_empty_text(metrics)
        normalized_metrics = {"primary": primary_metric} if primary_metric else {}
    if not _non_empty_text(normalized_metrics.get("primary")):
        normalized_metrics["primary"] = _fallback_metric_name(parsed, plan)
    plan["metrics"] = normalized_metrics

    compute = dict(plan.get("compute_budget") or {}) if isinstance(plan.get("compute_budget"), dict) else {}
    gpu_hours = (
        compute.get("total_gpu_hours")
        or compute.get("gpu_hours")
        or compute.get("gpu")
    )
    inferred_resource = _non_empty_text(parsed.get("resource_class")) or infer_resource_class(
        {
            **parsed,
            "proposed_method": method,
            "experimental_plan": plan,
        }
    )
    if gpu_hours in (None, "", "unknown") and inferred_resource != "cpu":
        gpu_hours = 24.0 if inferred_resource == "gpu_large" else 4.0
    elif inferred_resource == "cpu" and gpu_hours in (None, "", "unknown"):
        gpu_hours = 0.0
    if gpu_hours not in (None, ""):
        compute["total_gpu_hours"] = gpu_hours
    plan["compute_budget"] = compute
    plan = _ensure_real_benchmark_plan(parsed, method, plan, inferred_resource)

    ablations = _planned_ablations(method, plan)
    plan["ablations"] = [{"name": name} for name in ablations]
    publication_contract = _publication_evidence_contract(
        {**parsed, "proposed_method": method},
        plan,
        evidence_plan=parsed.get("evidence_plan") if isinstance(parsed.get("evidence_plan"), dict) else {},
        scaffold_kind="planned",
    )
    plan["publication_evidence_contract"] = publication_contract
    plan["paper_intent"] = publication_contract.get("paper_intent", {})
    return plan


def _autofill_experiment_contracts(
    insight: dict,
    *,
    llm_scope: dict | None = None,
) -> dict:
    parsed = _parse_insight_fields(insight)
    method = _enrich_proposed_method(parsed, dict(parsed.get("experimental_plan") or {}))
    plan = _enrich_experimental_plan(parsed, method, llm_scope=llm_scope)
    parsed["proposed_method"] = method
    parsed["experimental_plan"] = plan
    explicit_resource = _non_empty_text(parsed.get("resource_class"))
    inferred = explicit_resource or infer_resource_class(parsed)
    if EXPERIMENT_REQUIRE_REAL_BENCHMARK and not explicit_resource and _model_target_names(plan):
        model_text = " ".join(_model_target_names(plan)).lower()
        model_targets = plan.get("model_targets") if isinstance(plan.get("model_targets"), list) else []
        any_model_allows_cpu = any(
            isinstance(row, dict) and (row.get("cpu_allowed") or row.get("requires_cuda") is False)
            for row in model_targets
        )
        if not any_model_allows_cpu and any(token in model_text for token in ("qwen", "llama", "mistral", "mixtral", "gemma", "phi")):
            inferred = "gpu_large"
            compute = dict(plan.get("compute_budget") or {}) if isinstance(plan.get("compute_budget"), dict) else {}
            if not compute.get("total_gpu_hours"):
                compute["total_gpu_hours"] = 24.0
                plan["compute_budget"] = compute
    parsed["resource_class"] = inferred
    return parsed


def _persist_enriched_insight(insight_id: int, parsed: dict) -> None:
    agenda_id = int(parsed.get("agenda_id") or 0)
    if agenda_id <= 0:
        raise ValueError("enriched insight persistence requires agenda scope")
    db.execute(
        """
        UPDATE deep_insights
        SET proposed_method=?, experimental_plan=?, evidence_plan=?, resource_class=?, updated_at=CURRENT_TIMESTAMP
        WHERE id=? AND agenda_id=?
        """,
        (
            json.dumps(parsed.get("proposed_method") or {}, ensure_ascii=False),
            json.dumps(parsed.get("experimental_plan") or {}, ensure_ascii=False),
            json.dumps(parsed.get("evidence_plan") or {}, ensure_ascii=False),
            parsed.get("resource_class") or "cpu",
            insight_id,
            agenda_id,
        ),
    )
    db.commit()


def _safe_rmtree(path: Path | str) -> None:
    target = Path(path)
    if not target.exists():
        return

    def _retry_with_write_permission(func, value, _exc_info):
        try:
            os.chmod(value, 0o700)
            func(value)
        except FileNotFoundError:
            return

    shutil.rmtree(target, onerror=_retry_with_write_permission)


def _checkpoint_run_state(
    run_id: int,
    *,
    agenda_id: int,
    phase: str,
    workdir: Path | str | None = None,
    codebase: dict | None = None,
    program_md: str | None = None,
    proxy_config: dict | None = None,
    success_criteria: dict | None = None,
    baseline_metric_name: str | None = None,
) -> None:
    fields: dict[str, object] = {
        "status": "scaffolding",
        "phase": phase,
    }
    if workdir is not None:
        fields["workdir"] = str(workdir)
    if codebase is not None:
        fields["codebase_url"] = codebase.get("url", "scratch")
        fields["codebase_ref"] = codebase.get("name", "")
    if program_md is not None:
        fields["program_md"] = program_md
    if proxy_config is not None:
        fields["proxy_config"] = json.dumps(proxy_config)
    if success_criteria is not None:
        fields["success_criteria"] = json.dumps(success_criteria)
    if baseline_metric_name is not None:
        fields["baseline_metric_name"] = baseline_metric_name

    assignments = ", ".join(f"{key}=?" for key in fields)
    params = list(fields.values()) + [run_id, agenda_id]
    db.execute(
        f"UPDATE experiment_runs SET {assignments} WHERE id=? AND agenda_id=?",
        tuple(params),
    )
    db.commit()


def _git_binary() -> str | None:
    return shutil.which("git")


def _code_dir_has_content(code_dir: Path) -> bool:
    return code_dir.exists() and any(code_dir.iterdir())


def _github_archive_urls(repo_url: str) -> list[str]:
    parsed = urllib.parse.urlparse((repo_url or "").strip())
    if parsed.netloc not in {"github.com", "www.github.com"}:
        return []
    path = parsed.path.strip("/").removesuffix(".git")
    parts = [part for part in path.split("/") if part]
    if len(parts) < 2:
        return []
    owner, repo = parts[0], parts[1]
    base = f"https://github.com/{owner}/{repo}/archive/refs/heads"
    return [f"{base}/main.zip", f"{base}/master.zip"]


def _download_repo_archive(repo_url: str, code_dir: Path) -> bool:
    for archive_url in _github_archive_urls(repo_url):
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_root = Path(tmpdir)
                archive_path = tmp_root / "repo.zip"
                extract_dir = tmp_root / "extract"
                extract_dir.mkdir(parents=True, exist_ok=True)
                with urllib.request.urlopen(archive_url, timeout=30) as response:
                    archive_path.write_bytes(response.read())
                with zipfile.ZipFile(archive_path) as zf:
                    zf.extractall(extract_dir)
                roots = [path for path in extract_dir.iterdir() if path.is_dir()]
                source_root = roots[0] if roots else extract_dir
                for child in source_root.iterdir():
                    target = code_dir / child.name
                    if target.exists():
                        if target.is_dir():
                            _safe_rmtree(target)
                        else:
                            target.unlink()
                    shutil.move(str(child), str(target))
            return _code_dir_has_content(code_dir)
        except Exception as exc:
            print(f"[FORGE] Archive fetch failed for {archive_url}: {exc}", flush=True)
    return False


def _codebase_has_expected_entrypoint(code_dir: Path, codebase: dict) -> bool:
    expected = (codebase.get("main_train_file") or "").strip()
    if not expected:
        return _code_dir_has_content(code_dir)
    expected_path = code_dir / expected
    return expected_path.exists()


def _candidate_train_entrypoints(code_dir: Path) -> list[Path]:
    """Heuristically locate a plausible training entrypoint inside a cloned repo."""
    if not code_dir.exists():
        return []

    candidates: list[Path] = []
    for path in code_dir.rglob("train.py"):
        try:
            path.relative_to(code_dir / ".git")
            continue
        except ValueError:
            pass
        rel = path.relative_to(code_dir).as_posix()
        if rel.startswith(".git/"):
            continue
        candidates.append(path)

    # Prefer conventional locations first (stable ordering for deterministic picks).
    preference = (
        "train.py",
        "src/train.py",
        "scripts/train.py",
        "training/train.py",
    )
    rank = {name: idx for idx, name in enumerate(preference)}

    def sort_key(p: Path) -> tuple[int, int, str]:
        rel = p.relative_to(code_dir).as_posix()
        return (rank.get(rel, 999), len(rel), rel)

    candidates.sort(key=sort_key)
    return candidates


def repair_codebase_entrypoint(code_dir: Path, codebase: dict) -> dict:
    """If the declared train entrypoint is missing, try to infer a better one from disk."""
    if (codebase or {}).get("url") in {None, "", "scratch"}:
        return codebase

    repaired = dict(codebase)
    if _codebase_has_expected_entrypoint(code_dir, repaired):
        return repaired

    candidates = _candidate_train_entrypoints(code_dir)
    if not candidates:
        return repaired

    chosen = candidates[0]
    rel = chosen.relative_to(code_dir).as_posix()
    repaired["main_train_file"] = rel

    eval_cmd = _non_empty_text(repaired.get("main_eval_command"))
    if not eval_cmd or eval_cmd.lower() in {"python train.py", "python ./train.py"}:
        repaired["main_eval_command"] = f"python {rel}"

    return repaired


def _scratch_codebase(reason: str = "") -> dict:
    return {
        "url": "scratch",
        "name": "minimal",
        "reason": reason or "selected repository was unsuitable for the requested experiment",
        "main_train_file": "train.py",
        "main_eval_command": "python train.py",
        "real_benchmark_runner": True,
    }


def _normalize_codebase_metadata(codebase: dict) -> dict:
    normalized = dict(codebase or {})
    repo_url = _non_empty_text(normalized.get("url"))
    if repo_url and repo_url != "scratch":
        placeholder_values = {"scratch", "minimal", "n/a", "none", "unknown"}
        main_train_file = _non_empty_text(normalized.get("main_train_file")).lower()
        if main_train_file in placeholder_values:
            normalized["main_train_file"] = ""
        main_eval_command = _non_empty_text(normalized.get("main_eval_command")).lower()
        if main_eval_command in placeholder_values:
            normalized["main_eval_command"] = ""
    return normalized


def _resource_granted_proposer_json(
    system_prompt: str,
    user_prompt: str,
    *,
    llm_scope: dict,
    operation: str,
    max_tokens: int,
) -> tuple[dict, int, dict]:
    """Run one forge LLM operation under an already validated ResourceGrant."""
    required = ("agenda_id", "idea_id", "resource_grant_id", "stage")
    missing = [key for key in required if not llm_scope.get(key)]
    if missing:
        raise PermissionError(
            "resource-granted LLM scope is incomplete: " + ",".join(missing)
        )
    # A settled reservation cannot be reused - the ledger refuses the key and
    # the call dies as "idempotency key already exists with status settled" -
    # but it also cannot be replayed, because only the token accounting was
    # kept, not the model's answer. So a genuine retry has to be a new call
    # with a new key, while attempt 1 stays stable within itself.
    prior = db.fetchone(
        "SELECT COUNT(*) AS c FROM resource_grant_usage_reservations"
        " WHERE resource_grant_id=? AND idempotency_key LIKE ?",
        (int(llm_scope["resource_grant_id"]), f"{operation}:%"),
    )
    attempt = int(dict(prior or {}).get("c") or 0)
    digest = hashlib.sha256(
        "\n".join(
            (
                str(llm_scope["agenda_id"]),
                str(llm_scope["idea_id"]),
                str(llm_scope["resource_grant_id"]),
                operation,
                str(attempt),
                user_prompt,
            )
        ).encode("utf-8")
    ).hexdigest()
    result, tokens, route = call_llm_json_for_role(
        system_prompt,
        user_prompt,
        agenda_id=int(llm_scope["agenda_id"]),
        idea_id=int(llm_scope["idea_id"]),
        role="proposer",
        stage=str(llm_scope["stage"]),
        resource_grant_id=int(llm_scope["resource_grant_id"]),
        operation=operation,
        idempotency_key=f"{operation}:{digest}",
        prompt_version=configured_role_prompt_version("proposer"),
        max_tokens=max_tokens,
    )
    if not isinstance(result, dict):
        raise ValueError(f"{operation} must return a JSON object")
    return result, tokens, route


def scout_codebase(insight: dict, *, llm_scope: dict | None = None) -> dict:
    """Find the best codebase for implementing a hypothesis."""
    if llm_scope is None:
        raise PermissionError(
            "experiment codebase selection requires a ResourceGrant-backed "
            "LLM scope"
        )
    # The legacy agentic scout has its own unconstrained LLM calls. A
    # resource-granted forge uses only the routed implementation.
    return _scout_codebase_single_shot(insight, llm_scope=llm_scope)


def _scout_codebase_single_shot(
    insight: dict,
    *,
    llm_scope: dict | None = None,
) -> dict:
    """Legacy single-shot LLM repo suggestion (fallback)."""
    parsed = _parse_insight_fields(insight)
    method = parsed.get("proposed_method", {})
    plan = _ensure_real_benchmark_plan(
        parsed,
        method,
        parsed.get("experimental_plan", {}),
        parsed.get("resource_class") or infer_resource_class(parsed),
    )
    evidence_plan = parsed.get("evidence_plan", {})
    node_ids = parsed.get("source_node_ids", [])

    context_parts = [f"# Method to Implement\n"]
    context_parts.append(f"Name: {method.get('name', 'Unknown')}")
    context_parts.append(f"Type: {method.get('type', 'unknown')}")
    context_parts.append(f"Summary: {method.get('one_line', '')}")
    if method.get("definition"):
        context_parts.append(f"Definition: {method['definition'][:600]}")

    context_parts.append(f"\n# Experimental Plan")
    if plan.get("baselines"):
        context_parts.append("Baselines:")
        for b in plan["baselines"][:5]:
            name = b.get("name", b) if isinstance(b, dict) else str(b)
            model = b.get("model", "") if isinstance(b, dict) else ""
            context_parts.append(f"  - {name} {model}")
    if plan.get("datasets"):
        context_parts.append("Datasets:")
        for d in plan["datasets"][:5]:
            name = d.get("name", d) if isinstance(d, dict) else str(d)
            context_parts.append(f"  - {name}")

    context_parts.append(f"\n# Research Area")
    context_parts.append(f"Taxonomy nodes: {', '.join(node_ids[:5])}")

    if parsed.get("problem_statement"):
        context_parts.append(f"\n# Problem")
        context_parts.append(parsed["problem_statement"][:400])

    graph_methods = db.fetchall("""
        SELECT DISTINCT ge.canonical_name, ge.description
        FROM graph_entities ge
        JOIN paper_entity_mentions pem ON pem.entity_id = ge.id
        WHERE ge.entity_type = 'method'
          AND pem.node_id IN ({})
        ORDER BY ge.canonical_name
        LIMIT 15
    """.format(",".join("?" * len(node_ids))), tuple(node_ids)) if node_ids else []

    if graph_methods:
        context_parts.append("\n# Known Methods in This Area (from knowledge graph)")
        for m in graph_methods:
            desc = f" — {m['description'][:80]}" if m.get("description") else ""
            context_parts.append(f"  - {m['canonical_name']}{desc}")

    prompt = "\n".join(context_parts)

    if llm_scope is None:
        raise PermissionError(
            "experiment code scout requires a ResourceGrant-backed LLM scope"
        )
    result, _, route = _resource_granted_proposer_json(
        CODE_SCOUT_SYSTEM,
        prompt,
        llm_scope=llm_scope,
        operation="experiment_forge.code_scout",
        max_tokens=4000,
    )
    codebase = result.get("codebase", {"url": "scratch", "name": "minimal", "reason": "no suitable repo found"})
    normalized = _normalize_codebase_metadata(codebase)
    if route:
        normalized["llm_route"] = route
    return normalized


def setup_workspace(insight_id: int, run_id: int, codebase: dict, *, insight: dict | None = None) -> Path:
    """Create the experiment workspace directory with the codebase."""
    run_info = ensure_run_workspace(insight_id, run_id, insight=insight)
    workdir = Path(run_info["run_root"])
    code_dir = Path(run_info["code_root"])
    Path(run_info["results_root"]).mkdir(parents=True, exist_ok=True)
    Path(run_info["spec_root"]).mkdir(parents=True, exist_ok=True)
    Path(run_info["codex_root"]).mkdir(parents=True, exist_ok=True)
    url = codebase.get("url", "scratch")

    if url != "scratch" and not _code_dir_has_content(code_dir):
        if code_dir.exists():
            _safe_rmtree(code_dir)
        code_dir.mkdir(parents=True, exist_ok=True)
        git_bin = _git_binary()
        clone_ok = False
        if git_bin:
            try:
                subprocess.run(
                    [git_bin, "clone", "--depth", "1", url, str(code_dir)],
                    timeout=120,
                    capture_output=True,
                    check=True,
                )
                clone_ok = True
                print(f"[FORGE] Cloned {url} to {code_dir}", flush=True)
            except Exception as e:
                print(f"[FORGE] Clone failed for {url}: {e}. Trying archive fallback.", flush=True)
        else:
            print(f"[FORGE] git not available; trying archive fallback for {url}", flush=True)
        if not clone_ok and not _code_dir_has_content(code_dir):
            archive_ok = _download_repo_archive(url, code_dir)
            if archive_ok:
                print(f"[FORGE] Downloaded archive fallback for {url} into {code_dir}", flush=True)
            else:
                print(f"[FORGE] Archive fallback failed for {url}. Using scratch workspace.", flush=True)
    elif not code_dir.exists():
        code_dir.mkdir(parents=True, exist_ok=True)

    return workdir


def generate_scaffold(
    insight: dict,
    codebase: dict,
    workdir: Path,
    *,
    llm_scope: dict | None = None,
) -> dict:
    """Generate program.md, evaluate.py, and success_criteria.json using LLM."""
    parsed = _parse_insight_fields(insight)
    method = parsed.get("proposed_method", {})
    plan = parsed.get("experimental_plan", {})
    evidence_plan = parsed.get("evidence_plan", {})
    resource_class = parsed.get("resource_class") or infer_resource_class(parsed)
    publication_contract = _publication_evidence_contract(
        parsed,
        plan,
        codebase=codebase,
        evidence_plan=evidence_plan,
        scaffold_kind="planned",
    )
    benchmark_manifest = publication_contract.get("benchmark_manifest") or _benchmark_manifest(
        parsed,
        plan,
        codebase=codebase,
        scaffold_kind="planned",
    )
    benchmark_protocol = publication_contract.get("benchmark_protocol") or benchmark_manifest.get("benchmark_protocol") or {}

    code_dir = workdir / "code"
    code_structure = ""
    if code_dir.exists():
        try:
            result = subprocess.run(
                ["find", str(code_dir), "-name", "*.py", "-maxdepth", "3"],
                capture_output=True, text=True, timeout=10
            )
            files = [f.replace(str(code_dir) + "/", "") for f in result.stdout.strip().split("\n") if f][:30]
            code_structure = "\n".join(f"  {f}" for f in files)
        except Exception:
            code_structure = "(could not list files)"

    prompt_parts = [
        f"# Proposed Method",
        f"Name: {method.get('name', '?')}",
        f"Type: {method.get('type', '?')}",
        f"Summary: {method.get('one_line') or ''}",
        f"Definition:\n{(method.get('definition') or 'N/A')[:800]}",
    ]
    if method.get("pseudocode"):
        prompt_parts.append(f"Pseudocode:\n{(method.get('pseudocode') or '')[:500]}")
    if method.get("key_properties"):
        prompt_parts.append(f"Key Properties: {json.dumps((method.get('key_properties') or [])[:5])}")
    if method.get("hyperparameters"):
        prompt_parts.append(f"Hyperparameters: {json.dumps((method.get('hyperparameters') or [])[:5])}")

    prompt_parts.append(f"\n# Experimental Plan")
    prompt_parts.append(f"Baselines: {json.dumps(plan.get('baselines', []))[:500]}")
    prompt_parts.append(f"Datasets: {json.dumps(plan.get('datasets', []))[:500]}")
    prompt_parts.append(f"Metrics: {json.dumps(plan.get('metrics', {}))[:300]}")
    prompt_parts.append(f"Expected Results: {json.dumps(plan.get('expected_results', {}))[:300]}")
    prompt_parts.append(f"Resource class: {resource_class}")
    if resource_class != "cpu":
        prompt_parts.append(
            "GPU requirement: generated code must use PyTorch CUDA, print peak_vram_mb, "
            "and avoid numpy/scipy-only proxy scripts."
        )
    if evidence_plan:
        prompt_parts.append(f"\n# Adaptive Evidence Plan")
        prompt_parts.append(json.dumps(evidence_plan, ensure_ascii=False)[:1200])
        prompt_parts.append("Honor this plan. Do not invent ablations or visual analyses when they are disabled.")
    prompt_parts.append(f"\n# Publication Evidence Contract")
    prompt_parts.append(json.dumps(publication_contract, ensure_ascii=False)[:2400])
    prompt_parts.append(
        "The scaffold must make this contract operational. Bootstrap/proxy evidence must be labeled as such."
    )
    prompt_parts.append(f"\n# Benchmark Protocol")
    prompt_parts.append(json.dumps(benchmark_protocol, ensure_ascii=False)[:3000])
    prompt_parts.append(
        "This protocol is binding for paper evidence. If official benchmark instructions are unavailable, inspect/materialize the dataset and record the actual split and counts instead of applying global thresholds."
    )
    prompt_parts.append(f"\n# Benchmark Manifest")
    prompt_parts.append(json.dumps(benchmark_manifest, ensure_ascii=False)[:3000])
    prompt_parts.append(
        "Use the manifest to separate sanity execution from full benchmark execution. "
        "Do not let the sanity runner satisfy paper-evidence gates."
    )

    prompt_parts.append(f"\n# Codebase")
    prompt_parts.append(f"Repo: {codebase.get('url', 'scratch')} ({codebase.get('name', '')})")
    prompt_parts.append(f"Main train file: {codebase.get('main_train_file', 'train.py')}")
    prompt_parts.append(f"Eval command: {codebase.get('main_eval_command', 'python evaluate.py')}")
    if code_structure:
        prompt_parts.append(f"File structure:\n{code_structure}")

    prompt_parts.append(f"\n# Problem Context")
    prompt_parts.append((parsed.get("problem_statement") or "")[:300])
    prompt_parts.append(f"Weakness: {(parsed.get('existing_weakness') or '')[:300]}")

    prompt = "\n".join(prompt_parts)

    used_fallback = False
    llm_route = None
    real_runner_required = bool(EXPERIMENT_REQUIRE_REAL_BENCHMARK and not EXPERIMENT_ALLOW_SYNTHETIC_FALLBACK)
    recipe_blocked = False
    if _plan_uses_executable_probe(plan):
        print("[FORGE] Executable benchmark probe detected; using deterministic scaffold.", flush=True)
        result = _fallback_scaffold(method, plan, codebase)
        used_fallback = True
        tokens = 0
    else:
        if llm_scope is None:
            raise PermissionError(
                "experiment scaffold requires a ResourceGrant-backed LLM scope"
            )
        result, tokens, llm_route = _resource_granted_proposer_json(
            SCAFFOLD_SYSTEM,
            prompt,
            llm_scope=llm_scope,
            operation="experiment_forge.scaffold",
            max_tokens=12000,
        )

    program_md = result.get("program_md", "")
    evaluate_py = result.get("evaluate_py", "")
    success = result.get("success_criteria", {})
    train_py = result.get("train_py", "")
    baseline_command_override = None

    if codebase.get("url") == "scratch" and len(train_py or "") <= 50:
        fallback = _fallback_scaffold(method, plan, codebase)
        train_py = fallback.get("train_py", train_py)
        success = success or fallback.get("success_criteria", {})
        used_fallback = True

    if real_runner_required and not _train_py_is_real_benchmark_runner(train_py):
        metric_name = _metric_name_from_success_or_plan(success, plan)
        try:
            train_py = _real_llm_benchmark_train_py(
                method_name=str(method.get("name") or "candidate_method"),
                metric_name=metric_name,
                plan=plan,
            )
        except ValueError as exc:
            train_py = _benchmark_recipe_blocker_train_py(
                metric_name=metric_name,
                error=str(exc),
                plan=plan,
            )
            recipe_blocked = True
        success = success or {}
        success["metric_name"] = metric_name
        success.setdefault("metric_direction", "higher")
        success.setdefault("exciting", 0.02)
        success.setdefault("solid", 0.01)
        success.setdefault("disappointing", 0.0)
        baseline_command_override = "python train.py"
        used_fallback = True
        print("[FORGE] Real-benchmark guard injected Hugging Face benchmark runner", flush=True)

    if resource_class != "cpu" and not _train_py_uses_cuda(train_py):
        metric_name = _metric_name_from_success_or_plan(success, plan)
        if real_runner_required:
            try:
                train_py = _real_llm_benchmark_train_py(
                    method_name=str(method.get("name") or "candidate_method"),
                    metric_name=metric_name,
                    plan=plan,
                )
            except ValueError as exc:
                train_py = _benchmark_recipe_blocker_train_py(
                    metric_name=metric_name,
                    error=str(exc),
                    plan=plan,
                )
                recipe_blocked = True
            print("[FORGE] GPU guard replaced non-CUDA scaffold with real LLM benchmark runner", flush=True)
        else:
            train_py = _gpu_bootstrap_train_py(
                method_name=str(method.get("name") or "gpu_method"),
                metric_name=metric_name,
                resource_class=resource_class,
            )
            print(
                f"[FORGE] GPU scaffold guard injected CUDA bootstrap for {resource_class}",
                flush=True,
            )
        success = success or {}
        success.setdefault("metric_name", metric_name)
        success.setdefault("metric_direction", "higher")
        success.setdefault("exciting", 1.0)
        success.setdefault("solid", 0.7)
        success.setdefault("disappointing", 0.1)
        baseline_command_override = "python train.py"
        used_fallback = True

    if real_runner_required and _train_py_is_real_benchmark_runner(train_py):
        baseline_command_override = "python train.py"

    scaffold_kind = (
        "real_benchmark_recipe_blocked"
        if recipe_blocked
        else "full_benchmark_compiled" if (used_fallback and real_runner_required)
        else "bootstrap_probe" if used_fallback
        else "planned"
    )
    publication_contract = _publication_evidence_contract(
        parsed,
        plan,
        codebase=codebase,
        evidence_plan=evidence_plan,
        scaffold_kind=scaffold_kind,
    )
    benchmark_manifest = publication_contract.get("benchmark_manifest") or _benchmark_manifest(
        parsed,
        plan,
        codebase=codebase,
        scaffold_kind=scaffold_kind,
    )
    benchmark_protocol = publication_contract.get("benchmark_protocol") or benchmark_manifest.get("benchmark_protocol") or {}
    success = _normalize_success_criteria(success, plan, publication_contract)

    spec_dir = workdir / "spec"
    spec_dir.mkdir(parents=True, exist_ok=True)
    (spec_dir / "program.md").write_text(program_md, encoding="utf-8")
    (spec_dir / "evaluate.py").write_text(evaluate_py, encoding="utf-8")
    (spec_dir / "success_criteria.json").write_text(
        json.dumps(success, indent=2), encoding="utf-8")
    (spec_dir / "benchmark_manifest.json").write_text(
        json.dumps(benchmark_manifest, indent=2), encoding="utf-8")
    (spec_dir / "benchmark_protocol.json").write_text(
        json.dumps(benchmark_protocol, indent=2), encoding="utf-8")

    code_dir = workdir / "code"
    code_dir.mkdir(parents=True, exist_ok=True)
    if train_py and len(train_py) > 50:
        (code_dir / "train.py").write_text(train_py, encoding="utf-8")
        if real_runner_required:
            (code_dir / "requirements.txt").write_text(_real_llm_requirements_txt(), encoding="utf-8")
        print(f"[FORGE] train.py written ({len(train_py)} chars)", flush=True)
    elif not list(code_dir.glob("*.py")):
        print(f"[FORGE] WARNING: No train.py and no code in {code_dir}. Loop will likely fail.", flush=True)

    return {
        "program_md": program_md,
        "evaluate_py": evaluate_py,
        "success_criteria": success,
        "publication_evidence_contract": publication_contract,
        "benchmark_manifest": benchmark_manifest,
        "benchmark_protocol": benchmark_protocol,
        "claim_route": publication_contract.get("claim_route", {}),
        "train_py_written": bool(train_py and len(train_py) > 50),
        "baseline_command_override": baseline_command_override,
        "tokens": tokens,
        "llm_route": llm_route,
    }


def _metric_name_from_success_or_plan(success: dict, plan: dict) -> str:
    if isinstance(success, dict) and success.get("metric_name"):
        return str(success["metric_name"])
    metrics = plan.get("metrics", {}) if isinstance(plan, dict) else {}
    if isinstance(metrics, dict) and metrics.get("primary"):
        return str(metrics["primary"])
    if isinstance(metrics, list) and metrics:
        first = metrics[0]
        if isinstance(first, dict) and first.get("name"):
            return str(first["name"])
        return str(first)
    return "gpu_probe_score"


def _optional_nonnegative_int(value, default: int = 0) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return int(default)
    return max(0, parsed)


def _real_benchmark_defaults(plan: dict) -> dict:
    targets = plan.get("benchmark_targets") if isinstance(plan.get("benchmark_targets"), list) else []
    normalized_targets = [
        _normalize_benchmark_target(row)
        for row in targets
        if not _looks_like_synthetic_dataset(
            (row.get("name") or row.get("hf_dataset") or row.get("dataset")) if isinstance(row, dict) else row
        )
    ]
    if not normalized_targets:
        for name in _named_values(plan.get("datasets"), keys=("name", "dataset", "hf_dataset")):
            if not _looks_like_synthetic_dataset(name):
                normalized_targets.append(_normalize_benchmark_target({"name": name}))
    if not normalized_targets:
        raise ValueError(
            "Cannot generate a paper-grade benchmark runner without explicit benchmark_targets or real datasets; "
            "run benchmark design/harness repair before execution."
        )
    blockers = [
        target.get("generated_runner_blocker")
        or f"{target.get('name') or target.get('hf_dataset') or 'benchmark'} is not executable by the generated runner."
        for target in normalized_targets
        if target.get("generated_runner_supported") is False
    ]
    if blockers:
        raise ValueError(
            "Cannot generate a paper-grade benchmark runner until requested targets have executable recipes: "
            + " | ".join(blockers)
        )
    target = normalized_targets[0]
    models = plan.get("model_targets") if isinstance(plan.get("model_targets"), list) else []
    valid_models = [row for row in models if isinstance(row, dict)]
    valid_models.sort(
        key=lambda row: (
            0 if bool(row.get("requires_cuda")) else 1,
            0 if "/" in str(row.get("hf_model") or row.get("model") or row.get("name") or "") else 1,
            0 if "qwen" in str(row.get("hf_model") or row.get("model") or row.get("name") or "").lower() else 1,
        )
    )
    model = valid_models[0] if valid_models else {}
    dataset_id = str(target.get("hf_dataset") or "").strip()
    if not dataset_id and target.get("direct_files"):
        first_direct = target.get("direct_files")[0] if target.get("direct_files") else {}
        dataset_id = _non_empty_text(first_direct.get("id") if isinstance(first_direct, dict) else "") or target.get("name")
    if (
        not target.get("derive_from_loaded_benchmarks")
        and not target.get("direct_files")
        and (not dataset_id or "/" not in dataset_id)
    ):
        raise ValueError(
            "Cannot generate a paper-grade benchmark runner for "
            f"{target.get('name') or 'benchmark'} without a concrete Hugging Face dataset id or direct file recipe."
        )
    model_id = model.get("hf_model") or model.get("model") or model.get("name") or EXPERIMENT_REAL_LLM_MODEL
    return {
        "targets": normalized_targets,
        "dataset_id": dataset_id,
        "dataset_config": target.get("config", EXPERIMENT_REAL_BENCHMARK_DATASET_CONFIG),
        "dataset_split": target.get("split") or "test",
        "question_field": target.get("question_field") or "question",
        "answer_field": target.get("answer_field") or "answer",
        "model_id": model_id,
        "model_targets": valid_models or ([model] if model else []),
        "model_requires_cuda": bool(model.get("requires_cuda")),
        "model_cpu_allowed": bool(model.get("cpu_allowed") or not model.get("requires_cuda")),
        "model_load_in_4bit": bool(model.get("load_in_4bit")),
        "max_examples": _optional_nonnegative_int(plan.get("max_eval_examples"), _optional_nonnegative_int(target.get("max_eval_examples"), 0)),
        "seeds": max(1, _optional_nonnegative_int(plan.get("minimum_seeds"), 1)),
        "baselines": _planned_baselines(plan),
        "ablations": _unique_non_empty(
            _named_values(plan.get("ablations"), keys=("name", "component", "factor"))
            + _STANDARD_REASONING_ABLATIONS
        ),
    }


def _real_llm_requirements_txt() -> str:
    return "\n".join(
        [
            "torch",
            "transformers>=4.42",
            "datasets>=2.19",
            "accelerate>=0.30",
            "bitsandbytes>=0.43; platform_system != 'Windows'",
            "modelscope>=1.15",
            "",
        ]
    )


def _benchmark_recipe_blocker_train_py(*, metric_name: str, error: str, plan: dict) -> str:
    blocker_payload = {
        "metric_name": str(metric_name),
        "metric_value": 0.0,
        "full_benchmark_completed": False,
        "error": str(error),
        "benchmark_recipe_blockers": plan.get("benchmark_recipe_blockers") or [],
    }
    blocker_json = json.dumps(blocker_payload, ensure_ascii=False).replace("'''", "\\u0027\\u0027\\u0027")
    return textwrap.dedent(f"""\
    import json
    import sys

    BLOCKER = json.loads(r'''{blocker_json}''')

    def main():
        print("BENCHMARK_STAGE: recipe_blocked " + json.dumps(BLOCKER, ensure_ascii=False), flush=True)
        print(json.dumps(BLOCKER, ensure_ascii=False), flush=True)
        sys.exit(2)

    if __name__ == "__main__":
        main()
    """)


def _executable_probe_train_py(*, method_name: str, metric_name: str, plan: dict) -> str:
    """Render the audited GSM8K runner for a manuscript-blocked V1 probe."""
    defaults = _real_benchmark_defaults(plan)
    target = defaults["targets"][0]
    target_key = _canonical_name(target.get("name") or target.get("hf_dataset"))
    dataset_key = _canonical_name(defaults.get("dataset_id"))
    if "gsm8k" not in target_key and "gsm8k" not in dataset_key:
        return _benchmark_recipe_blocker_train_py(
            metric_name=metric_name,
            error=(
                "the audited executable-probe runner currently supports GSM8K only; "
                f"received {target.get('name') or defaults.get('dataset_id')}"
            ),
            plan=plan,
        )

    payload = {
        "method_name": str(method_name),
        "metric_name": str(metric_name or "exact_match"),
        "model_id": str(defaults["model_id"]),
        "dataset_id": str(defaults["dataset_id"] or "openai/gsm8k"),
        "dataset_config": str(defaults.get("dataset_config") or "main"),
        "dataset_split": str(defaults.get("dataset_split") or "test"),
        "max_examples": int(defaults.get("max_examples") or 32),
        "seeds": max(1, int(defaults.get("seeds") or 1)),
        "procedure": _non_empty_text(plan.get("procedure"))[:1200],
    }
    encoded = json.dumps(payload, ensure_ascii=False).replace("'''", "\\u0027\\u0027\\u0027")
    return textwrap.dedent(f"""\
    import json
    import os
    import random
    import re
    import time
    from pathlib import Path

    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer


    DEFAULTS = json.loads(r'''{encoded}''')
    BASELINE_METHOD = "direct_answer_baseline"
    CANDIDATE_METHOD = "process_guided_candidate"


    def normalize_answer(value):
        text = str(value or "").strip().replace(",", "")
        if "####" in text:
            text = text.rsplit("####", 1)[-1].strip()
        matches = re.findall(r"-?\\d+(?:\\.\\d+)?", text)
        if not matches:
            return text.lower()
        number = matches[-1]
        try:
            parsed = float(number)
            return str(int(parsed)) if parsed.is_integer() else str(parsed)
        except ValueError:
            return number


    def render_prompt(tokenizer, question, method):
        if method == BASELINE_METHOD:
            instruction = "Solve the problem and give only the final numeric answer."
        else:
            procedure = DEFAULTS.get("procedure") or DEFAULTS["method_name"]
            instruction = (
                "Solve independently with an explicit step-by-step reasoning process, "
                "check each intermediate step, then end with '#### <numeric answer>'. "
                "The pre-registered process policy is: " + procedure
            )
        messages = [
            {{"role": "system", "content": instruction}},
            {{"role": "user", "content": str(question)}},
        ]
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return instruction + "\\n\\nProblem: " + str(question) + "\\nAnswer:"


    def main():
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the executable benchmark probe")
        started = time.time()
        model_id = os.getenv("DEEPGRAPH_BENCHMARK_MODEL", DEFAULTS["model_id"])
        dataset_id = os.getenv("DEEPGRAPH_BENCHMARK_DATASET", DEFAULTS["dataset_id"])
        dataset_config = os.getenv("DEEPGRAPH_BENCHMARK_DATASET_CONFIG", DEFAULTS["dataset_config"])
        requested = int(os.getenv("DEEPGRAPH_BENCHMARK_MAX_EXAMPLES", str(DEFAULTS["max_examples"])))
        cap = int(os.getenv("DEEPGRAPH_BENCHMARK_MAX_EXAMPLES_CAP", "64"))
        max_examples = min(requested if requested > 0 else DEFAULTS["max_examples"], cap)
        max_examples = max(1, max_examples)
        requested_seeds = int(os.getenv("DEEPGRAPH_BENCHMARK_SEEDS", str(DEFAULTS["seeds"])))
        seed_cap = int(os.getenv("DEEPGRAPH_BENCHMARK_SEEDS_CAP", "3"))
        seeds = list(range(max(1, min(requested_seeds, seed_cap))))

        dataset = load_dataset(dataset_id, dataset_config or None, split=DEFAULTS["dataset_split"])
        examples = list(dataset.select(range(min(max_examples, len(dataset)))))
        if not examples:
            raise RuntimeError("real benchmark dataset resolved to zero examples")

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()
        torch.cuda.reset_peak_memory_stats()

        methods = [BASELINE_METHOD, CANDIDATE_METHOD]
        correct = {{name: 0 for name in methods}}
        totals = {{name: 0 for name in methods}}
        token_totals = {{name: 0 for name in methods}}
        seed_results = []
        predictions = []
        for seed in seeds:
            random.seed(seed)
            torch.manual_seed(seed)
            seed_methods = {{}}
            for method in methods:
                seed_correct = 0
                seed_tokens = 0
                for index, row in enumerate(examples):
                    question = row["question"]
                    target = normalize_answer(row["answer"])
                    prompt = render_prompt(tokenizer, question, method)
                    inputs = tokenizer(prompt, return_tensors="pt")
                    input_ids = inputs["input_ids"].to(model.device)
                    attention_mask = inputs.get("attention_mask")
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(model.device)
                    with torch.inference_mode():
                        generated = model.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            max_new_tokens=160 if method == CANDIDATE_METHOD else 48,
                            do_sample=False,
                            pad_token_id=tokenizer.eos_token_id,
                        )
                    new_tokens = generated[0, input_ids.shape[1]:]
                    prediction_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
                    prediction = normalize_answer(prediction_text)
                    is_correct = prediction == target
                    seed_correct += int(is_correct)
                    seed_tokens += int(new_tokens.numel())
                    predictions.append({{
                        "seed": seed,
                        "index": index,
                        "method": method,
                        "prediction": prediction,
                        "target": target,
                        "correct": is_correct,
                    }})
                score = seed_correct / len(examples)
                correct[method] += seed_correct
                totals[method] += len(examples)
                token_totals[method] += seed_tokens
                seed_methods[method] = {{"metric_value": score, "exact_match": score}}
            seed_results.append({{"seed": seed, "methods": seed_methods}})

        per_method = {{}}
        for method in methods:
            score = correct[method] / totals[method]
            per_method[method] = {{
                "metric_value": score,
                "exact_match": score,
                "num_correct": correct[method],
                "num_examples": totals[method],
                "avg_new_tokens": token_totals[method] / totals[method],
            }}
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        summary = {{
            "primary_metric": "exact_match",
            "metric_name": "exact_match",
            "candidate_method": CANDIDATE_METHOD,
            "best_method": max(per_method, key=lambda name: per_method[name]["exact_match"]),
            "per_method": per_method,
            "seed_results": seed_results,
            "num_seeds": len(seeds),
            "dataset": {{"name": "GSM8K", "id": dataset_id, "split": DEFAULTS["dataset_split"], "num_examples": len(examples)}},
            "datasets": [{{"name": "GSM8K", "id": dataset_id, "split": DEFAULTS["dataset_split"], "num_examples": len(examples)}}],
            "model": {{"id": model_id, "backend": "transformers", "cuda": True}},
            "models": [{{"id": model_id, "backend": "transformers", "cuda": True}}],
            "peak_vram_mb": peak_vram_mb,
            "duration_seconds": time.time() - started,
            "load_failures": [],
            "probe_completed": True,
            "probe_only": True,
            "blocks_manuscript": True,
            "full_benchmark_completed": False,
            "label_fallback_used": False,
        }}
        results_dir = Path(__file__).resolve().parents[1] / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        (results_dir / "benchmark_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        with (results_dir / "raw_predictions.jsonl").open("w", encoding="utf-8") as handle:
            for row in predictions:
                handle.write(json.dumps(row, ensure_ascii=False) + "\\n")
        print(f"peak_vram_mb: {{peak_vram_mb:.1f}}")
        print("FINAL_RESULTS: " + json.dumps(summary, ensure_ascii=False))


    if __name__ == "__main__":
        main()
    """)


def _real_llm_benchmark_train_py(*, method_name: str, metric_name: str, plan: dict) -> str:
    """Render a benchmark runner only through an explicit implementation plugin.

    The historical CGGR/VOC template is isolated in the non-production example
    plugin. A generic candidate must provide its own audited runner plugin;
    silently mapping an unknown method to a topic runner would invalidate the
    benchmark contract.
    """
    if _plan_uses_executable_probe(plan or {}):
        return _executable_probe_train_py(
            method_name=method_name,
            metric_name=metric_name,
            plan=plan,
        )

    runner_plugin = str((plan or {}).get("runner_plugin") or "").strip()
    if runner_plugin != "example.cggr":
        return _benchmark_recipe_blocker_train_py(
            metric_name=metric_name,
            error="no explicit audited runner_plugin for the proposed method",
            plan=plan,
        )
    enabled = os.getenv("DEEPGRAPH_ENABLE_NONPROD_EXAMPLE_PLUGINS", "").strip().lower()
    if enabled not in {"1", "true", "yes"}:
        return _benchmark_recipe_blocker_train_py(
            metric_name=metric_name,
            error="example.cggr runner is disabled and non-production",
            plan=plan,
        )
    from plugins.examples.cggr.experiment_runner import (
        render_historical_benchmark_runner,
    )

    return render_historical_benchmark_runner(
        method_name=method_name,
        metric_name=metric_name,
        plan=plan,
    )
def _train_py_uses_cuda(train_py: str | None) -> bool:
    if not train_py:
        return False
    text = train_py.lower()
    return "torch" in text and ("cuda" in text or ".to(device" in text or ".cuda(" in text)


def _train_py_looks_like_proxy(train_py: str | None) -> bool:
    if not train_py:
        return True
    text = train_py.lower()
    proxy_markers = (
        "synthetic",
        "simulated",
        "random.randn",
        "torch.randn",
        "np.random",
        "reserve_vram",
        "gpu_workload_target",
        "cuda bootstrap",
        "toy",
        "dummy",
    )
    real_markers = ("load_dataset", "datasets", "from_pretrained", "benchmark")
    return any(marker in text for marker in proxy_markers) and not any(marker in text for marker in real_markers)


def _train_py_is_real_benchmark_runner(train_py: str | None) -> bool:
    if not train_py:
        return False
    text = train_py.lower()
    if _train_py_looks_like_proxy(train_py):
        return False
    required = ("load_dataset", "from_pretrained", "final_results:", "per_method", "candidate_method")
    if not all(marker in text for marker in required):
        return False
    if "pass\n" in text or "todo" in text:
        return False
    return "automodelforcausallm" in text or "vllm" in text or "openai" in text


def _gpu_bootstrap_train_py(*, method_name: str, metric_name: str, resource_class: str) -> str:
    default_target_gb = "10.0" if resource_class == "gpu_large" else "6.0"
    safe_metric_name = metric_name.replace("\\", "\\\\").replace('"', '\\"')
    safe_method_name = method_name.replace("\\", "\\\\").replace('"', '\\"')
    return textwrap.dedent(f"""\
    import json
    import os
    import time

    import torch


    METRIC_NAME = "{safe_metric_name}"
    METHOD_NAME = "{safe_method_name}"


    def reserve_vram(device, target_gb):
        blocks = []
        current = float(target_gb)
        while current >= 0.5:
            try:
                numel = int(current * (1024 ** 3) / 2)  # float16 bytes
                block = torch.empty(numel, dtype=torch.float16, device=device)
                block.normal_(0, 0.01)
                blocks.append(block)
                return blocks, current
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                torch.cuda.empty_cache()
                current *= 0.8
        return blocks, 0.0


    def main():
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for this gpu resource_class experiment.")

        device = torch.device("cuda")
        torch.cuda.reset_peak_memory_stats(device)
        torch.manual_seed(7)

        target_gb = float(os.getenv("DEEPGRAPH_GPU_WORKLOAD_TARGET_GB", "{default_target_gb}"))
        hold_seconds = float(os.getenv("DEEPGRAPH_GPU_WORKLOAD_HOLD_SECONDS", "8"))
        reserve_blocks, reserved_gb = reserve_vram(device, target_gb)

        batch = 4096 if reserved_gb >= 6 else 2048
        dim = 2048
        x = torch.randn(batch, dim, device=device, dtype=torch.float16)
        w1 = torch.randn(dim, dim, device=device, dtype=torch.float16) / dim ** 0.5
        w2 = torch.randn(dim, dim, device=device, dtype=torch.float16) / dim ** 0.5
        labels = torch.randn(batch, dim, device=device, dtype=torch.float16)

        optimizer_signal = 0.0
        for step in range(24):
            y = torch.relu(x @ w1)
            y = y @ w2
            loss = torch.mean((y - labels) ** 2)
            optimizer_signal += float(loss.detach().cpu())
            x = x + 0.001 * torch.tanh(y)

        torch.cuda.synchronize(device)
        if hold_seconds > 0:
            time.sleep(hold_seconds)
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        score = max(0.0, min(2.0, (reserved_gb / max(target_gb, 0.1)) + 1.0 / (1.0 + optimizer_signal / 24.0)))

        print(f"method: {{METHOD_NAME}}")
        print(f"device: {{torch.cuda.get_device_name(device)}}")
        print(f"reserved_vram_gb: {{reserved_gb:.2f}}")
        print(f"peak_vram_mb: {{peak_mb:.1f}}")
        print(f"{{METRIC_NAME}}: {{score:.6f}}")
        print("FINAL_RESULTS: " + json.dumps({{
            METRIC_NAME: score,
            "peak_vram_mb": peak_mb,
            "reserved_vram_gb": reserved_gb,
            "target_vram_gb": target_gb,
            "cuda_device": torch.cuda.get_device_name(device),
        }}))

        del reserve_blocks
        torch.cuda.empty_cache()


    if __name__ == "__main__":
        main()
    """)


def _fallback_scaffold(method: dict, plan: dict, codebase: dict) -> dict:
    """Generate a minimal scaffold without LLM if the call fails."""
    plan = _ensure_real_benchmark_plan({"proposed_method": method}, method, dict(plan or {}), None)
    method_name = method.get("name", "ProposedMethod")
    method_def = method.get("definition", "See method description")
    train_file = codebase.get("main_train_file", "train.py")

    metrics = plan.get("metrics", {})
    primary_metric = metrics.get("primary", "accuracy") if isinstance(metrics, dict) else "accuracy"

    program_md = textwrap.dedent(f"""\
    # SciForge Experiment: {method_name}

    ## Setup
    1. Read all files in the code/ directory for context.
    2. Establish a baseline by running the training script as-is.
    3. Record the baseline metric.

    ## Experimentation
    **File to modify**: `code/{train_file}`
    **Goal**: Implement {method_name} and achieve a better {primary_metric} than baseline.

    ### Method to Implement
    {method_def[:1000]}

    ### Constraints
    - Only modify `code/{train_file}`.
    - Each run has a fixed time budget of {EXPERIMENT_TIME_BUDGET} seconds.
    - Evaluate using: `python spec/evaluate.py`

    ## The Experiment Loop
    LOOP FOREVER:
    1. Modify the code with an experimental idea based on the method above.
    2. git commit
    3. Run: `cd code && python {train_file} > ../run.log 2>&1`
    4. Evaluate: `python spec/evaluate.py run.log`
    5. If metric improved, keep. If worse, git reset.
    6. Log results to results.tsv
    7. NEVER STOP until manually interrupted.
    """)

    evaluate_py = textwrap.dedent(f"""\
    import json
    import re
    import sys

    def main():
        log_file = sys.argv[1] if len(sys.argv) > 1 else "run.log"
        try:
            with open(log_file) as f:
                text = f.read()
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except Exception:
                    payload = None
                if isinstance(payload, dict):
                    raw_value = payload.get("metric_value")
                    if raw_value is not None:
                        try:
                            print(f"metric_value: {{float(raw_value)}}")
                            return
                        except Exception:
                            pass
            patterns = [
                r'"metric_value"\\s*:\\s*([\\d.]+)',
                r'metric_value[:\\s]+([\\d.]+)',
                r'{primary_metric}[:\\s]+([\\d.]+)',
            ]
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    print(f"metric_value: {{matches[-1]}}")
                    return
            print("metric_value: 0.0")
        except Exception as e:
            print(f"metric_value: 0.0")

    if __name__ == "__main__":
        main()
    """)

    if EXPERIMENT_REQUIRE_REAL_BENCHMARK and not EXPERIMENT_ALLOW_SYNTHETIC_FALLBACK:
        try:
            train_py = _real_llm_benchmark_train_py(
                method_name=str(method_name),
                metric_name=str(primary_metric),
                plan=plan,
            )
            scaffold_kind = "full_benchmark_compiled"
        except ValueError as exc:
            train_py = _benchmark_recipe_blocker_train_py(
                metric_name=str(primary_metric),
                error=str(exc),
                plan=plan,
            )
            scaffold_kind = "real_benchmark_recipe_blocked"
    else:
        train_py = textwrap.dedent(f"""\
        import json
        import random

        def main():
            random.seed(13)
            baseline = 0.62
            noise = 0.03
            metric = baseline + (random.random() - 0.5) * noise
            result = {{
                "method": {method_name!r},
                "metric_name": {primary_metric!r},
                "metric_value": round(metric, 4),
                "notes": {method_def[:200]!r},
            }}
            print(json.dumps(result))
            print(f"{primary_metric}: {{result['metric_value']}}")

        if __name__ == "__main__":
            main()
        """)
        scaffold_kind = "bootstrap_probe"

    publication_contract = _publication_evidence_contract(
        {"title": method_name, "proposed_method": method},
        plan,
        codebase=codebase,
        scaffold_kind=scaffold_kind,
    )
    success = _normalize_success_criteria(
        {
            "metric_name": primary_metric,
            "metric_direction": publication_contract.get("metric_direction", "higher"),
            "exciting": 0.0,
            "solid": 0.0,
            "disappointing": 0.0,
        },
        plan,
        publication_contract,
    )

    return {
        "program_md": program_md,
        "evaluate_py": evaluate_py,
        "train_py": train_py,
        "success_criteria": success,
    }


def build_proxy_config(plan: dict, codebase: dict | None = None, *, judgement=None) -> dict:
    """Build proxy task configuration for time-budgeted experiments."""
    compute = plan.get("compute_budget", {}) if isinstance(plan, dict) else {}
    codebase = codebase or {}
    real_benchmark = bool(plan.get("real_benchmark_required") or plan.get("benchmark_targets"))
    time_budget_seconds = (
        max(EXPERIMENT_TIME_BUDGET, EXPERIMENT_REAL_BENCHMARK_TIME_BUDGET)
        if real_benchmark and EXPERIMENT_REQUIRE_REAL_BENCHMARK
        else EXPERIMENT_TIME_BUDGET
    )

    proxy = {
        "data_fraction": EXPERIMENT_PROXY_DATA_FRACTION,
        "max_epochs": EXPERIMENT_PROXY_MAX_EPOCHS,
        "time_budget_seconds": time_budget_seconds,
        "early_stop_threshold": EXPERIMENT_EARLY_STOP_THRESHOLD,
        "max_iterations": EXPERIMENT_MAX_ITERATIONS,
        "reproduction_iterations": EXPERIMENT_REPRODUCTION_ITERS,
        "refute_min_iterations": EXPERIMENT_REFUTE_MIN_ITERS,
        "estimated_gpu_hours": compute.get("total_gpu_hours", "unknown"),
        "main_train_file": codebase.get("main_train_file"),
        "baseline_command": codebase.get("main_eval_command"),
        "real_benchmark_required": bool(real_benchmark and EXPERIMENT_REQUIRE_REAL_BENCHMARK),
        "synthetic_fallback_allowed": bool(EXPERIMENT_ALLOW_SYNTHETIC_FALLBACK),
        "budget_policy": {
            "per_iteration_time_budget_seconds": time_budget_seconds,
            "max_hypothesis_iterations": EXPERIMENT_MAX_ITERATIONS,
            "reproduction_iterations": EXPERIMENT_REPRODUCTION_ITERS,
            "refute_min_iterations": EXPERIMENT_REFUTE_MIN_ITERS,
            "estimated_gpu_hours": compute.get("total_gpu_hours", "unknown"),
            "gpu_devices": list(GPU_VISIBLE_DEVICES),
            "gpu_model": GPU_DEFAULT_MODEL,
            "gpu_vram_gb": GPU_DEFAULT_VRAM_GB,
        },
        "benchmark_model": EXPERIMENT_REAL_LLM_MODEL,
        "benchmark_dataset": EXPERIMENT_REAL_BENCHMARK_DATASET,
        "benchmark_dataset_config": EXPERIMENT_REAL_BENCHMARK_DATASET_CONFIG,
        "benchmark_max_examples_per_seed": _optional_nonnegative_int(plan.get("max_eval_examples"), EXPERIMENT_REAL_BENCHMARK_MAX_EXAMPLES),
        "benchmark_seeds": max(1, _optional_nonnegative_int(plan.get("minimum_seeds"), EXPERIMENT_REAL_BENCHMARK_SEEDS)),
        "benchmark_time_budget_seconds": EXPERIMENT_REAL_BENCHMARK_TIME_BUDGET,
    }
    publication_contract = plan.get("publication_evidence_contract") if isinstance(plan, dict) else {}
    if isinstance(publication_contract, dict) and publication_contract.get("benchmark_manifest"):
        proxy["benchmark_manifest"] = publication_contract["benchmark_manifest"]
    if isinstance(publication_contract, dict) and publication_contract.get("benchmark_protocol"):
        proxy["benchmark_protocol"] = publication_contract["benchmark_protocol"]
    if isinstance(publication_contract, dict) and publication_contract.get("claim_route"):
        proxy["claim_route"] = publication_contract["claim_route"]
        proxy["claim_strength"] = publication_contract.get("claim_strength")
    if judgement is not None:
        proxy["formal_experiment"] = judgement.formal_experiment
        proxy["smoke_test_only"] = judgement.smoke_test_only
        proxy["experiment_judgement"] = judgement.to_dict()
    return proxy


def _plan_uses_executable_probe(plan: dict) -> bool:
    targets = plan.get("benchmark_targets") if isinstance(plan.get("benchmark_targets"), list) else []
    return any(
        isinstance(target, dict) and target.get("benchmark_role") == "executable_probe"
        for target in targets
    )


def forge_experiment(insight_id: int, *, resource_grant_id: int | None = None) -> dict:
    """Full forge pipeline: scout codebase -> setup workspace -> generate scaffold.

    Creates an experiment_run row and returns all paths/configs needed
    for the validation loop.
    """
    print(f"[FORGE] Starting experiment forge for insight {insight_id}...", flush=True)

    insight = db.fetchone("SELECT * FROM deep_insights WHERE id=?", (insight_id,))
    if not insight:
        return {"error": f"Deep insight {insight_id} not found"}
    agenda_id = int(insight.get("agenda_id") or 0)
    if agenda_id <= 0:
        return {
            "error": "experiment forge requires an agenda-scoped insight",
            "route": "blocked",
        }
    try:
        resource_grant_id = int(resource_grant_id or 0)
    except (TypeError, ValueError):
        resource_grant_id = 0
    grant = db.fetchone(
        """
        SELECT id, agenda_id, idea_id, stage, status, expires_at
        FROM resource_grants
        WHERE id=? AND agenda_id=? AND idea_id=?
          AND status='active' AND expires_at > CURRENT_TIMESTAMP
        """,
        (resource_grant_id, agenda_id, insight_id),
    )
    if not grant or str(grant.get("stage") or "") not in {"experiment_forge", "pilot"}:
        return {
            "error": "valid experiment_forge/pilot ResourceGrant is required",
            "route": "blocked",
        }
    llm_scope = {
        "agenda_id": agenda_id,
        "idea_id": insight_id,
        "resource_grant_id": resource_grant_id,
        "stage": str(grant["stage"]),
    }

    gate = evosci_strict_gate_insight(dict(insight))
    if gate:
        print(f"[FORGE] Blocked by EvoScientist strict gate: {gate.get('error')}", flush=True)
        return gate

    parsed = _autofill_experiment_contracts(
        dict(insight),
        llm_scope=llm_scope,
    )
    _persist_enriched_insight(insight_id, parsed)
    spec = DeepInsightSpec.from_raw(parsed)
    plan = spec.experimental_plan
    evidence_plan = spec.evidence_plan
    layout = get_idea_workspace(insight_id, insight=parsed, create=True, sync_db=True)

    # Step 1: Scout codebase
    if _plan_uses_executable_probe(plan):
        codebase = _scratch_codebase("executable benchmark probe uses generated runner; deferred formal target remains in benchmark_harness_jobs")
        print("[FORGE] Skipping codebase scout for executable benchmark probe; using scratch runner.", flush=True)
    else:
        print(f"[FORGE] Scouting codebase...", flush=True)
        try:
            codebase = scout_codebase(parsed, llm_scope=llm_scope)
        except Exception as exc:
            return {
                "error": f"resource-granted code scout unavailable: {exc}",
                "route": "manual_review_required",
            }
    print(f"[FORGE] Selected: {codebase.get('name', '?')} ({codebase.get('url', '?')})", flush=True)

    run_id = db.insert_returning_id(
        """
        INSERT INTO experiment_runs
            (agenda_id, resource_grant_id, deep_insight_id, experiment_suite,
             status, phase, workdir, codebase_url, codebase_ref,
             baseline_metric_name, scientific_evidence_state)
        VALUES (?, ?, ?, ?, 'scaffolding', 'setup', ?, ?, ?, ?, 'planned')
        RETURNING id
        """,
        (
            agenda_id,
            resource_grant_id,
            insight_id,
            str(parsed.get("experiment_suite") or "main").strip() or "main",
            "",
            codebase.get("url", "scratch"),
            codebase.get("name", ""),
            "metric",
        ),
    )
    db.commit()

    # Step 2: Setup workspace
    workdir = setup_workspace(insight_id, run_id, codebase, insight=parsed)
    _checkpoint_run_state(
        run_id,
        agenda_id=agenda_id,
        phase="workspace_ready",
        workdir=workdir,
        codebase=codebase,
        baseline_metric_name="metric",
    )
    code_dir = workdir / "code"
    codebase = repair_codebase_entrypoint(code_dir, codebase)
    entrypoint_available = None
    if codebase.get("url") != "scratch" and not _codebase_has_expected_entrypoint(
        code_dir, codebase
    ):
        entrypoint_available = False
        print(
            f"[FORGE] Selected repo missing expected entrypoint "
            f"{codebase.get('main_train_file', 'unknown')}; falling back to scratch.",
            flush=True,
        )
        if code_dir.exists():
            _safe_rmtree(code_dir)
        code_dir.mkdir(parents=True, exist_ok=True)
        codebase = _scratch_codebase(
            reason=(
                f"repo {codebase.get('name', '?')} missing expected entrypoint "
                f"{codebase.get('main_train_file', 'unknown')}"
            )
        )
    elif codebase.get("url") != "scratch":
        entrypoint_available = True
    print(f"[FORGE] Workspace: {workdir}", flush=True)

    judgement = review_experiment_candidate(
        spec,
        codebase=codebase,
        entrypoint_available=entrypoint_available,
    )
    if judgement.recommended_route == "blocked":
        summary = judgement.summary or "Experiment review blocked formalization"
        blocked_payload = {
            "formal_experiment": False,
            "smoke_test_only": True,
            "review_route": judgement.recommended_route,
            "experiment_judgement": judgement.to_dict(),
        }
        db.execute(
            """
            UPDATE experiment_runs
            SET status='failed',
                phase='experiment_review_blocked',
                proxy_config=?,
                error_message=?,
                completed_at=CURRENT_TIMESTAMP
            WHERE id=? AND agenda_id=?
            """,
            (
                json.dumps(blocked_payload, ensure_ascii=False),
                summary,
                run_id,
                agenda_id,
            ),
        )
        db.commit()
        return {
            "error": summary,
            "judgement": judgement.to_dict(),
            "route": judgement.recommended_route,
            "harness_required": bool((judgement.environment_review or {}).get("benchmark_harness_required")),
            "harness_queue": (judgement.environment_review or {}).get("harness_queue") or "",
        }
    print(
        f"[FORGE] Review route={judgement.recommended_route} "
        f"formal={judgement.formal_experiment} smoke={judgement.smoke_test_only}",
        flush=True,
    )

    proxy = build_proxy_config(plan, codebase=codebase, judgement=judgement)
    proxy["evidence_plan"] = evidence_plan
    proxy["publication_evidence_contract"] = _publication_evidence_contract(
        parsed,
        plan,
        codebase=codebase,
        evidence_plan=evidence_plan,
        scaffold_kind="planned",
    )
    proxy["benchmark_manifest"] = proxy["publication_evidence_contract"].get("benchmark_manifest", {})
    proxy["benchmark_protocol"] = proxy["publication_evidence_contract"].get("benchmark_protocol", {})
    proxy["paper_intent"] = proxy["publication_evidence_contract"].get("paper_intent", {})
    proxy["claim_route"] = proxy["publication_evidence_contract"].get("claim_route", {})
    proxy["claim_strength"] = proxy["publication_evidence_contract"].get("claim_strength")
    _checkpoint_run_state(
        run_id,
        agenda_id=agenda_id,
        phase="review_decision_ready",
        workdir=workdir,
        codebase=codebase,
        proxy_config=proxy,
        baseline_metric_name="metric",
    )

    # Step 3: Generate scaffold
    print(f"[FORGE] Generating scaffold (program.md, evaluate.py, success_criteria)...", flush=True)
    try:
        scaffold = generate_scaffold(
            parsed,
            codebase,
            workdir,
            llm_scope=llm_scope,
        )
    except Exception as exc:
        db.execute(
            """
            UPDATE experiment_runs
            SET status='failed',
                phase='scaffold_route_unavailable',
                error_message=?,
                completed_at=CURRENT_TIMESTAMP
            WHERE id=? AND agenda_id=?
            """,
            (
                f"resource-granted scaffold route unavailable: {exc}",
                run_id,
                agenda_id,
            ),
        )
        db.commit()
        return {
            "error": f"resource-granted scaffold route unavailable: {exc}",
            "route": "manual_review_required",
            "run_id": run_id,
        }

    # Step 4: Build proxy config
    success = scaffold.get("success_criteria", {})
    if scaffold.get("publication_evidence_contract"):
        proxy["publication_evidence_contract"] = scaffold["publication_evidence_contract"]
        proxy["benchmark_manifest"] = scaffold["publication_evidence_contract"].get("benchmark_manifest", {})
        proxy["benchmark_protocol"] = scaffold["publication_evidence_contract"].get("benchmark_protocol", {})
        proxy["paper_intent"] = scaffold["publication_evidence_contract"].get("paper_intent", {})
        proxy["claim_route"] = scaffold["publication_evidence_contract"].get("claim_route", {})
        proxy["claim_strength"] = scaffold["publication_evidence_contract"].get("claim_strength")
    if scaffold.get("benchmark_manifest"):
        proxy["benchmark_manifest"] = scaffold["benchmark_manifest"]
    if scaffold.get("benchmark_protocol"):
        proxy["benchmark_protocol"] = scaffold["benchmark_protocol"]
    if scaffold.get("baseline_command_override"):
        proxy["baseline_command"] = scaffold["baseline_command_override"]
        proxy["main_train_file"] = "train.py"
        codebase["main_eval_command"] = scaffold["baseline_command_override"]
        codebase["main_train_file"] = "train.py"
    plan_paths = write_plan_files(
        insight_id,
        run_id=run_id,
        insight=parsed,
        files={
            "program.md": scaffold.get("program_md", ""),
            "evaluate.py": scaffold.get("evaluate_py", ""),
            "success_criteria.json": success,
            "proxy_config.json": proxy,
            "benchmark_manifest.json": proxy.get("benchmark_manifest") or scaffold.get("benchmark_manifest") or {},
            "benchmark_protocol.json": proxy.get("benchmark_protocol") or scaffold.get("benchmark_protocol") or {},
            "evidence_plan.json": evidence_plan,
            "experiment_judgement.json": judgement.to_dict(),
        },
    )

    experiment_spec = ExperimentSpec.from_sources(
        run_id=run_id,
        insight=spec,
        workdir=str(workdir),
        codebase=codebase,
        judgement=judgement,
        success_criteria=success,
        proxy_config=proxy,
        artifact_paths={
            "program_md": plan_paths["program.md"],
            "evaluate_py": plan_paths["evaluate.py"],
            "success_criteria": plan_paths["success_criteria.json"],
            "proxy_config": plan_paths["proxy_config.json"],
            "benchmark_manifest": plan_paths["benchmark_manifest.json"],
            "benchmark_protocol": plan_paths["benchmark_protocol.json"],
            "evidence_plan": plan_paths["evidence_plan.json"],
            "experiment_judgement": plan_paths["experiment_judgement.json"],
        },
    )
    plan_paths.update(
        write_plan_files(
            insight_id,
            run_id=run_id,
            insight=parsed,
            files={"experiment_spec.json": experiment_spec.to_dict()},
        )
    )
    _checkpoint_run_state(
        run_id,
        agenda_id=agenda_id,
        phase="scaffold_ready",
        workdir=workdir,
        codebase=codebase,
        program_md=scaffold.get("program_md", ""),
        proxy_config=proxy,
        success_criteria=success,
        baseline_metric_name=success.get("metric_name", "metric"),
    )
    db.execute(
        """
        INSERT INTO experiment_artifacts
            (agenda_id, run_id, artifact_type, path, metadata)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            agenda_id,
            run_id,
            "source_data",
            plan_paths["experiment_spec.json"],
            json.dumps({"contract_type": "ExperimentSpec"}),
        ),
    )
    db.execute(
        """
        INSERT INTO experiment_artifacts
            (agenda_id, run_id, artifact_type, path, metadata)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            agenda_id,
            run_id,
            "source_data",
            plan_paths["experiment_judgement.json"],
            json.dumps({"contract_type": "ExperimentJudgement"}),
        ),
    )
    db.commit()

    # Update deep_insight status
    new_insight_status = "forged" if judgement.formal_experiment else "smoke_only"
    db.execute(
        "UPDATE deep_insights SET status=?, evoscientist_workdir=?, updated_at=CURRENT_TIMESTAMP WHERE id=? AND agenda_id=?",
        (
            new_insight_status,
            str(layout["workspace_root"]),
            insight_id,
            agenda_id,
        ),
    )
    db.commit()
    promote_canonical_run(insight_id, run_id, insight=parsed)
    write_latest_status(
        insight_id,
        {
            "stage": "experiment_forged",
            "status": new_insight_status,
            "workdir": str(workdir),
            "canonical_run_id": run_id,
            "formal_experiment": judgement.formal_experiment,
            "smoke_test_only": judgement.smoke_test_only,
            "evidence_tier": (proxy.get("publication_evidence_contract") or {}).get("evidence_tier"),
            "proxy_config_path": plan_paths["proxy_config.json"],
            "experiment_spec_path": plan_paths["experiment_spec.json"],
        },
        run_id=run_id,
        insight=parsed,
    )

    if judgement.formal_experiment:
        apply_experiment_queued_deep(insight_id, note=f"experiment_run_id={run_id}")

    print(f"[FORGE] Experiment forged: run_id={run_id}, workdir={workdir}", flush=True)

    return {
        "run_id": run_id,
        "insight_id": insight_id,
        "workdir": str(workdir),
        "codebase": codebase,
        "success_criteria": success,
        "proxy_config": proxy,
        "evidence_plan": evidence_plan,
        "judgement": judgement.to_dict(),
        "formal_experiment": judgement.formal_experiment,
        "smoke_test_only": judgement.smoke_test_only,
        "scaffold_tokens": scaffold.get("tokens", 0),
        "scout_llm_route": codebase.get("llm_route"),
        "scaffold_llm_route": scaffold.get("llm_route"),
    }
