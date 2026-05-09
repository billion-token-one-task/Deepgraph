"""Experiment Forge: bridge from deep_insights to runnable experiments.

Three sub-components:
  2a. Code Scout — find/clone relevant codebases for a hypothesis
  2b. Scaffold Builder — generate program.md, evaluate.py, success_criteria.json
  2c. Proxy Task Builder — configure time-budgeted experiment for fast iteration

This is the hardest layer: translating a structured method description
into something an autonomous coding agent can actually run.
"""
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
from agents.evosci_requirements import evosci_strict_gate_insight
from agents.experiment_review import review_experiment_candidate
from agents.idea_route import classify_idea_route
from agents.llm_client import call_llm, call_llm_json
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


_STANDARD_REASONING_ABLATIONS = [
    "no_counterfactual_delta",
    "no_lcb",
    "no_self_divergence_penalty",
    "no_qstruct_term",
    "oracle_router",
]


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


def _generated_runner_support_reason(target: dict) -> tuple[bool, str]:
    """Return whether the built-in runner can execute this benchmark target."""
    name = _non_empty_text(target.get("name") or target.get("hf_dataset") or "benchmark")
    if target.get("derive_from_loaded_benchmarks"):
        return True, ""

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
    target["name"] = _non_empty_text(target.get("name") or name or target.get("hf_dataset")) or "GSM8K"
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
    target["max_eval_examples"] = int(
        target.get("max_eval_examples") or EXPERIMENT_REAL_BENCHMARK_MAX_EXAMPLES
    )
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
                "max_eval_examples": EXPERIMENT_REAL_BENCHMARK_MAX_EXAMPLES,
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
                "max_eval_examples": EXPERIMENT_REAL_BENCHMARK_MAX_EXAMPLES,
            }
        ]
    return [
        _normalize_benchmark_target("GSM8K", parsed=parsed)
    ]


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
    return [
        {
            "name": EXPERIMENT_REAL_LLM_MODEL,
            "hf_model": EXPERIMENT_REAL_LLM_MODEL,
            "backend": "transformers",
            "role": "candidate_base_model",
            "load_in_4bit": True,
            "requires_cuda": True,
        }
    ]


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
    return _unique_non_empty(_named_values(rows, keys=("name", "hf_model", "model")))


def _ensure_real_benchmark_plan(parsed: dict, method: dict, plan: dict, resource_class: str | None = None) -> dict:
    plan = dict(plan or {})
    if not EXPERIMENT_REQUIRE_REAL_BENCHMARK:
        return plan
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
    plan["benchmark_targets"] = real_targets
    plan["datasets"] = [{"name": row.get("name") or row.get("hf_dataset") or row.get("dataset")} for row in real_targets]
    plan["baselines"] = [{"name": name} for name in _planned_baselines(plan)]

    model_targets = plan.get("model_targets") if isinstance(plan.get("model_targets"), list) else []
    normalized_models = []
    for row in model_targets:
        if isinstance(row, dict):
            name = _non_empty_text(row.get("name") or row.get("hf_model") or row.get("model"))
            if name and name.lower() not in {"toy", "dummy", "mock", "synthetic"}:
                normalized_models.append(dict(row))
    if not normalized_models:
        normalized_models = _default_real_model_targets(parsed, resource_class)
    plan["model_targets"] = normalized_models
    plan["real_benchmark_required"] = True
    plan["proxy_allowed"] = bool(EXPERIMENT_ALLOW_SYNTHETIC_FALLBACK)
    plan["minimum_seeds"] = max(int(plan.get("minimum_seeds") or 0), EXPERIMENT_REAL_BENCHMARK_SEEDS)
    plan["max_eval_examples"] = int(plan.get("max_eval_examples") or EXPERIMENT_REAL_BENCHMARK_MAX_EXAMPLES)
    plan["requires_real_model"] = True
    plan["requires_real_dataset"] = True
    plan["generated_runner_supported"] = not recipe_blockers
    if recipe_blockers:
        plan["benchmark_recipe_blockers"] = recipe_blockers
        plan["deferred_benchmark_targets"] = [
            item["name"] for item in recipe_blockers if item.get("name")
        ]
    else:
        plan.pop("benchmark_recipe_blockers", None)
        plan.pop("deferred_benchmark_targets", None)
    plan["benchmark_execution"] = {
        "mode": "real_benchmark",
        "synthetic_fallback_allowed": bool(EXPERIMENT_ALLOW_SYNTHETIC_FALLBACK),
        "generated_runner_supported": plan["generated_runner_supported"],
        "default_model": normalized_models[0].get("hf_model") or normalized_models[0].get("name"),
            "default_dataset": real_targets[0].get("hf_dataset") or real_targets[0].get("name"),
            "target_count": len(real_targets),
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
    seed_raw = plan.get("minimum_seeds") or plan.get("seeds")
    try:
        if isinstance(seed_raw, list):
            minimum_seeds = max(len(seed_raw), EXPERIMENT_REAL_BENCHMARK_SEEDS or 3)
        elif seed_raw not in (None, "", "unknown"):
            minimum_seeds = max(int(seed_raw), EXPERIMENT_REAL_BENCHMARK_SEEDS or 3)
        else:
            minimum_seeds = EXPERIMENT_REAL_BENCHMARK_SEEDS or 3
    except (TypeError, ValueError):
        minimum_seeds = EXPERIMENT_REAL_BENCHMARK_SEEDS or 3
    seed_list = list(range(max(1, minimum_seeds)))
    full_artifacts = [
        "run_config.json",
        "raw_predictions.jsonl",
        "routing_decisions.jsonl",
        "per_seed_results.json",
        "per_dataset_results.json",
        "main_results_table.json",
        "cost_utility_tradeoff_table.json",
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
            "experiment_contract_architect": "freeze datasets/baselines/metrics/artifact gates",
            "sanity_runner_builder": "small real-data runner for environment and signal checks",
            "full_benchmark_compiler": "expand the locked contract into a job matrix",
            "method_worker": "change method implementation without changing benchmark contract",
            "evidence_auditor": "audit artifacts before paper claims",
            "manuscript_writer": "write only audited claims",
        },
        "locked_fields": [
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
            "max_examples_per_seed": int(plan.get("max_eval_examples") or EXPERIMENT_REAL_BENCHMARK_MAX_EXAMPLES),
            "datasets": real_benchmarks[:1],
            "models": models[:1],
            "methods": ["baseline", "candidate"],
            "seeds": seed_list[: min(len(seed_list), EXPERIMENT_REAL_BENCHMARK_SEEDS)],
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
            "statistical_tests": [
                "paired_bootstrap_ci",
                "paired_permutation_test",
            ],
            "required_artifacts": full_artifacts,
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
    minimum_seeds = 3
    seed_raw = plan.get("minimum_seeds") or plan.get("seeds")
    try:
        if isinstance(seed_raw, list):
            minimum_seeds = max(minimum_seeds, len(seed_raw))
        elif seed_raw not in (None, "", "unknown"):
            minimum_seeds = max(minimum_seeds, int(seed_raw))
    except (TypeError, ValueError):
        minimum_seeds = 3

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
    required_artifacts = ["run_config", "raw_metrics_jsonl", "seed_variance_table"]
    if main_table_enabled:
        required_artifacts.insert(0, "main_results_table")
    required_artifacts.append("ablation_table")
    required_artifacts.extend(
        [
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
        "statistical_test": "paired bootstrap confidence interval plus paired permutation test across seeds/tasks",
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


def _enrich_experimental_plan(parsed: dict, method: dict) -> dict:
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

    dataset_names = _unique_non_empty(_named_values(plan.get("datasets"), keys=("name", "dataset")))
    if not dataset_names or (EXPERIMENT_REQUIRE_REAL_BENCHMARK and all(_looks_like_synthetic_dataset(name) for name in dataset_names)):
        real_targets = _default_real_benchmark_targets({**parsed, "proposed_method": method})
        dataset_names = [str(row.get("name") or row.get("hf_dataset")) for row in real_targets if row.get("name") or row.get("hf_dataset")]
        plan["benchmark_targets"] = real_targets
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


def _autofill_experiment_contracts(insight: dict) -> dict:
    parsed = _parse_insight_fields(insight)
    method = _enrich_proposed_method(parsed, dict(parsed.get("experimental_plan") or {}))
    plan = _enrich_experimental_plan(parsed, method)
    parsed["proposed_method"] = method
    parsed["experimental_plan"] = plan
    inferred = _non_empty_text(parsed.get("resource_class")) or infer_resource_class(parsed)
    if EXPERIMENT_REQUIRE_REAL_BENCHMARK and _model_target_names(plan):
        model_text = " ".join(_model_target_names(plan)).lower()
        if any(token in model_text for token in ("qwen", "llama", "mistral", "mixtral", "gemma", "phi")):
            inferred = "gpu_large"
            compute = dict(plan.get("compute_budget") or {}) if isinstance(plan.get("compute_budget"), dict) else {}
            if not compute.get("total_gpu_hours"):
                compute["total_gpu_hours"] = 24.0
                plan["compute_budget"] = compute
    parsed["resource_class"] = inferred
    return parsed


def _persist_enriched_insight(insight_id: int, parsed: dict) -> None:
    db.execute(
        """
        UPDATE deep_insights
        SET proposed_method=?, experimental_plan=?, resource_class=?, updated_at=CURRENT_TIMESTAMP
        WHERE id=?
        """,
        (
            json.dumps(parsed.get("proposed_method") or {}, ensure_ascii=False),
            json.dumps(parsed.get("experimental_plan") or {}, ensure_ascii=False),
            parsed.get("resource_class") or "cpu",
            insight_id,
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
    params = list(fields.values()) + [run_id]
    db.execute(f"UPDATE experiment_runs SET {assignments} WHERE id=?", tuple(params))
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


def scout_codebase(insight: dict) -> dict:
    """Find the best codebase for implementing a hypothesis.

    Uses LLM to suggest repos based on the method description and
    knowledge graph context about what methods/datasets are involved.
    """
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

    try:
        result, _ = call_llm_json(CODE_SCOUT_SYSTEM, prompt)
        codebase = result.get("codebase", {"url": "scratch", "name": "minimal", "reason": "no suitable repo found"})
        return _normalize_codebase_metadata(codebase)
    except Exception as e:
        print(f"[FORGE] Code scout failed: {e}", flush=True)
        return {"url": "scratch", "name": "minimal", "reason": f"scout error: {e}"}


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


def generate_scaffold(insight: dict, codebase: dict, workdir: Path) -> dict:
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
    real_runner_required = bool(EXPERIMENT_REQUIRE_REAL_BENCHMARK and not EXPERIMENT_ALLOW_SYNTHETIC_FALLBACK)
    recipe_blocked = False
    try:
        result, tokens = call_llm_json(SCAFFOLD_SYSTEM, prompt)
    except Exception as e:
        print(f"[FORGE] Scaffold generation failed: {e}", flush=True)
        result = _fallback_scaffold(method, plan, codebase)
        used_fallback = True
        tokens = 0

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
    success = _normalize_success_criteria(success, plan, publication_contract)

    spec_dir = workdir / "spec"
    spec_dir.mkdir(parents=True, exist_ok=True)
    (spec_dir / "program.md").write_text(program_md, encoding="utf-8")
    (spec_dir / "evaluate.py").write_text(evaluate_py, encoding="utf-8")
    (spec_dir / "success_criteria.json").write_text(
        json.dumps(success, indent=2), encoding="utf-8")
    (spec_dir / "benchmark_manifest.json").write_text(
        json.dumps(benchmark_manifest, indent=2), encoding="utf-8")

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
        "claim_route": publication_contract.get("claim_route", {}),
        "train_py_written": bool(train_py and len(train_py) > 50),
        "baseline_command_override": baseline_command_override,
        "tokens": tokens,
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
        normalized_targets = [_normalize_benchmark_target("GSM8K")]
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
    model = next((row for row in models if isinstance(row, dict)), {})
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
        "max_examples": int(plan.get("max_eval_examples") or target.get("max_eval_examples") or EXPERIMENT_REAL_BENCHMARK_MAX_EXAMPLES),
        "seeds": int(plan.get("minimum_seeds") or EXPERIMENT_REAL_BENCHMARK_SEEDS),
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


def _real_llm_benchmark_train_py(*, method_name: str, metric_name: str, plan: dict) -> str:
    defaults = _real_benchmark_defaults(plan)
    defaults_payload = {
        "method_name": method_name,
        "metric_name": metric_name,
        "model_id": defaults["model_id"],
        "targets": defaults["targets"],
        "max_examples": defaults["max_examples"],
        "seeds": defaults["seeds"],
        "baselines": defaults["baselines"],
        "ablations": defaults["ablations"],
    }
    defaults_json = json.dumps(defaults_payload, ensure_ascii=False).replace("'''", "\\u0027\\u0027\\u0027")
    return textwrap.dedent("""\
    import collections
    import hashlib
    import importlib.metadata
    import io
    import itertools
    import json
    import math
    import os
    import platform
    import random
    import re
    import statistics
    import sys
    import time
    import traceback
    import urllib.request

    os.environ.setdefault("HF_ENDPOINT", os.getenv("DEEPGRAPH_HF_ENDPOINT", "https://hf-mirror.com"))
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "180")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    DEFAULTS = json.loads(r'''__DEEPGRAPH_DEFAULTS_JSON__''')
    METHOD_NAME = DEFAULTS["method_name"]
    METRIC_NAME = DEFAULTS["metric_name"]
    DEFAULT_MODEL_ID = DEFAULTS["model_id"]
    DEFAULT_MAX_EXAMPLES = int(DEFAULTS["max_examples"])
    DEFAULT_SEEDS = int(DEFAULTS["seeds"])
    DEFAULT_LOCAL_JSONL = os.path.join(os.path.dirname(__file__), "benchmark_data", "gsm8k_test.jsonl")
    DEFAULT_JSONL_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
    DEFAULT_JSONL_SHA256 = "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14"
    DEFAULT_REPAIR_MAX_EXAMPLES_CAP = 2
    DEFAULT_REPAIR_SEEDS_CAP = 1
    DEFAULT_REPAIR_METHODS = (
        "Vanilla Direct Answering",
        "Always-Reason Chain-of-Thought",
        "CGGR",
        "CGGR/no_counterfactual_delta",
    )

    METHOD_SPECS = collections.OrderedDict([
        ("Vanilla Direct Answering", {"kind": "direct", "max_new_tokens": 48}),
        ("Always-Reason Chain-of-Thought", {"kind": "fixed_cot", "max_new_tokens": 192}),
        ("Self-Consistency Reasoning", {"kind": "self_consistency", "max_new_tokens": 192}),
        ("Least-to-Most Prompting", {"kind": "least_to_most", "max_new_tokens": 192}),
        ("Confidence Gate", {"kind": "confidence_gate", "max_new_tokens": 192}),
        ("Disagreement Routing", {"kind": "disagreement_gate", "max_new_tokens": 192}),
        ("Random Budget-Matched Routing", {"kind": "random_budget_matched", "max_new_tokens": 192}),
        ("CGGR", {"kind": "cggr", "max_new_tokens": 192}),
    ])
    ABLATION_SPECS = collections.OrderedDict([
        ("no_counterfactual_delta", {"kind": "cggr_ablate_counterfactual", "max_new_tokens": 192}),
        ("no_lcb", {"kind": "cggr_ablate_lcb", "max_new_tokens": 192}),
        ("no_self_divergence_penalty", {"kind": "cggr_ablate_divergence", "max_new_tokens": 192}),
        ("no_qstruct_term", {"kind": "cggr_ablate_qstruct", "max_new_tokens": 192}),
    ])
    BASELINE_ALIASES = {
        "Vanilla Direct Answering": ["direct", "vanilla", "direct_answering"],
        "Always-Reason Chain-of-Thought": ["fixed_cot", "cot", "chain_of_thought"],
        "Self-Consistency Reasoning": ["self_consistency", "sc"],
        "Least-to-Most Prompting": ["least_to_most", "ltm"],
        "Confidence Gate": ["confidence_gate", "adaptive_gate"],
        "Disagreement Routing": ["disagreement_gate", "disagreement", "self_consistency_gate"],
        "Random Budget-Matched Routing": ["random_budget_matched", "random_routing", "budget_matched_random"],
        "CGGR/oracle_router": ["oracle", "oracle_router", "upper_bound"],
        "CGGR": ["cggr", "candidate", "proposed_method"],
    }


    def _results_dir():
        path = os.path.abspath(os.path.join(os.getcwd(), "..", "results"))
        os.makedirs(path, exist_ok=True)
        return path


    def _write_json(name, payload):
        path = os.path.join(_results_dir(), name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        return path


    def _append_jsonl(name, payload):
        path = os.path.join(_results_dir(), name)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\\n")
        return path


    def _touch_result_file(name):
        path = os.path.join(_results_dir(), name)
        with open(path, "a", encoding="utf-8"):
            pass
        return path


    def _package_version(name):
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None


    def _write_environment_report(model_id, method_specs, seed_values, max_examples):
        cuda_available = bool(torch.cuda.is_available())
        report = {
            "schema_version": "benchmark_environment_report_v1",
            "python": sys.version,
            "platform": platform.platform(),
            "packages": {
                "torch": getattr(torch, "__version__", None),
                "transformers": _package_version("transformers"),
                "datasets": _package_version("datasets"),
                "accelerate": _package_version("accelerate"),
                "bitsandbytes": _package_version("bitsandbytes"),
                "modelscope": _package_version("modelscope"),
            },
            "cuda": {
                "available": cuda_available,
                "torch_cuda": getattr(torch.version, "cuda", None),
                "device_count": torch.cuda.device_count() if cuda_available else 0,
                "current_device": torch.cuda.current_device() if cuda_available else None,
                "device_name": torch.cuda.get_device_name(0) if cuda_available else None,
            },
            "model_id": model_id,
            "methods": list(method_specs.keys()),
            "seed_values": list(seed_values),
            "max_examples_per_dataset_seed": max_examples,
            "env": {
                key: os.getenv(key)
                for key in sorted(os.environ)
                if key.startswith("DEEPGRAPH_BENCHMARK_")
                or key
                in {
                    "CUDA_VISIBLE_DEVICES",
                    "HF_ENDPOINT",
                    "HF_HUB_ETAG_TIMEOUT",
                    "HF_HUB_DOWNLOAD_TIMEOUT",
                    "HF_HUB_DISABLE_XET",
                }
            },
        }
        return _write_json("environment_report.json", report)


    def _read_jsonl_rows(path):
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows


    def _rewrite_hf_url(url):
        endpoint = (os.getenv("DEEPGRAPH_HF_ENDPOINT") or os.getenv("HF_ENDPOINT") or "").strip().rstrip("/")
        if endpoint and url.startswith("https://huggingface.co/"):
            return endpoint + url[len("https://huggingface.co"):]
        return url


    def _download_bytes(url, *, timeout=90):
        url = _rewrite_hf_url(url)
        retries = max(1, int(os.getenv("DEEPGRAPH_DIRECT_DATASET_RETRIES", "2")))
        last_exc = None
        for attempt in range(retries):
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "DeepGraphBenchmarkRunner/1.0"},
            )
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    return response.read()
            except Exception as exc:
                last_exc = exc
                if attempt + 1 < retries:
                    time.sleep(min(8, 1.5 ** attempt))
        raise RuntimeError(f"direct download failed after {retries} attempts for {url}: {last_exc}")


    def _download_jsonl_rows(url, expected_sha):
        payload = _download_bytes(url, timeout=180)
        digest = hashlib.sha256(payload).hexdigest()
        if expected_sha and digest != expected_sha:
            raise RuntimeError("Downloaded benchmark checksum mismatch: " + digest)
        return [json.loads(line) for line in payload.decode("utf-8").splitlines() if line.strip()]


    def _rows_from_json_payload(payload):
        if isinstance(payload, list):
            return [row for row in payload if isinstance(row, dict)]
        if not isinstance(payload, dict):
            return []
        preferred_keys = (
            "data",
            "examples",
            "questions",
            "train",
            "validation",
            "dev",
            "test",
            "rows",
        )
        for key in preferred_keys:
            value = payload.get(key)
            rows = _rows_from_json_payload(value)
            if rows:
                return rows
        rows = []
        for value in payload.values():
            rows.extend(_rows_from_json_payload(value))
            if len(rows) >= 100000:
                break
        return rows


    def _download_direct_rows(target, errors):
        for spec in target.get("direct_files") or []:
            if not isinstance(spec, dict):
                continue
            url = str(spec.get("url") or "").strip()
            fmt = str(spec.get("format") or "").strip().lower()
            if not url:
                continue
            try:
                print(
                    "BENCHMARK_STAGE: direct_download "
                    + str(target.get("name"))
                    + " "
                    + str(spec.get("id") or url),
                    flush=True,
                )
                payload = _download_bytes(url, timeout=int(os.getenv("DEEPGRAPH_DIRECT_DATASET_TIMEOUT", "90")))
                if fmt == "jsonl" or url.endswith(".jsonl"):
                    rows = [json.loads(line) for line in payload.decode("utf-8").splitlines() if line.strip()]
                elif fmt == "parquet" or url.endswith(".parquet"):
                    import pandas as pd
                    rows = pd.read_parquet(io.BytesIO(payload)).to_dict("records")
                else:
                    rows = _rows_from_json_payload(json.loads(payload.decode("utf-8")))
                if rows:
                    return rows, {
                        "name": target.get("name"),
                        "id": spec.get("id") or url,
                        "config": spec.get("config") or "direct_file",
                        "split": spec.get("split") or target.get("split") or "validation",
                        "direct_file": True,
                    }
                errors.append(f"{spec.get('id') or url}: downloaded but no row objects were found")
            except Exception as exc:
                print(
                    "BENCHMARK_STAGE: direct_download_failed "
                    + str(target.get("name"))
                    + " "
                    + str(spec.get("id") or url)
                    + " "
                    + str(exc)[:300],
                    flush=True,
                )
                errors.append(f"{spec.get('id') or url}: {exc}")
        return None, None


    def _unique(values):
        out = []
        seen = set()
        for value in values:
            text = str(value or "").strip()
            key = text.lower()
            if text and key not in seen:
                seen.add(key)
                out.append(text)
        return out


    def _env_flag(name, default=False):
        value = os.getenv(name)
        if value is None:
            return bool(default)
        return value.strip().lower() in {"1", "true", "yes", "on"}


    def _env_int(name, default):
        value = os.getenv(name)
        if value is None or str(value).strip() == "":
            return int(default)
        try:
            return int(value)
        except ValueError as exc:
            raise ValueError(f"{name} must be an integer, got {value!r}") from exc


    def _apply_runtime_budget(requested_max_examples, requested_seeds):
        if _env_flag("DEEPGRAPH_BENCHMARK_FULL_RUN"):
            return requested_max_examples, requested_seeds
        max_examples_cap = _env_int("DEEPGRAPH_BENCHMARK_MAX_EXAMPLES_CAP", DEFAULT_REPAIR_MAX_EXAMPLES_CAP)
        seeds_cap = _env_int("DEEPGRAPH_BENCHMARK_SEEDS_CAP", DEFAULT_REPAIR_SEEDS_CAP)
        max_examples = requested_max_examples
        seeds = requested_seeds
        if max_examples_cap > 0 and (max_examples <= 0 or max_examples > max_examples_cap):
            max_examples = max_examples_cap
        if seeds_cap > 0 and seeds > seeds_cap:
            seeds = seeds_cap
        if max_examples != requested_max_examples or seeds != requested_seeds:
            print(
                "BENCHMARK_STAGE: runtime_budget_capped "
                + json.dumps(
                    {
                        "requested_max_examples": requested_max_examples,
                        "effective_max_examples": max_examples,
                        "requested_seeds": requested_seeds,
                        "effective_seeds": seeds,
                        "disable_with": "DEEPGRAPH_BENCHMARK_FULL_RUN=1",
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        return max_examples, seeds


    def _selected_seed_values(seeds):
        offset = _env_int("DEEPGRAPH_BENCHMARK_SEED_OFFSET", 0)
        count_raw = os.getenv("DEEPGRAPH_BENCHMARK_SEED_COUNT", "").strip()
        count = _env_int("DEEPGRAPH_BENCHMARK_SEED_COUNT", seeds) if count_raw else seeds
        if offset < 0:
            raise ValueError("DEEPGRAPH_BENCHMARK_SEED_OFFSET must be non-negative")
        if count <= 0:
            raise ValueError("DEEPGRAPH_BENCHMARK_SEED_COUNT must be positive")
        values = list(range(seeds))[offset : offset + count]
        if not values:
            raise ValueError("seed shard is empty; check DEEPGRAPH_BENCHMARK_SEED_OFFSET/COUNT")
        if values != list(range(seeds)):
            print(
                "BENCHMARK_STAGE: seed_shard "
                + json.dumps({"seed_values": values, "total_declared_seeds": seeds}, ensure_ascii=False),
                flush=True,
            )
        return values


    def _method_specs_for_run():
        method_specs = collections.OrderedDict(METHOD_SPECS)
        for name, spec in ABLATION_SPECS.items():
            if name in DEFAULTS.get("ablations", []):
                method_specs["CGGR/" + name] = spec
        requested = os.getenv("DEEPGRAPH_BENCHMARK_METHODS", "").strip()
        requested_all = requested.lower() in {"all", "*"}
        if _env_flag("DEEPGRAPH_BENCHMARK_FULL_RUN") and (not requested or requested_all):
            print(
                "BENCHMARK_STAGE: methods_selected "
                + json.dumps({"methods": list(method_specs.keys()), "mode": "full"}, ensure_ascii=False),
                flush=True,
            )
            return method_specs
        if requested_all:
            print(
                "BENCHMARK_STAGE: methods_selected "
                + json.dumps({"methods": list(method_specs.keys()), "mode": "full"}, ensure_ascii=False),
                flush=True,
            )
            return method_specs
        explicit_subset = bool(requested)
        names = [item.strip() for item in requested.split(",") if item.strip()] if requested else list(DEFAULT_REPAIR_METHODS)
        selected = collections.OrderedDict()
        missing = []
        for name in names:
            key = name if name in method_specs else "CGGR/" + name if "CGGR/" + name in method_specs else None
            if key:
                selected[key] = method_specs[key]
            else:
                missing.append(name)
        if not explicit_subset and "CGGR" not in selected and "CGGR" in method_specs:
            selected["CGGR"] = method_specs["CGGR"]
        if not explicit_subset and not any(name in selected for name in ("Vanilla Direct Answering", "Always-Reason Chain-of-Thought")):
            selected = collections.OrderedDict(
                [("Vanilla Direct Answering", method_specs["Vanilla Direct Answering"]), *selected.items()]
            )
        print(
            "BENCHMARK_STAGE: methods_selected "
            + json.dumps(
                {
                    "methods": list(selected.keys()),
                    "mode": "method_shard" if explicit_subset else "bounded_core",
                    "missing_requested": missing,
                    "override_with": "DEEPGRAPH_BENCHMARK_METHODS=all or DEEPGRAPH_BENCHMARK_FULL_RUN=1",
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return selected


    def _load_hf_rows(target):
        if target.get("derive_from_loaded_benchmarks"):
            return [], {
                "name": target.get("name"),
                "id": "derived_from_loaded_benchmarks",
                "config": "",
                "split": "derived",
                "derived": True,
            }
        local_jsonl = os.getenv("DEEPGRAPH_BENCHMARK_LOCAL_JSONL", "")
        if target.get("name") == "GSM8K" and not local_jsonl:
            local_jsonl = DEFAULT_LOCAL_JSONL
        if local_jsonl and os.path.exists(local_jsonl) and target.get("name") == "GSM8K":
            return _read_jsonl_rows(local_jsonl), {
                "name": target.get("name"),
                "id": "openai/gsm8k:local_jsonl",
                "config": target.get("config") or "main",
                "split": "test",
            }

        errors = []
        rows, meta = _download_direct_rows(target, errors)
        if rows:
            return rows, meta
        candidates = _unique(target.get("hf_candidates") or [target.get("hf_dataset")])
        configs = target.get("config_candidates") or [target.get("config") or ""]
        splits = target.get("split_candidates") or [target.get("split") or "test"]
        for dataset_id in candidates:
            for config in configs:
                for split in splits:
                    if not dataset_id or split == "derived":
                        continue
                    try:
                        print(
                            "BENCHMARK_STAGE: load_dataset "
                            + str(target.get("name"))
                            + " "
                            + dataset_id
                            + "/"
                            + (config or "-")
                            + ":"
                            + split,
                            flush=True,
                        )
                        if config:
                            data = load_dataset(dataset_id, config, split=split)
                        else:
                            data = load_dataset(dataset_id, split=split)
                        return list(data), {
                            "name": target.get("name") or dataset_id,
                            "id": dataset_id,
                            "config": config,
                            "split": split,
                        }
                    except Exception as exc:
                        errors.append(f"{dataset_id}/{config or '-'}:{split}: {exc}")
        rows, meta = _download_direct_rows(target, errors)
        if rows:
            return rows, meta
        if target.get("name") == "GSM8K":
            url = os.getenv("DEEPGRAPH_BENCHMARK_JSONL_URL", DEFAULT_JSONL_URL)
            checksum = os.getenv("DEEPGRAPH_BENCHMARK_JSONL_SHA256", DEFAULT_JSONL_SHA256)
            return _download_jsonl_rows(url, checksum), {
                "name": "GSM8K",
                "id": "openai/gsm8k:jsonl_url",
                "config": "main",
                "split": "test",
            }
        raise RuntimeError("Could not load benchmark target " + str(target.get("name")) + ": " + " | ".join(errors[-5:]))


    def _field_value(row, candidates):
        for key in candidates:
            if key and isinstance(row, dict) and key in row and row[key] not in (None, ""):
                return row[key]
        return None


    def _answer_to_text(value):
        if isinstance(value, bool):
            return "yes" if value else "no"
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, list):
            if not value:
                return ""
            return _answer_to_text(value[0])
        if isinstance(value, dict):
            for key in ("text", "answer", "value", "label", "aliases"):
                if key in value:
                    return _answer_to_text(value[key])
            return json.dumps(value, ensure_ascii=False)
        text = str(value or "").strip()
        if "####" in text:
            text = text.split("####")[-1].strip()
        return text


    def _question_to_text(value):
        if isinstance(value, list):
            return " ".join(str(item) for item in value)
        if isinstance(value, dict):
            for key in ("question", "text", "query", "prompt", "input"):
                if key in value:
                    return _question_to_text(value[key])
            return json.dumps(value, ensure_ascii=False)
        return str(value or "").strip()


    def _difficulty_proxy(question, task_type="qa"):
        text = str(question)
        numbers = len(re.findall(r"\\d+", text))
        operators = sum(text.count(ch) for ch in "+-*/=%")
        clauses = len(re.findall(r"\\b(if|unless|because|before|after|except|not)\\b", text.lower()))
        score = (len(text.split()) / 90.0) + 0.07 * numbers + 0.05 * operators + 0.04 * clauses
        task = str(task_type or "").lower()
        if "multihop" in task:
            score = max(score, 0.46)
        elif "boolean" in task:
            score = max(score, 0.35)
        return min(1.0, score)


    def _materialize_examples(rows, target, meta, max_examples):
        q_candidates = _unique([
            target.get("question_field"),
            "question",
            "query",
            "input",
            "prompt",
            "problem",
            "text",
        ])
        a_candidates = _unique([
            target.get("answer_field"),
            "answer",
            "answers",
            "target",
            "label",
            "gold",
            "final_answer",
            "output",
        ])
        examples = []
        for idx, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            question = _question_to_text(_field_value(row, q_candidates))
            answer = _answer_to_text(_field_value(row, a_candidates))
            if not question or not answer:
                continue
            example_id = str(row.get("id") or row.get("qid") or row.get("question_id") or idx)
            examples.append({
                "example_id": example_id,
                "question": question,
                "answer": answer,
                "dataset_name": meta.get("name") or target.get("name"),
                "dataset_id": meta.get("id"),
                "dataset_config": meta.get("config"),
                "split": meta.get("split"),
                "task_type": target.get("task_type") or "qa",
                "difficulty": _difficulty_proxy(question, target.get("task_type") or "qa"),
            })
        if max_examples > 0:
            examples = examples[: max_examples * 4]
        return examples


    def _load_benchmark_suites(max_examples):
        suites = []
        failures = []
        loaded_pool = []
        target_limit = int(os.getenv("DEEPGRAPH_BENCHMARK_TARGET_LIMIT", "0") or "0")
        target_name_filter = {
            item.strip().lower()
            for item in os.getenv("DEEPGRAPH_BENCHMARK_TARGET_NAMES", "").split(",")
            if item.strip()
        }
        targets = list(DEFAULTS["targets"])
        if target_name_filter:
            nonderived = [
                t
                for t in targets
                if not t.get("derive_from_loaded_benchmarks")
                and (
                    str(t.get("name") or "").lower() in target_name_filter
                    or str(t.get("hf_dataset") or "").lower() in target_name_filter
                )
            ]
            derived = [t for t in targets if t.get("derive_from_loaded_benchmarks")]
            targets = nonderived + derived
        if target_limit > 0:
            nonderived = [t for t in targets if not t.get("derive_from_loaded_benchmarks")][:target_limit]
            derived = [t for t in targets if t.get("derive_from_loaded_benchmarks")]
            targets = nonderived + derived
        for target in targets:
            if target.get("derive_from_loaded_benchmarks"):
                continue
            try:
                print("BENCHMARK_STAGE: materialize " + str(target.get("name")), flush=True)
                rows, meta = _load_hf_rows(target)
                examples = _materialize_examples(rows, target, meta, max_examples)
                if not examples:
                    raise RuntimeError("loaded dataset but could not infer question/answer fields")
                print(
                    "BENCHMARK_STAGE: materialized "
                    + str(target.get("name"))
                    + " examples="
                    + str(len(examples)),
                    flush=True,
                )
                suites.append({"target": target, "meta": meta, "examples": examples})
                loaded_pool.extend(examples)
            except Exception as exc:
                failures.append({"target": target.get("name"), "error": str(exc)})
                _append_jsonl("failure_cases.jsonl", {"stage": "load_dataset", "target": target.get("name"), "error": str(exc)})
        for target in targets:
            if not target.get("derive_from_loaded_benchmarks"):
                continue
            if not loaded_pool:
                failures.append({"target": target.get("name"), "error": "no loaded examples for derived stress split"})
                continue
            sorted_pool = sorted(loaded_pool, key=lambda ex: ex.get("difficulty", 0.0))
            k = min(max_examples if max_examples > 0 else 64, max(2, len(sorted_pool) // 2))
            easy = sorted_pool[: max(1, k // 2)]
            hard = sorted_pool[-max(1, k // 2):]
            stress = []
            for ex in easy + hard:
                row = dict(ex)
                row["dataset_name"] = target.get("name")
                row["dataset_id"] = "derived_from_loaded_benchmarks"
                row["split"] = "simple_vs_hard"
                row["task_type"] = "derived_stress_split"
                stress.append(row)
            suites.append({
                "target": target,
                "meta": {
                    "name": target.get("name"),
                    "id": "derived_from_loaded_benchmarks",
                    "config": "",
                    "split": "simple_vs_hard",
                    "derived": True,
                },
                "examples": stress,
            })
        if not suites:
            raise RuntimeError("No real benchmark suites loaded; refusing synthetic fallback. Failures: " + json.dumps(failures, ensure_ascii=False))
        return suites, failures


    def _extract_number(text):
        matches = re.findall(r"[-+]?\\d+(?:\\.\\d+)?", str(text).replace(",", ""))
        return matches[-1] if matches else ""


    def _normalize_text(text):
        return re.sub(r"\\s+", " ", re.sub(r"[^a-z0-9\\s]+", " ", str(text or "").lower())).strip()


    def _extract_final_answer(text):
        raw = str(text or "")
        markers = ["final answer:", "answer:"]
        lowered = raw.lower()
        for marker in markers:
            if marker in lowered:
                raw = raw[lowered.rfind(marker) + len(marker):]
                break
        return raw.strip()


    def _token_f1(prediction, gold):
        pred_tokens = _normalize_text(prediction).split()
        gold_tokens = _normalize_text(gold).split()
        if not pred_tokens or not gold_tokens:
            return 0.0
        overlap = collections.Counter(pred_tokens) & collections.Counter(gold_tokens)
        common = sum(overlap.values())
        if common == 0:
            return 0.0
        precision = common / len(pred_tokens)
        recall = common / len(gold_tokens)
        return 2 * precision * recall / (precision + recall)


    def _score_answer(prediction, gold, task_type):
        final = _extract_final_answer(prediction)
        gold_text = _answer_to_text(gold)
        pred_norm = _normalize_text(final)
        gold_norm = _normalize_text(gold_text)
        pred_num = _extract_number(final)
        gold_num = _extract_number(gold_text)
        numeric_exact = 0.0
        if pred_num and gold_num:
            try:
                numeric_exact = 1.0 if math.isclose(float(pred_num), float(gold_num), rel_tol=1e-4, abs_tol=1e-4) else 0.0
            except ValueError:
                numeric_exact = 1.0 if pred_num == gold_num else 0.0
        bool_gold = gold_norm in {"yes", "no", "true", "false"}
        bool_pred = "yes" if re.search(r"\\byes\\b|\\btrue\\b", pred_norm) else "no" if re.search(r"\\bno\\b|\\bfalse\\b", pred_norm) else pred_norm
        exact = 1.0 if pred_norm == gold_norm else numeric_exact
        if bool_gold:
            exact = 1.0 if bool_pred in {gold_norm, "yes" if gold_norm == "true" else "no" if gold_norm == "false" else gold_norm} else 0.0
        f1 = max(exact, _token_f1(final, gold_text))
        primary = exact if task_type in {"math_qa", "boolean_qa"} else f1
        return {
            "exact": float(exact),
            "f1": float(f1),
            "primary_score": float(primary),
            "prediction_answer": final,
            "gold_answer": gold_text,
        }


    def _build_prompt(question, kind, *, difficulty=0.0):
        if kind == "direct":
            return "Answer the question. Give only the final answer.\\nQuestion: " + question + "\\nAnswer:"
        if kind == "fixed_cot":
            return "Answer the question. Think step by step, then write 'Final answer: <answer>'.\\nQuestion: " + question + "\\nSolution:"
        if kind == "least_to_most":
            return (
                "Decompose the question into the smallest useful subquestions, solve them in order, "
                "then write 'Final answer: <answer>'.\\nQuestion: " + question + "\\nSolution:"
            )
        if kind.startswith("cggr") or kind in {"confidence_gate", "disagreement_gate", "random_budget_matched"}:
            return (
                "Choose the smallest sufficient response. If the answer is clear, do not reason. "
                "If deliberation is useful, use at most two concise reasoning sentences. "
                "Use deliberate reasoning only when the question structure, counterfactual risk, or uncertainty justifies it. "
                "End with exactly one line: 'Final answer: <answer>'. Do not repeat the final answer or add text after it."
                "\\nQuestion: " + question + f"\\nDifficulty proxy: {difficulty:.3f}\\nSolution:"
            )
        return "Answer the question and end with 'Final answer: <answer>'.\\nQuestion: " + question + "\\nSolution:"


    def _max_tokens_for_kind(kind, difficulty):
        if kind == "direct":
            return 48
        if kind == "confidence_gate":
            return 192 if difficulty >= 0.50 else 56
        if kind == "disagreement_gate":
            return 192 if difficulty >= 0.50 else 56
        if kind == "random_budget_matched":
            return 192 if difficulty >= 0.50 else 56
        if kind == "cggr":
            return 224 if difficulty >= 0.42 else 64
        if kind == "cggr_ablate_counterfactual":
            return 192 if difficulty >= 0.58 else 56
        if kind == "cggr_ablate_lcb":
            return 224 if difficulty >= 0.34 else 80
        if kind == "cggr_ablate_divergence":
            return 192 if difficulty >= 0.38 else 64
        if kind == "cggr_ablate_qstruct":
            return 192 if len(str(difficulty).split()) > 999 else 96
        return 192


    def _coerce_tokenizer_encoding(encoded):
        if hasattr(encoded, "data") and isinstance(encoded.data, dict):
            encoded = dict(encoded.data)
        elif isinstance(encoded, dict):
            encoded = dict(encoded)
        else:
            encoded = {"input_ids": encoded}
        if "input_ids" not in encoded:
            raise RuntimeError("Tokenizer encoding missing input_ids")
        return encoded


    def _encode_prompt(tokenizer, prompt):
        use_chat_template = os.getenv("DEEPGRAPH_BENCHMARK_USE_CHAT_TEMPLATE", "1").strip().lower() not in {"0", "false", "no", "off"}
        if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
            try:
                messages = [{"role": "user", "content": prompt}]
                try:
                    encoded = tokenizer.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=True,
                        return_tensors="pt",
                        return_dict=True,
                    )
                except TypeError:
                    encoded = tokenizer.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=True,
                        return_tensors="pt",
                    )
                return _coerce_tokenizer_encoding(encoded)
            except Exception as exc:
                print(
                    "BENCHMARK_STAGE: chat_template_fallback "
                    + json.dumps({"error_type": type(exc).__name__, "error": repr(exc)[:300]}, ensure_ascii=False),
                    flush=True,
                )
        return _coerce_tokenizer_encoding(
            tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=int(os.getenv("DEEPGRAPH_BENCHMARK_MAX_INPUT_TOKENS", "1536")),
            )
        )


    def _generate(model, tokenizer, prompt, *, max_new_tokens, do_sample=False, temperature=0.0):
        encoded = _encode_prompt(tokenizer, prompt)
        encoded = {
            key: value.to(model.device) if hasattr(value, "to") else torch.as_tensor(value, device=model.device)
            for key, value in encoded.items()
        }
        before = int(encoded["input_ids"].shape[-1])
        kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": tokenizer.eos_token_id,
            "do_sample": bool(do_sample),
        }
        if do_sample:
            kwargs["temperature"] = max(0.05, float(temperature))
            kwargs["top_p"] = 0.95
        with torch.no_grad():
            out = model.generate(**encoded, **kwargs)
        generated = out[0, before:]
        token_count = int(generated.numel())
        if token_count <= 0:
            raise RuntimeError("LLM generation returned zero new tokens")
        text = tokenizer.decode(generated, skip_special_tokens=True).strip()
        if not text:
            raise RuntimeError("LLM generation returned empty decoded text")
        return text, token_count


    def _modelscope_snapshot(model_id):
        disabled = os.getenv("DEEPGRAPH_DISABLE_MODELSCOPE_FALLBACK", "").strip().lower()
        if disabled in {"1", "true", "yes", "on"}:
            raise RuntimeError("ModelScope fallback disabled")
        from modelscope import snapshot_download
        return snapshot_download(os.getenv("DEEPGRAPH_MODELSCOPE_MODEL", model_id))


    def _load_model():
        model_id = os.getenv("DEEPGRAPH_BENCHMARK_MODEL", DEFAULT_MODEL_ID)
        if not torch.cuda.is_available():
            raise RuntimeError("Real LLM benchmark requires CUDA. No synthetic or mocked fallback is allowed.")
        model_path = model_id
        prefer_modelscope_default = "1" if "qwen" in model_id.lower() else "0"
        prefer_modelscope = os.getenv("DEEPGRAPH_PREFER_MODELSCOPE", prefer_modelscope_default).strip().lower()
        if prefer_modelscope in {"1", "true", "yes", "on"}:
            try:
                print("BENCHMARK_STAGE: modelscope_snapshot " + str(model_id), flush=True)
                model_path = _modelscope_snapshot(model_id)
            except Exception as exc:
                print("WARNING: ModelScope prefetch failed; falling back to Hugging Face: " + str(exc), flush=True)
                model_path = model_id
        print("BENCHMARK_STAGE: load_tokenizer " + str(model_id), flush=True)
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        except Exception as exc:
            if not os.getenv("DEEPGRAPH_MODELSCOPE_MODEL") and "qwen" not in model_id.lower():
                raise
            print("WARNING: Hugging Face tokenizer load failed; trying ModelScope snapshot: " + str(exc), flush=True)
            model_path = _modelscope_snapshot(model_id)
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        load_kwargs = {"torch_dtype": torch.float16, "device_map": "auto", "trust_remote_code": True}
        if os.getenv("DEEPGRAPH_BENCHMARK_LOAD_IN_4BIT", "1").strip().lower() in {"1", "true", "yes", "on"}:
            try:
                from transformers import BitsAndBytesConfig
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
            except Exception as exc:
                print("WARNING: 4bit quantization unavailable; continuing with fp16 real-model load: " + str(exc), flush=True)
        print("BENCHMARK_STAGE: load_model " + str(model_path), flush=True)
        try:
            model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        except Exception as exc:
            if model_path != model_id:
                raise
            if not os.getenv("DEEPGRAPH_MODELSCOPE_MODEL") and "qwen" not in model_id.lower():
                raise
            print("WARNING: Hugging Face weight load failed; trying ModelScope snapshot: " + str(exc), flush=True)
            model_path = _modelscope_snapshot(model_id)
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        model.eval()
        print("BENCHMARK_STAGE: model_ready " + str(model_id), flush=True)
        return model, tokenizer, model_id


    def _sample_examples(examples, seed, max_examples):
        rng = random.Random(seed)
        if max_examples <= 0 or len(examples) <= max_examples:
            return list(examples)
        indices = rng.sample(range(len(examples)), max_examples)
        return [examples[i] for i in indices]


    def _run_single(model, tokenizer, example, method_name, spec, *, seed):
        kind = spec["kind"]
        difficulty = float(example.get("difficulty") or 0.0)
        if kind == "random_budget_matched":
            rng = random.Random(str(seed) + "::" + str(example.get("example_id") or example.get("question") or ""))
            deliberate = rng.random() < 0.5
            prompt_kind = "fixed_cot" if deliberate else "direct"
            output, tokens = _generate(
                model,
                tokenizer,
                _build_prompt(example["question"], prompt_kind, difficulty=difficulty),
                max_new_tokens=192 if deliberate else 56,
            )
            return output, tokens, {
                "difficulty": difficulty,
                "kind": kind,
                "routed_to_deliberation": deliberate,
                "random_budget_matched": True,
            }
        if kind == "disagreement_gate":
            direct_prompt = _build_prompt(example["question"], "direct", difficulty=difficulty)
            out_a, tok_a = _generate(model, tokenizer, direct_prompt, max_new_tokens=48, do_sample=False)
            out_b, tok_b = _generate(model, tokenizer, direct_prompt, max_new_tokens=48, do_sample=True, temperature=0.7)
            disagree = _normalize_text(_extract_final_answer(out_a)) != _normalize_text(_extract_final_answer(out_b))
            if disagree:
                cot_prompt = _build_prompt(example["question"], "fixed_cot", difficulty=difficulty)
                output, tok_c = _generate(model, tokenizer, cot_prompt, max_new_tokens=192)
                return output, tok_a + tok_b + tok_c, {
                    "difficulty": difficulty,
                    "kind": kind,
                    "routed_to_deliberation": True,
                    "short_answer_a": _extract_final_answer(out_a),
                    "short_answer_b": _extract_final_answer(out_b),
                }
            return out_a, tok_a + tok_b, {
                "difficulty": difficulty,
                "kind": kind,
                "routed_to_deliberation": False,
                "short_answer_a": _extract_final_answer(out_a),
                "short_answer_b": _extract_final_answer(out_b),
            }
        if kind == "self_consistency":
            samples = max(1, int(os.getenv("DEEPGRAPH_SELF_CONSISTENCY_SAMPLES", "3")))
            outputs = []
            total_tokens = 0
            for sample_idx in range(samples):
                prompt = _build_prompt(example["question"], "fixed_cot", difficulty=difficulty)
                output, tokens = _generate(
                    model,
                    tokenizer,
                    prompt,
                    max_new_tokens=192,
                    do_sample=sample_idx > 0,
                    temperature=0.7,
                )
                outputs.append(output)
                total_tokens += tokens
            votes = collections.Counter(_normalize_text(_extract_final_answer(text)) for text in outputs)
            winner_norm = votes.most_common(1)[0][0] if votes else ""
            chosen = next((text for text in outputs if _normalize_text(_extract_final_answer(text)) == winner_norm), outputs[-1])
            return chosen, total_tokens, {"samples": samples, "vote": winner_norm}
        max_new_tokens = _max_tokens_for_kind(kind, difficulty)
        selective_kind = kind.startswith("cggr") or kind in {"confidence_gate", "random_budget_matched"}
        prompt_kind = "direct" if kind == "direct" or (selective_kind and max_new_tokens <= 80) else kind
        output, tokens = _generate(
            model,
            tokenizer,
            _build_prompt(example["question"], prompt_kind, difficulty=difficulty),
            max_new_tokens=max_new_tokens,
        )
        route = {
            "difficulty": difficulty,
            "kind": kind,
            "max_new_tokens": max_new_tokens,
            "routed_to_deliberation": bool(max_new_tokens > 80),
        }
        return output, tokens, route


    def _mean(values):
        return float(sum(values) / max(1, len(values)))


    def _std(values):
        return float(statistics.stdev(values)) if len(values) > 1 else 0.0


    def _paired_permutation_pvalue(candidate, baseline):
        pairs = [(float(c), float(b)) for c, b in zip(candidate, baseline)]
        if not pairs:
            return 1.0
        observed = abs(sum(c - b for c, b in pairs) / len(pairs))
        count = 0
        extreme = 0
        for signs in itertools.product([-1, 1], repeat=len(pairs)):
            diff = abs(sum(sign * (c - b) for sign, (c, b) in zip(signs, pairs)) / len(pairs))
            count += 1
            if diff >= observed - 1e-12:
                extreme += 1
        return float(extreme / max(1, count))


    def _bootstrap_ci(values, rounds=2000):
        if not values:
            return [0.0, 0.0]
        rng = random.Random(12345)
        means = []
        for _ in range(rounds):
            sample = [values[rng.randrange(len(values))] for _ in values]
            means.append(_mean(sample))
        means.sort()
        lo = means[int(0.025 * (len(means) - 1))]
        hi = means[int(0.975 * (len(means) - 1))]
        return [float(lo), float(hi)]


    def main():
        started = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        requested_max_examples = _env_int("DEEPGRAPH_BENCHMARK_MAX_EXAMPLES", DEFAULT_MAX_EXAMPLES)
        requested_seeds = _env_int("DEEPGRAPH_BENCHMARK_SEEDS", DEFAULT_SEEDS)
        max_examples, seeds = _apply_runtime_budget(requested_max_examples, requested_seeds)
        seed_values = _selected_seed_values(seeds)
        lambda_cost = float(os.getenv("DEEPGRAPH_BENCHMARK_COST_LAMBDA", "0.03"))
        print(
            "BENCHMARK_STAGE: start "
            + json.dumps(
                {
                    "max_examples": max_examples,
                    "requested_max_examples": requested_max_examples,
                    "seeds": seeds,
                    "seed_values": seed_values,
                    "requested_seeds": requested_seeds,
                    "targets": [t.get("name") for t in DEFAULTS["targets"]],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        suites, load_failures = _load_benchmark_suites(max_examples)
        print("BENCHMARK_STAGE: datasets_ready count=" + str(len(suites)), flush=True)
        model, tokenizer, model_id = _load_model()
        method_specs = _method_specs_for_run()
        _touch_result_file("failure_cases.jsonl")
        environment_report_path = _write_environment_report(model_id, method_specs, seed_values, max_examples)
        seed_results = []
        aggregate = {
            method: {"score": 0.0, "exact": 0.0, "f1": 0.0, "count": 0, "tokens": 0.0, "latency": 0.0, "routed": 0.0}
            for method in method_specs
        }
        per_dataset_results = {}
        per_seed_method_values = {method: [] for method in method_specs}
        per_example_scores = {}
        difficulty_breakdown_acc = {}

        _write_json("run_config.json", {
            "method": METHOD_NAME,
            "metric_name": METRIC_NAME,
            "model_id": model_id,
            "targets": DEFAULTS["targets"],
            "seeds": seeds,
            "seed_values": seed_values,
            "max_examples_per_dataset_seed": max_examples,
            "methods": list(method_specs.keys()),
            "ablations": DEFAULTS.get("ablations", []),
            "cost_lambda": lambda_cost,
            "prompt_template": "method-specific direct, chain-of-thought, gating, disagreement, and CGGR prompts in _build_prompt",
            "decoding": {"default": "greedy", "self_consistency_extra_samples": "temperature=0.7, top_p=0.95"},
            "reasoning_budget": {"direct": 48, "short_gate": 56, "cot": 192, "cggr": "64-224 max_new_tokens by difficulty"},
        })

        for seed in seed_values:
            print("BENCHMARK_STAGE: eval_seed seed=" + str(seed), flush=True)
            seed_row = {"seed": seed, "datasets": {}, "methods": {}}
            for suite in suites:
                dataset_name = suite["meta"].get("name") or suite["target"].get("name")
                examples = _sample_examples(suite["examples"], seed, max_examples)
                print(
                    "BENCHMARK_STAGE: eval_dataset "
                    + json.dumps({"seed": seed, "dataset": dataset_name, "examples": len(examples)}, ensure_ascii=False),
                    flush=True,
                )
                seed_row["datasets"][dataset_name] = {"num_examples": len(examples), "methods": {}}
                per_dataset_results.setdefault(dataset_name, {})
                for method_name, spec in method_specs.items():
                    print(
                        "BENCHMARK_STAGE: eval_method "
                        + json.dumps({"seed": seed, "dataset": dataset_name, "method": method_name}, ensure_ascii=False),
                        flush=True,
                    )
                    total_score = 0.0
                    total_exact = 0.0
                    total_f1 = 0.0
                    total_tokens = 0.0
                    total_latency = 0.0
                    total_routed = 0.0
                    for ex in examples:
                        call_start = time.time()
                        try:
                            prediction, tokens, route = _run_single(model, tokenizer, ex, method_name, spec, seed=seed)
                            score = _score_answer(prediction, ex["answer"], ex.get("task_type") or "")
                        except Exception as exc:
                            failure = {
                                "stage": "generation_or_scoring",
                                "seed": seed,
                                "dataset": dataset_name,
                                "method": method_name,
                                "example_id": ex.get("example_id"),
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                                "error_repr": repr(exc),
                                "traceback": "".join(traceback.format_exception_only(type(exc), exc)).strip()[:4000],
                            }
                            _append_jsonl("failure_cases.jsonl", failure)
                            if os.getenv("DEEPGRAPH_BENCHMARK_CONTINUE_ON_ERROR", "0").strip().lower() not in {"1", "true", "yes", "on"}:
                                raise RuntimeError(
                                    "generation_or_scoring failed for "
                                    + json.dumps(
                                        {
                                            "seed": seed,
                                            "dataset": dataset_name,
                                            "method": method_name,
                                            "example_id": ex.get("example_id"),
                                            "error_type": type(exc).__name__,
                                            "error": str(exc),
                                            "error_repr": repr(exc),
                                        },
                                        ensure_ascii=False,
                                    )
                                ) from exc
                            prediction = ""
                            tokens = 0
                            route = {"error": str(exc), "error_type": type(exc).__name__, "error_repr": repr(exc)}
                            score = {"exact": 0.0, "f1": 0.0, "primary_score": 0.0, "prediction_answer": "", "gold_answer": ex["answer"]}
                        latency_seconds = time.time() - call_start
                        total_score += score["primary_score"]
                        total_exact += score["exact"]
                        total_f1 += score["f1"]
                        total_tokens += tokens
                        total_latency += latency_seconds
                        total_routed += 1.0 if route.get("routed_to_deliberation") else 0.0
                        key = (seed, dataset_name, ex.get("example_id"), method_name)
                        per_example_scores[key] = score["primary_score"]
                        difficulty = float(ex.get("difficulty") or 0.0)
                        difficulty_bucket = "easy" if difficulty < 0.33 else "medium" if difficulty < 0.66 else "hard"
                        bucket_acc = difficulty_breakdown_acc.setdefault(method_name, {}).setdefault(
                            difficulty_bucket,
                            {"score": 0.0, "tokens": 0.0, "latency": 0.0, "routed": 0.0, "count": 0},
                        )
                        bucket_acc["score"] += score["primary_score"]
                        bucket_acc["tokens"] += tokens
                        bucket_acc["latency"] += latency_seconds
                        bucket_acc["routed"] += 1.0 if route.get("routed_to_deliberation") else 0.0
                        bucket_acc["count"] += 1
                        raw_row = {
                            "seed": seed,
                            "dataset": dataset_name,
                            "dataset_id": ex.get("dataset_id"),
                            "split": ex.get("split"),
                            "method": method_name,
                            "example_id": ex.get("example_id"),
                            "question": ex.get("question"),
                            "gold": score.get("gold_answer"),
                            "prediction": prediction,
                            "prediction_answer": score.get("prediction_answer"),
                            "exact": score["exact"],
                            "f1": score["f1"],
                            "primary_score": score["primary_score"],
                            "new_tokens": tokens,
                            "latency_seconds": latency_seconds,
                        }
                        _append_jsonl("raw_predictions.jsonl", raw_row)
                        if spec["kind"].startswith("cggr") or spec["kind"] == "confidence_gate":
                            _append_jsonl("routing_decisions.jsonl", {
                                "seed": seed,
                                "dataset": dataset_name,
                                "method": method_name,
                                "example_id": ex.get("example_id"),
                                **route,
                            })
                        if method_name == "CGGR" and score["primary_score"] < 0.5:
                            _append_jsonl("failure_cases.jsonl", raw_row)
                    count = max(1, len(examples))
                    metric_value = (total_score / count) - lambda_cost * ((total_tokens / count) / 192.0)
                    row = {
                        "score": float(total_score / count),
                        "exact": float(total_exact / count),
                        "f1": float(total_f1 / count),
                        "avg_new_tokens": float(total_tokens / count),
                        "avg_latency_seconds": float(total_latency / count),
                        "route_rate": float(total_routed / count),
                        "cost_adjusted_accuracy": float(metric_value),
                        "metric_value": float(metric_value),
                        "count": count,
                    }
                    print(
                        "BENCHMARK_STAGE: eval_method_done "
                        + json.dumps(
                            {
                                "seed": seed,
                                "dataset": dataset_name,
                                "method": method_name,
                                "metric_value": row["metric_value"],
                                "count": count,
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                    seed_row["datasets"][dataset_name]["methods"][method_name] = row
                    seed_row["methods"].setdefault(method_name, {"score": 0.0, "tokens": 0.0, "latency": 0.0, "routed": 0.0, "count": 0})
                    seed_row["methods"][method_name]["score"] += total_score
                    seed_row["methods"][method_name]["tokens"] += total_tokens
                    seed_row["methods"][method_name]["latency"] += total_latency
                    seed_row["methods"][method_name]["routed"] += total_routed
                    seed_row["methods"][method_name]["count"] += count
                    aggregate[method_name]["score"] += total_score
                    aggregate[method_name]["exact"] += total_exact
                    aggregate[method_name]["f1"] += total_f1
                    aggregate[method_name]["tokens"] += total_tokens
                    aggregate[method_name]["latency"] += total_latency
                    aggregate[method_name]["routed"] += total_routed
                    aggregate[method_name]["count"] += count
                    bucket = per_dataset_results[dataset_name].setdefault(method_name, {"score": 0.0, "tokens": 0.0, "latency": 0.0, "routed": 0.0, "count": 0})
                    bucket["score"] += total_score
                    bucket["tokens"] += total_tokens
                    bucket["latency"] += total_latency
                    bucket["routed"] += total_routed
                    bucket["count"] += count
            for method_name, row in seed_row["methods"].items():
                count = max(1, int(row["count"]))
                value = (row["score"] / count) - lambda_cost * ((row["tokens"] / count) / 192.0)
                row["cost_adjusted_accuracy"] = float(value)
                row["metric_value"] = float(value)
                row["avg_latency_seconds"] = float(row["latency"] / count)
                row["route_rate"] = float(row["routed"] / count)
                per_seed_method_values[method_name].append(float(value))
            seed_results.append(seed_row)

        oracle_values = []
        for seed_row in seed_results:
            oracle_score = 0.0
            oracle_count = 0
            oracle_tokens = 0.0
            for dataset_name, dataset_row in seed_row["datasets"].items():
                direct = dataset_row["methods"].get("Vanilla Direct Answering", {})
                cot = dataset_row["methods"].get("Always-Reason Chain-of-Thought", {})
                oracle_score += max(float(direct.get("score", 0.0)), float(cot.get("score", 0.0))) * max(1, int(direct.get("count") or cot.get("count") or 0))
                oracle_tokens += min(float(direct.get("avg_new_tokens", 0.0) or 0.0), float(cot.get("avg_new_tokens", 0.0) or 0.0)) * max(1, int(direct.get("count") or cot.get("count") or 0))
                oracle_count += max(1, int(direct.get("count") or cot.get("count") or 0))
            oracle_metric = (oracle_score / max(1, oracle_count)) - lambda_cost * ((oracle_tokens / max(1, oracle_count)) / 192.0)
            oracle_values.append(float(oracle_metric))

        per_method = {}
        per_method_std = {}
        for method_name, row in aggregate.items():
            count = max(1, int(row["count"]))
            metric_value = (row["score"] / count) - lambda_cost * ((row["tokens"] / count) / 192.0)
            per_method[method_name] = {
                "score": float(row["score"] / count),
                "exact": float(row["exact"] / count),
                "f1": float(row["f1"] / count),
                "avg_new_tokens": float(row["tokens"] / count),
                "avg_latency_seconds": float(row["latency"] / count),
                "route_rate": float(row["routed"] / count),
                "cost_adjusted_accuracy": float(metric_value),
                "metric_value": float(metric_value),
                "count": count,
            }
            per_method_std[method_name] = _std(per_seed_method_values.get(method_name, []))
        if oracle_values:
            per_method["CGGR/oracle_router"] = {
                "cost_adjusted_accuracy": _mean(oracle_values),
                "metric_value": _mean(oracle_values),
                "score": _mean(oracle_values),
                "avg_new_tokens": 0.0,
                "avg_latency_seconds": 0.0,
                "route_rate": 1.0,
                "count": sum(int(row.get("num_examples") or 0) for seed_row in seed_results for row in seed_row.get("datasets", {}).values()),
                "upper_bound": True,
            }
            per_method_std["CGGR/oracle_router"] = _std(oracle_values)

        for dataset_name, methods in per_dataset_results.items():
            for method_name, row in methods.items():
                count = max(1, int(row["count"]))
                row["cost_adjusted_accuracy"] = float((row["score"] / count) - lambda_cost * ((row["tokens"] / count) / 192.0))
                row["metric_value"] = row["cost_adjusted_accuracy"]
                row["score"] = float(row["score"] / count)
                row["avg_new_tokens"] = float(row["tokens"] / count)
                row["avg_latency_seconds"] = float(row.get("latency", 0.0) / count)
                row["route_rate"] = float(row.get("routed", 0.0) / count)

        best_method = max(per_method, key=lambda key: per_method[key]["metric_value"])
        candidate_values = per_seed_method_values.get("CGGR", [])
        baseline_name = "Always-Reason Chain-of-Thought" if "Always-Reason Chain-of-Thought" in per_seed_method_values else "Vanilla Direct Answering"
        baseline_values = per_seed_method_values.get(baseline_name, [])
        bootstrap = {
            "candidate_method": "CGGR",
            "baseline_method": baseline_name,
            "candidate_ci95": _bootstrap_ci(candidate_values),
            "baseline_ci95": _bootstrap_ci(baseline_values),
            "paired_permutation_p": _paired_permutation_pvalue(candidate_values, baseline_values),
        }
        ablation_table = []
        for name in DEFAULTS.get("ablations", []):
            key = "CGGR/" + name
            if key in per_method:
                ablation_table.append({
                    "ablation": name,
                    "method": key,
                    "metric_value": per_method[key]["metric_value"],
                    "delta_vs_cggr": per_method[key]["metric_value"] - per_method.get("CGGR", {}).get("metric_value", 0.0),
                })
        if "CGGR/oracle_router" in per_method:
            ablation_table.append({
                "ablation": "oracle_router",
                "method": "CGGR/oracle_router",
                "metric_value": per_method["CGGR/oracle_router"]["metric_value"],
                "delta_vs_cggr": per_method["CGGR/oracle_router"]["metric_value"] - per_method.get("CGGR", {}).get("metric_value", 0.0),
                "upper_bound": True,
            })

        latency_tokens_table = []
        always_tokens = float(per_method.get("Always-Reason Chain-of-Thought", {}).get("avg_new_tokens", 0.0) or 0.0)
        always_latency = float(per_method.get("Always-Reason Chain-of-Thought", {}).get("avg_latency_seconds", 0.0) or 0.0)
        for method_name, row in per_method.items():
            avg_tokens = float(row.get("avg_new_tokens", 0.0) or 0.0)
            avg_latency = float(row.get("avg_latency_seconds", 0.0) or 0.0)
            latency_tokens_table.append({
                "method": method_name,
                "metric_value": float(row.get("metric_value", 0.0) or 0.0),
                "accuracy": float(row.get("score", 0.0) or 0.0),
                "avg_new_tokens": avg_tokens,
                "avg_latency_seconds": avg_latency,
                "route_rate": float(row.get("route_rate", 0.0) or 0.0),
                "token_saving_vs_always_reason": float(1.0 - (avg_tokens / always_tokens)) if always_tokens > 0 else 0.0,
                "latency_saving_vs_always_reason": float(1.0 - (avg_latency / always_latency)) if always_latency > 0 else 0.0,
            })
        cost_utility_tradeoff_table = latency_tokens_table

        difficulty_breakdown_table = []
        for method_name, buckets in difficulty_breakdown_acc.items():
            for bucket_name, row in buckets.items():
                count = max(1, int(row.get("count", 0)))
                difficulty_breakdown_table.append({
                    "method": method_name,
                    "difficulty": bucket_name,
                    "accuracy": float(row.get("score", 0.0) / count),
                    "avg_new_tokens": float(row.get("tokens", 0.0) / count),
                    "avg_latency_seconds": float(row.get("latency", 0.0) / count),
                    "route_rate": float(row.get("routed", 0.0) / count),
                    "count": count,
                })
        direct_easy = next((row for row in difficulty_breakdown_table if row["method"] == "Vanilla Direct Answering" and row["difficulty"] == "easy"), {})
        cggr_easy = next((row for row in difficulty_breakdown_table if row["method"] == "CGGR" and row["difficulty"] == "easy"), {})
        simple_case_degradation = {
            "subset": "easy",
            "baseline_method": "Vanilla Direct Answering",
            "candidate_method": "CGGR",
            "baseline_accuracy": direct_easy.get("accuracy"),
            "candidate_accuracy": cggr_easy.get("accuracy"),
            "degradation": (
                float(cggr_easy.get("accuracy", 0.0) - direct_easy.get("accuracy", 0.0))
                if direct_easy and cggr_easy
                else None
            ),
            "candidate_route_rate": cggr_easy.get("route_rate"),
        }
        calibration_reliability = []
        for bucket_name, proxy_value in (("easy", 0.17), ("medium", 0.50), ("hard", 0.83)):
            direct_row = next((row for row in difficulty_breakdown_table if row["method"] == "Vanilla Direct Answering" and row["difficulty"] == bucket_name), {})
            cggr_row = next((row for row in difficulty_breakdown_table if row["method"] == "CGGR" and row["difficulty"] == bucket_name), {})
            if direct_row and cggr_row:
                calibration_reliability.append({
                    "difficulty_bucket": bucket_name,
                    "difficulty_proxy": proxy_value,
                    "observed_gain_vs_direct": float(cggr_row.get("accuracy", 0.0) - direct_row.get("accuracy", 0.0)),
                    "route_rate": cggr_row.get("route_rate"),
                    "count": cggr_row.get("count"),
                })
        routing_analysis = {
            "methods": [
                {
                    "method": row["method"],
                    "route_rate": row["route_rate"],
                    "cost_saving": row["token_saving_vs_always_reason"],
                    "latency_saving": row["latency_saving_vs_always_reason"],
                    "avg_new_tokens": row["avg_new_tokens"],
                    "avg_latency_seconds": row["avg_latency_seconds"],
                    "utility": row["metric_value"],
                }
                for row in latency_tokens_table
                if any(token in row["method"].lower() for token in ("gate", "routing", "cggr", "oracle"))
            ],
            "easy_medium_hard_breakdown": difficulty_breakdown_table,
            "simple_case_degradation": simple_case_degradation,
            "calibration_reliability": calibration_reliability,
        }

        datasets_observed = [
            {
                "name": suite["meta"].get("name") or suite["target"].get("name"),
                "id": suite["meta"].get("id"),
                "config": suite["meta"].get("config"),
                "split": suite["meta"].get("split"),
                "num_materialized_examples": len(suite["examples"]),
                "license_or_source": suite["target"].get("hf_dataset") or suite["meta"].get("id") or suite["target"].get("name"),
                "preprocessing": "Answer normalization with exact/F1 scoring and task-specific numeric/boolean extraction.",
            }
            for suite in suites
        ]
        required_names = [target.get("name") for target in DEFAULTS["targets"]]
        observed_names = {str(row["name"]).lower() for row in datasets_observed}
        completed_required_datasets = all(str(name or "").lower() in observed_names for name in required_names if name)
        requested_methods = os.getenv("DEEPGRAPH_BENCHMARK_METHODS", "").strip()
        method_shard = bool(requested_methods and requested_methods.lower() not in {"all", "*"})
        target_shard = bool(
            os.getenv("DEEPGRAPH_BENCHMARK_TARGET_NAMES", "").strip()
            or int(os.getenv("DEEPGRAPH_BENCHMARK_TARGET_LIMIT", "0") or "0") > 0
        )
        seed_shard = seed_values != list(range(seeds))
        sharded_run = bool(method_shard or target_shard or seed_shard)
        full_completed = bool(
            not sharded_run
            and not load_failures
            and completed_required_datasets
            and len(seed_values) >= DEFAULT_SEEDS
            and all(name in per_method for name in METHOD_SPECS)
            and len(ablation_table) >= min(1, len(DEFAULTS.get("ablations", [])))
        )
        artifacts = {
            "run_config": _write_json("run_config.json", {
                "method": METHOD_NAME,
                "metric_name": METRIC_NAME,
                "model_id": model_id,
                "targets": DEFAULTS["targets"],
                "seeds": seeds,
                "seed_values": seed_values,
                "sharded_run": sharded_run,
                "shard_axes": {
                    "method": method_shard,
                    "target": target_shard,
                    "seed": seed_shard,
                },
                "max_examples_per_dataset_seed": max_examples,
                "methods": list(method_specs.keys()),
                "ablations": DEFAULTS.get("ablations", []),
                "cost_lambda": lambda_cost,
                "prompt_template": "method-specific direct, chain-of-thought, gating, disagreement, and CGGR prompts in _build_prompt",
                "decoding": {"default": "greedy", "self_consistency_extra_samples": "temperature=0.7, top_p=0.95"},
                "reasoning_budget": {"direct": 48, "short_gate": 56, "cot": 192, "cggr": "64-224 max_new_tokens by difficulty"},
            }),
            "per_seed_results": _write_json("per_seed_results.json", seed_results),
            "per_dataset_results": _write_json("per_dataset_results.json", per_dataset_results),
            "main_results_table": _write_json("main_results_table.json", per_method),
            "cost_utility_tradeoff_table": _write_json("cost_utility_tradeoff_table.json", cost_utility_tradeoff_table),
            "ablation_table": _write_json("ablation_table.json", ablation_table),
            "difficulty_breakdown_table": _write_json("difficulty_breakdown_table.json", difficulty_breakdown_table),
            "routing_analysis": _write_json("routing_analysis.json", routing_analysis),
            "latency_tokens_table": _write_json("latency_tokens_table.json", latency_tokens_table),
            "simple_case_degradation": _write_json("simple_case_degradation.json", simple_case_degradation),
            "calibration_reliability": _write_json("calibration_reliability.json", calibration_reliability),
            "bootstrap_ci": _write_json("bootstrap_ci.json", bootstrap),
        }
        artifacts["environment_report"] = environment_report_path
        artifacts["raw_predictions"] = os.path.join(_results_dir(), "raw_predictions.jsonl")
        artifacts["routing_decisions"] = os.path.join(_results_dir(), "routing_decisions.jsonl")
        artifacts["failure_cases"] = os.path.join(_results_dir(), "failure_cases.jsonl")
        peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0
        result = {
            "primary_metric": "cost_adjusted_accuracy",
            "metric_name": "cost_adjusted_accuracy",
            "candidate_method": "CGGR",
            "best_method": best_method,
            "per_method": per_method,
            "per_method_std": per_method_std,
            "seed_results": seed_results,
            "num_seeds": seeds,
            "datasets": datasets_observed,
            "dataset": datasets_observed[0] if datasets_observed else {},
            "dataset_aliases": _unique(
                [row.get("name") for row in datasets_observed]
                + [row.get("id") for row in datasets_observed]
            ),
            "model": {"id": model_id, "backend": "transformers", "cuda": bool(torch.cuda.is_available())},
            "baseline_aliases": BASELINE_ALIASES,
            "method_aliases": BASELINE_ALIASES,
            "ablations": [row["ablation"] for row in ablation_table],
            "ablation_results": ablation_table,
            "ablation_table": ablation_table,
            "cost_utility_tradeoff_table": cost_utility_tradeoff_table,
            "difficulty_breakdown_table": difficulty_breakdown_table,
            "routing_analysis": routing_analysis,
            "latency_tokens_table": latency_tokens_table,
            "simple_case_degradation": simple_case_degradation,
            "calibration_reliability": calibration_reliability,
            "bootstrap_ci": bootstrap,
            "load_failures": load_failures,
            "budget": {
                "seeds": seeds,
                "max_examples_per_dataset_seed": max_examples,
                "methods": list(method_specs.keys()),
                "cost_lambda": lambda_cost,
                "target_count": len(DEFAULTS["targets"]),
            },
            "method": METHOD_NAME,
            "duration_seconds": time.time() - started,
            "peak_vram_mb": peak_mb,
            "hardware": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "full_benchmark_completed": full_completed,
            "artifact_paths": artifacts,
            METRIC_NAME: per_method.get("CGGR", {}).get("metric_value", 0.0),
        }
        result["model"] = {
            **result["model"],
            "hardware": result["hardware"],
            "prompt_template": "method-specific direct, chain-of-thought, gating, disagreement, and CGGR prompts in _build_prompt",
            "decoding": {"default": "greedy", "self_consistency_extra_samples": "temperature=0.7, top_p=0.95"},
            "reasoning_budget": {"direct": 48, "short_gate": 56, "cot": 192, "cggr": "64-224 max_new_tokens by difficulty"},
        }
        artifacts["artifact_manifest"] = _write_json("artifact_manifest.json", {
            "full_benchmark_completed": full_completed,
            "artifacts": artifacts,
            "datasets": datasets_observed,
            "methods": list(per_method.keys()),
            "model": result["model"],
            "hardware": result["hardware"],
            "load_failures": load_failures,
        })
        result["artifact_paths"] = artifacts
        _write_json("benchmark_summary.json", result)
        print("method: " + METHOD_NAME)
        print("model: " + model_id)
        print("datasets: " + ", ".join(row["name"] for row in datasets_observed))
        print(f"peak_vram_mb: {peak_mb:.1f}")
        print(f"{METRIC_NAME}: {per_method.get('CGGR', {}).get('metric_value', 0.0):.6f}")
        print("FINAL_RESULTS: " + json.dumps(result, ensure_ascii=False))


    if __name__ == "__main__":
        main()
    """).replace("__DEEPGRAPH_DEFAULTS_JSON__", defaults_json)


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
        "benchmark_max_examples_per_seed": EXPERIMENT_REAL_BENCHMARK_MAX_EXAMPLES,
        "benchmark_seeds": EXPERIMENT_REAL_BENCHMARK_SEEDS,
        "benchmark_time_budget_seconds": EXPERIMENT_REAL_BENCHMARK_TIME_BUDGET,
    }
    publication_contract = plan.get("publication_evidence_contract") if isinstance(plan, dict) else {}
    if isinstance(publication_contract, dict) and publication_contract.get("benchmark_manifest"):
        proxy["benchmark_manifest"] = publication_contract["benchmark_manifest"]
    if isinstance(publication_contract, dict) and publication_contract.get("claim_route"):
        proxy["claim_route"] = publication_contract["claim_route"]
        proxy["claim_strength"] = publication_contract.get("claim_strength")
    if judgement is not None:
        proxy["formal_experiment"] = judgement.formal_experiment
        proxy["smoke_test_only"] = judgement.smoke_test_only
        proxy["experiment_judgement"] = judgement.to_dict()
    return proxy


def forge_experiment(insight_id: int) -> dict:
    """Full forge pipeline: scout codebase -> setup workspace -> generate scaffold.

    Creates an experiment_run row and returns all paths/configs needed
    for the validation loop.
    """
    print(f"[FORGE] Starting experiment forge for insight {insight_id}...", flush=True)

    insight = db.fetchone("SELECT * FROM deep_insights WHERE id=?", (insight_id,))
    if not insight:
        return {"error": f"Deep insight {insight_id} not found"}

    gate = evosci_strict_gate_insight(dict(insight))
    if gate:
        print(f"[FORGE] Blocked by EvoScientist strict gate: {gate.get('error')}", flush=True)
        return gate

    parsed = _autofill_experiment_contracts(dict(insight))
    _persist_enriched_insight(insight_id, parsed)
    spec = DeepInsightSpec.from_raw(parsed)
    plan = spec.experimental_plan
    evidence_plan = spec.evidence_plan
    layout = get_idea_workspace(insight_id, insight=parsed, create=True, sync_db=True)

    # Step 1: Scout codebase
    print(f"[FORGE] Scouting codebase...", flush=True)
    codebase = scout_codebase(parsed)
    print(f"[FORGE] Selected: {codebase.get('name', '?')} ({codebase.get('url', '?')})", flush=True)

    run_id = db.insert_returning_id(
        """
        INSERT INTO experiment_runs (deep_insight_id, experiment_suite, status, phase, workdir, codebase_url, codebase_ref, baseline_metric_name)
        VALUES (?, ?, 'scaffolding', 'setup', ?, ?, ?, ?)
        RETURNING id
        """,
        (
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
        db.execute("DELETE FROM experiment_runs WHERE id=?", (run_id,))
        db.commit()
        return {
            "error": judgement.summary or "Experiment review blocked formalization",
            "judgement": judgement.to_dict(),
            "route": judgement.recommended_route,
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
    proxy["paper_intent"] = proxy["publication_evidence_contract"].get("paper_intent", {})
    proxy["claim_route"] = proxy["publication_evidence_contract"].get("claim_route", {})
    proxy["claim_strength"] = proxy["publication_evidence_contract"].get("claim_strength")
    _checkpoint_run_state(
        run_id,
        phase="review_decision_ready",
        workdir=workdir,
        codebase=codebase,
        proxy_config=proxy,
        baseline_metric_name="metric",
    )

    # Step 3: Generate scaffold
    print(f"[FORGE] Generating scaffold (program.md, evaluate.py, success_criteria)...", flush=True)
    scaffold = generate_scaffold(parsed, codebase, workdir)

    # Step 4: Build proxy config
    success = scaffold.get("success_criteria", {})
    if scaffold.get("publication_evidence_contract"):
        proxy["publication_evidence_contract"] = scaffold["publication_evidence_contract"]
        proxy["benchmark_manifest"] = scaffold["publication_evidence_contract"].get("benchmark_manifest", {})
        proxy["paper_intent"] = scaffold["publication_evidence_contract"].get("paper_intent", {})
        proxy["claim_route"] = scaffold["publication_evidence_contract"].get("claim_route", {})
        proxy["claim_strength"] = scaffold["publication_evidence_contract"].get("claim_strength")
    if scaffold.get("benchmark_manifest"):
        proxy["benchmark_manifest"] = scaffold["benchmark_manifest"]
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
        INSERT INTO experiment_artifacts (run_id, artifact_type, path, metadata)
        VALUES (?, ?, ?, ?)
        """,
        (
            run_id,
            "source_data",
            plan_paths["experiment_spec.json"],
            json.dumps({"contract_type": "ExperimentSpec"}),
        ),
    )
    db.execute(
        """
        INSERT INTO experiment_artifacts (run_id, artifact_type, path, metadata)
        VALUES (?, ?, ?, ?)
        """,
        (
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
        "UPDATE deep_insights SET status=?, evoscientist_workdir=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
        (new_insight_status, str(layout["workspace_root"]), insight_id),
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
    }
