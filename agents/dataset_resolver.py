"""Resolve benchmark names into executable public dataset recipes.

Idea discovery often names a plausible benchmark before the experiment forge
knows how to load it. This module fills that gap conservatively: it searches
Hugging Face dataset metadata, scores candidates, and returns loadable recipes.
It never marks a dataset executable unless a concrete dataset id is available.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

import httpx

from config import (
    DATASET_RESOLVER_ALLOW_NETWORK,
    DATASET_RESOLVER_ENABLED,
    DATASET_RESOLVER_HF_LIMIT,
    DATASET_RESOLVER_MIN_CONFIDENCE,
    DATASET_RESOLVER_TIMEOUT_SECONDS,
    EXPERIMENT_REAL_BENCHMARK_MAX_EXAMPLES,
)


COMMON_HF_DATASETS: dict[str, dict[str, Any]] = {
    "gsm8k": {
        "name": "GSM8K",
        "hf_dataset": "openai/gsm8k",
        "config": "main",
        "split": "test",
        "task_type": "math_qa",
        "question_field": "question",
        "answer_field": "answer",
        "confidence": 0.98,
    },
    "strategyqa": {
        "name": "StrategyQA",
        "hf_dataset": "tasksource/strategy-qa",
        "config": "",
        "split": "validation",
        "task_type": "boolean_qa",
        "question_field": "question",
        "answer_field": "answer",
        "confidence": 0.9,
    },
    "boolq": {
        "name": "BoolQ",
        "hf_dataset": "google/boolq",
        "config": "",
        "split": "validation",
        "task_type": "boolean_qa",
        "question_field": "question",
        "answer_field": "answer",
        "confidence": 0.94,
    },
    "commonsenseqa": {
        "name": "CommonsenseQA",
        "hf_dataset": "tau/commonsense_qa",
        "config": "",
        "split": "validation",
        "task_type": "qa",
        "question_field": "question",
        "answer_field": "answerKey",
        "confidence": 0.92,
    },
    "openbookqa": {
        "name": "OpenBookQA",
        "hf_dataset": "allenai/openbookqa",
        "config": "main",
        "config_candidates": ["main", ""],
        "split": "validation",
        "split_candidates": ["validation", "test", "train"],
        "task_type": "qa",
        "question_field": "question_stem",
        "question_field_candidates": ["question_stem", "question"],
        "answer_field": "answerKey",
        "answer_field_candidates": ["answerKey", "answer"],
        "confidence": 0.98,
    },
    "qasc": {
        "name": "QASC",
        "hf_dataset": "allenai/qasc",
        "config": "default",
        "config_candidates": ["default", ""],
        "split": "validation",
        "split_candidates": ["validation", "test", "train"],
        "task_type": "qa",
        "question_field": "question",
        "question_field_candidates": ["question", "formatted_question"],
        "answer_field": "answerKey",
        "answer_field_candidates": ["answerKey", "answer"],
        "confidence": 0.98,
    },
    "ai2arc": {
        "name": "AI2 ARC",
        "hf_dataset": "allenai/ai2_arc",
        "config": "ARC-Challenge",
        "split": "test",
        "task_type": "qa",
        "question_field": "question",
        "answer_field": "answerKey",
        "confidence": 0.9,
    },
    "arcchallenge": {
        "name": "ARC-Challenge",
        "hf_dataset": "allenai/ai2_arc",
        "config": "ARC-Challenge",
        "split": "test",
        "task_type": "qa",
        "question_field": "question",
        "answer_field": "answerKey",
        "confidence": 0.9,
    },
    "mmlu": {
        "name": "MMLU",
        "hf_dataset": "cais/mmlu",
        "config": "all",
        "split": "test",
        "task_type": "qa",
        "question_field": "question",
        "answer_field": "answer",
        "confidence": 0.86,
    },
    "mbpp": {
        "name": "MBPP",
        "hf_dataset": "google-research-datasets/mbpp",
        "config": "",
        "split": "test",
        "task_type": "code_generation",
        "question_field": "text",
        "answer_field": "code",
        "confidence": 0.92,
    },
    "cifar10": {
        "name": "CIFAR-10",
        "hf_dataset": "cifar10",
        "config": "",
        "split": "test",
        "task_type": "image_classification",
        "confidence": 0.92,
    },
}

UNRESOLVABLE_MARKERS = (
    "foia",
    "privilege",
    "trec 2010 legal",
    "enron privilege",
    "private",
    "proprietary",
    "restricted",
)


@dataclass(frozen=True)
class DatasetCandidate:
    dataset_id: str
    score: float
    downloads: int = 0
    likes: int = 0
    tags: tuple[str, ...] = ()
    card_data: dict[str, Any] | None = None


def _text(value: Any) -> str:
    return str(value or "").strip()


def normalize_query(value: str) -> str:
    text = _text(value).lower()
    text = re.sub(r"\([^)]*\)", " ", text)
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _canonical(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", _text(value).lower())


def _tokens(value: str) -> set[str]:
    return {token for token in normalize_query(value).split() if len(token) >= 3}


def _common_recipe(name: str) -> dict[str, Any] | None:
    key = _canonical(name)
    for alias, recipe in COMMON_HF_DATASETS.items():
        if key == alias or alias in key or key in alias:
            out = dict(recipe)
            out["requested_name"] = name
            out["source"] = "local_registry"
            out["resolved"] = True
            out["url"] = f"https://huggingface.co/datasets/{out['hf_dataset']}"
            return out
    return None


def _auth_headers() -> dict[str, str]:
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    headers = {"User-Agent": "DeepGraph-DatasetResolver/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _candidate_score(query: str, row: dict[str, Any]) -> float:
    dataset_id = _text(row.get("id") or row.get("_id"))
    if not dataset_id:
        return 0.0
    query_tokens = _tokens(query)
    id_tokens = _tokens(dataset_id.replace("/", " "))
    tags = [str(tag) for tag in row.get("tags") or []]
    tag_tokens = set().union(*(_tokens(tag) for tag in tags)) if tags else set()
    overlap = len(query_tokens & (id_tokens | tag_tokens))
    denom = max(1, len(query_tokens))
    lexical = overlap / denom
    exact_bonus = 0.35 if _canonical(query) in _canonical(dataset_id) else 0.0
    data_bonus = 0.0
    siblings = row.get("siblings") or []
    if any(str((sib or {}).get("rfilename", "")).endswith((".json", ".jsonl", ".parquet", ".csv")) for sib in siblings if isinstance(sib, dict)):
        data_bonus = 0.08
    popularity = min(0.15, (int(row.get("downloads") or 0) / 50000.0) * 0.1 + (int(row.get("likes") or 0) / 1000.0) * 0.05)
    return min(1.0, lexical * 0.55 + exact_bonus + data_bonus + popularity)


def search_huggingface_datasets(
    query: str,
    *,
    limit: int | None = None,
    timeout: float | None = None,
    client: httpx.Client | None = None,
) -> list[DatasetCandidate]:
    """Search Hugging Face dataset metadata and return scored candidates.

    Network errors are deliberately swallowed; callers should inspect an empty
    result and keep the idea blocked rather than inventing a recipe.
    """

    if not DATASET_RESOLVER_ENABLED or not DATASET_RESOLVER_ALLOW_NETWORK:
        return []
    clean_query = normalize_query(query)
    if not clean_query:
        return []
    close_client = client is None
    client = client or httpx.Client(timeout=timeout or DATASET_RESOLVER_TIMEOUT_SECONDS)
    try:
        response = client.get(
            "https://huggingface.co/api/datasets",
            params={"search": clean_query, "limit": limit or DATASET_RESOLVER_HF_LIMIT, "full": "true"},
            headers=_auth_headers(),
        )
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return []
    finally:
        if close_client:
            client.close()
    if not isinstance(payload, list):
        return []
    candidates: list[DatasetCandidate] = []
    for row in payload:
        if not isinstance(row, dict):
            continue
        dataset_id = _text(row.get("id") or row.get("_id"))
        if not dataset_id or "/" not in dataset_id and _canonical(dataset_id) not in COMMON_HF_DATASETS:
            continue
        score = _candidate_score(query, row)
        if score <= 0:
            continue
        candidates.append(
            DatasetCandidate(
                dataset_id=dataset_id,
                score=score,
                downloads=int(row.get("downloads") or 0),
                likes=int(row.get("likes") or 0),
                tags=tuple(str(tag) for tag in row.get("tags") or []),
                card_data=row.get("cardData") if isinstance(row.get("cardData"), dict) else None,
            )
        )
    candidates.sort(key=lambda item: (item.score, item.downloads, item.likes), reverse=True)
    return candidates


def _infer_task_type(name: str, dataset_id: str, tags: tuple[str, ...] = ()) -> str:
    corpus = " ".join([name, dataset_id, *tags]).lower()
    if any(token in corpus for token in ("image", "vision", "cifar", "imagenet")):
        return "image_classification"
    if any(token in corpus for token in ("code", "mbpp", "humaneval", "python")):
        return "code_generation"
    if any(token in corpus for token in ("math", "gsm8k", "aime")):
        return "math_qa"
    if any(token in corpus for token in ("bool", "yes/no", "strategyqa")):
        return "boolean_qa"
    if any(token in corpus for token in ("qa", "question", "mmlu", "arc", "commonsense")):
        return "qa"
    return "qa"


def _recipe_from_candidate(name: str, candidate: DatasetCandidate) -> dict[str, Any]:
    task_type = _infer_task_type(name, candidate.dataset_id, candidate.tags)
    recipe = {
        "name": name,
        "requested_name": name,
        "hf_dataset": candidate.dataset_id,
        "hf_candidates": [candidate.dataset_id],
        "config": "",
        "config_candidates": ["", "main", "default"],
        "split": "test",
        "split_candidates": ["test", "validation", "train"],
        "task_type": task_type,
        "max_eval_examples": EXPERIMENT_REAL_BENCHMARK_MAX_EXAMPLES,
        "generated_runner_supported": True,
        "source": "huggingface_search",
        "resolved": True,
        "confidence": round(candidate.score, 4),
        "url": f"https://huggingface.co/datasets/{candidate.dataset_id}",
        "load_command": f"load_dataset({candidate.dataset_id!r})",
    }
    if task_type == "code_generation":
        recipe.update({"question_field": "text", "answer_field": "code"})
    elif task_type == "image_classification":
        recipe.update({"question_field": "image", "answer_field": "label"})
    else:
        recipe.update({"question_field": "question", "answer_field": "answer"})
    return recipe


def _first_existing_field(features: Any, candidates: list[str]) -> str:
    if not features:
        return ""
    keys = set()
    try:
        keys = set(features.keys())
    except Exception:
        return ""
    for field in candidates:
        if field in keys:
            return field
    return ""


def refine_recipe_with_hf_metadata(recipe: dict[str, Any], *, client: httpx.Client | None = None) -> dict[str, Any]:
    """Probe Hugging Face metadata before the runner downloads examples.

    This is intentionally a metadata pass: it verifies config names, splits,
    and top-level feature fields so generated benchmark runners do not rely on
    guessed recipes such as ``QASC/main`` or ``OpenBookQA/question``.
    """

    out = dict(recipe or {})
    dataset_id = _text(out.get("hf_dataset"))
    if not DATASET_RESOLVER_ENABLED or not DATASET_RESOLVER_ALLOW_NETWORK or not dataset_id:
        return out
    try:
        from datasets import get_dataset_config_names, get_dataset_split_names, load_dataset_builder
    except Exception as exc:
        out.setdefault("metadata_probe", {"status": "skipped", "reason": f"datasets_unavailable:{exc}"})
        return out

    requested_configs = [_text(value) for value in _as_list(out.get("config_candidates"))]
    if _text(out.get("config")) not in requested_configs:
        requested_configs.insert(0, _text(out.get("config")))
    requested_configs = [value for value in requested_configs if value or value == ""]
    try:
        config_names = get_dataset_config_names(dataset_id)
    except Exception as exc:
        out["metadata_probe"] = {"status": "failed", "stage": "configs", "error": str(exc)[:500]}
        return out
    config_names = [_text(value) for value in config_names]
    config_order = list(requested_configs or [""])
    if config_names:
        config_order = [value for value in config_order if value] + [value for value in config_order if not value]
    valid_configs: list[str] = []
    for value in config_order:
        if value == "" or value in config_names:
            valid_configs.append(value)
    if not valid_configs and config_names:
        valid_configs.append(config_names[0])
    if not valid_configs:
        valid_configs = [""]

    selected_config = valid_configs[0]
    requested_splits = [_text(value) for value in _as_list(out.get("split_candidates"))]
    if _text(out.get("split")) not in requested_splits:
        requested_splits.insert(0, _text(out.get("split")))
    requested_splits = [value for value in requested_splits if value]
    try:
        if selected_config:
            split_names = get_dataset_split_names(dataset_id, selected_config)
        else:
            split_names = get_dataset_split_names(dataset_id)
    except Exception as exc:
        out["metadata_probe"] = {
            "status": "failed",
            "stage": "splits",
            "config": selected_config,
            "error": str(exc)[:500],
        }
        return out
    split_names = [_text(value) for value in split_names]
    selected_split = next((value for value in requested_splits if value in split_names), "")
    if not selected_split and split_names:
        selected_split = split_names[0]

    features = None
    try:
        if selected_config:
            builder = load_dataset_builder(dataset_id, selected_config)
        else:
            builder = load_dataset_builder(dataset_id)
        features = getattr(getattr(builder, "info", None), "features", None)
    except Exception as exc:
        out.setdefault("metadata_probe", {})["feature_error"] = str(exc)[:500]

    question_candidates = [
        *_as_list(out.get("question_field_candidates")),
        out.get("question_field"),
        "question",
        "question_stem",
        "formatted_question",
        "text",
        "input",
        "prompt",
    ]
    answer_candidates = [
        *_as_list(out.get("answer_field_candidates")),
        out.get("answer_field"),
        "answer",
        "answerKey",
        "label",
        "target",
        "output",
    ]
    question_field = _first_existing_field(features, [_text(value) for value in question_candidates if _text(value)])
    answer_field = _first_existing_field(features, [_text(value) for value in answer_candidates if _text(value)])
    if question_field:
        out["question_field"] = question_field
    if answer_field:
        out["answer_field"] = answer_field

    out["config"] = selected_config
    out["config_candidates"] = valid_configs
    out["split"] = selected_split or out.get("split") or "test"
    out["split_candidates"] = split_names or requested_splits or [out["split"]]
    out["resolved"] = True
    out["metadata_verified"] = True
    out["metadata_probe"] = {
        "status": "ok",
        "configs": config_names,
        "selected_config": selected_config,
        "splits": split_names,
        "selected_split": out["split"],
        "fields": list(features.keys()) if features is not None and hasattr(features, "keys") else [],
    }
    return out


def resolve_dataset_name(name: str, *, client: httpx.Client | None = None) -> dict[str, Any]:
    """Resolve one named dataset into a conservative executable recipe."""

    clean = _text(name)
    if not clean:
        return {"name": name, "resolved": False, "reason": "empty_dataset_name"}
    lowered = normalize_query(clean)
    if any(marker in lowered for marker in UNRESOLVABLE_MARKERS):
        return {
            "name": clean,
            "resolved": False,
            "reason": "dataset appears restricted or lacks a public Hugging Face benchmark recipe",
        }
    common = _common_recipe(clean)
    if common:
        return refine_recipe_with_hf_metadata(common, client=client)
    candidates = search_huggingface_datasets(clean, client=client)
    if not candidates:
        return {"name": clean, "resolved": False, "reason": "no_huggingface_candidate"}
    best = candidates[0]
    if best.score < DATASET_RESOLVER_MIN_CONFIDENCE:
        return {
            "name": clean,
            "resolved": False,
            "reason": f"best_huggingface_candidate_below_threshold:{best.dataset_id}:{best.score:.3f}",
            "candidates": [candidate.dataset_id for candidate in candidates[:3]],
        }
    return refine_recipe_with_hf_metadata(_recipe_from_candidate(clean, best), client=client)


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if value in (None, "", "unknown"):
        return []
    return [value]


def _dataset_names(plan: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for key in ("benchmark_targets", "datasets"):
        for row in _as_list(plan.get(key)):
            if isinstance(row, dict):
                value = _text(row.get("name") or row.get("dataset") or row.get("hf_dataset"))
            else:
                value = _text(row)
            if value and value.lower() not in {name.lower() for name in names}:
                names.append(value)
    return names


def resolve_plan_datasets(plan: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``plan`` with public dataset recipes attached when possible."""

    if not DATASET_RESOLVER_ENABLED:
        return dict(plan or {})
    out = dict(plan or {})
    targets = [dict(row) for row in _as_list(out.get("benchmark_targets")) if isinstance(row, dict)]
    names = _dataset_names(out)
    existing = {
        _canonical(row.get("name") or row.get("hf_dataset") or row.get("dataset")): row
        for row in targets
        if isinstance(row, dict)
    }
    resolved_rows: list[dict[str, Any]] = []
    unresolved_rows: list[dict[str, Any]] = []
    for name in names:
        key = _canonical(name)
        current = existing.get(key)
        if current and (_text(current.get("hf_dataset")) or current.get("direct_files") or current.get("derive_from_loaded_benchmarks")):
            common = _common_recipe(name) or {}
            row = {**dict(current), **common}
            row = refine_recipe_with_hf_metadata(row)
            row.setdefault("resolved", True)
            resolved_rows.append(row)
            continue
        recipe = resolve_dataset_name(name)
        if recipe.get("resolved"):
            merged = {**(current or {}), **recipe}
            resolved_rows.append(merged)
        else:
            unresolved_rows.append(recipe)
            if current:
                targets.append(current)
    if not names and not targets:
        out["dataset_resolution"] = {"status": "unresolved", "resolved": [], "unresolved": [{"reason": "no_dataset_names"}]}
        return out
    merged_targets: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in [*resolved_rows, *targets]:
        name = _canonical(row.get("name") or row.get("hf_dataset") or row.get("dataset"))
        if not name or name in seen:
            continue
        seen.add(name)
        merged_targets.append(row)
    if resolved_rows:
        out["benchmark_targets"] = merged_targets
        out["datasets"] = [
            {
                "name": row.get("name") or row.get("hf_dataset"),
                "hf_dataset": row.get("hf_dataset"),
                "split": row.get("split"),
                "source": row.get("source"),
            }
            for row in resolved_rows
        ]
    if unresolved_rows and not resolved_rows:
        out["generated_runner_supported"] = False
        out["benchmark_recipe_blockers"] = [
            {"name": row.get("name"), "reason": row.get("reason")}
            for row in unresolved_rows
            if row.get("name")
        ]
        out["deferred_benchmark_targets"] = [row.get("name") for row in unresolved_rows if row.get("name")]
    elif resolved_rows:
        resolved_names = {
            str(row.get(key) or "").lower()
            for row in resolved_rows
            for key in ("name", "hf_dataset", "dataset")
            if row.get(key)
        }
        blockers = [
            row
            for row in _as_list(out.get("benchmark_recipe_blockers"))
            if not isinstance(row, dict) or str(row.get("name") or "").lower() not in resolved_names
        ]
        seen_blockers = {
            str(row.get("name") or "").lower()
            for row in blockers
            if isinstance(row, dict) and row.get("name")
        }
        for row in unresolved_rows:
            name = row.get("name")
            key = str(name or "").lower()
            if name and key not in seen_blockers:
                blockers.append({"name": name, "reason": row.get("reason")})
                seen_blockers.add(key)
        if blockers:
            out["benchmark_recipe_blockers"] = blockers
            out["deferred_benchmark_targets"] = [
                row.get("name") if isinstance(row, dict) else str(row)
                for row in blockers
                if (row.get("name") if isinstance(row, dict) else str(row))
            ]
            out["deferred_benchmark_target_details"] = unresolved_rows
            out["benchmark_harness_deferred"] = True
        else:
            out.pop("benchmark_recipe_blockers", None)
            out.pop("deferred_benchmark_targets", None)
            out.pop("deferred_benchmark_target_details", None)
            out.pop("benchmark_harness_deferred", None)
        out["generated_runner_supported"] = True
    out["dataset_resolution"] = {
        "status": "resolved" if resolved_rows and not unresolved_rows else "partial" if resolved_rows else "unresolved",
        "resolved": [
            {
                "name": row.get("name"),
                "hf_dataset": row.get("hf_dataset"),
                "source": row.get("source"),
                "confidence": row.get("confidence"),
            }
            for row in resolved_rows
        ],
        "unresolved": unresolved_rows,
    }
    return out
