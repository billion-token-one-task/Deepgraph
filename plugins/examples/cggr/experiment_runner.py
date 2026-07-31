"""Historical CGGR/VOC benchmark runner template.

This implementation is preserved for reproducibility but is disabled by
default, non-production, and has no ResourceGrant authority.
"""

from __future__ import annotations

import json
import textwrap

from agents.experiment_forge import _real_benchmark_defaults


def render_historical_benchmark_runner(*, method_name: str, metric_name: str, plan: dict) -> str:
    defaults = _real_benchmark_defaults(plan)
    method_lower = str(method_name or "").lower()
    cggr_mode = "cggr" in method_lower or "counterfactual gain gated" in method_lower
    defaults_payload = {
        "method_name": method_name,
        "candidate_method_name": "CGGR" if cggr_mode else method_name,
        "candidate_kind": "cggr" if cggr_mode else "voc_metareasoning",
        "cggr_mode": cggr_mode,
        "metric_name": metric_name,
        "model_id": defaults["model_id"],
        "model_targets": defaults.get("model_targets") or [],
        "model_requires_cuda": defaults.get("model_requires_cuda", True),
        "model_cpu_allowed": defaults.get("model_cpu_allowed", False),
        "model_load_in_4bit": defaults.get("model_load_in_4bit", True),
        "targets": defaults["targets"],
        "max_examples": defaults["max_examples"],
        "seeds": defaults["seeds"],
        "baselines": defaults["baselines"],
        "ablations": defaults["ablations"] if cggr_mode else [],
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
    CANDIDATE_METHOD = DEFAULTS.get("candidate_method_name") or METHOD_NAME
    CANDIDATE_KIND = DEFAULTS.get("candidate_kind") or "voc_metareasoning"
    CGGR_MODE = bool(DEFAULTS.get("cggr_mode"))
    METRIC_NAME = DEFAULTS["metric_name"]
    DEFAULT_MODEL_ID = DEFAULTS["model_id"]
    DEFAULT_MODEL_REQUIRES_CUDA = bool(DEFAULTS.get("model_requires_cuda", True))
    DEFAULT_MODEL_CPU_ALLOWED = bool(DEFAULTS.get("model_cpu_allowed", False))
    DEFAULT_MODEL_LOAD_IN_4BIT = bool(DEFAULTS.get("model_load_in_4bit", True))
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
        CANDIDATE_METHOD,
    )

    BASE_METHOD_SPECS = collections.OrderedDict([
        ("Vanilla Direct Answering", {"kind": "direct", "max_new_tokens": 48}),
        ("Always-Reason Chain-of-Thought", {"kind": "fixed_cot", "max_new_tokens": 192}),
        ("Self-Consistency Reasoning", {"kind": "self_consistency", "max_new_tokens": 192}),
        ("Least-to-Most Prompting", {"kind": "least_to_most", "max_new_tokens": 192}),
        ("Confidence Gate", {"kind": "confidence_gate", "max_new_tokens": 192}),
        ("Disagreement Routing", {"kind": "disagreement_gate", "max_new_tokens": 192}),
        ("Random Budget-Matched Routing", {"kind": "random_budget_matched", "max_new_tokens": 192}),
    ])
    METHOD_SPECS = collections.OrderedDict(BASE_METHOD_SPECS)
    METHOD_SPECS[CANDIDATE_METHOD] = {"kind": CANDIDATE_KIND, "max_new_tokens": 192}
    TOP_VENUE_BASELINE_SPECS = collections.OrderedDict([
        ("CAR-Style Certainty Adaptive Routing", {"kind": "car_certainty_gate", "max_new_tokens": 192}),
        ("Self-Route-Style Mode Routing", {"kind": "self_route_mode", "max_new_tokens": 192}),
        ("Rational-Metareasoning VOC Routing", {"kind": "voc_metareasoning", "max_new_tokens": 192}),
    ])
    ABLATION_SPECS = collections.OrderedDict()
    if CGGR_MODE:
        ABLATION_SPECS.update([
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
        "CAR-Style Certainty Adaptive Routing": ["car", "car_style", "certainty_adaptive_routing", "certainty_routing"],
        "Self-Route-Style Mode Routing": ["self_route", "self_route_style", "mode_routing"],
        "Rational-Metareasoning VOC Routing": ["voc", "value_of_computation", "rational_metareasoning"],
        "CGGR/oracle_router": ["oracle", "oracle_router", "upper_bound"],
    }
    BASELINE_ALIASES[CANDIDATE_METHOD] = ["candidate", "proposed_method", "method_under_test"]
    if CGGR_MODE:
        BASELINE_ALIASES["CGGR"] = ["cggr", "candidate", "proposed_method"]


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
        requested = os.getenv("DEEPGRAPH_BENCHMARK_METHODS", "").strip()
        requested_names = [item.strip() for item in requested.split(",") if item.strip()] if requested else []
        include_top_venue = _env_flag("DEEPGRAPH_BENCHMARK_INCLUDE_TOP_VENUE_BASELINES") or any(
            name in TOP_VENUE_BASELINE_SPECS for name in requested_names
        )
        if include_top_venue:
            method_specs.update(TOP_VENUE_BASELINE_SPECS)
        for name, spec in ABLATION_SPECS.items():
            if name in DEFAULTS.get("ablations", []):
                method_specs["CGGR/" + name] = spec
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
            key = name if name in method_specs else "CGGR/" + name if CGGR_MODE and "CGGR/" + name in method_specs else None
            if key:
                selected[key] = method_specs[key]
            else:
                missing.append(name)
        if not explicit_subset and CANDIDATE_METHOD not in selected and CANDIDATE_METHOD in method_specs:
            selected[CANDIDATE_METHOD] = method_specs[CANDIDATE_METHOD]
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


    def _choice_map(row):
        raw = None
        if isinstance(row, dict):
            raw = row.get("choices") or row.get("options") or row.get("candidates")
        labels = []
        texts = []
        if isinstance(raw, dict):
            labels = raw.get("label") or raw.get("labels") or raw.get("keys") or []
            texts = raw.get("text") or raw.get("texts") or raw.get("choices") or raw.get("options") or []
        elif isinstance(raw, list):
            for item in raw:
                if isinstance(item, dict):
                    labels.append(item.get("label") or item.get("key") or item.get("id") or "")
                    texts.append(item.get("text") or item.get("answer") or item.get("value") or item.get("option") or "")
                else:
                    texts.append(str(item))
        if isinstance(labels, str):
            labels = list(labels)
        if isinstance(texts, str):
            texts = [texts]
        labels = [str(label or "").strip().upper() for label in labels]
        texts = [str(item or "").strip() for item in texts]
        if texts and (not labels or len(labels) != len(texts)):
            labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"[: len(texts)])
        out = {}
        for label, value in zip(labels, texts):
            if label and value:
                out[label] = value
        return out


    def _question_with_choices(question, choices):
        if not choices:
            return question
        lines = [f"{label}. {value}" for label, value in choices.items()]
        return question + "\\nChoices:\\n" + "\\n".join(lines)


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
            choices = _choice_map(row)
            question_for_prompt = _question_with_choices(question, choices)
            if not question_for_prompt or not answer:
                continue
            example_id = str(row.get("id") or row.get("qid") or row.get("question_id") or idx)
            examples.append({
                "example_id": example_id,
                "question": question_for_prompt,
                "answer": answer,
                "choices": choices,
                "dataset_name": meta.get("name") or target.get("name"),
                "dataset_id": meta.get("id"),
                "dataset_config": meta.get("config"),
                "split": meta.get("split"),
                "task_type": target.get("task_type") or "qa",
                "difficulty": _difficulty_proxy(question_for_prompt, target.get("task_type") or "qa"),
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
        if not suites:
            failures.append({
                "target": "all_requested_benchmarks",
                "error": "no requested benchmark suite loaded; refusing cross-domain GSM8K fallback",
            })
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


    def _extract_choice_label(final, choices):
        if not choices:
            return ""
        raw = str(final or "").strip()
        labels = set(str(label).upper() for label in choices)
        compact = re.sub(r"[^A-Za-z]", "", raw).upper()
        if len(compact) == 1 and compact in labels:
            return compact
        for pattern in (
            r"(?i)\\b(?:option|choice|answer|final answer)\\s*[:#\\-]?\\s*([A-Z])\\b",
            r"(?m)^\\s*([A-Z])\\s*[\\).:-]",
        ):
            match = re.search(pattern, raw)
            if match:
                label = match.group(1).upper()
                if label in labels:
                    return label
        return ""


    def _choice_discussion_labels(text, choices):
        labels = []
        raw = str(text or "")
        for label in choices or {}:
            pattern = r"(?i)\\b(?:choice|option)\\s*" + re.escape(str(label)) + r"\\b"
            if re.search(pattern, raw):
                labels.append(str(label).upper())
        return set(labels)


    def _score_answer(prediction, gold, task_type, choices=None):
        choices = choices or {}
        raw_prediction = str(prediction or "")
        lowered_prediction = raw_prediction.lower()
        has_answer_marker = any(marker in lowered_prediction for marker in ("final answer:", "answer:"))
        final = _extract_final_answer(prediction)
        unmarked_reasoning_cue = bool(re.search(r"(?i)\\b(step\\s*\\d+|firstly|secondly|analy[sz]e|evaluate|therefore|because)\\b", final))
        multi_choice_discussion = len(_choice_discussion_labels(final, choices)) > 1
        short_unmarked_answer = (
            (not has_answer_marker)
            and final.count("\\n") <= 1
            and len(final.split()) <= 24
            and not unmarked_reasoning_cue
            and not multi_choice_discussion
        )
        allow_choice_parse = has_answer_marker or short_unmarked_answer
        gold_text = _answer_to_text(gold)
        gold_label = str(gold_text or "").strip().upper()
        gold_choice_text = choices.get(gold_label, "") if isinstance(choices, dict) else ""
        pred_label = _extract_choice_label(final, choices) if allow_choice_parse else ""
        pred_norm = _normalize_text(final)
        gold_norm = _normalize_text(gold_text)
        gold_choice_norm = _normalize_text(gold_choice_text)
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
        if choices and gold_label:
            if pred_label == gold_label:
                exact = 1.0
            elif gold_choice_norm and allow_choice_parse and (pred_norm == gold_choice_norm or gold_choice_norm in pred_norm):
                exact = 1.0
        if bool_gold:
            exact = 1.0 if bool_pred in {gold_norm, "yes" if gold_norm == "true" else "no" if gold_norm == "false" else gold_norm} else 0.0
        f1 = max(exact, _token_f1(final, gold_text), _token_f1(final, gold_choice_text) if gold_choice_text else 0.0)
        primary = exact if choices or task_type in {"math_qa", "boolean_qa"} else f1
        return {
            "exact": float(exact),
            "f1": float(f1),
            "primary_score": float(primary),
            "prediction_answer": final,
            "prediction_label": pred_label,
            "gold_answer": gold_text,
            "gold_choice_text": gold_choice_text,
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
        if kind == "voc_metareasoning":
            return (
                "Use the residual decision packet only as a private check. "
                "Select exactly one answer option. "
                "Output one line only: \"Final answer: <option label or concise answer>\". "
                "Do not include reasoning, alternatives, multiple labels, uncertainty text, or any text after the final answer."
                "\\nQuestion: " + question + f"\\nDifficulty proxy: {difficulty:.3f}\\nAnswer:"
            )
        if kind.startswith("cggr") or kind in {
            "confidence_gate",
            "disagreement_gate",
            "random_budget_matched",
            "car_certainty_gate",
            "self_route_mode",
        }:
            return (
                "Choose the smallest sufficient response. If the answer is clear, do not reason. "
                "If deliberation is useful, use at most two concise reasoning sentences. "
                "Use deliberate reasoning only when the question structure, counterfactual risk, or uncertainty justifies it. "
                "End with exactly one line: 'Final answer: <option label or concise answer>'. Do not repeat the final answer or add text after it."
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
        if kind == "car_certainty_gate":
            return 192 if difficulty >= 0.50 else 56
        if kind == "self_route_mode":
            return 192 if difficulty >= 0.46 else 56
        if kind == "voc_metareasoning":
            return 64 if difficulty >= 0.44 else 48
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
        use_cuda = bool(torch.cuda.is_available())
        if DEFAULT_MODEL_REQUIRES_CUDA and not use_cuda:
            raise RuntimeError("Real LLM benchmark requires CUDA. No synthetic or mocked fallback is allowed.")
        if not use_cuda and not DEFAULT_MODEL_CPU_ALLOWED:
            raise RuntimeError("Selected real benchmark model is not marked CPU-compatible.")
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
        if use_cuda:
            load_kwargs = {"torch_dtype": torch.float16, "device_map": "auto", "trust_remote_code": True}
        else:
            load_kwargs = {"torch_dtype": torch.float32, "trust_remote_code": True}
        load_in_4bit_default = "1" if DEFAULT_MODEL_LOAD_IN_4BIT and use_cuda else "0"
        if os.getenv("DEEPGRAPH_BENCHMARK_LOAD_IN_4BIT", load_in_4bit_default).strip().lower() in {"1", "true", "yes", "on"}:
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
        if use_cuda:
            device_map = getattr(model, "hf_device_map", {}) or {}
            offloaded = {
                name: str(device).lower()
                for name, device in device_map.items()
                if str(device).lower() in {"cpu", "disk"}
            }
            if offloaded and not _env_flag("DEEPGRAPH_ALLOW_CPU_OFFLOAD"):
                sample = dict(list(offloaded.items())[:5])
                raise RuntimeError(
                    "Real GPU benchmark disallows CPU/disk offload; free GPU memory or set "
                    "DEEPGRAPH_ALLOW_CPU_OFFLOAD=1 for non-paper diagnostics. Offloaded modules: "
                    + json.dumps(sample, ensure_ascii=False)
                )
        if not use_cuda:
            model.to("cpu")
        model.eval()
        print("BENCHMARK_STAGE: model_ready " + str(model_id), flush=True)
        return model, tokenizer, model_id


    def _sample_examples(examples, seed, max_examples):
        rng = random.Random(seed)
        if max_examples <= 0 or len(examples) <= max_examples:
            return list(examples)
        indices = rng.sample(range(len(examples)), max_examples)
        return [examples[i] for i in indices]


    def _certainty_proxy(question, direct_output, difficulty):
        answer = _extract_final_answer(direct_output)
        norm = _normalize_text(answer)
        uncertainty_markers = r"\\b(maybe|unknown|unclear|unsure|cannot determine|not enough|it depends)\\b"
        confidence = 1.0 - 0.55 * float(difficulty)
        if norm and len(norm.split()) <= 8:
            confidence += 0.12
        if re.search(uncertainty_markers, norm):
            confidence -= 0.35
        if not norm:
            confidence -= 0.40
        if re.search(r"\\b(yes|no|true|false)\\b", norm):
            confidence += 0.04
        return max(0.0, min(1.0, float(confidence)))


    def _question_structure_signal(question, difficulty):
        text = str(question or "")
        lowered = text.lower()
        multihop_terms = len(re.findall(r"\\b(before|after|because|which|who|where|when|compare|between|except|unless)\\b", lowered))
        numbers = len(re.findall(r"\\d+", text))
        signal = float(difficulty) + 0.035 * multihop_terms + 0.025 * numbers
        return max(0.0, min(1.0, signal))


    def _route_with_strategy(model, tokenizer, example, strategy, *, difficulty, max_new_tokens):
        return _generate(
            model,
            tokenizer,
            _build_prompt(example["question"], strategy, difficulty=difficulty),
            max_new_tokens=max_new_tokens,
        )


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
                "max_new_tokens": 192 if deliberate else 56,
                "routed_to_deliberation": deliberate,
                "random_budget_matched": True,
            }
        if kind == "car_certainty_gate":
            direct_prompt = _build_prompt(example["question"], "direct", difficulty=difficulty)
            direct_output, direct_tokens = _generate(model, tokenizer, direct_prompt, max_new_tokens=48)
            certainty = _certainty_proxy(example["question"], direct_output, difficulty)
            threshold = float(os.getenv("DEEPGRAPH_CAR_CERTAINTY_THRESHOLD", "0.58"))
            deliberate = certainty < threshold
            if deliberate:
                cot_output, cot_tokens = _route_with_strategy(
                    model,
                    tokenizer,
                    example,
                    "fixed_cot",
                    difficulty=difficulty,
                    max_new_tokens=192,
                )
                return cot_output, direct_tokens + cot_tokens, {
                    "difficulty": difficulty,
                    "kind": kind,
                    "max_new_tokens": 192,
                    "routed_to_deliberation": True,
                    "certainty_proxy": certainty,
                    "certainty_threshold": threshold,
                    "short_answer": _extract_final_answer(direct_output),
                }
            return direct_output, direct_tokens, {
                "difficulty": difficulty,
                "kind": kind,
                "max_new_tokens": 48,
                "routed_to_deliberation": False,
                "certainty_proxy": certainty,
                "certainty_threshold": threshold,
                "short_answer": _extract_final_answer(direct_output),
            }
        if kind == "self_route_mode":
            signal = _question_structure_signal(example["question"], difficulty)
            threshold = float(os.getenv("DEEPGRAPH_SELF_ROUTE_THRESHOLD", "0.46"))
            deliberate = signal >= threshold
            strategy = "fixed_cot" if deliberate else "direct"
            output, tokens = _route_with_strategy(
                model,
                tokenizer,
                example,
                strategy,
                difficulty=difficulty,
                max_new_tokens=192 if deliberate else 56,
            )
            return output, tokens, {
                "difficulty": difficulty,
                "kind": kind,
                "max_new_tokens": 192 if deliberate else 56,
                "routed_to_deliberation": deliberate,
                "routing_signal": signal,
                "routing_threshold": threshold,
                "route_before_answer": True,
            }
        if kind == "voc_metareasoning":
            structure_signal = _question_structure_signal(example["question"], difficulty)
            reasoning_cost = float(os.getenv("DEEPGRAPH_VOC_REASONING_COST", "0.11"))
            simple_case_penalty = 0.10 if difficulty < 0.30 else 0.0
            expected_value = 0.52 * structure_signal - reasoning_cost - simple_case_penalty
            threshold = float(os.getenv("DEEPGRAPH_VOC_THRESHOLD", "0.28"))
            deliberate = expected_value >= threshold
            strategy = "voc_metareasoning" if deliberate else "direct"
            max_new_tokens = _env_int("DEEPGRAPH_VOC_DELIBERATE_MAX_NEW_TOKENS", 64) if deliberate else 48
            output, tokens = _route_with_strategy(
                model,
                tokenizer,
                example,
                strategy,
                difficulty=difficulty,
                max_new_tokens=max_new_tokens,
            )
            return output, tokens, {
                "difficulty": difficulty,
                "kind": kind,
                "max_new_tokens": max_new_tokens,
                "routed_to_deliberation": deliberate,
                "structure_signal": structure_signal,
                "expected_value_of_computation": expected_value,
                "routing_threshold": threshold,
                "reasoning_cost": reasoning_cost,
                "route_before_answer": True,
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
                    "max_new_tokens": 192,
                    "routed_to_deliberation": True,
                    "short_answer_a": _extract_final_answer(out_a),
                    "short_answer_b": _extract_final_answer(out_b),
                }
            return out_a, tok_a + tok_b, {
                "difficulty": difficulty,
                "kind": kind,
                "max_new_tokens": 48,
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
            "prompt_template": "method-specific direct, chain-of-thought, gating, disagreement, candidate-routing, and optional CGGR prompts in _build_prompt",
            "decoding": {"default": "greedy", "self_consistency_extra_samples": "temperature=0.7, top_p=0.95"},
            "reasoning_budget": {"direct": 48, "short_gate": 56, "cot": 192, "candidate": "adaptive 56-224 max_new_tokens by difficulty", "top_venue_baselines": "48-token short branch plus optional 192-token deliberation"},
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
                            score = _score_answer(prediction, ex["answer"], ex.get("task_type") or "", ex.get("choices"))
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
                            score = {"exact": 0.0, "f1": 0.0, "primary_score": 0.0, "prediction_answer": "", "prediction_label": "", "gold_answer": ex["answer"], "gold_choice_text": ""}
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
                        if (
                            spec["kind"].startswith("cggr")
                            or spec["kind"]
                            in {
                                "confidence_gate",
                                "disagreement_gate",
                                "random_budget_matched",
                                "car_certainty_gate",
                                "self_route_mode",
                                "voc_metareasoning",
                            }
                        ):
                            _append_jsonl("routing_decisions.jsonl", {
                                "seed": seed,
                                "dataset": dataset_name,
                                "method": method_name,
                                "example_id": ex.get("example_id"),
                                **route,
                            })
                        if method_name == CANDIDATE_METHOD and score["primary_score"] < 0.5:
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
        oracle_method_name = CANDIDATE_METHOD + "/oracle_router"
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
            per_method[oracle_method_name] = {
                "cost_adjusted_accuracy": _mean(oracle_values),
                "metric_value": _mean(oracle_values),
                "score": _mean(oracle_values),
                "avg_new_tokens": 0.0,
                "avg_latency_seconds": 0.0,
                "route_rate": 1.0,
                "count": sum(int(row.get("num_examples") or 0) for seed_row in seed_results for row in seed_row.get("datasets", {}).values()),
                "upper_bound": True,
            }
            per_method_std[oracle_method_name] = _std(oracle_values)

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
        candidate_values = per_seed_method_values.get(CANDIDATE_METHOD, [])
        baseline_name = "Always-Reason Chain-of-Thought" if "Always-Reason Chain-of-Thought" in per_seed_method_values else "Vanilla Direct Answering"
        baseline_values = per_seed_method_values.get(baseline_name, [])
        bootstrap = {
            "candidate_method": CANDIDATE_METHOD,
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
                    "delta_vs_candidate": per_method[key]["metric_value"] - per_method.get(CANDIDATE_METHOD, {}).get("metric_value", 0.0),
                })
        if oracle_method_name in per_method:
            ablation_table.append({
                "ablation": "oracle_router",
                "method": oracle_method_name,
                "metric_value": per_method[oracle_method_name]["metric_value"],
                "delta_vs_candidate": per_method[oracle_method_name]["metric_value"] - per_method.get(CANDIDATE_METHOD, {}).get("metric_value", 0.0),
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
        quality_cost_frontier = sorted(
            [
                {
                    "method": row["method"],
                    "quality": row["accuracy"],
                    "utility": row["metric_value"],
                    "avg_new_tokens": row["avg_new_tokens"],
                    "avg_latency_seconds": row["avg_latency_seconds"],
                    "route_rate": row["route_rate"],
                }
                for row in latency_tokens_table
            ],
            key=lambda row: (row["avg_new_tokens"], -row["quality"]),
        )
        route_rate_sweep = [
            {
                "method": row["method"],
                "route_rate": row["route_rate"],
                "quality": row["accuracy"],
                "utility": row["metric_value"],
                "avg_new_tokens": row["avg_new_tokens"],
                "avg_latency_seconds": row["avg_latency_seconds"],
            }
            for row in sorted(latency_tokens_table, key=lambda item: item["route_rate"])
            if row["method"] == CANDIDATE_METHOD
            or any(token in row["method"].lower() for token in ("gate", "routing", "route", "budget", "random", "chain-of-thought"))
        ]

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
        candidate_easy = next((row for row in difficulty_breakdown_table if row["method"] == CANDIDATE_METHOD and row["difficulty"] == "easy"), {})
        simple_case_degradation = {
            "subset": "easy",
            "baseline_method": "Vanilla Direct Answering",
            "candidate_method": CANDIDATE_METHOD,
            "baseline_accuracy": direct_easy.get("accuracy"),
            "candidate_accuracy": candidate_easy.get("accuracy"),
            "degradation": (
                float(candidate_easy.get("accuracy", 0.0) - direct_easy.get("accuracy", 0.0))
                if direct_easy and candidate_easy
                else None
            ),
            "candidate_route_rate": candidate_easy.get("route_rate"),
        }
        calibration_reliability = []
        for bucket_name, proxy_value in (("easy", 0.17), ("medium", 0.50), ("hard", 0.83)):
            direct_row = next((row for row in difficulty_breakdown_table if row["method"] == "Vanilla Direct Answering" and row["difficulty"] == bucket_name), {})
            candidate_row = next((row for row in difficulty_breakdown_table if row["method"] == CANDIDATE_METHOD and row["difficulty"] == bucket_name), {})
            if direct_row and candidate_row:
                calibration_reliability.append({
                    "difficulty_bucket": bucket_name,
                    "difficulty_proxy": proxy_value,
                    "observed_gain_vs_direct": float(candidate_row.get("accuracy", 0.0) - direct_row.get("accuracy", 0.0)),
                    "route_rate": candidate_row.get("route_rate"),
                    "count": candidate_row.get("count"),
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
                if row["method"] == CANDIDATE_METHOD
                or any(token in row["method"].lower() for token in ("gate", "routing", "cggr", "oracle"))
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
                "prompt_template": "method-specific direct, chain-of-thought, gating, disagreement, candidate-routing, and optional CGGR prompts in _build_prompt",
                "decoding": {"default": "greedy", "self_consistency_extra_samples": "temperature=0.7, top_p=0.95"},
                "reasoning_budget": {"direct": 48, "short_gate": 56, "cot": 192, "candidate": "adaptive 56-224 max_new_tokens by difficulty", "top_venue_baselines": "48-token short branch plus optional 192-token deliberation"},
            }),
            "per_seed_results": _write_json("per_seed_results.json", seed_results),
            "per_dataset_results": _write_json("per_dataset_results.json", per_dataset_results),
            "main_results_table": _write_json("main_results_table.json", per_method),
            "cost_utility_tradeoff_table": _write_json("cost_utility_tradeoff_table.json", cost_utility_tradeoff_table),
            "quality_cost_frontier": _write_json("quality_cost_frontier.json", quality_cost_frontier),
            "route_rate_sweep": _write_json("route_rate_sweep_table.json", route_rate_sweep),
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
            "candidate_method": CANDIDATE_METHOD,
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
            "quality_cost_frontier": quality_cost_frontier,
            "route_rate_sweep": route_rate_sweep,
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
            METRIC_NAME: per_method.get(CANDIDATE_METHOD, {}).get("metric_value", 0.0),
        }
        result["model"] = {
            **result["model"],
            "hardware": result["hardware"],
            "prompt_template": "method-specific direct, chain-of-thought, gating, disagreement, candidate-routing, and optional CGGR prompts in _build_prompt",
            "decoding": {"default": "greedy", "self_consistency_extra_samples": "temperature=0.7, top_p=0.95"},
            "reasoning_budget": {"direct": 48, "short_gate": 56, "cot": 192, "candidate": "adaptive 56-224 max_new_tokens by difficulty", "top_venue_baselines": "48-token short branch plus optional 192-token deliberation"},
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
        print(f"{METRIC_NAME}: {per_method.get(CANDIDATE_METHOD, {}).get('metric_value', 0.0):.6f}")
        print("FINAL_RESULTS: " + json.dumps(result, ensure_ascii=False))


    if __name__ == "__main__":
        main()
    """).replace("__DEEPGRAPH_DEFAULTS_JSON__", defaults_json)
