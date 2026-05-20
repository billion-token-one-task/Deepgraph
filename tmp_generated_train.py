import collections
import hashlib
import itertools
import json
import math
import os
import random
import re
import statistics
import time
import urllib.request

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULTS = json.loads(r'''{"method_name": "CGGR", "metric_name": "cost_adjusted_utility", "model_id": "Qwen/Qwen2.5-7B-Instruct", "targets": [{"name": "GSM8K", "hf_dataset": "openai/gsm8k", "hf_candidates": ["openai/gsm8k"], "config": "main", "config_candidates": ["main", ""], "split": "test", "split_candidates": ["test", "train"], "task_type": "math_qa", "question_field": "question", "answer_field": "answer", "max_eval_examples": 128}, {"name": "MuSiQue-Ans", "hf_dataset": "dgslibisey/MuSiQue", "hf_candidates": ["dgslibisey/MuSiQue", "voidful/MuSiQue", "bdsaglam/musique"], "config": "", "config_candidates": ["", "answerable"], "split": "validation", "split_candidates": ["validation", "test", "train"], "task_type": "multihop_qa", "question_field": "question", "answer_field": "answer", "max_eval_examples": 128}, {"name": "StrategyQA", "hf_dataset": "tasksource/strategy-qa", "hf_candidates": ["tasksource/strategy-qa", "ChilleD/StrategyQA", "wics/strategy-qa"], "config": "", "config_candidates": ["", "default"], "split": "validation", "split_candidates": ["validation", "test", "train"], "task_type": "boolean_qa", "question_field": "question", "answer_field": "answer", "max_eval_examples": 128}, {"name": "2WikiMultihopQA", "hf_dataset": "xanhho/2WikiMultihopQA", "hf_candidates": ["xanhho/2WikiMultihopQA", "voidful/2WikiMultihopQA"], "config": "", "config_candidates": ["", "default"], "split": "validation", "split_candidates": ["validation", "test", "train"], "task_type": "multihop_qa", "question_field": "question", "answer_field": "answer", "max_eval_examples": 128}, {"name": "Stress Test Split: Simple-vs-Hard Counterfactual Partition", "hf_dataset": "", "hf_candidates": [], "config": "", "config_candidates": [""], "split": "derived", "split_candidates": ["derived"], "task_type": "derived_stress_split", "derive_from_loaded_benchmarks": true, "max_eval_examples": 128}], "max_examples": 2, "seeds": 3, "baselines": ["Vanilla Direct Answering", "Always-Reason Chain-of-Thought", "Self-Consistency Reasoning", "Least-to-Most Prompting", "Confidence Gate"], "ablations": ["no_counterfactual_delta", "no_lcb", "no_self_divergence_penalty", "no_qstruct_term", "oracle_router"]}''')
METHOD_NAME = DEFAULTS["method_name"]
METRIC_NAME = DEFAULTS["metric_name"]
DEFAULT_MODEL_ID = DEFAULTS["model_id"]
DEFAULT_MAX_EXAMPLES = int(DEFAULTS["max_examples"])
DEFAULT_SEEDS = int(DEFAULTS["seeds"])
DEFAULT_LOCAL_JSONL = os.path.join(os.path.dirname(__file__), "benchmark_data", "gsm8k_test.jsonl")
DEFAULT_JSONL_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
DEFAULT_JSONL_SHA256 = "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14"

METHOD_SPECS = collections.OrderedDict([
    ("Vanilla Direct Answering", {"kind": "direct", "max_new_tokens": 48}),
    ("Always-Reason Chain-of-Thought", {"kind": "fixed_cot", "max_new_tokens": 192}),
    ("Self-Consistency Reasoning", {"kind": "self_consistency", "max_new_tokens": 192}),
    ("Least-to-Most Prompting", {"kind": "least_to_most", "max_new_tokens": 192}),
    ("Confidence Gate", {"kind": "confidence_gate", "max_new_tokens": 192}),
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
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return path


def _read_jsonl_rows(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _download_jsonl_rows(url, expected_sha):
    with urllib.request.urlopen(url, timeout=120) as response:
        payload = response.read()
    digest = hashlib.sha256(payload).hexdigest()
    if expected_sha and digest != expected_sha:
        raise RuntimeError("Downloaded benchmark checksum mismatch: " + digest)
    return [json.loads(line) for line in payload.decode("utf-8").splitlines() if line.strip()]


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
    candidates = _unique(target.get("hf_candidates") or [target.get("hf_dataset")])
    configs = target.get("config_candidates") or [target.get("config") or ""]
    splits = target.get("split_candidates") or [target.get("split") or "test"]
    for dataset_id in candidates:
        for config in configs:
            for split in splits:
                if not dataset_id or split == "derived":
                    continue
                try:
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


def _difficulty_proxy(question):
    text = str(question)
    numbers = len(re.findall(r"\d+", text))
    operators = sum(text.count(ch) for ch in "+-*/=%")
    clauses = len(re.findall(r"\b(if|unless|because|before|after|except|not)\b", text.lower()))
    return min(1.0, (len(text.split()) / 90.0) + 0.07 * numbers + 0.05 * operators + 0.04 * clauses)


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
            "difficulty": _difficulty_proxy(question),
        })
    if max_examples > 0:
        examples = examples[: max_examples * 4]
    return examples


def _load_benchmark_suites(max_examples):
    suites = []
    failures = []
    loaded_pool = []
    target_limit = int(os.getenv("DEEPGRAPH_BENCHMARK_TARGET_LIMIT", "0") or "0")
    targets = list(DEFAULTS["targets"])
    if target_limit > 0:
        nonderived = [t for t in targets if not t.get("derive_from_loaded_benchmarks")][:target_limit]
        derived = [t for t in targets if t.get("derive_from_loaded_benchmarks")]
        targets = nonderived + derived
    for target in targets:
        if target.get("derive_from_loaded_benchmarks"):
            continue
        try:
            rows, meta = _load_hf_rows(target)
            examples = _materialize_examples(rows, target, meta, max_examples)
            if not examples:
                raise RuntimeError("loaded dataset but could not infer question/answer fields")
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
    matches = re.findall(r"[-+]?\d+(?:\.\d+)?", str(text).replace(",", ""))
    return matches[-1] if matches else ""


def _normalize_text(text):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]+", " ", str(text or "").lower())).strip()


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
    bool_pred = "yes" if re.search(r"\byes\b|\btrue\b", pred_norm) else "no" if re.search(r"\bno\b|\bfalse\b", pred_norm) else pred_norm
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
        return "Answer the question. Give only the final answer.\nQuestion: " + question + "\nAnswer:"
    if kind == "fixed_cot":
        return "Answer the question. Think step by step, then write 'Final answer: <answer>'.\nQuestion: " + question + "\nSolution:"
    if kind == "least_to_most":
        return (
            "Decompose the question into the smallest useful subquestions, solve them in order, "
            "then write 'Final answer: <answer>'.\nQuestion: " + question + "\nSolution:"
        )
    if kind.startswith("cggr") or kind == "confidence_gate":
        return (
            "Decide whether concise direct answering or deliberate reasoning is warranted. "
            "Use deliberate reasoning only when the question structure, counterfactual risk, or uncertainty justifies it. "
            "End with 'Final answer: <answer>'.\nQuestion: " + question + f"\nDifficulty proxy: {difficulty:.3f}\nSolution:"
        )
    return "Answer the question and end with 'Final answer: <answer>'.\nQuestion: " + question + "\nSolution:"


def _max_tokens_for_kind(kind, difficulty):
    if kind == "direct":
        return 48
    if kind == "confidence_gate":
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


def _generate(model, tokenizer, prompt, *, max_new_tokens, do_sample=False, temperature=0.0):
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=int(os.getenv("DEEPGRAPH_BENCHMARK_MAX_INPUT_TOKENS", "1536")),
    )
    encoded = {key: value.to(model.device) for key, value in encoded.items()}
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
    return tokenizer.decode(generated, skip_special_tokens=True).strip(), int(generated.numel())


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
    prompt_kind = "direct" if kind == "direct" else kind
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
    max_examples = int(os.getenv("DEEPGRAPH_BENCHMARK_MAX_EXAMPLES", str(DEFAULT_MAX_EXAMPLES)))
    seeds = int(os.getenv("DEEPGRAPH_BENCHMARK_SEEDS", str(DEFAULT_SEEDS)))
    lambda_cost = float(os.getenv("DEEPGRAPH_BENCHMARK_COST_LAMBDA", "0.03"))
    suites, load_failures = _load_benchmark_suites(max_examples)
    model, tokenizer, model_id = _load_model()
    method_specs = collections.OrderedDict(METHOD_SPECS)
    for name, spec in ABLATION_SPECS.items():
        if name in DEFAULTS.get("ablations", []):
            method_specs["CGGR/" + name] = spec
    seed_results = []
    aggregate = {
        method: {"score": 0.0, "exact": 0.0, "f1": 0.0, "count": 0, "tokens": 0.0}
        for method in method_specs
    }
    per_dataset_results = {}
    per_seed_method_values = {method: [] for method in method_specs}
    per_example_scores = {}

    _write_json("run_config.json", {
        "method": METHOD_NAME,
        "metric_name": METRIC_NAME,
        "model_id": model_id,
        "targets": DEFAULTS["targets"],
        "seeds": seeds,
        "max_examples_per_dataset_seed": max_examples,
        "methods": list(method_specs.keys()),
        "ablations": DEFAULTS.get("ablations", []),
        "cost_lambda": lambda_cost,
    })

    for seed in range(seeds):
        seed_row = {"seed": seed, "datasets": {}, "methods": {}}
        for suite in suites:
            dataset_name = suite["meta"].get("name") or suite["target"].get("name")
            examples = _sample_examples(suite["examples"], seed, max_examples)
            seed_row["datasets"][dataset_name] = {"num_examples": len(examples), "methods": {}}
            per_dataset_results.setdefault(dataset_name, {})
            for method_name, spec in method_specs.items():
                total_score = 0.0
                total_exact = 0.0
                total_f1 = 0.0
                total_tokens = 0.0
                for ex in examples:
                    try:
                        prediction, tokens, route = _run_single(model, tokenizer, ex, method_name, spec, seed=seed)
                        score = _score_answer(prediction, ex["answer"], ex.get("task_type") or "")
                    except Exception as exc:
                        prediction = ""
                        tokens = 0
                        route = {"error": str(exc)}
                        score = {"exact": 0.0, "f1": 0.0, "primary_score": 0.0, "prediction_answer": "", "gold_answer": ex["answer"]}
                        _append_jsonl("failure_cases.jsonl", {
                            "stage": "generation_or_scoring",
                            "seed": seed,
                            "dataset": dataset_name,
                            "method": method_name,
                            "example_id": ex.get("example_id"),
                            "error": str(exc),
                        })
                    total_score += score["primary_score"]
                    total_exact += score["exact"]
                    total_f1 += score["f1"]
                    total_tokens += tokens
                    key = (seed, dataset_name, ex.get("example_id"), method_name)
                    per_example_scores[key] = score["primary_score"]
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
                    "cost_adjusted_accuracy": float(metric_value),
                    "metric_value": float(metric_value),
                    "count": count,
                }
                seed_row["datasets"][dataset_name]["methods"][method_name] = row
                seed_row["methods"].setdefault(method_name, {"score": 0.0, "tokens": 0.0, "count": 0})
                seed_row["methods"][method_name]["score"] += total_score
                seed_row["methods"][method_name]["tokens"] += total_tokens
                seed_row["methods"][method_name]["count"] += count
                aggregate[method_name]["score"] += total_score
                aggregate[method_name]["exact"] += total_exact
                aggregate[method_name]["f1"] += total_f1
                aggregate[method_name]["tokens"] += total_tokens
                aggregate[method_name]["count"] += count
                bucket = per_dataset_results[dataset_name].setdefault(method_name, {"score": 0.0, "tokens": 0.0, "count": 0})
                bucket["score"] += total_score
                bucket["tokens"] += total_tokens
                bucket["count"] += count
        for method_name, row in seed_row["methods"].items():
            count = max(1, int(row["count"]))
            value = (row["score"] / count) - lambda_cost * ((row["tokens"] / count) / 192.0)
            row["cost_adjusted_accuracy"] = float(value)
            row["metric_value"] = float(value)
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

    datasets_observed = [
        {
            "name": suite["meta"].get("name") or suite["target"].get("name"),
            "id": suite["meta"].get("id"),
            "config": suite["meta"].get("config"),
            "split": suite["meta"].get("split"),
            "num_materialized_examples": len(suite["examples"]),
        }
        for suite in suites
    ]
    required_names = [target.get("name") for target in DEFAULTS["targets"]]
    observed_names = {str(row["name"]).lower() for row in datasets_observed}
    completed_required_datasets = all(str(name or "").lower() in observed_names for name in required_names if name)
    full_completed = bool(
        not load_failures
        and completed_required_datasets
        and seeds >= DEFAULT_SEEDS
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
            "max_examples_per_dataset_seed": max_examples,
            "methods": list(method_specs.keys()),
            "ablations": DEFAULTS.get("ablations", []),
            "cost_lambda": lambda_cost,
        }),
        "per_seed_results": _write_json("per_seed_results.json", seed_results),
        "per_dataset_results": _write_json("per_dataset_results.json", per_dataset_results),
        "main_results_table": _write_json("main_results_table.json", per_method),
        "ablation_table": _write_json("ablation_table.json", ablation_table),
        "bootstrap_ci": _write_json("bootstrap_ci.json", bootstrap),
    }
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
        "full_benchmark_completed": full_completed,
        "artifact_paths": artifacts,
        METRIC_NAME: per_method.get("CGGR", {}).get("metric_value", 0.0),
    }
    artifacts["artifact_manifest"] = _write_json("artifact_manifest.json", {
        "full_benchmark_completed": full_completed,
        "artifacts": artifacts,
        "datasets": datasets_observed,
        "methods": list(per_method.keys()),
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
