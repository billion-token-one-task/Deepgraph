"""Generic real-data/real-model runner for two structured task protocols."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from meta_harness.failure_policy import classify_failure
from meta_harness.runner_capability import ExperimentRequirements
from meta_harness.runner_contract import (
    ResearchRunner,
    RunnerContractError,
    recompute_metric,
    validate_final_results,
)


def _dump(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalized_input_sha256(value: str) -> str:
    normalized = " ".join(str(value).split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _load_candidate(path: Path, protocol: str):
    if not path.is_file():
        raise RunnerContractError("runner_contract_violation", "candidate_adapter_missing")
    spec = importlib.util.spec_from_file_location("deepgraph_candidate_adapter", path)
    if spec is None or spec.loader is None:
        raise RunnerContractError("runner_contract_violation", "candidate_adapter_unloadable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    method_name = str(getattr(module, "CANDIDATE_METHOD", "")).strip()
    if not method_name:
        raise RunnerContractError("runner_contract_violation", "candidate_method_missing")
    hook = "candidate_prompt" if protocol == "generative_qa" else "candidate_text"
    if not callable(getattr(module, hook, None)):
        raise RunnerContractError("runner_contract_violation", f"{hook}_missing")
    return module, method_name, hook


class GenericTransformersRunner(ResearchRunner):
    """One model and dataset revision, paired baseline/candidate evaluation."""

    BASELINE_METHOD = "unmodified_input_baseline"

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        candidate_adapter_path: str | Path,
        output_dir: str | Path,
    ):
        requirement_payload = config.get("requirements") or config
        self.requirements = ExperimentRequirements.from_dict(requirement_payload)
        self.config = dict(config)
        self.dataset_revision = str(
            config.get("resolved_dataset_revision")
            or self.requirements.dataset.revision
        )
        self.model_revision = str(
            config.get("resolved_model_revision")
            or self.requirements.model.revision
        )
        self.output_dir = Path(output_dir)
        self.candidate_path = Path(candidate_adapter_path)
        self.candidate_module = None
        self.candidate_method = ""
        self.candidate_hook = ""
        self.dataset_rows: list[dict[str, Any]] = []
        self.tokenizer = None
        self.model = None
        self.torch = None
        self.predictions: list[dict[str, Any]] = []
        self.metrics: dict[str, float] = {}
        self.started_at = time.time()

    def prepare(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.candidate_module, self.candidate_method, self.candidate_hook = _load_candidate(
            self.candidate_path,
            self.requirements.task_protocol,
        )
        try:
            import torch
        except ImportError as exc:
            raise RunnerContractError("dependency_missing", "torch") from exc
        self.torch = torch
        for seed in self.requirements.seeds:
            random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

    def load_dataset(self) -> Sequence[Mapping[str, Any]]:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RunnerContractError("dependency_missing", "datasets") from exc
        dataset_args = [self.requirements.dataset.repository_id]
        if self.requirements.dataset.config:
            dataset_args.append(self.requirements.dataset.config)
        try:
            dataset = load_dataset(
                *dataset_args,
                split=self.requirements.dataset.split,
                revision=self.dataset_revision,
            )
        except Exception as exc:
            reason = classify_failure(message=f"dataset unavailable:{exc}")
            raise RunnerContractError(reason, str(exc)) from exc
        cap = self.requirements.sample_cap or len(dataset)
        self.dataset_rows = [dict(dataset[index]) for index in range(min(len(dataset), cap))]
        if not self.dataset_rows:
            raise RunnerContractError("dataset_unavailable", "empty_split")
        missing = sorted(
            set(self.requirements.dataset.field_mapping.values())
            - set(self.dataset_rows[0])
        )
        if missing:
            raise RunnerContractError("dataset_schema_mismatch", ",".join(missing))
        return self.dataset_rows

    def load_model(self) -> Any:
        try:
            from transformers import (
                AutoModelForCausalLM,
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )
        except ImportError as exc:
            raise RunnerContractError("dependency_missing", "transformers") from exc
        kwargs: dict[str, Any] = {"revision": self.model_revision}
        if self.torch.cuda.is_available():
            kwargs["device_map"] = "auto"
            kwargs["torch_dtype"] = "auto"
        runtime_adjustments = dict(self.config.get("runtime_adjustments") or {})
        use_4bit = self.requirements.model.quantization == "4bit" or bool(
            runtime_adjustments.get("prefer_quantized")
        )
        if use_4bit:
            try:
                from transformers import BitsAndBytesConfig

                kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
            except ImportError as exc:
                raise RunnerContractError("dependency_missing", "bitsandbytes") from exc
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.requirements.model.repository_id,
                revision=self.model_revision,
            )
            model_class = (
                AutoModelForCausalLM
                if self.requirements.task_protocol == "generative_qa"
                else AutoModelForSequenceClassification
            )
            self.model = model_class.from_pretrained(
                self.requirements.model.repository_id,
                **kwargs,
            )
            self.model.eval()
        except Exception as exc:
            reason = classify_failure(message=f"model load:{exc}")
            raise RunnerContractError(reason, str(exc)) from exc
        return self.model

    def _device(self):
        try:
            return next(self.model.parameters()).device
        except (StopIteration, AttributeError):
            return self.torch.device("cuda" if self.torch.cuda.is_available() else "cpu")

    def _qa_prediction(self, prompt: str) -> str:
        tokens = self.tokenizer(prompt, return_tensors="pt")
        tokens = {key: value.to(self._device()) for key, value in tokens.items()}
        runtime_adjustments = dict(self.config.get("runtime_adjustments") or {})
        max_new_tokens = int(
            runtime_adjustments.get("max_new_tokens")
            or self.config.get("max_new_tokens")
            or 64
        )
        with self.torch.inference_mode():
            generated = self.model.generate(
                **tokens,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=(
                    self.tokenizer.pad_token_id
                    if self.tokenizer.pad_token_id is not None
                    else self.tokenizer.eos_token_id
                ),
            )
        continuation = generated[0][tokens["input_ids"].shape[1] :]
        return self.tokenizer.decode(continuation, skip_special_tokens=True).strip()

    def _classification_prediction(self, text: str) -> str:
        tokens = self.tokenizer(text, return_tensors="pt", truncation=True)
        tokens = {key: value.to(self._device()) for key, value in tokens.items()}
        with self.torch.inference_mode():
            logits = self.model(**tokens).logits
        return str(int(logits.argmax(dim=-1).item()))

    def _run_method(self, method: str, *, candidate: bool) -> list[dict[str, Any]]:
        mapping = self.requirements.dataset.field_mapping
        output: list[dict[str, Any]] = []
        for seed in self.requirements.seeds:
            random.seed(seed)
            self.torch.manual_seed(seed)
            for index, example in enumerate(self.dataset_rows):
                if self.requirements.task_protocol == "generative_qa":
                    baseline_input = str(example[mapping["prompt"]])
                    candidate_example = {
                        key: value
                        for key, value in example.items()
                        if key != mapping["target"]
                    }
                    model_input = (
                        str(
                            self.candidate_module.candidate_prompt(
                                candidate_example, baseline_input
                            )
                        )
                        if candidate
                        else baseline_input
                    )
                    prediction = self._qa_prediction(model_input)
                    target = str(example[mapping["target"]])
                else:
                    baseline_input = str(example[mapping["text"]])
                    candidate_example = {
                        key: value
                        for key, value in example.items()
                        if key != mapping["label"]
                    }
                    model_input = (
                        str(
                            self.candidate_module.candidate_text(
                                candidate_example, baseline_input
                            )
                        )
                        if candidate
                        else baseline_input
                    )
                    prediction = self._classification_prediction(model_input)
                    target = str(example[mapping["label"]])
                output.append(
                    {
                        "method": method,
                        "seed": seed,
                        "sample_index": index,
                        "prediction": prediction,
                        "target": target,
                        "input_sha256": hashlib.sha256(
                            model_input.encode("utf-8")
                        ).hexdigest(),
                        "normalized_input_sha256": _normalized_input_sha256(
                            model_input
                        ),
                    }
                )
        self.predictions.extend(output)
        return output

    def run_baseline(self) -> Sequence[Mapping[str, Any]]:
        return self._run_method(self.BASELINE_METHOD, candidate=False)

    def run_candidate(self) -> Sequence[Mapping[str, Any]]:
        candidate_rows = self._run_method(self.candidate_method, candidate=True)
        baseline_hashes = {
            (int(row["seed"]), int(row["sample_index"])): str(
                row["normalized_input_sha256"]
            )
            for row in self.predictions
            if row["method"] == self.BASELINE_METHOD
        }
        candidate_hashes = {
            (int(row["seed"]), int(row["sample_index"])): str(
                row["normalized_input_sha256"]
            )
            for row in candidate_rows
        }
        if set(candidate_hashes) != set(baseline_hashes):
            raise RunnerContractError(
                "runner_contract_violation", "candidate_pairing_mismatch"
            )
        if candidate_hashes and all(
            candidate_hashes[key] == baseline_hashes[key]
            for key in candidate_hashes
        ):
            raise RunnerContractError(
                "runner_contract_violation", "candidate_adapter_identity"
            )
        return candidate_rows

    def compute_metrics(self) -> Mapping[str, Any]:
        metric_name = self.requirements.metric.name
        baseline_rows = [
            row for row in self.predictions if row["method"] == self.BASELINE_METHOD
        ]
        candidate_rows = [
            row for row in self.predictions if row["method"] == self.candidate_method
        ]
        self.metrics = {
            self.BASELINE_METHOD: recompute_metric(baseline_rows, metric_name),
            self.candidate_method: recompute_metric(candidate_rows, metric_name),
        }
        return self.metrics

    def _gpu_environment(self) -> dict[str, Any]:
        available = bool(self.torch.cuda.is_available())
        return {
            "available": available,
            "device_count": int(self.torch.cuda.device_count()) if available else 0,
            "device_name": self.torch.cuda.get_device_name(0) if available else "cpu",
            "cuda_version": self.torch.version.cuda,
            "torch_version": self.torch.__version__,
            "python_version": platform.python_version(),
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        }

    def emit_final_results(self) -> Mapping[str, Any]:
        if not self.metrics:
            raise RunnerContractError("metric_missing")
        raw_path = self.output_dir / "raw_predictions.jsonl"
        raw_path.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in self.predictions
            ),
            encoding="utf-8",
        )
        environment_path = self.output_dir / "environment_manifest.json"
        dataset_path = self.output_dir / "dataset_manifest.json"
        model_path = self.output_dir / "model_manifest.json"
        _dump(environment_path, self._gpu_environment())
        _dump(
            dataset_path,
            {
                "repository_id": self.requirements.dataset.repository_id,
                "revision": self.dataset_revision,
                "config": self.requirements.dataset.config,
                "split": self.requirements.dataset.split,
                "field_mapping": dict(self.requirements.dataset.field_mapping),
                "num_examples": len(self.dataset_rows),
            },
        )
        _dump(
            model_path,
            {
                "repository_id": self.requirements.model.repository_id,
                "revision": self.model_revision,
                "framework": self.requirements.model.framework,
                "task": self.requirements.model.task,
                "quantization": self.requirements.model.quantization,
            },
        )
        baseline = float(self.metrics[self.BASELINE_METHOD])
        candidate = float(self.metrics[self.candidate_method])
        direction = self.requirements.metric.direction
        negative = candidate <= baseline if direction == "higher" else candidate >= baseline
        artifacts = {
            "final_results": {"path": "final_results.json"},
            "raw_predictions": {"path": raw_path.name},
            "environment_manifest": {"path": environment_path.name},
            "dataset_manifest": {"path": dataset_path.name},
            "model_manifest": {"path": model_path.name},
        }
        hashes = {
            "raw_predictions": _sha256(raw_path),
            "environment_manifest": _sha256(environment_path),
            "dataset_manifest": _sha256(dataset_path),
            "model_manifest": _sha256(model_path),
            "candidate_adapter": _sha256(self.candidate_path),
        }
        result = {
            "schema_version": "final_results_v1",
            "task_protocol": self.requirements.task_protocol,
            "dataset_id": self.requirements.dataset.repository_id,
            "dataset_revision": self.dataset_revision,
            "model_id": self.requirements.model.repository_id,
            "model_revision": self.model_revision,
            "seeds": list(self.requirements.seeds),
            "num_seeds": len(self.requirements.seeds),
            "num_examples": len(self.dataset_rows),
            "baseline_method": self.BASELINE_METHOD,
            "candidate_method": self.candidate_method,
            "metric_name": self.requirements.metric.name,
            "primary_metric": self.requirements.metric.name,
            "metric_direction": direction,
            "metric_value": candidate,
            "baseline_metric_value": baseline,
            "best_metric_value": candidate,
            "per_method": {
                self.BASELINE_METHOD: {
                    self.requirements.metric.name: baseline,
                    "metric_value": baseline,
                },
                self.candidate_method: {
                    self.requirements.metric.name: candidate,
                    "metric_value": candidate,
                },
            },
            "seed_results": [
                {
                    "seed": seed,
                    "baseline": recompute_metric(
                        [
                            row
                            for row in self.predictions
                            if row["method"] == self.BASELINE_METHOD
                            and row["seed"] == seed
                        ],
                        self.requirements.metric.name,
                    ),
                    "candidate": recompute_metric(
                        [
                            row
                            for row in self.predictions
                            if row["method"] == self.candidate_method
                            and row["seed"] == seed
                        ],
                        self.requirements.metric.name,
                    ),
                }
                for seed in self.requirements.seeds
            ],
            "scientific_negative_result": negative,
            "execution_reason_code": (
                "scientific_negative_result" if negative else "attempt_completed"
            ),
            "label_fallback_used": False,
            "gpu_environment": self._gpu_environment(),
            "artifacts": artifacts,
            "artifact_hashes": hashes,
            "wall_seconds": time.time() - self.started_at,
        }
        validate_final_results(result)
        final_path = self.output_dir / "final_results.json"
        _dump(final_path, result)
        return result

    def run(self) -> Mapping[str, Any]:
        self.prepare()
        print("BENCHMARK_STAGE: prepare_complete", flush=True)
        self.load_dataset()
        print("BENCHMARK_STAGE: dataset_ready", flush=True)
        self.load_model()
        print("BENCHMARK_STAGE: model_ready", flush=True)
        self.run_baseline()
        self.run_candidate()
        self.compute_metrics()
        result = self.emit_final_results()
        print("FINAL_RESULTS: " + json.dumps(result, ensure_ascii=False), flush=True)
        return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--candidate-adapter", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    try:
        config = json.loads(Path(args.config).read_text(encoding="utf-8"))
        GenericTransformersRunner(
            config,
            candidate_adapter_path=args.candidate_adapter,
            output_dir=args.output_dir,
        ).run()
        return 0
    except Exception as exc:
        reason = (
            exc.reason_code
            if isinstance(exc, RunnerContractError)
            else classify_failure(message=str(exc), returncode=1)
        )
        print(
            "RUNNER_ERROR: "
            + json.dumps(
                {"reason_code": reason, "detail": str(exc)},
                ensure_ascii=False,
            ),
            file=sys.stderr,
            flush=True,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
